use assets::{
    image::LinearImage,
    neural_fourier::{FourierNeuralTextureModel, HIDDEN_WIDTH},
};
use bytemuck::{Pod, Zeroable};
use renderer::RenderContext;
use std::sync::Arc;
use winit::{dpi::PhysicalSize, event::WindowEvent, event_loop::EventLoop, window::Window};

struct SourceTexture {
    image: LinearImage,
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    egui_id: egui::TextureId,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewSettings {
    parameter_count: u32,
    mode: u32,
    error_scale: f32,
    _padding: f32,
}

#[derive(Clone, Copy, PartialEq)]
enum ViewMode {
    Source,
    Reconstruction,
    Error,
    AmplifiedError,
    Split,
}
impl ViewMode {
    const ALL: [Self; 5] = [
        Self::Source,
        Self::Reconstruction,
        Self::Error,
        Self::AmplifiedError,
        Self::Split,
    ];
    const fn label(self) -> &'static str {
        match self {
            Self::Source => "Source",
            Self::Reconstruction => "Reconstruction",
            Self::Error => "Absolute error",
            Self::AmplifiedError => "Amplified error",
            Self::Split => "Split",
        }
    }
    const fn index(self) -> u32 {
        match self {
            Self::Source => 0,
            Self::Reconstruction => 1,
            Self::Error => 2,
            Self::AmplifiedError => 3,
            Self::Split => 4,
        }
    }
}

struct NeuralRenderer {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    parameter_buffer: wgpu::Buffer,
    settings_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

fn full_image_mse_and_psnr(model: &FourierNeuralTextureModel, image: &LinearImage) -> (f32, f32) {
    let mse = image
        .pixels
        .iter()
        .enumerate()
        .map(|(index, target)| {
            let x = index % image.width as usize;
            let y = index / image.width as usize;
            let output = model.forward([
                x as f32 / (image.width - 1).max(1) as f32,
                y as f32 / (image.height - 1).max(1) as f32,
            ]);
            (0..3)
                .map(|channel| (output[channel] - target[channel]).powi(2))
                .sum::<f32>()
                / 3.0
        })
        .sum::<f32>()
        / image.pixels.len() as f32;
    (mse, -10.0 * mse.max(1e-12).log10())
}

impl NeuralRenderer {
    fn new(context: &RenderContext, parameter_count: usize) -> Self {
        let layout = context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("neural texture layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let shader = context.device.create_shader_module(wgpu::include_wgsl!(
            "../../../shaders/generated/neural_fourier_texture.wgsl"
        ));
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("neural texture pipeline layout"),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                });
        let pipeline = context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("neural texture pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });
        let parameter_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("neural texture parameters"),
            size: (parameter_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let settings_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("neural texture view settings"),
            size: std::mem::size_of::<ViewSettings>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            pipeline,
            layout,
            parameter_buffer,
            settings_buffer,
            sampler,
        }
    }
    fn render(
        &self,
        context: &RenderContext,
        source: &SourceTexture,
        model: &FourierNeuralTextureModel,
        mode: ViewMode,
        target: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        context.queue.write_buffer(
            &self.parameter_buffer,
            0,
            bytemuck::cast_slice(&model.parameters),
        );
        context.queue.write_buffer(
            &self.settings_buffer,
            0,
            bytemuck::bytes_of(&ViewSettings {
                parameter_count: model.parameters.len() as u32,
                mode: mode.index(),
                error_scale: 8.0,
                _padding: 0.0,
            }),
        );
        let group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("neural texture bind group"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.parameter_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&source.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.settings_buffer.as_entire_binding(),
                    },
                ],
            });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("neural texture inference"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
}

impl SourceTexture {
    fn new(
        context: &RenderContext,
        egui_renderer: &mut egui_wgpu::Renderer,
        image: LinearImage,
    ) -> Self {
        let texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("neural lab source texture"),
            size: wgpu::Extent3d {
                width: image.width,
                height: image.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        context.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &image.rgba8_srgb(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * image.width),
                rows_per_image: Some(image.height),
            },
            wgpu::Extent3d {
                width: image.width,
                height: image.height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&Default::default());
        let egui_id = egui_renderer.register_native_texture(
            &context.device,
            &view,
            wgpu::FilterMode::Nearest,
        );
        Self {
            image,
            _texture: texture,
            view,
            egui_id,
        }
    }
}

// The lab starts as a deliberately unlit 2D reference path. Subsequent steps
// replace this displayed source with Slang MLP reconstruction/error views.
#[allow(deprecated)]
fn main() {
    let event_loop = EventLoop::new().expect("creating event loop");
    let window = Arc::new(
        event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("3dempire · neural texture lab")
                    .with_inner_size(PhysicalSize::new(1280, 820)),
            )
            .expect("creating window"),
    );
    let context =
        pollster::block_on(RenderContext::new(window.clone())).expect("initializing renderer");
    let egui_context = egui::Context::default();
    let mut egui_state = egui_winit::State::new(
        egui_context.clone(),
        egui::ViewportId::ROOT,
        &window,
        None,
        None,
        None,
    );
    let mut egui_renderer = egui_wgpu::Renderer::new(
        &context.device,
        context.config.format,
        egui_wgpu::RendererOptions::default(),
    );
    let initial_image = std::env::args()
        .nth(1)
        .map(LinearImage::load)
        .transpose()
        .expect("loading source image")
        .unwrap_or_else(|| LinearImage::gradient_circle(512, 512));
    let source = SourceTexture::new(&context, &mut egui_renderer, initial_image);
    // Deterministic 10% subset keeps the experiment reproducible and ensures
    // training cost does not scale as aggressively with source resolution.
    let training_indices: Vec<usize> = (0..source.image.pixels.len())
        .filter(|index| index % 10 == 0)
        .collect();
    let mut model = FourierNeuralTextureModel::new();
    let neural_renderer = NeuralRenderer::new(&context, model.parameters.len());
    let reconstruction_texture = context.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("neural reconstruction target"),
        size: wgpu::Extent3d {
            width: source.image.width,
            height: source.image.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let reconstruction_view = reconstruction_texture.create_view(&Default::default());
    let reconstruction_id = egui_renderer.register_native_texture(
        &context.device,
        &reconstruction_view,
        wgpu::FilterMode::Nearest,
    );
    let mut learning = false;
    let mut loss_history = Vec::new();
    let mut view_mode = ViewMode::Source;
    let mut full_metrics = full_image_mse_and_psnr(&model, &source.image);
    let mut frames_since_metrics = 0_u32;

    event_loop
        .run(move |event, event_loop| match event {
            winit::event::Event::WindowEvent { window_id, event } if window_id == window.id() => {
                let response = egui_state.on_window_event(&window, &event);
                match event {
                    WindowEvent::CloseRequested => event_loop.exit(),
                    WindowEvent::RedrawRequested => {
                        if learning {
                            for _ in 0..8 {
                                loss_history.push(model.train_step(
                                    &source.image,
                                    &training_indices,
                                    512,
                                    0.01,
                                ));
                            }
                            if loss_history.len() > 240 {
                                loss_history.drain(..loss_history.len() - 240);
                            }
                        }
                        frames_since_metrics += 1;
                        if frames_since_metrics >= 30 {
                            full_metrics = full_image_mse_and_psnr(&model, &source.image);
                            frames_since_metrics = 0;
                        }
                        let raw_input = egui_state.take_egui_input(&window);
                        let output = egui_context.run(raw_input, |ui_context| {
                            egui::SidePanel::left("neural-controls")
                                .default_width(270.0)
                                .show(ui_context, |ui| {
                                    ui.heading("Neural Texture Lab");
                                    ui.label("Phase N2 · CPU train + Slang inference");
                                    ui.separator();
                                    ui.label(format!(
                                        "Source: {} × {}",
                                        source.image.width, source.image.height
                                    ));
                                    ui.label("Color space: sRGB → linear RGB");
                                    ui.label(format!(
                                        "Training texels: {} / {} (10%)",
                                        training_indices.len(),
                                        source.image.pixels.len()
                                    ));
                                    ui.separator();
                                    ui.strong("Viewport");
                                    for mode in ViewMode::ALL {
                                        ui.radio_value(&mut view_mode, mode, mode.label());
                                    }
                                    ui.separator();
                                    ui.strong("Training");
                                    if ui
                                        .button(if learning { "Pause" } else { "Learn" })
                                        .clicked()
                                    {
                                        learning = !learning;
                                    }
                                    if ui.button("Reset weights").clicked() {
                                        model = FourierNeuralTextureModel::new();
                                        loss_history.clear();
                                    }
                                    ui.label(format!(
                                        "MLP: {INPUT} → {HIDDEN_WIDTH} → {HIDDEN_WIDTH} → 3 ReLU",
                                        INPUT = 14
                                    ));
                                    ui.label(format!("Adam steps: {}", model.step));
                                    if let Some(loss) = loss_history.last() {
                                        ui.label(format!("Batch MSE: {loss:.6}"));
                                    }
                                    let (mse, psnr) = full_metrics;
                                    ui.label(format!("Full image MSE: {mse:.6}"));
                                    ui.label(format!("PSNR: {psnr:.2} dB"));
                                    if loss_history.len() > 1 {
                                        let max = loss_history
                                            .iter()
                                            .copied()
                                            .fold(0.000001_f32, f32::max);
                                        let (rect, _) = ui.allocate_exact_size(
                                            egui::vec2(ui.available_width(), 64.0),
                                            egui::Sense::hover(),
                                        );
                                        let points = loss_history
                                            .iter()
                                            .enumerate()
                                            .map(|(index, loss)| {
                                                egui::pos2(
                                                    rect.left()
                                                        + rect.width() * index as f32
                                                            / (loss_history.len() - 1) as f32,
                                                    rect.bottom() - rect.height() * (*loss / max),
                                                )
                                            })
                                            .collect();
                                        ui.painter().add(egui::Shape::line(
                                            points,
                                            egui::Stroke::new(1.5_f32, egui::Color32::LIGHT_GREEN),
                                        ));
                                    }
                                    ui.separator();
                                    ui.small("Run: cargo run -p neural_lab -- path/to/image.png");
                                });
                            egui::CentralPanel::default().show(ui_context, |ui| {
                                ui.heading(view_mode.label());
                                let available = ui.available_size();
                                let aspect = source.image.width as f32 / source.image.height as f32;
                                let size = if available.x / available.y > aspect {
                                    egui::vec2(available.y * aspect, available.y)
                                } else {
                                    egui::vec2(available.x, available.x / aspect)
                                };
                                let texture_id = if view_mode == ViewMode::Source {
                                    source.egui_id
                                } else {
                                    reconstruction_id
                                };
                                ui.add(egui::Image::new((texture_id, size)));
                            });
                        });
                        egui_state.handle_platform_output(&window, output.platform_output);
                        let paint_jobs =
                            egui_context.tessellate(output.shapes, egui_context.pixels_per_point());
                        for (id, delta) in &output.textures_delta.set {
                            egui_renderer.update_texture(
                                &context.device,
                                &context.queue,
                                *id,
                                delta,
                            );
                        }
                        let screen = egui_wgpu::ScreenDescriptor {
                            size_in_pixels: [context.config.width, context.config.height],
                            pixels_per_point: egui_context.pixels_per_point(),
                        };
                        let mut encoder = context.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("neural lab encoder"),
                            },
                        );
                        if view_mode != ViewMode::Source {
                            neural_renderer.render(
                                &context,
                                &source,
                                &model,
                                view_mode,
                                &reconstruction_view,
                                &mut encoder,
                            );
                        }
                        egui_renderer.update_buffers(
                            &context.device,
                            &context.queue,
                            &mut encoder,
                            &paint_jobs,
                            &screen,
                        );
                        let frame = match context.surface.get_current_texture() {
                            Ok(frame) => frame,
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                event_loop.exit();
                                return;
                            }
                            Err(_) => return,
                        };
                        let view = frame.texture.create_view(&Default::default());
                        {
                            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("neural lab ui"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    depth_slice: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });
                            egui_renderer.render(&mut pass.forget_lifetime(), &paint_jobs, &screen);
                        }
                        context.queue.submit(Some(encoder.finish()));
                        frame.present();
                        for id in &output.textures_delta.free {
                            egui_renderer.free_texture(id);
                        }
                    }
                    _ => {}
                }
                if response.repaint {
                    window.request_redraw();
                }
            }
            winit::event::Event::AboutToWait => window.request_redraw(),
            _ => {}
        })
        .expect("running event loop");
}
