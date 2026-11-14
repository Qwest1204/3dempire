use editor_ui::EditorUiState;
use engine::{
    camera::FlyCamera, diagnostics::FrameStatistics, input::InputState, time::FrameClock,
};
use glam::Vec3;
use renderer::{
    DebugRenderer, DeferredRenderer, DrawParams, GpuMesh, LightingSettings, RenderContext,
};
use scene::{
    Material, MeshHandle, PreviousTransform, Renderable, Transform, advance_transform_history,
    extract_render_world,
};
use shaders::{SlangCompileResult, SlangHotReload};
use std::sync::Arc;
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::{CursorGrabMode, Window},
};

// `ApplicationHandler` migration is intentionally deferred until winit/egui-winit
// expose a stable shared lifecycle pattern for this async GPU bootstrap path.
// Keep this localized compatibility allowance so workspace CI remains strict.
#[allow(deprecated)]
fn main() {
    let event_loop = EventLoop::new().expect("creating event loop");
    let window = Arc::new(
        event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("3dempire · sandbox")
                    .with_inner_size(PhysicalSize::new(1440, 900)),
            )
            .expect("creating window"),
    );
    let mut context =
        pollster::block_on(RenderContext::new(window.clone())).expect("initializing renderer");
    let shader_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../shaders/generated/debug_output.wgsl"
    );
    let mut debug_renderer =
        DebugRenderer::new(&context, shader_path).expect("creating debug pipeline");
    let mut scene_renderer = DeferredRenderer::new(&context);
    let mut meshes: Vec<GpuMesh> = Vec::new();
    meshes.push(scene_renderer.upload_mesh(&context, &assets::primitives::plane()));
    meshes.push(scene_renderer.upload_mesh(&context, &assets::primitives::cube()));
    meshes.push(scene_renderer.upload_mesh(&context, &assets::primitives::uv_sphere(32, 16)));
    let suzanne_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/models/suzanne.glb"
    );
    if let Ok(suzanne) = assets::gltf::load_first_mesh(suzanne_path) {
        meshes.push(scene_renderer.upload_mesh(&context, &suzanne));
    }
    let mut world = hecs::World::new();
    let plane_transform = Transform {
        translation: Vec3::new(0.0, -1.0, 0.0),
        rotation: glam::Quat::IDENTITY,
        scale: Vec3::splat(10.0),
    };
    world.spawn((
        plane_transform,
        PreviousTransform(plane_transform.matrix()),
        Renderable {
            mesh: MeshHandle(0),
            material: Material {
                base_color: Vec3::new(0.22, 0.25, 0.30),
                metallic: 0.0,
                roughness: 0.9,
                emissive: Vec3::ZERO,
                occlusion: 1.0,
            },
        },
    ));
    let cube_transform = Transform {
        translation: Vec3::new(-2.0, 0.0, 0.0),
        rotation: glam::Quat::IDENTITY,
        scale: Vec3::splat(0.75),
    };
    let cube = world.spawn((
        cube_transform,
        PreviousTransform(cube_transform.matrix()),
        Renderable {
            mesh: MeshHandle(1),
            material: Material {
                base_color: Vec3::new(0.15, 0.52, 0.95),
                metallic: 0.7,
                roughness: 0.22,
                emissive: Vec3::ZERO,
                occlusion: 1.0,
            },
        },
    ));
    let sphere_transform = Transform {
        translation: Vec3::new(2.0, 0.0, 0.0),
        rotation: glam::Quat::IDENTITY,
        scale: Vec3::splat(0.9),
    };
    world.spawn((
        sphere_transform,
        PreviousTransform(sphere_transform.matrix()),
        Renderable {
            mesh: MeshHandle(2),
            material: Material {
                base_color: Vec3::new(0.96, 0.30, 0.16),
                metallic: 0.05,
                roughness: 0.38,
                emissive: Vec3::new(0.04, 0.0, 0.0),
                occlusion: 1.0,
            },
        },
    ));
    if meshes.len() > 3 {
        let suzanne_transform = Transform {
            translation: Vec3::new(0.0, 0.0, 0.0),
            rotation: glam::Quat::IDENTITY,
            scale: Vec3::splat(1.25),
        };
        world.spawn((
            suzanne_transform,
            PreviousTransform(suzanne_transform.matrix()),
            Renderable {
                mesh: MeshHandle(3),
                material: Material {
                    base_color: Vec3::new(0.75, 0.35, 0.87),
                    metallic: 0.25,
                    roughness: 0.42,
                    emissive: Vec3::ZERO,
                    occlusion: 1.0,
                },
            },
        ));
    }
    let mut slang_hot_reload =
        SlangHotReload::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")).ok();

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
    let mut input = InputState::default();
    let mut camera = FlyCamera::default();
    let mut previous_view_projection = glam::Mat4::IDENTITY;
    let mut clock = FrameClock::default();
    let mut statistics = FrameStatistics::default();
    let mut ui_state = EditorUiState::default();
    let mut lighting = LightingSettings::default();

    event_loop
        .run(move |event, event_loop| match &event {
            Event::DeviceEvent { event, .. } => input.device_event(event),
            Event::WindowEvent { window_id, event } if *window_id == window.id() => {
                let response = egui_state.on_window_event(&window, event);
                if !response.consumed {
                    input.window_event(event);
                }
                match event {
                    WindowEvent::CloseRequested => event_loop.exit(),
                    WindowEvent::Resized(size) => {
                        context.resize(*size);
                        scene_renderer.resize(&context);
                    }
                    WindowEvent::RedrawRequested => {
                        clock.tick();
                        statistics.record_frame(clock.delta);
                        camera.update(&input, clock.delta_seconds());
                        if let Ok(mut transform) = world.get::<&mut Transform>(cube) {
                            transform.rotation =
                                glam::Quat::from_rotation_y(clock.elapsed.as_secs_f32() * 0.8);
                        }
                        if let Some(slang) = &mut slang_hot_reload {
                            match slang.update() {
                                SlangCompileResult::Compiled => {
                                    debug_renderer.reload(&context);
                                    scene_renderer.reload_shaders(&context);
                                }
                                SlangCompileResult::Failed(error) => {
                                    eprintln!("Slang hot reload failed:\n{error}")
                                }
                                SlangCompileResult::Unchanged | SlangCompileResult::Compiling => {}
                            }
                        }
                        let slang_status = slang_hot_reload
                            .as_ref()
                            .map(|slang| slang.status().clone())
                            .unwrap_or_default();
                        let shader_error = scene_renderer
                            .shader_error()
                            .or(debug_renderer.shader_error());
                        let raw_input = egui_state.take_egui_input(&window);
                        let mut recompile_shaders = false;
                        let output = egui_context.run(raw_input, |egui_context| {
                            recompile_shaders = editor_ui::draw(
                                egui_context,
                                &mut ui_state,
                                &mut camera,
                                &statistics,
                                &mut lighting,
                                shader_error,
                                &slang_status,
                            );
                        });
                        if recompile_shaders && let Some(slang) = &mut slang_hot_reload {
                            slang.request_compile();
                        }
                        egui_state.handle_platform_output(&window, output.platform_output);
                        let paint_jobs =
                            egui_context.tessellate(output.shapes, egui_context.pixels_per_point());
                        for (id, image_delta) in &output.textures_delta.set {
                            egui_renderer.update_texture(
                                &context.device,
                                &context.queue,
                                *id,
                                image_delta,
                            );
                        }
                        let screen = egui_wgpu::ScreenDescriptor {
                            size_in_pixels: [context.config.width, context.config.height],
                            pixels_per_point: egui_context.pixels_per_point(),
                        };
                        let mut encoder = context.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("frame encoder"),
                            },
                        );
                        egui_renderer.update_buffers(
                            &context.device,
                            &context.queue,
                            &mut encoder,
                            &paint_jobs,
                            &screen,
                        );
                        let frame = match context.surface.get_current_texture() {
                            Ok(frame) => frame,
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                context.resize(context.size);
                                return;
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                event_loop.exit();
                                return;
                            }
                            Err(wgpu::SurfaceError::Timeout) => return,
                            Err(wgpu::SurfaceError::Other) => return,
                        };
                        let view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());
                        // Slang-generated sky is an offscreen background. The deferred
                        // composite samples it only where the G-buffer has no geometry.
                        debug_renderer.draw(
                            &mut encoder,
                            scene_renderer.sky_target(),
                            camera.view_matrix().inverse(),
                            &context.queue,
                        );
                        let render_world = extract_render_world(&world);
                        let aspect = context.config.width as f32 / context.config.height as f32;
                        let view_projection =
                            camera.projection_matrix(aspect) * camera.view_matrix();
                        lighting.camera_position = camera.position;
                        scene_renderer.draw(
                            DrawParams {
                                encoder: &mut encoder,
                                target: &view,
                                view_projection,
                                previous_view_projection,
                                output: ui_state.viewport_output,
                                lighting,
                                queue: &context.queue,
                            },
                            render_world.instances.iter().filter_map(|instance| {
                                meshes.get(instance.mesh.0).map(|mesh| {
                                    (
                                        mesh,
                                        instance.model,
                                        instance.previous_model,
                                        instance.material.base_color,
                                        instance.material.metallic,
                                        instance.material.roughness,
                                        instance.material.emissive,
                                        instance.material.occlusion,
                                    )
                                })
                            }),
                        );
                        {
                            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("egui pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    depth_slice: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
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
                        previous_view_projection = view_projection;
                        advance_transform_history(&mut world);
                        for id in &output.textures_delta.free {
                            egui_renderer.free_texture(id);
                        }
                        input.begin_frame();
                    }
                    WindowEvent::MouseInput { .. } => {
                        let mode = if input.look_active {
                            CursorGrabMode::Locked
                        } else {
                            CursorGrabMode::None
                        };
                        let _ = window.set_cursor_grab(mode);
                        window.set_cursor_visible(!input.look_active);
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        })
        .expect("running event loop");
}
