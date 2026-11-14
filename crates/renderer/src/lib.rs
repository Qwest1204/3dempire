use anyhow::{Context, Result};
use assets::{MeshData, Vertex};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::{num::NonZeroU64, path::Path, sync::Arc};
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewportOutput {
    FinalComposite,
    BaseColor,
    Metallic,
    WorldNormal,
    Roughness,
    DepthRaw,
    DepthLinear,
    MotionVectors,
}

impl ViewportOutput {
    pub const ALL: [Self; 8] = [
        Self::FinalComposite,
        Self::BaseColor,
        Self::Metallic,
        Self::WorldNormal,
        Self::Roughness,
        Self::DepthRaw,
        Self::DepthLinear,
        Self::MotionVectors,
    ];
    pub const fn label(self) -> &'static str {
        match self {
            Self::FinalComposite => "Final composite",
            Self::BaseColor => "Base color",
            Self::Metallic => "Metallic",
            Self::WorldNormal => "World normal",
            Self::Roughness => "Roughness",
            Self::DepthRaw => "Depth (raw)",
            Self::DepthLinear => "Depth (linear)",
            Self::MotionVectors => "Motion vectors",
        }
    }
}

pub struct RenderContext {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: PhysicalSize<u32>,
}

impl RenderContext {
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window)
            .context("creating wgpu surface")?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("no compatible GPU adapter found")?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("3dempire device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .context("requesting wgpu device")?;
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(caps.formats[0]);
        let present_mode = caps
            .present_modes
            .iter()
            .copied()
            .find(|mode| *mode == wgpu::PresentMode::Fifo)
            .unwrap_or(caps.present_modes[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
        })
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.size = size;
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
    }
}

pub struct DebugRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    shader_path: std::path::PathBuf,
    last_shader_error: Option<String>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SkyUniform {
    camera_to_world: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_projection: [[f32; 4]; 4],
    previous_view_projection: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ObjectUniform {
    model: [[f32; 4]; 4],
    previous_model: [[f32; 4]; 4],
    color_metallic: [f32; 4],
    roughness: f32,
    emissive_occlusion: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightingUniform {
    inverse_view_projection: [[f32; 4]; 4],
    camera_position: [f32; 4],
    directional_direction: [f32; 4],
    directional_color_intensity: [f32; 4],
    point_position_range: [f32; 4],
    point_color_intensity: [f32; 4],
    spot_position_range: [f32; 4],
    spot_direction_inner: [f32; 4],
    spot_color_intensity: [f32; 4],
    spot_outer: [f32; 4],
    ambient_exposure: [f32; 4],
    environment_sky: [f32; 4],
    environment_ground: [f32; 4],
}

#[derive(Debug, Clone, Copy)]
pub struct LightingSettings {
    pub camera_position: Vec3,
    pub directional_direction: Vec3,
    pub directional_color: Vec3,
    pub directional_intensity: f32,
    pub point_position: Vec3,
    pub point_color: Vec3,
    pub point_intensity: f32,
    pub point_range: f32,
    pub spot_position: Vec3,
    pub spot_direction: Vec3,
    pub spot_color: Vec3,
    pub spot_intensity: f32,
    pub spot_range: f32,
    pub spot_inner_cos: f32,
    pub spot_outer_cos: f32,
    pub ambient_intensity: f32,
    pub exposure: f32,
    pub environment_intensity: f32,
    pub environment_rotation: f32,
}

impl Default for LightingSettings {
    fn default() -> Self {
        Self {
            camera_position: Vec3::ZERO,
            directional_direction: Vec3::new(-0.35, -0.8, -0.45),
            directional_color: Vec3::ONE,
            directional_intensity: 2.0,
            point_position: Vec3::new(-2.5, 2.0, 1.5),
            point_color: Vec3::new(0.35, 0.55, 1.0),
            point_intensity: 28.0,
            point_range: 8.0,
            spot_position: Vec3::new(2.5, 3.0, 2.5),
            spot_direction: Vec3::new(-0.5, -0.8, -0.5),
            spot_color: Vec3::new(1.0, 0.35, 0.16),
            spot_intensity: 35.0,
            spot_range: 10.0,
            spot_inner_cos: 0.94,
            spot_outer_cos: 0.82,
            ambient_intensity: 0.06,
            exposure: 1.0,
            environment_intensity: 0.35,
            environment_rotation: 0.0,
        }
    }
}

const OBJECT_UNIFORM_STRIDE: u64 = 256;
const MAX_SCENE_OBJECTS: u64 = 1_024;

pub struct GpuMesh {
    vertex: wgpu::Buffer,
    index: wgpu::Buffer,
    index_count: u32,
}

pub struct DrawParams<'a> {
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub target: &'a wgpu::TextureView,
    pub view_projection: Mat4,
    pub previous_view_projection: Mat4,
    pub output: ViewportOutput,
    pub lighting: LightingSettings,
    pub queue: &'a wgpu::Queue,
}

struct GBuffer {
    albedo: wgpu::TextureView,
    normal_roughness: wgpu::TextureView,
    motion: wgpu::TextureView,
    emissive_occlusion: wgpu::TextureView,
    depth: wgpu::TextureView,
    sky: wgpu::TextureView,
}

pub struct DeferredRenderer {
    geometry_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    debug_pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    object_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    object_bind_group: wgpu::BindGroup,
    composite_bind_group: wgpu::BindGroup,
    debug_bind_group: wgpu::BindGroup,
    texture_layout: wgpu::BindGroupLayout,
    camera_layout: wgpu::BindGroupLayout,
    object_layout: wgpu::BindGroupLayout,
    lighting_layout: wgpu::BindGroupLayout,
    final_shader_path: std::path::PathBuf,
    debug_shader_path: std::path::PathBuf,
    last_shader_error: Option<String>,
    lighting_buffer: wgpu::Buffer,
    lighting_bind_group: wgpu::BindGroup,
    sky_bind_group: wgpu::BindGroup,
    gbuffer: GBuffer,
}

impl DeferredRenderer {
    pub fn new(context: &RenderContext) -> Self {
        let camera_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("deferred camera layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
        let object_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("deferred object layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
        let texture_layout = Self::texture_layout(context);
        let lighting_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("lighting layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
        let camera_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deferred camera uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let object_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deferred object uniform"),
            size: OBJECT_UNIFORM_STRIDE * MAX_SCENE_OBJECTS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lighting_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lighting uniform"),
            size: std::mem::size_of::<LightingUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lighting_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lighting bind group"),
                layout: &lighting_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lighting_buffer.as_entire_binding(),
                }],
            });
        let camera_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("deferred camera bind group"),
                layout: &camera_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }],
            });
        let object_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("deferred object bind group"),
                layout: &object_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &object_buffer,
                        offset: 0,
                        size: NonZeroU64::new(OBJECT_UNIFORM_STRIDE),
                    }),
                }],
            });
        let gbuffer = Self::create_gbuffer(context);
        let (composite_bind_group, debug_bind_group, sky_bind_group) =
            Self::texture_bind_groups(context, &texture_layout, &gbuffer);
        let final_shader_path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../shaders/generated/deferred.final.wgsl"
        ));
        let debug_shader_path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../shaders/generated/deferred.debug.wgsl"
        ));
        let (geometry_pipeline, composite_pipeline, debug_pipeline) = Self::create_pipelines(
            context,
            &final_shader_path,
            &debug_shader_path,
            &camera_layout,
            &object_layout,
            &texture_layout,
            &lighting_layout,
        )
        .expect("creating deferred pipelines from generated Slang WGSL");
        Self {
            geometry_pipeline,
            composite_pipeline,
            debug_pipeline,
            camera_buffer,
            object_buffer,
            camera_bind_group,
            object_bind_group,
            composite_bind_group,
            debug_bind_group,
            texture_layout,
            camera_layout,
            object_layout,
            lighting_layout,
            final_shader_path,
            debug_shader_path,
            last_shader_error: None,
            lighting_buffer,
            lighting_bind_group,
            sky_bind_group,
            gbuffer,
        }
    }

    fn create_pipelines(
        context: &RenderContext,
        final_shader_path: &Path,
        debug_shader_path: &Path,
        camera_layout: &wgpu::BindGroupLayout,
        object_layout: &wgpu::BindGroupLayout,
        texture_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
    ) -> Result<(
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
    )> {
        let create_module = |path: &Path| -> Result<wgpu::ShaderModule> {
            let source = std::fs::read_to_string(path)
                .with_context(|| format!("reading shader {}", path.display()))?;
            Ok(context
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: path.to_str(),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                }))
        };
        let shader = create_module(final_shader_path)?;
        let debug_shader = create_module(debug_shader_path)?;
        let geometry_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("gbuffer layout"),
                    bind_group_layouts: &[camera_layout, object_layout],
                    push_constant_ranges: &[],
                });
        let geometry_pipeline =
            context
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("gbuffer pipeline"),
                    layout: Some(&geometry_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("gbuffer_vs"),
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<Vertex>() as u64,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x3,
                                    offset: 0,
                                    shader_location: 0,
                                },
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x3,
                                    offset: 12,
                                    shader_location: 1,
                                },
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x2,
                                    offset: 24,
                                    shader_location: 2,
                                },
                            ],
                        }],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("gbuffer_fs"),
                        targets: &[
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rgba16Float,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rg16Float,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rgba16Float,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                        ],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        cull_mode: Some(wgpu::Face::Back),
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample: Default::default(),
                    multiview: None,
                    cache: None,
                });
        let fullscreen_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("deferred fullscreen layout"),
                    bind_group_layouts: &[texture_layout, object_layout, lighting_layout],
                    push_constant_ranges: &[],
                });
        let make_fullscreen = |module: &wgpu::ShaderModule, entry: &str, label: &str| {
            context
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&fullscreen_layout),
                    vertex: wgpu::VertexState {
                        module,
                        entry_point: Some("fullscreen_vs"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module,
                        entry_point: Some(entry),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: context.config.format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    multiview: None,
                    cache: None,
                })
        };
        Ok((
            geometry_pipeline,
            make_fullscreen(&shader, "composite_fs", "deferred composite pipeline"),
            make_fullscreen(&debug_shader, "composite_fs", "gbuffer debug pipeline"),
        ))
    }

    /// Replaces all production deferred pipelines only after the complete generated
    /// Slang module has been accepted by wgpu. A failed candidate leaves rendering intact.
    pub fn reload_shaders(&mut self, context: &RenderContext) {
        match Self::create_pipelines(
            context,
            &self.final_shader_path,
            &self.debug_shader_path,
            &self.camera_layout,
            &self.object_layout,
            &self.texture_layout,
            &self.lighting_layout,
        ) {
            Ok((geometry, composite, debug)) => {
                self.geometry_pipeline = geometry;
                self.composite_pipeline = composite;
                self.debug_pipeline = debug;
                self.last_shader_error = None;
            }
            Err(error) => self.last_shader_error = Some(format!("{error:#}")),
        }
    }

    pub fn shader_error(&self) -> Option<&str> {
        self.last_shader_error.as_deref()
    }

    fn texture_layout(context: &RenderContext) -> wgpu::BindGroupLayout {
        context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gbuffer textures layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
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
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            })
    }
    fn create_texture(
        context: &RenderContext,
        label: &str,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> wgpu::TextureView {
        context
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: context.config.width,
                    height: context.config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
            .create_view(&Default::default())
    }
    fn create_gbuffer(context: &RenderContext) -> GBuffer {
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        GBuffer {
            albedo: Self::create_texture(
                context,
                "gbuffer albedo",
                wgpu::TextureFormat::Rgba8Unorm,
                usage,
            ),
            normal_roughness: Self::create_texture(
                context,
                "gbuffer normal roughness",
                wgpu::TextureFormat::Rgba16Float,
                usage,
            ),
            motion: Self::create_texture(
                context,
                "gbuffer motion",
                wgpu::TextureFormat::Rg16Float,
                usage,
            ),
            emissive_occlusion: Self::create_texture(
                context,
                "gbuffer emissive occlusion",
                wgpu::TextureFormat::Rgba16Float,
                usage,
            ),
            depth: Self::create_texture(
                context,
                "gbuffer depth",
                wgpu::TextureFormat::Depth32Float,
                usage,
            ),
            sky: Self::create_texture(context, "sky color", context.config.format, usage),
        }
    }
    fn texture_bind_groups(
        context: &RenderContext,
        layout: &wgpu::BindGroupLayout,
        gbuffer: &GBuffer,
    ) -> (wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup) {
        let sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("gbuffer sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let create = |label| {
            context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&gbuffer.albedo),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&gbuffer.sky),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::TextureView(
                                &gbuffer.emissive_occlusion,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&gbuffer.normal_roughness),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&gbuffer.motion),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&gbuffer.depth),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                    ],
                })
        };
        // Bind groups retain the sampler; each group has an identical immutable binding set.
        (
            create("composite gbuffer bind group"),
            create("debug gbuffer bind group"),
            create("sky gbuffer bind group"),
        )
    }
    pub fn resize(&mut self, context: &RenderContext) {
        self.gbuffer = Self::create_gbuffer(context);
        (
            self.composite_bind_group,
            self.debug_bind_group,
            self.sky_bind_group,
        ) = Self::texture_bind_groups(context, &self.texture_layout, &self.gbuffer);
    }
    pub fn upload_mesh(&self, context: &RenderContext, mesh: &MeshData) -> GpuMesh {
        GpuMesh {
            vertex: context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{} vertices", mesh.name)),
                    contents: bytemuck::cast_slice(&mesh.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            index: context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{} indices", mesh.name)),
                    contents: bytemuck::cast_slice(&mesh.indices),
                    usage: wgpu::BufferUsages::INDEX,
                }),
            index_count: mesh.indices.len() as u32,
        }
    }
    pub fn draw<'a>(
        &self,
        params: DrawParams<'_>,
        drawables: impl Iterator<Item = (&'a GpuMesh, Mat4, Mat4, Vec3, f32, f32, Vec3, f32)>,
    ) {
        let DrawParams {
            encoder,
            target,
            view_projection,
            previous_view_projection,
            output,
            lighting,
            queue,
        } = params;
        let drawables: Vec<_> = drawables.take(MAX_SCENE_OBJECTS as usize).collect();
        let inverse_view_projection = view_projection.inverse();
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&CameraUniform {
                view_projection: view_projection.to_cols_array_2d(),
                previous_view_projection: previous_view_projection.to_cols_array_2d(),
            }),
        );
        queue.write_buffer(
            &self.lighting_buffer,
            0,
            bytemuck::bytes_of(&LightingUniform {
                inverse_view_projection: inverse_view_projection.to_cols_array_2d(),
                camera_position: lighting.camera_position.extend(1.0).to_array(),
                directional_direction: lighting.directional_direction.extend(0.0).to_array(),
                directional_color_intensity: lighting
                    .directional_color
                    .extend(lighting.directional_intensity)
                    .to_array(),
                point_position_range: lighting
                    .point_position
                    .extend(lighting.point_range)
                    .to_array(),
                point_color_intensity: lighting
                    .point_color
                    .extend(lighting.point_intensity)
                    .to_array(),
                spot_position_range: lighting
                    .spot_position
                    .extend(lighting.spot_range)
                    .to_array(),
                spot_direction_inner: lighting
                    .spot_direction
                    .normalize_or_zero()
                    .extend(lighting.spot_inner_cos)
                    .to_array(),
                spot_color_intensity: lighting
                    .spot_color
                    .extend(lighting.spot_intensity)
                    .to_array(),
                spot_outer: [lighting.spot_outer_cos, 0.0, 0.0, 0.0],
                ambient_exposure: [lighting.ambient_intensity, lighting.exposure, 0.0, 0.0],
                environment_sky: [0.12, 0.28, 0.55, lighting.environment_intensity],
                environment_ground: [0.035, 0.025, 0.018, lighting.environment_rotation],
            }),
        );
        for (index, (_, model, previous_model, color, metallic, roughness, emissive, occlusion)) in
            drawables.iter().enumerate()
        {
            queue.write_buffer(
                &self.object_buffer,
                index as u64 * OBJECT_UNIFORM_STRIDE,
                bytemuck::bytes_of(&ObjectUniform {
                    model: model.to_cols_array_2d(),
                    previous_model: previous_model.to_cols_array_2d(),
                    color_metallic: [color.x, color.y, color.z, *metallic],
                    roughness: *roughness,
                    emissive_occlusion: [emissive.x, emissive.y, emissive.z, *occlusion],
                }),
            );
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gbuffer pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.gbuffer.albedo,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.gbuffer.normal_roughness,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.gbuffer.motion,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.gbuffer.emissive_occlusion,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.gbuffer.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.geometry_pipeline);
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        for (index, (mesh, _, _, _, _, _, _, _)) in drawables.iter().enumerate() {
            pass.set_bind_group(
                1,
                &self.object_bind_group,
                &[index as u32 * OBJECT_UNIFORM_STRIDE as u32],
            );
            pass.set_vertex_buffer(0, mesh.vertex.slice(..));
            pass.set_index_buffer(mesh.index.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
        drop(pass);
        let (pipeline, bind_group) = if output == ViewportOutput::FinalComposite {
            (&self.composite_pipeline, &self.composite_bind_group)
        } else {
            (&self.debug_pipeline, &self.debug_bind_group)
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("deferred compose pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.01,
                        g: 0.015,
                        b: 0.03,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_bind_group(1, &self.object_bind_group, &[0]);
        pass.set_bind_group(2, &self.lighting_bind_group, &[]);
        pass.draw(0..3, output as u32..output as u32 + 1);
    }

    pub fn sky_target(&self) -> &wgpu::TextureView {
        &self.gbuffer.sky
    }
}

impl DebugRenderer {
    pub fn new(context: &RenderContext, shader_path: impl AsRef<Path>) -> Result<Self> {
        let shader_path = shader_path.as_ref().to_path_buf();
        let layout = context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sky uniform layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let uniform_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky uniform"),
            size: std::mem::size_of::<SkyUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sky bind group"),
                layout: &layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });
        let pipeline = Self::create_pipeline(context, &shader_path, &layout)?;
        Ok(Self {
            pipeline,
            uniform_buffer,
            bind_group,
            bind_group_layout: layout,
            shader_path,
            last_shader_error: None,
        })
    }

    fn create_pipeline(
        context: &RenderContext,
        path: &Path,
        layout: &wgpu::BindGroupLayout,
    ) -> Result<wgpu::RenderPipeline> {
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("reading shader {}", path.display()))?;
        let module = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: path.to_str(),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        Ok(context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("debug output pipeline"),
                layout: Some(&context.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: Some("sky pipeline layout"),
                        bind_group_layouts: &[layout],
                        push_constant_ranges: &[],
                    },
                )),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: context.config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            }))
    }

    pub fn reload(&mut self, context: &RenderContext) {
        match Self::create_pipeline(context, &self.shader_path, &self.bind_group_layout) {
            Ok(pipeline) => {
                self.pipeline = pipeline;
                self.last_shader_error = None;
            }
            Err(error) => self.last_shader_error = Some(format!("{error:#}")),
        }
    }

    pub fn shader_error(&self) -> Option<&str> {
        self.last_shader_error.as_deref()
    }

    pub fn draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        camera_to_world: Mat4,
        queue: &wgpu::Queue,
    ) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&SkyUniform {
                camera_to_world: camera_to_world.to_cols_array_2d(),
            }),
        );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("debug output pass"),
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
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
