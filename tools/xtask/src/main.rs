use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::{
    collections::HashSet,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Deserialize)]
struct Manifest {
    shader: Vec<Shader>,
}
#[derive(Deserialize)]
struct Shader {
    name: String,
    source: PathBuf,
    output: PathBuf,
    entries: Vec<Entry>,
    #[serde(default)]
    definitions: Vec<Definition>,
    #[serde(default)]
    variant: Vec<Variant>,
}
#[derive(Deserialize)]
struct Entry {
    name: String,
    stage: String,
}
#[derive(Deserialize)]
struct Definition {
    name: String,
    value: String,
}
#[derive(Deserialize)]
struct Variant {
    name: String,
    #[serde(default)]
    definitions: Vec<Definition>,
}

fn validate_manifest(manifest: &Manifest) -> Result<()> {
    let mut shader_names = HashSet::new();
    let mut outputs = HashSet::new();
    for shader in &manifest.shader {
        if shader.name.trim().is_empty() || !shader_names.insert(&shader.name) {
            bail!(
                "every shader must have a unique, non-empty name; got `{}`",
                shader.name
            );
        }
        if shader.entries.is_empty() {
            bail!("{} must declare at least one entry point", shader.name);
        }
        let mut entries = HashSet::new();
        for entry in &shader.entries {
            if entry.name.trim().is_empty() || !entries.insert(&entry.name) {
                bail!("{} has a duplicate or empty entry point", shader.name);
            }
        }
        validate_definitions(&shader.name, &shader.definitions)?;
        let mut variants = HashSet::new();
        for variant in &shader.variant {
            if variant.name.trim().is_empty() || !variants.insert(&variant.name) {
                bail!("{} has a duplicate or empty variant name", shader.name);
            }
            validate_definitions(
                &format!("{} [{}]", shader.name, variant.name),
                &variant.definitions,
            )?;
            let output = variant_output_path(&shader.output, &variant.name);
            if !outputs.insert(output.clone()) {
                bail!(
                    "generated shader output is duplicated: {}",
                    output.display()
                );
            }
        }
        if !outputs.insert(shader.output.clone()) {
            bail!(
                "generated shader output is duplicated: {}",
                shader.output.display()
            );
        }
    }
    Ok(())
}

fn validate_definitions(owner: &str, definitions: &[Definition]) -> Result<()> {
    let mut names = HashSet::new();
    for definition in definitions {
        if definition.name.trim().is_empty()
            || !definition.name.chars().all(|character| {
                character.is_ascii_uppercase() || character == '_' || character.is_ascii_digit()
            })
            || !names.insert(&definition.name)
        {
            bail!(
                "{owner} has an invalid or duplicate compile definition `{}`",
                definition.name
            );
        }
        if definition.value.trim().is_empty() {
            bail!("{owner} has an empty value for `{}`", definition.name);
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    match env::args().nth(1).as_deref() {
        Some("shaders") => compile_shaders(),
        Some("validate-shaders") => validate_generated_shaders(),
        _ => {
            eprintln!("Usage: cargo run -p xtask -- shaders | validate-shaders");
            Ok(())
        }
    }
}

fn validate_generated_shaders() -> Result<()> {
    let root = workspace_root();
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .context("requesting headless wgpu adapter for shader validation")?;
    let (device, _) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("shader validation device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    }))
    .context("creating headless wgpu device for shader validation")?;

    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let sky = shader_module(&device, root.join("shaders/generated/debug_output.wgsl"))?;
    let final_shader = shader_module(&device, root.join("shaders/generated/deferred.final.wgsl"))?;
    let debug_shader = shader_module(&device, root.join("shaders/generated/deferred.debug.wgsl"))?;
    let _ = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("validate sky pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &sky,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &sky,
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
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: 32,
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
    };
    let _ = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("validate deferred gbuffer pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &final_shader,
            entry_point: Some("gbuffer_vs"),
            buffers: &[vertex_layout],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &final_shader,
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
        primitive: Default::default(),
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
    for (module, label) in [
        (&final_shader, "validate deferred final pipeline"),
        (&debug_shader, "validate deferred debug pipeline"),
    ] {
        let _ = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: None,
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("fullscreen_vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some("composite_fs"),
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
    }
    device.poll(wgpu::PollType::wait_indefinitely())?;
    if let Some(error) = pollster::block_on(device.pop_error_scope()) {
        bail!("generated WGSL failed headless wgpu validation: {error}");
    }
    println!("Validated generated WGSL modules and render pipelines with headless wgpu.");
    Ok(())
}

fn shader_module(device: &wgpu::Device, path: PathBuf) -> Result<wgpu::ShaderModule> {
    let source =
        fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: path.to_str(),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    }))
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

fn compile_shaders() -> Result<()> {
    let root = workspace_root();
    let manifest_path = root.join("shaders/shader_manifest.toml");
    let manifest: Manifest =
        toml::from_str(&fs::read_to_string(&manifest_path).context("reading shader manifest")?)
            .context("parsing shader manifest")?;
    validate_manifest(&manifest)?;
    let slangc = env::var_os("SLANGC").unwrap_or_else(|| "slangc".into());
    let version = Command::new(&slangc)
        .arg("-v")
        .output()
        .context("starting slangc; set SLANGC if it is not on PATH")?;
    if !version.status.success() {
        bail!("slangc -v failed");
    }
    let version_text = String::from_utf8_lossy(&version.stdout);
    let version_text = if version_text.trim().is_empty() {
        String::from_utf8_lossy(&version.stderr).trim().to_owned()
    } else {
        version_text.trim().to_owned()
    };
    if version_text != "2026.12" {
        bail!("unsupported Slang version `{version_text}`; expected 2026.12");
    }
    println!("Using Slang {version_text}");
    for shader in manifest.shader {
        compile_shader(&root, &slangc, &shader, None)?;
        for variant in &shader.variant {
            compile_shader(&root, &slangc, &shader, Some(variant))?;
        }
    }
    Ok(())
}

fn compile_shader(
    root: &Path,
    slangc: &std::ffi::OsStr,
    shader: &Shader,
    variant: Option<&Variant>,
) -> Result<()> {
    let output_path = match variant {
        Some(variant) => variant_output_path(&shader.output, &variant.name),
        None => shader.output.clone(),
    };
    let output = root.join(&output_path);
    fs::create_dir_all(output.parent().context("shader output has no parent")?)?;
    let mut command = Command::new(slangc);
    command
        .current_dir(root)
        .arg("-target")
        .arg("wgsl")
        .arg("-matrix-layout-column-major");
    for entry in &shader.entries {
        let stage = match entry.stage.as_str() {
            "vertex" => "vertex",
            "fragment" => "fragment",
            "compute" => "compute",
            value => bail!("{}: unsupported stage {value}", shader.name),
        };
        command
            .arg("-entry")
            .arg(&entry.name)
            .arg("-stage")
            .arg(stage);
    }
    for definition in &shader.definitions {
        command
            .arg("-D")
            .arg(format!("{}={}", definition.name, definition.value));
    }
    if let Some(variant) = variant {
        for definition in &variant.definitions {
            command
                .arg("-D")
                .arg(format!("{}={}", definition.name, definition.value));
        }
    }
    let result = command
        .arg("-o")
        .arg(&output)
        .arg(&shader.source)
        .output()
        .with_context(|| format!("compiling {}", shader.name))?;
    if !result.status.success() {
        bail!(
            "Slang compilation failed for {}:\n{}",
            shader.name,
            String::from_utf8_lossy(&result.stderr)
        );
    }
    println!(
        "Compiled {}{} -> {}",
        shader.source.display(),
        variant
            .map(|variant| format!(" [{}]", variant.name))
            .unwrap_or_default(),
        output_path.display()
    );
    Ok(())
}

fn variant_output_path(output: &Path, variant: &str) -> PathBuf {
    let stem = output.file_stem().unwrap_or_default().to_string_lossy();
    let extension = output.extension().unwrap_or_default().to_string_lossy();
    output.with_file_name(format!("{stem}.{variant}.{extension}"))
}
