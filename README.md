# 3dempire

Experimental Rust 3D graphics laboratory for shader and rendering research.

## Phase 4 status

- `winit` window and event loop
- `wgpu` renderer (Metal on macOS)
- WASD + mouse fly camera
- `egui` renderer controls and frame statistics
- deferred G-buffer rendering with albedo, normal/roughness, motion, and depth views
- camera-oriented Slang sky background and production deferred Slang shaders
- Slang → WGSL compilation through a versioned shader manifest
- background Slang recompilation with safe error reporting and frame-time graph

## Run

```sh
cargo run -p xtask -- shaders
cargo run -p sandbox
```

Hold the right mouse button to look around. Use `W`, `A`, `S`, `D`, `Space`, and
`Left Ctrl` to move. Mouse wheel changes movement speed.

## Prerequisites

Use an up-to-date stable Rust toolchain. On macOS, the recommended installation is
[`rustup`](https://rustup.rs/), rather than a system/Homebrew compiler tied to a
specific LLVM runtime.

## Shaders

From phase 4, new production shaders use Slang source in `shaders/slang/`.
`shaders/shader_manifest.toml` defines entry points and outputs. Compile every
manifest entry with:

```sh
cargo run -p xtask -- shaders
```

The command locates `slangc` through `SLANGC` or `PATH`, writes WGSL to
`shaders/generated/`, and fails on a compiler error. Generated WGSL is not
tracked by Git. The current required compiler baseline is Slang `2026.12`.

Each manifest item can declare `definitions`, which are passed to Slang as
`-D name=value`. This is the stable foundation for explicit shader variants;
generated files are only reloaded after a successful background compilation.

The deferred renderer deliberately has no hand-written WGSL fallback:
`deferred.final.wgsl` and `deferred.debug.wgsl` are explicit generated variants.
Validate all generated modules and representative render pipelines without opening
the sandbox window with:

```sh
cargo run -p xtask -- validate-shaders
```

The Renderer window is height-limited and organized into collapsible sections
for viewport output, lighting, camera, statistics, and Slang diagnostics.

For a non-standard installation:

```sh
SLANGC=/absolute/path/to/slangc cargo run -p xtask -- shaders
```
