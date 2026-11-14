# 3dempire — Contributor and Agent Guide

## Project purpose

3dempire is an experimental Rust graphics laboratory. Its current emphasis is
research into **neural textures**, neural materials, and eventually neural
shaders, rather than feature-complete game-engine development. The project
uses Rust for application, asset, and CPU-reference code; `wgpu` for graphics;
and Slang as the canonical shader source language.

The present neural-texture milestone is deliberately narrow and reproducible:

```text
linear source image
  -> CPU reference MLP training
  -> packed runtime GPU parameter buffer
  -> Slang fragment inference
  -> reconstruction / error visualisation
```

Keep this path understandable and measurable. Do not prematurely replace it
with opaque GPU-only training or model-specific generated shader constants.

## Repository map

```text
apps/
  neural_lab/        Interactive 2D neural-texture experiment.
  sandbox/           Interactive deferred-rendering sandbox.

crates/
  assets/            Image I/O, colour-space conversion, neural CPU models.
  editor_ui/         Editor-oriented UI support.
  engine/            Higher-level engine structures.
  renderer/          wgpu device/context and deferred rendering path.
  scene/             Scene data/import integration.
  shaders/           Shader-related Rust helpers.

shaders/
  slang/             Canonical, hand-authored Slang source files.
  generated/         Generated WGSL; ignored by Git, never edit manually.
  shader_manifest.toml
                       Slang compile entries, definitions, and variants.

tools/xtask/         Shader compilation and headless generated-WGSL validation.
assets/              Runtime assets, including imported scenes.
web/                 Web-facing project material, if applicable.
```

## Rust workspace conventions

- The root `Cargo.toml` defines the workspace. Use workspace dependency
  versions where they already exist instead of adding per-crate versions.
- The workspace uses Rust edition 2024.
- Keep formatting idiomatic (`cargo fmt --all`) and code lint-clean under the
  strict Clippy command in the verification section.
- Avoid unnecessary dependencies. Prefer small local types/functions for an
  experiment over pulling in a framework.
- Never hide correctness problems behind `#[allow(...)]` unless there is a
  documented platform/API reason and the narrowest possible scope is used.
- Public neural-model behaviour should be documented where a CPU/GPU layout or
  mathematical convention must remain identical.

## Shader source and generation rules

### Slang is canonical

All new production shaders belong in `shaders/slang/`. Add each source file to
`shaders/shader_manifest.toml` with its output and entry points. Compile with:

```sh
cargo run -p xtask -- shaders
```

Required Slang baseline: **2026.12**. The compiler is found through `SLANGC`
or `PATH`; use `SLANGC=/absolute/path/to/slangc` when needed.

### Generated WGSL is disposable

`shaders/generated/` is Git-ignored. It is a build product, not source:

- do not edit generated `.wgsl` files;
- do not commit generated files;
- regenerate after editing Slang or the manifest;
- use `wgpu::include_wgsl!` only for generated output that comes from a
  manifest entry.

Validate generated shaders and representative pipelines without opening a UI:

```sh
cargo run -p xtask -- validate-shaders
```

### Shader layouts are contracts

When Rust and Slang share buffers, update both sides in the same change:

- Rust `#[repr(C)]` uniform/storage structures;
- Slang buffer declarations and field order;
- bind-group layouts, binding numbers, and shader manifest if applicable;
- any CPU parameter offset calculation.

Use `bytemuck::{Pod, Zeroable}` for Rust data written as uniforms when the
layout is plain-data compatible. Maintain explicit padding where required by
WGSL/Slang uniform layout rules.

## Neural-texture experiment policy

### Preserve old experiments

Neural experiments are educational reference points. **Never overwrite an
existing neural model or shader to introduce a new architecture.** Add a new,
independent Rust model and Slang shader instead, then explicitly select it in
the lab application.

Current distinct experiments:

```text
Baseline experiment
  Rust:   crates/assets/src/neural.rs
  Slang:  shaders/slang/neural_texture.slang
  Shape:  2 -> 32 -> 3

Fourier-feature experiment (active lab model)
  Rust:   crates/assets/src/neural_fourier.rs
  Slang:  shaders/slang/neural_fourier_texture.slang
  Shape:  14 -> 32 -> 32 -> 3
```

Keep their parameter layouts, activation functions, and source files separate.
This is intentional even when code might otherwise be shared.

### Active Fourier model specification

The active model is `FourierNeuralTextureModel` in
`crates/assets/src/neural_fourier.rs`.

Input encoding, in this exact order:

```text
x, y,
sin(2x), cos(2x), sin(2y), cos(2y),
sin(4x), cos(4x), sin(4y), cos(4y),
sin(8x), cos(8x), sin(8y), cos(8y)
```

Network:

```text
14 -> Linear(32) -> ReLU -> Linear(32) -> ReLU -> Linear(3)
   -> exp(clamp(value, -16, 0))
```

The exponential final activation is a required property inherited from the
original neural-material experiment. Output biases start in log space at
`ln(0.08)` so initial RGB is neutral and dark rather than `exp(0) == 1`.

Packed parameter ordering must remain identical in Rust and Slang:

```text
W0: 14 x 32
B0: 32
W1: 32 x 32
B1: 32
W2: 32 x 3
B2: 3
```

Before changing dimensions, encodings, activations, or ordering, introduce a
new experiment rather than mutating this one. If a bug must be fixed, update
the CPU and GPU implementations together and add/extend parity coverage.

### Training and colour-space invariants

- Train against **linear RGB** values in `LinearImage`.
- Source images are decoded from sRGB to linear on load.
- The display source texture stores sRGB-encoded `Rgba8Unorm` texels; the
  Fourier Slang shader converts sampled source colour back to linear before
  computing error.
- MLP reconstruction/error is converted from linear RGB to sRGB only for UI
  presentation.
- The default lab source is `LinearImage::gradient_circle(512, 512)`.
- Loading an explicit image remains supported:

  ```sh
  cargo run -p neural_lab -- path/to/image.png
  ```

- The default training subset is deterministic: image indices for which
  `index % 10 == 0`. This is approximately 10% of texel centres.
- Do not quietly change this subset to all pixels or random sampling. Make
  sampling behaviour explicit in UI/configuration and retain reproducibility.
- Training presently uses CPU Adam as the reference implementation. GPU
  compute training is a later milestone, after measured CPU/GPU inference
  parity and checkpointing are stable.

### Where responsibilities live

```text
crates/assets/src/image.rs
  LinearImage storage, sRGB/linear conversion, procedural gradient circle.

crates/assets/src/neural_fourier.rs
  Fourier encoding, CPU forward pass, MSE backpropagation, Adam state/update.

apps/neural_lab/src/main.rs
  Window/UI, deterministic 10% training subset, training scheduling,
  full-image MSE/PSNR, uploading model parameters to wgpu, viewport rendering.

shaders/slang/neural_fourier_texture.slang
  GPU inference that must mirror CPU encoding/layout/activation exactly.
```

`NeuralRenderer` uploads `model.parameters` to a runtime storage buffer. Do
not embed trained weights into Slang shader source except for a deliberately
separate export experiment with explicit documentation.

### Required future neural safeguards

When extending the active experiment, prioritise these in order:

1. Numeric CPU/GPU parity: render/read back a small UV grid, compare with
   `FourierNeuralTextureModel::forward`, report mean/max absolute error, and
   enforce an explicit epsilon.
2. Portable checkpoint save/load: model metadata, parameter vector, Adam
   state if training resume is supported, step, Fourier frequencies, colour
   space, dimensions, and metrics. Reconstruction-only use must work without
   retaining the original source texture.
3. CPU trainer efficiency: replace per-sample temporary `Vec`s with fixed
   `[f32; 32]` arrays or reusable buffers without changing numerical meaning.
4. Only then consider Slang compute forward/backward/Adam training.

## Rendering and asset conventions

- The deferred renderer has production Slang sources and generated WGSL only;
  do not introduce a hand-written WGSL fallback for it.
- Existing phase-6 foundations include imported glTF UVs, material emissive and
  occlusion data, and an environment/IBL-like lighting path. Maintain vertex
  ABI compatibility across importer, CPU structs, Rust GPU buffers, and Slang.
- Preserve the existing safe background shader-reload diagnostics workflow.
- Prefer scoped research additions over broad engine/editor/physics work.
  Physics, TAA, and large-scene features are currently out of scope unless the
  task explicitly asks for them.

## Standard commands

Run from the workspace root.

```sh
# Regenerate all ignored WGSL from canonical Slang.
cargo run -p xtask -- shaders

# Validate generated WGSL modules and representative pipelines headlessly.
cargo run -p xtask -- validate-shaders

# Format all Rust code.
cargo fmt --all

# Type-check the entire workspace.
cargo check --workspace

# Enforce project lint policy.
cargo clippy --workspace --all-targets -- -D warnings

# Run all tests.
cargo test --workspace

# Start the neural texture lab with its default gradient-circle image.
cargo run -p neural_lab

# Start the deferred sandbox.
cargo run -p sandbox
```

The `block v0.1.6` future-incompatibility warning originates in a third-party
dependency. Do not suppress it by weakening workspace checks; distinguish it
from warnings in project code.

## Change workflow

1. Read the surrounding Rust/Slang implementation before changing ABI, model
   mathematics, or colour handling.
2. Preserve old neural experiments; create a separately named model and shader
   for a materially different experiment.
3. Modify Rust and Slang in the same change whenever a shared contract changes.
4. Regenerate WGSL after every Slang or manifest change.
5. Run formatter, checks, strict Clippy, tests, and shader validation before
   claiming a change complete. Start a GUI app when relevant; it is expected to
   keep running until manually closed.
6. Summarise user-visible behaviour, retained experiments, and verification
   commands in the final response.

## Git hygiene

- Inspect `git status` before staging.
- Do not add `target/`, IDE files, `.DS_Store`, or `shaders/generated/`; the
  root `.gitignore` already excludes these.
- Do not use destructive Git operations (`reset --hard`, broad checkout, or
  deletion of unknown work) without explicit user approval.
- Keep commits focused and describe the outcome, e.g. `Add Fourier neural
  texture reference experiment`.
- This repository may be used locally before a remote is configured. A local
  commit does not require a remote; do not invent or add one unless asked.
