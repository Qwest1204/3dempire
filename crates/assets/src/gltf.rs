use crate::{MeshData, Vertex};
use anyhow::{Context, Result, bail};
use glam::Vec3;
use std::path::Path;

/// Imports the first triangle mesh primitive from a glTF/GLB asset.
pub fn load_first_mesh(path: impl AsRef<Path>) -> Result<MeshData> {
    let path = path.as_ref();
    let (document, buffers, _) =
        gltf::import(path).with_context(|| format!("loading {}", path.display()))?;
    let primitive = document
        .meshes()
        .flat_map(|mesh| mesh.primitives())
        .next()
        .context("asset contains no mesh primitive")?;
    if primitive.mode() != gltf::mesh::Mode::Triangles {
        bail!("only triangle primitives are supported");
    }
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .context("mesh has no POSITION attribute")?
        .collect();
    let normals: Vec<[f32; 3]> = reader
        .read_normals()
        .map(Iterator::collect)
        .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
    let uvs: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .map(|coordinates| coordinates.into_f32().collect())
        .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
    let vertices: Vec<Vertex> = positions
        .into_iter()
        .zip(normals.into_iter().zip(uvs))
        .map(|(position, (normal, uv))| {
            Vertex::with_uv(
                Vec3::from(position),
                Vec3::from(normal).normalize_or_zero(),
                uv,
            )
        })
        .collect();
    let indices = reader
        .read_indices()
        .map(|read| read.into_u32().collect())
        .unwrap_or_else(|| (0..vertices.len() as u32).collect());
    Ok(MeshData {
        name: path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("glTF mesh")
            .into(),
        vertices,
        indices,
    })
}
