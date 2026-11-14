use bytemuck::{Pod, Zeroable};
use glam::Vec3;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl Vertex {
    pub fn new(position: Vec3, normal: Vec3) -> Self {
        Self {
            position: position.to_array(),
            normal: normal.to_array(),
            uv: [0.0, 0.0],
        }
    }

    pub fn with_uv(position: Vec3, normal: Vec3, uv: [f32; 2]) -> Self {
        Self {
            position: position.to_array(),
            normal: normal.to_array(),
            uv,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MeshData {
    pub name: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl MeshData {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}
