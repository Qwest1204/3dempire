use crate::{MeshData, Vertex};
use glam::Vec3;

pub fn cube() -> MeshData {
    let faces = [
        (
            Vec3::X,
            [
                Vec3::new(1., -1., -1.),
                Vec3::new(1., -1., 1.),
                Vec3::new(1., 1., 1.),
                Vec3::new(1., 1., -1.),
            ],
        ),
        (
            -Vec3::X,
            [
                Vec3::new(-1., -1., 1.),
                Vec3::new(-1., -1., -1.),
                Vec3::new(-1., 1., -1.),
                Vec3::new(-1., 1., 1.),
            ],
        ),
        (
            Vec3::Y,
            [
                Vec3::new(-1., 1., -1.),
                Vec3::new(1., 1., -1.),
                Vec3::new(1., 1., 1.),
                Vec3::new(-1., 1., 1.),
            ],
        ),
        (
            -Vec3::Y,
            [
                Vec3::new(-1., -1., 1.),
                Vec3::new(1., -1., 1.),
                Vec3::new(1., -1., -1.),
                Vec3::new(-1., -1., -1.),
            ],
        ),
        (
            Vec3::Z,
            [
                Vec3::new(1., -1., 1.),
                Vec3::new(-1., -1., 1.),
                Vec3::new(-1., 1., 1.),
                Vec3::new(1., 1., 1.),
            ],
        ),
        (
            -Vec3::Z,
            [
                Vec3::new(-1., -1., -1.),
                Vec3::new(1., -1., -1.),
                Vec3::new(1., 1., -1.),
                Vec3::new(-1., 1., -1.),
            ],
        ),
    ];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (face, (normal, points)) in faces.into_iter().enumerate() {
        let start = (face * 4) as u32;
        vertices.extend(points.into_iter().map(|point| Vertex::new(point, normal)));
        indices.extend_from_slice(&[start, start + 2, start + 1, start, start + 3, start + 2]);
    }
    MeshData {
        name: "Cube".into(),
        vertices,
        indices,
    }
}

pub fn plane() -> MeshData {
    let normal = Vec3::Y;
    let vertices = vec![
        Vertex::new(Vec3::new(-1., 0., -1.), normal),
        Vertex::new(Vec3::new(1., 0., -1.), normal),
        Vertex::new(Vec3::new(1., 0., 1.), normal),
        Vertex::new(Vec3::new(-1., 0., 1.), normal),
    ];
    MeshData {
        name: "Plane".into(),
        vertices,
        indices: vec![0, 2, 1, 0, 3, 2],
    }
}

pub fn uv_sphere(segments: u32, rings: u32) -> MeshData {
    let mut vertices = Vec::with_capacity(((segments + 1) * (rings + 1)) as usize);
    let mut indices = Vec::with_capacity((segments * rings * 6) as usize);
    for y in 0..=rings {
        let v = y as f32 / rings as f32;
        let theta = v * std::f32::consts::PI;
        for x in 0..=segments {
            let u = x as f32 / segments as f32;
            let phi = u * std::f32::consts::TAU;
            let normal = Vec3::new(
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            );
            vertices.push(Vertex::new(normal, normal));
        }
    }
    for y in 0..rings {
        for x in 0..segments {
            let a = y * (segments + 1) + x;
            let b = a + segments + 1;
            indices.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    MeshData {
        name: "UV Sphere".into(),
        vertices,
        indices,
    }
}
