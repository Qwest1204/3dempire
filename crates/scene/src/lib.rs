use glam::{Mat4, Quat, Vec3};
use hecs::{Entity, World};

#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

#[derive(Debug, Clone, Copy)]
pub struct PreviousTransform(pub Mat4);

impl Transform {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };
    pub fn matrix(self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MeshHandle(pub usize);

#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub base_color: Vec3,
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: Vec3,
    pub occlusion: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Renderable {
    pub mesh: MeshHandle,
    pub material: Material,
}

#[derive(Debug, Clone)]
pub struct RenderInstance {
    pub entity: Entity,
    pub model: Mat4,
    pub previous_model: Mat4,
    pub mesh: MeshHandle,
    pub material: Material,
}

#[derive(Debug, Default)]
pub struct RenderWorld {
    pub instances: Vec<RenderInstance>,
}

pub fn extract_render_world(world: &World) -> RenderWorld {
    let mut instances = Vec::new();
    for (entity, (transform, previous, renderable)) in
        &mut world.query::<(&Transform, &PreviousTransform, &Renderable)>()
    {
        instances.push(RenderInstance {
            entity,
            model: transform.matrix(),
            previous_model: previous.0,
            mesh: renderable.mesh,
            material: renderable.material,
        });
    }
    RenderWorld { instances }
}

pub fn advance_transform_history(world: &mut World) {
    for (_, (transform, previous)) in &mut world.query::<(&Transform, &mut PreviousTransform)>() {
        previous.0 = transform.matrix();
    }
}
