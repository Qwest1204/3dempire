use crate::input::InputState;
use glam::{Mat4, Vec3};
use winit::keyboard::KeyCode;

#[derive(Debug)]
pub struct FlyCamera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y_radians: f32,
    pub near: f32,
    pub far: f32,
    pub speed: f32,
    pub sensitivity: f32,
}

impl Default for FlyCamera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 1.5, 5.0),
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: -0.15,
            fov_y_radians: 60_f32.to_radians(),
            near: 0.05,
            far: 2_000.0,
            speed: 5.0,
            sensitivity: 0.0025,
        }
    }
}

impl FlyCamera {
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward(), Vec3::Y)
    }
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y_radians, aspect, self.near, self.far)
    }

    pub fn update(&mut self, input: &InputState, delta_seconds: f32) {
        self.speed = (self.speed * (1.0 + input.wheel_delta * 0.1)).clamp(0.1, 500.0);
        if input.look_active {
            self.yaw += input.mouse_delta.0 * self.sensitivity;
            self.pitch = (self.pitch - input.mouse_delta.1 * self.sensitivity).clamp(-1.55, 1.55);
        }
        let forward = self.forward();
        let right = forward.cross(Vec3::Y).normalize();
        let mut movement = Vec3::ZERO;
        if input.key_down(KeyCode::KeyW) {
            movement += forward;
        }
        if input.key_down(KeyCode::KeyS) {
            movement -= forward;
        }
        if input.key_down(KeyCode::KeyD) {
            movement += right;
        }
        if input.key_down(KeyCode::KeyA) {
            movement -= right;
        }
        if input.key_down(KeyCode::Space) {
            movement += Vec3::Y;
        }
        if input.key_down(KeyCode::ControlLeft) {
            movement -= Vec3::Y;
        }
        if movement.length_squared() > 0.0 {
            let multiplier = if input.key_down(KeyCode::ShiftLeft) {
                4.0
            } else {
                1.0
            };
            self.position += movement.normalize() * self.speed * multiplier * delta_seconds;
        }
    }
}
