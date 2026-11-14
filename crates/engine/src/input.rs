use std::collections::HashSet;
use winit::{
    event::{DeviceEvent, ElementState, MouseButton, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
};

#[derive(Debug, Default)]
pub struct InputState {
    pressed: HashSet<KeyCode>,
    pub look_active: bool,
    pub mouse_delta: (f32, f32),
    pub wheel_delta: f32,
}

impl InputState {
    pub fn begin_frame(&mut self) {
        self.mouse_delta = (0.0, 0.0);
        self.wheel_delta = 0.0;
    }

    pub fn key_down(&self, key: KeyCode) -> bool {
        self.pressed.contains(&key)
    }

    pub fn window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            self.pressed.insert(key);
                        }
                        ElementState::Released => {
                            self.pressed.remove(&key);
                        }
                    }
                }
            }
            WindowEvent::MouseInput {
                button: MouseButton::Right,
                state,
                ..
            } => {
                self.look_active = *state == ElementState::Pressed;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.wheel_delta += match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 40.0,
                };
            }
            _ => {}
        }
    }

    pub fn device_event(&mut self, event: &DeviceEvent) {
        if self.look_active
            && let DeviceEvent::MouseMotion { delta } = event
        {
            self.mouse_delta.0 += delta.0 as f32;
            self.mouse_delta.1 += delta.1 as f32;
        }
    }
}
