use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct FrameClock {
    last_frame: Instant,
    pub delta: Duration,
    pub elapsed: Duration,
    pub frame_index: u64,
}

impl Default for FrameClock {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            last_frame: now,
            delta: Duration::ZERO,
            elapsed: Duration::ZERO,
            frame_index: 0,
        }
    }
}

impl FrameClock {
    pub fn tick(&mut self) {
        let now = Instant::now();
        self.delta = (now - self.last_frame).min(Duration::from_millis(250));
        self.last_frame = now;
        self.elapsed += self.delta;
        self.frame_index += 1;
    }

    pub fn delta_seconds(&self) -> f32 {
        self.delta.as_secs_f32()
    }
}
