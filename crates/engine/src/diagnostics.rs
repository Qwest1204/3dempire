use std::{collections::VecDeque, time::Duration};

#[derive(Debug, Default)]
pub struct FrameStatistics {
    smoothed_frame_time: Duration,
    samples: u32,
    frame_times_ms: VecDeque<f32>,
}

impl FrameStatistics {
    pub fn record_frame(&mut self, delta: Duration) {
        self.frame_times_ms.push_back(delta.as_secs_f32() * 1000.0);
        if self.frame_times_ms.len() > 180 {
            self.frame_times_ms.pop_front();
        }
        self.samples = (self.samples + 1).min(120);
        let previous = self.smoothed_frame_time.as_secs_f32();
        let weight = 1.0 / self.samples as f32;
        self.smoothed_frame_time =
            Duration::from_secs_f32(previous + (delta.as_secs_f32() - previous) * weight);
    }

    pub fn fps(&self) -> f32 {
        let seconds = self.smoothed_frame_time.as_secs_f32();
        if seconds > 0.0 { 1.0 / seconds } else { 0.0 }
    }

    pub fn frame_time_ms(&self) -> f32 {
        self.smoothed_frame_time.as_secs_f32() * 1000.0
    }

    pub fn frame_times_ms(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        self.frame_times_ms.iter().copied()
    }
}
