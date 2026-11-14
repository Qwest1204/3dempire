use anyhow::{Context, Result};
use std::path::Path;

/// CPU-side source image in linear RGB. Neural training always uses this
/// representation; the original encoded bytes are not part of the model.
#[derive(Clone, Debug)]
pub struct LinearImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[f32; 4]>,
}

impl LinearImage {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let image = image::open(path)
            .with_context(|| format!("loading image {}", path.display()))?
            .to_rgba8();
        let (width, height) = image.dimensions();
        let pixels = image
            .pixels()
            .map(|pixel| {
                let channel = |value: u8| srgb_to_linear(value as f32 / 255.0);
                [
                    channel(pixel[0]),
                    channel(pixel[1]),
                    channel(pixel[2]),
                    pixel[3] as f32 / 255.0,
                ]
            })
            .collect();
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn gradient_circle(width: u32, height: u32) -> Self {
        let pixels = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let u = x as f32 / (width - 1).max(1) as f32;
                    let v = y as f32 / (height - 1).max(1) as f32;
                    let dx = (u - 0.5) * 2.0;
                    let dy = (v - 0.5) * 2.0;
                    let radius = (dx * dx + dy * dy).sqrt();
                    let mask = (1.0 - (radius - 0.72) / 0.08).clamp(0.0, 1.0);
                    let radial = (1.0 - radius / 0.8).clamp(0.0, 1.0);
                    let angle = dy.atan2(dx);
                    [
                        (0.5 + 0.5 * angle.cos()) * radial * mask,
                        (0.5 + 0.5 * (angle + 2.094).cos()) * radial * mask,
                        (0.5 + 0.5 * (angle + 4.189).cos()) * radial * mask,
                        1.0,
                    ]
                })
            })
            .collect();
        Self {
            width,
            height,
            pixels,
        }
    }

    pub fn rgba8_srgb(&self) -> Vec<u8> {
        self.pixels
            .iter()
            .flat_map(|pixel| {
                [
                    (linear_to_srgb(pixel[0]).clamp(0.0, 1.0) * 255.0).round() as u8,
                    (linear_to_srgb(pixel[1]).clamp(0.0, 1.0) * 255.0).round() as u8,
                    (linear_to_srgb(pixel[2]).clamp(0.0, 1.0) * 255.0).round() as u8,
                    (pixel[3].clamp(0.0, 1.0) * 255.0).round() as u8,
                ]
            })
            .collect()
    }
}

pub fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

pub fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}
