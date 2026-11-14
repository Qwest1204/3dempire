use crate::image::LinearImage;

/// Deliberately small CPU reference MLP. Its packed parameter order is the
/// contract later shared with Slang storage-buffer inference.
#[derive(Clone, Debug)]
pub struct NeuralTextureModel {
    pub hidden_width: usize,
    pub parameters: Vec<f32>,
    gradients: Vec<f32>,
    adam_m: Vec<f32>,
    adam_v: Vec<f32>,
    pub step: u32,
}

impl NeuralTextureModel {
    pub fn new(hidden_width: usize) -> Self {
        let count = hidden_width * 2 + hidden_width + hidden_width * 3 + 3;
        let mut parameters = vec![0.0; count];
        for (index, weight) in parameters.iter_mut().enumerate() {
            *weight = (((index * 1_103_515_245 + 12_345) & 0xffff) as f32 / 65_535.0 - 0.5) * 0.15;
        }
        Self {
            hidden_width,
            gradients: vec![0.0; count],
            adam_m: vec![0.0; count],
            adam_v: vec![0.0; count],
            parameters,
            step: 0,
        }
    }

    fn offsets(&self) -> (usize, usize, usize, usize) {
        let w0 = 0;
        let b0 = w0 + self.hidden_width * 2;
        let w1 = b0 + self.hidden_width;
        let b1 = w1 + self.hidden_width * 3;
        (w0, b0, w1, b1)
    }

    pub fn forward(&self, uv: [f32; 2]) -> [f32; 3] {
        let (w0, b0, w1, b1) = self.offsets();
        let hidden: Vec<f32> = (0..self.hidden_width)
            .map(|unit| {
                (self.parameters[w0 + unit * 2] * uv[0]
                    + self.parameters[w0 + unit * 2 + 1] * uv[1]
                    + self.parameters[b0 + unit])
                    .max(0.0)
            })
            .collect();
        std::array::from_fn(|channel| {
            let value = (0..self.hidden_width)
                .map(|unit| hidden[unit] * self.parameters[w1 + channel * self.hidden_width + unit])
                .sum::<f32>()
                + self.parameters[b1 + channel];
            value.clamp(0.0, 1.0)
        })
    }

    pub fn train_step(
        &mut self,
        image: &LinearImage,
        batch_size: usize,
        learning_rate: f32,
    ) -> f32 {
        self.gradients.fill(0.0);
        let (w0, b0, w1, b1) = self.offsets();
        let mut loss = 0.0;
        for sample in 0..batch_size {
            let index = ((self.step as usize * 977 + sample * 7_919) % image.pixels.len())
                .min(image.pixels.len() - 1);
            let x = index % image.width as usize;
            let y = index / image.width as usize;
            let uv = [
                x as f32 / (image.width - 1).max(1) as f32,
                y as f32 / (image.height - 1).max(1) as f32,
            ];
            let target = image.pixels[index];
            let pre_hidden: Vec<f32> = (0..self.hidden_width)
                .map(|unit| {
                    self.parameters[w0 + unit * 2] * uv[0]
                        + self.parameters[w0 + unit * 2 + 1] * uv[1]
                        + self.parameters[b0 + unit]
                })
                .collect();
            let hidden: Vec<f32> = pre_hidden.iter().map(|value| value.max(0.0)).collect();
            let pre_output: [f32; 3] = std::array::from_fn(|channel| {
                (0..self.hidden_width)
                    .map(|unit| {
                        hidden[unit] * self.parameters[w1 + channel * self.hidden_width + unit]
                    })
                    .sum::<f32>()
                    + self.parameters[b1 + channel]
            });
            let output: [f32; 3] = pre_output.map(|value| value.clamp(0.0, 1.0));
            let output_gradient: [f32; 3] = std::array::from_fn(|channel| {
                let error = output[channel] - target[channel];
                loss += error * error;
                if pre_output[channel] > 0.0 && pre_output[channel] < 1.0 {
                    2.0 * error / 3.0
                } else {
                    0.0
                }
            });
            for (channel, output_gradient) in output_gradient.iter().enumerate() {
                self.gradients[b1 + channel] += *output_gradient;
                for (unit, hidden) in hidden.iter().enumerate() {
                    self.gradients[w1 + channel * self.hidden_width + unit] +=
                        *output_gradient * *hidden;
                }
            }
            for (unit, pre_hidden) in pre_hidden.iter().enumerate() {
                let hidden_gradient = (0..3)
                    .map(|channel| {
                        output_gradient[channel]
                            * self.parameters[w1 + channel * self.hidden_width + unit]
                    })
                    .sum::<f32>();
                if *pre_hidden > 0.0 {
                    self.gradients[b0 + unit] += hidden_gradient;
                    self.gradients[w0 + unit * 2] += hidden_gradient * uv[0];
                    self.gradients[w0 + unit * 2 + 1] += hidden_gradient * uv[1];
                }
            }
        }
        self.step += 1;
        let scale = 1.0 / batch_size as f32;
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        for index in 0..self.parameters.len() {
            let gradient = self.gradients[index] * scale;
            self.adam_m[index] = beta1 * self.adam_m[index] + (1.0 - beta1) * gradient;
            self.adam_v[index] = beta2 * self.adam_v[index] + (1.0 - beta2) * gradient * gradient;
            let m = self.adam_m[index] / (1.0 - beta1.powi(self.step as i32));
            let v = self.adam_v[index] / (1.0 - beta2.powi(self.step as i32));
            self.parameters[index] -= learning_rate * m / (v.sqrt() + 1e-8);
        }
        loss * scale / 3.0
    }
}
