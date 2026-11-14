use crate::image::LinearImage;

pub const INPUT_WIDTH: usize = 14;
pub const HIDDEN_WIDTH: usize = 32;

/// Independent Fourier-feature experiment. It intentionally does not share
/// implementation or parameter layout with `neural::NeuralTextureModel`.
/// Architecture: 14 (UV + six sine/cosine features) → 32 → 32 → RGB.
#[derive(Clone, Debug)]
pub struct FourierNeuralTextureModel {
    pub parameters: Vec<f32>,
    gradients: Vec<f32>,
    adam_m: Vec<f32>,
    adam_v: Vec<f32>,
    pub step: u32,
}

impl FourierNeuralTextureModel {
    pub const fn parameter_count() -> usize {
        INPUT_WIDTH * HIDDEN_WIDTH
            + HIDDEN_WIDTH
            + HIDDEN_WIDTH * HIDDEN_WIDTH
            + HIDDEN_WIDTH
            + HIDDEN_WIDTH * 3
            + 3
    }

    pub fn new() -> Self {
        let mut parameters = vec![0.0; Self::parameter_count()];
        for (index, value) in parameters.iter_mut().enumerate() {
            *value = (((index.wrapping_mul(1_103_515_245).wrapping_add(12_345)) & 0xffff) as f32
                / 65_535.0
                - 0.5)
                * 0.12;
        }
        // The output activation is exp(). Initialize its bias in log space so
        // the network starts at a dark, neutral color instead of exp(0) = 1.
        let (_, _, _, _, _, output_bias) = Self::offsets();
        parameters[output_bias..output_bias + 3].fill(0.08_f32.ln());
        let count = parameters.len();
        Self {
            parameters,
            gradients: vec![0.0; count],
            adam_m: vec![0.0; count],
            adam_v: vec![0.0; count],
            step: 0,
        }
    }

    fn offsets() -> (usize, usize, usize, usize, usize, usize) {
        let w0 = 0;
        let b0 = w0 + INPUT_WIDTH * HIDDEN_WIDTH;
        let w1 = b0 + HIDDEN_WIDTH;
        let b1 = w1 + HIDDEN_WIDTH * HIDDEN_WIDTH;
        let w2 = b1 + HIDDEN_WIDTH;
        let b2 = w2 + HIDDEN_WIDTH * 3;
        (w0, b0, w1, b1, w2, b2)
    }

    pub fn encode(uv: [f32; 2]) -> [f32; INPUT_WIDTH] {
        let mut encoded = [0.0; INPUT_WIDTH];
        encoded[0] = uv[0];
        encoded[1] = uv[1];
        for (index, frequency) in [2.0_f32, 4.0, 8.0].into_iter().enumerate() {
            let x = uv[0] * frequency;
            let y = uv[1] * frequency;
            let base = 2 + index * 4;
            encoded[base] = x.sin();
            encoded[base + 1] = x.cos();
            encoded[base + 2] = y.sin();
            encoded[base + 3] = y.cos();
        }
        encoded
    }

    pub fn forward(&self, uv: [f32; 2]) -> [f32; 3] {
        let input = Self::encode(uv);
        let (w0, b0, w1, b1, w2, b2) = Self::offsets();
        let first: Vec<f32> = (0..HIDDEN_WIDTH)
            .map(|unit| {
                (0..INPUT_WIDTH)
                    .map(|feature| {
                        self.parameters[w0 + unit * INPUT_WIDTH + feature] * input[feature]
                    })
                    .sum::<f32>()
                    + self.parameters[b0 + unit]
            })
            .map(|value| value.max(0.0))
            .collect();
        let second: Vec<f32> = (0..HIDDEN_WIDTH)
            .map(|unit| {
                (0..HIDDEN_WIDTH)
                    .map(|input| self.parameters[w1 + unit * HIDDEN_WIDTH + input] * first[input])
                    .sum::<f32>()
                    + self.parameters[b1 + unit]
            })
            .map(|value| value.max(0.0))
            .collect();
        std::array::from_fn(|channel| {
            ((0..HIDDEN_WIDTH)
                .map(|unit| self.parameters[w2 + channel * HIDDEN_WIDTH + unit] * second[unit])
                .sum::<f32>()
                + self.parameters[b2 + channel])
                .clamp(-16.0, 0.0)
                .exp()
        })
    }

    pub fn train_step(
        &mut self,
        image: &LinearImage,
        training_indices: &[usize],
        batch_size: usize,
        learning_rate: f32,
    ) -> f32 {
        self.gradients.fill(0.0);
        let (w0, b0, w1, b1, w2, b2) = Self::offsets();
        let mut loss = 0.0;
        for sample in 0..batch_size {
            let index = training_indices
                [(self.step as usize * 977 + sample * 7_919) % training_indices.len()];
            let x = index % image.width as usize;
            let y = index / image.width as usize;
            let input = Self::encode([
                x as f32 / (image.width - 1).max(1) as f32,
                y as f32 / (image.height - 1).max(1) as f32,
            ]);
            let pre0: Vec<f32> = (0..HIDDEN_WIDTH)
                .map(|unit| {
                    (0..INPUT_WIDTH)
                        .map(|feature| {
                            self.parameters[w0 + unit * INPUT_WIDTH + feature] * input[feature]
                        })
                        .sum::<f32>()
                        + self.parameters[b0 + unit]
                })
                .collect();
            let h0: Vec<f32> = pre0.iter().map(|value| value.max(0.0)).collect();
            let pre1: Vec<f32> = (0..HIDDEN_WIDTH)
                .map(|unit| {
                    (0..HIDDEN_WIDTH)
                        .map(|feature| {
                            self.parameters[w1 + unit * HIDDEN_WIDTH + feature] * h0[feature]
                        })
                        .sum::<f32>()
                        + self.parameters[b1 + unit]
                })
                .collect();
            let h1: Vec<f32> = pre1.iter().map(|value| value.max(0.0)).collect();
            let pre_out: [f32; 3] = std::array::from_fn(|channel| {
                (0..HIDDEN_WIDTH)
                    .map(|unit| self.parameters[w2 + channel * HIDDEN_WIDTH + unit] * h1[unit])
                    .sum::<f32>()
                    + self.parameters[b2 + channel]
            });
            // Exponential output matches the original neural-material model.
            // The clamp bounds avoid overflow; output remains in [~0, 1].
            let out = pre_out.map(|value| value.clamp(-16.0, 0.0).exp());
            let target = image.pixels[index];
            let grad_out: [f32; 3] = std::array::from_fn(|channel| {
                let error = out[channel] - target[channel];
                loss += error * error;
                if (-16.0..0.0).contains(&pre_out[channel]) {
                    2.0 * error * out[channel] / 3.0
                } else {
                    0.0
                }
            });
            for (channel, gradient) in grad_out.iter().enumerate() {
                self.gradients[b2 + channel] += *gradient;
                for (unit, value) in h1.iter().enumerate() {
                    self.gradients[w2 + channel * HIDDEN_WIDTH + unit] += *gradient * *value;
                }
            }
            let grad_h1: Vec<f32> = (0..HIDDEN_WIDTH)
                .map(|unit| {
                    (0..3)
                        .map(|channel| {
                            grad_out[channel] * self.parameters[w2 + channel * HIDDEN_WIDTH + unit]
                        })
                        .sum()
                })
                .collect();
            for (unit, gradient) in grad_h1.iter().enumerate() {
                if pre1[unit] > 0.0 {
                    self.gradients[b1 + unit] += *gradient;
                    for (feature, value) in h0.iter().enumerate() {
                        self.gradients[w1 + unit * HIDDEN_WIDTH + feature] += *gradient * *value;
                    }
                }
            }
            let grad_h0: Vec<f32> = (0..HIDDEN_WIDTH)
                .map(|unit| {
                    (0..HIDDEN_WIDTH)
                        .map(|next| {
                            grad_h1[next] * self.parameters[w1 + next * HIDDEN_WIDTH + unit]
                        })
                        .sum()
                })
                .collect();
            for (unit, gradient) in grad_h0.iter().enumerate() {
                if pre0[unit] > 0.0 {
                    self.gradients[b0 + unit] += *gradient;
                    for (feature, value) in input.iter().enumerate() {
                        self.gradients[w0 + unit * INPUT_WIDTH + feature] += *gradient * *value;
                    }
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

impl Default for FourierNeuralTextureModel {
    fn default() -> Self {
        Self::new()
    }
}
