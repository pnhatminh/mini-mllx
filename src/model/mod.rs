use mlx_rs::{error::Exception, Array};

pub mod rms_norm;
pub mod linear;

pub struct Linear {
    weight: Array,
}

impl Linear {
    pub fn new(weight: Array) -> Self {
        assert_eq!(
            weight.shape().len(),
            2,
            "Linear weight must be 2D matrix"
        );

        Self { weight }
    }

    pub fn forward(&self, input: &Array) -> Result<Array, Exception> {
        let input_shape = input.shape();
        let weight_shape = self.weight.shape();
        assert!(
            !input_shape.is_empty(),
            "Input must have at least one dimension"
        );

        // input shape: [4,1024]
        // weight shape: [3072, 1024] => valid number of features
        let in_features = weight_shape[1];
        assert_eq!(
            // another way to write it
             *input_shape.last().unwrap(),
            in_features,
            // input_shape.last().copied(),
            // Some(in_features),
            "Input features must match weight"
        );

        let weight_t = self.weight.transpose()?;
        input.matmul(&weight_t)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::{with_device, Device};

    #[test]
    fn linear_projection_produces_expected_values() {
        let input = Array::from_slice(
            &[1.0_f32, 2.0, 3.0, 4.0],
            &[2, 2],
        );

        // Shape [3, 2]
        let weight = Array::from_slice(
            &[
                1.0_f32, 0.0,
                0.0, 1.0,
                1.0, 1.0,
            ],
            &[3, 2],
        );

        let linear = Linear::new(weight);

        let output = with_device(Device::cpu(), || {
            linear.forward(&input)
        })
        .unwrap();

        output.eval().unwrap();

        assert_eq!(output.shape(), &[2, 3]);

        let expected = [
            1.0_f32, 2.0, 3.0,
            3.0, 4.0, 7.0,
        ];

        let actual = output.as_slice::<f32>();

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() < 1e-5,
                "Expected {e}, got {a}"
            );
        }
    }
}
