use mlx_rs::{Array, error::Exception, ops};

pub struct RmsNorm {
    weight: Array,
    eps: f32,
}

impl RmsNorm {
    pub fn new(weight: Array, eps: f32) -> Self {
        assert_eq!(weight.shape().len(), 1, "Weight must be an 1D array");
        assert!(eps > 0.0, "Epsilon must be positive");
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        assert_eq!(
            x.shape().last().copied(),
            Some(self.weight.shape()[0]),
            "Input size must match the weight size"
        );

        let squared = x.square()?;
        let mean_squared = squared.mean_axis(-1, true)?;
        let stabilized = mean_squared.add(Array::from_f32(self.eps))?;
        let inv_rms = ops::rsqrt(&stabilized)?;
        let normalized = x.multiply(&inv_rms)?;
        normalized.multiply(&self.weight)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_vector_is_stable() {
        let input = Array::from_slice(
            &[0.0_f32, 0.0],
            &[1, 2],
        );

        let weight = Array::from_slice(
            &[1.0_f32, 2.0],
            &[2],
        );

        let norm = RmsNorm::new(weight, 1e-6);

        let output = norm.forward(&input).unwrap();
        output.eval().unwrap();

        let values = output.as_slice::<f32>();

        assert_eq!(values, &[0.0, 0.0]);
    }
}
