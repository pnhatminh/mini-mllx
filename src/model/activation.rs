use mlx_rs::{ops, Array, error::Exception};

pub fn sigmoid(input: &Array) -> Result<Array, Exception> {
    let neg_x = ops::negative(input)?;
    let exp_neg_x = ops::exp(neg_x)?;
    let one = Array::from(1.0);
    let denominator = ops::add(&one, &exp_neg_x)?;
    ops::divide(&one, &denominator)
}

pub fn silu(input: &Array) -> Result<Array, Exception> {
    let sigmoid_x = sigmoid(&input)?;
    ops::multiply(&input, &sigmoid_x)
}


#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::{with_device, Device, nn};

    // Independent scalar reference implementation.
    fn reference_silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    #[test]
    fn silu_matches_scalar_reference() {
        let values = [-2.0_f32, -1.0, 0.0, 1.0, 2.0];

        let input = Array::from_slice(&values, &[5]);

        let output = with_device(Device::cpu(), || {
            silu(&input)
        }).unwrap();

        output.eval().unwrap();

        for (actual, x) in output
            .as_slice::<f32>()
            .iter()
            .zip(values.iter())
        {
            let expected = reference_silu(*x);

            assert!(
                (actual - expected).abs() < 1e-5,
                "For x={x}, expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn silu_preserves_tensor_shape() {
        let input = Array::from_slice(
            &[1.0_f32, -1.0, 2.0, -2.0, 0.0, 3.0],
            &[2, 3],
        );

        let output = silu(&input).unwrap();

        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn silu_matches_mlx_builtin() {
        let input = Array::from_slice(
            &[-2.0_f32, -1.0, 0.0, 1.0, 2.0],
            &[5],
        );

        let ours = silu(&input).unwrap();
        let builtin = nn::silu(&input).unwrap();

        ours.eval().unwrap();
        builtin.eval().unwrap();

        for (a, b) in ours
            .as_slice::<f32>()
            .iter()
            .zip(builtin.as_slice::<f32>().iter())
        {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
