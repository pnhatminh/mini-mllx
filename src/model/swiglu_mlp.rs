use mlx_rs::{error::Exception, Array};
use super::activation::silu;
use super::linear::Linear;

pub struct SwiGluMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SwiGluMlp {
    pub fn new(gate_weight: Array, up_weight: Array, down_weight: Array) -> Self {
        let gate_shape = gate_weight.shape().to_vec();
        let up_shape = up_weight.shape().to_vec();
        let down_shape = down_weight.shape().to_vec();

        assert_eq!(gate_shape.len(), 2);
        assert_eq!(up_shape.len(), 2);
        assert_eq!(down_shape.len(), 2);

        assert_eq!(
            gate_shape, up_shape,
            "Gate and up projection shapes must match"
        );

        assert_eq!(
            down_shape[1], gate_shape[0],
            "Down shape must match intermediate size"
        );

        assert_eq!(
            down_shape[0], gate_shape[1],
            "Down shape must match hidden size"
        );

        Self {
            gate_proj: Linear::new(gate_weight),
            up_proj: Linear::new(up_weight),
            down_proj: Linear::new(down_weight),
        }
    }

    pub fn forward(&self, input: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(&input)?;
        let up = self.up_proj.forward(&input)?;
        let activated_gate = silu(&gate)?;
        let hidden = activated_gate.multiply(&up)?;
        self.down_proj.forward(&hidden)
    }

}


#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_mlp() -> SwiGluMlp {
        // Gate: [3, 2]
        // For x=[2,1], gate=[2,0,-1].
        let gate_weight = Array::from_slice(
            &[
                1.0_f32, 0.0,
                0.0,     0.0,
                0.0,    -1.0,
            ],
            &[3, 2],
        );

        // Up: [3, 2]
        // For x=[2,1], up=[2,4,3].
        let up_weight = Array::from_slice(
            &[
                1.0_f32, 0.0,
                2.0,     0.0,
                1.0,     1.0,
            ],
            &[3, 2],
        );

        // Down: [2, 3]
        let down_weight = Array::from_slice(
            &[
                1.0_f32, 1.0, 0.0,
                0.0,     1.0, 1.0,
            ],
            &[2, 3],
        );

        SwiGluMlp::new(
            gate_weight,
            up_weight,
            down_weight,
        )
    }

    #[test]
    fn matches_hand_calculated_output() {
        let mlp = make_test_mlp();

        // Two tokens, two features each.
        let input = Array::from_slice(
            &[2.0_f32, 1.0, 0.0, 1.0],
            &[2, 2],
        );

        let output = mlp.forward(&input).unwrap();
        output.eval().unwrap();

        assert_eq!(output.shape(), &[2, 2]);

        let expected = [
            3.523188_f32, -0.806824,
            0.0,          -0.268941,
        ];

        let actual = output.as_slice::<f32>();

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() < 1e-4,
                "Expected {e}, got {a}"
            );
        }
    }

    #[test]
    fn preserves_leading_dimensions() {
        let mlp = make_test_mlp();

        // One batch containing two tokens.
        let input = Array::from_slice(
            &[2.0_f32, 1.0, 0.0, 1.0],
            &[1, 2, 2],
        );

        let output = mlp.forward(&input).unwrap();

        // Hidden dimension goes 2 -> 3 -> 2.
        assert_eq!(output.shape(), &[1, 2, 2]);

        output.eval().unwrap();
    }

    #[test]
    #[should_panic(expected = "Gate and up projection shapes must match")]
    fn rejects_incompatible_projections() {
        let gate = Array::from_slice(
            &[1.0_f32; 6],
            &[3, 2],
        );

        let up = Array::from_slice(
            &[1.0_f32; 8],
            &[4, 2],
        );

        let down = Array::from_slice(
            &[1.0_f32; 6],
            &[2, 3],
        );

        SwiGluMlp::new(gate, up, down);
    }
}
