mod model;
use mlx_rs::{Array, Device, with_device};
use model::rms_norm::RmsNorm;
use std::error::Error;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = Array::from_slice(&[3.0_f32, 4.0, 6.0, 8.0], &[2, 2]);
    let weight = Array::from_slice(&[1.0_f32, 2.0], &[2]);

    let norm = RmsNorm::new(weight, 1e-6);
    let output = with_device(Device::gpu(), || norm.forward(&input))?;

    output.eval()?;

    let actual = output.as_slice::<f32>();

    let expected = [0.848528, 2.262742, 0.848528, 2.262742];

    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-4, "Expected {e}, got {a}");
    }

    println!("Input:  {:?}", input);
    println!("Output: {:?}", actual);
    println!("M1.1 PASSED");

    Ok(())
}
