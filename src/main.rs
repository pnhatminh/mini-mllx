use mlx_rs::{Device, array, assert_array_eq, ops, with_device};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello, world!");
    let a = array!([1.0_f32, 2.0, 3.0, 4.0]);
    let b = array!([10.0_f32, 20.0, 30.0, 40.0]);
    let result = with_device(Device::gpu(), || ops::add(&a, &b))?;
    assert_array_eq!(result, array!([11.0_f32, 22.0, 33.0, 44.0]));
    println!("Result: {:?}", result);
    Ok(())
}
