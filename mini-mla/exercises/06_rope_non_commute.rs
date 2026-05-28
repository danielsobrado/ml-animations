#![allow(unused_variables)]

pub fn rotate_2d(x: [f32; 2], theta: f32) -> [f32; 2] {
    // TODO:
    // Apply 2D rotation.
    todo!()
}

pub fn scale_then_rotate(x: [f32; 2], scale: [f32; 2], theta: f32) -> [f32; 2] {
    // TODO:
    // First scale coordinates, then rotate.
    todo!()
}

pub fn rotate_then_scale(x: [f32; 2], scale: [f32; 2], theta: f32) -> [f32; 2] {
    // TODO:
    // First rotate, then scale coordinates.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_and_scaling_do_not_commute() {
        let x = [1.0, 2.0];
        let scale = [2.0, 3.0];
        let theta = 0.7;

        let a = scale_then_rotate(x, scale, theta);
        let b = rotate_then_scale(x, scale, theta);

        assert!((a[0] - b[0]).abs() > 1e-3 || (a[1] - b[1]).abs() > 1e-3);
    }
}
