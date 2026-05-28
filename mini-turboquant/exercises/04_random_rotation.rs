pub fn rotate_2d(x: [f32; 2], theta: f32) -> [f32; 2] {
    // TODO:
    // [cos -sin; sin cos] x
    todo!()
}

pub fn l2_norm(x: &[f32]) -> f32 {
    // TODO:
    // sqrt(sum x_i^2)
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_preserves_norm() {
        let x = [3.0, 4.0];
        let y = rotate_2d(x, std::f32::consts::FRAC_PI_4);

        assert!((l2_norm(&x) - l2_norm(&y)).abs() < 1e-5);
    }
}
