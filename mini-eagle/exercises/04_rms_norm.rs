pub fn rms_norm(x: &[f32], eps: f32) -> Vec<f32> {
    // TODO:
    // rms = sqrt(mean(x_i^2) + eps)
    // output_i = x_i / rms
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(x: &[f32]) -> f32 {
        let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        mean_sq.sqrt()
    }

    #[test]
    fn normalizes_rms_close_to_one() {
        let y = rms_norm(&[3.0, 4.0], 1e-6);
        assert!((rms(&y) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn preserves_direction_ratio() {
        let y = rms_norm(&[2.0, 4.0], 1e-6);
        assert!((y[1] / y[0] - 2.0).abs() < 1e-5);
    }
}
