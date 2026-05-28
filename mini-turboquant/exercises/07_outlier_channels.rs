pub fn top_abs_indices(x: &[f32], k: usize) -> Vec<usize> {
    // TODO:
    // Return indices of the k largest absolute values, descending by abs value.
    todo!()
}

pub fn effective_bits(num_channels: usize, outlier_channels: usize, regular_bits: f32, outlier_bits: f32) -> f32 {
    // TODO:
    // Weighted average bit-width.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_outliers() {
        let x = vec![0.1, -9.0, 2.0, 7.0];
        assert_eq!(top_abs_indices(&x, 2), vec![1, 3]);
    }

    #[test]
    fn computes_effective_precision() {
        let bits = effective_bits(128, 32, 2.0, 3.0);
        assert!((bits - 2.25).abs() < 1e-6);
    }
}
