pub fn mean(_xs: &[f32]) -> f32 {
    // TODO:
    // Return average value.
    // If xs is empty, return 0.0.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_mean_reward() {
        let rewards = vec![1.0, 0.0, 1.0, 0.0];
        assert!((mean(&rewards) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn empty_mean_is_zero() {
        assert_eq!(mean(&[]), 0.0);
    }
}
