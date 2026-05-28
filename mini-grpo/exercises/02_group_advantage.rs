fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f32>() / xs.len() as f32
}

pub fn stddev(_xs: &[f32]) -> f32 {
    // TODO:
    // Population standard deviation.
    // If variance is zero, return 0.0.
    todo!()
}

pub fn group_advantages(_rewards: &[f32]) -> Vec<f32> {
    // TODO:
    // A_i = (r_i - mean) / std
    // If std is zero, return all zeros.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_rewards() {
        let rewards = vec![1.0, 0.0, 1.0, 0.0];
        let adv = group_advantages(&rewards);

        assert_eq!(adv.len(), 4);
        assert!(adv[0] > 0.0);
        assert!(adv[1] < 0.0);
        assert!(adv[2] > 0.0);
        assert!(adv[3] < 0.0);
    }

    #[test]
    fn zero_variance_gives_no_signal() {
        assert_eq!(group_advantages(&[1.0, 1.0, 1.0]), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn computes_population_stddev() {
        assert!((stddev(&[1.0, 0.0, 1.0, 0.0]) - 0.5).abs() < 1e-6);
    }
}
