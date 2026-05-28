#![allow(unused)]

pub fn mean(xs: &[f32]) -> f32 {
    // TODO:
    // Average, or 0.0 for empty.
    todo!()
}

pub fn stddev(xs: &[f32]) -> f32 {
    // TODO:
    // Population standard deviation.
    todo!()
}

pub fn normalized_advantages(rewards: &[f32]) -> Vec<f32> {
    // TODO:
    // A_i = (r_i - mean) / std.
    // If std is zero, return zeros.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_rewards_give_positive_and_negative_advantages() {
        let adv = normalized_advantages(&[1.0, 0.0, 1.0, 0.0]);

        assert!(adv[0] > 0.0);
        assert!(adv[1] < 0.0);
        assert!(adv[2] > 0.0);
        assert!(adv[3] < 0.0);
    }

    #[test]
    fn no_variance_gives_zero_advantages() {
        assert_eq!(normalized_advantages(&[1.0, 1.0]), vec![0.0, 0.0]);
    }
}
