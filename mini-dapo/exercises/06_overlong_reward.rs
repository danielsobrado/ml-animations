#![allow(unused)]

pub fn overlong_penalty(length: usize, max_len: usize, soft_margin: usize) -> f32 {
    // TODO:
    // If length <= max_len - soft_margin: penalty = 0.0
    // If length >= max_len: penalty = 1.0
    // Otherwise linearly increase from 0 to 1 across the soft margin.
    todo!()
}

pub fn shaped_reward(base_reward: f32, length: usize, max_len: usize, soft_margin: usize) -> f32 {
    // TODO:
    // reward = base_reward - overlong_penalty(...)
    // Clamp to minimum 0.0.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_penalty_far_from_limit() {
        assert_eq!(overlong_penalty(800, 1000, 100), 0.0);
    }

    #[test]
    fn full_penalty_at_limit() {
        assert_eq!(overlong_penalty(1000, 1000, 100), 1.0);
    }

    #[test]
    fn partial_penalty_near_limit() {
        let p = overlong_penalty(950, 1000, 100);
        assert!((p - 0.5).abs() < 1e-6);
    }

    #[test]
    fn reward_does_not_go_negative() {
        assert_eq!(shaped_reward(0.25, 1000, 1000, 100), 0.0);
    }
}
