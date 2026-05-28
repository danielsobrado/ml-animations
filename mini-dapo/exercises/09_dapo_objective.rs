#![allow(unused)]

#[derive(Debug, Clone, Copy)]
pub struct ClipRange {
    pub lower: f32,
    pub upper: f32,
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub reward: f32,
    pub length: usize,
    pub token_ratios: Vec<f32>,
}

fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }

    xs.iter().sum::<f32>() / xs.len() as f32
}

fn stddev(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }

    let m = mean(xs);
    let variance = xs.iter().map(|x| (x - m).powi(2)).sum::<f32>() / xs.len() as f32;
    variance.sqrt()
}

fn clipped_ratio(ratio: f32, range: ClipRange) -> f32 {
    ratio.clamp(range.lower, range.upper)
}

fn token_level_policy_loss(ratios: &[f32], advantage: f32, range: ClipRange) -> f32 {
    if ratios.is_empty() {
        return 0.0;
    }

    let sum = ratios
        .iter()
        .map(|ratio| {
            let unclipped = ratio * advantage;
            let clipped = clipped_ratio(*ratio, range) * advantage;
            unclipped.min(clipped)
        })
        .sum::<f32>();

    -sum / ratios.len() as f32
}

fn overlong_penalty(length: usize, max_len: usize, soft_margin: usize) -> f32 {
    let safe_len = max_len.saturating_sub(soft_margin);
    if length <= safe_len {
        0.0
    } else if length >= max_len {
        1.0
    } else {
        (length - safe_len) as f32 / soft_margin.max(1) as f32
    }
}

fn shaped_reward(base_reward: f32, length: usize, max_len: usize, soft_margin: usize) -> f32 {
    (base_reward - overlong_penalty(length, max_len, soft_margin)).max(0.0)
}

pub fn dapo_sample_loss(
    sample: &Sample,
    group_rewards: &[f32],
    range: ClipRange,
    max_len: usize,
    soft_margin: usize,
) -> f32 {
    // TODO:
    // 1. Shape this sample's reward using overlong reward shaping.
    // 2. Shape all group rewards with no length info by using them directly.
    //    For this toy exercise, compute mean/std over group_rewards.
    // 3. advantage = (shaped_reward - mean(group_rewards)) / std(group_rewards)
    // 4. return token_level_policy_loss(token_ratios, advantage, range)
    // If std is zero, advantage = 0.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_reward_with_good_ratios_gets_negative_loss() {
        let sample = Sample {
            reward: 1.0,
            length: 800,
            token_ratios: vec![1.0, 1.1, 1.2],
        };

        let group_rewards = vec![1.0, 0.0, 1.0, 0.0];
        let range = ClipRange { lower: 0.8, upper: 1.28 };

        let loss = dapo_sample_loss(&sample, &group_rewards, range, 1000, 100);
        assert!(loss < 0.0);
    }
}
