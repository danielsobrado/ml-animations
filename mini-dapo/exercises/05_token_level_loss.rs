#![allow(unused)]

#[derive(Debug, Clone, Copy)]
pub struct ClipRange {
    pub lower: f32,
    pub upper: f32,
}

pub fn clipped_ratio(ratio: f32, range: ClipRange) -> f32 {
    ratio.clamp(range.lower, range.upper)
}

pub fn token_level_policy_loss(
    ratios: &[f32],
    advantage: f32,
    range: ClipRange,
) -> f32 {
    // TODO:
    // For each token:
    //   unclipped = ratio * advantage
    //   clipped = clipped_ratio(ratio, range) * advantage
    //   contribution = min(unclipped, clipped)
    // Return negative mean contribution because optimizers minimize loss.
    //
    // Note: f32::min works for this toy objective.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn averages_over_tokens() {
        let ratios = vec![1.0, 1.1, 1.5];
        let range = ClipRange { lower: 0.8, upper: 1.2 };

        let loss = token_level_policy_loss(&ratios, 1.0, range);

        // contributions: 1.0, 1.1, min(1.5, 1.2) = 1.2
        assert!((loss + 1.1).abs() < 1e-6);
    }

    #[test]
    fn empty_tokens_have_zero_loss() {
        let range = ClipRange { lower: 0.8, upper: 1.2 };
        assert_eq!(token_level_policy_loss(&[], 1.0, range), 0.0);
    }
}
