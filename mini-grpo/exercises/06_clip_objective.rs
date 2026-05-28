pub fn clipped_surrogate(_ratio: f32, _advantage: f32, _clip_eps: f32) -> f32 {
    // TODO:
    // PPO-style term:
    // min(ratio * advantage, clipped_ratio * advantage)
    // where clipped_ratio is ratio clipped to [1 - eps, 1 + eps].
    //
    // Be careful: min behaves differently for negative advantages.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clips_positive_advantage() {
        let unclipped = 2.0 * 1.0;
        let clipped = clipped_surrogate(2.0, 1.0, 0.2);

        assert!(clipped < unclipped);
        assert!((clipped - 1.2).abs() < 1e-6);
    }

    #[test]
    fn handles_negative_advantage() {
        let val = clipped_surrogate(0.5, -1.0, 0.2);
        assert!((val - -0.8).abs() < 1e-6);
    }
}
