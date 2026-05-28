#[derive(Debug, Clone, Copy)]
pub struct DriftScore {
    pub sink_mass: f32,
    pub recent_draft_mass: f32,
    pub drift: f32,
}

pub fn attention_drift_score(
    attention: &[f32],
    sink_indices: &[usize],
    recent_draft_indices: &[usize],
) -> DriftScore {
    // TODO:
    // sink_mass = sum attention over sink_indices
    // recent_draft_mass = sum attention over recent_draft_indices
    // drift = recent_draft_mass - sink_mass
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_when_draft_tokens_dominate() {
        let attn = vec![0.10, 0.10, 0.15, 0.30, 0.35];
        let score = attention_drift_score(&attn, &[0], &[3, 4]);

        assert!((score.sink_mass - 0.10).abs() < 1e-6);
        assert!((score.recent_draft_mass - 0.65).abs() < 1e-6);
        assert!(score.drift > 0.0);
    }

    #[test]
    fn negative_when_sink_dominates() {
        let attn = vec![0.60, 0.10, 0.10, 0.10, 0.10];
        let score = attention_drift_score(&attn, &[0], &[3, 4]);

        assert!(score.drift < 0.0);
    }
}
