#[derive(Debug, Clone)]
pub struct SparsePlan {
    pub name: &'static str,
    pub estimated_quality: f32,
    pub tokens_loaded: usize,
}

pub fn efficiency_score(_plan: &SparsePlan) -> f32 {
    // TODO:
    // Quality per loaded token.
    todo!()
}

pub fn best_plan(_plans: &[SparsePlan]) -> Option<&SparsePlan> {
    // TODO:
    // Pick highest efficiency_score.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_balanced_plan() {
        let plans = vec![
            SparsePlan { name: "too small", estimated_quality: 0.40, tokens_loaded: 512 },
            SparsePlan { name: "balanced", estimated_quality: 0.90, tokens_loaded: 1024 },
            SparsePlan { name: "too large", estimated_quality: 0.95, tokens_loaded: 4096 },
        ];

        assert_eq!(best_plan(&plans).unwrap().name, "balanced");
    }
}
