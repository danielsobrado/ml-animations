#![allow(unused_variables)]

#[derive(Debug, Clone)]
pub struct AttentionStrategy {
    pub name: &'static str,
    pub cache_elements: usize,
    pub projection_flops: usize,
    pub head_diversity_score: f32,
}

pub fn memory_efficiency_score(strategy: &AttentionStrategy) -> f32 {
    // TODO:
    // Higher is better. Use diversity divided by cache elements.
    todo!()
}

pub fn best_memory_efficiency(strategies: &[AttentionStrategy]) -> Option<&AttentionStrategy> {
    // TODO:
    // Pick the strategy with highest memory_efficiency_score.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_compact_expressive_strategy() {
        let strategies = vec![
            AttentionStrategy {
                name: "MHA",
                cache_elements: 4096,
                projection_flops: 100,
                head_diversity_score: 1.0,
            },
            AttentionStrategy {
                name: "GQA",
                cache_elements: 1024,
                projection_flops: 100,
                head_diversity_score: 0.45,
            },
            AttentionStrategy {
                name: "MLA",
                cache_elements: 576,
                projection_flops: 160,
                head_diversity_score: 0.85,
            },
        ];

        assert_eq!(best_memory_efficiency(&strategies).unwrap().name, "MLA");
    }
}
