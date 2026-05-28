#[derive(Debug, Clone)]
pub struct Strategy {
    pub name: &'static str,
    pub expected_accepted: f32,
    pub latency_ms: f32,
}

pub fn accepted_token_throughput(s: &Strategy) -> f32 {
    // TODO:
    // Return accepted tokens per second.
    todo!()
}

pub fn best_strategy(strategies: &[Strategy]) -> Option<&Strategy> {
    // TODO:
    // Pick max accepted-token throughput.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_tokens_per_second() {
        let s = Strategy { name: "medium tree", expected_accepted: 6.0, latency_ms: 40.0 };

        assert!((accepted_token_throughput(&s) - 150.0).abs() < 1e-6);
    }

    #[test]
    fn picks_highest_throughput_not_highest_acceptance() {
        let strategies = vec![
            Strategy { name: "deep tree", expected_accepted: 8.0, latency_ms: 80.0 },
            Strategy { name: "medium tree", expected_accepted: 6.0, latency_ms: 40.0 },
        ];

        assert_eq!(best_strategy(&strategies).unwrap().name, "medium tree");
    }
}
