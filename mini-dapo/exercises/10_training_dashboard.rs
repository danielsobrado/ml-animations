#![allow(unused)]

#[derive(Debug, Clone)]
pub struct BatchStats {
    pub total_groups: usize,
    pub effective_groups: usize,
    pub mean_reward: f32,
    pub entropy: f32,
    pub overlong_rate: f32,
}

pub fn training_health(stats: &BatchStats) -> &'static str {
    // TODO:
    // Return:
    // "low-signal" if effective_groups < total_groups / 2
    // "entropy-collapse" if entropy < 0.5
    // "too-overlong" if overlong_rate > 0.3
    // otherwise "healthy"
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_low_signal_batches() {
        let stats = BatchStats {
            total_groups: 100,
            effective_groups: 20,
            mean_reward: 0.5,
            entropy: 1.0,
            overlong_rate: 0.1,
        };

        assert_eq!(training_health(&stats), "low-signal");
    }

    #[test]
    fn flags_healthy_batch() {
        let stats = BatchStats {
            total_groups: 100,
            effective_groups: 80,
            mean_reward: 0.5,
            entropy: 1.2,
            overlong_rate: 0.1,
        };

        assert_eq!(training_health(&stats), "healthy");
    }
}
