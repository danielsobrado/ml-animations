#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub prompt: String,
    pub response: String,
    pub reward: f32,
}

pub fn keep_high_reward(_samples: &[Sample], _threshold: f32) -> Vec<Sample> {
    // TODO:
    // Return samples with reward >= threshold.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_teacher_outputs() {
        let samples = vec![
            Sample { prompt: "2+2?".into(), response: "4".into(), reward: 1.0 },
            Sample { prompt: "2+2?".into(), response: "5".into(), reward: 0.0 },
        ];

        assert_eq!(keep_high_reward(&samples, 0.5).len(), 1);
    }
}
