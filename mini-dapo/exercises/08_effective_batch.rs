#![allow(unused)]

#[derive(Debug, Clone, PartialEq)]
pub struct PromptGroup {
    pub prompt_id: usize,
    pub rewards: Vec<f32>,
}

fn group_accuracy(rewards: &[f32]) -> f32 {
    if rewards.is_empty() {
        return 0.0;
    }

    let correct = rewards.iter().filter(|reward| **reward > 0.0).count();
    correct as f32 / rewards.len() as f32
}

fn has_contrast(rewards: &[f32]) -> bool {
    let accuracy = group_accuracy(rewards);
    accuracy > 0.0 && accuracy < 1.0
}

pub fn effective_group_count(groups: &[PromptGroup]) -> usize {
    // TODO:
    // Count groups with contrast.
    todo!()
}

pub fn effective_sample_count(groups: &[PromptGroup]) -> usize {
    // TODO:
    // Count samples inside groups with contrast.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_only_useful_groups_and_samples() {
        let groups = vec![
            PromptGroup { prompt_id: 1, rewards: vec![1.0, 1.0, 1.0] },
            PromptGroup { prompt_id: 2, rewards: vec![1.0, 0.0, 1.0] },
            PromptGroup { prompt_id: 3, rewards: vec![0.0, 0.0, 0.0] },
            PromptGroup { prompt_id: 4, rewards: vec![0.0, 1.0, 0.0] },
        ];

        assert_eq!(effective_group_count(&groups), 2);
        assert_eq!(effective_sample_count(&groups), 6);
    }
}
