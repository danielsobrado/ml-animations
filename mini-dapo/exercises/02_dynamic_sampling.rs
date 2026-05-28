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

pub fn dynamic_sampling(groups: &[PromptGroup], target_count: usize) -> Vec<PromptGroup> {
    // TODO:
    // Keep only groups with contrast.
    // Return at most target_count groups, preserving order.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_uninformative_groups() {
        let groups = vec![
            PromptGroup { prompt_id: 1, rewards: vec![1.0, 1.0] },
            PromptGroup { prompt_id: 2, rewards: vec![1.0, 0.0] },
            PromptGroup { prompt_id: 3, rewards: vec![0.0, 0.0] },
            PromptGroup { prompt_id: 4, rewards: vec![0.0, 1.0] },
        ];

        let kept = dynamic_sampling(&groups, 10);
        let ids: Vec<usize> = kept.iter().map(|g| g.prompt_id).collect();

        assert_eq!(ids, vec![2, 4]);
    }

    #[test]
    fn honors_target_count() {
        let groups = vec![
            PromptGroup { prompt_id: 1, rewards: vec![1.0, 0.0] },
            PromptGroup { prompt_id: 2, rewards: vec![0.0, 1.0] },
            PromptGroup { prompt_id: 3, rewards: vec![1.0, 0.0] },
        ];

        let ids: Vec<usize> = dynamic_sampling(&groups, 2)
            .iter()
            .map(|g| g.prompt_id)
            .collect();

        assert_eq!(ids, vec![1, 2]);
    }
}
