pub fn accepted_prefix_len(draft: &[usize], target: &[usize]) -> usize {
    draft
        .iter()
        .zip(target.iter())
        .take_while(|(draft_token, target_token)| draft_token == target_token)
        .count()
}

pub fn speculative_round(prefix: &[usize], draft: &[usize], target: &[usize]) -> Vec<usize> {
    // TODO:
    // 1. Start with prefix.
    // 2. Append accepted draft tokens.
    // 3. If draft is not fully accepted, append the target token at the rejection position.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_accepted_prefix_and_replacement() {
        let prefix = vec![10, 11];
        let draft = vec![1, 2, 9, 4];
        let target = vec![1, 2, 3, 4];

        assert_eq!(
            speculative_round(&prefix, &draft, &target),
            vec![10, 11, 1, 2, 3]
        );
    }

    #[test]
    fn appends_all_when_fully_accepted() {
        let prefix = vec![10];
        let draft = vec![1, 2, 3];
        let target = vec![1, 2, 3];

        assert_eq!(
            speculative_round(&prefix, &draft, &target),
            vec![10, 1, 2, 3]
        );
    }
}
