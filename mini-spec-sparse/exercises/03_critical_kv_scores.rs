pub fn average_logits(first: &[f32], bonus: &[f32]) -> Vec<f32> {
    assert_eq!(first.len(), bonus.len());

    // TODO:
    // Average the two logit vectors elementwise.
    todo!()
}

pub fn top_k_indices(scores: &[f32], k: usize) -> Vec<usize> {
    // TODO:
    // Return indices of the top-k scores, highest first.
    // Break ties by lower index first.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn averages_logits() {
        let first = vec![1.0, 5.0, 2.0, 0.0];
        let bonus = vec![1.0, 1.0, 6.0, 0.0];

        assert_eq!(average_logits(&first, &bonus), vec![1.0, 3.0, 4.0, 0.0]);
    }

    #[test]
    fn selects_critical_entries() {
        let first = vec![1.0, 5.0, 2.0, 0.0];
        let bonus = vec![1.0, 1.0, 6.0, 0.0];

        let scores = average_logits(&first, &bonus);
        assert_eq!(top_k_indices(&scores, 2), vec![2, 1]);
    }

    #[test]
    fn top_k_clamps_to_available_scores() {
        assert_eq!(top_k_indices(&[2.0, 5.0], 5), vec![1, 0]);
    }
}
