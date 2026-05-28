pub fn shared_group_scores(_head_scores: &[Vec<f32>]) -> Vec<f32> {
    // TODO:
    // Average block scores across query heads in one GQA group.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn averages_scores_across_heads() {
        let scores = vec![vec![1.0, 5.0, 2.0], vec![3.0, 1.0, 4.0]];

        assert_eq!(shared_group_scores(&scores), vec![2.0, 3.0, 3.0]);
    }
}
