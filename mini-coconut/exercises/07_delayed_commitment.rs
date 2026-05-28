pub fn commitment_step(_branch_distributions: &[Vec<f32>], _threshold: f32) -> Option<usize> {
    // TODO:
    // Return the first step where max probability >= threshold.
    // Return None if no step commits.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_when_model_commits() {
        let steps = vec![
            vec![0.34, 0.33, 0.33],
            vec![0.50, 0.30, 0.20],
            vec![0.91, 0.05, 0.04],
        ];

        assert_eq!(commitment_step(&steps, 0.9), Some(2));
    }
}
