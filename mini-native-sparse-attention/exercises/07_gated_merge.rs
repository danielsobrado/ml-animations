pub fn gated_merge(outputs: &[Vec<f32>], gates: &[f32]) -> Vec<f32> {
    assert_eq!(outputs.len(), gates.len());

    // TODO:
    // Weighted sum of branch outputs.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combines_three_branches() {
        let outputs = vec![
            vec![1.0, 0.0],
            vec![0.0, 2.0],
            vec![4.0, 4.0],
        ];
        let gates = vec![0.5, 0.25, 0.25];

        assert_eq!(gated_merge(&outputs, &gates), vec![1.5, 1.5]);
    }
}
