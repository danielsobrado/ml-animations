pub fn top_k_indices(_scores: &[f32], _k: usize) -> Vec<usize> {
    // TODO:
    // Return indices of the top-k scores, highest first.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_top_blocks() {
        let scores = vec![0.1, 4.0, 2.0, 9.0];
        assert_eq!(top_k_indices(&scores, 2), vec![3, 1]);
    }
}
