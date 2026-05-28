pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    // TODO
    todo!()
}

pub fn block_scores(_query: &[f32], _compressed_keys: &[Vec<f32>]) -> Vec<f32> {
    // TODO:
    // Score each compressed key with q dot c_k.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_blocks() {
        let q = vec![1.0, 2.0];
        let c = vec![vec![1.0, 0.0], vec![0.0, 3.0]];

        assert_eq!(block_scores(&q, &c), vec![1.0, 6.0]);
    }
}
