pub fn mean_vector(_vectors: &[Vec<f32>]) -> Vec<f32> {
    // TODO:
    // Average vectors elementwise.
    todo!()
}

pub fn compress_blocks(_keys: &[Vec<f32>], _block_size: usize) -> Vec<Vec<f32>> {
    // TODO:
    // Split keys into blocks and mean-pool each block.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compresses_by_mean() {
        let keys = vec![vec![1.0, 3.0], vec![3.0, 5.0], vec![10.0, 20.0]];

        assert_eq!(
            compress_blocks(&keys, 2),
            vec![vec![2.0, 4.0], vec![10.0, 20.0]]
        );
    }
}
