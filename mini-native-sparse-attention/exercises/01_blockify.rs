pub fn block_ranges(_seq_len: usize, _block_size: usize) -> Vec<(usize, usize)> {
    // TODO:
    // Return half-open ranges [start, end) covering the sequence.
    // The final block may be shorter.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_sequence_with_blocks() {
        assert_eq!(block_ranges(10, 4), vec![(0, 4), (4, 8), (8, 10)]);
    }
}
