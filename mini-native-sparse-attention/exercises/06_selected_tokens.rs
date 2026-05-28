pub fn selected_token_indices(
    _selected_blocks: &[usize],
    _block_size: usize,
    _seq_len: usize,
) -> Vec<usize> {
    // TODO:
    // Expand block ids into token indices, clipping at seq_len.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_blocks() {
        assert_eq!(
            selected_token_indices(&[1, 3], 4, 15),
            vec![4, 5, 6, 7, 12, 13, 14]
        );
    }
}
