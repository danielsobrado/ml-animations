use std::collections::BTreeSet;

pub fn exact_merged_schedule(query_blocks: &[Vec<usize>]) -> Vec<usize> {
    // TODO:
    // Return sorted union of all selected blocks.
    todo!()
}

pub fn query_owns_block(query_blocks: &[usize], block: usize) -> bool {
    // TODO:
    // Return whether this query originally selected this block.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_and_deduplicates() {
        let queries = vec![
            vec![0, 2, 5, 7],
            vec![0, 2, 4, 7],
            vec![0, 3, 4, 7],
        ];

        assert_eq!(exact_merged_schedule(&queries), vec![0, 2, 3, 4, 5, 7]);
    }

    #[test]
    fn keeps_mask_semantics() {
        let q1 = vec![0, 2, 5, 7];

        assert!(query_owns_block(&q1, 5));
        assert!(!query_owns_block(&q1, 4));
    }
}
