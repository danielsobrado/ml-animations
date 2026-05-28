pub fn shared_index_schedule(
    query_blocks: &[Vec<usize>],
    representative: usize,
) -> Vec<Vec<usize>> {
    // TODO:
    // Return one copied selected-block set for every query.
    // Use query_blocks[representative] as the shared layout.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shares_representative_layout() {
        let queries = vec![
            vec![0, 2, 5, 7],
            vec![0, 2, 4, 7],
            vec![1, 3, 4, 7],
        ];

        let shared = shared_index_schedule(&queries, 1);
        assert_eq!(shared, vec![
            vec![0, 2, 4, 7],
            vec![0, 2, 4, 7],
            vec![0, 2, 4, 7],
        ]);
    }

    #[test]
    fn preserves_query_count() {
        let queries = vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8]];

        assert_eq!(shared_index_schedule(&queries, 2).len(), 4);
    }
}
