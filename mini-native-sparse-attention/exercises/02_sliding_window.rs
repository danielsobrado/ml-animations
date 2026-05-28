pub fn sliding_window_indices(_query_pos: usize, _window: usize) -> Vec<usize> {
    // TODO:
    // Return previous token indices visible to query_pos,
    // including query_pos itself.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_recent_context() {
        assert_eq!(sliding_window_indices(10, 4), vec![7, 8, 9, 10]);
    }

    #[test]
    fn clips_at_start() {
        assert_eq!(sliding_window_indices(2, 5), vec![0, 1, 2]);
    }
}
