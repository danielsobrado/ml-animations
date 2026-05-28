pub fn selected_kv_count(prefix_len: usize, sparse_ratio: f32) -> usize {
    // TODO:
    // Return ceil(prefix_len * sparse_ratio), at least 1 when prefix_len > 0.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_selected_count() {
        assert_eq!(selected_kv_count(1000, 0.07), 70);
        assert_eq!(selected_kv_count(10, 0.01), 1);
    }

    #[test]
    fn keeps_empty_prefix_empty() {
        assert_eq!(selected_kv_count(0, 0.10), 0);
    }

    #[test]
    fn rounds_up_fractional_counts() {
        assert_eq!(selected_kv_count(33, 0.10), 4);
    }
}
