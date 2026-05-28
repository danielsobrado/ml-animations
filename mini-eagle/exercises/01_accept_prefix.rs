pub fn accepted_prefix_len(draft: &[usize], target: &[usize]) -> usize {
    // TODO:
    // Count matching tokens from the start.
    // Stop at the first mismatch.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_full_match() {
        assert_eq!(accepted_prefix_len(&[1, 2, 3], &[1, 2, 3]), 3);
    }

    #[test]
    fn stops_at_first_rejection() {
        assert_eq!(accepted_prefix_len(&[1, 2, 9, 4], &[1, 2, 3, 4]), 2);
    }

    #[test]
    fn accepts_zero_when_first_token_differs() {
        assert_eq!(accepted_prefix_len(&[8, 2, 3], &[1, 2, 3]), 0);
    }
}
