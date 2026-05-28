pub fn accepted_prefix_len(draft: &[usize], target: &[usize]) -> usize {
    // TODO:
    // Return the number of matching tokens before the first mismatch.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stops_at_first_mismatch() {
        assert_eq!(accepted_prefix_len(&[1, 2, 9, 4], &[1, 2, 3, 4]), 2);
    }

    #[test]
    fn accepts_full_match() {
        assert_eq!(accepted_prefix_len(&[5, 6, 7], &[5, 6, 7]), 3);
    }

    #[test]
    fn handles_short_target() {
        assert_eq!(accepted_prefix_len(&[5, 6, 7], &[5, 6]), 2);
    }
}
