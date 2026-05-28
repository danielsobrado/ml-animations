pub fn collect_two_query_indices(num_draft_tokens: usize) -> Option<(usize, usize)> {
    // Convention:
    // index 0 = first draft token
    // index num_draft_tokens - 1 = bonus token
    //
    // TODO:
    // Return None if there are no draft tokens.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_first_and_bonus() {
        assert_eq!(collect_two_query_indices(8), Some((0, 7)));
    }

    #[test]
    fn handles_single_token() {
        assert_eq!(collect_two_query_indices(1), Some((0, 0)));
    }

    #[test]
    fn handles_empty() {
        assert_eq!(collect_two_query_indices(0), None);
    }
}
