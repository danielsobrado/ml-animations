use std::collections::HashSet;

pub fn overlap_ratio(a: &[usize], b: &[usize]) -> f32 {
    // TODO:
    // Return intersection_size / union_size.
    // Return 1.0 when both sets are empty.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_overlap() {
        let a = vec![0, 2, 5, 7];
        let b = vec![0, 2, 4, 7];

        let ratio = overlap_ratio(&a, &b);
        assert!((ratio - 0.6).abs() < 1e-6);
    }

    #[test]
    fn handles_no_overlap() {
        let ratio = overlap_ratio(&[1, 2], &[3, 4]);
        assert!((ratio - 0.0).abs() < 1e-6);
    }

    #[test]
    fn treats_empty_sets_as_identical() {
        let ratio = overlap_ratio(&[], &[]);
        assert!((ratio - 1.0).abs() < 1e-6);
    }
}
