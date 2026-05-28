pub fn has_contrast(_rewards: &[f32]) -> bool {
    // TODO:
    // Return true if not all rewards are identical.
    todo!()
}

pub fn count_useful_groups(_groups: &[Vec<f32>]) -> usize {
    // TODO:
    // Count groups with contrast.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_contrast() {
        assert!(!has_contrast(&[1.0, 1.0, 1.0]));
        assert!(!has_contrast(&[0.0, 0.0, 0.0]));
        assert!(has_contrast(&[1.0, 0.0, 1.0]));
    }

    #[test]
    fn counts_useful_groups() {
        let groups = vec![
            vec![1.0, 1.0],
            vec![1.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 1.0],
        ];

        assert_eq!(count_useful_groups(&groups), 2);
    }
}
