#![allow(unused)]

pub fn group_accuracy(rewards: &[f32]) -> f32 {
    // TODO:
    // Treat reward > 0.0 as correct.
    // Return fraction correct.
    // Empty group returns 0.0.
    todo!()
}

pub fn has_contrast(rewards: &[f32]) -> bool {
    // TODO:
    // Return true when group accuracy is strictly between 0 and 1.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_mixed_group() {
        let rewards = vec![1.0, 0.0, 1.0, 0.0];

        assert!((group_accuracy(&rewards) - 0.5).abs() < 1e-6);
        assert!(has_contrast(&rewards));
    }

    #[test]
    fn all_correct_has_no_contrast() {
        assert!(!has_contrast(&[1.0, 1.0, 1.0]));
    }

    #[test]
    fn all_wrong_has_no_contrast() {
        assert!(!has_contrast(&[0.0, 0.0, 0.0]));
    }
}
