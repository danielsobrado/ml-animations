pub fn exact_answer_reward(_predicted: &str, _target: &str) -> f32 {
    // TODO:
    // Trim whitespace and return 1.0 for exact match, else 0.0.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewards_correct_answer() {
        assert_eq!(exact_answer_reward(" 42 ", "42"), 1.0);
    }

    #[test]
    fn rejects_wrong_answer() {
        assert_eq!(exact_answer_reward("41", "42"), 0.0);
    }
}
