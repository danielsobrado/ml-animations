pub fn has_answer_tag(_text: &str) -> bool {
    // TODO:
    // Return true if text contains both "<answer>" and "</answer>".
    todo!()
}

pub fn format_reward(_text: &str) -> f32 {
    // TODO:
    // Return 0.1 if answer tags exist, else 0.0.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewards_answer_format() {
        assert_eq!(format_reward("<answer>42</answer>"), 0.1);
    }

    #[test]
    fn no_reward_without_tags() {
        assert_eq!(format_reward("42"), 0.0);
    }
}
