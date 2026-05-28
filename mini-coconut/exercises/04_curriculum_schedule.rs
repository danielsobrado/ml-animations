pub fn replaced_steps_at_stage(_stage: usize, _total_steps: usize) -> usize {
    // TODO:
    // At stage 0, replace 0 steps.
    // At stage s, replace min(s, total_steps) reasoning steps.
    todo!()
}

pub fn curriculum_example(_total_steps: usize, _stage: usize) -> Vec<&'static str> {
    // TODO:
    // Return a vector of "latent" for replaced steps and "text" for remaining steps.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_replaces_prefix_steps() {
        assert_eq!(replaced_steps_at_stage(2, 5), 2);
        assert_eq!(replaced_steps_at_stage(10, 5), 5);
    }

    #[test]
    fn builds_curriculum_labels() {
        assert_eq!(
            curriculum_example(4, 2),
            vec!["latent", "latent", "text", "text"]
        );
    }
}
