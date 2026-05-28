pub struct ReasoningBudget {
    pub text_tokens: usize,
    pub latent_steps: usize,
}

pub fn total_visible_tokens(_budget: ReasoningBudget) -> usize {
    // TODO:
    // Only text tokens are visible/generated language tokens.
    todo!()
}

pub fn total_compute_steps(_budget: ReasoningBudget) -> usize {
    // TODO:
    // Both text tokens and latent steps require forward computation.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_visible_tokens_from_compute_steps() {
        let b = ReasoningBudget {
            text_tokens: 5,
            latent_steps: 3,
        };

        assert_eq!(total_visible_tokens(b), 5);

        let b = ReasoningBudget {
            text_tokens: 5,
            latent_steps: 3,
        };

        assert_eq!(total_compute_steps(b), 8);
    }
}
