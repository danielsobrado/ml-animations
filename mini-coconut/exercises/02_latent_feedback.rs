#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Language,
    Latent,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InputStep {
    TokenEmbedding(Vec<f32>),
    LatentThought(Vec<f32>),
}

pub fn next_input_embedding(
    _mode: Mode,
    _sampled_token_embedding: Vec<f32>,
    _last_hidden_state: Vec<f32>,
) -> Vec<f32> {
    // TODO:
    // In language mode, use sampled_token_embedding.
    // In latent mode, use last_hidden_state.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_mode_uses_hidden_state() {
        let token = vec![1.0, 2.0];
        let hidden = vec![9.0, 8.0];

        assert_eq!(
            next_input_embedding(Mode::Latent, token, hidden.clone()),
            hidden
        );
    }

    #[test]
    fn language_mode_uses_token_embedding() {
        let token = vec![1.0, 2.0];
        let hidden = vec![9.0, 8.0];

        assert_eq!(
            next_input_embedding(Mode::Language, token.clone(), hidden),
            token
        );
    }
}
