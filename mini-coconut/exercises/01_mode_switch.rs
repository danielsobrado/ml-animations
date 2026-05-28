#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Language,
    Latent,
}

pub fn next_mode(_current: Mode, _token: &str) -> Mode {
    // TODO:
    // "<bot>" switches into Latent mode.
    // "<eot>" switches back to Language mode.
    // Otherwise keep current mode.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switches_into_latent_mode() {
        assert_eq!(next_mode(Mode::Language, "<bot>"), Mode::Latent);
    }

    #[test]
    fn switches_back_to_language_mode() {
        assert_eq!(next_mode(Mode::Latent, "<eot>"), Mode::Language);
    }

    #[test]
    fn ordinary_token_keeps_mode() {
        assert_eq!(next_mode(Mode::Latent, "thinking"), Mode::Latent);
    }
}
