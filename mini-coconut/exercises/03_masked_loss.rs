#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionKind {
    Question,
    LatentThought,
    TextTarget,
}

pub fn loss_mask(_kinds: &[PositionKind]) -> Vec<bool> {
    // TODO:
    // Loss applies only to TextTarget positions.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_questions_and_latents() {
        let kinds = vec![
            PositionKind::Question,
            PositionKind::LatentThought,
            PositionKind::TextTarget,
            PositionKind::TextTarget,
        ];

        assert_eq!(loss_mask(&kinds), vec![false, false, true, true]);
    }
}
