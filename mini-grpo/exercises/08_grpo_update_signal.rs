#[derive(Debug, PartialEq, Eq)]
pub enum UpdateDirection {
    Reinforce,
    Suppress,
    Neutral,
}

pub fn update_direction(_advantage: f32) -> UpdateDirection {
    // TODO:
    // Positive -> Reinforce
    // Negative -> Suppress
    // Zero -> Neutral
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_advantage_to_direction() {
        assert_eq!(update_direction(0.7), UpdateDirection::Reinforce);
        assert_eq!(update_direction(-0.2), UpdateDirection::Suppress);
        assert_eq!(update_direction(0.0), UpdateDirection::Neutral);
    }
}
