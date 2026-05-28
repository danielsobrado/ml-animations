pub fn pause_embedding(_base_pause: &[f32]) -> Vec<f32> {
    // TODO:
    // A pause token uses the same learned embedding every time in this toy model.
    todo!()
}

pub fn continuous_thought(_hidden_state: &[f32]) -> Vec<f32> {
    // TODO:
    // A continuous thought depends on the current hidden state.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pause_is_fixed_but_thought_changes() {
        let pause = vec![0.1, 0.2];
        let h1 = vec![1.0, 2.0];
        let h2 = vec![3.0, 4.0];

        assert_eq!(pause_embedding(&pause), pause_embedding(&pause));
        assert_ne!(continuous_thought(&h1), continuous_thought(&h2));
    }
}
