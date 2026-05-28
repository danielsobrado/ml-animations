pub fn dot(_a: &[f32], _b: &[f32]) -> f32 {
    assert_eq!(_a.len(), _b.len());

    // TODO:
    // sum_i a_i * b_i
    todo!()
}

pub fn nearest_token<'a>(
    _hidden: &[f32],
    _vocab: &'a [(&'a str, Vec<f32>)],
) -> Option<&'a str> {
    // TODO:
    // Return token with largest dot product with hidden.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probes_latent_state() {
        let hidden = vec![1.0, 0.0];
        let vocab = vec![
            ("left", vec![0.9, 0.1]),
            ("right", vec![0.1, 0.9]),
        ];

        assert_eq!(nearest_token(&hidden, &vocab), Some("left"));
    }
}
