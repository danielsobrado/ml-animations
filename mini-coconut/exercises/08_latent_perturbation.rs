pub fn add_perturbation(_h: &[f32], _noise: &[f32], _scale: f32) -> Vec<f32> {
    assert_eq!(_h.len(), _noise.len());

    // TODO:
    // h_i + scale * noise_i
    todo!()
}

pub fn l2_distance(_a: &[f32], _b: &[f32]) -> f32 {
    assert_eq!(_a.len(), _b.len());

    // TODO:
    // sqrt(sum_i (a_i - b_i)^2)
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perturbation_changes_hidden_state() {
        let h = vec![1.0, 2.0];
        let noise = vec![0.5, -0.5];

        let hp = add_perturbation(&h, &noise, 2.0);
        assert_eq!(hp, vec![2.0, 1.0]);
        assert!(l2_distance(&h, &hp) > 0.0);
    }
}
