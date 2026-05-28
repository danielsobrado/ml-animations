pub fn expected_acceptance_length(probs: &[f32]) -> f32 {
    // TODO:
    // If probs[k] is probability token k is accepted conditional on previous accepts,
    // expected accepted length = p1 + p1*p2 + p1*p2*p3 + ...
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_probs_give_longer_acceptance() {
        let drifting = vec![0.95, 0.80, 0.55, 0.30];
        let stable = vec![0.90, 0.88, 0.86, 0.84];

        assert!(expected_acceptance_length(&stable) > expected_acceptance_length(&drifting));
    }

    #[test]
    fn zero_first_prob_means_zero_length() {
        assert_eq!(expected_acceptance_length(&[0.0, 1.0, 1.0]), 0.0);
    }
}
