#![allow(unused)]

pub fn entropy(probs: &[f32]) -> f32 {
    // TODO:
    // -sum p * ln(p), skipping p == 0.
    todo!()
}

pub fn entropy_collapsed(probs: &[f32], threshold: f32) -> bool {
    // TODO:
    // Return true if entropy is below threshold.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peaked_distribution_has_low_entropy() {
        let diverse = vec![0.25, 0.25, 0.25, 0.25];
        let collapsed = vec![0.97, 0.01, 0.01, 0.01];

        assert!(entropy(&diverse) > entropy(&collapsed));
    }

    #[test]
    fn detects_entropy_collapse() {
        let collapsed = vec![0.97, 0.01, 0.01, 0.01];
        assert!(entropy_collapsed(&collapsed, 0.25));
    }
}
