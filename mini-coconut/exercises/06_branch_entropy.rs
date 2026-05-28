pub fn entropy(_probs: &[f32]) -> f32 {
    // TODO:
    // Return -sum p * ln(p), skipping p == 0.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_has_higher_entropy_than_peaked() {
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let peaked = vec![0.97, 0.01, 0.01, 0.01];

        assert!(entropy(&uniform) > entropy(&peaked));
    }
}
