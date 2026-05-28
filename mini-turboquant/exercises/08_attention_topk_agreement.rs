pub fn top_k(scores: &[f32], k: usize) -> Vec<usize> {
    // TODO:
    // Return indices of top-k scores, highest first.
    todo!()
}

pub fn agreement_at_k(a: &[usize], b: &[usize]) -> f32 {
    // TODO:
    // Return intersection_size / k.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_attention_ranking_agreement() {
        let full = vec![0, 2, 4];
        let quant = vec![2, 0, 5];

        assert!((agreement_at_k(&full, &quant) - 2.0 / 3.0).abs() < 1e-6);
    }
}
