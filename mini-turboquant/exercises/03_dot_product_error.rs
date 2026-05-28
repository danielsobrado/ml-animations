pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    // TODO:
    // Compute sum_i a[i] * b[i].
    todo!()
}

pub fn dot_error(query: &[f32], key: &[f32], key_hat: &[f32]) -> f32 {
    // TODO:
    // Return absolute difference between q*k and q*k_hat.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_dot_product() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn computes_attention_score_error() {
        let q = vec![1.0, 0.0];
        let k = vec![2.0, 3.0];
        let kh = vec![1.5, 3.0];

        assert!((dot_error(&q, &k, &kh) - 0.5).abs() < 1e-6);
    }
}
