pub fn mse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    // TODO:
    // mean_i (a[i] - b[i])^2
    todo!()
}

pub fn signed_dot_error(query: &[f32], key: &[f32], key_hat: &[f32]) -> f32 {
    // TODO:
    // Return q*key_hat - q*key.
    // This signed error reveals bias direction.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_mse_can_have_signed_dot_error() {
        let q = vec![10.0, 0.0];
        let k = vec![1.0, 1.0];
        let kh = vec![0.9, 1.1];

        assert!(mse(&k, &kh) < 0.02);
        assert!((signed_dot_error(&q, &k, &kh) + 1.0).abs() < 1e-5);
    }
}
