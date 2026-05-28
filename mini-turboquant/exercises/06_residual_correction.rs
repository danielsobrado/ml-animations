pub fn residual(original: &[f32], approx: &[f32]) -> Vec<f32> {
    assert_eq!(original.len(), approx.len());

    // TODO:
    // original - approx elementwise
    todo!()
}

pub fn sign_sketch(x: &[f32]) -> Vec<i8> {
    // TODO:
    // Return +1 for x >= 0, -1 otherwise.
    todo!()
}

pub fn corrected_dot_toy(query: &[f32], approx_key: &[f32], residual_signs: &[i8], scale: f32) -> f32 {
    assert_eq!(query.len(), approx_key.len());
    assert_eq!(query.len(), residual_signs.len());

    // TODO:
    // dot(query, approx_key) + scale * sum_i query[i] * sign_i
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_captures_leftover_direction() {
        let k = vec![1.0, -2.0];
        let kh = vec![0.75, -1.5];

        assert_eq!(residual(&k, &kh), vec![0.25, -0.5]);
        assert_eq!(sign_sketch(&residual(&k, &kh)), vec![1, -1]);
    }
}
