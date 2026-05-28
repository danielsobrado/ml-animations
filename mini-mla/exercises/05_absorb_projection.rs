#![allow(unused_variables)]

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    // TODO:
    // Return the dot product.
    todo!()
}

pub fn transpose(matrix: &[Vec<f32>]) -> Vec<Vec<f32>> {
    // TODO:
    // Return matrix^T.
    todo!()
}

pub fn matvec(matrix: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    // TODO:
    // Return matrix * x.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absorption_equivalence() {
        // W_up maps latent dim 2 -> full dim 3.
        let w_up = vec![
            vec![1.0, 0.0],
            vec![0.0, 2.0],
            vec![1.0, 1.0],
        ];

        let c = vec![3.0, 4.0];
        let q = vec![2.0, 1.0, -1.0];

        let expanded_k = matvec(&w_up, &c);
        let left = dot(&q, &expanded_k);

        let absorbed_q = matvec(&transpose(&w_up), &q);
        let right = dot(&absorbed_q, &c);

        assert!((left - right).abs() < 1e-6);
    }
}
