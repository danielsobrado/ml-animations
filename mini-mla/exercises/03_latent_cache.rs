#![allow(unused_variables)]

pub fn matvec(matrix: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    // TODO:
    // matrix shape: rows x cols
    // output_i = dot(matrix[i], x)
    todo!()
}

pub fn down_project(w_down: &[Vec<f32>], hidden: &[f32]) -> Vec<f32> {
    // TODO:
    // compressed latent cKV = W_down * hidden
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projects_to_latent_dim() {
        let w = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 1.0],
        ];
        let h = vec![2.0, 3.0, 4.0];

        assert_eq!(down_project(&w, &h), vec![2.0, 7.0]);
    }
}
