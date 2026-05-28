#![allow(unused_variables)]

pub fn up_project(w_up: &[Vec<f32>], latent: &[f32]) -> Vec<f32> {
    // TODO:
    // expanded = W_up * latent
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstructs_head_view() {
        let w_up = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let c = vec![2.0, 3.0];

        assert_eq!(up_project(&w_up, &c), vec![2.0, 3.0, 5.0]);
    }
}
