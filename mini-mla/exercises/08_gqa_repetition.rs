#![allow(unused_variables)]

pub fn repeat_kv_heads(kv: &[Vec<f32>], repeat: usize) -> Vec<Vec<f32>> {
    // TODO:
    // Repeat each KV head `repeat` times.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeats_group_heads() {
        let kv = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];

        let repeated = repeat_kv_heads(&kv, 3);

        assert_eq!(
            repeated,
            vec![
                vec![1.0, 2.0],
                vec![1.0, 2.0],
                vec![1.0, 2.0],
                vec![3.0, 4.0],
                vec![3.0, 4.0],
                vec![3.0, 4.0],
            ]
        );
    }
}
