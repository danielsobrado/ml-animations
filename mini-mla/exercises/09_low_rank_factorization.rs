#![allow(unused_variables)]

pub fn count_distinct_rows(rows: &[Vec<f32>]) -> usize {
    // TODO:
    // Count unique rows exactly. This is a toy proxy for rank.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_gqa_rows_have_low_diversity() {
        let rows = vec![
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
        ];

        assert_eq!(count_distinct_rows(&rows), 2);
    }
}
