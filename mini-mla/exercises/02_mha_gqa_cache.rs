#![allow(unused_variables)]

pub fn compression_ratio(full: usize, compressed: usize) -> f32 {
    // TODO:
    // Return full / compressed.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_ratio() {
        let full = 2 * 32 * 128;
        let compressed = 512 + 64;

        let ratio = compression_ratio(full, compressed);
        assert!(ratio > 14.0);
    }
}
