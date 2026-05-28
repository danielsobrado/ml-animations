pub fn kv_cache_bits(
    layers: usize,
    tokens: usize,
    kv_heads: usize,
    head_dim: usize,
    bits_per_value: usize,
) -> usize {
    // TODO:
    // KV cache stores both K and V.
    todo!()
}

pub fn compression_ratio(full_bits: usize, compressed_bits: usize) -> f32 {
    // TODO:
    // Return full_bits / compressed_bits as f32.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn includes_keys_and_values() {
        let bits = kv_cache_bits(2, 10, 4, 8, 16);
        assert_eq!(bits, 2 * 10 * 4 * 8 * 2 * 16);
    }

    #[test]
    fn computes_ratio() {
        assert!((compression_ratio(16, 4) - 4.0).abs() < 1e-6);
    }
}
