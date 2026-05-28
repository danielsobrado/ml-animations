#![allow(unused_variables)]

pub fn mha_kv_cache_elements(num_heads: usize, head_dim: usize) -> usize {
    // TODO:
    // MHA stores K and V for every head.
    todo!()
}

pub fn gqa_kv_cache_elements(num_kv_heads: usize, head_dim: usize) -> usize {
    // TODO:
    // GQA stores K and V for fewer KV heads.
    todo!()
}

pub fn mla_kv_cache_elements(latent_dim: usize, rope_dim: usize) -> usize {
    // TODO:
    // MLA stores compressed latent content plus decoupled RoPE key.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_cache_sizes() {
        assert_eq!(mha_kv_cache_elements(8, 64), 2 * 8 * 64);
        assert_eq!(gqa_kv_cache_elements(2, 64), 2 * 2 * 64);
        assert_eq!(mla_kv_cache_elements(128, 32), 160);
    }
}
