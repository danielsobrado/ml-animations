#![allow(unused_variables)]

pub struct MlaConfig {
    pub latent_dim: usize,
    pub rope_dim: usize,
    pub include_values: bool,
}

pub fn decoupled_mla_cache_elements(config: MlaConfig) -> usize {
    // TODO:
    // Cache latent content plus RoPE key.
    // The latent jointly represents K/V content.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_plus_position() {
        let cfg = MlaConfig {
            latent_dim: 512,
            rope_dim: 64,
            include_values: true,
        };

        assert_eq!(decoupled_mla_cache_elements(cfg), 576);
    }
}
