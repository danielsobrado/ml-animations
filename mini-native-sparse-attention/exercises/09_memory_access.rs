pub struct NsaConfig {
    pub seq_len: usize,
    pub compression_block: usize,
    pub selected_blocks: usize,
    pub selection_block: usize,
    pub sliding_window: usize,
}

pub fn nsa_tokens_loaded(_cfg: NsaConfig) -> usize {
    // TODO:
    // Approximate tokens/representations read:
    // compressed blocks + selected fine tokens + sliding window.
    todo!()
}

pub fn full_attention_tokens_loaded(seq_len: usize) -> usize {
    seq_len
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nsa_loads_fewer_than_full_attention() {
        let cfg = NsaConfig {
            seq_len: 65_536,
            compression_block: 64,
            selected_blocks: 32,
            selection_block: 64,
            sliding_window: 512,
        };

        assert!(nsa_tokens_loaded(cfg) < full_attention_tokens_loaded(65_536));
    }
}
