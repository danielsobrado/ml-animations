#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub name: &'static str,
    pub bits_per_value: f32,
    pub dot_error: f32,
}

pub fn memory_ratio_vs_fp16(bits_per_value: f32) -> f32 {
    // TODO:
    // FP16 uses 16 bits per value.
    // Return compressed_size / fp16_size.
    todo!()
}

pub fn pick_best_under_error(configs: &[QuantConfig], max_error: f32) -> Option<&QuantConfig> {
    // TODO:
    // Among configs with dot_error <= max_error,
    // pick the one with smallest bits_per_value.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_smallest_safe_config() {
        let configs = vec![
            QuantConfig { name: "8-bit", bits_per_value: 8.0, dot_error: 0.01 },
            QuantConfig { name: "4-bit", bits_per_value: 4.0, dot_error: 0.03 },
            QuantConfig { name: "2-bit", bits_per_value: 2.0, dot_error: 0.20 },
        ];

        assert_eq!(pick_best_under_error(&configs, 0.05).unwrap().name, "4-bit");
    }
}
