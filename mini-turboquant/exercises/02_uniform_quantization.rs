#[derive(Debug, Clone, Copy)]
pub struct QuantParams {
    pub min: f32,
    pub max: f32,
    pub levels: usize,
}

pub fn quantize_scalar(x: f32, params: QuantParams) -> usize {
    // TODO:
    // Clamp x to [min, max], then map to the nearest integer code in [0, levels - 1].
    todo!()
}

pub fn dequantize_scalar(code: usize, params: QuantParams) -> f32 {
    // TODO:
    // Map code back to the value of its quantization level.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantizes_endpoints() {
        let p = QuantParams { min: -1.0, max: 1.0, levels: 5 };
        assert_eq!(quantize_scalar(-1.0, p), 0);
        assert_eq!(quantize_scalar(1.0, p), 4);
    }

    #[test]
    fn roundtrip_is_close() {
        let p = QuantParams { min: -1.0, max: 1.0, levels: 9 };
        let code = quantize_scalar(0.5, p);
        let y = dequantize_scalar(code, p);
        assert!((y - 0.5).abs() <= 0.25);
    }
}
