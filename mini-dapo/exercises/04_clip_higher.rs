#![allow(unused)]

#[derive(Debug, Clone, Copy)]
pub struct ClipRange {
    pub lower: f32,
    pub upper: f32,
}

pub fn clipped_ratio(ratio: f32, range: ClipRange) -> f32 {
    // TODO:
    // Clamp ratio to [lower, upper].
    todo!()
}

pub fn clip_higher(base_eps: f32, higher_eps: f32) -> ClipRange {
    // TODO:
    // lower = 1.0 - base_eps
    // upper = 1.0 + higher_eps
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_asymmetric_clip_range() {
        let range = clip_higher(0.2, 0.28);

        assert!((range.lower - 0.8).abs() < 1e-6);
        assert!((range.upper - 1.28).abs() < 1e-6);
    }

    #[test]
    fn clamps_to_range() {
        let range = ClipRange { lower: 0.8, upper: 1.28 };

        assert_eq!(clipped_ratio(2.0, range), 1.28);
        assert_eq!(clipped_ratio(0.1, range), 0.8);
        assert_eq!(clipped_ratio(1.1, range), 1.1);
    }
}
