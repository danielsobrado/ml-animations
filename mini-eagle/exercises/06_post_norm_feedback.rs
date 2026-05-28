pub fn rms(x: &[f32]) -> f32 {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    mean_sq.sqrt()
}

pub fn rms_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let denom = (rms(x).powi(2) + eps).sqrt();
    x.iter().map(|v| v / denom).collect()
}

pub fn draft_step(h: &[f32]) -> Vec<f32> {
    // Toy residual update: a real drafter would use attention and an MLP.
    h.iter().map(|v| v + 0.5 * v.tanh()).collect()
}

pub fn run_drafter_without_post_norm(mut h: Vec<f32>, steps: usize) -> Vec<f32> {
    for _ in 0..steps {
        h = draft_step(&h);
    }
    h
}

pub fn run_drafter_with_post_norm(mut h: Vec<f32>, steps: usize) -> Vec<f32> {
    for _ in 0..steps {
        // TODO:
        // Apply draft_step, then RMSNorm before the next step.
        todo!()
    }
    h
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_norm_keeps_scale_bounded() {
        let h0 = vec![1.0, 2.0, 3.0, 4.0];

        let raw = run_drafter_without_post_norm(h0.clone(), 10);
        let normed = run_drafter_with_post_norm(h0, 10);

        assert!(rms(&raw) > 2.0);
        assert!((rms(&normed) - 1.0).abs() < 1e-4);
    }
}
