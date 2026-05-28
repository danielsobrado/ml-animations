pub fn concat(xs: &[Vec<f32>]) -> Vec<f32> {
    xs.iter().flat_map(|v| v.iter().copied()).collect()
}

pub fn rms_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let denom = (mean_sq + eps).sqrt();
    x.iter().map(|v| v / denom).collect()
}

pub fn fuse_without_norm(low: &[f32], mid: &[f32], high: &[f32]) -> Vec<f32> {
    concat(&[low.to_vec(), mid.to_vec(), high.to_vec()])
}

pub fn fuse_with_fc_norm(low: &[f32], mid: &[f32], high: &[f32]) -> Vec<f32> {
    // TODO:
    // RMS-normalize each stream separately, then concatenate.
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2(x: &[f32]) -> f32 {
        x.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    #[test]
    fn high_layer_no_longer_dominates_after_norm() {
        let low = vec![1.0, 1.0];
        let mid = vec![2.0, 2.0];
        let high = vec![100.0, 100.0];

        let fused = fuse_with_fc_norm(&low, &mid, &high);

        let low_part = &fused[0..2];
        let mid_part = &fused[2..4];
        let high_part = &fused[4..6];

        assert!((l2(low_part) - l2(mid_part)).abs() < 1e-4);
        assert!((l2(mid_part) - l2(high_part)).abs() < 1e-4);
    }
}
