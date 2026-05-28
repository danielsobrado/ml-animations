pub fn kl_discrete(_p: &[f32], _q: &[f32]) -> f32 {
    assert_eq!(_p.len(), _q.len());

    // TODO:
    // KL(p || q) = sum p_i * ln(p_i / q_i)
    // Skip terms where p_i == 0.
    todo!()
}

pub fn objective_with_kl(_reward_objective: f32, _kl: f32, _beta: f32) -> f32 {
    // TODO:
    // Larger is better: reward_objective - beta * kl
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_when_same_distribution() {
        let p = vec![0.5, 0.5];
        assert!(kl_discrete(&p, &p).abs() < 1e-6);
    }

    #[test]
    fn kl_reduces_objective() {
        assert_eq!(objective_with_kl(10.0, 2.0, 0.5), 9.0);
    }
}
