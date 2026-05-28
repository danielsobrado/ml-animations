pub fn policy_ratio(_new_logprob: f32, _old_logprob: f32) -> f32 {
    // TODO:
    // ratio = exp(new_logprob - old_logprob)
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_logprob_ratio_one() {
        assert!((policy_ratio(-2.0, -2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn higher_new_logprob_ratio_above_one() {
        assert!(policy_ratio(-1.0, -2.0) > 1.0);
    }
}
