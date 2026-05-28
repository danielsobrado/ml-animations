# mini-grpo

Rustlings-style exercises for the GRPO reasoning RL lesson.

Each exercise contains one or more `todo!()` calls and tests. The project is expected to fail until the learner implements the missing pieces.

Run the pack with:

```bash
cargo test --bins
```

Use `cargo check --bins` when you only want to confirm the stubs compile.

Suggested order:

1. `01_group_mean.rs`
2. `02_group_advantage.rs`
3. `03_reward_correctness.rs`
4. `04_format_reward.rs`
5. `05_policy_ratio.rs`
6. `06_clip_objective.rs`
7. `07_kl_penalty.rs`
8. `08_grpo_update_signal.rs`
9. `09_batch_filtering.rs`
10. `10_distillation_dataset.rs`
