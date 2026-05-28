# mini-dapo

Rustlings-style exercises for the DAPO reasoning RL lesson.

Each exercise contains one or more `todo!()` calls and tests. The project is expected to fail until the learner implements the missing pieces.

Run the pack with:

```bash
cargo test --bins
```

Use `cargo check --bins` when you only want to confirm the stubs compile.

Suggested order:

1. `01_group_accuracy.rs`
2. `02_dynamic_sampling.rs`
3. `03_group_advantage.rs`
4. `04_clip_higher.rs`
5. `05_token_level_loss.rs`
6. `06_overlong_reward.rs`
7. `07_entropy_collapse.rs`
8. `08_effective_batch.rs`
9. `09_dapo_objective.rs`
10. `10_training_dashboard.rs`
