# mini-coconut

Rustlings-style exercises for the Coconut continuous latent reasoning lesson.

Each exercise contains one or more `todo!()` calls and tests. The project is expected to fail until the learner implements the missing pieces.

Run the pack with:

```bash
cargo test --bins
```

Use `cargo check --bins` when you only want to confirm the stubs compile.

Suggested order:

1. `01_mode_switch.rs`
2. `02_latent_feedback.rs`
3. `03_masked_loss.rs`
4. `04_curriculum_schedule.rs`
5. `05_pause_vs_continuous.rs`
6. `06_branch_entropy.rs`
7. `07_delayed_commitment.rs`
8. `08_latent_perturbation.rs`
9. `09_token_budget.rs`
10. `10_probe_nearest_tokens.rs`
