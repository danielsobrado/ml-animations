# mini-mla

Rustlings-style exercises for the MLA / TransMLA lesson.

Each exercise contains one or more `todo!()` calls and tests. The project is expected to fail until the learner implements the missing pieces.

Run the pack with:

```bash
cargo test --bins
```

Use `cargo check --bins` when you only want to confirm the stubs compile.

Suggested order:

1. `01_kv_cache_size.rs`
2. `02_mha_gqa_cache.rs`
3. `03_latent_cache.rs`
4. `04_down_up_projection.rs`
5. `05_absorb_projection.rs`
6. `06_rope_non_commute.rs`
7. `07_decoupled_rope_cache.rs`
8. `08_gqa_repetition.rs`
9. `09_low_rank_factorization.rs`
10. `10_strategy_tradeoff.rs`
