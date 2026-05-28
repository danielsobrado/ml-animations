# mini-spec-sparse

Rustlings-style exercises for the SpecSA / SpecAttn sparse speculative decoding lesson.

Each exercise contains one or more `todo!()` calls and tests. The project is expected to fail until the learner implements the missing pieces.

Run the pack with:

```bash
cargo test --bins
```

Suggested order:

1. `01_accept_prefix.rs`
2. `02_sparse_ratio.rs`
3. `03_critical_kv_scores.rs`
4. `04_collect_two_query.rs`
5. `05_block_overlap.rs`
6. `06_exact_merged_schedule.rs`
7. `07_shared_index.rs`
8. `08_refresh_reuse.rs`
9. `09_strategy_planner.rs`
