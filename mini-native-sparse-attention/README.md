# mini-native-sparse-attention

Rustlings-style exercises for the Native Sparse Attention lesson.

Each exercise contains one or more `todo!()` calls and tests. The project is expected to fail until the learner implements the missing pieces.

Run the pack with:

```bash
cargo test --bins
```

Use `cargo check --bins` when you only want to confirm the stubs compile.

Suggested order:

1. `01_blockify.rs`
2. `02_sliding_window.rs`
3. `03_compress_blocks.rs`
4. `04_block_scores.rs`
5. `05_topk_blocks.rs`
6. `06_selected_tokens.rs`
7. `07_gated_merge.rs`
8. `08_gqa_shared_selection.rs`
9. `09_memory_access.rs`
10. `10_nsa_budget.rs`
