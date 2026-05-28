# mini-turboquant

Rustlings-style exercises for the TurboQuant lesson.

Run compile checks:

```bash
cargo test --bins --no-run
```

Run the exercise tests after implementing TODOs:

```bash
cargo test --bins
```

The TODOs intentionally fail until solved. Each file focuses on one concept from low-bit KV-cache quantization: cache sizing, scalar quantization, dot-product error, rotation, MSE versus inner-product objectives, residual correction, outlier channels, attention ranking agreement, and compression tradeoff planning.
