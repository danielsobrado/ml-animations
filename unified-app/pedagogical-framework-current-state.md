# Pedagogical Framework Current State

This note corrects an earlier analytical review that described several now-stale gaps in the Machine Learning Visualized repository.

## Current Verified State

- The GitHub Pages deployment is accessible at `https://danielsobrado.github.io/Machine-Learning-Visualized/`.
- Live lessons have six-branch concept maps covering prerequisites, mechanism, intuitions, code/formula, traps, and applications/next concepts.
- Concept-map exports and live-lesson coverage are protected by integrity tests.
- Rustlings-style code labs exist for every live lesson, with worker-backed test execution in the browser.
- The larger lab catalogue includes linear algebra, neural networks, transformers, language modeling, RAG, evaluation, and experimentation exercises.
- Shipped lab solutions are smoke-tested against their embedded tests.
- Optimizers, dropout/batch normalization, reinforcement learning, causal ML, RAG, model reliability, security/robustness, and frontier-system topics are active to varying depth.

## Remaining Priorities

- Make executable code-lab progress persistent and visible without making it a hard completion gate.
- Keep improving validated formative assessment so practice evidence is stronger than manual checkbox completion alone.
- Add a curriculum-wide graph overview later if learners need a global dependency map beyond per-lesson concept maps.
- Treat security/devops improvements, such as `SECURITY.md`, Dependabot, and container onboarding, as a separate infrastructure batch.

## Assessment Direction

For the current batch, code-lab progress remains browser-local and stores only pass metadata. Learner source code is not persisted. Progress can be exported and imported as JSON for manual backup or transfer between browsers.
