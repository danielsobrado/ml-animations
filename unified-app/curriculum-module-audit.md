# Curriculum Module Audit

Generated: 2026-05-28T09:36:45.330Z

## Review Policy

- Tier A: excellent custom lesson with strong interaction and assessment-facing mechanics.
- Tier B: meaningful lesson with reusable controls and working conceptual workflow.
- Tier C: adequate but currently shallow or shared lesson wrapper.
- Tier D: placeholder or insufficient quality requiring immediate conversion.
- Source `manual` means the lesson quality tier is manifest-claimed; source `auto` means it was inferred from inspected source shape.
- Release checklist: `npm test`, `npm run audit:quality`, `npm run test:smoke`, `npm run build`.

- Total active lessons: 152
- Priority paths covered: Start Here, Probability To ML, NLP To LLMs, RAG And Retrieval, Model Reliability, Experimentation & Causal, Vision And Generation, RL And Algorithms

## Priority Track Coverage

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Assessment Count | Lab Count | Glossary Count | Est. Min | Status | Next Action |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ab-testing-foundations | A (excellent) | manual | 11996 | 0 | 100 | 1 | 13 | 18 | stable | Add sequential testing and peeking warnings with alpha-spending illustration. |
| actor-critic | B (good) | auto | 5640 | 0 | 100 | 1 | 8 | 22 |  | Review and set explicit manual quality entry. |
| agentic-coding-systems | A (excellent) | auto | 19783 | 0 | 100 | 3 | 256 | 300 |  | Review and set explicit manual quality entry. |
| attention-masks | B (good) | auto | 12546 | 0 | 100 | 1 | 10 | 18 |  | Review and set explicit manual quality entry. |
| attention-mechanism | B (good) | auto | 5624 | 1 | 100 | 1 | 10 | 18 |  | Review and set explicit manual quality entry. |
| bayes-rule-ml | B (good) | manual | 12273 | 0 | 100 | 1 | 8 | 16 | stable | Add a confusion-matrix bridge for calibrated classifier outputs. |
| bias-variance-tradeoff | B (good) | manual | 12684 | 0 | 100 | 1 | 12 | 18 | stable | Add explicit regularization-family comparisons and noisy-label cases. |
| calibration | A (excellent) | manual | 11706 | 0 | 100 | 1 | 12 | 18 | stable | Add dataset-shift mini-playground with live recalibration step. |
| causal-graphs-dags | B (good) | manual | 3409 | 0 | 100 | 1 | 13 | 20 | stable | Add mediator controls and frontdoor-path examples. |
| classification-metrics | A (excellent) | manual | 187 | 0 | 100 | 1 | 12 | 20 | stable | Add subgroup slicing and calibration-by-decision-surface follow-up. |
| classifier-free-guidance | B (good) | auto | 6558 | 0 | 100 | 1 | 9 | 18 |  | Review and set explicit manual quality entry. |
| clip-encoder | B (good) | auto | 2703 | 7 | 100 | 0 | 9 | 25 |  | Review and set explicit manual quality entry. |
| coconut-latent-reasoning | A (excellent) | manual | 41934 | 0 | 100 | 3 | 10 | 55 | stable | Add a real embedding-probe demo that maps latent states to nearest vocabulary tokens from a small toy vocabulary. |
| computation-graph-backprop | A (excellent) | manual | 13122 | 0 | 100 | 1 | 14 | 22 | stable | Add chain-rule failure mode examples and exploding-grad examples. |
| conditional-probability | B (good) | manual | 3058 | 3 | 100 | 0 | 8 | 14 | stable | Add a real-case confusion example with noisy observations. |
| confounding-simpsons-paradox | B (good) | manual | 3814 | 0 | 100 | 1 | 13 | 18 | stable | Add matching and stratified-standardization examples. |
| conv2d | B (good) | manual | 13787 | 2 | 100 | 1 | 14 | 18 | stable | Add multi-channel input and multiple output filter examples. |
| cosine-similarity | B (good) | manual | 3278 | 3 | 100 | 1 | 8 | 12 | stable | Add sparse-vs-dense vector contrast and explicit bias/relevance audit examples. |
| cross-entropy | B (good) | manual | 2734 | 5 | 100 | 0 | 8 | 15 | stable | Add targeted counterexample where confidence collapses at wrong scale. |
| cross-validation | B (good) | manual | 16863 | 0 | 100 | 1 | 12 | 18 | stable | Add repeated-stratified CV and time-series split variants. |
| cuped-variance-reduction | B (good) | manual | 3545 | 0 | 100 | 1 | 13 | 18 | stable | Add covariate quality checks and multiple pre-treatment covariates. |
| dapo-reasoning-rl | A (excellent) | manual | 45685 | 0 | 100 | 3 | 8 | 60 | stable | Add side-by-side ablation presets from public verl reproduction runs. |
| data-leakage-deep-dive | A (excellent) | manual | 11185 | 0 | 100 | 1 | 12 | 18 | stable | Add split-by-time and leakage lineage examples for longitudinal data. |
| diffusion-basics | B (good) | auto | 6905 | 0 | 100 | 1 | 9 | 16 |  | Review and set explicit manual quality entry. |
| diffusion-language-models | A (excellent) | auto | 33051 | 0 | 100 | 4 | 256 | 60 |  | Review and set explicit manual quality entry. |
| diffusion-sampling | B (good) | auto | 7394 | 0 | 100 | 1 | 9 | 18 |  | Review and set explicit manual quality entry. |
| diffusion-vae | B (good) | auto | 3822 | 7 | 100 | 0 | 9 | 24 |  | Review and set explicit manual quality entry. |
| dit | B (good) | auto | 2531 | 7 | 100 | 0 | 9 | 25 |  | Review and set explicit manual quality entry. |
| efficient-llm-serving | A (excellent) | auto | 36574 | 0 | 100 | 4 | 256 | 75 |  | Review and set explicit manual quality entry. |
| embeddings | B (good) | auto | 1739 | 6 | 100 | 1 | 8 | 14 |  | Review and set explicit manual quality entry. |
| entropy | B (good) | manual | 2734 | 2 | 100 | 0 | 8 | 14 | stable | Add sequence entropy and information gain contrast cases. |
| expected-value-variance | B (good) | manual | 3049 | 3 | 100 | 0 | 8 | 14 | stable | Add finite-sample variability case studies. |
| feature-scaling-preprocessing | A (excellent) | manual | 15848 | 0 | 100 | 1 | 12 | 18 | stable | Add additional preprocessing recipes and feature-card audit summary. |
| fine-tuning | B (good) | auto | 3783 | 5 | 100 | 1 | 10 | 26 |  | Review and set explicit manual quality entry. |
| flow-matching | B (good) | auto | 4227 | 7 | 100 | 0 | 9 | 25 |  | Review and set explicit manual quality entry. |
| frontier-evaluation-safety | A (excellent) | auto | 35199 | 0 | 100 | 4 | 256 | 360 |  | Review and set explicit manual quality entry. |
| frontier-llm-architecture-overview | A (excellent) | manual | 45614 | 0 | 100 | 0 | 256 | 40 | stable | Add live KV-byte calculator presets tied to published model cards. |
| frontier-moe-systems | A (excellent) | manual | 62203 | 0 | 100 | 0 | 256 | 50 | stable | Incorporate hardware-specific FLOP counters and exact interconnect bandwidth calculators. |
| gpt2-comprehensive | A (excellent) | auto | 5370 | 10 | 100 | 0 | 10 | 30 |  | Review and set explicit manual quality entry. |
| gradient-descent | B (good) | manual | 4288 | 7 | 100 | 1 | 9 | 18 | stable | Add saddle point and bad-conditioning playbook. |
| grpo-reasoning | A (excellent) | manual | 39741 | 0 | 100 | 3 | 8 | 55 | stable | Add a custom prompt authoring mode with deterministic toy verifier traces. |
| hypothesis-testing-intuition | B (good) | manual | 10721 | 0 | 100 | 1 | 8 | 16 | stable | Add one-sided/two-sided mode and multiple-testing correction examples. |
| initialization | A (excellent) | manual | 7603 | 0 | 100 | 1 | 14 | 18 | stable | Add variance explosion/suppression case studies. |
| joint-attention | B (good) | auto | 2606 | 7 | 100 | 0 | 9 | 25 |  | Review and set explicit manual quality entry. |
| k-means | A (excellent) | manual | 8941 | 0 | 100 | 1 | 12 | 16 | stable | Add centroid drift diagnostics and K-selection guidance. |
| knn-naive-bayes-svm | A (excellent) | manual | 13699 | 0 | 100 | 1 | 12 | 22 | stable | Add class-imbalance and boundary brittleness checks. |
| linear-regression | B (good) | manual | 3056 | 3 | 100 | 1 | 9 | 18 | stable | Add outlier and heteroscedastic noise counterexamples. |
| llm-training-objectives | B (good) | auto | 9294 | 0 | 100 | 1 | 10 | 18 |  | Review and set explicit manual quality entry. |
| logistic-regression | B (good) | manual | 17742 | 0 | 100 | 1 | 12 | 18 | stable | Add class imbalance and cost-sensitive threshold scenarios. |
| long-context-frontier-models | A (excellent) | auto | 19370 | 0 | 100 | 3 | 256 | 60 |  | Review and set explicit manual quality entry. |
| loss-functions-likelihoods | B (good) | manual | 16750 | 0 | 100 | 1 | 8 | 18 | stable | Add multiclass categorical NLL and label-smoothing variants. |
| markov-chains | B (good) | auto | 3339 | 4 | 100 | 0 | 8 | 18 |  | Review and set explicit manual quality entry. |
| matrix-multiplication | B (good) | manual | 3063 | 2 | 100 | 1 | 9 | 14 | stable | Add geometric interpretation quick checks for rectangular dimensions. |
| max-pooling | B (good) | manual | 12569 | 3 | 100 | 1 | 14 | 16 | stable | Add average pooling and strided convolution comparisons. |
| maximum-likelihood-estimation | B (good) | manual | 17696 | 0 | 100 | 1 | 8 | 18 | stable | Add a prior-vs-likelihood contrast before introducing MAP estimation. |
| mdp-formalism | B (good) | auto | 5803 | 0 | 100 | 1 | 8 | 18 |  | Review and set explicit manual quality entry. |
| model-debugging | B (good) | manual | 19984 | 0 | 100 | 1 | 14 | 20 | stable | Add one "before/after intervention" replay mode with saved snapshots. |
| model-fairness | B (good) | manual | 15990 | 0 | 100 | 1 | 14 | 22 | stable | Add explicit subgroup-level business metric and threshold optimization walkthrough. |
| model-interpretability | B (good) | manual | 17870 | 0 | 100 | 1 | 14 | 20 | stable | Add method comparison between linear and model-agnostic explanations. |
| model-monitoring | B (good) | manual | 16259 | 0 | 100 | 1 | 14 | 20 | stable | Add incident annotation and recovery-action history. |
| multi-head-latent-attention | A (excellent) | manual | 34798 | 0 | 100 | 3 | 256 | 50 | stable | Add model-specific DeepSeek-V2/V3 dimension presets and measured kernel bandwidth data. |
| native-sparse-attention | A (excellent) | manual | 36603 | 0 | 100 | 3 | 12 | 50 | stable | Add measured kernel presets across A100/H100 and smaller GQA group layouts. |
| neural-network | A (excellent) | manual | 40273 | 1 | 100 | 1 | 14 | 18 | stable | Add architecture ablation and under/over-parameterization checks. |
| omni-multimodal-architectures | A (excellent) | auto | 33697 | 0 | 100 | 4 | 256 | 60 |  | Review and set explicit manual quality entry. |
| optimizers | A (excellent) | manual | 14650 | 1 | 100 | 2 | 14 | 20 | stable | Add momentum warm-up and sparse-gradient diagnostics. |
| overfitting | B (good) | manual | 16557 | 0 | 100 | 1 | 12 | 16 | stable | Add train/validation/test replay snapshots for repeated model-selection risk. |
| pagerank | B (good) | auto | 3921 | 3 | 100 | 0 | 8 | 14 |  | Review and set explicit manual quality entry. |
| pca | A (excellent) | manual | 10089 | 0 | 100 | 1 | 9 | 18 | stable | Add reconstruction-error and component-reuse examples. |
| policy-gradients | B (good) | auto | 6492 | 0 | 100 | 1 | 8 | 20 |  | Review and set explicit manual quality entry. |
| policy-iteration | B (good) | auto | 7866 | 0 | 100 | 1 | 8 | 18 |  | Review and set explicit manual quality entry. |
| power-sample-size | A (excellent) | manual | 11423 | 0 | 100 | 1 | 13 | 18 | stable | Add two-proportion and continuous-metric formula toggles with paired designs. |
| ppo-clipped-policy-gradient | A (excellent) | auto | 14842 | 0 | 100 | 1 | 8 | 24 |  | Review and set explicit manual quality entry. |
| probability-distributions | B (good) | manual | 3071 | 3 | 100 | 0 | 8 | 14 | stable | Add comparative examples that connect distribution assumptions to downstream model behavior. |
| propensity-scores | B (good) | manual | 3591 | 0 | 100 | 1 | 13 | 20 | stable | Add standardized mean difference tables and trimming examples. |
| q-learning | B (good) | manual | 3328 | 3 | 100 | 1 | 8 | 20 | stable | Add explicit epsilon control and compare off-policy Q-learning with on-policy SARSA. |
| rag | B (good) | auto | 3325 | 4 | 100 | 0 | 16 | 22 |  | Review and set explicit manual quality entry. |
| rag-chunking-context | B (good) | auto | 9305 | 0 | 100 | 1 | 16 | 18 |  | Review and set explicit manual quality entry. |
| rag-failure-modes | A (excellent) | auto | 20594 | 0 | 100 | 1 | 16 | 16 |  | Review and set explicit manual quality entry. |
| rag-reranking-grounding | A (excellent) | auto | 16659 | 0 | 100 | 1 | 16 | 18 |  | Review and set explicit manual quality entry. |
| rag-retrieval-evaluation | B (good) | auto | 12004 | 0 | 100 | 1 | 16 | 20 |  | Review and set explicit manual quality entry. |
| rag-vector-indexing | B (good) | auto | 8883 | 0 | 100 | 1 | 16 | 18 |  | Review and set explicit manual quality entry. |
| reasoning-rlvr-grpo | A (excellent) | manual | 64875 | 0 | 100 | 7 | 256 | 60 | stable | Add a live Python code sandbox to test custom verifiable outcome regex parsers. |
| regularization | B (good) | manual | 17348 | 0 | 100 | 1 | 12 | 18 | stable | Add model-family examples where regularization appears as depth limits, dropout, and data augmentation. |
| relu | B (good) | manual | 3019 | 3 | 100 | 1 | 14 | 10 | stable | Add gradient-flow edge-case checks across negative saturation. |
| reward-shaping | B (good) | auto | 7792 | 0 | 100 | 1 | 8 | 18 |  | Review and set explicit manual quality entry. |
| rl-exploration | B (good) | manual | 3344 | 3 | 100 | 1 | 8 | 18 | stable | Add UCB and Thompson sampling comparisons for bandit-style exploration. |
| rl-foundations | B (good) | manual | 3257 | 3 | 100 | 1 | 8 | 16 | stable | Add stochastic transition examples and a policy diagram that updates after each action. |
| roc-pr-curves | A (excellent) | manual | 10286 | 0 | 100 | 1 | 12 | 18 | stable | Add decision-threshold failure mode examples for class imbalance. |
| sampling-confidence-intervals | B (good) | manual | 10755 | 0 | 100 | 1 | 8 | 16 | stable | Add bootstrap interval comparison and non-proportion examples. |
| sampling-strategies | B (good) | auto | 11914 | 0 | 100 | 1 | 10 | 18 |  | Review and set explicit manual quality entry. |
| sd3-overview | B (good) | auto | 3117 | 7 | 100 | 0 | 9 | 25 |  | Review and set explicit manual quality entry. |
| self-attention | B (good) | manual | 13380 | 3 | 100 | 1 | 10 | 20 | stable | Add multi-head comparison and learned projection examples. |
| sequential-testing-peeking | B (good) | manual | 3688 | 0 | 100 | 1 | 13 | 18 | stable | Add Pocock and OBrien-Fleming boundary presets. |
| softmax | B (good) | manual | 4225 | 3 | 100 | 0 | 14 | 12 | stable | Add temperature scheduling and label-shift mini-studio. |
| t5-encoder | B (good) | auto | 2708 | 7 | 100 | 0 | 9 | 25 |  | Review and set explicit manual quality entry. |
| test-time-compute-thinking-budgets | A (excellent) | manual | 49049 | 0 | 100 | 4 | 256 | 50 | stable | Add live computation of optimal N using real benchmark data and model-specific cost profiles. |
| tokenization | B (good) | auto | 3313 | 4 | 100 | 1 | 8 | 15 |  | Review and set explicit manual quality entry. |
| tool-using-reasoning-models | A (excellent) | manual | 122628 | 0 | 100 | 7 | 256 | 75 | stable | Incorporate parallel tool execution and multi-node sandboxed Python interpreters. |
| train-validation-test-split | B (good) | manual | 17574 | 0 | 100 | 1 | 12 | 15 | stable | Add grouped entity split and train/serve skew examples. |
| training-loop-dynamics | A (excellent) | manual | 7656 | 0 | 100 | 1 | 14 | 20 | stable | Add metric drift alerts and early-stop recommendation panel. |
| transformer | B (good) | auto | 3897 | 6 | 100 | 1 | 10 | 25 |  | Review and set explicit manual quality entry. |
| transformer-architecture-families | B (good) | auto | 9912 | 0 | 100 | 1 | 10 | 18 |  | Review and set explicit manual quality entry. |
| transformer-token-generation | B (good) | auto | 12113 | 0 | 100 | 1 | 10 | 20 |  | Review and set explicit manual quality entry. |
| treatment-effects | B (good) | manual | 3416 | 0 | 100 | 1 | 13 | 18 | stable | Add confidence intervals for subgroup effects and multiple-testing warnings. |
| tree-ensembles | A (excellent) | manual | 13530 | 0 | 100 | 1 | 12 | 20 | stable | Add feature-attribution edge cases and overfit detection hooks. |
| uncertainty-estimation | B (good) | manual | 14573 | 0 | 100 | 1 | 14 | 20 | stable | Add a dedicated calibration panel with empirical reliability buckets. |
| unet-vs-dit | B (good) | auto | 7826 | 0 | 100 | 1 | 9 | 18 |  | Review and set explicit manual quality entry. |
| vae | B (good) | auto | 3877 | 6 | 100 | 0 | 16 | 24 |  | Review and set explicit manual quality entry. |
| value-iteration | B (good) | auto | 6711 | 0 | 100 | 1 | 8 | 20 |  | Review and set explicit manual quality entry. |

## All Modules

### nlp

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Assessment Count | Lab Count | Glossary Count | Est. Min | Status | Next Action |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| bag-of-words | B (good) | auto | 3840 | 6 | 100 | 0 | 8 | 12 |  | Review and set explicit manual quality entry. |
| embeddings | B (good) | auto | 1739 | 6 | 100 | 1 | 8 | 14 |  | Review and set explicit manual quality entry. |
| fasttext | B (good) | auto | 4109 | 7 | 100 | 0 | 8 | 12 |  | Review and set explicit manual quality entry. |
| glove | B (good) | auto | 3919 | 6 | 100 | 0 | 8 | 12 |  | Review and set explicit manual quality entry. |
| tokenization | B (good) | auto | 3313 | 4 | 100 | 1 | 8 | 15 |  | Review and set explicit manual quality entry. |
| word2vec | B (good) | auto | 4160 | 7 | 100 | 0 | 8 | 12 |  | Review and set explicit manual quality entry. |

### transformers

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Assessment Count | Lab Count | Glossary Count | Est. Min | Status | Next Action |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| attention-masks | B (good) | auto | 12546 | 0 | 100 | 1 | 10 | 18 |  | Review and set explicit manual quality entry. |
| attention-mechanism | B (good) | auto | 5624 | 1 | 100 | 1 | 10 | 18 |  | Review and set explicit manual quality entry. |
| bert | A (excellent) | auto | 4837 | 9 | 100 | 0 | 10 | 24 |  | Review and set explicit manual quality entry. |
| coconut-latent-reasoning | A (excellent) | manual | 41934 | 0 | 100 | 3 | 10 | 55 | stable | Add a real embedding-probe demo that maps latent states to nearest vocabulary tokens from a small toy vocabulary. |
| efficient-inference-compression-track | B (good) | manual | 3556 | 0 | 100 | 1 | 10 | 24 | stable | Add model-size, context-length, and KV-cache memory calculators. |
| fine-tuning | B (good) | auto | 3783 | 5 | 100 | 1 | 10 | 26 |  | Review and set explicit manual quality entry. |
| flash-attention | B (good) | manual | 12149 | 3 | 100 | 1 | 10 | 18 | stable | Add causal masking and backward-pass memory examples. |
| gpt2-comprehensive | A (excellent) | auto | 5370 | 10 | 100 | 0 | 10 | 30 |  | Review and set explicit manual quality entry. |
| grouped-query-attention | B (good) | manual | 12886 | 3 | 100 | 1 | 10 | 18 | stable | Add per-layer GQA examples and model-family presets. |
| kv-cache | B (good) | manual | 12716 | 3 | 100 | 1 | 10 | 18 | stable | Add paged-cache fragmentation and batch-size memory examples. |
| llm-training-objectives | B (good) | auto | 9294 | 0 | 100 | 1 | 10 | 18 |  | Review and set explicit manual quality entry. |
| moe | B (good) | auto | 5203 | 4 | 100 | 0 | 10 | 22 |  | Review and set explicit manual quality entry. |
| positional-encoding | B (good) | manual | 13615 | 3 | 100 | 1 | 10 | 16 | stable | Add relative position bias and RoPE bridge examples. |
| recommender-systems-ranking-track | B (good) | manual | 3539 | 0 | 100 | 1 | 10 | 24 | stable | Add matrix-factorization and nDCG worked examples. |
| residual-stream | B (good) | manual | 12116 | 3 | 100 | 1 | 10 | 22 | stable | Add pre-norm versus post-norm architecture presets. |
| rope | B (good) | manual | 14154 | 3 | 100 | 1 | 10 | 22 | stable | Add RoPE scaling variants and long-context aliasing examples. |
| sampling-strategies | B (good) | auto | 11914 | 0 | 100 | 1 | 10 | 18 |  | Review and set explicit manual quality entry. |
| self-attention | B (good) | manual | 13380 | 3 | 100 | 1 | 10 | 20 | stable | Add multi-head comparison and learned projection examples. |
| spec-sparse-attention | A (excellent) | auto | 43729 | 0 | 100 | 3 | 10 | 40 |  | Review and set explicit manual quality entry. |
| transformer | B (good) | auto | 3897 | 6 | 100 | 1 | 10 | 25 |  | Review and set explicit manual quality entry. |
| transformer-architecture-families | B (good) | auto | 9912 | 0 | 100 | 1 | 10 | 18 |  | Review and set explicit manual quality entry. |
| transformer-token-generation | B (good) | auto | 12113 | 0 | 100 | 1 | 10 | 20 |  | Review and set explicit manual quality entry. |
| turboquant | A (excellent) | auto | 39573 | 0 | 100 | 3 | 10 | 35 |  | Review and set explicit manual quality entry. |

### papers

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Assessment Count | Lab Count | Glossary Count | Est. Min | Status | Next Action |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| eagle-3-1-speculative-decoding | A (excellent) | auto | 35929 | 0 | 100 | 3 | 12 | 35 |  | Review and set explicit manual quality entry. |
| native-sparse-attention | A (excellent) | manual | 36603 | 0 | 100 | 3 | 12 | 50 | stable | Add measured kernel presets across A100/H100 and smaller GQA group layouts. |

### frontier-llms

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Assessment Count | Lab Count | Glossary Count | Est. Min | Status | Next Action |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| agentic-coding-systems | A (excellent) | auto | 19783 | 0 | 100 | 3 | 256 | 300 |  | Review and set explicit manual quality entry. |
| diffusion-language-models | A (excellent) | auto | 33051 | 0 | 100 | 4 | 256 | 60 |  | Review and set explicit manual quality entry. |
| efficient-llm-serving | A (excellent) | auto | 36574 | 0 | 100 | 4 | 256 | 75 |  | Review and set explicit manual quality entry. |
| frontier-evaluation-safety | A (excellent) | auto | 35199 | 0 | 100 | 4 | 256 | 360 |  | Review and set explicit manual quality entry. |
| frontier-llm-architecture-overview | A (excellent) | manual | 45614 | 0 | 100 | 0 | 256 | 40 | stable | Add live KV-byte calculator presets tied to published model cards. |
| frontier-moe-systems | A (excellent) | manual | 62203 | 0 | 100 | 0 | 256 | 50 | stable | Incorporate hardware-specific FLOP counters and exact interconnect bandwidth calculators. |
| long-context-frontier-models | A (excellent) | auto | 19370 | 0 | 100 | 3 | 256 | 60 |  | Review and set explicit manual quality entry. |
| multi-head-latent-attention | A (excellent) | manual | 34798 | 0 | 100 | 3 | 256 | 50 | stable | Add model-specific DeepSeek-V2/V3 dimension presets and measured kernel bandwidth data. |
| omni-multimodal-architectures | A (excellent) | auto | 33697 | 0 | 100 | 4 | 256 | 60 |  | Review and set explicit manual quality entry. |
| reasoning-rlvr-grpo | A (excellent) | manual | 64875 | 0 | 100 | 7 | 256 | 60 | stable | Add a live Python code sandbox to test custom verifiable outcome regex parsers. |
| test-time-compute-thinking-budgets | A (excellent) | manual | 49049 | 0 | 100 | 4 | 256 | 50 | stable | Add live computation of optimal N using real benchmark data and model-specific cost profiles. |
| tool-using-reasoning-models | A (excellent) | manual | 122628 | 0 | 100 | 7 | 256 | 75 | stable | Incorporate parallel tool execution and multi-node sandboxed Python interpreters. |

### neural-networks

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Assessment Count | Lab Count | Glossary Count | Est. Min | Status | Next Action |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| computation-graph-backprop | A (excellent) | manual | 13122 | 0 | 100 | 1 | 14 | 22 | stable | Add chain-rule failure mode examples and exploding-grad examples. |
| conv-relu | B (good) | manual | 12482 | 2 | 100 | 1 | 14 | 16 | stable | Add backward-pass gradient tracing for clipped versus active cells. |
| conv2d | B (good) | manual | 13787 | 2 | 100 | 1 | 14 | 18 | stable | Add multi-channel input and multiple output filter examples. |
| dropout-batchnorm | B (good) | auto | 8692 | 0 | 100 | 1 | 14 | 18 |  | Review and set explicit manual quality entry. |
| gradient-problems | B (good) | manual | 9829 | 3 | 100 | 1 | 14 | 16 | stable | Add activation-specific presets for sigmoid, tanh, ReLU, and GELU chains. |
| initialization | A (excellent) | manual | 7603 | 0 | 100 | 1 | 14 | 18 | stable | Add variance explosion/suppression case studies. |
| layer-normalization | B (good) | manual | 11905 | 3 | 100 | 1 | 14 | 16 | stable | Add RMSNorm and scale-only normalization comparison. |
| leaky-relu | B (good) | manual | 10448 | 3 | 100 | 1 | 14 | 10 | stable | Add PReLU and ELU comparisons for learned or smooth negative branches. |
| lstm | B (good) | auto | 3304 | 4 | 100 | 0 | 14 | 16 |  | Review and set explicit manual quality entry. |
| max-pooling | B (good) | manual | 12569 | 3 | 100 | 1 | 14 | 16 | stable | Add average pooling and strided convolution comparisons. |
| neural-network | A (excellent) | manual | 40273 | 1 | 100 | 1 | 14 | 18 | stable | Add architecture ablation and under/over-parameterization checks. |
| optimizers | A (excellent) | manual | 14650 | 1 | 100 | 2 | 14 | 20 | stable | Add momentum warm-up and sparse-gradient diagnostics. |
| relu | B (good) | manual | 3019 | 3 | 100 | 1 | 14 | 10 | stable | Add gradient-flow edge-case checks across negative saturation. |
| softmax | B (good) | manual | 4225 | 3 | 100 | 0 | 14 | 12 | stable | Add temperature scheduling and label-shift mini-studio. |
| training-loop-dynamics | A (excellent) | manual | 7656 | 0 | 100 | 1 | 14 | 20 | stable | Add metric drift alerts and early-stop recommendation panel. |

### advanced-models

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Assessment Count | Lab Count | Glossary Count | Est. Min | Status | Next Action |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| multimodal-llm | B (good) | auto | 3335 | 4 | 100 | 0 | 16 | 24 |  | Review and set explicit manual quality entry. |
| rag | B (good) | auto | 3325 | 4 | 100 | 0 | 16 | 22 |  | Review and set explicit manual quality entry. |
| rag-chunking-context | B (good) | auto | 9305 | 0 | 100 | 1 | 16 | 18 |  | Review and set explicit manual quality entry. |
| rag-failure-modes | A (excellent) | auto | 20594 | 0 | 100 | 1 | 16 | 16 |  | Review and set explicit manual quality entry. |
| rag-reranking-grounding | A (excellent) | auto | 16659 | 0 | 100 | 1 | 16 | 18 |  | Review and set explicit manual quality entry. |
| rag-retrieval-evaluation | B (good) | auto | 12004 | 0 | 100 | 1 | 16 | 20 |  | Review and set explicit manual quality entry. |
| rag-vector-indexing | B (good) | auto | 8883 | 0 | 100 | 1 | 16 | 18 |  | Review and set explicit manual quality entry. |
| vae | B (good) | auto | 3877 | 6 | 100 | 0 | 16 | 24 |  | Review and set explicit manual quality entry. |

### math-fundamentals

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Assessment Count | Lab Count | Glossary Count | Est. Min | Status | Next Action |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| change-of-basis | B (good) | auto | 5625 | 0 | 100 | 0 | 9 | 15 |  | Review and set explicit manual quality entry. |
| condition-number | B (good) | auto | 4276 | 0 | 100 | 0 | 9 | 15 |  | Review and set explicit manual quality entry. |
| determinant-volume | B (good) | auto | 4094 | 0 | 100 | 0 | 9 | 15 |  | Review and set explicit manual quality entry. |
| eigenvalue | B (good) | auto | 3706 | 5 | 100 | 0 | 9 | 15 |  | Review and set explicit manual quality entry. |
| fundamental-subspaces | B (good) | manual | 2831 | 3 | 100 | 1 | 9 | 15 | stable | Add numeric row-reduction examples that derive bases for all four subspaces. |
| gradient-descent | B (good) | manual | 4288 | 7 | 100 | 1 | 9 | 18 | stable | Add saddle point and bad-conditioning playbook. |
| least-squares-projection | B (good) | auto | 6443 | 0 | 100 | 0 | 9 | 15 |  | Review and set explicit manual quality entry. |
| linear-regression | B (good) | manual | 3056 | 3 | 100 | 1 | 9 | 18 | stable | Add outlier and heteroscedastic noise counterexamples. |
| low-rank-approximation | B (good) | auto | 5270 | 0 | 100 | 0 | 9 | 15 |  | Review and set explicit manual quality entry. |
| matrix-decompositions | B (good) | manual | 2443 | 2 | 100 | 1 | 9 | 15 | stable | Add side-by-side numeric examples for LU, QR, SVD, and Cholesky on small matrices. |
| matrix-multiplication | B (good) | manual | 3063 | 2 | 100 | 1 | 9 | 14 | stable | Add geometric interpretation quick checks for rectangular dimensions. |
| optimization | B (good) | auto | 1265 | 4 | 100 | 0 | 9 | 15 |  | Review and set explicit manual quality entry. |
| pca | A (excellent) | manual | 10089 | 0 | 100 | 1 | 9 | 18 | stable | Add reconstruction-error and component-reuse examples. |
| projection-matrices | B (good) | auto | 4420 | 0 | 100 | 0 | 9 | 15 |  | Review and set explicit manual quality entry. |
| pseudoinverse | B (good) | auto | 4929 | 0 | 100 | 0 | 9 | 15 |  | Review and set explicit manual quality entry. |
| qr-decomposition | B (good) | manual | 2975 | 2 | 100 | 1 | 9 | 15 | stable | Add Householder QR comparison for numerical stability. |
| svd | B (good) | manual | 2948 | 2 | 100 | 1 | 9 | 15 | stable | Add explicit rank-k reconstruction error controls. |

### core-ml

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Assessment Count | Lab Count | Glossary Count | Est. Min | Status | Next Action |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| bias-variance-tradeoff | B (good) | manual | 12684 | 0 | 100 | 1 | 12 | 18 | stable | Add explicit regularization-family comparisons and noisy-label cases. |
| calibration | A (excellent) | manual | 11706 | 0 | 100 | 1 | 12 | 18 | stable | Add dataset-shift mini-playground with live recalibration step. |
| classification-metrics | A (excellent) | manual | 187 | 0 | 100 | 1 | 12 | 20 | stable | Add subgroup slicing and calibration-by-decision-surface follow-up. |
| cross-validation | B (good) | manual | 16863 | 0 | 100 | 1 | 12 | 18 | stable | Add repeated-stratified CV and time-series split variants. |
| data-engineering-for-ml-track | B (good) | manual | 3586 | 0 | 100 | 1 | 12 | 22 | stable | Add target-encoding leakage and feature-store materialization examples. |
| data-leakage-deep-dive | A (excellent) | manual | 11185 | 0 | 100 | 1 | 12 | 18 | stable | Add split-by-time and leakage lineage examples for longitudinal data. |
| feature-scaling-preprocessing | A (excellent) | manual | 15848 | 0 | 100 | 1 | 12 | 18 | stable | Add additional preprocessing recipes and feature-card audit summary. |
| k-means | A (excellent) | manual | 8941 | 0 | 100 | 1 | 12 | 16 | stable | Add centroid drift diagnostics and K-selection guidance. |
| knn-naive-bayes-svm | A (excellent) | manual | 13699 | 0 | 100 | 1 | 12 | 22 | stable | Add class-imbalance and boundary brittleness checks. |
| logistic-regression | B (good) | manual | 17742 | 0 | 100 | 1 | 12 | 18 | stable | Add class imbalance and cost-sensitive threshold scenarios. |
| overfitting | B (good) | manual | 16557 | 0 | 100 | 1 | 12 | 16 | stable | Add train/validation/test replay snapshots for repeated model-selection risk. |
| regularization | B (good) | manual | 17348 | 0 | 100 | 1 | 12 | 18 | stable | Add model-family examples where regularization appears as depth limits, dropout, and data augmentation. |
| roc-pr-curves | A (excellent) | manual | 10286 | 0 | 100 | 1 | 12 | 18 | stable | Add decision-threshold failure mode examples for class imbalance. |
| time-series-forecasting-track | B (good) | manual | 3466 | 0 | 100 | 1 | 12 | 22 | stable | Add explicit MAE/RMSE/MAPE/pinball metric scenarios. |
| train-validation-test-split | B (good) | manual | 17574 | 0 | 100 | 1 | 12 | 15 | stable | Add grouped entity split and train/serve skew examples. |
| tree-ensembles | A (excellent) | manual | 13530 | 0 | 100 | 1 | 12 | 20 | stable | Add feature-attribution edge cases and overfit detection hooks. |

### model-reliability

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Assessment Count | Lab Count | Glossary Count | Est. Min | Status | Next Action |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ml-security-robustness-track | B (good) | manual | 3627 | 0 | 100 | 1 | 14 | 24 | stable | Add concrete prompt-injection and retrieval-poisoning case studies. |
| model-debugging | B (good) | manual | 19984 | 0 | 100 | 1 | 14 | 20 | stable | Add one "before/after intervention" replay mode with saved snapshots. |
| model-fairness | B (good) | manual | 15990 | 0 | 100 | 1 | 14 | 22 | stable | Add explicit subgroup-level business metric and threshold optimization walkthrough. |
| model-interpretability | B (good) | manual | 17870 | 0 | 100 | 1 | 14 | 20 | stable | Add method comparison between linear and model-agnostic explanations. |
| model-monitoring | B (good) | manual | 16259 | 0 | 100 | 1 | 14 | 20 | stable | Add incident annotation and recovery-action history. |
| uncertainty-estimation | B (good) | manual | 14573 | 0 | 100 | 1 | 14 | 20 | stable | Add a dedicated calibration panel with empirical reliability buckets. |

### experimentation-causal-ml

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Assessment Count | Lab Count | Glossary Count | Est. Min | Status | Next Action |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ab-testing-foundations | A (excellent) | manual | 11996 | 0 | 100 | 1 | 13 | 18 | stable | Add sequential testing and peeking warnings with alpha-spending illustration. |
| causal-graphs-dags | B (good) | manual | 3409 | 0 | 100 | 1 | 13 | 20 | stable | Add mediator controls and frontdoor-path examples. |
| confounding-simpsons-paradox | B (good) | manual | 3814 | 0 | 100 | 1 | 13 | 18 | stable | Add matching and stratified-standardization examples. |
| cuped-variance-reduction | B (good) | manual | 3545 | 0 | 100 | 1 | 13 | 18 | stable | Add covariate quality checks and multiple pre-treatment covariates. |
| power-sample-size | A (excellent) | manual | 11423 | 0 | 100 | 1 | 13 | 18 | stable | Add two-proportion and continuous-metric formula toggles with paired designs. |
| propensity-scores | B (good) | manual | 3591 | 0 | 100 | 1 | 13 | 20 | stable | Add standardized mean difference tables and trimming examples. |
| sequential-testing-peeking | B (good) | manual | 3688 | 0 | 100 | 1 | 13 | 18 | stable | Add Pocock and OBrien-Fleming boundary presets. |
| treatment-effects | B (good) | manual | 3416 | 0 | 100 | 1 | 13 | 18 | stable | Add confidence intervals for subgroup effects and multiple-testing warnings. |

### probability-stats

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Assessment Count | Lab Count | Glossary Count | Est. Min | Status | Next Action |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| bayes-rule-ml | B (good) | manual | 12273 | 0 | 100 | 1 | 8 | 16 | stable | Add a confusion-matrix bridge for calibrated classifier outputs. |
| conditional-probability | B (good) | manual | 3058 | 3 | 100 | 0 | 8 | 14 | stable | Add a real-case confusion example with noisy observations. |
| cosine-similarity | B (good) | manual | 3278 | 3 | 100 | 1 | 8 | 12 | stable | Add sparse-vs-dense vector contrast and explicit bias/relevance audit examples. |
| cross-entropy | B (good) | manual | 2734 | 5 | 100 | 0 | 8 | 15 | stable | Add targeted counterexample where confidence collapses at wrong scale. |
| entropy | B (good) | manual | 2734 | 2 | 100 | 0 | 8 | 14 | stable | Add sequence entropy and information gain contrast cases. |
| expected-value-variance | B (good) | manual | 3049 | 3 | 100 | 0 | 8 | 14 | stable | Add finite-sample variability case studies. |
| hypothesis-testing-intuition | B (good) | manual | 10721 | 0 | 100 | 1 | 8 | 16 | stable | Add one-sided/two-sided mode and multiple-testing correction examples. |
| loss-functions-likelihoods | B (good) | manual | 16750 | 0 | 100 | 1 | 8 | 18 | stable | Add multiclass categorical NLL and label-smoothing variants. |
| maximum-likelihood-estimation | B (good) | manual | 17696 | 0 | 100 | 1 | 8 | 18 | stable | Add a prior-vs-likelihood contrast before introducing MAP estimation. |
| probability-distributions | B (good) | manual | 3071 | 3 | 100 | 0 | 8 | 14 | stable | Add comparative examples that connect distribution assumptions to downstream model behavior. |
| sampling-confidence-intervals | B (good) | manual | 10755 | 0 | 100 | 1 | 8 | 16 | stable | Add bootstrap interval comparison and non-proportion examples. |
| spearman-correlation | B (good) | manual | 3321 | 3 | 100 | 1 | 8 | 14 | stable | Add tied-rank handling and a small monotonic-vs-nonmonotonic scenario picker. |

### reinforcement-learning

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Assessment Count | Lab Count | Glossary Count | Est. Min | Status | Next Action |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| actor-critic | B (good) | auto | 5640 | 0 | 100 | 1 | 8 | 22 |  | Review and set explicit manual quality entry. |
| dapo-reasoning-rl | A (excellent) | manual | 45685 | 0 | 100 | 3 | 8 | 60 | stable | Add side-by-side ablation presets from public verl reproduction runs. |
| grpo-reasoning | A (excellent) | manual | 39741 | 0 | 100 | 3 | 8 | 55 | stable | Add a custom prompt authoring mode with deterministic toy verifier traces. |
| markov-chains | B (good) | auto | 3339 | 4 | 100 | 0 | 8 | 18 |  | Review and set explicit manual quality entry. |
| mdp-formalism | B (good) | auto | 5803 | 0 | 100 | 1 | 8 | 18 |  | Review and set explicit manual quality entry. |
| policy-gradients | B (good) | auto | 6492 | 0 | 100 | 1 | 8 | 20 |  | Review and set explicit manual quality entry. |
| policy-iteration | B (good) | auto | 7866 | 0 | 100 | 1 | 8 | 18 |  | Review and set explicit manual quality entry. |
| ppo-clipped-policy-gradient | A (excellent) | auto | 14842 | 0 | 100 | 1 | 8 | 24 |  | Review and set explicit manual quality entry. |
| q-learning | B (good) | manual | 3328 | 3 | 100 | 1 | 8 | 20 | stable | Add explicit epsilon control and compare off-policy Q-learning with on-policy SARSA. |
| reward-shaping | B (good) | auto | 7792 | 0 | 100 | 1 | 8 | 18 |  | Review and set explicit manual quality entry. |
| rl-exploration | B (good) | manual | 3344 | 3 | 100 | 1 | 8 | 18 | stable | Add UCB and Thompson sampling comparisons for bandit-style exploration. |
| rl-foundations | B (good) | manual | 3257 | 3 | 100 | 1 | 8 | 16 | stable | Add stochastic transition examples and a policy diagram that updates after each action. |
| value-iteration | B (good) | auto | 6711 | 0 | 100 | 1 | 8 | 20 |  | Review and set explicit manual quality entry. |

### algorithms

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Assessment Count | Lab Count | Glossary Count | Est. Min | Status | Next Action |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| bloom-filter | B (good) | manual | 3289 | 3 | 100 | 1 | 8 | 14 | stable | Add a counting Bloom filter variant and side-by-side exact set memory comparison. |
| pagerank | B (good) | auto | 3921 | 3 | 100 | 0 | 8 | 14 |  | Review and set explicit manual quality entry. |

### diffusion-models

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Assessment Count | Lab Count | Glossary Count | Est. Min | Status | Next Action |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| classifier-free-guidance | B (good) | auto | 6558 | 0 | 100 | 1 | 9 | 18 |  | Review and set explicit manual quality entry. |
| clip-encoder | B (good) | auto | 2703 | 7 | 100 | 0 | 9 | 25 |  | Review and set explicit manual quality entry. |
| diffusion-basics | B (good) | auto | 6905 | 0 | 100 | 1 | 9 | 16 |  | Review and set explicit manual quality entry. |
| diffusion-sampling | B (good) | auto | 7394 | 0 | 100 | 1 | 9 | 18 |  | Review and set explicit manual quality entry. |
| diffusion-vae | B (good) | auto | 3822 | 7 | 100 | 0 | 9 | 24 |  | Review and set explicit manual quality entry. |
| dit | B (good) | auto | 2531 | 7 | 100 | 0 | 9 | 25 |  | Review and set explicit manual quality entry. |
| flow-matching | B (good) | auto | 4227 | 7 | 100 | 0 | 9 | 25 |  | Review and set explicit manual quality entry. |
| joint-attention | B (good) | auto | 2606 | 7 | 100 | 0 | 9 | 25 |  | Review and set explicit manual quality entry. |
| sd3-overview | B (good) | auto | 3117 | 7 | 100 | 0 | 9 | 25 |  | Review and set explicit manual quality entry. |
| t5-encoder | B (good) | auto | 2708 | 7 | 100 | 0 | 9 | 25 |  | Review and set explicit manual quality entry. |
| tokenizer-bpe | B (good) | auto | 2651 | 7 | 100 | 0 | 9 | 25 |  | Review and set explicit manual quality entry. |
| unet-vs-dit | B (good) | auto | 7826 | 0 | 100 | 1 | 9 | 18 |  | Review and set explicit manual quality entry. |

## Immediate Remediation

No unexpected Tier D items were detected.
