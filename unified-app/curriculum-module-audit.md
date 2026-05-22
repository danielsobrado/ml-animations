# Curriculum Module Audit

Generated: 2026-05-22T07:22:37.820Z

## Review Policy

- Tier A: excellent custom lesson with strong interaction and assessment-facing mechanics.
- Tier B: meaningful lesson with reusable controls and working conceptual workflow.
- Tier C: adequate but currently shallow or shared lesson wrapper.
- Tier D: placeholder or insufficient quality requiring immediate conversion.

- Total active lessons: 119
- Priority paths covered: matrix-multiplication, linear-regression, probability-distributions, conditional-probability, bayes-rule-ml, expected-value-variance, sampling-confidence-intervals, hypothesis-testing-intuition, maximum-likelihood-estimation, entropy, softmax, cross-entropy, loss-functions-likelihoods, train-validation-test-split, cross-validation, data-leakage-deep-dive, feature-scaling-preprocessing, pca, k-means, logistic-regression, classification-metrics, roc-pr-curves, calibration, overfitting, bias-variance-tradeoff, regularization, knn-naive-bayes-svm, tree-ensembles, gradient-descent, neural-network, relu, computation-graph-backprop, initialization, optimizers, training-loop-dynamics, model-debugging, model-interpretability, uncertainty-estimation, model-monitoring, model-fairness

## Priority Track Coverage

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Status | Next Action |
| --- | --- | --- | ---: | ---: | --- | --- |
| bayes-rule-ml | B (good) | manual | 12020 | 0 | stable | Add a confusion-matrix bridge for calibrated classifier outputs. |
| bias-variance-tradeoff | B (good) | manual | 12684 | 0 | stable | Add explicit regularization-family comparisons and noisy-label cases. |
| calibration | A (excellent) | manual | 11706 | 0 | stable | Add dataset-shift mini-playground with live recalibration step. |
| classification-metrics | A (excellent) | manual | 187 | 0 | stable | Add subgroup slicing and calibration-by-decision-surface follow-up. |
| computation-graph-backprop | A (excellent) | manual | 13122 | 0 | stable | Add chain-rule failure mode examples and exploding-grad examples. |
| conditional-probability | B (good) | manual | 3058 | 3 | stable | Add a real-case confusion example with noisy observations. |
| cross-entropy | B (good) | manual | 2734 | 5 | stable | Add targeted counterexample where confidence collapses at wrong scale. |
| cross-validation | B (good) | manual | 16506 | 0 | stable | Add repeated-stratified CV and time-series split variants. |
| data-leakage-deep-dive | A (excellent) | manual | 11185 | 0 | stable | Add split-by-time and leakage lineage examples for longitudinal data. |
| entropy | B (good) | manual | 2734 | 2 | stable | Add sequence entropy and information gain contrast cases. |
| expected-value-variance | B (good) | manual | 3049 | 3 | stable | Add finite-sample variability case studies. |
| feature-scaling-preprocessing | A (excellent) | manual | 15848 | 0 | stable | Add additional preprocessing recipes and feature-card audit summary. |
| gradient-descent | B (good) | manual | 4288 | 7 | stable | Add saddle point and bad-conditioning playbook. |
| hypothesis-testing-intuition | B (good) | manual | 10509 | 0 | stable | Add one-sided/two-sided mode and multiple-testing correction examples. |
| initialization | A (excellent) | manual | 7603 | 0 | stable | Add variance explosion/suppression case studies. |
| k-means | A (excellent) | manual | 8941 | 0 | stable | Add centroid drift diagnostics and K-selection guidance. |
| knn-naive-bayes-svm | A (excellent) | manual | 13699 | 0 | stable | Add class-imbalance and boundary brittleness checks. |
| linear-regression | B (good) | manual | 3056 | 3 | stable | Add outlier and heteroscedastic noise counterexamples. |
| logistic-regression | B (good) | manual | 17282 | 0 | stable | Add class imbalance and cost-sensitive threshold scenarios. |
| loss-functions-likelihoods | B (good) | manual | 16379 | 0 | stable | Add multiclass categorical NLL and label-smoothing variants. |
| matrix-multiplication | B (good) | manual | 2737 | 2 | stable | Add geometric interpretation quick checks for rectangular dimensions. |
| maximum-likelihood-estimation | B (good) | manual | 17345 | 0 | stable | Add a prior-vs-likelihood contrast before introducing MAP estimation. |
| model-debugging | B (good) | manual | 19516 | 0 | stable | Add one "before/after intervention" replay mode with saved snapshots. |
| model-fairness | B (good) | manual | 15733 | 0 | stable | Add explicit subgroup-level business metric and threshold optimization walkthrough. |
| model-interpretability | B (good) | manual | 17464 | 0 | stable | Add method comparison between linear and model-agnostic explanations. |
| model-monitoring | B (good) | manual | 15880 | 0 | stable | Add incident annotation and recovery-action history. |
| neural-network | A (excellent) | manual | 40231 | 1 | stable | Add architecture ablation and under/over-parameterization checks. |
| optimizers | A (excellent) | manual | 13443 | 0 | stable | Add momentum warm-up and sparse-gradient diagnostics. |
| overfitting | B (good) | manual | 16210 | 0 | stable | Add train/validation/test replay snapshots for repeated model-selection risk. |
| pca | A (excellent) | manual | 9907 | 0 | stable | Add reconstruction-error and component-reuse examples. |
| probability-distributions | B (good) | manual | 3071 | 3 | stable | Add comparative examples that connect distribution assumptions to downstream model behavior. |
| regularization | B (good) | manual | 16990 | 0 | stable | Add model-family examples where regularization appears as depth limits, dropout, and data augmentation. |
| relu | B (good) | manual | 3019 | 3 | stable | Add gradient-flow edge-case checks across negative saturation. |
| roc-pr-curves | A (excellent) | manual | 10286 | 0 | stable | Add decision-threshold failure mode examples for class imbalance. |
| sampling-confidence-intervals | B (good) | manual | 10531 | 0 | stable | Add bootstrap interval comparison and non-proportion examples. |
| softmax | B (good) | manual | 4225 | 3 | stable | Add temperature scheduling and label-shift mini-studio. |
| train-validation-test-split | B (good) | manual | 17205 | 0 | stable | Add grouped entity split and train/serve skew examples. |
| training-loop-dynamics | A (excellent) | manual | 7656 | 0 | stable | Add metric drift alerts and early-stop recommendation panel. |
| tree-ensembles | A (excellent) | manual | 13530 | 0 | stable | Add feature-attribution edge cases and overfit detection hooks. |
| uncertainty-estimation | B (good) | manual | 14241 | 0 | stable | Add a dedicated calibration panel with empirical reliability buckets. |

## All Modules

### nlp

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Status | Next Action |
| --- | --- | --- | ---: | ---: | --- | --- |
| bag-of-words | B (good) | auto | 3840 | 6 |  | Review and set explicit manual quality entry. |
| embeddings | B (good) | auto | 1739 | 6 |  | Review and set explicit manual quality entry. |
| fasttext | B (good) | auto | 4109 | 7 |  | Review and set explicit manual quality entry. |
| glove | B (good) | auto | 3919 | 6 |  | Review and set explicit manual quality entry. |
| tokenization | B (good) | auto | 3313 | 4 |  | Review and set explicit manual quality entry. |
| word2vec | B (good) | auto | 4160 | 7 |  | Review and set explicit manual quality entry. |

### transformers

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Status | Next Action |
| --- | --- | --- | ---: | ---: | --- | --- |
| attention-masks | B (good) | auto | 12546 | 0 |  | Review and set explicit manual quality entry. |
| attention-mechanism | B (good) | auto | 5624 | 1 |  | Review and set explicit manual quality entry. |
| bert | A (excellent) | auto | 4837 | 9 |  | Review and set explicit manual quality entry. |
| fine-tuning | B (good) | auto | 3783 | 5 |  | Review and set explicit manual quality entry. |
| flash-attention | B (good) | manual | 11871 | 3 | stable | Add causal masking and backward-pass memory examples. |
| gpt2-comprehensive | A (excellent) | auto | 5370 | 10 |  | Review and set explicit manual quality entry. |
| grouped-query-attention | B (good) | manual | 12587 | 3 | stable | Add per-layer GQA examples and model-family presets. |
| kv-cache | B (good) | manual | 12446 | 3 | stable | Add paged-cache fragmentation and batch-size memory examples. |
| llm-training-objectives | B (good) | auto | 9294 | 0 |  | Review and set explicit manual quality entry. |
| moe | B (good) | auto | 5203 | 4 |  | Review and set explicit manual quality entry. |
| positional-encoding | B (good) | manual | 13319 | 3 | stable | Add relative position bias and RoPE bridge examples. |
| residual-stream | B (good) | manual | 11866 | 3 | stable | Add pre-norm versus post-norm architecture presets. |
| rope | B (good) | manual | 13838 | 3 | stable | Add RoPE scaling variants and long-context aliasing examples. |
| sampling-strategies | B (good) | auto | 11608 | 0 |  | Review and set explicit manual quality entry. |
| self-attention | B (good) | manual | 13077 | 3 | stable | Add multi-head comparison and learned projection examples. |
| transformer | B (good) | auto | 3897 | 6 |  | Review and set explicit manual quality entry. |
| transformer-architecture-families | B (good) | auto | 9912 | 0 |  | Review and set explicit manual quality entry. |
| transformer-token-generation | B (good) | auto | 11684 | 0 |  | Review and set explicit manual quality entry. |

### neural-networks

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Status | Next Action |
| --- | --- | --- | ---: | ---: | --- | --- |
| computation-graph-backprop | A (excellent) | manual | 13122 | 0 | stable | Add chain-rule failure mode examples and exploding-grad examples. |
| conv-relu | B (good) | manual | 12197 | 2 | stable | Add backward-pass gradient tracing for clipped versus active cells. |
| conv2d | C (adequate) | auto | 2723 | 2 |  | Review and set explicit manual quality entry. |
| dropout-batchnorm | B (good) | auto | 8692 | 0 |  | Review and set explicit manual quality entry. |
| gradient-problems | C (adequate) | auto | 1115 | 3 |  | Review and set explicit manual quality entry. |
| initialization | A (excellent) | manual | 7603 | 0 | stable | Add variance explosion/suppression case studies. |
| layer-normalization | C (adequate) | auto | 3059 | 3 |  | Review and set explicit manual quality entry. |
| leaky-relu | C (adequate) | auto | 3049 | 3 |  | Review and set explicit manual quality entry. |
| lstm | B (good) | auto | 3304 | 4 |  | Review and set explicit manual quality entry. |
| max-pooling | C (adequate) | auto | 1800 | 3 |  | Review and set explicit manual quality entry. |
| neural-network | A (excellent) | manual | 40231 | 1 | stable | Add architecture ablation and under/over-parameterization checks. |
| optimizers | A (excellent) | manual | 13443 | 0 | stable | Add momentum warm-up and sparse-gradient diagnostics. |
| relu | B (good) | manual | 3019 | 3 | stable | Add gradient-flow edge-case checks across negative saturation. |
| softmax | B (good) | manual | 4225 | 3 | stable | Add temperature scheduling and label-shift mini-studio. |
| training-loop-dynamics | A (excellent) | manual | 7656 | 0 | stable | Add metric drift alerts and early-stop recommendation panel. |

### advanced-models

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Status | Next Action |
| --- | --- | --- | ---: | ---: | --- | --- |
| multimodal-llm | B (good) | auto | 3335 | 4 |  | Review and set explicit manual quality entry. |
| rag | B (good) | auto | 3325 | 4 |  | Review and set explicit manual quality entry. |
| rag-chunking-context | B (good) | auto | 9305 | 0 |  | Review and set explicit manual quality entry. |
| rag-failure-modes | A (excellent) | auto | 20097 | 0 |  | Review and set explicit manual quality entry. |
| rag-reranking-grounding | A (excellent) | auto | 16659 | 0 |  | Review and set explicit manual quality entry. |
| rag-retrieval-evaluation | B (good) | auto | 12004 | 0 |  | Review and set explicit manual quality entry. |
| rag-vector-indexing | B (good) | auto | 8670 | 0 |  | Review and set explicit manual quality entry. |
| vae | B (good) | auto | 3877 | 6 |  | Review and set explicit manual quality entry. |

### math-fundamentals

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Status | Next Action |
| --- | --- | --- | ---: | ---: | --- | --- |
| change-of-basis | B (good) | auto | 5625 | 0 |  | Review and set explicit manual quality entry. |
| condition-number | B (good) | auto | 4276 | 0 |  | Review and set explicit manual quality entry. |
| determinant-volume | B (good) | auto | 4094 | 0 |  | Review and set explicit manual quality entry. |
| eigenvalue | B (good) | auto | 3706 | 5 |  | Review and set explicit manual quality entry. |
| fundamental-subspaces | C (adequate) | auto | 2597 | 3 |  | Review and set explicit manual quality entry. |
| gradient-descent | B (good) | manual | 4288 | 7 | stable | Add saddle point and bad-conditioning playbook. |
| least-squares-projection | B (good) | auto | 6443 | 0 |  | Review and set explicit manual quality entry. |
| linear-regression | B (good) | manual | 3056 | 3 | stable | Add outlier and heteroscedastic noise counterexamples. |
| low-rank-approximation | B (good) | auto | 5270 | 0 |  | Review and set explicit manual quality entry. |
| matrix-decompositions | C (adequate) | auto | 2209 | 2 |  | Review and set explicit manual quality entry. |
| matrix-multiplication | B (good) | manual | 2737 | 2 | stable | Add geometric interpretation quick checks for rectangular dimensions. |
| optimization | B (good) | auto | 1265 | 4 |  | Review and set explicit manual quality entry. |
| pca | A (excellent) | manual | 9907 | 0 | stable | Add reconstruction-error and component-reuse examples. |
| projection-matrices | B (good) | auto | 4420 | 0 |  | Review and set explicit manual quality entry. |
| pseudoinverse | B (good) | auto | 4929 | 0 |  | Review and set explicit manual quality entry. |
| qr-decomposition | C (adequate) | auto | 2738 | 2 |  | Review and set explicit manual quality entry. |
| svd | C (adequate) | auto | 2724 | 2 |  | Review and set explicit manual quality entry. |

### core-ml

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Status | Next Action |
| --- | --- | --- | ---: | ---: | --- | --- |
| bias-variance-tradeoff | B (good) | manual | 12684 | 0 | stable | Add explicit regularization-family comparisons and noisy-label cases. |
| calibration | A (excellent) | manual | 11706 | 0 | stable | Add dataset-shift mini-playground with live recalibration step. |
| classification-metrics | A (excellent) | manual | 187 | 0 | stable | Add subgroup slicing and calibration-by-decision-surface follow-up. |
| cross-validation | B (good) | manual | 16506 | 0 | stable | Add repeated-stratified CV and time-series split variants. |
| data-leakage-deep-dive | A (excellent) | manual | 11185 | 0 | stable | Add split-by-time and leakage lineage examples for longitudinal data. |
| feature-scaling-preprocessing | A (excellent) | manual | 15848 | 0 | stable | Add additional preprocessing recipes and feature-card audit summary. |
| k-means | A (excellent) | manual | 8941 | 0 | stable | Add centroid drift diagnostics and K-selection guidance. |
| knn-naive-bayes-svm | A (excellent) | manual | 13699 | 0 | stable | Add class-imbalance and boundary brittleness checks. |
| logistic-regression | B (good) | manual | 17282 | 0 | stable | Add class imbalance and cost-sensitive threshold scenarios. |
| overfitting | B (good) | manual | 16210 | 0 | stable | Add train/validation/test replay snapshots for repeated model-selection risk. |
| regularization | B (good) | manual | 16990 | 0 | stable | Add model-family examples where regularization appears as depth limits, dropout, and data augmentation. |
| roc-pr-curves | A (excellent) | manual | 10286 | 0 | stable | Add decision-threshold failure mode examples for class imbalance. |
| train-validation-test-split | B (good) | manual | 17205 | 0 | stable | Add grouped entity split and train/serve skew examples. |
| tree-ensembles | A (excellent) | manual | 13530 | 0 | stable | Add feature-attribution edge cases and overfit detection hooks. |

### model-reliability

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Status | Next Action |
| --- | --- | --- | ---: | ---: | --- | --- |
| model-debugging | B (good) | manual | 19516 | 0 | stable | Add one "before/after intervention" replay mode with saved snapshots. |
| model-fairness | B (good) | manual | 15733 | 0 | stable | Add explicit subgroup-level business metric and threshold optimization walkthrough. |
| model-interpretability | B (good) | manual | 17464 | 0 | stable | Add method comparison between linear and model-agnostic explanations. |
| model-monitoring | B (good) | manual | 15880 | 0 | stable | Add incident annotation and recovery-action history. |
| uncertainty-estimation | B (good) | manual | 14241 | 0 | stable | Add a dedicated calibration panel with empirical reliability buckets. |

### probability-stats

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Status | Next Action |
| --- | --- | --- | ---: | ---: | --- | --- |
| bayes-rule-ml | B (good) | manual | 12020 | 0 | stable | Add a confusion-matrix bridge for calibrated classifier outputs. |
| conditional-probability | B (good) | manual | 3058 | 3 | stable | Add a real-case confusion example with noisy observations. |
| cosine-similarity | C (adequate) | auto | 3026 | 3 |  | Review and set explicit manual quality entry. |
| cross-entropy | B (good) | manual | 2734 | 5 | stable | Add targeted counterexample where confidence collapses at wrong scale. |
| entropy | B (good) | manual | 2734 | 2 | stable | Add sequence entropy and information gain contrast cases. |
| expected-value-variance | B (good) | manual | 3049 | 3 | stable | Add finite-sample variability case studies. |
| hypothesis-testing-intuition | B (good) | manual | 10509 | 0 | stable | Add one-sided/two-sided mode and multiple-testing correction examples. |
| loss-functions-likelihoods | B (good) | manual | 16379 | 0 | stable | Add multiclass categorical NLL and label-smoothing variants. |
| maximum-likelihood-estimation | B (good) | manual | 17345 | 0 | stable | Add a prior-vs-likelihood contrast before introducing MAP estimation. |
| probability-distributions | B (good) | manual | 3071 | 3 | stable | Add comparative examples that connect distribution assumptions to downstream model behavior. |
| sampling-confidence-intervals | B (good) | manual | 10531 | 0 | stable | Add bootstrap interval comparison and non-proportion examples. |
| spearman-correlation | C (adequate) | auto | 3063 | 3 |  | Review and set explicit manual quality entry. |

### reinforcement-learning

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Status | Next Action |
| --- | --- | --- | ---: | ---: | --- | --- |
| actor-critic | B (good) | auto | 5439 | 0 |  | Review and set explicit manual quality entry. |
| markov-chains | B (good) | auto | 3339 | 4 |  | Review and set explicit manual quality entry. |
| mdp-formalism | B (good) | auto | 5803 | 0 |  | Review and set explicit manual quality entry. |
| policy-gradients | B (good) | auto | 6229 | 0 |  | Review and set explicit manual quality entry. |
| policy-iteration | B (good) | auto | 7866 | 0 |  | Review and set explicit manual quality entry. |
| q-learning | C (adequate) | auto | 3090 | 3 |  | Review and set explicit manual quality entry. |
| reward-shaping | B (good) | auto | 7792 | 0 |  | Review and set explicit manual quality entry. |
| rl-exploration | C (adequate) | auto | 3085 | 3 |  | Review and set explicit manual quality entry. |
| rl-foundations | C (adequate) | auto | 3011 | 3 |  | Review and set explicit manual quality entry. |
| value-iteration | B (good) | auto | 6711 | 0 |  | Review and set explicit manual quality entry. |

### algorithms

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Status | Next Action |
| --- | --- | --- | ---: | ---: | --- | --- |
| bloom-filter | C (adequate) | auto | 3047 | 3 |  | Review and set explicit manual quality entry. |
| pagerank | B (good) | auto | 3921 | 3 |  | Review and set explicit manual quality entry. |

### diffusion-models

| Lesson | Tier | Source | Size (bytes) | Side-Panel Files | Status | Next Action |
| --- | --- | --- | ---: | ---: | --- | --- |
| classifier-free-guidance | B (good) | auto | 6390 | 0 |  | Review and set explicit manual quality entry. |
| clip-encoder | B (good) | auto | 2703 | 7 |  | Review and set explicit manual quality entry. |
| diffusion-basics | B (good) | auto | 6905 | 0 |  | Review and set explicit manual quality entry. |
| diffusion-sampling | B (good) | auto | 7168 | 0 |  | Review and set explicit manual quality entry. |
| diffusion-vae | B (good) | auto | 3822 | 7 |  | Review and set explicit manual quality entry. |
| dit | B (good) | auto | 2531 | 7 |  | Review and set explicit manual quality entry. |
| flow-matching | B (good) | auto | 4227 | 7 |  | Review and set explicit manual quality entry. |
| joint-attention | B (good) | auto | 2606 | 7 |  | Review and set explicit manual quality entry. |
| sd3-overview | B (good) | auto | 3117 | 7 |  | Review and set explicit manual quality entry. |
| t5-encoder | B (good) | auto | 2708 | 7 |  | Review and set explicit manual quality entry. |
| tokenizer-bpe | B (good) | auto | 2651 | 7 |  | Review and set explicit manual quality entry. |
| unet-vs-dit | B (good) | auto | 7644 | 0 |  | Review and set explicit manual quality entry. |

## Immediate Remediation

No unexpected Tier D items were detected.
