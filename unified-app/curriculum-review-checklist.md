# Curriculum Review Checklist

This checklist tracks the reliability pass after the broad curriculum expansion. The goal is to keep the course trustworthy before adding more topic clusters.

## Global Gates

- [x] Catalog entries have matching animation registry routes.
- [x] Registry entries resolve to catalog lessons.
- [x] Learning-path nodes exist in the catalog.
- [x] Prerequisite ids exist in the catalog.
- [x] Track animation ids exist in the catalog.
- [x] Assessment lesson ids exist in the catalog.
- [x] Quiz answer indices are valid.
- [x] Labs include title, prompt, and success criteria.
- [x] Lazy-loaded registry modules resolve on disk.
- [x] Route smoke coverage includes every `allAnimations` id.
- [x] No Tier C or Tier D lessons remain in the generated module audit.
- [x] Learning paths and curriculum tracks show completion percentages from persisted progress.

## Advanced Lesson Correctness Pass

Use this checklist for each advanced lesson before marking it reviewed:

- [ ] Displayed formula matches the code path.
- [ ] Toy numbers and toy worlds are clearly labeled as examples.
- [ ] Approximations are described as approximations.
- [ ] Copy avoids implying guarantees when the method is heuristic or probabilistic.
- [ ] Visual sequence preserves the real causal order.
- [ ] Assessment tests the main misconception.

Priority lessons:

| Lesson | Formula/code checked | Toy labels checked | Causal order checked | Assessment misconception checked | Notes |
| --- | --- | --- | --- | --- | --- |
| transformer-token-generation | [ ] | [ ] | [ ] | [ ] | Verify top-k/top-p filtering, temperature wording, and KV-cache savings explanation. |
| sampling-strategies | [ ] | [ ] | [ ] | [ ] | Verify top-p cutoff behavior and deterministic vs stochastic copy. |
| rag-vector-indexing | [ ] | [ ] | [ ] | [ ] | Verify ANN recall/latency trade-off wording and index-build flow. |
| rag-retrieval-evaluation | [ ] | [ ] | [ ] | [ ] | Verify recall@k, MRR, nDCG examples and reranker candidate-set limits. |
| policy-gradients | [ ] | [ ] | [ ] | [ ] | Verify return weighting and baseline wording. |
| actor-critic | [ ] | [ ] | [ ] | [ ] | Verify actor/critic update order and advantage explanation. |
| diffusion-sampling | [ ] | [ ] | [ ] | [ ] | Verify DDPM/DDIM/flow-style sampling labels and step-count trade-offs. |
| classifier-free-guidance | [ ] | [ ] | [ ] | [ ] | Verify guidance formula, oversaturation warning, and unconditional branch wording. |
| unet-vs-dit | [ ] | [ ] | [ ] | [ ] | Verify architecture comparison avoids implying one universal winner. |

## Assessment Depth Pass

Target depth for priority lessons:

- [ ] 3-5 quiz questions.
- [ ] One predict-before-running task.
- [ ] One explain-the-failure-mode task.
- [ ] One practical lab.
- [ ] Review mode for incorrect answers.

Start Here priority batch:

| Lesson | 3-5 quiz | Predict task | Failure-mode task | Practical lab | Notes |
| --- | --- | --- | --- | --- | --- |
| matrix-multiplication | [ ] | [ ] | [ ] | [x] | Add one more misconception question. |
| linear-regression | [ ] | [ ] | [ ] | [x] | Add residual/failure-mode prompt. |
| bayes-rule-ml | [x] | [ ] | [ ] | [x] | Add predict-before-updating-prior task. |
| sampling-confidence-intervals | [x] | [ ] | [ ] | [x] | Add coverage misconception task. |
| hypothesis-testing-intuition | [x] | [ ] | [ ] | [x] | Add p-value misinterpretation task. |
| maximum-likelihood-estimation | [x] | [ ] | [ ] | [x] | Add likelihood-vs-probability failure task. |
| loss-functions-likelihoods | [x] | [ ] | [ ] | [x] | Add mismatch-between-loss-and-noise task. |
| train-validation-test-split | [ ] | [ ] | [ ] | [x] | Add leakage-specific quiz depth. |
| cross-validation | [ ] | [ ] | [ ] | [x] | Add fold variance and grouped-data task. |
| data-leakage-deep-dive | [x] | [ ] | [ ] | [x] | Add predict-score-inflation task. |
| feature-scaling-preprocessing | [x] | [ ] | [ ] | [x] | Add outlier and train-only fitting task. |
| pca | [x] | [ ] | [ ] | [x] | Add variance-vs-label-signal misconception. |
| k-means | [x] | [ ] | [ ] | [x] | Add initialization failure task. |
| logistic-regression | [x] | [ ] | [ ] | [x] | Add threshold-vs-probability task. |
| classification-metrics | [x] | [ ] | [ ] | [x] | Add class-imbalance prediction task. |
| roc-pr-curves | [x] | [ ] | [ ] | [x] | Add ROC-vs-PR selection task. |
| calibration | [x] | [ ] | [ ] | [x] | Add high-confidence-miscalibration task. |
| overfitting | [x] | [ ] | [ ] | [x] | Add validation-curve diagnosis task. |
| bias-variance-tradeoff | [x] | [ ] | [ ] | [x] | Add underfit/overfit prediction task. |
| regularization | [x] | [ ] | [ ] | [x] | Add too-strong-regularization task. |
| knn-naive-bayes-svm | [x] | [ ] | [ ] | [x] | Add model-family selection task. |
| tree-ensembles | [x] | [ ] | [ ] | [x] | Add bagging-vs-boosting failure task. |
| gradient-descent | [ ] | [ ] | [ ] | [x] | Add learning-rate prediction task. |
| neural-network | [ ] | [ ] | [ ] | [ ] | Add structured assessment. |
| relu | [x] | [ ] | [ ] | [x] | Add dead-unit diagnosis task. |
| computation-graph-backprop | [x] | [ ] | [ ] | [x] | Add local-gradient chain prediction. |
| initialization | [x] | [ ] | [ ] | [x] | Add saturation/variance failure task. |
| optimizers | [x] | [ ] | [ ] | [x] | Add Adam vs SGD prediction task. |
| training-loop-dynamics | [x] | [ ] | [ ] | [x] | Add early-stopping failure task. |
