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
| transformer-token-generation | [x] | [x] | [x] | [ ] | Clarified top-p cutoff behavior and changed KV-cache metric to reusable prior rows. |
| sampling-strategies | [x] | [x] | [x] | [ ] | Clarified top-p threshold inclusion and rank-vs-mass filtering. |
| rag-vector-indexing | [x] | [x] | [x] | [ ] | Set exact search to 100% recall baseline and labeled ANN recall as simulated recovery. |
| rag-retrieval-evaluation | [x] | [x] | [x] | [ ] | Metrics and reranker candidate-set limits verified; no code change needed. |
| policy-gradients | [x] | [x] | [x] | [ ] | Labeled preference update as toy and clarified softmax renormalization. |
| actor-critic | [x] | [x] | [x] | [ ] | Clarified toy actor signal and critic value update wording. |
| diffusion-sampling | [x] | [x] | [x] | [ ] | Confirmed sampler labels and clarified beginner flow/ODE comparison wording. |
| classifier-free-guidance | [x] | [x] | [x] | [ ] | Confirmed guidance formula and clarified toy scalar prediction view. |
| unet-vs-dit | [x] | [x] | [x] | [ ] | Reworded comparison label to avoid universal-winner framing. |

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
| matrix-multiplication | [x] | [x] | [ ] | [x] | Added shape prediction and dimension-mismatch questions. |
| linear-regression | [x] | [x] | [x] | [x] | Added outlier failure and slope-change prediction questions. |
| bayes-rule-ml | [x] | [x] | [x] | [x] | Added base-rate shift prediction and false-alarm failure framing. |
| sampling-confidence-intervals | [x] | [x] | [x] | [x] | Added long-run coverage misconception; existing sample-size item covers prediction. |
| hypothesis-testing-intuition | [x] | [x] | [x] | [x] | Added p-value/null-probability failure; existing sample-size item covers prediction. |
| maximum-likelihood-estimation | [x] | [x] | [x] | [x] | Added Bernoulli peak prediction; existing likelihood-not-prior item covers failure. |
| loss-functions-likelihoods | [x] | [x] | [x] | [x] | Added mismatched-loss failure; existing loss-choice item covers prediction setup. |
| train-validation-test-split | [x] | [x] | [x] | [x] | Added suspicious-score leakage diagnosis. |
| cross-validation | [x] | [x] | [x] | [x] | Added subgroup fold prediction and grouped-fold lab. |
| data-leakage-deep-dive | [x] | [x] | [x] | [x] | Added score-inflation prediction and grouped-entity leakage failure. |
| feature-scaling-preprocessing | [x] | [x] | [x] | [x] | Added outlier scaler prediction and fit-before-split failure. |
| pca | [x] | [x] | [x] | [x] | Added PC1 direction prediction and variance-vs-label-signal failure. |
| k-means | [x] | [x] | [x] | [x] | Added centroid-move prediction and initialization-local-minimum failure. |
| logistic-regression | [x] | [x] | [x] | [x] | Added probability-vs-decision failure; existing threshold move covers prediction. |
| classification-metrics | [x] | [x] | [x] | [x] | Added rare-class baseline prediction and precision-only failure. |
| roc-pr-curves | [x] | [x] | [x] | [x] | Added threshold-sweep prediction and ROC-vs-PR selection under imbalance. |
| calibration | [x] | [x] | [x] | [x] | Added overconfidence bucket prediction and ranking-vs-calibration failure. |
| overfitting | [x] | [x] | [x] | [x] | Existing validation-curve and underfit contrast items cover prediction and failure. |
| bias-variance-tradeoff | [x] | [x] | [x] | [x] | Added underfit and overfit pattern predictions from train/validation errors. |
| regularization | [x] | [x] | [x] | [x] | Added too-large-lambda prediction; existing too-strong item covers failure. |
| knn-naive-bayes-svm | [x] | [x] | [x] | [x] | Added local kNN prediction and Naive Bayes redundancy failure. |
| tree-ensembles | [x] | [x] | [x] | [x] | Added forest averaging prediction and boosting-overfit failure. |
| gradient-descent | [x] | [x] | [x] | [x] | Added small-step and next-step prediction questions. |
| neural-network | [x] | [x] | [x] | [x] | Added structured XOR forward/backward assessment. |
| relu | [x] | [ ] | [ ] | [x] | Add dead-unit diagnosis task. |
| computation-graph-backprop | [x] | [ ] | [ ] | [x] | Add local-gradient chain prediction. |
| initialization | [x] | [ ] | [ ] | [x] | Add saturation/variance failure task. |
| optimizers | [x] | [ ] | [ ] | [x] | Add Adam vs SGD prediction task. |
| training-loop-dynamics | [x] | [ ] | [ ] | [x] | Add early-stopping failure task. |
