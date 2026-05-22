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

- [x] Displayed formula matches the code path.
- [x] Toy numbers and toy worlds are clearly labeled as examples.
- [x] Approximations are described as approximations.
- [x] Copy avoids implying guarantees when the method is heuristic or probabilistic.
- [x] Visual sequence preserves the real causal order.
- [x] Assessment tests the main misconception.

Priority lessons:

| Lesson | Formula/code checked | Toy labels checked | Causal order checked | Assessment misconception checked | Notes |
| --- | --- | --- | --- | --- | --- |
| transformer-token-generation | [x] | [x] | [x] | [x] | Clarified top-p cutoff behavior, KV-cache reuse, and cache misconception. |
| sampling-strategies | [x] | [x] | [x] | [x] | Clarified top-p threshold inclusion and added fixed-token-count misconception. |
| rag-vector-indexing | [x] | [x] | [x] | [x] | Set exact search to 100% recall baseline and added ANN perfect-recall misconception. |
| rag-retrieval-evaluation | [x] | [x] | [x] | [x] | Metrics verified and added reranker-missing-evidence misconception. |
| policy-gradients | [x] | [x] | [x] | [x] | Labeled toy update and added noisy-local-update misconception. |
| actor-critic | [x] | [x] | [x] | [x] | Clarified actor/critic roles and added critic-action-choice misconception. |
| diffusion-sampling | [x] | [x] | [x] | [x] | Clarified beginner flow/ODE comparison and added deterministic-quality misconception. |
| classifier-free-guidance | [x] | [x] | [x] | [x] | Confirmed guidance formula and added max-guidance misconception. |
| unet-vs-dit | [x] | [x] | [x] | [x] | Reworded comparison label and added universal-winner misconception. |

## Assessment Depth Pass

Target depth for priority lessons:

- [x] 3-5 quiz questions.
- [x] One predict-before-running task.
- [x] One explain-the-failure-mode task.
- [x] One practical lab.
- [x] Review mode for incorrect answers.

Start Here priority batch:

| Lesson | 3-5 quiz | Predict task | Failure-mode task | Practical lab | Notes |
| --- | --- | --- | --- | --- | --- |
| matrix-multiplication | [x] | [x] | [x] | [x] | Added shape prediction, dimension mismatch, and elementwise-confusion failure questions. |
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
| relu | [x] | [x] | [x] | [x] | Added cross-zero prediction and dead-unit diagnosis. |
| computation-graph-backprop | [x] | [x] | [x] | [x] | Added local-gradient multiplication prediction and missing-derivative failure. |
| initialization | [x] | [x] | [x] | [x] | Added tiny-weight prediction and symmetry failure. |
| optimizers | [x] | [x] | [x] | [x] | Added momentum path prediction and Adam-vs-SGD validation caveat. |
| training-loop-dynamics | [x] | [x] | [x] | [x] | Added overshoot prediction and early-stopping failure. |
