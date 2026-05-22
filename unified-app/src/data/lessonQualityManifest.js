export const MODULE_QUALITY_TIERS = {
  A: 'excellent',
  B: 'good',
  C: 'adequate',
  D: 'insufficient',
};

export const AUDIT_REMARK_LIMIT = 100;

export const MANUAL_LESSON_QUALITY = {
  'bayes-rule-ml': {
    tier: 'B',
    status: 'stable',
    reason: 'Custom Bayes workbench with population counts, posterior decomposition, false-alarm sweep, and action-threshold diagnostics.',
    nextAction: 'Add a confusion-matrix bridge for calibrated classifier outputs.',
  },
  'loss-functions-likelihoods': {
    tier: 'B',
    status: 'stable',
    reason: 'Custom loss-assumption lab comparing Gaussian, Laplace, and Bernoulli NLL with active examples and parameter sweeps.',
    nextAction: 'Add multiclass categorical NLL and label-smoothing variants.',
  },
  'maximum-likelihood-estimation': {
    tier: 'B',
    status: 'stable',
    reason: 'Custom fitting workbench with Bernoulli/Gaussian examples, candidate sweeps, log-likelihood, NLL, and MLE comparisons.',
    nextAction: 'Add a prior-vs-likelihood contrast before introducing MAP estimation.',
  },
  'sampling-confidence-intervals': {
    tier: 'B',
    status: 'stable',
    reason: 'Custom repeated-sampling lab with interval coverage, confidence/sample-size controls, and square-root margin diagnostics.',
    nextAction: 'Add bootstrap interval comparison and non-proportion examples.',
  },
  'hypothesis-testing-intuition': {
    tier: 'B',
    status: 'stable',
    reason: 'Custom signal-vs-noise simulator with p-value, alpha, power, null distribution, and practical-importance checks.',
    nextAction: 'Add one-sided/two-sided mode and multiple-testing correction examples.',
  },
  'train-validation-test-split': {
    tier: 'B',
    status: 'stable',
    reason: 'Custom split simulator with ratio controls, random/stratified/time modes, drift diagnostics, and leakage boundary warnings.',
    nextAction: 'Add grouped entity split and train/serve skew examples.',
  },
  'cross-validation': {
    tier: 'B',
    status: 'stable',
    reason: 'Custom fold-rotation lesson with grouped splits, preprocessing scope, fold variance, and leakage audit controls.',
    nextAction: 'Add repeated-stratified CV and time-series split variants.',
  },
  'logistic-regression': {
    tier: 'B',
    status: 'stable',
    reason: 'Custom classifier workbench with weight, bias, threshold, decision-boundary, probability, and confusion-matrix controls.',
    nextAction: 'Add class imbalance and cost-sensitive threshold scenarios.',
  },
  'classification-metrics': {
    tier: 'A',
    status: 'stable',
    reason: 'Has a solid practical metric surface and threshold tuning controls.',
    nextAction: 'Add subgroup slicing and calibration-by-decision-surface follow-up.',
  },
  'roc-pr-curves': {
    tier: 'A',
    status: 'stable',
    reason: 'Interactive threshold and ROC/PR visualizations are sufficiently complete.',
    nextAction: 'Add decision-threshold failure mode examples for class imbalance.',
  },
  'calibration': {
    tier: 'A',
    status: 'stable',
    reason: 'Reliable interactive explanation of reliability and confidence errors.',
    nextAction: 'Add dataset-shift mini-playground with live recalibration step.',
  },
  'overfitting': {
    tier: 'B',
    status: 'stable',
    reason: 'Custom train-vs-validation diagnostic with complexity, noise, sample-size, early-stopping, and regularization controls.',
    nextAction: 'Add train/validation/test replay snapshots for repeated model-selection risk.',
  },
  'bias-variance-tradeoff': {
    tier: 'B',
    status: 'stable',
    reason: 'Clear decomposition lesson with useful interactive controls.',
    nextAction: 'Add explicit regularization-family comparisons and noisy-label cases.',
  },
  'regularization': {
    tier: 'B',
    status: 'stable',
    reason: 'Custom penalty workbench with L1/L2/elastic-net comparisons, lambda sweep, weight shrinkage, and loss decomposition.',
    nextAction: 'Add model-family examples where regularization appears as depth limits, dropout, and data augmentation.',
  },
  'data-leakage-deep-dive': {
    tier: 'A',
    status: 'stable',
    reason: 'Hands-on workflow with mode toggles and validation audit messaging.',
    nextAction: 'Add split-by-time and leakage lineage examples for longitudinal data.',
  },
  'feature-scaling-preprocessing': {
    tier: 'A',
    status: 'stable',
    reason: 'Strong custom controls for fit scope, outlier behavior, and transformed geometry.',
    nextAction: 'Add additional preprocessing recipes and feature-card audit summary.',
  },
  'pca': {
    tier: 'A',
    status: 'stable',
    reason: 'Clear projection surface with interpretable variance capture controls.',
    nextAction: 'Add reconstruction-error and component-reuse examples.',
  },
  'probability-distributions': {
    tier: 'B',
    status: 'stable',
    reason: 'Good custom lesson structure with distinct panels and practical interactive controls.',
    nextAction: 'Add comparative examples that connect distribution assumptions to downstream model behavior.',
  },
  'k-means': {
    tier: 'A',
    status: 'stable',
    reason: 'Good custom clustering mechanics with cluster assignment feedback.',
    nextAction: 'Add centroid drift diagnostics and K-selection guidance.',
  },
  'knn-naive-bayes-svm': {
    tier: 'A',
    status: 'stable',
    reason: 'Multi-family comparison already includes meaningful controls and comparison outputs.',
    nextAction: 'Add class-imbalance and boundary brittleness checks.',
  },
  'tree-ensembles': {
    tier: 'A',
    status: 'stable',
    reason: 'Strong visual pathway through impurity, depth, and ensemble effects.',
    nextAction: 'Add feature-attribution edge cases and overfit detection hooks.',
  },
  'matrix-multiplication': {
    tier: 'B',
    status: 'stable',
    reason: 'Core mathematical lesson with reliable visual interaction patterns.',
    nextAction: 'Add geometric interpretation quick checks for rectangular dimensions.',
  },
  'linear-regression': {
    tier: 'B',
    status: 'stable',
    reason: 'Longstanding foundational lesson with active model-parameter controls.',
    nextAction: 'Add outlier and heteroscedastic noise counterexamples.',
  },
  'conditional-probability': {
    tier: 'B',
    status: 'stable',
    reason: 'Sound introductory treatment with useful event conditioning controls.',
    nextAction: 'Add a real-case confusion example with noisy observations.',
  },
  'entropy': {
    tier: 'B',
    status: 'stable',
    reason: 'Clear link from uncertainty to expectation already in place.',
    nextAction: 'Add sequence entropy and information gain contrast cases.',
  },
  'softmax': {
    tier: 'B',
    status: 'stable',
    reason: 'Includes theorem/marginalia surfaces and stable chart controls.',
    nextAction: 'Add temperature scheduling and label-shift mini-studio.',
  },
  'cross-entropy': {
    tier: 'B',
    status: 'stable',
    reason: 'Good connection from classification confidence to loss with readable math support.',
    nextAction: 'Add targeted counterexample where confidence collapses at wrong scale.',
  },
  'gradient-descent': {
    tier: 'B',
    status: 'stable',
    reason: 'Reliable optimization walkthrough with trajectory controls and interpretation.',
    nextAction: 'Add saddle point and bad-conditioning playbook.',
  },
  'neural-network': {
    tier: 'A',
    status: 'stable',
    reason: 'Dense component layer interactions and architecture-level controls.',
    nextAction: 'Add architecture ablation and under/over-parameterization checks.',
  },
  relu: {
    tier: 'B',
    status: 'stable',
    reason: 'Activation behavior is represented with interactive inputs and outputs.',
    nextAction: 'Add gradient-flow edge-case checks across negative saturation.',
  },
  'computation-graph-backprop': {
    tier: 'A',
    status: 'stable',
    reason: 'Strong pedagogical bridge with step tracing and derivative flow.',
    nextAction: 'Add chain-rule failure mode examples and exploding-grad examples.',
  },
  initialization: {
    tier: 'A',
    status: 'stable',
    reason: 'Interactive init behavior and convergence diagnostics are present.',
    nextAction: 'Add variance explosion/suppression case studies.',
  },
  optimizers: {
    tier: 'A',
    status: 'stable',
    reason: 'Compares learning-rate schedules and update behavior with practical controls.',
    nextAction: 'Add momentum warm-up and sparse-gradient diagnostics.',
  },
  'training-loop-dynamics': {
    tier: 'A',
    status: 'stable',
    reason: 'High-quality training loop sequence and validation behavior view.',
    nextAction: 'Add metric drift alerts and early-stop recommendation panel.',
  },
  'model-debugging': {
    tier: 'B',
    status: 'stable',
    reason: 'Provides a complete debugging loop with staged evidence checks, slice-level metrics, and intervention simulation.',
    nextAction: 'Add one "before/after intervention" replay mode with saved snapshots.',
  },
  'model-interpretability': {
    tier: 'B',
    status: 'stable',
    reason: 'Delivers local/global attribution views, correlation mode, and counterfactual perturbation checks.',
    nextAction: 'Add method comparison between linear and model-agnostic explanations.',
  },
  'uncertainty-estimation': {
    tier: 'B',
    status: 'stable',
    reason: 'Interactive interval decomposition, coverage tracking, and abstain policy simulation are in place.',
    nextAction: 'Add a dedicated calibration panel with empirical reliability buckets.',
  },
  'model-monitoring': {
    tier: 'B',
    status: 'stable',
    reason: 'Covers drift, throughput, calibration, and alert threshold behavior over a simulated timeline.',
    nextAction: 'Add incident annotation and recovery-action history.',
  },
  'model-fairness': {
    tier: 'B',
    status: 'stable',
    reason: 'Offers subgroup confusion summaries, threshold strategy controls, and fairness-gap tradeoff checks.',
    nextAction: 'Add explicit subgroup-level business metric and threshold optimization walkthrough.',
  },
  'self-attention': {
    tier: 'B',
    status: 'stable',
    reason: 'Custom attention workbench with query/key score tracing, softmax temperature, causal masking, value mixing, and full matrix view.',
    nextAction: 'Add multi-head comparison and learned projection examples.',
  },
  'kv-cache': {
    tier: 'B',
    status: 'stable',
    reason: 'Custom decode-step simulator showing cached key/value reuse, prefix recomputation cost, sliding-window tradeoffs, and memory growth.',
    nextAction: 'Add paged-cache fragmentation and batch-size memory examples.',
  },
  'grouped-query-attention': {
    tier: 'B',
    status: 'stable',
    reason: 'Custom MHA/MQA/GQA comparison with query-to-KV grouping, cache-memory ratio, bandwidth, and specialization tradeoff controls.',
    nextAction: 'Add per-layer GQA examples and model-family presets.',
  },
  'flash-attention': {
    tier: 'B',
    status: 'stable',
    reason: 'Custom tiled-attention workbench showing exact attention, online softmax state, SRAM working set, and avoided score-matrix memory.',
    nextAction: 'Add causal masking and backward-pass memory examples.',
  },
  'positional-encoding': {
    tier: 'B',
    status: 'stable',
    reason: 'Custom position-signal workbench comparing no position, sinusoidal, and learned absolute encodings with order ambiguity and extrapolation diagnostics.',
    nextAction: 'Add relative position bias and RoPE bridge examples.',
  },
  rope: {
    tier: 'B',
    status: 'stable',
    reason: 'Custom RoPE workbench showing Q/K pair rotations, relative-distance score behavior, frequency schedule, and context extrapolation caveats.',
    nextAction: 'Add RoPE scaling variants and long-context aliasing examples.',
  },
  'expected-value-variance': {
    tier: 'B',
    status: 'stable',
    reason: 'Statistical baseline with expectation/variance visualization hooks.',
    nextAction: 'Add finite-sample variability case studies.',
  },
};

export const D_TIER_PLACEHOLDERS = Object.entries(MANUAL_LESSON_QUALITY)
  .filter(([, info]) => info.tier === 'D')
  .map(([id]) => id);

export const STABLE_HIGH_PRIORITY_TIERS = {
  A: 'A',
  B: 'B',
  C: 'C',
};

