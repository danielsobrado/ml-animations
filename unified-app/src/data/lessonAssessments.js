export const PRIORITY_ASSESSMENT_LESSON_IDS = [
  'matrix-multiplication',
  'linear-regression',
  'pca',
  'k-means',
  'train-validation-test-split',
  'cross-validation',
  'data-leakage-deep-dive',
  'feature-scaling-preprocessing',
  'logistic-regression',
  'classification-metrics',
  'roc-pr-curves',
  'calibration',
  'overfitting',
  'bias-variance-tradeoff',
  'regularization',
  'knn-naive-bayes-svm',
  'tree-ensembles',
  'gradient-descent',
  'initialization',
  'optimizers',
  'training-loop-dynamics',
  'dropout-batchnorm',
  'relu',
  'computation-graph-backprop',
  'tokenization',
  'embeddings',
  'attention-mechanism',
  'self-attention',
  'attention-masks',
  'transformer',
  'transformer-architecture-families',
  'llm-training-objectives',
  'transformer-token-generation',
  'sampling-strategies',
  'fine-tuning',
  'rag-chunking-context',
  'rag-vector-indexing',
  'rag-reranking-grounding',
  'rag-failure-modes',
  'rag-retrieval-evaluation',
  'diffusion-basics',
  'diffusion-sampling',
  'classifier-free-guidance',
  'unet-vs-dit',
];

export const EMPTY_ASSESSMENT = Object.freeze({
  quiz: Object.freeze([]),
  labs: Object.freeze([]),
});

export const lessonAssessments = {
  'matrix-multiplication': {
    quiz: [
      {
        id: 'row-column-entry',
        prompt: 'For C = A B, what creates entry C[1,2]?',
        choices: [
          'Row 1 of A dotted with column 2 of B',
          'Column 1 of A dotted with row 2 of B',
          'A[1,2] multiplied by B[1,2]',
        ],
        answerIndex: 0,
        explanation: 'Each output entry combines one row from the left matrix with one column from the right matrix.',
      },
      {
        id: 'composition-order',
        prompt: 'Why does matrix multiplication order matter?',
        choices: [
          'The left and right matrices define a sequence of transformations',
          'The larger matrix always has to be first',
          'Only square matrices can be multiplied',
        ],
        answerIndex: 0,
        explanation: 'A B applies transformations in a specific order, so swapping them usually changes the result.',
      },
    ],
    labs: [
      {
        id: 'compute-one-cell',
        title: 'Compute one cell by hand',
        prompt: 'Pick one highlighted output cell, write the row-column dot product, then compare it with the animation.',
        successCriteria: 'Your written terms match the multiplied row and column entries in order.',
      },
    ],
  },
  'linear-regression': {
    quiz: [
      {
        id: 'line-error',
        prompt: 'What does linear regression adjust during training?',
        choices: [
          'The slope and intercept that minimize prediction error',
          'The raw labels in the dataset',
          'The train/test split after every prediction',
        ],
        answerIndex: 0,
        explanation: 'Training changes model parameters, not the observed labels or evaluation split.',
      },
      {
        id: 'residual-meaning',
        prompt: 'What is a residual?',
        choices: [
          'The difference between an observed value and the prediction',
          'The model weight after regularization',
          'The average of all feature values',
        ],
        answerIndex: 0,
        explanation: 'Residuals show how far predictions are from the observed targets.',
      },
    ],
    labs: [
      {
        id: 'move-line',
        title: 'Move the fitted line',
        prompt: 'Change the slope and intercept until residuals shrink, then identify which points still dominate the error.',
        successCriteria: 'You can name at least one point whose residual pulls the fitted line.',
      },
    ],
  },
  pca: {
    quiz: [
      {
        id: 'center-before-variance',
        prompt: 'Why does PCA center the data first?',
        choices: [
          'So variance directions describe spread around the mean',
          'So labels can be subtracted from predictions',
          'So every feature becomes a class probability',
        ],
        answerIndex: 0,
        explanation: 'PCA analyzes covariance around the mean; without centering, the origin can distort the directions.',
      },
      {
        id: 'largest-eigenvalue',
        prompt: 'Which principal component comes first?',
        choices: [
          'The covariance eigenvector with the largest eigenvalue',
          'The feature with the smallest raw units',
          'The direction chosen by the target label',
        ],
        answerIndex: 0,
        explanation: 'The largest eigenvalue marks the direction that explains the most variance in the input data.',
      },
    ],
    labs: [
      {
        id: 'projection-error',
        title: 'Compare one and two components',
        prompt: 'Switch between 1D and 2D projection, then identify when the 1D reconstruction loses the most information.',
        successCriteria: 'You can connect higher noise or weaker correlation to lower PC1 explained variance.',
      },
    ],
  },
  'k-means': {
    quiz: [
      {
        id: 'assignment-step',
        prompt: 'During the assignment step, where does each point go?',
        choices: [
          'To the nearest centroid',
          'To the cluster with the largest label value',
          'To every centroid equally',
        ],
        answerIndex: 0,
        explanation: 'K-means assigns each point to the currently nearest centroid before updating centroid positions.',
      },
      {
        id: 'centroid-update',
        prompt: 'How is a centroid updated after points are assigned?',
        choices: [
          'Move it to the mean of its assigned points',
          'Move it to the farthest point in the cluster',
          'Move it to the origin every time',
        ],
        answerIndex: 0,
        explanation: 'The centroid is the coordinate-wise average of the points currently assigned to that cluster.',
      },
    ],
    labs: [
      {
        id: 'k-vs-inertia',
        title: 'Compare k and inertia',
        prompt: 'Change k and the number of iterations, then describe when lower inertia starts splitting a natural group.',
        successCriteria: 'You can explain why inertia alone cannot choose the best k.',
      },
    ],
  },
  'train-validation-test-split': {
    quiz: [
      {
        id: 'test-set-role',
        prompt: 'What happens if the test set guides model selection?',
        choices: [
          'It turns into another validation set',
          'It becomes a larger training set',
          'It prevents overfitting automatically',
        ],
        answerIndex: 0,
        explanation: 'Repeated test-set decisions leak information and make the final score optimistic.',
      },
      {
        id: 'validation-role',
        prompt: 'Which split should tune thresholds and hyperparameters?',
        choices: [
          'Validation',
          'Test',
          'Production-only traffic',
        ],
        answerIndex: 0,
        explanation: 'Validation data is the development feedback loop; test data is reserved for the final estimate.',
      },
    ],
    labs: [
      {
        id: 'spot-leakage',
        title: 'Spot a leakage path',
        prompt: 'Move the split controls and explain one way preprocessing before splitting could leak information.',
        successCriteria: 'Your example names what statistic or label information crosses the split boundary.',
      },
    ],
  },
  'cross-validation': {
    quiz: [
      {
        id: 'pipeline-inside-fold',
        prompt: 'Why should preprocessing be fitted inside each cross-validation fold?',
        choices: [
          'So validation-fold information cannot shape the training pipeline',
          'So every fold uses a different target variable',
          'So the final test set can be used during model selection',
        ],
        answerIndex: 0,
        explanation: 'Scaling, imputation, feature selection, or embedding fitting can leak validation statistics if done before the fold split.',
      },
      {
        id: 'grouped-folds',
        prompt: 'When repeated users appear in a dataset, what split is usually safer?',
        choices: [
          'Group all rows from the same user into the same fold',
          'Randomly scatter every row independently',
          'Put all users into every validation fold',
        ],
        answerIndex: 0,
        explanation: 'Grouped folds prevent a model from seeing one row for a user in training and another row for the same user in validation.',
      },
    ],
    labs: [
      {
        id: 'leakage-audit',
        title: 'Audit a fold pipeline',
        prompt: 'Pick one preprocessing step and decide whether it must be learned inside each fold.',
        successCriteria: 'You can explain what information would leak if the step ran before splitting folds.',
      },
    ],
  },
  'data-leakage-deep-dive': {
    quiz: [
      {
        id: 'duplicate-users',
        prompt: 'Why can duplicate users create leakage?',
        choices: [
          'The model can learn user-specific patterns in train and see the same user in validation',
          'Duplicate users always make the dataset smaller',
          'Validation rows cannot have user ids',
        ],
        answerIndex: 0,
        explanation: 'If rows from the same entity cross the split, validation no longer measures generalization to unseen entities.',
      },
      {
        id: 'target-derived-feature',
        prompt: 'Which feature is a target-leakage warning sign?',
        choices: [
          'A value created from information known only after the outcome',
          'A feature scaled using training data only',
          'A categorical feature observed before prediction time',
        ],
        answerIndex: 0,
        explanation: 'Target leakage often enters through post-outcome fields, future labels, or aggregates that include the answer.',
      },
    ],
    labs: [
      {
        id: 'leakage-mode-audit',
        title: 'Audit one leakage mode',
        prompt: 'Select each leakage mode and state the boundary being crossed and the safer split or pipeline rule.',
        successCriteria: 'You can name the leaked information and the concrete prevention rule for at least three modes.',
      },
    ],
  },
  'feature-scaling-preprocessing': {
    quiz: [
      {
        id: 'train-only-fit',
        prompt: 'Why should a scaler be fitted on training data only?',
        choices: [
          'So validation or test statistics cannot shape the model pipeline',
          'So validation examples become part of the training labels',
          'So every feature keeps its original unit scale',
        ],
        answerIndex: 0,
        explanation: 'Preprocessing parameters are learned from data, so fitting them on validation or test rows leaks evaluation information.',
      },
      {
        id: 'unit-dominance',
        prompt: 'Why can raw feature units hurt distance-based models?',
        choices: [
          'A large-unit column can dominate Euclidean distance',
          'Distances ignore feature values after scaling',
          'Standardization removes the need for labels',
        ],
        answerIndex: 0,
        explanation: 'If income is measured in thousands and age in years, raw distances mostly reflect income unless features are scaled.',
      },
    ],
    labs: [
      {
        id: 'outlier-scaler-comparison',
        title: 'Compare scaler sensitivity',
        prompt: 'Toggle the outlier and compare standard, min-max, and robust scaling on the same selected point.',
        successCriteria: 'You can explain which scaler moved most and why robust scaling is less sensitive to one extreme value.',
      },
    ],
  },
  'logistic-regression': {
    quiz: [
      {
        id: 'sigmoid-role',
        prompt: 'What does the sigmoid do in binary logistic regression?',
        choices: [
          'It maps a linear score into a 0-to-1 probability',
          'It sorts examples by class label',
          'It computes the confusion matrix',
        ],
        answerIndex: 0,
        explanation: 'The model first computes a logit, then sigmoid converts that score into a probability-like output.',
      },
      {
        id: 'threshold-tradeoff',
        prompt: 'If the threshold rises from 0.5 to 0.7, what usually happens?',
        choices: [
          'Fewer examples are predicted positive',
          'More examples are predicted positive',
          'The fitted weights are retrained immediately',
        ],
        answerIndex: 0,
        explanation: 'A higher threshold requires stronger positive evidence before predicting class 1.',
      },
    ],
    labs: [
      {
        id: 'threshold-flips',
        title: 'Find threshold flips',
        prompt: 'Move the threshold and identify which points change class first.',
        successCriteria: 'You can explain each flip by comparing its probability with the threshold.',
      },
    ],
  },
  'classification-metrics': {
    quiz: [
      {
        id: 'precision-question',
        prompt: 'What does precision measure?',
        choices: [
          'Among predicted positives, how many were actually positive',
          'Among actual positives, how many were found',
          'Among all examples, how many were negative',
        ],
        answerIndex: 0,
        explanation: 'Precision focuses on the trustworthiness of positive predictions.',
      },
      {
        id: 'rare-positive-risk',
        prompt: 'Why can accuracy be misleading for rare positives?',
        choices: [
          'A model can predict mostly negative and still look accurate',
          'Accuracy ignores true negatives entirely',
          'Accuracy can only be used with continuous targets',
        ],
        answerIndex: 0,
        explanation: 'When negatives dominate, a high accuracy score can hide missed positives.',
      },
    ],
    labs: [
      {
        id: 'threshold-counts',
        title: 'Trace the confusion matrix',
        prompt: 'Move the threshold and predict which count changes before reading the metric tiles.',
        successCriteria: 'You can connect at least one threshold move to TP, FP, FN, or TN.',
      },
    ],
  },
  'roc-pr-curves': {
    quiz: [
      {
        id: 'roc-axis',
        prompt: 'What does the ROC curve put on its axes?',
        choices: [
          'True positive rate versus false positive rate',
          'Precision versus training loss',
          'Accuracy versus model size',
        ],
        answerIndex: 0,
        explanation: 'ROC curves sweep thresholds and compare recall/TPR against the false positive rate.',
      },
      {
        id: 'pr-rare-positive',
        prompt: 'Why are precision-recall curves often more revealing for rare positives?',
        choices: [
          'They focus on predicted-positive quality and positive-class coverage',
          'They ignore false positives completely',
          'They choose the threshold automatically',
        ],
        answerIndex: 0,
        explanation: 'PR curves show how many predicted positives are real and how many real positives are recovered.',
      },
    ],
    labs: [
      {
        id: 'threshold-costs',
        title: 'Choose an operating threshold',
        prompt: 'Move the threshold and pick a cutoff for a case where false negatives are more expensive than false positives.',
        successCriteria: 'You can justify the threshold using recall, precision, and the mistake costs.',
      },
    ],
  },
  calibration: {
    quiz: [
      {
        id: 'probability-meaning',
        prompt: 'What does a calibrated 0.8 score mean?',
        choices: [
          'About 80 percent of similar scored examples should be positive',
          'The example is guaranteed to be positive',
          'The model has 80 percent accuracy overall',
        ],
        answerIndex: 0,
        explanation: 'Calibration is about observed frequency within score buckets, not certainty for one row or global accuracy.',
      },
      {
        id: 'sigmoid-warning',
        prompt: 'Why is a sigmoid output not automatically calibrated?',
        choices: [
          'The model score can be overconfident or underconfident after training',
          'Sigmoid outputs are never between 0 and 1',
          'Calibration only applies to regression models',
        ],
        answerIndex: 0,
        explanation: 'A bounded probability-shaped score can still disagree with observed outcome frequencies.',
      },
    ],
    labs: [
      {
        id: 'reliability-gap',
        title: 'Find the largest reliability gap',
        prompt: 'Switch between calibrated, overconfident, and underconfident modes, then identify the bucket with the largest gap.',
        successCriteria: 'You can compare predicted probability with observed positive rate and describe the calibration error.',
      },
    ],
  },
  overfitting: {
    quiz: [
      {
        id: 'gap-signal',
        prompt: 'What is the classic overfitting signal?',
        choices: [
          'Training error falls while validation error rises',
          'Training and validation error are both high from the start',
          'Validation error is never measured',
        ],
        answerIndex: 0,
        explanation: 'Overfitting appears when the model keeps improving on training data but generalizes worse.',
      },
      {
        id: 'complexity-risk',
        prompt: 'What does too much model complexity often learn?',
        choices: [
          'Noise and quirks in the training sample',
          'Only the simplest trend',
          'The test-set labels directly',
        ],
        answerIndex: 0,
        explanation: 'A flexible model can bend around noise instead of learning reusable signal.',
      },
    ],
    labs: [
      {
        id: 'find-sweet-spot',
        title: 'Find the validation sweet spot',
        prompt: 'Move complexity until validation error is lowest, then compare it with the lowest training error.',
        successCriteria: 'You can explain why the best validation point is not necessarily the most complex model.',
      },
    ],
  },
  'bias-variance-tradeoff': {
    quiz: [
      {
        id: 'high-bias-signal',
        prompt: 'What usually signals high bias?',
        choices: [
          'Training and validation error are both high',
          'Training error is near zero while validation error rises sharply',
          'The test set was used exactly once at the end',
        ],
        answerIndex: 0,
        explanation: 'High bias means the model is too simple for the real pattern, so it performs poorly even on training data.',
      },
      {
        id: 'high-variance-fix',
        prompt: 'Which action often helps a high-variance model?',
        choices: [
          'Collect more data or regularize the model',
          'Make the model more flexible immediately',
          'Stop using validation data',
        ],
        answerIndex: 0,
        explanation: 'High variance comes from sample sensitivity, so more data, simpler models, regularization, or averaging can help.',
      },
    ],
    labs: [
      {
        id: 'complexity-sweep',
        title: 'Sweep complexity and sample size',
        prompt: 'Switch between simple, balanced, and flexible models, then change sample size and noise.',
        successCriteria: 'You can explain when the dominant problem is bias versus variance from the train and validation errors.',
      },
    ],
  },
  regularization: {
    quiz: [
      {
        id: 'penalty-purpose',
        prompt: 'Why add a regularization penalty?',
        choices: [
          'To discourage unnecessarily large or complex parameters',
          'To make the training set larger',
          'To remove the need for validation data',
        ],
        answerIndex: 0,
        explanation: 'The penalty asks the model to earn complexity with enough predictive value.',
      },
      {
        id: 'too-strong',
        prompt: 'What can happen when regularization is too strong?',
        choices: [
          'The model underfits by becoming too simple',
          'The test set becomes invalid',
          'The loss function stops using labels',
        ],
        answerIndex: 0,
        explanation: 'Excess penalty can shrink useful signal along with noisy parameters.',
      },
    ],
    labs: [
      {
        id: 'lambda-sweep',
        title: 'Sweep lambda',
        prompt: 'Increase the penalty and watch which weights shrink fastest.',
        successCriteria: 'You can describe the tradeoff between data loss, penalty, and total loss.',
      },
    ],
  },
  'knn-naive-bayes-svm': {
    quiz: [
      {
        id: 'knn-scale',
        prompt: 'Why does kNN need feature scaling?',
        choices: [
          'Euclidean distance can be dominated by the largest-unit feature',
          'Scaling changes labels into probabilities',
          'kNN ignores feature values after training',
        ],
        answerIndex: 0,
        explanation: 'kNN compares distances directly, so one unscaled large-unit column can decide most neighbor relationships.',
      },
      {
        id: 'model-assumption',
        prompt: 'Which statement best matches Gaussian Naive Bayes?',
        choices: [
          'It combines per-feature likelihoods under a conditional-independence assumption',
          'It votes among the nearest k training examples',
          'It learns a maximum-margin boundary only from support vectors',
        ],
        answerIndex: 0,
        explanation: 'Naive Bayes is probabilistic: it combines class priors with feature likelihoods, often with an independence assumption.',
      },
    ],
    labs: [
      {
        id: 'boundary-comparison',
        title: 'Compare decision changes',
        prompt: 'Move the query point near the boundary and switch between kNN, Naive Bayes, and SVM.',
        successCriteria: 'You can explain whether the decision came from local neighbors, likelihood scores, or margin side.',
      },
    ],
  },
  'tree-ensembles': {
    quiz: [
      {
        id: 'forest-variance',
        prompt: 'Why does a random forest average many trees?',
        choices: [
          'To reduce variance from any one unstable tree',
          'To make every tree deeper than before',
          'To remove the need for validation data',
        ],
        answerIndex: 0,
        explanation: 'Bagging and feature randomness make trees differ, and averaging their votes reduces variance.',
      },
      {
        id: 'boosting-sequence',
        prompt: 'What does gradient boosting add at each round?',
        choices: [
          'A weak tree that corrects remaining errors',
          'A duplicate of the first tree',
          'A random train/test split',
        ],
        answerIndex: 0,
        explanation: 'Boosting is sequential: each new weak learner targets the residual signal left by the current ensemble.',
      },
    ],
    labs: [
      {
        id: 'depth-vs-ensemble',
        title: 'Compare depth and ensemble size',
        prompt: 'Change tree depth, forest tree count, and boosting rounds, then identify which control most risks memorizing quirks.',
        successCriteria: 'You can explain the difference between deeper single trees, forest averaging, and boosting rounds.',
      },
    ],
  },
  'gradient-descent': {
    quiz: [
      {
        id: 'negative-gradient',
        prompt: 'Why does gradient descent subtract the gradient?',
        choices: [
          'The negative gradient points toward the steepest local decrease',
          'The gradient is always negative',
          'Subtraction makes the loss exactly zero',
        ],
        answerIndex: 0,
        explanation: 'The gradient points uphill, so subtracting a scaled gradient steps downhill locally.',
      },
      {
        id: 'learning-rate',
        prompt: 'What can a learning rate that is too large cause?',
        choices: [
          'Overshooting or unstable loss',
          'A guaranteed global minimum',
          'No parameter updates at all',
        ],
        answerIndex: 0,
        explanation: 'Large steps can jump past useful regions and make training oscillate or diverge.',
      },
    ],
    labs: [
      {
        id: 'tune-step-size',
        title: 'Tune step size',
        prompt: 'Try a small, medium, and large learning rate and compare the loss trace.',
        successCriteria: 'You can identify which run converges, crawls, or overshoots.',
      },
    ],
  },
  optimizers: {
    quiz: [
      {
        id: 'momentum-purpose',
        prompt: 'What does momentum add to plain SGD?',
        choices: [
          'A velocity term that accumulates repeated gradient directions',
          'A second validation set after every step',
          'A guarantee that every update reaches the global minimum',
        ],
        answerIndex: 0,
        explanation: 'Momentum keeps a running velocity, so consistent directions build speed while alternating directions get damped.',
      },
      {
        id: 'batch-noise',
        prompt: 'What usually happens when mini-batch size increases?',
        choices: [
          'Gradient estimates become smoother but each step uses more examples',
          'The learning rate becomes irrelevant',
          'Adam stops using squared-gradient estimates',
        ],
        answerIndex: 0,
        explanation: 'Larger batches average more examples, so the gradient is less noisy, but each update costs more computation.',
      },
    ],
    labs: [
      {
        id: 'compare-update-rules',
        title: 'Compare update rules',
        prompt: 'Run SGD, momentum, and Adam with the same learning rate and batch size, then identify which path zigzags least.',
        successCriteria: 'You can connect the visible path to velocity, adaptive scaling, or mini-batch noise.',
      },
    ],
  },
  initialization: {
    quiz: [
      {
        id: 'he-for-relu',
        prompt: 'Why does He initialization usually fit ReLU networks better than Xavier?',
        choices: [
          'It compensates for ReLU zeroing many activations',
          'It removes the need for backpropagation',
          'It makes every weight start with the same value',
        ],
        answerIndex: 0,
        explanation: 'He initialization uses fan-in scaling that accounts for ReLU discarding roughly half the signal.',
      },
      {
        id: 'bad-variance',
        prompt: 'What can happen when initial weight variance is far too large?',
        choices: [
          'Activations and gradients can explode through depth',
          'The model becomes perfectly regularized',
          'The optimizer no longer needs a learning rate',
        ],
        answerIndex: 0,
        explanation: 'Large initial weights can amplify signals layer after layer, making training unstable before it begins.',
      },
    ],
    labs: [
      {
        id: 'signal-scale',
        title: 'Find a stable signal scale',
        prompt: 'Switch between tiny, Xavier, He, and huge initialization while changing activation and depth.',
        successCriteria: 'You can explain which setup keeps layer variance stable and why another one vanishes or explodes.',
      },
    ],
  },
  'dropout-batchnorm': {
    quiz: [
      {
        id: 'different-jobs',
        prompt: 'What is the key difference between BatchNorm and dropout?',
        choices: [
          'BatchNorm stabilizes activation scale; dropout regularizes by masking units during training',
          'BatchNorm deletes units permanently; dropout computes batch statistics',
          'They are two names for the same inference-time operation',
        ],
        answerIndex: 0,
        explanation: 'BatchNorm normalizes and learns scale/shift, while dropout randomly masks activations during training.',
      },
      {
        id: 'dropout-inference',
        prompt: 'What happens to dropout at inference time?',
        choices: [
          'Units are no longer randomly masked',
          'The dropout rate is usually doubled',
          'Batch statistics replace every weight',
        ],
        answerIndex: 0,
        explanation: 'Dropout is a training-time regularizer; inference uses the full network with the learned weights.',
      },
    ],
    labs: [
      {
        id: 'mode-switch',
        title: 'Compare training and inference',
        prompt: 'Toggle training mode and change dropout rate, then explain why the visible output no longer masks units at inference.',
        successCriteria: 'You can separate BatchNorm scaling effects from dropout masking effects.',
      },
    ],
  },
  'training-loop-dynamics': {
    quiz: [
      {
        id: 'batch-noise-diagnosis',
        prompt: 'What usually happens when mini-batch size is very small?',
        choices: [
          'Gradient estimates become noisier from step to step',
          'Validation loss becomes unnecessary',
          'The learning rate automatically becomes zero',
        ],
        answerIndex: 0,
        explanation: 'Small batches average fewer examples, so each gradient estimate is more variable.',
      },
      {
        id: 'validation-signal',
        prompt: 'Why watch validation loss during training?',
        choices: [
          'It helps detect overfitting even while training loss falls',
          'It replaces the need for a training objective',
          'It makes every optimizer deterministic',
        ],
        answerIndex: 0,
        explanation: 'Validation loss shows whether improvements on training data are transferring to held-out data.',
      },
    ],
    labs: [
      {
        id: 'loop-diagnosis',
        title: 'Diagnose a training loop',
        prompt: 'Increase learning rate, shrink batch size, and raise validation difficulty, then identify noisy, overshooting, or overfitting behavior.',
        successCriteria: 'You can explain why train loss, validation loss, and update stability must be read together.',
      },
    ],
  },
  relu: {
    quiz: [
      {
        id: 'relu-output',
        prompt: 'What does ReLU output for a negative input?',
        choices: [
          '0',
          'The same negative number',
          'A probability',
        ],
        answerIndex: 0,
        explanation: 'ReLU passes positive values through and clips negative values to zero.',
      },
      {
        id: 'dead-zone',
        prompt: 'Why can inactive ReLUs stop learning for an example?',
        choices: [
          'Their local derivative is zero in the negative region',
          'Their weights are not part of the graph',
          'They convert labels into logits',
        ],
        answerIndex: 0,
        explanation: 'When the pre-activation is negative, ReLU blocks the local gradient.',
      },
    ],
    labs: [
      {
        id: 'cross-zero',
        title: 'Cross the kink',
        prompt: 'Move the input across zero and watch both output and slope change.',
        successCriteria: 'You can name the active region and the blocked region.',
      },
    ],
  },
  'computation-graph-backprop': {
    quiz: [
      {
        id: 'reverse-accumulation',
        prompt: 'What does backpropagation accumulate as it walks backward?',
        choices: [
          'Upstream gradients multiplied by local derivatives',
          'New input examples',
          'Only the final loss value',
        ],
        answerIndex: 0,
        explanation: 'Backprop is the chain rule applied in reverse graph order.',
      },
      {
        id: 'relu-block',
        prompt: 'In this lesson, what happens to gradients when z is below zero?',
        choices: [
          'ReLU gives a local derivative of zero and blocks them',
          'They double automatically',
          'They bypass the weight update',
        ],
        answerIndex: 0,
        explanation: 'The MSE/ReLU example makes the activation gate visible in the backward pass.',
      },
    ],
    labs: [
      {
        id: 'predict-update',
        title: 'Predict the update',
        prompt: 'Before switching to the update tab, predict whether the next loss will fall.',
        successCriteria: 'Your prediction uses the gradient sign and the learning rate.',
      },
    ],
  },
  tokenization: {
    quiz: [
      {
        id: 'subword-purpose',
        prompt: 'Why do many modern tokenizers use subwords?',
        choices: [
          'They balance vocabulary size with the ability to represent rare words',
          'They guarantee every word is one token',
          'They remove the need for embeddings',
        ],
        answerIndex: 0,
        explanation: 'Subwords let the model compose rare or new words from familiar pieces.',
      },
      {
        id: 'boundary-risk',
        prompt: 'What can token boundaries change?',
        choices: [
          'The sequence length and the units the model sees',
          'The model architecture category',
          'The labels in supervised data',
        ],
        answerIndex: 0,
        explanation: 'Different splits change both context length and the text fragments fed downstream.',
      },
    ],
    labs: [
      {
        id: 'compare-splits',
        title: 'Compare token splits',
        prompt: 'Tokenize the same phrase with two modes and identify the longest changed fragment.',
        successCriteria: 'You can explain why one split creates more or fewer tokens.',
      },
    ],
  },
  embeddings: {
    quiz: [
      {
        id: 'embedding-purpose',
        prompt: 'What is an embedding?',
        choices: [
          'A dense vector that represents an item in a learned space',
          'A one-hot label that never changes',
          'A loss value for a classifier',
        ],
        answerIndex: 0,
        explanation: 'Embeddings turn discrete items into vectors that can be compared and transformed.',
      },
      {
        id: 'distance-meaning',
        prompt: 'What does distance or direction in embedding space usually suggest?',
        choices: [
          'A learned relationship, not a guaranteed human concept',
          'A fixed dictionary order',
          'The exact probability of the next token',
        ],
        answerIndex: 0,
        explanation: 'Embedding geometry is learned from data and objectives, so it needs interpretation.',
      },
    ],
    labs: [
      {
        id: 'nearest-neighbor',
        title: 'Inspect a neighbor',
        prompt: 'Move or choose a vector and compare its nearest neighbors.',
        successCriteria: 'You can separate semantic similarity from incidental training-data correlation.',
      },
    ],
  },
  'attention-mechanism': {
    quiz: [
      {
        id: 'attention-role',
        prompt: 'What does attention compute?',
        choices: [
          'A weighted mixture of value vectors based on query-key scores',
          'A fixed average of every token',
          'A single train/test split',
        ],
        answerIndex: 0,
        explanation: 'Queries score keys, softmax turns scores into weights, and those weights mix values.',
      },
      {
        id: 'score-meaning',
        prompt: 'What does a larger query-key score usually mean?',
        choices: [
          'That value receives more weight after softmax',
          'That token is deleted from context',
          'That the loss is already minimized',
        ],
        answerIndex: 0,
        explanation: 'Higher compatible scores become larger attention weights relative to competing keys.',
      },
    ],
    labs: [
      {
        id: 'shift-query',
        title: 'Shift the query',
        prompt: 'Change a query and predict which value will receive more weight.',
        successCriteria: 'Your prediction follows the largest query-key score.',
      },
    ],
  },
  'self-attention': {
    quiz: [
      {
        id: 'scaled-scores',
        prompt: 'Why are dot-product attention scores scaled by sqrt(d_k)?',
        choices: [
          'To keep large vector dimensions from making softmax too sharp',
          'To convert values into labels',
          'To remove the need for positional information',
        ],
        answerIndex: 0,
        explanation: 'Scaling keeps score magnitudes in a range where softmax can still distribute attention.',
      },
      {
        id: 'token-specific',
        prompt: 'In self-attention, what is recomputed for each token position?',
        choices: [
          'A context vector made from weighted values',
          'The dataset split',
          'The vocabulary itself',
        ],
        answerIndex: 0,
        explanation: 'Every position uses its own query to mix information from the sequence.',
      },
    ],
    labs: [
      {
        id: 'attention-row',
        title: 'Read one attention row',
        prompt: 'Pick a token and explain which other tokens it attends to most.',
        successCriteria: 'You can connect one row of weights to the resulting context vector.',
      },
    ],
  },
  'attention-masks': {
    quiz: [
      {
        id: 'causal-purpose',
        prompt: 'Why does a decoder use a causal attention mask?',
        choices: [
          'To prevent a position from reading future tokens during next-token prediction',
          'To make every token attend only to padding',
          'To replace softmax with a sigmoid',
        ],
        answerIndex: 0,
        explanation: 'A causal mask keeps training and generation honest by hiding tokens that come after the current position.',
      },
      {
        id: 'padding-purpose',
        prompt: 'What should a padding mask do?',
        choices: [
          'Remove [PAD] positions from attention scores before softmax',
          'Increase the probability of every padding token',
          'Change the tokenizer vocabulary size',
        ],
        answerIndex: 0,
        explanation: 'Padding is sequence-shaping filler, not content, so its scores are replaced by a very negative value before softmax.',
      },
    ],
    labs: [
      {
        id: 'trace-visible-keys',
        title: 'Trace visible keys',
        prompt: 'Pick one query row, switch between mask types, and list which keys remain visible before softmax.',
        successCriteria: 'You can justify each visible or blocked key using causal order, padding, or cross-attention memory.',
      },
    ],
  },
  transformer: {
    quiz: [
      {
        id: 'block-stack',
        prompt: 'What does a transformer block combine?',
        choices: [
          'Attention, feed-forward layers, residual paths, and normalization',
          'Only a tokenizer and a confusion matrix',
          'Only convolution and pooling',
        ],
        answerIndex: 0,
        explanation: 'A transformer block repeatedly mixes token information, transforms it, and stabilizes the stream.',
      },
      {
        id: 'residual-purpose',
        prompt: 'Why are residual connections important in transformers?',
        choices: [
          'They preserve an information path across many transformations',
          'They replace the need for weights',
          'They force every token to attend only to itself',
        ],
        answerIndex: 0,
        explanation: 'Residual paths let layers add updates without destroying the existing representation.',
      },
    ],
    labs: [
      {
        id: 'trace-token',
        title: 'Trace one token',
        prompt: 'Follow one token through attention, feed-forward, residual, and normalization stages.',
        successCriteria: 'You can say what each stage changes and what it preserves.',
      },
    ],
  },
  'transformer-architecture-families': {
    quiz: [
      {
        id: 'bert-family',
        prompt: 'Which transformer family best describes BERT-style models?',
        choices: [
          'Encoder-only with bidirectional attention',
          'Decoder-only with causal attention',
          'Encoder-decoder with cross-attention only',
        ],
        answerIndex: 0,
        explanation: 'BERT-style models encode an input with bidirectional self-attention to produce contextual representations.',
      },
      {
        id: 'gpt-generation',
        prompt: 'Why are GPT-style models naturally used for left-to-right generation?',
        choices: [
          'Causal self-attention and next-token prediction match autoregressive decoding',
          'Bidirectional masks let every future token be visible during generation',
          'They do not use token probabilities',
        ],
        answerIndex: 0,
        explanation: 'Decoder-only models learn to predict the next token from the prefix, then append each sampled token.',
      },
    ],
    labs: [
      {
        id: 'choose-family',
        title: 'Choose the right family',
        prompt: 'Pick search reranking, chat completion, and translation tasks, then choose the architecture family for each.',
        successCriteria: 'You can justify each choice using attention visibility, objective, and output type.',
      },
    ],
  },
  'llm-training-objectives': {
    quiz: [
      {
        id: 'next-token-target',
        prompt: 'What does next-token prediction train a decoder-only model to do?',
        choices: [
          'Predict the next token from the previous prefix',
          'Read future tokens bidirectionally during generation',
          'Compare two completed answers without token probabilities',
        ],
        answerIndex: 0,
        explanation: 'Autoregressive pretraining predicts each next token from the prefix, which matches left-to-right generation.',
      },
      {
        id: 'preference-signal',
        prompt: 'What does preference optimization usually compare?',
        choices: [
          'A chosen response against a rejected response for the same prompt',
          'A validation fold against a test fold',
          'A masked token against every source token',
        ],
        answerIndex: 0,
        explanation: 'Preference training uses comparative feedback to make chosen responses more likely than rejected alternatives.',
      },
    ],
    labs: [
      {
        id: 'match-objective-stage',
        title: 'Match objective to behavior',
        prompt: 'Select each objective and decide whether it mainly teaches continuation, representation, instruction following, or preference-shaped response behavior.',
        successCriteria: 'You can justify each match using the target in the loss.',
      },
    ],
  },
  'transformer-token-generation': {
    quiz: [
      {
        id: 'one-token-at-a-time',
        prompt: 'What does an autoregressive transformer generate at each decoding step?',
        choices: [
          'One next token chosen from a probability distribution',
          'The entire answer as one fixed vector',
          'A new tokenizer vocabulary',
        ],
        answerIndex: 0,
        explanation: 'The selected token is appended to the context, then the model repeats the same next-token process.',
      },
      {
        id: 'temperature-filtering',
        prompt: 'What do temperature, top-k, and top-p control?',
        choices: [
          'Which next-token candidates remain likely or eligible before selection',
          'Whether the model has encoder layers',
          'How many training examples the model sees',
        ],
        answerIndex: 0,
        explanation: 'Temperature reshapes probabilities, while top-k and top-p filter the candidate set before sampling or greedy choice.',
      },
    ],
    labs: [
      {
        id: 'sampling-contrast',
        title: 'Compare two decoding settings',
        prompt: 'Generate a few steps with low temperature and greedy mode, then compare it with higher temperature and sampling.',
        successCriteria: 'You can explain which setting is more deterministic and which setting keeps more alternatives alive.',
      },
    ],
  },
  'sampling-strategies': {
    quiz: [
      {
        id: 'temperature-effect',
        prompt: 'What does a higher temperature usually do before sampling?',
        choices: [
          'Flattens the next-token distribution so weaker candidates receive more probability',
          'Changes the model weights to remember new facts',
          'Forces the model to always choose the top token',
        ],
        answerIndex: 0,
        explanation: 'Temperature rescales logits before softmax; higher values make the distribution less sharp.',
      },
      {
        id: 'top-p-meaning',
        prompt: 'How does top-p sampling choose its candidate set?',
        choices: [
          'It keeps the smallest ranked set whose cumulative probability reaches p',
          'It always keeps exactly p tokens',
          'It searches multiple full sequences and picks the best final score',
        ],
        answerIndex: 0,
        explanation: 'Top-p is nucleus sampling: the number of kept tokens changes with the probability mass in the current context.',
      },
    ],
    labs: [
      {
        id: 'decode-for-task',
        title: 'Choose decoding for a task',
        prompt: 'Pick settings for factual QA, brainstorming, and translation, then explain which one should be more deterministic or diverse.',
        successCriteria: 'You can justify each setting using candidate count, probability mass, and sequence-score tradeoffs.',
      },
    ],
  },
  'fine-tuning': {
    quiz: [
      {
        id: 'lora-trains-what',
        prompt: 'What does LoRA train during fine-tuning?',
        choices: [
          'Small low-rank adapter matrices while the base weights stay frozen',
          'Only the tokenizer vocabulary',
          'A retrieval index with no gradient updates',
        ],
        answerIndex: 0,
        explanation: 'LoRA keeps the pretrained weights fixed and learns low-rank update matrices that approximate the needed weight change.',
      },
      {
        id: 'preference-data-shape',
        prompt: 'What data shape is most natural for DPO-style preference tuning?',
        choices: [
          'A prompt with a chosen answer and a rejected answer',
          'Unlabeled rows for k-means clustering',
          'A single test-set metric with no examples',
        ],
        answerIndex: 0,
        explanation: 'Preference tuning compares candidate answers for the same prompt and increases the relative likelihood of the chosen one.',
      },
    ],
    labs: [
      {
        id: 'choose-finetune-method',
        title: 'Choose a fine-tuning method',
        prompt: 'Pick one scenario with limited GPU memory, one with demonstrations, and one with preference pairs, then choose LoRA, SFT, or DPO/RLHF.',
        successCriteria: 'Your choice matches the available data signal and the memory or behavior constraint.',
      },
    ],
  },
  'rag-chunking-context': {
    quiz: [
      {
        id: 'overlap-tradeoff',
        prompt: 'What is the main tradeoff of adding chunk overlap?',
        choices: [
          'It can preserve boundary facts but also creates duplicate context and index entries',
          'It removes the need for embeddings',
          'It guarantees every generated answer is correct',
        ],
        answerIndex: 0,
        explanation: 'Overlap helps when an answer crosses chunk boundaries, but repeated text can consume retrieval slots and context budget.',
      },
      {
        id: 'context-packing-budget',
        prompt: 'Why can a high top-k still fail to help a RAG answer?',
        choices: [
          'Retrieved chunks may not fit inside the remaining context budget',
          'Top-k changes the model weights during generation',
          'The tokenizer stops splitting text into tokens',
        ],
        answerIndex: 0,
        explanation: 'Retrieval and context packing are separate steps; chunks returned by retrieval can be dropped when the context budget is full.',
      },
    ],
    labs: [
      {
        id: 'tune-chunk-budget',
        title: 'Tune chunking and packing',
        prompt: 'Adjust chunk size, overlap, top-k, and context budget until both relevant refund facts fit without too much duplicate text.',
        successCriteria: 'You can explain which control recovered the boundary fact and which control limited packed evidence.',
      },
    ],
  },
  'rag-vector-indexing': {
    quiz: [
      {
        id: 'ann-tradeoff',
        prompt: 'What is the core tradeoff in approximate nearest neighbor vector search?',
        choices: [
          'Lower latency in exchange for possible recall loss',
          'No need to create embeddings',
          'Guaranteed perfect reranking',
        ],
        answerIndex: 0,
        explanation: 'ANN methods avoid comparing every vector, which reduces latency but can miss some nearest neighbors.',
      },
      {
        id: 'search-breadth',
        prompt: 'What usually happens when search breadth increases in an ANN index?',
        choices: [
          'Recall tends to improve and latency tends to increase',
          'The model weights are fine-tuned',
          'Chunk overlap becomes zero',
        ],
        answerIndex: 0,
        explanation: 'Broader search probes more candidates or graph paths, so it can find more relevant vectors at higher cost.',
      },
    ],
    labs: [
      {
        id: 'choose-index',
        title: 'Choose an index strategy',
        prompt: 'Compare exact, IVF, and HNSW modes for a small corpus, a large corpus, and a high-recall support bot.',
        successCriteria: 'You can justify the choice using latency, recall risk, and corpus scale.',
      },
    ],
  },
  'rag-retrieval-evaluation': {
    quiz: [
      {
        id: 'missing-evidence',
        prompt: 'What happens if the relevant chunk never appears in the top-k retrieval set?',
        choices: [
          'The generator has no grounded evidence to cite',
          'The reranker can always recreate the missing chunk',
          'nDCG automatically becomes perfect',
        ],
        answerIndex: 0,
        explanation: 'Reranking can reorder candidates, but it cannot use evidence that first-pass retrieval failed to return.',
      },
      {
        id: 'metric-purpose',
        prompt: 'Which metric checks whether relevant evidence was recovered somewhere in the top-k set?',
        choices: [
          'Recall@k',
          'Learning rate',
          'Cross-entropy',
        ],
        answerIndex: 0,
        explanation: 'Recall@k measures how much known relevant evidence appears within the retrieved candidate set.',
      },
    ],
    labs: [
      {
        id: 'chunking-rerank-audit',
        title: 'Audit a retrieval setting',
        prompt: 'Change chunk size, overlap, top-k, and reranking, then pick the setting with the clearest grounded evidence.',
        successCriteria: 'You can justify the setting with recall@k, MRR, nDCG, and the text of the top result.',
      },
    ],
  },
  'rag-reranking-grounding': {
    quiz: [
      {
        id: 'rerank-function',
        prompt: 'Which statement best describes reranking?',
        choices: [
          'It changes the ordering of already-retrieved candidates before generation',
          'It generates completely new chunks that were never retrieved',
          'It replaces the split of train/validation/test',
        ],
        answerIndex: 0,
        explanation: 'A reranker re-scores candidates from the retrieval set; it cannot retrieve evidence that was never there.',
      },
      {
        id: 'grounding-rule',
        prompt: 'Why can a strict grounding rule drop a claim even when the claim appears in top-k?',
        choices: [
          'Because strict grounding may reject stale, conflicting, or low-trust evidence',
          'Because the reranker is too shallow',
          'Because cosine similarity is always wrong',
        ],
        answerIndex: 0,
        explanation: 'Grounding checks evidence quality, recency, and consistency before accepting citations.',
      },
    ],
    labs: [
      {
        id: 'grounding-audit',
        title: 'Audit grounding behavior',
        prompt: 'Use strictness and top-k to make a stale conflict visible, then make every claim grounded with usable evidence.',
        successCriteria: 'You can explain why strictness blocked one claim and why adjusting top-k changed grounded coverage.',
      },
    ],
  },
  'rag-failure-modes': {
    quiz: [
      {
        id: 'dominant-failure',
        prompt: 'Which failure should you diagnose first when top-k is full of unrelated chunks?',
        choices: [
          'Irrelevant retrieval evidence entered the candidate set',
          'The candidate scoring function is too selective',
          'The answer head has too few parameters',
        ],
        answerIndex: 0,
        explanation: 'When candidate quality is noisy, first improve retrieval and candidate generation, not decoding or reranker settings.',
      },
      {
        id: 'grounding-before-answer',
        prompt: 'What is the biggest risk if strictness is very high but top-k still includes no usable evidence?',
        choices: [
          'Grounded claims fail even if the model sounds fluent',
          'The model learns a better token distribution',
          'Recall metrics automatically improve',
        ],
        answerIndex: 0,
        explanation: 'Fluency is not equivalent to groundedness; without usable evidence, output quality is not evidence-based.',
      },
    ],
    labs: [
      {
        id: 'failure-tune',
        title: 'Tune one failure',
        prompt: 'Use reranker mode, top-k, and strictness to reduce stale or conflicting behavior for at least two claims.',
        successCriteria: 'You can report the top-k/strictness region with fewer stale/conflicting tags and a higher grounded count.',
      },
    ],
  },
  'mdp-formalism': {
    quiz: [
      {
        id: 'mdp-parts',
        prompt: 'What does the transition model P describe in an MDP?',
        choices: [
          'The probability distribution over next states after taking an action',
          'The list of all parameters in a neural network',
          'The final answer chosen by a language model',
        ],
        answerIndex: 0,
        explanation: 'P describes environment dynamics: from a state and action, it gives probabilities for possible next states.',
      },
      {
        id: 'discount-role',
        prompt: 'What changes when gamma is reduced toward zero?',
        choices: [
          'Immediate rewards dominate delayed rewards',
          'Future rewards become more important than immediate rewards',
          'Transition probabilities stop summing to one',
        ],
        answerIndex: 0,
        explanation: 'A smaller discount factor puts less weight on future rewards, so near-term reward matters more.',
      },
    ],
    labs: [
      {
        id: 'gamma-comparison',
        title: 'Compare action values',
        prompt: 'Switch between actions and lower gamma until the safer immediate reward becomes more attractive.',
        successCriteria: 'You can explain how the same transition probabilities lead to different action choices when gamma changes.',
      },
    ],
  },
  'value-iteration': {
    quiz: [
      {
        id: 'bellman-backup',
        prompt: 'What does a Bellman optimality backup do in value iteration?',
        choices: [
          'It compares action lookaheads and keeps the highest expected return',
          'It samples one action and permanently stores that single result',
          'It tokenizes rewards into subword units',
        ],
        answerIndex: 0,
        explanation: 'Value iteration uses the known model to compute expected return for each action, then takes the maximum.',
      },
      {
        id: 'model-vs-samples',
        prompt: 'Why is value iteration considered planning rather than model-free learning?',
        choices: [
          'It uses known transitions and rewards to update values before acting',
          'It ignores transition probabilities',
          'It only works when there are no rewards',
        ],
        answerIndex: 0,
        explanation: 'Planning methods use a model of the environment; Q-learning can learn from sampled experience without that full model.',
      },
    ],
    labs: [
      {
        id: 'sweep-propagation',
        title: 'Watch reward propagation',
        prompt: 'Set sweeps to zero, then increase one step at a time and observe which states change after the goal value appears.',
        successCriteria: 'You can identify how terminal rewards propagate backward through Bellman sweeps.',
      },
    ],
  },
  'policy-iteration': {
    quiz: [
      {
        id: 'two-phases',
        prompt: 'What are the two repeating phases of policy iteration?',
        choices: [
          'Evaluate the current policy, then greedily improve it',
          'Tokenize the state, then sample a reward',
          'Split train and test data, then fit a classifier',
        ],
        answerIndex: 0,
        explanation: 'Policy iteration alternates policy evaluation with policy improvement until the greedy policy stops changing.',
      },
      {
        id: 'stable-policy',
        prompt: 'What does it mean when the improved policy matches the current policy?',
        choices: [
          'The policy is stable under the current value estimates',
          'The transition model has been deleted',
          'Rewards no longer affect decisions',
        ],
        answerIndex: 0,
        explanation: 'If greedy improvement no longer changes any state action, policy iteration has reached a stable policy for the model.',
      },
    ],
    labs: [
      {
        id: 'policy-flip',
        title: 'Find a policy flip',
        prompt: 'Increase improvement rounds and identify the first state whose action changes from the initial policy.',
        successCriteria: 'You can name the state, the old action, and the improved greedy action.',
      },
    ],
  },
  'policy-gradients': {
    quiz: [
      {
        id: 'positive-advantage',
        prompt: 'What happens to the sampled action when its advantage is positive?',
        choices: [
          'Its log-probability is pushed upward',
          'Its probability is forced to zero',
          'The transition model is recomputed exactly',
        ],
        answerIndex: 0,
        explanation: 'Policy gradients reinforce sampled actions that beat the baseline by increasing their log-probability.',
      },
      {
        id: 'policy-object',
        prompt: 'What is optimized directly in policy-gradient methods?',
        choices: [
          'The stochastic policy action probabilities',
          'Only a fixed Q-table',
          'Only the train/test split',
        ],
        answerIndex: 0,
        explanation: 'Unlike tabular value methods, policy gradients update policy parameters that control action probabilities.',
      },
    ],
    labs: [
      {
        id: 'advantage-flip',
        title: 'Flip the advantage',
        prompt: 'Move return below and above the baseline and watch the sampled action probability move down or up.',
        successCriteria: 'You can explain how the sign of advantage changes the policy update direction.',
      },
    ],
  },
  'actor-critic': {
    quiz: [
      {
        id: 'actor-vs-critic',
        prompt: 'What does the critic provide to the actor?',
        choices: [
          'A value baseline used to compute advantage',
          'The final sampled action with no policy',
          'A train/test split for supervised learning',
        ],
        answerIndex: 0,
        explanation: 'The critic estimates value; the actor uses return minus that value as an advantage signal.',
      },
      {
        id: 'negative-advantage',
        prompt: 'If return is below the critic value, what should happen to the sampled action?',
        choices: [
          'The actor should reduce its log-probability',
          'The critic should choose the action directly',
          'The action must become deterministic',
        ],
        answerIndex: 0,
        explanation: 'A negative advantage means the action underperformed the critic baseline, so the actor is pushed away from it.',
      },
    ],
    labs: [
      {
        id: 'critic-baseline',
        title: 'Find the sign flip',
        prompt: 'Move critic value above and below return and watch the actor update switch from reinforcing to discouraging.',
        successCriteria: 'You can explain why the critic changes update sign without choosing actions itself.',
      },
    ],
  },
  'reward-shaping': {
    quiz: [
      {
        id: 'sparse-vs-shaped',
        prompt: 'Why add a shaping reward to a sparse-reward task?',
        choices: [
          'To provide denser learning feedback while the real goal remains sparse',
          'To replace the environment objective with an easier one',
          'To remove the need for discounting or value estimates',
        ],
        answerIndex: 0,
        explanation: 'Shaping gives the agent intermediate feedback, but the task reward should still define success.',
      },
      {
        id: 'potential-based-safety',
        prompt: 'What is the main reason to prefer potential-based shaping?',
        choices: [
          'It can guide progress without changing which policy is optimal',
          'It always makes rewards larger than zero',
          'It turns policy gradients into supervised learning',
        ],
        answerIndex: 0,
        explanation: 'Potential-based shaping rewards progress between states and is designed to preserve the intended optimum.',
      },
    ],
    labs: [
      {
        id: 'sparse-to-dense',
        title: 'Turn sparse reward into a hint',
        prompt: 'Move the next state toward and away from the goal and compare task reward, shaping bonus, and total signal.',
        successCriteria: 'You can identify when shaping helps exploration and when it would risk changing the objective.',
      },
    ],
  },
  'diffusion-basics': {
    quiz: [
      {
        id: 'predict-noise',
        prompt: 'In a basic noise-prediction diffusion lesson, what does the model learn to predict?',
        choices: [
          'The noise added to the clean sample at a timestep',
          'Only the final image caption',
          'A train/test split for the dataset',
        ],
        answerIndex: 0,
        explanation: 'The common beginner formulation trains the model to estimate the noise so it can be removed.',
      },
      {
        id: 'many-steps',
        prompt: 'Why is diffusion usually described as an iterative denoising process?',
        choices: [
          'Generation repeatedly removes noise across timesteps',
          'The model only runs once at t = 0',
          'Noise is added only after the image is complete',
        ],
        answerIndex: 0,
        explanation: 'A diffusion sampler walks from noisy latents toward cleaner latents over multiple denoising steps.',
      },
    ],
    labs: [
      {
        id: 'noise-prediction-error',
        title: 'Tune the noise prediction',
        prompt: 'Move timestep and prediction error to see how noisy sample and denoised estimate diverge from the clean signal.',
        successCriteria: 'You can explain why a better noise estimate produces a cleaner recovered sample.',
      },
    ],
  },
  'diffusion-sampling': {
    quiz: [
      {
        id: 'ddpm-vs-ddim',
        prompt: 'What is the key difference between DDPM and DDIM sampling in this beginner comparison?',
        choices: [
          'DDPM keeps stochastic reverse noise, while DDIM can follow a deterministic path',
          'DDIM trains a classifier, while DDPM trains a tokenizer',
          'DDPM only works for text and DDIM only works for images',
        ],
        answerIndex: 0,
        explanation: 'Both use the denoising model, but DDIM can remove the extra sampling randomness that DDPM retains.',
      },
      {
        id: 'step-tradeoff',
        prompt: 'What is the main risk when reducing the number of reverse sampling steps?',
        choices: [
          'Each step must do more denoising work, so prediction errors matter more',
          'The forward noising process becomes impossible to define',
          'The model stops using latents and switches to supervised labels',
        ],
        answerIndex: 0,
        explanation: 'Fewer steps can be faster, but each update covers more distance and can amplify denoising mistakes.',
      },
    ],
    labs: [
      {
        id: 'compare-samplers',
        title: 'Compare sampler paths',
        prompt: 'Switch between DDPM, DDIM, and flow/ODE while changing step count and prediction quality.',
        successCriteria: 'You can describe how stochasticity, step count, and prediction quality change the path from noise to sample.',
      },
    ],
  },
  'classifier-free-guidance': {
    quiz: [
      {
        id: 'cfg-combination',
        prompt: 'What two predictions does classifier-free guidance combine?',
        choices: [
          'A prompt-conditioned prediction and an unconditional prediction',
          'A train-set prediction and a test-set prediction',
          'A classifier label and a retrieval score',
        ],
        answerIndex: 0,
        explanation: 'CFG amplifies the direction from unconditional denoising toward prompt-conditioned denoising.',
      },
      {
        id: 'scale-tradeoff',
        prompt: 'What is the main tradeoff when increasing guidance scale too far?',
        choices: [
          'Prompt adherence can improve, but diversity and visual quality can suffer',
          'The model stops using text conditioning entirely',
          'The sampler must switch from images to tabular data',
        ],
        answerIndex: 0,
        explanation: 'High scale can force prompt features harder, but it can also overshoot and create artifacts.',
      },
    ],
    labs: [
      {
        id: 'guidance-scale-sweep',
        title: 'Sweep guidance scale',
        prompt: 'Move guidance scale from low to high and compare prompt match, diversity, and artifact risk.',
        successCriteria: 'You can explain why a moderate scale can be better than the maximum scale.',
      },
    ],
  },
  'unet-vs-dit': {
    quiz: [
      {
        id: 'unet-bias',
        prompt: 'What image-processing bias does a diffusion U-Net provide naturally?',
        choices: [
          'Local convolutional structure with downsample-upsample skip connections',
          'A next-token language-modeling objective',
          'A retrieval index over text chunks',
        ],
        answerIndex: 0,
        explanation: 'U-Nets are strong for images because convolutions and skip connections preserve local structure across scales.',
      },
      {
        id: 'dit-patches',
        prompt: 'Why do DiT-style models split latents into patches?',
        choices: [
          'So the image latent can be processed as a transformer token sequence',
          'So the model can avoid attention entirely',
          'So the sampler no longer needs denoising steps',
        ],
        answerIndex: 0,
        explanation: 'A DiT treats image or latent patches like tokens, which lets transformer attention mix global information.',
      },
    ],
    labs: [
      {
        id: 'patch-cost',
        title: 'Inspect patch-token cost',
        prompt: 'Change resolution and patch size and watch token count and attention-pair cost move.',
        successCriteria: 'You can explain why smaller patches improve detail but increase transformer attention cost.',
      },
    ],
  },
};

export function getLessonAssessment(lessonId) {
  return lessonAssessments[lessonId] || EMPTY_ASSESSMENT;
}

export function hasAssessmentContent(assessment) {
  return Boolean(assessment?.quiz?.length || assessment?.labs?.length);
}

export function getAssessmentStats(assessments = lessonAssessments) {
  return Object.values(assessments).reduce(
    (stats, assessment) => ({
      totalQuizQuestions: stats.totalQuizQuestions + (assessment.quiz?.length || 0),
      totalLabs: stats.totalLabs + (assessment.labs?.length || 0),
    }),
    { totalQuizQuestions: 0, totalLabs: 0 },
  );
}
