export const PRIORITY_ASSESSMENT_LESSON_IDS = [
  'matrix-multiplication',
  'linear-regression',
  'pca',
  'train-validation-test-split',
  'cross-validation',
  'logistic-regression',
  'classification-metrics',
  'overfitting',
  'regularization',
  'gradient-descent',
  'relu',
  'computation-graph-backprop',
  'tokenization',
  'embeddings',
  'attention-mechanism',
  'self-attention',
  'transformer',
  'transformer-token-generation',
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
