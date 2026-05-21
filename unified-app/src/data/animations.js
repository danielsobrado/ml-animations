import {
  Brain,
  Calculator,
  Layers,
  GitBranch,
  Sparkles,
  Network,
  Binary,
  LineChart,
  Box,
  Shuffle,
  Target,
  Lightbulb,
  BookOpen,
  Atom,
  Activity,
  Grid3X3,
  Cpu,
  Zap,
  TrendingUp,
  BarChart3,
  Settings,
  FileText,
  Hash,
  Boxes,
  CircleDot,
  Workflow,
  Database,
  Dice1,
  Sigma,
  ArrowRightLeft,
  Image,
  MessageSquare,
  TrendingDown,
  Maximize,
  Users,
  RotateCcw,
  GitMerge,
  ShieldCheck
} from 'lucide-react';

// Animation categories and their items
export const categories = [
  {
    id: 'nlp',
    name: 'Natural Language Processing',
    icon: MessageSquare,
    color: 'from-blue-500 to-cyan-500',
    items: [
      { id: 'bag-of-words', name: 'Bag of Words', icon: FileText, description: 'Text representation using word frequencies' },
      { id: 'word2vec', name: 'Word2Vec', icon: Network, description: 'Word embeddings using neural networks' },
      { id: 'glove', name: 'GloVe', icon: Sparkles, description: 'Global vectors for word representation' },
      { id: 'fasttext', name: 'FastText', icon: Zap, description: 'Subword-based word embeddings' },
      { id: 'tokenization', name: 'Tokenization', icon: Hash, description: 'Breaking text into tokens' },
      { id: 'embeddings', name: 'Embeddings', icon: Boxes, description: 'Dense vector representations' },
    ],
  },
  {
    id: 'transformers',
    name: 'Transformers & Attention',
    icon: Brain,
    color: 'from-purple-500 to-pink-500',
    items: [
      { id: 'attention-mechanism', name: 'Attention Mechanism', icon: Target, description: 'The foundation of modern NLP' },
      { id: 'self-attention', name: 'Self-Attention', icon: CircleDot, description: 'Relating positions in a sequence' },
      { id: 'positional-encoding', name: 'Positional Encoding', icon: Workflow, description: 'Adding position information' },
      { id: 'rope', name: 'RoPE (Rotary Embeddings)', icon: RotateCcw, description: 'Modern position encoding via rotation' },
      { id: 'transformer', name: 'Transformer Architecture', icon: Cpu, description: 'Complete transformer model' },
      { id: 'grouped-query-attention', name: 'Grouped-Query Attention', icon: Users, description: 'Efficient attention with grouped queries' },
      { id: 'kv-cache', name: 'KV Cache', icon: Database, description: 'Caching keys and values for fast inference' },
      { id: 'flash-attention', name: 'Flash Attention', icon: Zap, description: 'Hardware-aware tiled attention for efficiency' },
      { id: 'residual-stream', name: 'Residual Stream', icon: GitMerge, description: 'How information flows and accumulates through layers' },
      { id: 'bert', name: 'BERT', icon: BookOpen, description: 'Bidirectional encoder representations' },
      { id: 'gpt2-comprehensive', name: 'GPT-2 Comprehensive', icon: MessageSquare, description: 'Detailed look at GPT-2 architecture' },
      { id: 'moe', name: 'Mixture of Experts', icon: GitBranch, description: 'Scaling models with conditional computation' },
      { id: 'fine-tuning', name: 'Fine-Tuning', icon: Settings, description: 'Adapting pretrained models' },
    ],
  },
  {
    id: 'neural-networks',
    name: 'Neural Networks',
    icon: Network,
    color: 'from-green-500 to-emerald-500',
    items: [
      { id: 'relu', name: 'ReLU Activation', icon: Zap, description: 'Rectified linear unit function' },
      { id: 'leaky-relu', name: 'Leaky ReLU', icon: Activity, description: 'ReLU with small negative slope' },
      { id: 'softmax', name: 'Softmax', icon: BarChart3, description: 'Probability distribution output' },
      { id: 'layer-normalization', name: 'Layer Normalization', icon: Layers, description: 'Normalizing layer activations' },
      { id: 'lstm', name: 'LSTM', icon: GitBranch, description: 'Long short-term memory networks' },
      { id: 'conv2d', name: 'Conv2D', icon: Grid3X3, description: '2D convolutional layers' },
      { id: 'max-pooling', name: 'Max Pooling', icon: Maximize, description: 'Downsampling in convolutional networks' },
      { id: 'conv-relu', name: 'Conv + ReLU', icon: Box, description: 'Convolution with activation' },
      { id: 'neural-network', name: 'Neural Network Overview', icon: Network, description: 'How artificial neural networks function' },
      { id: 'computation-graph-backprop', name: 'Computation Graph & Backpropagation', icon: Workflow, description: 'Forward values, local derivatives, reverse accumulation, and updates' },
      { id: 'gradient-problems', name: 'Gradient Problems', icon: TrendingDown, description: 'Vanishing and exploding gradients' },
    ],
  },
  {
    id: 'advanced-models',
    name: 'Advanced Models',
    icon: Atom,
    color: 'from-orange-500 to-red-500',
    items: [
      { id: 'vae', name: 'Variational Autoencoder', icon: Shuffle, description: 'Generative latent variable model' },
      { id: 'rag', name: 'RAG', icon: Database, description: 'Retrieval-augmented generation' },
      { id: 'multimodal-llm', name: 'Multimodal LLM', icon: Image, description: 'Multi-modal language models' },
    ],
  },
  {
    id: 'math-fundamentals',
    name: 'Math Fundamentals',
    icon: Calculator,
    color: 'from-indigo-500 to-violet-500',
    items: [
      { id: 'matrix-multiplication', name: 'Matrix Multiplication', icon: Grid3X3, description: 'Core linear algebra operation' },
      { id: 'matrix-decompositions', name: 'Matrix Decompositions', icon: Workflow, description: 'One-sheet guide to LU, QR, SVD, eigen, and more' },
      { id: 'fundamental-subspaces', name: 'Fundamental Subspaces', icon: Calculator, description: 'Row, null, column, and left-null spaces' },
      { id: 'least-squares-projection', name: 'Least Squares Projection', icon: Target, description: 'Projection, residuals, and normal equations' },
      { id: 'pseudoinverse', name: 'Pseudoinverse', icon: ArrowRightLeft, description: 'SVD-based inverse for rectangular systems' },
      { id: 'change-of-basis', name: 'Change of Basis', icon: Workflow, description: 'Coordinates under different bases' },
      { id: 'condition-number', name: 'Condition Number', icon: Activity, description: 'Sensitivity, stretching, and near-null directions' },
      { id: 'determinant-volume', name: 'Determinant as Volume', icon: Box, description: 'Area and volume scaling by matrices' },
      { id: 'projection-matrices', name: 'Projection Matrices', icon: Target, description: 'Idempotent maps onto subspaces' },
      { id: 'low-rank-approximation', name: 'Low-Rank Approximation', icon: Layers, description: 'Truncated SVD and reconstruction error' },
      { id: 'eigenvalue', name: 'Eigenvalues', icon: Sigma, description: 'Matrix eigendecomposition' },
      { id: 'svd', name: 'SVD', icon: ArrowRightLeft, description: 'Singular value decomposition' },
      { id: 'qr-decomposition', name: 'QR Decomposition', icon: Boxes, description: 'Orthogonal matrix factorization' },
      { id: 'gradient-descent', name: 'Gradient Descent', icon: TrendingUp, description: 'Optimization algorithm' },
      { id: 'optimization', name: 'Optimization Landscape', icon: Target, description: 'Exploring loss surfaces and minima' },
      { id: 'linear-regression', name: 'Linear Regression', icon: LineChart, description: 'Fitting linear models' },
    ],
  },
  {
    id: 'core-ml',
    name: 'Core ML',
    icon: Target,
    color: 'from-cyan-600 to-blue-600',
    items: [
      { id: 'train-validation-test-split', name: 'Train / Validation / Test Split', icon: Shuffle, description: 'Separating data for learning, tuning, and honest evaluation' },
      { id: 'overfitting', name: 'Overfitting', icon: TrendingDown, description: 'When a model memorizes training quirks instead of learning the pattern' },
      { id: 'logistic-regression', name: 'Logistic Regression', icon: CircleDot, description: 'Linear classification through sigmoid probabilities' },
      { id: 'classification-metrics', name: 'Classification Metrics', icon: BarChart3, description: 'Confusion matrix and threshold tradeoffs' },
      { id: 'regularization', name: 'Regularization', icon: ShieldCheck, description: 'Penalizing complexity so models generalize better' },
    ],
  },
  {
    id: 'probability-stats',
    name: 'Probability & Statistics',
    icon: Dice1,
    color: 'from-teal-500 to-cyan-500',
    items: [
      { id: 'probability-distributions', name: 'Probability Distributions', icon: BarChart3, description: 'Common probability distributions' },
      { id: 'conditional-probability', name: 'Conditional Probability', icon: GitBranch, description: 'P(A|B) and Bayes theorem' },
      { id: 'expected-value-variance', name: 'Expected Value & Variance', icon: Calculator, description: 'Statistical moments' },
      { id: 'entropy', name: 'Entropy', icon: Activity, description: 'Information entropy measure' },
      { id: 'cross-entropy', name: 'Cross-Entropy', icon: Shuffle, description: 'Loss function for classification' },
      { id: 'cosine-similarity', name: 'Cosine Similarity', icon: Target, description: 'Vector similarity measure' },
      { id: 'spearman-correlation', name: 'Spearman Correlation', icon: TrendingUp, description: 'Rank-based correlation' },
    ],
  },
  {
    id: 'reinforcement-learning',
    name: 'Reinforcement Learning',
    icon: Lightbulb,
    color: 'from-amber-500 to-orange-500',
    items: [
      { id: 'rl-foundations', name: 'RL Foundations', icon: BookOpen, description: 'Basic RL concepts' },
      { id: 'q-learning', name: 'Q-Learning', icon: Brain, description: 'Value-based learning algorithm' },
      { id: 'rl-exploration', name: 'Exploration vs Exploitation', icon: Shuffle, description: 'The exploration-exploitation tradeoff' },
      { id: 'markov-chains', name: 'Markov Chains', icon: Workflow, description: 'State transition models' },
    ],
  },
  {
    id: 'algorithms',
    name: 'Algorithms & Data Structures',
    icon: Binary,
    color: 'from-rose-500 to-pink-500',
    items: [
      { id: 'bloom-filter', name: 'Bloom Filter', icon: Hash, description: 'Probabilistic data structure' },
      { id: 'pagerank', name: 'PageRank', icon: Network, description: 'Graph ranking algorithm' },
    ],
  },
  {
    id: 'diffusion-models',
    name: 'Diffusion Models (SD3)',
    icon: Sparkles,
    color: 'from-fuchsia-500 to-purple-600',
    items: [
      { id: 'sd3-overview', name: 'SD3 Architecture Overview', icon: Cpu, description: 'Complete Stable Diffusion 3 pipeline' },
      { id: 'flow-matching', name: 'Flow Matching', icon: Workflow, description: 'Flow-based generative modeling' },
      { id: 'diffusion-vae', name: 'VAE for Diffusion', icon: Shuffle, description: 'Latent space encoding for images' },
      { id: 'tokenizer-bpe', name: 'BPE & Unigram Tokenizers', icon: Hash, description: 'Text tokenization methods' },
      { id: 'clip-encoder', name: 'CLIP Text Encoder', icon: Target, description: 'Contrastive language-image pretraining' },
      { id: 't5-encoder', name: 'T5 Text Encoder', icon: FileText, description: 'Text-to-text transfer transformer' },
      { id: 'joint-attention', name: 'Joint Attention', icon: CircleDot, description: 'Multi-modal attention mechanism' },
      { id: 'dit', name: 'DiT (Diffusion Transformer)', icon: Brain, description: 'Transformer-based diffusion backbone' },
    ],
  },
];

export const curriculumTracks = [
  {
    id: 'foundations',
    title: 'Foundations',
    description: 'Math and probability concepts learners need before model training feels concrete.',
    animationIds: [
      'matrix-multiplication',
      'matrix-decompositions',
      'fundamental-subspaces',
      'least-squares-projection',
      'pseudoinverse',
      'change-of-basis',
      'condition-number',
      'determinant-volume',
      'projection-matrices',
      'low-rank-approximation',
      'eigenvalue',
      'svd',
      'qr-decomposition',
      'probability-distributions',
      'conditional-probability',
      'expected-value-variance',
      'entropy',
      'spearman-correlation',
      'gradient-descent',
      'linear-regression',
    ],
  },
  {
    id: 'core-ml',
    title: 'Core ML',
    description: 'The supervised-learning loop: prediction, loss, optimization, and evaluation concepts.',
    animationIds: [
      'linear-regression',
      'train-validation-test-split',
      'logistic-regression',
      'classification-metrics',
      'overfitting',
      'regularization',
      'cross-entropy',
      'softmax',
      'gradient-descent',
      'optimization',
      'expected-value-variance',
      'entropy',
    ],
  },
  {
    id: 'neural-networks',
    title: 'Neural Networks',
    description: 'From neurons and activations to convolution, normalization, memory, and gradient stability.',
    animationIds: [
      'neural-network',
      'relu',
      'leaky-relu',
      'softmax',
      'cross-entropy',
      'computation-graph-backprop',
      'gradient-problems',
      'layer-normalization',
      'conv2d',
      'conv-relu',
      'max-pooling',
      'lstm',
    ],
  },
  {
    id: 'nlp-transformers',
    title: 'NLP To Transformers',
    description: 'Represent text as vectors, then build up attention, transformer blocks, and inference tools.',
    animationIds: [
      'tokenization',
      'bag-of-words',
      'embeddings',
      'cosine-similarity',
      'word2vec',
      'glove',
      'fasttext',
      'attention-mechanism',
      'self-attention',
      'positional-encoding',
      'transformer',
      'bert',
      'gpt2-comprehensive',
      'rope',
      'residual-stream',
      'grouped-query-attention',
      'kv-cache',
      'flash-attention',
      'fine-tuning',
      'moe',
    ],
  },
  {
    id: 'generative-ai',
    title: 'Generative AI',
    description: 'Latent-variable models, diffusion-era conditioning, multimodal attention, and RAG.',
    animationIds: [
      'vae',
      'diffusion-vae',
      'sd3-overview',
      'flow-matching',
      'tokenizer-bpe',
      'clip-encoder',
      't5-encoder',
      'joint-attention',
      'dit',
      'rag',
      'multimodal-llm',
      'fine-tuning',
      'moe',
    ],
  },
  {
    id: 'rl-algorithms',
    title: 'RL And Algorithms',
    description: 'State transitions, value learning, exploration, graph ranking, and probabilistic structures.',
    animationIds: [
      'markov-chains',
      'rl-foundations',
      'q-learning',
      'rl-exploration',
      'pagerank',
      'bloom-filter',
    ],
  },
];

export const curriculumBacklog = [
  { id: 'cross-validation', title: 'Cross-Validation & Data Leakage', trackId: 'core-ml' },
  { id: 'pca', title: 'PCA', trackId: 'foundations' },
  { id: 'k-means', title: 'K-Means Clustering', trackId: 'core-ml' },
  { id: 'knn-naive-bayes-svm', title: 'kNN, Naive Bayes, and SVM', trackId: 'core-ml' },
  { id: 'tree-ensembles', title: 'Decision Trees, Random Forests, and Gradient Boosting', trackId: 'core-ml' },
  { id: 'transformer-token-generation', title: 'Transformer Token Generation Loop', trackId: 'nlp-transformers' },
  { id: 'rag-retrieval-evaluation', title: 'Chunking, Reranking, and Retrieval Evaluation', trackId: 'generative-ai' },
];

const trackIdsByAnimation = curriculumTracks.reduce((acc, track) => {
  for (const animationId of track.animationIds) {
    acc[animationId] = [...(acc[animationId] || []), track.id];
  }
  return acc;
}, {});

const CATEGORY_CURRICULUM_DEFAULTS = {
  nlp: { difficulty: 'beginner', estimatedMinutes: 12, prerequisites: [] },
  transformers: { difficulty: 'advanced', estimatedMinutes: 22, prerequisites: ['embeddings', 'softmax'] },
  'neural-networks': { difficulty: 'intermediate', estimatedMinutes: 16, prerequisites: ['linear-regression', 'gradient-descent'] },
  'advanced-models': { difficulty: 'advanced', estimatedMinutes: 24, prerequisites: ['embeddings', 'probability-distributions'] },
  'math-fundamentals': { difficulty: 'intermediate', estimatedMinutes: 15, prerequisites: [] },
  'core-ml': { difficulty: 'beginner', estimatedMinutes: 16, prerequisites: ['linear-regression'] },
  'probability-stats': { difficulty: 'intermediate', estimatedMinutes: 14, prerequisites: ['probability-distributions'] },
  'reinforcement-learning': { difficulty: 'intermediate', estimatedMinutes: 18, prerequisites: ['expected-value-variance'] },
  algorithms: { difficulty: 'intermediate', estimatedMinutes: 14, prerequisites: ['matrix-multiplication'] },
  'diffusion-models': { difficulty: 'advanced', estimatedMinutes: 25, prerequisites: ['vae', 'self-attention'] },
};

const CURRICULUM_OVERRIDES = {
  'matrix-multiplication': {
    difficulty: 'beginner',
    estimatedMinutes: 14,
    prerequisites: [],
    learningObjectives: [
      'Compute one matrix product entry as a row-column dot product',
      'Explain why matrix multiplication composes linear transformations',
    ],
    commonMisconception: 'Matrix multiplication is not elementwise multiplication; each output entry combines a row with a column.',
  },
  softmax: {
    difficulty: 'beginner',
    estimatedMinutes: 12,
    prerequisites: ['expected-value-variance'],
    learningObjectives: [
      'Convert logits into a normalized probability distribution',
      'Predict how changing one logit affects every output probability',
    ],
    commonMisconception: 'Softmax does not simply rescale scores independently; every probability depends on every logit.',
  },
  'cross-entropy': {
    difficulty: 'intermediate',
    estimatedMinutes: 15,
    prerequisites: ['softmax', 'entropy'],
  },
  'gradient-descent': {
    difficulty: 'beginner',
    estimatedMinutes: 18,
    prerequisites: ['linear-regression'],
  },
  'linear-regression': {
    difficulty: 'beginner',
    estimatedMinutes: 18,
    prerequisites: ['matrix-multiplication'],
  },
  'train-validation-test-split': {
    difficulty: 'beginner',
    estimatedMinutes: 15,
    prerequisites: ['linear-regression'],
    learningObjectives: [
      'Explain why training, validation, and test data must stay separate',
      'Choose split sizes that preserve enough examples for learning and evaluation',
      'Identify how repeated test-set tuning creates data leakage',
    ],
    commonMisconception: 'A high test score is not trustworthy if the test set influenced model choices during development.',
  },
  'logistic-regression': {
    difficulty: 'beginner',
    estimatedMinutes: 18,
    prerequisites: ['linear-regression'],
    learningObjectives: [
      'Map a linear score through the sigmoid function to get a probability',
      'Move a classification threshold and predict how labels change',
      'Connect logistic regression to binary cross-entropy training',
    ],
    commonMisconception: 'Logistic regression is a classification model; despite the name, its output is a probability for a class.',
  },
  'classification-metrics': {
    difficulty: 'beginner',
    estimatedMinutes: 20,
    prerequisites: ['logistic-regression'],
    learningObjectives: [
      'Read true positives, false positives, false negatives, and true negatives from a confusion matrix',
      'Compare precision, recall, F1, accuracy, and ROC-style threshold behavior',
      'Choose a metric that matches the cost of different mistakes',
    ],
    commonMisconception: 'Accuracy can look excellent on imbalanced data while the model misses the class you actually care about.',
  },
  overfitting: {
    difficulty: 'beginner',
    estimatedMinutes: 16,
    prerequisites: ['train-validation-test-split'],
    learningObjectives: [
      'Recognize the gap between training error and validation error',
      'Explain why too much model flexibility can memorize noise',
      'Use validation behavior to decide when a model has become too complex',
    ],
    commonMisconception: 'Lower training loss is not always better; after a point it can signal memorization rather than generalization.',
  },
  regularization: {
    difficulty: 'intermediate',
    estimatedMinutes: 18,
    prerequisites: ['overfitting', 'classification-metrics'],
    learningObjectives: [
      'Describe how penalties discourage overly large parameters',
      'Compare under-regularized, balanced, and over-regularized models',
      'Use validation performance to tune the regularization strength',
    ],
    commonMisconception: 'Regularization is not a magic accuracy boost; too much penalty can underfit by making the model too simple.',
  },
  tokenization: {
    difficulty: 'beginner',
    estimatedMinutes: 15,
    prerequisites: [],
  },
  embeddings: {
    difficulty: 'beginner',
    estimatedMinutes: 14,
    prerequisites: ['matrix-multiplication'],
  },
  'cosine-similarity': {
    difficulty: 'beginner',
    estimatedMinutes: 12,
    prerequisites: ['matrix-multiplication'],
  },
  'attention-mechanism': {
    difficulty: 'intermediate',
    estimatedMinutes: 18,
    prerequisites: ['embeddings', 'softmax'],
  },
  'self-attention': {
    difficulty: 'intermediate',
    estimatedMinutes: 20,
    prerequisites: ['matrix-multiplication', 'softmax'],
    learningObjectives: [
      'Compute scaled attention scores from query and key vectors',
      'Explain why softmax turns scores into attention weights',
      'Describe how value vectors are mixed into a context-aware output',
    ],
    commonMisconception: 'Self-attention is not a lookup table; it recomputes weighted mixtures for each token from the current sequence.',
  },
  'positional-encoding': {
    difficulty: 'intermediate',
    estimatedMinutes: 16,
    prerequisites: ['self-attention'],
  },
  transformer: {
    difficulty: 'advanced',
    estimatedMinutes: 25,
    prerequisites: ['self-attention', 'positional-encoding', 'layer-normalization'],
  },
  bert: {
    difficulty: 'advanced',
    estimatedMinutes: 24,
    prerequisites: ['transformer', 'tokenization'],
  },
  'gpt2-comprehensive': {
    difficulty: 'advanced',
    estimatedMinutes: 30,
    prerequisites: ['transformer', 'positional-encoding'],
  },
  'flash-attention': {
    difficulty: 'advanced',
    estimatedMinutes: 18,
    prerequisites: ['self-attention', 'matrix-multiplication'],
  },
  'kv-cache': {
    difficulty: 'advanced',
    estimatedMinutes: 18,
    prerequisites: ['self-attention', 'gpt2-comprehensive'],
  },
  'grouped-query-attention': {
    difficulty: 'advanced',
    estimatedMinutes: 18,
    prerequisites: ['self-attention', 'kv-cache'],
  },
  'neural-network': {
    difficulty: 'beginner',
    estimatedMinutes: 18,
    prerequisites: ['linear-regression', 'gradient-descent'],
  },
  'computation-graph-backprop': {
    difficulty: 'intermediate',
    estimatedMinutes: 22,
    prerequisites: ['linear-regression', 'gradient-descent', 'relu'],
    learningObjectives: [
      'Trace forward values through a computation graph',
      'Compute local derivatives for each graph edge',
      'Apply the chain rule in reverse to accumulate parameter gradients',
      'Use gradients and a learning rate to update weights and reduce loss',
    ],
    commonMisconception: 'Backpropagation is not a mysterious new learning rule; it is the chain rule applied to local derivatives in reverse graph order.',
  },
  relu: {
    difficulty: 'beginner',
    estimatedMinutes: 10,
    prerequisites: ['neural-network'],
  },
  'leaky-relu': {
    difficulty: 'beginner',
    estimatedMinutes: 10,
    prerequisites: ['relu'],
  },
  'gradient-problems': {
    difficulty: 'intermediate',
    estimatedMinutes: 16,
    prerequisites: ['neural-network', 'relu'],
  },
  'layer-normalization': {
    difficulty: 'intermediate',
    estimatedMinutes: 16,
    prerequisites: ['neural-network', 'gradient-problems'],
  },
  conv2d: {
    difficulty: 'intermediate',
    estimatedMinutes: 18,
    prerequisites: ['matrix-multiplication', 'neural-network'],
  },
  vae: {
    difficulty: 'advanced',
    estimatedMinutes: 24,
    prerequisites: ['probability-distributions', 'neural-network'],
  },
  'diffusion-vae': {
    difficulty: 'advanced',
    estimatedMinutes: 24,
    prerequisites: ['vae'],
  },
  rag: {
    difficulty: 'advanced',
    estimatedMinutes: 22,
    prerequisites: ['embeddings', 'cosine-similarity'],
  },
  'rl-foundations': {
    difficulty: 'beginner',
    estimatedMinutes: 16,
    prerequisites: ['markov-chains'],
  },
  'q-learning': {
    difficulty: 'intermediate',
    estimatedMinutes: 20,
    prerequisites: ['rl-foundations', 'expected-value-variance'],
  },
};

function makeCurriculumMetadata(item, category) {
  const defaults = CATEGORY_CURRICULUM_DEFAULTS[category.id] || {
    difficulty: 'intermediate',
    estimatedMinutes: 15,
    prerequisites: [],
  };
  const override = CURRICULUM_OVERRIDES[item.id] || {};
  const difficulty = override.difficulty || defaults.difficulty;
  const estimatedMinutes = override.estimatedMinutes || defaults.estimatedMinutes;

  return {
    difficulty,
    prerequisites: override.prerequisites || defaults.prerequisites,
    estimatedMinutes,
    learningObjectives: override.learningObjectives || [
      `Explain the core idea behind ${item.name}`,
      `Use the animation to predict how ${item.description.toLowerCase()} changes the output`,
    ],
    commonMisconception:
      override.commonMisconception ||
      `${item.name} is a simplified teaching view; real models add scale, data, and implementation details.`,
    trackIds: trackIdsByAnimation[item.id] || [category.id],
  };
}

// Flatten all items for easy lookup
export const allAnimations = categories.flatMap(category =>
  category.items.map(item => ({
    ...item,
    categoryId: category.id,
    categoryName: category.name,
    categoryColor: category.color,
    ...makeCurriculumMetadata(item, category),
  }))
);

// Get animation by id
export const getAnimationById = (id) => allAnimations.find(a => a.id === id);

// Get category by id
export const getCategoryById = (id) => categories.find(c => c.id === id);
