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
  RotateCcw
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
      { id: 'eigenvalue', name: 'Eigenvalues', icon: Sigma, description: 'Matrix eigendecomposition' },
      { id: 'svd', name: 'SVD', icon: ArrowRightLeft, description: 'Singular value decomposition' },
      { id: 'qr-decomposition', name: 'QR Decomposition', icon: Boxes, description: 'Orthogonal matrix factorization' },
      { id: 'gradient-descent', name: 'Gradient Descent', icon: TrendingUp, description: 'Optimization algorithm' },
      { id: 'optimization', name: 'Optimization Landscape', icon: Target, description: 'Exploring loss surfaces and minima' },
      { id: 'linear-regression', name: 'Linear Regression', icon: LineChart, description: 'Fitting linear models' },
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

// Flatten all items for easy lookup
export const allAnimations = categories.flatMap(category =>
  category.items.map(item => ({
    ...item,
    categoryId: category.id,
    categoryName: category.name,
    categoryColor: category.color,
  }))
);

// Get animation by id
export const getAnimationById = (id) => allAnimations.find(a => a.id === id);

// Get category by id
export const getCategoryById = (id) => categories.find(c => c.id === id);
