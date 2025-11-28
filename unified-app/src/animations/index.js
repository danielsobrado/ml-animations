import { lazy } from 'react';

// Registry of available animation components
// All animations are now integrated from their individual projects
const animationRegistry = {
  // NLP & Text Processing
  'bag-of-words': lazy(() => import('./bag-of-words')),
  'word2vec': lazy(() => import('./word2vec')),
  'glove': lazy(() => import('./glove')),
  'fasttext': lazy(() => import('./fasttext')),
  'tokenization': lazy(() => import('./tokenization')),
  'embeddings': lazy(() => import('./embeddings')),
  
  // Transformers & Attention
  'attention-mechanism': lazy(() => import('./attention-mechanism')),
  'self-attention': lazy(() => import('./self-attention')),
  'transformer': lazy(() => import('./transformer')),
  'bert': lazy(() => import('./bert')),
  'positional-encoding': lazy(() => import('./positional-encoding')),
  
  // Neural Network Components
  'relu': lazy(() => import('./relu')),
  'leaky-relu': lazy(() => import('./leaky-relu')),
  'conv2d': lazy(() => import('./conv2d')),
  'conv-relu': lazy(() => import('./conv-relu')),
  'lstm': lazy(() => import('./lstm')),
  'layer-normalization': lazy(() => import('./layer-normalization')),
  'softmax': lazy(() => import('./softmax')),
  
  // Advanced Models
  'vae': lazy(() => import('./vae')),
  'rag': lazy(() => import('./rag')),
  'multimodal-llm': lazy(() => import('./multimodal-llm')),
  'fine-tuning': lazy(() => import('./fine-tuning')),
  
  // Linear Algebra & Math
  'matrix-multiplication': lazy(() => import('./matrix-multiplication')),
  'eigenvalue': lazy(() => import('./eigenvalue')),
  'svd': lazy(() => import('./svd')),
  'qr-decomposition': lazy(() => import('./qr-decomposition')),
  'linear-regression': lazy(() => import('./linear-regression')),
  'gradient-descent': lazy(() => import('./gradient-descent')),
  
  // Probability & Statistics
  'probability-distributions': lazy(() => import('./probability-distributions')),
  'conditional-probability': lazy(() => import('./conditional-probability')),
  'entropy': lazy(() => import('./entropy')),
  'cross-entropy': lazy(() => import('./cross-entropy')),
  'expected-value-variance': lazy(() => import('./expected-value-variance')),
  'cosine-similarity': lazy(() => import('./cosine-similarity')),
  'spearman-correlation': lazy(() => import('./spearman-correlation')),
  
  // Reinforcement Learning
  'rl-foundations': lazy(() => import('./rl-foundations')),
  'rl-exploration': lazy(() => import('./rl-exploration')),
  'q-learning': lazy(() => import('./q-learning')),
  
  // Algorithms & Data Structures
  'bloom-filter': lazy(() => import('./bloom-filter')),
  'markov-chains': lazy(() => import('./markov-chains')),
  'pagerank': lazy(() => import('./pagerank')),
};

export function getAnimationComponent(id) {
  return animationRegistry[id] || null;
}

export function isAnimationAvailable(id) {
  return id in animationRegistry;
}

export const availableAnimations = Object.keys(animationRegistry);
