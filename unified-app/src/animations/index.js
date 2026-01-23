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
  'grouped-query-attention': lazy(() => import('./grouped-query-attention')),
  'kv-cache': lazy(() => import('./kv-cache')),
  'transformer': lazy(() => import('./transformer')),
  'bert': lazy(() => import('./bert')),
  'gpt2-comprehensive': lazy(() => import('./gpt2-comprehensive')),
  'positional-encoding': lazy(() => import('./positional-encoding')),
  'rope': lazy(() => import('./rope')),

  // Neural Network Components
  'relu': lazy(() => import('./relu')),
  'leaky-relu': lazy(() => import('./leaky-relu')),
  'conv2d': lazy(() => import('./conv2d')),
  'max-pooling': lazy(() => import('./max-pooling')),
  'conv-relu': lazy(() => import('./conv-relu')),
  'neural-network': lazy(() => import('./neural-network')),
  'lstm': lazy(() => import('./lstm')),
  'layer-normalization': lazy(() => import('./layer-normalization')),
  'softmax': lazy(() => import('./softmax')),

  // Advanced Models
  'vae': lazy(() => import('./vae')),
  'moe': lazy(() => import('./moe')),
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
  'optimization': lazy(() => import('./optimization')),
  'gradient-problems': lazy(() => import('./gradient-problems')),

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

  // Diffusion Models (SD3)
  'sd3-overview': lazy(() => import('./sd3-overview')),
  'flow-matching': lazy(() => import('./flow-matching')),
  'diffusion-vae': lazy(() => import('./diffusion-vae')),
  'tokenizer-bpe': lazy(() => import('./tokenizer-bpe')),
  'clip-encoder': lazy(() => import('./clip-encoder')),
  't5-encoder': lazy(() => import('./t5-encoder')),
  'joint-attention': lazy(() => import('./joint-attention')),
  'dit': lazy(() => import('./dit')),
};

export function getAnimationComponent(id) {
  return animationRegistry[id] || null;
}

export function isAnimationAvailable(id) {
  return id in animationRegistry;
}

export const availableAnimations = Object.keys(animationRegistry);
