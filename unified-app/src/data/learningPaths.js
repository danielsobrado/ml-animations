export const HUB_LEARNING_PATHS = [
  {
    id: 'start-here',
    label: 'Start Here',
    description: 'A compact route through the algebra, fitting, validation, and classification ideas that make the rest of the catalog easier to read.',
    nodes: [
      'matrix-multiplication',
      'linear-regression',
      'train-validation-test-split',
      'logistic-regression',
      'classification-metrics',
      'computation-graph-backprop',
    ],
  },
  {
    id: 'llm-path',
    label: 'NLP To LLMs',
    description: 'A language-model route from tokens and embeddings into attention, self-attention, transformers, and GPT-style stacks.',
    nodes: [
      'tokenization',
      'embeddings',
      'attention-mechanism',
      'self-attention',
      'transformer',
      'gpt2-comprehensive',
    ],
  },
  {
    id: 'vision-path',
    label: 'Vision And Generation',
    description: 'A visual-model route from convolution and pooling into latent-variable models and modern diffusion systems.',
    nodes: [
      'conv2d',
      'max-pooling',
      'vae',
      'diffusion-vae',
      'sd3-overview',
    ],
  },
  {
    id: 'rl-path',
    label: 'RL And Algorithms',
    description: 'A decision-making route through Markov structure, rewards, Q-values, exploration, and graph ranking.',
    nodes: [
      'markov-chains',
      'rl-foundations',
      'q-learning',
      'rl-exploration',
      'pagerank',
    ],
  },
];
