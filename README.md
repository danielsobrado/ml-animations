# ML Animations

Interactive visualizations of Machine Learning and Linear Algebra concepts, built with React, Three.js, and GSAP.

![Unified App](screenshots/unified-app.png)

## 🚀 Unified Application

All animations are now available in a single unified React application with:
- 🎨 Consistent design system
- 🌓 Dark/Light mode toggle
- 📱 Responsive sidebar navigation
- ⚡ Lazy loading for performance

```bash
cd unified-app
npm install
npm run dev
```

---

## 📚 Table of Contents

- [Natural Language Processing](#natural-language-processing)
- [Transformers & Attention](#transformers--attention)
- [Neural Networks](#neural-networks)
- [Advanced Models](#advanced-models)
- [Math Fundamentals](#math-fundamentals)
- [Probability & Statistics](#probability--statistics)
- [Reinforcement Learning](#reinforcement-learning)
- [Algorithms & Data Structures](#algorithms--data-structures)
- [Information Theory](#information-theory)

---

## Natural Language Processing

### Bag of Words Animation

A foundational NLP concept showing how text is converted to numerical vectors.

![Bag of Words](screenshots/bag-of-words.png)

- **Visualizes:** Text → Word Frequency Vectors
- **Features:**
  - Interactive document input
  - Vocabulary building visualization
  - Vector representation display
  - Similarity calculations between documents

```bash
cd bag-of-words-animation && npm install && npm run dev
```

### Word2Vec Animation

Understanding word embeddings through the Skip-gram and CBOW models.

![Word2Vec](screenshots/word2vec.png)

- **Visualizes:** Words → Dense Vector Space
- **3 Learning Modes:**
  1. **📖 Concept**: Understanding context windows and word relationships
  2. **🎯 Skip-gram**: Predicting context words from center word
  3. **🔄 CBOW**: Predicting center word from context
- **Features:**
  - Interactive training visualization
  - 2D/3D embedding space exploration
  - Analogy demonstrations (King - Man + Woman = Queen)

```bash
cd word2vec-animation && npm install && npm run dev
```

### GloVe Animation

Global Vectors for Word Representation - learning from co-occurrence statistics.

![GloVe](screenshots/glove.png)

- **Visualizes:** Co-occurrence Matrix → Word Vectors
- **Features:**
  - Co-occurrence matrix construction
  - Objective function visualization
  - Comparison with Word2Vec approach
  - Interactive embedding exploration

```bash
cd glove-animation && npm install && npm run dev
```

### FastText Animation

Subword embeddings for handling rare and out-of-vocabulary words.

![FastText](screenshots/fasttext.png)

- **Visualizes:** Words → Character n-grams → Embeddings
- **Features:**
  - Subword decomposition visualization
  - OOV word handling demonstration
  - Morphologically rich language support
  - Comparison with Word2Vec

```bash
cd fasttext-animation && npm install && npm run dev
```

### Tokenization Animation

Breaking text into tokens - the first step in NLP pipelines.

![Tokenization](screenshots/tokenization.png)

- **Visualizes:** Text → Tokens → IDs
- **3 Learning Modes:**
  1. **✂️ Word Tokenization**: Simple whitespace and punctuation splitting
  2. **🧩 Subword (BPE)**: Byte Pair Encoding algorithm
  3. **🔤 SentencePiece**: Unigram language model tokenization
- **Features:**
  - Interactive tokenization playground
  - Vocabulary size impact visualization
  - Encoding/decoding demonstration

```bash
cd tokenization-animation && npm install && npm run dev
```

---

## Transformers & Attention

### Attention Mechanism Animation

The core intuition behind attention - learning what to focus on.

![Attention Mechanism](screenshots/attention-mechanism.png)

- **Visualizes:** Query-Key-Value attention mechanism
- **3 Learning Modes:**
  1. **📚 Library Search**: Intuitive analogy for Q, K, V
  2. **🌍 Translation**: Attention in sequence-to-sequence models
  3. **💬 Conversation**: Context-aware attention

```bash
cd attention-mechanism-animation && npm install && npm run dev
```

### Self-Attention Animation

The heart of the Transformer architecture.

![Self-Attention](screenshots/self-attention.png)

- **Visualizes:** softmax(QK^T / √d)V
- **3 Learning Modes:**
  1. **🔑 The Concept**: Intuitive "Library" analogy for Query, Key, and Value
  2. **🧮 The Mechanism**: Step-by-step visualization of the math
  3. **🕸️ Playground**: Interactive text attention visualization (Coreference Resolution)
- **Features:**
  - Interactive Q/K/V matching
  - Matrix multiplication visualization
  - Real-time attention heatmaps for text

```bash
cd self-attention-animation && npm install && npm run dev
```

### BERT Animation

Bidirectional Encoder Representations from Transformers.

![BERT](screenshots/bert.png)

- **Visualizes:** Pre-training objectives and fine-tuning
- **3 Learning Modes:**
  1. **🎭 Masked Language Model (MLM)**: Predicting masked tokens
  2. **📝 Next Sentence Prediction (NSP)**: Understanding sentence relationships
  3. **🔧 Fine-tuning**: Adapting BERT for downstream tasks
- **Features:**
  - Token embedding visualization
  - Attention pattern exploration
  - Interactive masking playground

```bash
cd bert-animation && npm install && npm run dev
```

### Transformer Animation

The complete Transformer architecture - "Attention Is All You Need".

![Transformer](screenshots/transformer.png)

- **Visualizes:** Encoder-Decoder architecture
- **4 Learning Modes:**
  1. **🏗️ Architecture Overview**: High-level component breakdown
  2. **🔄 Encoder Stack**: Multi-head attention and feed-forward layers
  3. **📤 Decoder Stack**: Masked attention and cross-attention
  4. **🎬 Full Forward Pass**: End-to-end sequence transformation
- **Features:**
  - Layer-by-layer visualization
  - Residual connections and layer normalization
  - Interactive token flow

```bash
cd transformer-animation && npm install && npm run dev
```

### Positional Encoding Animation

How Transformers understand word order.

![Positional Encoding](screenshots/positional-encoding.png)

- **Visualizes:** Position information in sequences
- **3 Learning Modes:**
  1. **🔀 The Problem**: Demonstrate why "Dog bites man" ≠ "Man bites dog"
  2. **🌊 Sinusoidal Encoding**: Visualize the wave patterns that encode position
  3. **🎮 Encoding Playground**: Compare Sinusoidal vs. Learned vs. Integer encoding
- **Features:**
  - Interactive sentence comparison
  - Waveform visualization with multiple frequencies

```bash
cd positional-encoding-animation && npm install && npm run dev
```

---

## Neural Networks

### ReLU Activation Animation

A visual explanation of the ReLU (Rectified Linear Unit) activation function.

![ReLU](screenshots/relu.png)

- **Visualizes:** z = W·X + b → ReLU(z) = max(0, z)
- **Features:**
  - Step-by-step animation showing dot product, bias addition, and ReLU application
  - Interactive ReLU graph visualization synchronized with steps
  - Practice mode with randomly generated problems
  - Formula reference and hints
  - Built with Three.js for 3D rendering

```bash
cd relu-animation && npm install && npm run dev
```

### Leaky ReLU Activation Animation

A visual explanation of the Leaky ReLU activation function.

![Leaky ReLU](screenshots/leaky-relu.png)

- **Visualizes:** z = W·X + b → Leaky ReLU(z) = z if z > 0, else α×z
- **Features:**
  - Step-by-step animation showing dot product, bias addition, and Leaky ReLU application
  - Interactive Leaky ReLU graph visualization with reference line
  - Practice mode with randomly generated problems
  - α (alpha) parameter visualization (default: 0.01)

```bash
cd leaky-relu-animation && npm install && npm run dev
```

### Softmax Animation

Converting logits to probabilities for classification.

![Softmax](screenshots/softmax.png)

- **Visualizes:** e^(xi) / Σe^(xj) → Probability Distribution
- **3 Learning Modes:**
  1. **📊 The Math**: Step-by-step calculation breakdown
  2. **🌡️ Temperature Scaling**: How temperature affects sharpness
  3. **🎯 Classification**: Real-world application in neural networks
- **Features:**
  - Interactive logit adjustment
  - Probability distribution visualization
  - Temperature parameter exploration

```bash
cd softmax-animation && npm install && npm run dev
```

### LSTM Animation (Deep Dive)

A "Bit-by-Bit" interactive guide to Long Short-Term Memory networks.

![LSTM](screenshots/lstm.png)

- **4-Mode Learning System:**
  1. **📚 The Conveyor Belt**: Intuitive analogy for cell state flow
  2. **🔬 Anatomy Lab**: Interactive component explorer (Sigmoid, Tanh, Gates)
  3. **🎬 Bit-by-Bit Flow**: Granular 8-step animation of a single time step
  4. **🔁 Sequence View**: Visualization of LSTM unrolled over time

```bash
cd lstm-animation && npm install && npm run dev
```

### 2D Convolution Animation

A visual guide to 2D convolution operations used in CNNs.

![Conv2D](screenshots/conv2d.png)

- **Visualizes:** Input (5×5) ∗ Kernel (3×3) = Output (3×3)
- **Features:**
  - Animated kernel sliding across input matrix
  - Real-time element-wise multiplication and summation display
  - Color-coded highlighting showing kernel position
  - Practice mode with different kernel types (edge detection, sharpen, blur)

```bash
cd conv2d-animation && npm install && npm run dev
```

### Multi-Input Neural Network (Conv + ReLU)

A visual demonstration of a two-layer neural network with three inputs and ReLU activations.

![Conv ReLU](screenshots/conv-relu.png)

- **Visualizes:** X × W₁ → ReLU → A₁ × W₂ → ReLU → Output
- **Features:**
  - Two-layer feedforward network with matrix multiplication
  - Step-by-step animation showing forward propagation
  - Clear visualization of pre-activation (Z) and post-activation (A) values
  - Practice mode with randomly generated matrix problems

```bash
cd conv-relu-animation && npm install && npm run dev
```

### Layer Normalization Animation

Stabilizing deep network training.

![Layer Normalization](screenshots/layer-normalization.png)

- **3 Learning Modes:**
  1. **📊 The Problem**: Visualize activation drift and gradient instability
  2. **⚖️ Layer Norm vs Batch Norm**: Compare normalization across different axes
  3. **🔄 Residual Connections**: Demonstrate the "Add & Norm" pattern in Transformers
- **Features:**
  - Activation distribution visualization
  - Interactive batch size adjustment
  - Gradient flow demonstration

```bash
cd layer-normalization-animation && npm install && npm run dev
```

---

## Advanced Models

### VAE Animation (Variational Autoencoder)

Understanding generative models through the lens of variational inference.

![VAE](screenshots/vae.png)

- **Visualizes:** Encoder → Latent Space → Decoder
- **3 Learning Modes:**
  1. **🔄 Autoencoder Basics**: Compression and reconstruction
  2. **📊 The Latent Space**: Continuous latent representation
  3. **🎲 Sampling & Generation**: Reparameterization trick
- **Features:**
  - Interactive latent space exploration
  - KL divergence visualization
  - Image generation playground

```bash
cd vae-animation && npm install && npm run dev
```

### RAG Animation (Retrieval-Augmented Generation)

Enhancing LLMs with external knowledge retrieval.

![RAG](screenshots/rag.png)

- **Visualizes:** Query → Retrieve → Augment → Generate
- **3 Learning Modes:**
  1. **📚 The Pipeline**: Understanding RAG architecture
  2. **🔍 Retrieval**: Vector similarity search
  3. **✨ Generation**: Augmented context for LLM
- **Features:**
  - Document embedding visualization
  - Relevance scoring demonstration
  - Interactive query playground

```bash
cd rag-animation && npm install && npm run dev
```

### Multimodal LLM Animation

Understanding models that process multiple modalities (text, images, audio).

![Multimodal LLM](screenshots/multimodal-llm.png)

- **Visualizes:** Multi-modal input processing and fusion
- **Features:**
  - Vision encoder architecture
  - Cross-modal attention mechanisms
  - Token embedding visualization
  - Interactive multimodal queries

```bash
cd multimodal-llm-animation && npm install && npm run dev
```

### Fine-tuning Animation

Adapting pre-trained models for specific tasks.

![Fine-tuning](screenshots/fine-tuning.png)

- **Visualizes:** Pre-trained → Fine-tuned model transformation
- **3 Learning Modes:**
  1. **🎯 Full Fine-tuning**: Updating all parameters
  2. **❄️ Freeze & Train**: Selective layer training
  3. **🔌 LoRA/Adapters**: Parameter-efficient fine-tuning
- **Features:**
  - Layer-wise training visualization
  - Learning rate strategies
  - Catastrophic forgetting demonstration

```bash
cd fine-tuning-animation && npm install && npm run dev
```

---

## Math Fundamentals

### Matrix Multiplication Animation

A step-by-step visual guide to matrix multiplication.

![Matrix Multiplication](screenshots/matrix-multiplication.png)

- **Visualizes:** Matrix A (2x2) × Matrix B (2x3) = Matrix C (2x3)
- **Features:**
  - Step-by-step animation of row-column dot products
  - Color-coded highlighting of active rows and columns
  - Interactive controls (Play, Reset, Next/Prev Step)
  - Practice mode with different matrices
  - Built with Three.js for 3D rendering

```bash
cd matrix-multiplication-animation && npm install && npm run dev
```

### SVD Animation

A comprehensive visualization of Singular Value Decomposition.

![SVD](screenshots/svd.png)

- **Visualizes:** A (m×n) = U (m×m) × Σ (m×n) × V^T (n×n)
- **Features:**
  - Step-by-step SVD decomposition animation (9 steps)
  - Shows U (left singular vectors), Σ (singular values), V^T (right singular vectors)
  - Visualizes reconstruction: A = UΣV^T
  - Practice mode with exercises
  - Educational info on ML applications (PCA, compression)

```bash
cd svd-animation && npm install && npm run dev
```

### Eigenvalue Decomposition Animation ✨

A comprehensive learning system teaching eigenvalues from first principles.

![Eigenvalue](screenshots/eigenvalue.png)

- **Visualizes:** A = Q Λ Q^T (for symmetric A)
- **5 Learning Modes:**
  1. **📚 Tutorial Mode**: 7-step conceptual learning
  2. **🌐 Geometric Visualizer**: Circle → ellipse transformation
  3. **🎮 Interactive Explorer**: Drag vectors to see transformation
  4. **🎬 Matrix Decomposition**: Step-by-step A = QΛQ^T animation
  5. **✏️ Practice Exercises**: Interactive problems with hints

```bash
cd eigenvalue-animation && npm install && npm run dev
```

### QR Decomposition Animation

A demonstration of QR decomposition using the Gram-Schmidt process.

![QR Decomposition](screenshots/qr-decomposition.png)

- **Visualizes:** A = Q × R (orthonormal Q, upper triangular R)
- **Features:**
  - Step-by-step Gram-Schmidt orthogonalization (6 steps)
  - Shows transformation of matrix columns into orthonormal basis
  - Visualizes Q (orthonormal columns) and R (upper triangular)
  - Practice mode with QR decomposition exercises

```bash
cd qr-decomposition-animation && npm install && npm run dev
```

### Gradient Descent Animation

A step-by-step guide to the "Learning Process" of neural networks.

![Gradient Descent](screenshots/gradient-descent.png)

- **4 Learning Modes:**
  1. **🏔️ The Hiker**: Intuitive analogy of a hiker in the fog
  2. **📉 2D Slope Lab**: Connecting the hiker to the math (dy/dx)
  3. **🏞️ 3D Landscape**: Exploring complex terrain with local minima
  4. **🎛️ Tuning Studio**: Experimenting with Learning Rates
- **Features:**
  - Interactive 3D visualization with Three.js
  - Learning rate comparison

```bash
cd gradient-descent-animation && npm install && npm run dev
```

### Linear Regression Animation

Finding the line of best fit.

![Linear Regression](screenshots/linear-regression.png)

- **3 Learning Modes:**
  1. **📏 The Residuals**: Manually drag the line to minimize error squares
  2. **✍️ Interactive Fitter**: Click to add points, see OLS formula update
  3. **🥣 The Cost Landscape**: Visualize MSE as a 3D surface bowl
- **Features:**
  - Interactive point manipulation
  - Real-time coefficient updates

```bash
cd linear-regression-animation && npm install && npm run dev
```

### Embeddings Animation

Where words become geometry.

![Embeddings](screenshots/embeddings.png)

- **3 Learning Modes:**
  1. **🧮 Word Algebra**: Visualizing "King - Man + Woman = Queen"
  2. **📐 Similarity Lab**: Interactive Cosine Similarity calculator
  3. **🌌 3D Semantic Space**: Fly through a galaxy of word clusters
- **Features:**
  - Vector arithmetic visualization
  - 3D Point Cloud with Three.js

```bash
cd embeddings-animation && npm install && npm run dev
```

### Cosine Similarity Animation

The math behind recommendations and search.

![Cosine Similarity](screenshots/cosine-similarity.png)

- **3 Learning Modes:**
  1. **✖️ The Dot Product**: Visualizing projections
  2. **🎬 Movie Matcher**: Build a recommender system
  3. **🔍 Search Engine**: Rank documents by relevance
- **Features:**
  - Interactive vector manipulation
  - Radar charts for multi-dimensional comparison

```bash
cd cosine-similarity-animation && npm install && npm run dev
```

---

## Probability & Statistics

### Conditional Probability & Bayes' Theorem Animation

Updating beliefs with evidence.

![Conditional Probability](screenshots/conditional-probability.png)

- **3 Learning Modes:**
  1. **🎲 Conditional Probability**: Interactive card deck demonstrating P(A|B)
  2. **🔄 Bayes' Theorem**: Formula breakdown with sliders
  3. **🏥 Medical Testing**: Real-world application with disease diagnosis
- **Features:**
  - Interactive filtering to visualize conditioning
  - Real-time Bayes' calculation
  - Population grid showing true/false positives

```bash
cd conditional-probability-animation && npm install && npm run dev
```

### Probability Distributions Animation

Modeling randomness with mathematics.

![Probability Distributions](screenshots/probability-distributions.png)

- **3 Learning Modes:**
  1. **🎲 Discrete Distributions**: Binomial and Poisson
  2. **📊 Continuous Distributions**: Normal and Exponential
  3. **⚖️ PMF vs PDF**: Understanding the critical difference
- **Features:**
  - Interactive parameter sliders
  - Real-time distribution visualization
  - Area under curve calculation

```bash
cd probability-distributions-animation && npm install && npm run dev
```

### Expected Value & Variance Animation

Quantifying center and spread.

![Expected Value Variance](screenshots/expected-value-variance.png)

- **3 Learning Modes:**
  1. **⚖️ Expected Value**: Visualize E[X] as the "balance point"
  2. **📏 Variance**: Compare narrow vs wide distributions
  3. **🎰 Decision Making**: Apply to investment choices
- **Features:**
  - Balance beam visualization for E[X]
  - Deviation visualization for variance
  - Risk-adjusted decision making

```bash
cd expected-value-variance-animation && npm install && npm run dev
```

### Markov Chains Animation

The engine behind PageRank and Text Generation.

![Markov Chains](screenshots/markov-chains.png)

- **4 Learning Modes:**
  1. **🐸 The Markov Property**: Frog simulator demonstrating memorylessness
  2. **🕸️ Transition Matrix**: Interactive graph-to-matrix builder
  3. **⚖️ Stationary Distribution**: Visualizing convergence
  4. **📝 Text Generation**: Mini-LLM using bigram models
- **Features:**
  - Animated simulations
  - Real-time matrix updates
  - Text generation playground

```bash
cd markov-chains-animation && npm install && npm run dev
```

### Spearman Correlation Animation

An interactive exploration of Rank Correlation and Robustness.

![Spearman Correlation](screenshots/spearman-correlation.png)

- **3 Learning Modes:**
  1. **💡 Concept**: Raw Space (curved) to Rank Space (linear)
  2. **🧮 Calculation Lab**: Step-by-step animated table
  3. **⚖️ Robustness**: Interactive outlier sensitivity (Pearson vs. Spearman)

```bash
cd spearman-correlation-animation && npm install && npm run dev
```

---

## Reinforcement Learning

### Part 1: RL Foundations

The building blocks of Reinforcement Learning.

![RL Foundations](screenshots/rl-foundations.png)

- **3 Learning Modes:**
  1. **🤖 The Agent**: Manual Gridworld to understand State-Action-Reward loops
  2. **💎 Rewards**: Design level rewards and analyze path returns
  3. **📉 Discounted Returns**: Visualize how Gamma affects long-term planning
- **Features:**
  - Playable Gridworld
  - Level Editor
  - Interactive Discount Factor visualization

```bash
cd rl-foundations-animation && npm install && npm run dev
```

### Part 2: Q-Learning Algorithm

The core algorithm where the agent learns from experience.

![Q-Learning](screenshots/q-learning.png)

- **3 Learning Modes:**
  1. **📊 The Q-Table**: Visualize the agent's brain (State-Action Values)
  2. **🧮 The Bellman Update**: Step-by-step math visualization
  3. **🏋️ Training Loop**: Watch the agent master the maze
- **Features:**
  - Interactive Q-Table inspection
  - Real-time training graphs
  - Adjustable learning parameters

```bash
cd q-learning-animation && npm install && npm run dev
```

### Part 3: Exploration & Optimization

Mastering the trade-offs in Reinforcement Learning.

![RL Exploration](screenshots/rl-exploration.png)

- **3 Learning Modes:**
  1. **🎲 Epsilon-Greedy**: Visualize Explore vs Exploit
  2. **🧗 The Cliff**: Risk vs Reward - why optimal isn't always safe
  3. **🎛️ Hyperparameters**: Interactive tuning lab for Alpha and Gamma
- **Features:**
  - Live exploration stats
  - Cliff walking simulation
  - Learning curve projections

```bash
cd rl-exploration-animation && npm install && npm run dev
```

---

## Algorithms & Data Structures

### Bloom Filter Animation

A probabilistic data structure explorer.

![Bloom Filter](screenshots/bloom-filter.png)

- **3 Learning Modes:**
  1. **🎮 Playground**: Interactive visualizer to Add and Check items
  2. **⚠️ False Positive Lab**: Create collisions to understand "Probably Yes"
  3. **🎛️ Tuning Studio**: Find the optimal k (hash functions)

```bash
cd bloom-filter-animation && npm install && npm run dev
```

### PageRank Animation

The algorithm that built Google.

![PageRank](screenshots/pagerank.png)

- **3 Learning Modes:**
  1. **🕸️ Graph Builder**: Build your own "Mini-Internet"
  2. **🏄 Random Surfer**: Monte Carlo simulation visualization
  3. **👑 Power Method**: Watch "Rank Juice" flow to steady state

```bash
cd pagerank-animation && npm install && npm run dev
```

---

## Information Theory

### Part 1: Entropy & Information

The math of uncertainty and surprise.

![Entropy](screenshots/entropy.png)

- **2 Learning Modes:**
  1. **💡 The Bit**: Visualize Information as "Surprise" (-log p)
  2. **📊 Entropy**: Interactive distribution editor to maximize/minimize uncertainty
- **Features:**
  - Real-time bit calculation
  - Dynamic probability adjustment

```bash
cd entropy-animation && npm install && npm run dev
```

### Part 2: Cross-Entropy & KL Divergence

The bridge between Probability and Machine Learning optimization.

![Cross-Entropy](screenshots/cross-entropy.png)

- **2 Learning Modes:**
  1. **🎯 Cross-Entropy**: Visualize the Loss Function used in classification
  2. **⚖️ KL Divergence**: Visualize the "Distance" between two distributions
- **Features:**
  - Interactive prediction matching
  - Gaussian distribution comparison

```bash
cd cross-entropy-animation && npm install && npm run dev
```

---

## Technologies Used

- **React**: UI and state management
- **Three.js**: 3D graphics rendering
- **GSAP**: Smooth animations
- **Vite**: Fast build tool and development server
- **Tailwind CSS**: Styling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your animation following the existing pattern
4. Submit a pull request

## License

MIT License - See LICENSE file for details
