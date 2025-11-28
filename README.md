# ML Animations

Interactive visualizations of Machine Learning and Linear Algebra concepts, built with React, Three.js, and GSAP.

## Projects

### Matrix Multiplication Animation

A step-by-step visual guide to matrix multiplication.

- **Visualizes:** Matrix A (2x2) × Matrix B (2x3) = Matrix C (2x3)
- **Features:**
  - Step-by-step animation of row-column dot products.
  - Color-coded highlighting of active rows and columns.
  - Interactive controls (Play, Reset, Next/Prev Step).
  - Practice mode with different matrices.
  - Built with Three.js for 3D rendering (orthographic view).

### ReLU Activation Animation

A visual explanation of the ReLU (Rectified Linear Unit) activation function.

- **Visualizes:** z = W·X + b → ReLU(z) = max(0, z)
- **Features:**
  - Step-by-step animation showing dot product, bias addition, and ReLU application.
  - Interactive ReLU graph visualization synchronized with steps.
  - Practice mode with randomly generated problems.
  - Formula reference and hints.
  - Built with Three.js for 3D rendering (orthographic view).

### Leaky ReLU Activation Animation

A visual explanation of the Leaky ReLU activation function.

- **Visualizes:** z = W·X + b → Leaky ReLU(z) = z if z > 0, else α×z
- **Features:**
  - Step-by-step animation showing dot product, bias addition, and Leaky ReLU application.
  - Interactive Leaky ReLU graph visualization with reference line.
  - Practice mode with randomly generated problems.
  - Comparison with standard ReLU (shows y=z reference line).
  - α (alpha) parameter visualization (default: 0.01).
  - Built with Three.js for 3D rendering (orthographic view).

### Multi-Input Neural Network (Conv + ReLU)

A visual demonstration of a two-layer neural network with three inputs and ReLU activations.

- **Visualizes:** X × W₁ → ReLU → A₁ × W₂ → ReLU → Output
- **Features:**
  - Two-layer feedforward network with matrix multiplication.
  - Step-by-step animation showing forward propagation through both layers.
  - Clear visualization of pre-activation (Z) and post-activation (A) values.
  - Practice mode with randomly generated matrix problems.
  - Shows how negative values become zero after ReLU.
  - Color-coded matrices for easy tracking.

### 2D Convolution Animation

A visual guide to 2D convolution operations used in Convolutional Neural Networks (CNNs).

- **Visualizes:** Input (5×5) ∗ Kernel (3×3) = Output (3×3)
- **Features:**
  - Animated kernel sliding across input matrix.
  - Real-time element-wise multiplication and summation display.
  - Color-coded highlighting showing kernel position on input.
  - Adjustable animation speed.
  - Click-to-jump to any convolution step.
  - Practice mode with different kernel types (edge detection, sharpen, blur, identity).
  - Interactive output cell computation with hints.

### SVD Animation

A comprehensive visualization of Singular Value Decomposition (SVD).

- **Visualizes:** A (m×n) = U (m×m) × Σ (m×n) × V^T (n×n)
- **Features:**
  - Step-by-step SVD decomposition animation (9 steps).
  - Shows U (left singular vectors), Σ (singular values diagonal), V^T (right singular vectors).
  - Visualizes reconstruction: A = UΣV^T.
  - Practice mode with exercises to find singular values.
  - Hints with formulas (eigenvalues of A^TA).
  - Educational info on ML applications (PCA, compression, recommender systems).

### Eigenvalue Decomposition Animation ✨ **Enhanced with Geometric Intuition**

A comprehensive learning system teaching eigenvalues **from first principles** with interactive exploration.

- **Visualizes:** A = Q Λ Q^T (for symmetric A)
- **5 Learning Modes:**
  - **📚 Tutorial Mode** - 7-step conceptual learning (transforms → eigenvectors → equation → decomposition).
  - **🌐 Geometric Visualizer** - Interactive circle → ellipse transformation showing eigenvectors as axes.
  - **🎮 Interactive Explorer** - Drag vectors to see transformation in real-time, detects eigenvector alignment.
  - **🎬 Matrix Decomposition** - Step-by-step animation of A = QΛQ^T (7 steps).
  - **✏️ Practice Exercises** - Interactive problems with hints and validation.
- **Features:**
  - Learn eigenvalues from scratch with no prior knowledge required.
  - Geometric intuition built before algebraic formulas.
  - Hands-on exploration with draggable vectors.
  - Tab-based interface for progressive learning.
  - ML applications highlighted (PCA, covariance analysis).

### QR Decomposition Animation

A demonstration of QR decomposition using the Gram-Schmidt process.

- **Visualizes:** A = Q × R (orthonormal Q, upper triangular R)
- **Features:**
  - Step-by-step Gram-Schmidt orthogonalization (6 steps).
  - Shows transformation of matrix columns into orthonormal basis.
  - Visualizes Q (orthonormal columns) and R (upper triangular).
  - Practice mode with QR decomposition exercises.
### LSTM Animation (Deep Dive)

A "Bit-by-Bit" interactive guide to Long Short-Term Memory networks.

- **4-Mode Learning System:**
  1. **📚 The Conveyor Belt**: Intuitive analogy for cell state flow.
  2. **🔬 Anatomy Lab**: Interactive component explorer (Sigmoid, Tanh, Gates).
  3. **🎬 Bit-by-Bit Flow**: Granular 8-step animation of a single time step.
  4. **🔁 Sequence View**: Visualization of LSTM unrolled over time.
- **Features:**
### Spearman Correlation Animation

An interactive exploration of Rank Correlation and Robustness.

- **3 Learning Modes:**
  1. **💡 Concept**: Visualizing the transformation from Raw Space (curved) to Rank Space (linear).
  2. **🧮 Calculation Lab**: Step-by-step animated table showing sorting, ranking, and formula application.
  3. **⚖️ Robustness**: Interactive playground to test outlier sensitivity (Pearson vs. Spearman).
- **Features:**
### Bloom Filter Animation

A probabilistic data structure explorer.

- **3 Learning Modes:**
  1. **🎮 Playground**: Interactive visualizer to Add and Check items, seeing bits flip in real-time.
  2. **⚠️ False Positive Lab**: Deliberately create collisions to understand why "Probably Yes" exists.
  3. **🎛️ Tuning Studio**: Interactive graph to find the optimal $k$ (hash functions) for your data.
- **Features:**
### Gradient Descent Animation

A step-by-step guide to the "Learning Process" of neural networks.

- **4 Learning Modes:**
  1. **🏔️ The Hiker**: Intuitive analogy of a hiker in the fog feeling the slope.
  2. **📉 2D Slope Lab**: Connecting the hiker to the math ($dy/dx$) and tangent lines.
  3. **🏞️ 3D Landscape**: Exploring complex terrain with local minima using Three.js.
  4. **🎛️ Tuning Studio**: Experimenting with Learning Rates (Too Slow vs. Too Fast).
- **Features:**
### PageRank Animation

The algorithm that built Google.

- **3 Learning Modes:**
  1. **🕸️ Graph Builder**: Build your own "Mini-Internet" by adding nodes and links.
  2. **🏄 Random Surfer**: Visualize a Monte Carlo simulation of a user clicking random links.
  3. **👑 Power Method**: Watch the "Rank Juice" flow and settle into a steady state mathematically.
- **Features:**
### Linear Regression Animation

Finding the line of best fit.

- **3 Learning Modes:**
  1. **📏 The Residuals**: Manually drag the line to minimize the error squares.
  2. **✍️ Interactive Fitter**: Click to add points and see the OLS formula update instantly.
  3. **🥣 The Cost Landscape**: Visualize the Mean Squared Error as a 3D surface bowl.
- **Features:**
### Embeddings Animation

Where words become geometry.

- **3 Learning Modes:**
  1. **🧮 Word Algebra**: Visualizing "King - Man + Woman = Queen" with vectors.
  2. **📐 Similarity Lab**: Interactive Cosine Similarity calculator.
  3. **🌌 3D Semantic Space**: Fly through a galaxy of word clusters.
- **Features:**
  - Vector arithmetic visualization.
  - 3D Point Cloud with Three.js.
### Cosine Similarity Animation

The math behind recommendations and search.

- **3 Learning Modes:**
  1. **✖️ The Dot Product**: Visualizing projections and the mechanics of the formula.
  2. **🎬 Movie Matcher**: Build a recommender system matching users by genre preferences.
  3. **🔍 Search Engine**: Rank documents by relevance to a search query.
- **Features:**
  - Interactive vector manipulation.
  - Radar charts for multi-dimensional comparison.
### Positional Encoding Animation

How Transformers understand word order.

- **3 Learning Modes:**
  1. **🔀 The Problem**: Demonstrate why "Dog bites man" ≠ "Man bites dog" without position info.
  2. **🌊 Sinusoidal Encoding**: Visualize the wave patterns that encode position.
  3. **🎮 Encoding Playground**: Compare Sinusoidal vs. Learned vs. Integer encoding.
- **Features:**
  - Interactive sentence comparison.
  - Waveform visualization with multiple frequencies.
### Layer Normalization Animation

Stabilizing deep network training.

- **3 Learning Modes:**
  1. **📊 The Problem**: Visualize activation drift and gradient instability without normalization.
  2. **⚖️ Layer Norm vs Batch Norm**: Compare normalization across different axes.
  3. **🔄 Residual Connections**: Demonstrate the "Add & Norm" pattern in Transformers.
- **Features:**
  - Activation distribution visualization.
  - Interactive batch size adjustment.
  - Gradient flow demonstration.

## Probability Theory Series

### Conditional Probability & Bayes' Theorem Animation

Updating beliefs with evidence.

- **3 Learning Modes:**
  1. **🎲 Conditional Probability**: Interactive card deck demonstrating P(A|B).
  2. **🔄 Bayes' Theorem**: Formula breakdown with sliders for prior, likelihood, and posterior.
  3. **🏥 Medical Testing**: Real-world application with disease diagnosis.
- **Features:**
  - Interactive filtering to visualize conditioning.
  - Real-time Bayes' calculation.
  - Population grid showing true/false positives.

### Probability Distributions Animation

Modeling randomness with mathematics.

- **3 Learning Modes:**
  1. **🎲 Discrete Distributions**: Binomial (coin flips) and Poisson (events per interval).
  2. **📊 Continuous Distributions**: Normal (bell curve) and Exponential (waiting times).
  3. **⚖️ PMF vs PDF**: Understanding the critical difference between discrete and continuous.
- **Features:**
  - Interactive parameter sliders.
  - Real-time distribution visualization.
  - Area under curve calculation for continuous distributions.

### Expected Value & Variance Animation

Quantifying center and spread.

- **3 Learning Modes:**
  1. **⚖️ Expected Value**: Visualize E[X] as the "balance point" with interactive die probabilities.
  2. **📏 Variance**: Compare narrow vs wide distributions with same mean.
  3. **🎰 Decision Making**: Apply to investment choices (Safe vs Risky).
- **Features:**
  - Balance beam visualization for E[X].
  - Deviation visualization for variance.
  - Risk-adjusted decision making with risk tolerance slider.

### Markov Chains Animation

The engine behind PageRank and Text Generation.

- **4 Learning Modes:**
  1. **🐸 The Markov Property**: Frog simulator demonstrating memorylessness.
  2. **🕸️ Transition Matrix**: Interactive graph-to-matrix builder.
  3. **⚖️ Stationary Distribution**: Visualizing convergence to steady state.
  4. **📝 Text Generation**: Mini-LLM using bigram models.
- **Features:**
  - Animated simulations.
  - Real-time matrix updates.
  - Text training and generation playground.

### Self-Attention Animation

The heart of the Transformer architecture.

- **3 Learning Modes:**
  1. **🔑 The Concept**: Intuitive "Library" analogy for Query, Key, and Value.
  2. **🧮 The Mechanism**: Step-by-step visualization of the math (`softmax(QK^T)V`).
  3. **🕸️ Playground**: Interactive text attention visualization (Coreference Resolution).
- **Features:**
  - Interactive Q/K/V matching.
  - Matrix multiplication visualization.
  - Real-time attention heatmaps for text.

## Reinforcement Learning Series

### Part 1: RL Foundations

The building blocks of Reinforcement Learning.

- **3 Learning Modes:**
  1. **🤖 The Agent**: Manual Gridworld to understand State-Action-Reward loops.
  2. **💎 Rewards**: Design level rewards and analyze path returns.
  3. **📉 Discounted Returns**: Visualize how Gamma affects long-term planning.
- **Features:**
  - Playable Gridworld.
  - Level Editor.
  - Interactive Discount Factor visualization.

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn

### Installation & Running

#### Matrix Multiplication Animation

```bash
cd matrix-multiplication-animation
npm install
npm run dev
```

#### ReLU Activation Animation

```bash
cd relu-animation
npm install
npm run dev
```

#### Leaky ReLU Activation Animation

```bash
cd leaky-relu-animation
npm install
npm run dev
```

#### Multi-Input Neural Network (Conv + ReLU)

```bash
cd conv-relu-animation
npm install
npm run dev
```

#### 2D Convolution Animation

```bash
cd conv2d-animation
npm install
npm run dev
```

#### SVD Animation

```bash
cd svd-animation
npm install
npm run dev
```

#### Eigenvalue Decomposition Animation

```bash
cd eigenvalue-animation
npm install
npm run dev
```

#### QR Decomposition Animation

```bash
cd qr-decomposition-animation
npm install
npm run dev
```

#### LSTM Animation

```bash
cd lstm-animation
npm install
npm run dev
```

#### Spearman Correlation Animation

```bash
cd spearman-correlation-animation
npm install
npm run dev
```

#### Bloom Filter Animation

```bash
cd bloom-filter-animation
npm install
npm run dev
```

#### Gradient Descent Animation

```bash
cd gradient-descent-animation
npm install
npm run dev
```

#### PageRank Animation

```bash
cd pagerank-animation
npm install
npm run dev
```

#### Linear Regression Animation

```bash
cd linear-regression-animation
npm install
npm run dev
```

#### Embeddings Animation

```bash
cd embeddings-animation
npm install
npm run dev
```

#### Cosine Similarity Animation

```bash
cd cosine-similarity-animation
npm install
npm run dev
```

#### Positional Encoding Animation

```bash
cd positional-encoding-animation
npm install
npm run dev
```

#### Layer Normalization Animation

```bash
cd layer-normalization-animation
npm install
npm run dev
```

#### Conditional Probability & Bayes' Theorem Animation

```bash
cd conditional-probability-animation
npm install
npm run dev
```

#### Probability Distributions Animation

```bash
cd probability-distributions-animation
npm install
npm run dev
```

#### Expected Value & Variance Animation

```bash
cd expected-value-variance-animation
npm install
npm run dev
```

#### Markov Chains Animation

```bash
cd markov-chains-animation
npm install
npm run dev
```

#### Self-Attention Animation

```bash
cd self-attention-animation
npm install
npm run dev
```

#### RL Foundations Animation (Part 1)

```bash
cd rl-foundations-animation
npm install
npm run dev
```

Open your browser at the URL shown in the terminal (usually `http://localhost:5173`).

## Technologies Used

- **React**: UI and state management.
- **Three.js**: 3D graphics rendering.
- **GSAP**: Smooth animations.
- **Vite**: Fast build tool and development server.
- **Tailwind CSS**: Styling.
