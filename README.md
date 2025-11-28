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

Open your browser at the URL shown in the terminal (usually `http://localhost:5173`).

## Technologies Used

- **React**: UI and state management.
- **Three.js**: 3D graphics rendering.
- **GSAP**: Smooth animations.
- **Vite**: Fast build tool and development server.
- **Tailwind CSS**: Styling.
