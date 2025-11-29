# GPT-2 Comprehensive Animation

An interactive, multi-step educational journey through GPT-2 architecture and training optimizations.

## Overview

This project provides a deep dive into GPT-2, covering:

### Implemented Steps (Phase 1)
1. **Tokenization & Embeddings**: BPE tokenization, learned embeddings
2. **Positional Encoding**: Learned absolute position embeddings  
3. **Multi-Head Self-Attention**: Q/K/V projections, causal masking, multi-head mechanism

### Coming Soon (Phase 2)
4. Feed-Forward Network
5. Layer Normalization & Residual Connections
6. Full Architecture Overview
7. Weight Tying Optimization
8. Training Optimizations (gradient accumulation, mixed precision, etc.)
9. Inference Optimizations (KV cache, sampling strategies)

## Features

- **Interactive Visualizations**: Three.js for 3D attention patterns, Canvas for positional encoding
- **Hands-on Exercises**: Quiz questions to validate understanding after each step
- **Progress Tracking**: Navigate through steps, mark completion
- **Dark Theme**: Easy on the eyes during long study sessions

## Getting Started

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

## Learning Objectives

After completing all steps, you will understand:
- How text is converted to numerical representations
- The self-attention mechanism that powers transformers
- Why GPT-2 uses causal masking for autoregressive generation
- Training and inference optimizations that make GPT-2 practical

## Project Structure

```
src/
├── App.jsx                 # Main app with navigation
├── stepsConfig.js          # Step definitions
└── steps/
    ├── Step1Tokenization.jsx
    ├── Step2Positional.jsx
    └── Step3Attention.jsx
```

## Extending This Project

To add Step 4 and beyond:
1. Create new component in `src/steps/`
2. Add route in `App.jsx`
3. Update `stepsConfig.js`

## License

MIT
