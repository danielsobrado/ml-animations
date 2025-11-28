# Cross-Entropy Loss Animation

A visual demonstration of the Cross-Entropy Loss function, built with React, Three.js, and Tailwind CSS.

## Overview

This project visualizes how Cross-Entropy Loss is calculated for classification models, specifically focusing on:

- **One-Hot Encoding**: Visualizing true labels as vectors.
- **Log Loss Curve**: Showing why we use $-\ln(p)$ to penalize wrong predictions.
- **Error Calculation**: Demonstrating that we only care about the probability of the *correct* class.

## Components

- **Animation Panel**: Step-by-step visualization of the calculation.
- **Graph Panel**: Interactive plot of the Log Loss curve ($y = -\ln(x)$).
- **Practice Panel**: Slider to adjust probabilities and see the loss explode as $p \to 0$.

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Run the development server:
   ```bash
   npm run dev
   ```

3. Open your browser to the URL shown (usually `http://localhost:5173`).

## License

MIT
