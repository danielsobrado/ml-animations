# Max Pooling Animation

A visual demonstration of Max Pooling, built with React, Three.js, and Tailwind CSS.

## Overview

This project visualizes how Max Pooling downsamples feature maps in Convolutional Neural Networks (CNNs). It shows:

- **Sliding Window**: A window that moves across the input feature map
- **Max Selection**: Highlighting the maximum value in each window region
- **Output Creation**: Building the downsampled output feature map
- **Dimension Reduction**: Visualizing how pooling reduces spatial dimensions

## Components

- **Max Pooling Panel**: Animated visualization with sliding window and max value highlighting
- **Config Panel**: Interactive controls for pool size, stride, and input randomization

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

## Features

- **Configurable Pool Size**: Try 2×2 or 3×3 pooling windows
- **Adjustable Stride**: Test with stride 1 (overlapping) or stride 2 (non-overlapping)
- **Random Input**: Generate different feature maps to see how pooling behaves

## Why Max Pooling?

- Reduces spatial dimensions (downsampling)
- Decreases computational cost
- Provides translation invariance
- Keeps the strongest features (maximum values)

## License

MIT
