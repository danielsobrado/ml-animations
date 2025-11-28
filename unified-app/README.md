# ML Animations - Unified Application

A professional, unified React application that consolidates all ML animation visualizations into a single cohesive experience with a sidebar navigation, consistent theming, and dark/light mode toggle.

## Features

- 🎨 **Unified Design System** - Consistent styling across all animations
- 🌓 **Dark/Light Mode** - Toggle between themes with system preference detection
- 📱 **Responsive Layout** - Works on desktop, tablet, and mobile
- 🗂️ **Sidebar Navigation** - Easy navigation through all animation categories
- ⚡ **Lazy Loading** - Animations load on demand for better performance
- 🎯 **Progress Tracking** - Navigate through multi-step animations

## Project Structure

```
unified-app/
├── public/
│   └── favicon.svg
├── src/
│   ├── animations/          # Animation modules
│   │   ├── index.js         # Animation registry
│   │   └── attention-mechanism/
│   │       ├── index.jsx    # Main animation component
│   │       └── IntuitionPanel.jsx
│   ├── components/
│   │   └── layout/
│   │       ├── Header.jsx   # Top navigation bar
│   │       └── Sidebar.jsx  # Side navigation
│   ├── context/
│   │   └── ThemeContext.jsx # Dark/light mode context
│   ├── data/
│   │   └── animations.js    # Animation metadata & categories
│   ├── pages/
│   │   ├── HomePage.jsx     # Landing page with all categories
│   │   └── AnimationPage.jsx # Individual animation view
│   ├── App.jsx              # Main app with routing
│   ├── main.jsx             # Entry point
│   └── index.css            # Global styles with Tailwind
├── index.html
├── package.json
├── tailwind.config.js
├── postcss.config.js
└── vite.config.js
```

## Getting Started

### Installation

```bash
cd unified-app
npm install
```

### Development

```bash
npm run dev
```

### Build for Production

```bash
npm run build
npm run preview
```

## Integrating New Animations

### Step 1: Create Animation Folder

Create a new folder in `src/animations/` with your animation ID:

```
src/animations/your-animation/
├── index.jsx          # Main component (default export)
├── Panel1.jsx         # Sub-panels
├── Panel2.jsx
└── ...
```

### Step 2: Adapt Components for Theme Support

Replace hardcoded dark theme colors with theme-aware classes:

```jsx
// Before (hardcoded dark)
<div className="bg-slate-800 text-white">

// After (theme-aware)
<div className="bg-white dark:bg-slate-800 text-slate-900 dark:text-white">
```

Use the provided utility classes:
- `card` - Theme-aware card container
- `btn-primary` - Primary gradient button
- `btn-secondary` - Secondary button
- `text-gradient` - Gradient text effect

### Step 3: Register the Animation

Add your animation to `src/animations/index.js`:

```javascript
const animationRegistry = {
  'attention-mechanism': lazy(() => import('./attention-mechanism')),
  'your-animation': lazy(() => import('./your-animation')), // Add this
};
```

### Step 4: Update Animation Metadata

The animation should already be listed in `src/data/animations.js`. If not, add it:

```javascript
{
  id: 'your-animation',
  name: 'Your Animation',
  icon: YourIcon,
  description: 'Brief description'
}
```

## Theme Classes Reference

### Colors (Light / Dark)

| Purpose | Light | Dark |
|---------|-------|------|
| Background | `bg-slate-50` | `dark:bg-slate-900` |
| Surface | `bg-white` | `dark:bg-slate-800` |
| Border | `border-slate-200` | `dark:border-slate-700` |
| Text Primary | `text-slate-900` | `dark:text-white` |
| Text Secondary | `text-slate-600` | `dark:text-slate-400` |

### Component Classes

```css
.card          /* Themed card container with border and shadow */
.btn-primary   /* Gradient primary button */
.btn-secondary /* Neutral secondary button */
.sidebar-item  /* Sidebar navigation item */
.text-gradient /* Blue-to-purple gradient text */
```

## Categories

The animations are organized into the following categories:

1. **Natural Language Processing** - Bag of Words, Word2Vec, GloVe, etc.
2. **Transformers & Attention** - Attention, BERT, Transformers, etc.
3. **Neural Networks** - ReLU, Softmax, LSTM, Conv2D, etc.
4. **Advanced Models** - VAE, RAG, Multimodal LLM
5. **Math Fundamentals** - Matrix ops, SVD, Gradient Descent, etc.
6. **Probability & Statistics** - Distributions, Entropy, etc.
7. **Reinforcement Learning** - Q-Learning, Markov Chains, etc.
8. **Algorithms** - Bloom Filter, PageRank

## Technology Stack

- **React 18** - UI framework
- **React Router 6** - Client-side routing
- **Tailwind CSS 3** - Utility-first CSS
- **Vite 5** - Build tool
- **Lucide React** - Icons
- **GSAP** - Animations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Integrate your animation following the steps above
4. Submit a pull request

## License

MIT License - See LICENSE file for details
