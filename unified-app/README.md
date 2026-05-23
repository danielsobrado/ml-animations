# ML Animations - Unified Application

A unified React application that consolidates the ML animation visualizations into a Distill-style educational experience with sidebar navigation, guided curriculum tracks, glossary-backed learning cards, and lazy-loaded animation modules.

## Features

- 🎨 **Distill-Style Design System** - Calm typography, equation strips, and paper-like educational panels
- 🧭 **Guided Curriculum Tracks** - Foundations, Core ML, Neural Networks, Experimentation & Causal ML, NLP to Transformers, Generative AI, and RL/Algorithms paths
- 🧠 **Learning Shell** - Prerequisites, estimated time, objectives, misconceptions, next steps, and glossary terms around each animation
- 📱 **Responsive Layout** - Works on desktop, tablet, and mobile
- 🗂️ **Sidebar Navigation** - Easy navigation through all animation categories
- ⚡ **Lazy Loading** - Animations load on demand for better performance
- 🎯 **Practice Panels** - Many modules include questions, sliders, and playground-style checks

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
│   │   ├── animation-shell/
│   │   │   └── AnimationShell.jsx # Curriculum shell around each animation
│   │   └── layout/
│   │       ├── Header.jsx   # Top navigation bar
│   │       └── Sidebar.jsx  # Side navigation
│   ├── data/
│   │   ├── animations.js    # Animation metadata, categories, tracks, backlog
│   │   ├── animationLearning.js # Learning-card and mindmap model
│   │   └── glossaryRepository.js # Glossary terms and generated images
│   ├── pages/
│   │   ├── HomePage.jsx     # Landing page with all categories
│   │   ├── AnimationPage.jsx # Individual animation view
│   │   └── GlossaryPage.jsx # Glossary term notes
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

### Step 2: Adapt Components For The Distill Shell

Prefer the shared shell and readable panel colors over module-specific full-page chrome:

```jsx
// Inside src/animations/your-animation/index.jsx
export default function YourAnimation() {
  return <YourPanel />;
}
```

When a panel uses a dark background, make labels and helper text high contrast:

```jsx
<div className="bg-slate-800 text-slate-100">
  <p className="text-slate-300">Readable explanatory text</p>
</div>
```

Avoid adding a second app header, sidebar, theme provider, or router inside an animation module. The unified app already supplies those.

### Step 3: Register the Animation

Add your animation to `src/animations/index.js`:

```javascript
const animationRegistry = {
  'attention-mechanism': lazy(() => import('./attention-mechanism')),
  'your-animation': lazy(() => import('./your-animation')), // Add this
};
```

### Step 4: Update Animation Metadata

The animation should already be listed in `src/data/animations.js`. If not, add it with curriculum metadata:

```javascript
{
  id: 'your-animation',
  name: 'Your Animation',
  icon: YourIcon,
  description: 'Brief description',
  difficulty: 'intermediate',
  prerequisites: ['matrix-multiplication'],
  estimatedMinutes: 15,
  learningObjectives: ['Explain the main idea', 'Predict the animation output'],
  commonMisconception: 'One common simplification to watch for',
  trackIds: ['foundations']
}
```

## Design Notes

- The app uses a static Distill-style paper theme, not a dark/light theme toggle.
- Keep formulas aligned with the actual animation math.
- Use `AnimationShell` metadata for prerequisites, estimated time, glossary terms, and next recommendations.
- Keep category routes stable; guided tracks are an additional curriculum layer.

## Categories

The animations are organized into the following categories:

1. **Natural Language Processing** - Bag of Words, Word2Vec, GloVe, etc.
2. **Transformers & Attention** - Attention, BERT, Transformers, etc.
3. **Neural Networks** - ReLU, Softmax, LSTM, Conv2D, etc.
4. **Advanced Models** - VAE, RAG, Multimodal LLM
5. **Math Fundamentals** - Matrix ops, SVD, Gradient Descent, etc.
6. **Model Reliability** - Debugging, interpretability, monitoring, fairness, and uncertainty
7. **Probability & Statistics** - Distributions, entropy, testing, likelihoods, and intervals
8. **Experimentation & Causal ML** - A/B testing, power analysis, peeking, CUPED, confounding, DAGs, treatment effects, and propensity scores
9. **Reinforcement Learning** - Q-Learning, Markov Chains, etc.
10. **Algorithms** - Bloom Filter, PageRank

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
