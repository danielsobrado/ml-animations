import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight, Home, Zap, Puzzle, Target, GitCompare, Code, Dumbbell } from 'lucide-react';
import IntroPanel from './IntroPanel';
import SubwordPanel from './SubwordPanel';
import NGramPanel from './NGramPanel';
import OOVPanel from './OOVPanel';
import ComparisonPanel from './ComparisonPanel';
import CodePanel from './CodePanel';
import PracticePanel from './PracticePanel';

const panels = [
  { id: 0, title: 'Introduction', icon: Home, component: IntroPanel },
  { id: 1, title: 'Subword Embeddings', icon: Puzzle, component: SubwordPanel },
  { id: 2, title: 'N-gram Vectors', icon: Zap, component: NGramPanel },
  { id: 3, title: 'OOV Handling', icon: Target, component: OOVPanel },
  { id: 4, title: 'Comparison', icon: GitCompare, component: ComparisonPanel },
  { id: 5, title: 'Code', icon: Code, component: CodePanel },
  { id: 6, title: 'Practice', icon: Dumbbell, component: PracticePanel },
];

function App() {
  const [currentPanel, setCurrentPanel] = useState(0);

  const goNext = () => setCurrentPanel((prev) => Math.min(prev + 1, panels.length - 1));
  const goPrev = () => setCurrentPanel((prev) => Math.max(prev - 1, 0));

  const CurrentComponent = panels[currentPanel].component;

  return (
    <div className="min-h-screen p-4 md:p-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-6"
      >
        <h1 className="text-4xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-pink-400 to-orange-400 mb-2">
          FastText
        </h1>
        <p className="text-purple-200/80 text-lg">Subword Embeddings & Morphological Awareness</p>
      </motion.div>

      {/* Navigation Dots */}
      <div className="flex justify-center gap-2 mb-6">
        {panels.map((panel) => (
          <motion.button
            key={panel.id}
            onClick={() => setCurrentPanel(panel.id)}
            whileHover={{ scale: 1.2 }}
            whileTap={{ scale: 0.9 }}
            className={`w-3 h-3 rounded-full transition-all duration-300 ${
              currentPanel === panel.id
                ? 'bg-gradient-to-r from-purple-400 to-pink-400 scale-125'
                : 'bg-white/30 hover:bg-white/50'
            }`}
            title={panel.title}
          />
        ))}
      </div>

      {/* Panel Title */}
      <motion.div
        key={currentPanel}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex items-center justify-center gap-2 mb-4"
      >
        {React.createElement(panels[currentPanel].icon, { className: 'w-5 h-5 text-purple-400' })}
        <span className="text-purple-200 font-medium">{panels[currentPanel].title}</span>
        <span className="text-purple-400/60 text-sm">({currentPanel + 1}/{panels.length})</span>
      </motion.div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentPanel}
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -50 }}
            transition={{ duration: 0.3 }}
            className="glass-panel p-6 md:p-8 min-h-[500px]"
          >
            <CurrentComponent />
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Navigation Buttons */}
      <div className="flex justify-center gap-4 mt-6">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={goPrev}
          disabled={currentPanel === 0}
          className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all ${
            currentPanel === 0
              ? 'bg-white/10 text-white/30 cursor-not-allowed'
              : 'bg-white/20 text-white hover:bg-white/30'
          }`}
        >
          <ChevronLeft className="w-5 h-5" />
          Previous
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={goNext}
          disabled={currentPanel === panels.length - 1}
          className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all ${
            currentPanel === panels.length - 1
              ? 'bg-white/10 text-white/30 cursor-not-allowed'
              : 'bg-gradient-to-r from-purple-500 to-pink-500 text-white hover:from-purple-600 hover:to-pink-600'
          }`}
        >
          Next
          <ChevronRight className="w-5 h-5" />
        </motion.button>
      </div>
    </div>
  );
}

export default App;
