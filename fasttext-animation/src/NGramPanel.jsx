import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Layers, ArrowRight, Plus } from 'lucide-react';

function NGramPanel() {
  const [step, setStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  const word = "unhappy";
  const ngrams = ['<un', 'unh', 'nha', 'hap', 'app', 'ppy', 'py>'];
  
  // Simulated vector values (normalized for visualization)
  const ngramVectors = {
    '<un': [0.3, -0.2, 0.5, 0.1],
    'unh': [-0.1, 0.4, 0.2, -0.3],
    'nha': [0.2, 0.1, -0.4, 0.3],
    'hap': [0.5, 0.3, 0.2, 0.4],
    'app': [0.1, 0.2, 0.3, -0.1],
    'ppy': [0.4, 0.5, -0.2, 0.2],
    'py>': [-0.2, 0.1, 0.4, 0.1],
  };

  const sumVector = [1.2, 1.4, 1.0, 0.7]; // Sum of all vectors

  useEffect(() => {
    if (isAnimating && step < ngrams.length + 1) {
      const timer = setTimeout(() => {
        setStep(step + 1);
      }, 800);
      return () => clearTimeout(timer);
    } else if (step >= ngrams.length + 1) {
      setIsAnimating(false);
    }
  }, [step, isAnimating]);

  const startAnimation = () => {
    setStep(0);
    setIsAnimating(true);
  };

  const resetAnimation = () => {
    setStep(0);
    setIsAnimating(false);
  };

  return (
    <div className="space-y-6">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold text-white mb-2">N-gram Vector Composition</h2>
        <p className="text-purple-200/70">
          Watch how individual n-gram vectors combine to form the word vector
        </p>
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-4 mb-6">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={startAnimation}
          disabled={isAnimating}
          className={`px-6 py-2 rounded-lg font-medium ${
            isAnimating
              ? 'bg-white/10 text-white/50'
              : 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
          }`}
        >
          {step === 0 ? 'Start Animation' : 'Replay'}
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={resetAnimation}
          className="px-6 py-2 rounded-lg font-medium bg-white/10 text-white hover:bg-white/20"
        >
          Reset
        </motion.button>
      </div>

      {/* Main Visualization */}
      <div className="bg-white/5 rounded-xl p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* N-grams List */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Layers className="w-5 h-5 text-purple-400" />
              N-gram Vectors for "{word}"
            </h3>
            <div className="space-y-2">
              {ngrams.map((ngram, i) => (
                <motion.div
                  key={ngram}
                  initial={{ opacity: 0.3 }}
                  animate={{
                    opacity: step > i ? 1 : 0.3,
                    scale: step === i + 1 ? 1.05 : 1,
                    backgroundColor: step === i + 1 ? 'rgba(168, 85, 247, 0.3)' : 'transparent'
                  }}
                  className="flex items-center gap-3 p-2 rounded-lg"
                >
                  <span className="font-mono text-purple-300 w-12">{ngram}</span>
                  <ArrowRight className="w-4 h-4 text-purple-400" />
                  <div className="flex gap-1">
                    {ngramVectors[ngram].map((val, j) => (
                      <motion.div
                        key={j}
                        animate={{
                          backgroundColor: step > i 
                            ? val > 0 ? `rgba(34, 197, 94, ${Math.abs(val)})` : `rgba(239, 68, 68, ${Math.abs(val)})`
                            : 'rgba(255, 255, 255, 0.1)'
                        }}
                        className="w-10 h-8 rounded flex items-center justify-center text-xs text-white font-mono"
                      >
                        {val.toFixed(1)}
                      </motion.div>
                    ))}
                  </div>
                  {step > i && step <= ngrams.length && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                    >
                      <Plus className="w-4 h-4 text-pink-400" />
                    </motion.div>
                  )}
                </motion.div>
              ))}
            </div>
          </div>

          {/* Result Vector */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Resulting Word Vector</h3>
            
            <motion.div
              animate={{
                opacity: step > ngrams.length ? 1 : 0.3,
                scale: step > ngrams.length ? 1 : 0.95
              }}
              className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl p-6 border border-purple-400/30"
            >
              <div className="text-center mb-4">
                <span className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
                  v("{word}")
                </span>
              </div>

              <div className="flex justify-center gap-2 mb-4">
                {sumVector.map((val, i) => (
                  <motion.div
                    key={i}
                    initial={{ height: 0 }}
                    animate={{ 
                      height: step > ngrams.length ? Math.abs(val) * 60 : 0 
                    }}
                    transition={{ delay: i * 0.1, duration: 0.5 }}
                    className={`w-12 rounded-t-lg flex items-end justify-center pb-1 text-xs text-white font-mono ${
                      val > 0 ? 'bg-gradient-to-t from-green-500 to-emerald-400' : 'bg-gradient-to-t from-red-500 to-rose-400'
                    }`}
                    style={{ minHeight: step > ngrams.length ? 30 : 0 }}
                  >
                    {step > ngrams.length && val.toFixed(1)}
                  </motion.div>
                ))}
              </div>

              <div className="text-center text-purple-300/70 text-sm">
                {step > ngrams.length 
                  ? 'Sum of all n-gram vectors'
                  : 'Waiting for n-gram sum...'}
              </div>
            </motion.div>

            {/* Progress Indicator */}
            <div className="mt-6">
              <div className="flex justify-between text-sm text-purple-300/70 mb-2">
                <span>Progress</span>
                <span>{Math.min(step, ngrams.length)}/{ngrams.length} n-grams</span>
              </div>
              <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                <motion.div
                  animate={{ width: `${(Math.min(step, ngrams.length) / ngrams.length) * 100}%` }}
                  className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Explanation */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="bg-white/5 rounded-xl p-4"
      >
        <p className="text-purple-200/80 text-sm">
          <strong className="text-purple-300">Key Insight:</strong> By summing n-gram vectors, 
          FastText captures morphological patterns. Words like "unhappy", "unhelpful", and "unfortunate" 
          share the "&lt;un" n-gram, making their vectors similar in the "negation" dimension.
        </p>
      </motion.div>
    </div>
  );
}

export default NGramPanel;
