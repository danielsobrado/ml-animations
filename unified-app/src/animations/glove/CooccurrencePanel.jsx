import React, { useState, useEffect, useRef } from 'react';
import { Play, RotateCcw, Grid3X3, ArrowRight } from 'lucide-react';
import gsap from 'gsap';

export default function CooccurrencePanel() {
  const [step, setStep] = useState(1);
  const [windowSize, setWindowSize] = useState(2);
  const [isPlaying, setIsPlaying] = useState(false);
  const animationRef = useRef(null);

  const corpus = [
    "I like deep learning",
    "I like NLP",
    "I enjoy machine learning"
  ];

  const words = ['i', 'like', 'enjoy', 'deep', 'learning', 'nlp', 'machine'];
  
  // Simple co-occurrence counts based on corpus
  const getCooccurrence = () => {
    const matrix = {};
    words.forEach(w1 => {
      matrix[w1] = {};
      words.forEach(w2 => {
        matrix[w1][w2] = 0;
      });
    });
    
    // Count co-occurrences
    const sentences = [
      ['i', 'like', 'deep', 'learning'],
      ['i', 'like', 'nlp'],
      ['i', 'enjoy', 'machine', 'learning']
    ];
    
    sentences.forEach(sent => {
      for (let i = 0; i < sent.length; i++) {
        for (let j = Math.max(0, i - windowSize); j <= Math.min(sent.length - 1, i + windowSize); j++) {
          if (i !== j) {
            matrix[sent[i]][sent[j]] += 1;
          }
        }
      }
    });
    
    return matrix;
  };

  const cooccurrence = getCooccurrence();

  const steps = [
    {
      title: 'Step 1: Define Corpus',
      description: 'Start with a text corpus. Each sentence contains words we want to embed.'
    },
    {
      title: 'Step 2: Set Context Window',
      description: `With window size ${windowSize}, we consider ${windowSize} words on each side of the center word.`
    },
    {
      title: 'Step 3: Count Co-occurrences',
      description: 'For each word, count how many times other words appear within the context window.'
    },
    {
      title: 'Step 4: Build Matrix X',
      description: 'The co-occurrence matrix X where X_ij = count of word j appearing in context of word i.'
    },
    {
      title: 'Step 5: Compute Probabilities',
      description: 'Convert counts to probabilities: P(j|i) = X_ij / Σ_k X_ik'
    }
  ];

  const playAnimation = () => {
    setIsPlaying(true);
    let currentStep = 1;
    
    const interval = setInterval(() => {
      currentStep++;
      if (currentStep > steps.length) {
        setIsPlaying(false);
        clearInterval(interval);
      } else {
        setStep(currentStep);
      }
    }, 2000);
    
    animationRef.current = interval;
  };

  const reset = () => {
    if (animationRef.current) {
      clearInterval(animationRef.current);
    }
    setStep(1);
    setIsPlaying(false);
  };

  useEffect(() => {
    return () => {
      if (animationRef.current) {
        clearInterval(animationRef.current);
      }
    };
  }, []);

  const getCellColor = (count) => {
    if (count === 0) return 'bg-gray-800';
    if (count === 1) return 'bg-violet-900/50';
    if (count === 2) return 'bg-violet-700/60';
    return 'bg-violet-500/70';
  };

  return (
    <div className="space-y-6 pb-20">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="gradient-text">Co-occurrence Matrix:</span> Counting Word Neighbors
        </h2>
        <p className="text-gray-400">
          Building the foundation for GloVe word vectors
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center justify-center gap-4">
        <button
          onClick={playAnimation}
          disabled={isPlaying}
          className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-700 disabled:opacity-50 rounded-lg transition-colors"
        >
          <Play size={18} />
          Play Animation
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
        >
          <RotateCcw size={18} />
          Reset
        </button>
        <div className="flex items-center gap-2">
          <span className="text-gray-400 text-sm">Window Size:</span>
          <select
            value={windowSize}
            onChange={(e) => setWindowSize(parseInt(e.target.value))}
            className="bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-sm"
          >
            {[1, 2, 3].map(n => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Step Buttons */}
      <div className="flex flex-wrap justify-center gap-2">
        {steps.map((s, i) => (
          <button
            key={i}
            onClick={() => setStep(i + 1)}
            className={`px-3 py-1 rounded-full text-sm transition-all ${
              step === i + 1
                ? 'bg-violet-600 text-white'
                : 'bg-white/10 text-gray-400 hover:bg-white/20'
            }`}
          >
            {i + 1}
          </button>
        ))}
      </div>

      {/* Current Step Info */}
      <div className="bg-gradient-to-r from-violet-900/30 to-cyan-900/30 rounded-xl p-4 border border-violet-500/30">
        <h3 className="text-lg font-bold text-violet-400 mb-1">{steps[step - 1].title}</h3>
        <p className="text-gray-300">{steps[step - 1].description}</p>
      </div>

      {/* Visualization */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Corpus */}
        <div className={`bg-black/30 rounded-xl p-6 border border-white/10 transition-opacity ${step >= 1 ? 'opacity-100' : 'opacity-30'}`}>
          <h4 className="text-lg font-bold text-violet-400 mb-4 flex items-center gap-2">
            <Grid3X3 size={20} />
            Sample Corpus
          </h4>
          <div className="space-y-3">
            {corpus.map((sentence, i) => (
              <div key={i} className="bg-black/30 rounded-lg p-3">
                <span className="text-gray-500 text-sm">Sentence {i + 1}:</span>
                <p className="text-gray-200 font-mono">{sentence}</p>
              </div>
            ))}
          </div>
          
          {step >= 2 && (
            <div className="mt-4 p-3 bg-cyan-900/20 rounded-lg border border-cyan-500/30">
              <p className="text-sm text-cyan-400">
                <strong>Context Window = {windowSize}</strong>
              </p>
              <p className="text-xs text-gray-400 mt-1">
                For each word, consider {windowSize} word(s) on each side
              </p>
            </div>
          )}
        </div>

        {/* Co-occurrence Matrix */}
        <div className={`bg-black/30 rounded-xl p-6 border border-white/10 transition-opacity ${step >= 4 ? 'opacity-100' : 'opacity-30'}`}>
          <h4 className="text-lg font-bold text-cyan-400 mb-4">
            Co-occurrence Matrix X
          </h4>
          <div className="overflow-x-auto">
            <table className="text-xs">
              <thead>
                <tr>
                  <th className="p-1"></th>
                  {words.map(w => (
                    <th key={w} className="p-1 text-violet-400 font-mono">{w}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {words.map(w1 => (
                  <tr key={w1}>
                    <td className="p-1 text-cyan-400 font-mono">{w1}</td>
                    {words.map(w2 => (
                      <td 
                        key={w2}
                        className={`p-1 text-center ${getCellColor(cooccurrence[w1][w2])} rounded matrix-cell`}
                      >
                        <span className={cooccurrence[w1][w2] > 0 ? 'text-white' : 'text-gray-600'}>
                          {cooccurrence[w1][w2]}
                        </span>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          <div className="mt-4 flex items-center gap-2 text-xs">
            <span className="text-gray-400">Intensity:</span>
            <div className="flex gap-1">
              <div className="w-4 h-4 rounded bg-gray-800"></div>
              <div className="w-4 h-4 rounded bg-violet-900/50"></div>
              <div className="w-4 h-4 rounded bg-violet-700/60"></div>
              <div className="w-4 h-4 rounded bg-violet-500/70"></div>
            </div>
            <span className="text-gray-400">0 → High</span>
          </div>
        </div>
      </div>

      {/* Context Window Example */}
      {step >= 3 && (
        <div className="bg-black/30 rounded-xl p-6 border border-white/10">
          <h4 className="text-lg font-bold text-violet-400 mb-4">
            Context Window Example
          </h4>
          <div className="flex flex-wrap items-center justify-center gap-2">
            {['I', 'like', 'deep', 'learning'].map((word, i) => (
              <span
                key={i}
                className={`px-4 py-2 rounded-lg font-mono text-lg ${
                  i === 2 
                    ? 'bg-violet-600 text-white ring-2 ring-violet-400' 
                    : i >= 2 - windowSize && i <= 2 + windowSize && i !== 2
                    ? 'bg-cyan-900/50 text-cyan-400 border border-cyan-500/50'
                    : 'bg-white/10 text-gray-400'
                }`}
              >
                {word}
              </span>
            ))}
          </div>
          <div className="text-center mt-4">
            <p className="text-sm text-gray-400">
              Center word: <span className="text-violet-400 font-mono">"deep"</span> | 
              Context words: <span className="text-cyan-400 font-mono">{windowSize === 1 ? '"like", "learning"' : windowSize === 2 ? '"I", "like", "learning"' : '"I", "like", "learning"'}</span>
            </p>
          </div>
        </div>
      )}

      {/* Probability Computation */}
      {step >= 5 && (
        <div className="bg-gradient-to-r from-violet-900/20 to-cyan-900/20 rounded-xl p-6 border border-violet-500/30">
          <h4 className="text-lg font-bold text-violet-400 mb-4">
            📐 Computing Probabilities
          </h4>
          <div className="bg-black/30 rounded-lg p-4 font-mono text-center">
            <p className="text-xl text-gray-200">
              P(j | i) = X<sub className="text-violet-400">ij</sub> / X<sub className="text-cyan-400">i</sub>
            </p>
            <p className="text-sm text-gray-400 mt-2">
              where X<sub className="text-cyan-400">i</sub> = Σ<sub>k</sub> X<sub>ik</sub> (row sum)
            </p>
          </div>
          
          <div className="mt-4 grid md:grid-cols-2 gap-4">
            <div className="bg-black/30 rounded-lg p-3">
              <p className="text-sm text-violet-400 font-medium">Example: P(deep | like)</p>
              <p className="text-xs text-gray-400 font-mono mt-1">
                = X<sub>like,deep</sub> / X<sub>like</sub> = count(like,deep) / Σ count(like,*)
              </p>
            </div>
            <div className="bg-black/30 rounded-lg p-3">
              <p className="text-sm text-cyan-400 font-medium">Why Probabilities?</p>
              <p className="text-xs text-gray-400 mt-1">
                Ratios P(k|i)/P(k|j) encode semantic relationships
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Key Points */}
      <div className="bg-black/30 rounded-xl p-6 border border-white/10">
        <h4 className="text-lg font-bold text-violet-400 mb-4">💡 Key Points</h4>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-violet-900/20 rounded-lg p-3">
            <p className="text-violet-400 font-medium mb-1">Symmetric Counting</p>
            <p className="text-sm text-gray-400">
              If "dog" appears near "cat", then "cat" also appears near "dog" (X_ij = X_ji for symmetric windows)
            </p>
          </div>
          <div className="bg-cyan-900/20 rounded-lg p-3">
            <p className="text-cyan-400 font-medium mb-1">Window Size Matters</p>
            <p className="text-sm text-gray-400">
              Smaller windows capture syntactic info, larger windows capture semantic/topical info
            </p>
          </div>
          <div className="bg-green-900/20 rounded-lg p-3">
            <p className="text-green-400 font-medium mb-1">Sparse Matrix</p>
            <p className="text-sm text-gray-400">
              Most word pairs never co-occur, so the matrix is very sparse (lots of zeros)
            </p>
          </div>
          <div className="bg-yellow-900/20 rounded-lg p-3">
            <p className="text-yellow-400 font-medium mb-1">One-time Computation</p>
            <p className="text-sm text-gray-400">
              Unlike Word2Vec, we build the matrix once before training (not during)
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
