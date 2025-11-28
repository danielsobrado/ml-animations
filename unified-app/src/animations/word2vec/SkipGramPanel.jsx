import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Target, ArrowRight } from 'lucide-react';

export default function SkipGramPanel() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [windowSize, setWindowSize] = useState(2);
  const [showTrainingPairs, setShowTrainingPairs] = useState(false);

  const sentence = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"];
  const centerIndex = 4; // "jumps"

  const steps = [
    { title: 'Input Sentence', description: 'Start with a sentence from your corpus' },
    { title: 'Choose Center Word', description: 'Select a word to be the center (input) word' },
    { title: 'Define Context Window', description: 'Words within window size are context words' },
    { title: 'Create Training Pairs', description: 'Each (center, context) pair is a training example' },
    { title: 'Train Neural Network', description: 'Predict context words from center word' },
    { title: 'Extract Embeddings', description: 'The learned weights become word vectors' },
  ];

  useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= steps.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 2500);
      return () => clearInterval(interval);
    }
  }, [isPlaying]);

  const reset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
    setShowTrainingPairs(false);
  };

  // Get context words based on window size
  const getContextIndices = () => {
    const indices = [];
    for (let i = Math.max(0, centerIndex - windowSize); i <= Math.min(sentence.length - 1, centerIndex + windowSize); i++) {
      if (i !== centerIndex) indices.push(i);
    }
    return indices;
  };

  const contextIndices = getContextIndices();

  // Generate training pairs for animation
  const trainingPairs = contextIndices.map(i => ({
    center: sentence[centerIndex],
    context: sentence[i],
    position: i < centerIndex ? 'left' : 'right'
  }));

  return (
    <div className="space-y-6 pb-20">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="text-green-400">Skip-gram</span>: Predict Context from Center
        </h2>
        <p className="text-gray-400">
          Given a word, predict the surrounding words
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap justify-center gap-3">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
        >
          {isPlaying ? <Pause size={18} /> : <Play size={18} />}
          {isPlaying ? 'Pause' : 'Play Animation'}
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
        >
          <RotateCcw size={18} />
          Reset
        </button>
        <div className="flex items-center gap-2 px-4 py-2 bg-white/5 rounded-lg">
          <span className="text-sm text-gray-400">Window Size:</span>
          <select
            value={windowSize}
            onChange={(e) => setWindowSize(parseInt(e.target.value))}
            className="bg-gray-700 rounded px-2 py-1 text-sm"
          >
            {[1, 2, 3, 4].map(n => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Step Progress */}
      <div className="flex flex-wrap justify-center gap-2">
        {steps.map((step, i) => (
          <button
            key={i}
            onClick={() => { setCurrentStep(i); setIsPlaying(false); }}
            className={`px-3 py-1 rounded-full text-sm transition-all ${
              i === currentStep 
                ? 'bg-green-500 text-black scale-110' 
                : i < currentStep 
                ? 'bg-green-900 text-green-300' 
                : 'bg-white/10 text-gray-500'
            }`}
          >
            {i + 1}. {step.title}
          </button>
        ))}
      </div>

      {/* Current Step Info */}
      <div className="bg-green-900/20 border border-green-500/30 rounded-xl p-4">
        <h3 className="font-bold text-green-400">Step {currentStep + 1}: {steps[currentStep].title}</h3>
        <p className="text-gray-300 mt-1">{steps[currentStep].description}</p>
      </div>

      {/* Main Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        {/* Sentence Display */}
        <div className="flex flex-wrap justify-center gap-2 mb-6">
          {sentence.map((word, i) => {
            const isCenter = i === centerIndex && currentStep >= 1;
            const isContext = contextIndices.includes(i) && currentStep >= 2;
            const isInWindow = Math.abs(i - centerIndex) <= windowSize && i !== centerIndex;
            
            return (
              <div
                key={i}
                className={`px-4 py-2 rounded-lg font-mono text-lg transition-all duration-300 ${
                  isCenter
                    ? 'bg-green-500 text-black scale-110 shadow-lg shadow-green-500/50'
                    : isContext
                    ? 'bg-blue-500 text-white scale-105 shadow-lg shadow-blue-500/50'
                    : currentStep >= 2 && isInWindow
                    ? 'bg-blue-900/50 border border-blue-500/50'
                    : 'bg-white/10 text-gray-400'
                }`}
              >
                {word}
              </div>
            );
          })}
        </div>

        {/* Window Visualization */}
        {currentStep >= 2 && (
          <div className="text-center mb-6 animate-fadeIn">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/5 rounded-lg">
              <span className="text-gray-400">Window size:</span>
              <span className="text-green-400 font-bold">{windowSize}</span>
              <span className="text-gray-400">→</span>
              <span className="text-blue-400">{contextIndices.length} context words</span>
            </div>
          </div>
        )}

        {/* Training Pairs */}
        {currentStep >= 3 && (
          <div className="space-y-4 animate-fadeIn">
            <h4 className="text-center text-gray-400 mb-2">Training Pairs (center → context):</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {trainingPairs.map((pair, i) => (
                <div
                  key={i}
                  className="bg-gradient-to-r from-green-900/30 to-blue-900/30 rounded-lg p-3 border border-white/10 animate-fadeIn"
                  style={{ animationDelay: `${i * 100}ms` }}
                >
                  <div className="flex items-center justify-center gap-2 font-mono">
                    <span className="text-green-400">{pair.center}</span>
                    <ArrowRight size={14} className="text-gray-500" />
                    <span className="text-blue-400">{pair.context}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Neural Network Visualization */}
        {currentStep >= 4 && (
          <div className="mt-8 animate-fadeIn">
            <h4 className="text-center text-gray-400 mb-4">Skip-gram Neural Network:</h4>
            <div className="flex justify-center items-center gap-4">
              {/* Input */}
              <div className="text-center">
                <div className="w-20 h-20 rounded-lg bg-green-900/50 border border-green-500 flex items-center justify-center">
                  <div className="font-mono text-green-400">jumps</div>
                </div>
                <div className="text-xs text-gray-500 mt-2">Input<br/>(one-hot)</div>
              </div>

              <ArrowRight className="text-gray-500" />

              {/* Hidden Layer */}
              <div className="text-center">
                <div className="w-24 h-20 rounded-lg bg-purple-900/50 border border-purple-500 flex flex-col items-center justify-center px-2">
                  <div className="text-xs text-purple-300 mb-1">W (V×D)</div>
                  <div className="text-xs text-gray-400">Embedding</div>
                </div>
                <div className="text-xs text-gray-500 mt-2">Hidden Layer<br/>(D dims)</div>
              </div>

              <ArrowRight className="text-gray-500" />

              {/* Output */}
              <div className="text-center">
                <div className="w-20 h-20 rounded-lg bg-blue-900/50 border border-blue-500 flex flex-col items-center justify-center">
                  <div className="font-mono text-xs text-blue-400">quick</div>
                  <div className="font-mono text-xs text-blue-400">brown</div>
                  <div className="font-mono text-xs text-blue-400">over</div>
                  <div className="font-mono text-xs text-blue-400">the</div>
                </div>
                <div className="text-xs text-gray-500 mt-2">Output<br/>(softmax)</div>
              </div>
            </div>
          </div>
        )}

        {/* Embedding Extraction */}
        {currentStep >= 5 && (
          <div className="mt-8 bg-gradient-to-r from-purple-900/30 to-pink-900/30 rounded-xl p-6 border border-purple-500/30 animate-fadeIn">
            <h4 className="text-purple-400 font-bold mb-3">🎯 The Word Embeddings</h4>
            <p className="text-gray-300 text-sm mb-4">
              After training, the weight matrix W becomes our word embedding matrix.
              Each row is the embedding vector for a word.
            </p>
            <div className="font-mono text-sm bg-black/30 rounded-lg p-4 overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr>
                    <th className="text-left text-gray-500 pr-4">Word</th>
                    <th className="text-gray-500" colSpan="4">Embedding (D=4 for demo)</th>
                  </tr>
                </thead>
                <tbody>
                  {['the', 'quick', 'brown', 'fox', 'jumps'].map((word, i) => (
                    <tr key={i}>
                      <td className="text-green-400 pr-4">{word}</td>
                      <td className="text-purple-300 px-2">[{(Math.random() * 2 - 1).toFixed(2)},</td>
                      <td className="text-purple-300 px-2">{(Math.random() * 2 - 1).toFixed(2)},</td>
                      <td className="text-purple-300 px-2">{(Math.random() * 2 - 1).toFixed(2)},</td>
                      <td className="text-purple-300 px-2">{(Math.random() * 2 - 1).toFixed(2)}]</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Skip-gram Formula */}
      <div className="bg-black/40 rounded-xl p-6 border border-white/10">
        <h4 className="font-bold text-white mb-4">📐 Skip-gram Objective</h4>
        <div className="bg-black/50 rounded-lg p-4 text-center font-mono">
          <p className="text-green-300">
            maximize Σ log P(context | center)
          </p>
          <p className="text-sm text-gray-400 mt-2">
            P(context | center) = softmax(W'<sub>context</sub> · W<sub>center</sub>)
          </p>
        </div>
        <p className="text-gray-400 text-sm mt-4">
          We want to maximize the probability of seeing actual context words given the center word.
        </p>
      </div>

      {/* Key Points */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-green-900/20 rounded-xl p-4 border border-green-500/30">
          <h4 className="font-bold text-green-400 mb-2">✅ Skip-gram Strengths</h4>
          <ul className="text-sm text-gray-300 space-y-1">
            <li>• Works well with small datasets</li>
            <li>• Better representation for rare words</li>
            <li>• Captures subtle semantic relationships</li>
          </ul>
        </div>
        <div className="bg-red-900/20 rounded-xl p-4 border border-red-500/30">
          <h4 className="font-bold text-red-400 mb-2">❌ Skip-gram Weaknesses</h4>
          <ul className="text-sm text-gray-300 space-y-1">
            <li>• Slower to train (more predictions per word)</li>
            <li>• Original softmax is expensive (vocabulary size)</li>
            <li>• Requires negative sampling for efficiency</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
