import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Shuffle, ArrowRight } from 'lucide-react';

export default function CbowPanel() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [windowSize, setWindowSize] = useState(2);

  const sentence = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"];
  const centerIndex = 4; // "jumps" - the word we're predicting

  const steps = [
    { title: 'Input Sentence', description: 'Start with a sentence from your corpus' },
    { title: 'Select Target Word', description: 'This is the word we want to predict' },
    { title: 'Identify Context Words', description: 'Words within window size are context (input)' },
    { title: 'Average Context Vectors', description: 'Combine context embeddings by averaging' },
    { title: 'Predict Target', description: 'Use averaged vector to predict center word' },
    { title: 'Learn Embeddings', description: 'Update weights based on prediction error' },
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
  };

  // Get context word indices
  const getContextIndices = () => {
    const indices = [];
    for (let i = Math.max(0, centerIndex - windowSize); i <= Math.min(sentence.length - 1, centerIndex + windowSize); i++) {
      if (i !== centerIndex) indices.push(i);
    }
    return indices;
  };

  const contextIndices = getContextIndices();
  const contextWords = contextIndices.map(i => sentence[i]);

  return (
    <div className="space-y-6 pb-20">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="text-yellow-400">CBOW</span>: Predict Center from Context
        </h2>
        <p className="text-gray-400">
          Continuous Bag of Words: given surrounding words, predict the center word
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap justify-center gap-3">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="flex items-center gap-2 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded-lg transition-colors"
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
                ? 'bg-yellow-500 text-black scale-110' 
                : i < currentStep 
                ? 'bg-yellow-900 text-yellow-300' 
                : 'bg-white/10 text-gray-500'
            }`}
          >
            {i + 1}
          </button>
        ))}
      </div>

      {/* Current Step Info */}
      <div className="bg-yellow-900/20 border border-yellow-500/30 rounded-xl p-4">
        <h3 className="font-bold text-yellow-400">Step {currentStep + 1}: {steps[currentStep].title}</h3>
        <p className="text-gray-300 mt-1">{steps[currentStep].description}</p>
      </div>

      {/* Main Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        {/* Sentence Display */}
        <div className="flex flex-wrap justify-center gap-2 mb-6">
          {sentence.map((word, i) => {
            const isTarget = i === centerIndex && currentStep >= 1;
            const isContext = contextIndices.includes(i) && currentStep >= 2;
            
            return (
              <div
                key={i}
                className={`px-4 py-2 rounded-lg font-mono text-lg transition-all duration-300 ${
                  isTarget
                    ? 'bg-yellow-500 text-black scale-110 shadow-lg shadow-yellow-500/50'
                    : isContext
                    ? 'bg-blue-500 text-white scale-105 shadow-lg shadow-blue-500/50'
                    : 'bg-white/10 text-gray-400'
                }`}
              >
                {word}
                {isTarget && currentStep >= 1 && (
                  <span className="text-xs ml-1">(?)</span>
                )}
              </div>
            );
          })}
        </div>

        {/* Context Words List */}
        {currentStep >= 2 && (
          <div className="text-center mb-6 animate-fadeIn">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-900/30 rounded-lg border border-blue-500/30">
              <span className="text-gray-400">Context words:</span>
              <div className="flex gap-1">
                {contextWords.map((w, i) => (
                  <span key={i} className="px-2 py-1 bg-blue-500/30 rounded text-blue-300 font-mono text-sm">
                    {w}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Averaging Visualization */}
        {currentStep >= 3 && (
          <div className="space-y-4 animate-fadeIn">
            <h4 className="text-center text-gray-400">Average Context Embeddings:</h4>
            <div className="flex justify-center items-center gap-4 flex-wrap">
              {contextWords.map((word, i) => (
                <div key={i} className="text-center">
                  <div className="bg-blue-900/30 rounded-lg p-3 border border-blue-500/30">
                    <div className="font-mono text-blue-400 text-sm">{word}</div>
                    <div className="text-xs text-gray-500 mt-1">
                      [{(Math.random() * 2 - 1).toFixed(1)}, {(Math.random() * 2 - 1).toFixed(1)}, ...]
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="text-center text-gray-500">↓ Average ↓</div>
            <div className="flex justify-center">
              <div className="bg-purple-900/30 rounded-lg p-4 border border-purple-500/30">
                <div className="font-mono text-purple-400">h = average(context vectors)</div>
                <div className="text-xs text-gray-400 mt-1">
                  [{(Math.random() * 2 - 1).toFixed(2)}, {(Math.random() * 2 - 1).toFixed(2)}, {(Math.random() * 2 - 1).toFixed(2)}, ...]
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Neural Network */}
        {currentStep >= 4 && (
          <div className="mt-8 animate-fadeIn">
            <h4 className="text-center text-gray-400 mb-4">CBOW Neural Network:</h4>
            <div className="flex justify-center items-center gap-4">
              {/* Input (context words) */}
              <div className="text-center">
                <div className="w-24 h-24 rounded-lg bg-blue-900/50 border border-blue-500 flex flex-col items-center justify-center p-2">
                  {contextWords.slice(0, 4).map((w, i) => (
                    <div key={i} className="font-mono text-xs text-blue-400">{w}</div>
                  ))}
                  {contextWords.length > 4 && <div className="text-xs text-gray-500">...</div>}
                </div>
                <div className="text-xs text-gray-500 mt-2">Context<br/>(one-hot each)</div>
              </div>

              <ArrowRight className="text-gray-500" />

              {/* Projection Layer */}
              <div className="text-center">
                <div className="w-20 h-24 rounded-lg bg-green-900/50 border border-green-500 flex flex-col items-center justify-center px-2">
                  <div className="text-xs text-green-300 mb-1">W</div>
                  <div className="text-xs text-gray-400">Lookup &</div>
                  <div className="text-xs text-gray-400">Average</div>
                </div>
                <div className="text-xs text-gray-500 mt-2">Projection</div>
              </div>

              <ArrowRight className="text-gray-500" />

              {/* Hidden (averaged) */}
              <div className="text-center">
                <div className="w-20 h-24 rounded-lg bg-purple-900/50 border border-purple-500 flex flex-col items-center justify-center">
                  <div className="text-xs text-purple-300">h</div>
                  <div className="text-xs text-gray-400 mt-1">D dims</div>
                </div>
                <div className="text-xs text-gray-500 mt-2">Hidden<br/>(averaged)</div>
              </div>

              <ArrowRight className="text-gray-500" />

              {/* Output */}
              <div className="text-center">
                <div className="w-20 h-24 rounded-lg bg-yellow-900/50 border border-yellow-500 flex flex-col items-center justify-center">
                  <div className="font-mono text-yellow-400">jumps</div>
                  <div className="text-xs text-gray-400 mt-1">P=0.85</div>
                </div>
                <div className="text-xs text-gray-500 mt-2">Prediction<br/>(softmax)</div>
              </div>
            </div>
          </div>
        )}

        {/* Learning Update */}
        {currentStep >= 5 && (
          <div className="mt-8 bg-gradient-to-r from-yellow-900/30 to-green-900/30 rounded-xl p-6 border border-yellow-500/30 animate-fadeIn">
            <h4 className="text-yellow-400 font-bold mb-3">🔄 Weight Update</h4>
            <p className="text-gray-300 text-sm mb-4">
              If prediction is wrong, adjust embeddings so context words 
              point more toward the correct center word.
            </p>
            <div className="font-mono text-sm bg-black/30 rounded-lg p-4">
              <p className="text-gray-400">// Gradient descent update</p>
              <p className="text-green-300">W_new = W_old - learning_rate × ∇Loss</p>
              <p className="text-gray-400 mt-2">// Both input and output weights updated</p>
            </div>
          </div>
        )}
      </div>

      {/* CBOW vs Skip-gram Comparison */}
      <div className="bg-black/40 rounded-xl p-6 border border-white/10">
        <h4 className="font-bold text-white mb-4">🔄 CBOW vs Skip-gram</h4>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-yellow-900/20 rounded-lg p-4 border border-yellow-500/30">
            <h5 className="text-yellow-400 font-medium mb-2">CBOW</h5>
            <p className="text-sm text-gray-400 mb-2">Context → Center</p>
            <div className="text-xs space-y-1 text-gray-500">
              <p>• One prediction per window position</p>
              <p>• Faster to train</p>
              <p>• Better for frequent words</p>
              <p>• Smooths over context</p>
            </div>
          </div>
          <div className="bg-green-900/20 rounded-lg p-4 border border-green-500/30">
            <h5 className="text-green-400 font-medium mb-2">Skip-gram</h5>
            <p className="text-sm text-gray-400 mb-2">Center → Context</p>
            <div className="text-xs space-y-1 text-gray-500">
              <p>• Multiple predictions per center word</p>
              <p>• Slower to train</p>
              <p>• Better for rare words</p>
              <p>• Captures more nuance</p>
            </div>
          </div>
        </div>
      </div>

      {/* CBOW Formula */}
      <div className="bg-black/40 rounded-xl p-6 border border-white/10">
        <h4 className="font-bold text-white mb-4">📐 CBOW Objective</h4>
        <div className="bg-black/50 rounded-lg p-4 text-center font-mono">
          <p className="text-yellow-300">
            maximize Σ log P(center | context)
          </p>
          <p className="text-sm text-gray-400 mt-2">
            h = (1/C) Σ W<sub>context_i</sub>
          </p>
          <p className="text-sm text-gray-400">
            P(center | context) = softmax(W' · h)
          </p>
        </div>
      </div>

      {/* Visual Comparison */}
      <div className="bg-gradient-to-r from-yellow-900/20 to-green-900/20 rounded-xl p-6 border border-white/10">
        <h4 className="font-bold text-white mb-4 text-center">Direction of Prediction</h4>
        <div className="grid md:grid-cols-2 gap-8">
          {/* CBOW */}
          <div className="text-center">
            <h5 className="text-yellow-400 mb-3">CBOW</h5>
            <div className="flex justify-center items-center gap-2">
              <div className="flex flex-col gap-1">
                <span className="px-2 py-1 bg-blue-500/30 rounded text-xs">quick</span>
                <span className="px-2 py-1 bg-blue-500/30 rounded text-xs">brown</span>
              </div>
              <ArrowRight className="text-yellow-400" />
              <span className="px-3 py-2 bg-yellow-500 text-black rounded font-bold">?</span>
              <ArrowRight className="text-yellow-400 rotate-180" />
              <div className="flex flex-col gap-1">
                <span className="px-2 py-1 bg-blue-500/30 rounded text-xs">over</span>
                <span className="px-2 py-1 bg-blue-500/30 rounded text-xs">the</span>
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-2">Many inputs → One output</p>
          </div>
          
          {/* Skip-gram */}
          <div className="text-center">
            <h5 className="text-green-400 mb-3">Skip-gram</h5>
            <div className="flex justify-center items-center gap-2">
              <div className="flex flex-col gap-1">
                <span className="px-2 py-1 bg-blue-500/30 rounded text-xs">?</span>
                <span className="px-2 py-1 bg-blue-500/30 rounded text-xs">?</span>
              </div>
              <ArrowRight className="text-green-400 rotate-180" />
              <span className="px-3 py-2 bg-green-500 text-black rounded font-bold">jumps</span>
              <ArrowRight className="text-green-400" />
              <div className="flex flex-col gap-1">
                <span className="px-2 py-1 bg-blue-500/30 rounded text-xs">?</span>
                <span className="px-2 py-1 bg-blue-500/30 rounded text-xs">?</span>
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-2">One input → Many outputs</p>
          </div>
        </div>
      </div>
    </div>
  );
}
