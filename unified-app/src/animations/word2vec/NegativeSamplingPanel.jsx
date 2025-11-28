import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Zap, ThumbsUp, ThumbsDown, TrendingUp } from 'lucide-react';

export default function NegativeSamplingPanel() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [numNegatives, setNumNegatives] = useState(5);

  const vocabulary = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 'cat', 'bird', 'tree', 'sky'];
  const centerWord = 'fox';
  const positiveContext = 'jumps';

  const steps = [
    { title: 'The Softmax Problem', description: 'Standard softmax over full vocabulary is expensive' },
    { title: 'Positive Pair', description: 'We have a real (center, context) pair from training data' },
    { title: 'Sample Negatives', description: 'Randomly select words that are NOT in the context' },
    { title: 'Binary Classification', description: 'Train: real pair → 1, fake pairs → 0' },
    { title: 'Update Embeddings', description: 'Only update selected word vectors (much faster!)' },
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
      }, 3000);
      return () => clearInterval(interval);
    }
  }, [isPlaying]);

  const reset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
  };

  // Get negative samples
  const getNegativeSamples = () => {
    const negatives = vocabulary.filter(w => w !== centerWord && w !== positiveContext);
    return negatives.slice(0, numNegatives);
  };

  const negativeSamples = getNegativeSamples();

  return (
    <div className="space-y-6 pb-20">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="text-orange-400">Negative Sampling</span>: Efficient Training
        </h2>
        <p className="text-gray-400">
          Turn expensive softmax into simple binary classification
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap justify-center gap-3">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="flex items-center gap-2 px-4 py-2 bg-orange-600 hover:bg-orange-700 rounded-lg transition-colors"
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
          <span className="text-sm text-gray-400"># Negatives:</span>
          <select
            value={numNegatives}
            onChange={(e) => setNumNegatives(parseInt(e.target.value))}
            className="bg-gray-700 rounded px-2 py-1 text-sm"
          >
            {[2, 5, 10, 15, 20].map(n => (
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
                ? 'bg-orange-500 text-black scale-110' 
                : i < currentStep 
                ? 'bg-orange-900 text-orange-300' 
                : 'bg-white/10 text-gray-500'
            }`}
          >
            {i + 1}
          </button>
        ))}
      </div>

      {/* Current Step Info */}
      <div className="bg-orange-900/20 border border-orange-500/30 rounded-xl p-4">
        <h3 className="font-bold text-orange-400">Step {currentStep + 1}: {steps[currentStep].title}</h3>
        <p className="text-gray-300 mt-1">{steps[currentStep].description}</p>
      </div>

      {/* Main Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        
        {/* Step 0: The Problem */}
        {currentStep === 0 && (
          <div className="space-y-6 animate-fadeIn">
            <h4 className="text-center text-gray-400 mb-4">The Softmax Problem</h4>
            
            <div className="grid md:grid-cols-2 gap-6">
              {/* Standard Softmax */}
              <div className="bg-red-900/20 rounded-xl p-4 border border-red-500/30">
                <h5 className="text-red-400 font-bold mb-3">❌ Standard Softmax</h5>
                <div className="font-mono text-sm bg-black/30 p-3 rounded">
                  <p className="text-gray-400">P(w|center) = </p>
                  <p className="text-red-300 ml-4">exp(v<sub>w</sub> · v<sub>c</sub>) /</p>
                  <p className="text-red-400 ml-4">Σ exp(v<sub>i</sub> · v<sub>c</sub>)</p>
                </div>
                <p className="text-sm text-gray-400 mt-3">
                  Must compute for ALL {vocabulary.length}+ words!
                </p>
                <p className="text-xs text-red-400 mt-2">
                  Real vocabulary: 10K - 1M words 😱
                </p>
              </div>

              {/* Negative Sampling */}
              <div className="bg-green-900/20 rounded-xl p-4 border border-green-500/30">
                <h5 className="text-green-400 font-bold mb-3">✅ Negative Sampling</h5>
                <div className="font-mono text-sm bg-black/30 p-3 rounded">
                  <p className="text-gray-400">Binary classification:</p>
                  <p className="text-green-300 ml-4">1 positive pair</p>
                  <p className="text-orange-300 ml-4">+ k negative pairs</p>
                </div>
                <p className="text-sm text-gray-400 mt-3">
                  Only update {numNegatives + 1} word vectors!
                </p>
                <p className="text-xs text-green-400 mt-2">
                  Speedup: O(V) → O(k) where k ≈ 5-20 🚀
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Step 1: Positive Pair */}
        {currentStep === 1 && (
          <div className="space-y-6 animate-fadeIn">
            <h4 className="text-center text-gray-400 mb-4">Real Training Pair</h4>
            
            <div className="flex justify-center items-center gap-8">
              <div className="text-center">
                <div className="w-24 h-24 rounded-xl bg-green-900/50 border-2 border-green-500 flex items-center justify-center">
                  <span className="font-mono text-xl text-green-400">{centerWord}</span>
                </div>
                <p className="text-sm text-gray-400 mt-2">Center Word</p>
              </div>
              
              <div className="flex flex-col items-center">
                <ThumbsUp className="text-green-400 mb-2" size={32} />
                <span className="text-green-400 font-bold">REAL PAIR</span>
                <span className="text-green-300 text-sm">Label = 1</span>
              </div>
              
              <div className="text-center">
                <div className="w-24 h-24 rounded-xl bg-blue-900/50 border-2 border-blue-500 flex items-center justify-center">
                  <span className="font-mono text-xl text-blue-400">{positiveContext}</span>
                </div>
                <p className="text-sm text-gray-400 mt-2">Context Word</p>
              </div>
            </div>

            <div className="text-center text-gray-500 text-sm mt-4">
              This pair came from the sentence: "...the quick brown <span className="text-green-400">fox</span> <span className="text-blue-400">jumps</span> over..."
            </div>
          </div>
        )}

        {/* Step 2: Negative Samples */}
        {currentStep === 2 && (
          <div className="space-y-6 animate-fadeIn">
            <h4 className="text-center text-gray-400 mb-4">Sample Negative Words</h4>
            
            {/* Positive */}
            <div className="flex justify-center items-center gap-4 mb-6">
              <div className="w-20 h-16 rounded-lg bg-green-900/50 border border-green-500 flex items-center justify-center">
                <span className="font-mono text-green-400">{centerWord}</span>
              </div>
              <span className="text-gray-500">+</span>
              <div className="w-20 h-16 rounded-lg bg-blue-900/50 border border-blue-500 flex items-center justify-center">
                <span className="font-mono text-blue-400">{positiveContext}</span>
              </div>
              <ThumbsUp className="text-green-400" />
            </div>

            {/* Negatives */}
            <div className="bg-red-900/20 rounded-xl p-4 border border-red-500/30">
              <div className="flex items-center gap-2 mb-3">
                <ThumbsDown className="text-red-400" />
                <span className="text-red-400 font-bold">Negative Samples (Random)</span>
              </div>
              <div className="flex flex-wrap justify-center gap-2">
                {negativeSamples.map((word, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-2 px-3 py-2 bg-black/30 rounded-lg animate-fadeIn"
                    style={{ animationDelay: `${i * 100}ms` }}
                  >
                    <span className="font-mono text-green-400">{centerWord}</span>
                    <span className="text-gray-500">+</span>
                    <span className="font-mono text-red-400">{word}</span>
                  </div>
                ))}
              </div>
              <p className="text-xs text-gray-500 mt-3 text-center">
                These words were NOT actually in the context of "{centerWord}"
              </p>
            </div>

            {/* Sampling Distribution */}
            <div className="bg-black/40 rounded-lg p-4">
              <h5 className="text-gray-400 text-sm mb-2">Sampling Distribution:</h5>
              <p className="text-xs text-gray-500">
                P(w) ∝ freq(w)<sup>0.75</sup>
              </p>
              <p className="text-xs text-gray-400 mt-1">
                The 0.75 power helps balance between very common and rare words
              </p>
            </div>
          </div>
        )}

        {/* Step 3: Binary Classification */}
        {currentStep === 3 && (
          <div className="space-y-6 animate-fadeIn">
            <h4 className="text-center text-gray-400 mb-4">Binary Classification</h4>
            
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr>
                    <th className="text-left p-2 text-gray-400">Center</th>
                    <th className="text-left p-2 text-gray-400">Word</th>
                    <th className="text-left p-2 text-gray-400">Label</th>
                    <th className="text-left p-2 text-gray-400">Type</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-t border-white/10 bg-green-900/20">
                    <td className="p-2 font-mono text-green-400">{centerWord}</td>
                    <td className="p-2 font-mono text-blue-400">{positiveContext}</td>
                    <td className="p-2 text-green-400 font-bold">1</td>
                    <td className="p-2 text-green-400">Positive ✓</td>
                  </tr>
                  {negativeSamples.map((word, i) => (
                    <tr key={i} className="border-t border-white/10 bg-red-900/10">
                      <td className="p-2 font-mono text-green-400">{centerWord}</td>
                      <td className="p-2 font-mono text-red-400">{word}</td>
                      <td className="p-2 text-red-400 font-bold">0</td>
                      <td className="p-2 text-red-400">Negative ✗</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="bg-black/40 rounded-lg p-4">
              <h5 className="text-orange-400 font-bold mb-2">Loss Function:</h5>
              <div className="font-mono text-sm">
                <p className="text-gray-300">L = -log σ(v<sub>context</sub> · v<sub>center</sub>)</p>
                <p className="text-gray-500">- Σ log σ(-v<sub>negative</sub> · v<sub>center</sub>)</p>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                σ = sigmoid function: σ(x) = 1 / (1 + e<sup>-x</sup>)
              </p>
            </div>
          </div>
        )}

        {/* Step 4: Update */}
        {currentStep === 4 && (
          <div className="space-y-6 animate-fadeIn">
            <h4 className="text-center text-gray-400 mb-4">Efficient Weight Updates</h4>
            
            <div className="grid md:grid-cols-2 gap-6">
              {/* What gets updated */}
              <div className="bg-green-900/20 rounded-xl p-4 border border-green-500/30">
                <h5 className="text-green-400 font-bold mb-3">✅ Updated Vectors</h5>
                <div className="space-y-2">
                  <div className="flex items-center gap-2 px-3 py-2 bg-black/30 rounded">
                    <span className="font-mono text-green-400">{centerWord}</span>
                    <span className="text-xs text-gray-500">← center</span>
                  </div>
                  <div className="flex items-center gap-2 px-3 py-2 bg-black/30 rounded">
                    <span className="font-mono text-blue-400">{positiveContext}</span>
                    <span className="text-xs text-gray-500">← positive</span>
                  </div>
                  {negativeSamples.slice(0, 3).map((word, i) => (
                    <div key={i} className="flex items-center gap-2 px-3 py-2 bg-black/30 rounded">
                      <span className="font-mono text-red-400">{word}</span>
                      <span className="text-xs text-gray-500">← negative</span>
                    </div>
                  ))}
                  {negativeSamples.length > 3 && (
                    <div className="text-gray-500 text-sm">+ {negativeSamples.length - 3} more...</div>
                  )}
                </div>
                <p className="text-xs text-green-400 mt-2">
                  Total: {2 + numNegatives} vectors updated
                </p>
              </div>

              {/* What stays the same */}
              <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-600/30">
                <h5 className="text-gray-400 font-bold mb-3">⏸️ Not Updated</h5>
                <div className="flex flex-wrap gap-2">
                  {vocabulary.filter(w => 
                    w !== centerWord && 
                    w !== positiveContext && 
                    !negativeSamples.includes(w)
                  ).map((word, i) => (
                    <span key={i} className="px-2 py-1 bg-black/30 rounded text-xs text-gray-500">
                      {word}
                    </span>
                  ))}
                  <span className="text-gray-600">+ thousands more...</span>
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  In real vocab: ~99.99% untouched per step
                </p>
              </div>
            </div>

            {/* Speed Comparison */}
            <div className="bg-gradient-to-r from-orange-900/30 to-green-900/30 rounded-xl p-6 border border-orange-500/30">
              <h5 className="text-orange-400 font-bold mb-4 flex items-center gap-2">
                <TrendingUp /> Speed Improvement
              </h5>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-black/30 rounded-lg p-3">
                  <p className="text-red-400 text-sm">Full Softmax</p>
                  <p className="text-2xl font-bold text-red-300">O(V)</p>
                  <p className="text-xs text-gray-500">V = vocabulary size (100K+)</p>
                </div>
                <div className="bg-black/30 rounded-lg p-3">
                  <p className="text-green-400 text-sm">Negative Sampling</p>
                  <p className="text-2xl font-bold text-green-300">O(k)</p>
                  <p className="text-xs text-gray-500">k = {numNegatives} (typically 5-20)</p>
                </div>
              </div>
              <p className="text-center text-gray-400 text-sm mt-4">
                Speedup: ~{Math.round(100000 / numNegatives).toLocaleString()}x faster per training step!
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Key Insights */}
      <div className="bg-black/40 rounded-xl p-6 border border-white/10">
        <h4 className="font-bold text-white mb-4">💡 Key Insights</h4>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white/5 rounded-lg p-4">
            <h5 className="text-orange-400 font-medium mb-2">Why It Works</h5>
            <p className="text-sm text-gray-400">
              We don't need perfect probabilities—we just need vectors where 
              real pairs have higher scores than random pairs.
            </p>
          </div>
          <div className="bg-white/5 rounded-lg p-4">
            <h5 className="text-orange-400 font-medium mb-2">How Many Negatives?</h5>
            <p className="text-sm text-gray-400">
              • Small datasets: 5-20 negatives<br/>
              • Large datasets: 2-5 negatives<br/>
              More data = fewer negatives needed
            </p>
          </div>
        </div>
      </div>

      {/* Formula */}
      <div className="bg-gradient-to-r from-purple-900/20 to-blue-900/20 rounded-xl p-6 border border-purple-500/30">
        <h4 className="font-bold text-purple-400 mb-4">📐 Negative Sampling Objective</h4>
        <div className="bg-black/50 rounded-lg p-4 font-mono text-center">
          <p className="text-purple-300">
            log σ(v'<sub>context</sub> · v<sub>center</sub>) + 
            Σ<sub>k</sub> 𝔼<sub>w~P(w)</sub>[log σ(-v'<sub>w</sub> · v<sub>center</sub>)]
          </p>
        </div>
        <p className="text-sm text-gray-400 mt-4 text-center">
          Maximize: dot product with positive context<br/>
          Minimize: dot product with k negative samples
        </p>
      </div>
    </div>
  );
}
