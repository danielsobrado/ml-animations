import React, { useState } from 'react';
import { Calculator, Zap, Target, Play, RotateCcw, ArrowRight, CheckCircle } from 'lucide-react';

export default function ObjectivePanel() {
  const [step, setStep] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);

  const steps = [
    {
      title: 'Step 1: The Ratio Hypothesis',
      description: 'Word meaning is captured by probability ratios: F(w_i, w_j, w̃_k) should encode P_ik / P_jk'
    },
    {
      title: 'Step 2: Derive the Form',
      description: 'Through mathematical derivation, we arrive at: w_i · w̃_k + b_i + b̃_k = log(X_ik)'
    },
    {
      title: 'Step 3: Define the Loss',
      description: 'Minimize the difference between dot product and log co-occurrence'
    },
    {
      title: 'Step 4: Add Weighting',
      description: 'Weight frequent co-occurrences less with f(X_ij) to prevent domination by common pairs'
    },
    {
      title: 'Step 5: Final Objective',
      description: 'The complete weighted least squares objective function'
    }
  ];

  const playAnimation = () => {
    setIsPlaying(true);
    let currentStep = 1;
    setStep(1);
    
    const interval = setInterval(() => {
      currentStep++;
      if (currentStep > steps.length) {
        setIsPlaying(false);
        clearInterval(interval);
      } else {
        setStep(currentStep);
      }
    }, 3000);
  };

  const reset = () => {
    setStep(1);
    setIsPlaying(false);
  };

  return (
    <div className="space-y-6 pb-20">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="gradient-text">GloVe Objective:</span> The Math Behind the Magic
        </h2>
        <p className="text-gray-400">
          Understanding how GloVe learns meaningful word vectors
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
                : step > i + 1
                ? 'bg-green-900/50 text-green-400'
                : 'bg-white/10 text-gray-400 hover:bg-white/20'
            }`}
          >
            {step > i + 1 ? <CheckCircle size={14} /> : i + 1}
          </button>
        ))}
      </div>

      {/* Current Step */}
      <div className="bg-gradient-to-r from-violet-900/30 to-cyan-900/30 rounded-xl p-4 border border-violet-500/30">
        <h3 className="text-lg font-bold text-violet-400 mb-1">{steps[step - 1].title}</h3>
        <p className="text-gray-300">{steps[step - 1].description}</p>
      </div>

      {/* Step 1: Ratio Hypothesis */}
      {step >= 1 && (
        <div className={`bg-black/30 rounded-xl p-6 border border-white/10 transition-all ${step === 1 ? 'ring-2 ring-violet-500' : ''}`}>
          <h4 className="flex items-center gap-2 text-lg font-bold text-violet-400 mb-4">
            <Target size={20} />
            The Probability Ratio Hypothesis
          </h4>
          
          <div className="bg-gradient-to-r from-violet-900/20 to-cyan-900/20 rounded-lg p-4 mb-4">
            <p className="text-center font-mono text-lg text-gray-200">
              F(w<sub>i</sub>, w<sub>j</sub>, w̃<sub>k</sub>) = P<sub>ik</sub> / P<sub>jk</sub>
            </p>
          </div>
          
          <p className="text-gray-300 mb-4">
            We want a function F that, given two word vectors and a context vector, 
            encodes how much more likely word i co-occurs with k compared to word j.
          </p>
          
          <div className="grid md:grid-cols-3 gap-3 text-sm">
            <div className="bg-violet-900/20 rounded-lg p-3 text-center">
              <p className="text-violet-400 font-mono">w<sub>i</sub></p>
              <p className="text-gray-400">Word vector for "ice"</p>
            </div>
            <div className="bg-cyan-900/20 rounded-lg p-3 text-center">
              <p className="text-cyan-400 font-mono">w<sub>j</sub></p>
              <p className="text-gray-400">Word vector for "steam"</p>
            </div>
            <div className="bg-green-900/20 rounded-lg p-3 text-center">
              <p className="text-green-400 font-mono">w̃<sub>k</sub></p>
              <p className="text-gray-400">Context vector for "solid"</p>
            </div>
          </div>
        </div>
      )}

      {/* Step 2: Derivation */}
      {step >= 2 && (
        <div className={`bg-black/30 rounded-xl p-6 border border-white/10 transition-all ${step === 2 ? 'ring-2 ring-violet-500' : ''}`}>
          <h4 className="flex items-center gap-2 text-lg font-bold text-cyan-400 mb-4">
            <Calculator size={20} />
            Mathematical Derivation
          </h4>
          
          <div className="space-y-4">
            <div className="bg-black/30 rounded-lg p-3 font-mono text-sm">
              <p className="text-gray-400">Starting point:</p>
              <p className="text-gray-200">F(w<sub>i</sub> - w<sub>j</sub>, w̃<sub>k</sub>) = P<sub>ik</sub> / P<sub>jk</sub></p>
            </div>
            
            <div className="flex justify-center">
              <ArrowRight className="text-violet-400" size={24} />
            </div>
            
            <div className="bg-black/30 rounded-lg p-3 font-mono text-sm">
              <p className="text-gray-400">Using dot product (F = exp):</p>
              <p className="text-gray-200">exp((w<sub>i</sub> - w<sub>j</sub>) · w̃<sub>k</sub>) = P<sub>ik</sub> / P<sub>jk</sub></p>
            </div>
            
            <div className="flex justify-center">
              <ArrowRight className="text-violet-400" size={24} />
            </div>
            
            <div className="bg-gradient-to-r from-violet-900/30 to-cyan-900/30 rounded-lg p-4 font-mono">
              <p className="text-gray-400">Final form:</p>
              <p className="text-xl text-white text-center mt-2">
                w<sub className="text-violet-400">i</sub> · w̃<sub className="text-cyan-400">k</sub> + b<sub className="text-violet-400">i</sub> + b̃<sub className="text-cyan-400">k</sub> = log(X<sub className="text-green-400">ik</sub>)
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Step 3: Loss Function */}
      {step >= 3 && (
        <div className={`bg-black/30 rounded-xl p-6 border border-white/10 transition-all ${step === 3 ? 'ring-2 ring-violet-500' : ''}`}>
          <h4 className="flex items-center gap-2 text-lg font-bold text-green-400 mb-4">
            <Zap size={20} />
            Basic Loss Function
          </h4>
          
          <div className="bg-gradient-to-r from-green-900/20 to-cyan-900/20 rounded-lg p-4 font-mono text-center">
            <p className="text-xl text-gray-200">
              J = Σ<sub>i,j</sub> (w<sub>i</sub> · w̃<sub>j</sub> + b<sub>i</sub> + b̃<sub>j</sub> - log X<sub>ij</sub>)²
            </p>
          </div>
          
          <p className="text-gray-300 mt-4 text-center">
            Minimize the squared difference between the model prediction and the log co-occurrence.
          </p>
          
          <div className="mt-4 p-3 bg-yellow-900/20 rounded-lg border border-yellow-500/30">
            <p className="text-yellow-400 font-medium">⚠️ Problem:</p>
            <p className="text-sm text-gray-300">
              Common word pairs (like "the, of") dominate the loss. We need weighting!
            </p>
          </div>
        </div>
      )}

      {/* Step 4: Weighting Function */}
      {step >= 4 && (
        <div className={`bg-black/30 rounded-xl p-6 border border-white/10 transition-all ${step === 4 ? 'ring-2 ring-violet-500' : ''}`}>
          <h4 className="flex items-center gap-2 text-lg font-bold text-yellow-400 mb-4">
            <Calculator size={20} />
            Weighting Function f(X)
          </h4>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <div className="bg-gradient-to-r from-yellow-900/20 to-orange-900/20 rounded-lg p-4 font-mono">
                <p className="text-sm text-gray-400 mb-2">Weighting function:</p>
                <div className="text-center">
                  <p className="text-gray-200">f(x) = </p>
                  <p className="text-gray-200 mt-1">(x / x<sub>max</sub>)<sup>α</sup> &nbsp; if x &lt; x<sub>max</sub></p>
                  <p className="text-gray-200 mt-1">1 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; otherwise</p>
                </div>
              </div>
              
              <div className="mt-4 space-y-2 text-sm">
                <p className="text-gray-300">
                  <span className="text-yellow-400 font-mono">x<sub>max</sub> = 100</span> — cap for frequent words
                </p>
                <p className="text-gray-300">
                  <span className="text-yellow-400 font-mono">α = 0.75</span> — smoothing parameter
                </p>
              </div>
            </div>
            
            <div className="bg-black/30 rounded-lg p-4">
              <p className="text-sm text-gray-400 mb-3">f(x) visualization:</p>
              <div className="h-32 flex items-end gap-1">
                {[1, 10, 25, 50, 75, 100, 150, 200].map((x, i) => {
                  const height = x <= 100 ? Math.pow(x / 100, 0.75) : 1;
                  return (
                    <div key={i} className="flex-1 flex flex-col items-center">
                      <div 
                        className="w-full bg-gradient-to-t from-yellow-600 to-yellow-400 rounded-t"
                        style={{ height: `${height * 100}%` }}
                      />
                      <span className="text-xs text-gray-500 mt-1">{x}</span>
                    </div>
                  );
                })}
              </div>
              <p className="text-xs text-gray-500 text-center mt-2">Co-occurrence count X</p>
            </div>
          </div>
          
          <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
            <div className="bg-green-900/20 rounded-lg p-3">
              <p className="text-green-400 font-medium">f(0) = 0</p>
              <p className="text-gray-400">No contribution from zero counts</p>
            </div>
            <div className="bg-cyan-900/20 rounded-lg p-3">
              <p className="text-cyan-400 font-medium">f(100) = 1</p>
              <p className="text-gray-400">Capped at x_max</p>
            </div>
          </div>
        </div>
      )}

      {/* Step 5: Final Objective */}
      {step >= 5 && (
        <div className={`bg-gradient-to-r from-violet-900/30 to-cyan-900/30 rounded-xl p-6 border border-violet-500/50 transition-all ${step === 5 ? 'ring-2 ring-violet-400 pulse-glow' : ''}`}>
          <h4 className="flex items-center gap-2 text-xl font-bold text-white mb-4">
            🎯 The Complete GloVe Objective
          </h4>
          
          <div className="bg-black/50 rounded-lg p-6 font-mono text-center">
            <p className="text-2xl text-white">
              J = Σ<sub className="text-violet-400">i,j=1</sub><sup className="text-cyan-400">V</sup> f(X<sub className="text-yellow-400">ij</sub>)(w<sub className="text-violet-400">i</sub> · w̃<sub className="text-cyan-400">j</sub> + b<sub className="text-violet-400">i</sub> + b̃<sub className="text-cyan-400">j</sub> - log X<sub className="text-green-400">ij</sub>)²
            </p>
          </div>
          
          <div className="mt-6 grid md:grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="bg-violet-900/30 rounded-lg p-3 text-center">
              <p className="text-violet-400 font-mono font-bold">w<sub>i</sub></p>
              <p className="text-xs text-gray-400">Word vector</p>
            </div>
            <div className="bg-cyan-900/30 rounded-lg p-3 text-center">
              <p className="text-cyan-400 font-mono font-bold">w̃<sub>j</sub></p>
              <p className="text-xs text-gray-400">Context vector</p>
            </div>
            <div className="bg-yellow-900/30 rounded-lg p-3 text-center">
              <p className="text-yellow-400 font-mono font-bold">f(X<sub>ij</sub>)</p>
              <p className="text-xs text-gray-400">Weighting function</p>
            </div>
            <div className="bg-green-900/30 rounded-lg p-3 text-center">
              <p className="text-green-400 font-mono font-bold">b<sub>i</sub>, b̃<sub>j</sub></p>
              <p className="text-xs text-gray-400">Bias terms</p>
            </div>
          </div>
          
          <div className="mt-6 bg-black/30 rounded-lg p-4">
            <h5 className="text-violet-400 font-medium mb-2">🎓 Final Word Vector</h5>
            <p className="text-gray-300 text-sm">
              After training, the final word embedding is: <span className="font-mono text-white">W + W̃</span>
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Since w and w̃ are symmetric, adding them gives slightly better performance.
            </p>
          </div>
        </div>
      )}

      {/* Key Properties */}
      <div className="bg-black/30 rounded-xl p-6 border border-white/10">
        <h4 className="text-lg font-bold text-violet-400 mb-4">💡 Key Properties of the GloVe Objective</h4>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-violet-900/20 rounded-lg p-3">
            <p className="text-violet-400 font-medium mb-1">Closed-form Statistics</p>
            <p className="text-sm text-gray-400">
              Uses pre-computed co-occurrence, not mini-batches
            </p>
          </div>
          <div className="bg-cyan-900/20 rounded-lg p-3">
            <p className="text-cyan-400 font-medium mb-1">Weighted Least Squares</p>
            <p className="text-sm text-gray-400">
              Not SGD on prediction task like Word2Vec
            </p>
          </div>
          <div className="bg-green-900/20 rounded-lg p-3">
            <p className="text-green-400 font-medium mb-1">Log Transform</p>
            <p className="text-sm text-gray-400">
              Working in log space captures ratios naturally
            </p>
          </div>
          <div className="bg-yellow-900/20 rounded-lg p-3">
            <p className="text-yellow-400 font-medium mb-1">Non-zero Entries Only</p>
            <p className="text-sm text-gray-400">
              Training only on observed co-occurrences (X_ij &gt; 0)
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
