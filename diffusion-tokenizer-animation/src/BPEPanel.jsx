import React, { useState, useRef, useEffect } from 'react';
import { Play, RotateCcw, ArrowRight, Merge } from 'lucide-react';
import gsap from 'gsap';

function BPEPanel() {
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const animationRef = useRef(null);

  const bpeSteps = [
    {
      title: "Start with characters",
      chars: ['l', 'o', 'w', ' ', 'l', 'o', 'w', 'e', 'r', ' ', 'n', 'e', 'w', 'e', 's', 't'],
      pairs: [],
      merged: null,
      explanation: "Begin with individual characters as initial 'vocabulary'"
    },
    {
      title: "Count pairs",
      chars: ['l', 'o', 'w', ' ', 'l', 'o', 'w', 'e', 'r', ' ', 'n', 'e', 'w', 'e', 's', 't'],
      pairs: [{ pair: 'lo', count: 2 }, { pair: 'ow', count: 2 }, { pair: 'we', count: 2 }],
      merged: null,
      explanation: "Find the most frequent adjacent pairs: 'lo' appears 2 times"
    },
    {
      title: "Merge most frequent",
      chars: ['lo', 'w', ' ', 'lo', 'w', 'e', 'r', ' ', 'n', 'e', 'w', 'e', 's', 't'],
      pairs: [{ pair: 'lo', count: 2, merged: true }],
      merged: 'lo',
      explanation: "Merge 'l' + 'o' → 'lo' everywhere in the corpus"
    },
    {
      title: "Repeat: count new pairs",
      chars: ['lo', 'w', ' ', 'lo', 'w', 'e', 'r', ' ', 'n', 'e', 'w', 'e', 's', 't'],
      pairs: [{ pair: 'low', count: 2 }, { pair: 'we', count: 2 }],
      merged: null,
      explanation: "Now 'low' appears twice (lo+w)"
    },
    {
      title: "Merge again",
      chars: ['low', ' ', 'low', 'e', 'r', ' ', 'n', 'e', 'w', 'e', 's', 't'],
      pairs: [{ pair: 'low', count: 2, merged: true }],
      merged: 'low',
      explanation: "Merge 'lo' + 'w' → 'low'"
    },
    {
      title: "Continue until vocab size reached",
      chars: ['low', ' ', 'low', 'er', ' ', 'new', 'est'],
      pairs: [],
      merged: null,
      explanation: "Keep merging until desired vocabulary size (~49K for CLIP)"
    }
  ];

  const currentStep = bpeSteps[step];

  const playAnimation = () => {
    if (step >= bpeSteps.length - 1) {
      setStep(0);
      return;
    }
    setIsPlaying(true);
    
    const interval = setInterval(() => {
      setStep(prev => {
        if (prev >= bpeSteps.length - 1) {
          clearInterval(interval);
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 1500);
  };

  const resetAnimation = () => {
    setStep(0);
    setIsPlaying(false);
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-orange-400 mb-2">Byte Pair Encoding (BPE)</h2>
        <p className="text-gray-300 max-w-3xl mx-auto">
          BPE is a data compression algorithm adapted for tokenization.
          It iteratively merges the most frequent pairs of characters/subwords
          until reaching the desired vocabulary size.
        </p>
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-4">
        <button
          onClick={playAnimation}
          disabled={isPlaying}
          className="flex items-center gap-2 px-4 py-2 bg-orange-600 hover:bg-orange-700 disabled:opacity-50 rounded-lg transition-colors"
        >
          <Play size={18} />
          {step === 0 ? 'Start BPE' : 'Continue'}
        </button>
        <button
          onClick={resetAnimation}
          className="flex items-center gap-2 px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <RotateCcw size={18} />
          Reset
        </button>
        <div className="flex items-center gap-2 px-4 py-2 bg-black/30 rounded-lg">
          <span className="text-gray-400">Step:</span>
          <span className="text-orange-400 font-bold">{step + 1}/{bpeSteps.length}</span>
        </div>
      </div>

      {/* Step Visualization */}
      <div ref={animationRef} className="bg-black/40 rounded-xl p-6 space-y-6">
        {/* Title */}
        <div className="text-center">
          <h3 className="text-xl font-semibold text-orange-400">{currentStep.title}</h3>
          <p className="text-gray-400 text-sm mt-1">{currentStep.explanation}</p>
        </div>

        {/* Current Tokens */}
        <div className="bg-black/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-3">Current tokenization of "low lower newest":</div>
          <div className="flex flex-wrap gap-2">
            {currentStep.chars.map((char, i) => (
              <div
                key={i}
                className={`px-3 py-2 rounded-lg font-mono text-sm transition-all ${
                  char === currentStep.merged
                    ? 'bg-orange-500/50 text-orange-200 border-2 border-orange-400 scale-110'
                    : char.length > 1
                    ? 'bg-purple-500/30 text-purple-300 border border-purple-500/50'
                    : char === ' '
                    ? 'bg-gray-700/50 text-gray-400 border border-gray-600'
                    : 'bg-blue-500/30 text-blue-300 border border-blue-500/50'
                }`}
              >
                {char === ' ' ? '⎵' : char}
              </div>
            ))}
          </div>
        </div>

        {/* Pair Frequencies */}
        {currentStep.pairs.length > 0 && (
          <div className="bg-black/30 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-3">Most frequent pairs:</div>
            <div className="flex flex-wrap gap-3">
              {currentStep.pairs.map((p, i) => (
                <div
                  key={i}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg ${
                    p.merged
                      ? 'bg-green-500/30 border-2 border-green-400'
                      : 'bg-gray-700/50 border border-gray-600'
                  }`}
                >
                  <span className={`font-mono ${p.merged ? 'text-green-300' : 'text-white'}`}>
                    "{p.pair}"
                  </span>
                  <span className="text-gray-400">×</span>
                  <span className={p.merged ? 'text-green-400 font-bold' : 'text-orange-400'}>
                    {p.count}
                  </span>
                  {p.merged && <Merge size={16} className="text-green-400" />}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Merge Action */}
        {currentStep.merged && (
          <div className="flex items-center justify-center gap-4 py-4">
            <div className="text-center">
              <div className="text-sm text-gray-400 mb-1">Before</div>
              <div className="flex gap-1">
                {currentStep.merged.split('').map((c, i) => (
                  <span key={i} className="px-2 py-1 bg-blue-500/30 rounded text-blue-300 font-mono">
                    {c}
                  </span>
                ))}
              </div>
            </div>
            <ArrowRight className="text-green-400" size={24} />
            <div className="text-center">
              <div className="text-sm text-gray-400 mb-1">After</div>
              <span className="px-3 py-1 bg-green-500/30 rounded text-green-300 font-mono border-2 border-green-400">
                {currentStep.merged}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* BPE Algorithm */}
      <div className="bg-gradient-to-r from-orange-500/10 to-red-500/10 rounded-xl p-6 border border-orange-500/30">
        <h3 className="font-semibold text-orange-400 mb-4">📝 BPE Algorithm</h3>
        <div className="bg-black/40 rounded-lg p-4 font-mono text-sm">
          <div className="text-gray-400"># Pseudocode</div>
          <div className="text-blue-400">def</div>
          <span className="text-yellow-400"> train_bpe</span>
          <span className="text-white">(corpus, vocab_size):</span>
          <div className="ml-4 text-gray-300">
            <div>vocab = set(all_characters(corpus))</div>
            <div className="text-blue-400">while</div>
            <span className="text-white"> len(vocab) &lt; vocab_size:</span>
            <div className="ml-4">
              <div>pairs = count_pairs(corpus)</div>
              <div>best_pair = most_frequent(pairs)</div>
              <div>merge(corpus, best_pair)</div>
              <div>vocab.add(best_pair)</div>
            </div>
            <div className="text-blue-400">return</div>
            <span className="text-white"> vocab</span>
          </div>
        </div>
      </div>

      {/* Key Properties */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-blue-500/10 rounded-lg p-4 border border-blue-500/30">
          <h4 className="font-semibold text-blue-400 mb-2">Deterministic</h4>
          <p className="text-sm text-gray-300">
            Same text always produces same tokens. Essential for reproducibility.
          </p>
        </div>
        <div className="bg-purple-500/10 rounded-lg p-4 border border-purple-500/30">
          <h4 className="font-semibold text-purple-400 mb-2">Frequency-based</h4>
          <p className="text-sm text-gray-300">
            Common words become single tokens. Rare words split into pieces.
          </p>
        </div>
        <div className="bg-green-500/10 rounded-lg p-4 border border-green-500/30">
          <h4 className="font-semibold text-green-400 mb-2">Open Vocabulary</h4>
          <p className="text-sm text-gray-300">
            Can encode ANY text - unknown words split into known subwords.
          </p>
        </div>
      </div>
    </div>
  );
}

export default BPEPanel;
