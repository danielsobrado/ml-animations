import React, { useState, useEffect, useRef } from 'react';
import { Play, RotateCcw, ArrowLeftRight, Eye } from 'lucide-react';
import gsap from 'gsap';

function BidirectionalPanel() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedToken, setSelectedToken] = useState(2);
  const animationRef = useRef(null);
  const timelineRef = useRef(null);

  const tokens = ['A', 'majestic', 'lion', 'on', 'a', 'cliff'];

  useEffect(() => {
    if (!animationRef.current) return;

    const ctx = gsap.context(() => {
      gsap.set('.attention-line', { opacity: 0 });
      gsap.set('.token-highlight', { scale: 1 });
    }, animationRef);

    return () => ctx.revert();
  }, [selectedToken]);

  const playAnimation = () => {
    if (!animationRef.current) return;
    setIsPlaying(true);

    const ctx = gsap.context(() => {
      timelineRef.current = gsap.timeline({
        onComplete: () => setIsPlaying(false)
      });

      // Highlight selected token
      timelineRef.current.to(`.token-${selectedToken}`, {
        scale: 1.2,
        duration: 0.3,
        ease: 'back.out(1.7)'
      });

      // Show all attention lines
      timelineRef.current.to('.attention-line', {
        opacity: 1,
        stagger: 0.1,
        duration: 0.3
      });

      // Pulse effect
      timelineRef.current.to('.attention-line', {
        opacity: 0.7,
        duration: 0.5,
        yoyo: true,
        repeat: 2
      });

    }, animationRef);
  };

  const resetAnimation = () => {
    if (timelineRef.current) timelineRef.current.kill();
    
    const ctx = gsap.context(() => {
      gsap.set('.attention-line', { opacity: 0 });
      gsap.set('.token-highlight', { scale: 1 });
    }, animationRef);

    setIsPlaying(false);
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-emerald-600 dark:text-emerald-400 mb-2">Bidirectional Attention</h2>
        <p className="text-gray-700 dark:text-gray-300 max-w-3xl mx-auto">
          Unlike CLIP's causal attention, T5 uses <strong>bidirectional self-attention</strong>.
          Every token can attend to ALL other tokens - both before and after it.
        </p>
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-4 flex-wrap">
        <button
          onClick={playAnimation}
          disabled={isPlaying}
          className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 rounded-lg transition-colors"
        >
          <Play size={18} />
          Animate Attention
        </button>
        <button
          onClick={resetAnimation}
          className="flex items-center gap-2 px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <RotateCcw size={18} />
          Reset
        </button>
      </div>

      {/* Token Selection */}
      <div className="bg-black/30 rounded-xl p-4">
        <div className="text-sm text-gray-800 dark:text-gray-400 mb-3">Select a token to see what it attends to:</div>
        <div className="flex flex-wrap gap-2">
          {tokens.map((token, i) => (
            <button
              key={i}
              onClick={() => {
                setSelectedToken(i);
                resetAnimation();
              }}
              className={`px-4 py-2 rounded-lg transition-colors ${
                selectedToken === i
                  ? 'bg-emerald-600 text-white'
                  : 'bg-white/10 text-gray-700 dark:text-gray-300 hover:bg-white/20'
              }`}
            >
              {token}
            </button>
          ))}
        </div>
      </div>

      {/* Bidirectional Visualization */}
      <div ref={animationRef} className="bg-black/40 rounded-xl p-6">
        <svg viewBox="0 0 800 250" className="w-full h-auto">
          {/* Tokens */}
          {tokens.map((token, i) => {
            const x = 80 + i * 110;
            const isSelected = i === selectedToken;
            
            return (
              <g key={i}>
                {/* Token box */}
                <g className={`token-highlight token-${i}`}>
                  <rect
                    x={x}
                    y="100"
                    width="80"
                    height="50"
                    rx="8"
                    fill={isSelected ? '#10b981' : '#1e3a5f'}
                    fillOpacity={isSelected ? 0.5 : 0.3}
                    stroke={isSelected ? '#34d399' : '#3b82f6'}
                    strokeWidth={isSelected ? 3 : 1}
                  />
                  <text
                    x={x + 40}
                    y="130"
                    textAnchor="middle"
                    fill={isSelected ? '#34d399' : 'white'}
                    fontSize="12"
                    fontWeight={isSelected ? 'bold' : 'normal'}
                  >
                    {token}
                  </text>
                </g>

                {/* Attention lines from selected token */}
                {i !== selectedToken && (
                  <g className="attention-line" style={{ opacity: 0 }}>
                    <line
                      x1={80 + selectedToken * 110 + 40}
                      y1={selectedToken < i ? 150 : 100}
                      x2={x + 40}
                      y2={selectedToken < i ? 100 : 150}
                      stroke="#34d399"
                      strokeWidth="2"
                      strokeDasharray={Math.abs(i - selectedToken) > 2 ? "4,2" : "none"}
                    />
                    {/* Arrow head */}
                    <circle
                      cx={x + 40}
                      cy={selectedToken < i ? 100 : 150}
                      r="4"
                      fill="#34d399"
                    />
                  </g>
                )}
              </g>
            );
          })}

          {/* Labels */}
          <text x="400" y="40" textAnchor="middle" fill="#34d399" fontSize="14" fontWeight="bold">
            Bidirectional: "{tokens[selectedToken]}" sees ALL tokens
          </text>
          <text x="400" y="200" textAnchor="middle" fill="#94a3b8" fontSize="11">
            Every token has full context of the entire sequence
          </text>
        </svg>
      </div>

      {/* Comparison: Causal vs Bidirectional */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-blue-500/10 rounded-xl p-6 border border-blue-500/30">
          <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-4 flex items-center gap-2">
            <ArrowLeftRight size={20} />
            CLIP: Causal (Unidirectional)
          </h3>
          
          <div className="bg-black/40 rounded-lg p-4 mb-4">
            <div className="font-mono text-xs text-gray-800 dark:text-gray-400 mb-2">Attention Mask:</div>
            <div className="grid grid-cols-6 gap-1 text-xs">
              {tokens.map((_, i) => (
                <div key={i} className="flex flex-col gap-1">
                  {tokens.map((_, j) => (
                    <div
                      key={j}
                      className={`w-full aspect-square rounded flex items-center justify-center ${
                        j <= i ? 'bg-blue-500/50 text-white' : 'bg-gray-800 text-gray-800 dark:text-gray-600'
                      }`}
                    >
                      {j <= i ? '1' : '0'}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
          
          <p className="text-sm text-gray-700 dark:text-gray-300">
            Token can only see <strong>itself and previous tokens</strong>.
            Good for generation, but limits understanding.
          </p>
        </div>

        <div className="bg-emerald-500/10 rounded-xl p-6 border border-emerald-500/30">
          <h3 className="font-semibold text-emerald-600 dark:text-emerald-400 mb-4 flex items-center gap-2">
            <Eye size={20} />
            T5: Bidirectional
          </h3>
          
          <div className="bg-black/40 rounded-lg p-4 mb-4">
            <div className="font-mono text-xs text-gray-800 dark:text-gray-400 mb-2">Attention Mask:</div>
            <div className="grid grid-cols-6 gap-1 text-xs">
              {tokens.map((_, i) => (
                <div key={i} className="flex flex-col gap-1">
                  {tokens.map((_, j) => (
                    <div
                      key={j}
                      className="w-full aspect-square rounded flex items-center justify-center bg-emerald-500/50 text-white"
                    >
                      1
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
          
          <p className="text-sm text-gray-700 dark:text-gray-300">
            Every token sees <strong>all other tokens</strong>.
            Full context for understanding complex relationships.
          </p>
        </div>
      </div>

      {/* Why Bidirectional Matters */}
      <div className="bg-gradient-to-r from-emerald-500/10 to-teal-500/10 rounded-xl p-6 border border-emerald-500/30">
        <h3 className="font-semibold text-emerald-600 dark:text-emerald-400 mb-4">🎯 Why Bidirectional Matters for SD3</h3>
        
        <div className="space-y-4">
          <div className="bg-black/30 rounded-lg p-4">
            <div className="font-semibold text-white mb-2">Example: "A red ball to the LEFT of a blue cube"</div>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div className="text-gray-700 dark:text-gray-300">
                <span className="text-blue-600 dark:text-blue-400">CLIP (causal):</span> When processing "LEFT", 
                it hasn't seen "blue cube" yet. Harder to understand the spatial relationship.
              </div>
              <div className="text-gray-700 dark:text-gray-300">
                <span className="text-emerald-600 dark:text-emerald-400">T5 (bidirectional):</span> "LEFT" can attend to 
                both "red ball" AND "blue cube" simultaneously. Full spatial context!
              </div>
            </div>
          </div>

          <div className="bg-black/30 rounded-lg p-4">
            <div className="font-semibold text-white mb-2">Example: "not a dog but a cat"</div>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div className="text-gray-700 dark:text-gray-300">
                <span className="text-blue-600 dark:text-blue-400">CLIP:</span> "dog" is encoded before seeing "but a cat", 
                potentially missing the negation context.
              </div>
              <div className="text-gray-700 dark:text-gray-300">
                <span className="text-emerald-600 dark:text-emerald-400">T5:</span> "dog" sees the full context including "not" 
                and "but a cat", properly encoding the negation.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Relative Position Bias */}
      <div className="bg-teal-500/10 rounded-xl p-6 border border-teal-500/30">
        <h3 className="font-semibold text-teal-600 dark:text-teal-400 mb-3">📍 Relative Position Bias</h3>
        <div className="text-gray-700 dark:text-sm space-y-2">
          <p>
            T5 uses <strong>relative position biases</strong> instead of absolute position embeddings:
          </p>
          <div className="bg-black/40 rounded p-4 font-mono text-xs">
            Attention(Q, K) = softmax((QKᵀ + B) / √d) V<br/>
            <span className="text-teal-600 dark:text-teal-400">where B = learned_bias(pos_i - pos_j)</span>
          </div>
          <p>
            This means T5 can generalize to longer sequences than it was trained on, 
            and the attention patterns are translation-invariant.
          </p>
        </div>
      </div>
    </div>
  );
}

export default BidirectionalPanel;
