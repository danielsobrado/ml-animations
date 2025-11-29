import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Layers, ArrowRight } from 'lucide-react';
import gsap from 'gsap';

function TransformerPanel() {
  const [activeLayer, setActiveLayer] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const animationRef = useRef(null);
  const timelineRef = useRef(null);

  const layers = [
    { name: 'Self-Attention', color: '#3b82f6', description: 'Tokens attend to each other' },
    { name: 'Layer Norm', color: '#10b981', description: 'Normalize activations' },
    { name: 'MLP', color: '#8b5cf6', description: 'Feed-forward processing' },
    { name: 'Layer Norm', color: '#10b981', description: 'Final normalization' },
  ];

  useEffect(() => {
    if (!animationRef.current) return;

    const ctx = gsap.context(() => {
      // Initial setup
      gsap.set('.attention-line', { scaleX: 0, transformOrigin: 'left' });
      gsap.set('.mlp-node', { scale: 0 });
    }, animationRef);

    return () => ctx.revert();
  }, []);

  const playAttentionAnimation = () => {
    if (!animationRef.current) return;
    setIsPlaying(true);

    const ctx = gsap.context(() => {
      timelineRef.current = gsap.timeline({
        onComplete: () => setIsPlaying(false)
      });

      // Attention lines animate
      timelineRef.current.to('.attention-line', {
        scaleX: 1,
        stagger: 0.05,
        duration: 0.3,
        ease: 'power2.out'
      });

      // MLP processing
      timelineRef.current.to('.mlp-node', {
        scale: 1,
        stagger: 0.1,
        duration: 0.2,
        ease: 'back.out(1.7)'
      }, '+=0.2');

      // Highlight output
      timelineRef.current.to('.output-token', {
        scale: 1.1,
        duration: 0.3,
        ease: 'power2.out'
      }, '+=0.2');

      timelineRef.current.to('.output-token', {
        scale: 1,
        duration: 0.2
      });

    }, animationRef);
  };

  const resetAnimation = () => {
    if (timelineRef.current) {
      timelineRef.current.kill();
    }
    
    const ctx = gsap.context(() => {
      gsap.set('.attention-line', { scaleX: 0 });
      gsap.set('.mlp-node', { scale: 0 });
      gsap.set('.output-token', { scale: 1 });
    }, animationRef);

    setIsPlaying(false);
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-blue-400 mb-2">CLIP Transformer Architecture</h2>
        <p className="text-gray-300 max-w-3xl mx-auto">
          CLIP's text encoder uses a standard Transformer architecture with <strong>causal self-attention</strong> 
          (each token can only attend to previous tokens). CLIP-L has 12 layers, CLIP-G has 32 layers.
        </p>
      </div>

      {/* Architecture Comparison */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-blue-500/10 rounded-xl p-4 border border-blue-500/30">
          <h3 className="font-semibold text-blue-400 mb-3">CLIP-L/14 Architecture</h3>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="bg-black/30 rounded px-3 py-2">
              <span className="text-gray-400">Layers:</span>
              <span className="text-white ml-2">12</span>
            </div>
            <div className="bg-black/30 rounded px-3 py-2">
              <span className="text-gray-400">Hidden:</span>
              <span className="text-white ml-2">768</span>
            </div>
            <div className="bg-black/30 rounded px-3 py-2">
              <span className="text-gray-400">Heads:</span>
              <span className="text-white ml-2">12</span>
            </div>
            <div className="bg-black/30 rounded px-3 py-2">
              <span className="text-gray-400">MLP:</span>
              <span className="text-white ml-2">3072</span>
            </div>
          </div>
        </div>
        <div className="bg-purple-500/10 rounded-xl p-4 border border-purple-500/30">
          <h3 className="font-semibold text-purple-400 mb-3">CLIP-G/14 Architecture</h3>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="bg-black/30 rounded px-3 py-2">
              <span className="text-gray-400">Layers:</span>
              <span className="text-white ml-2">32</span>
            </div>
            <div className="bg-black/30 rounded px-3 py-2">
              <span className="text-gray-400">Hidden:</span>
              <span className="text-white ml-2">1280</span>
            </div>
            <div className="bg-black/30 rounded px-3 py-2">
              <span className="text-gray-400">Heads:</span>
              <span className="text-white ml-2">20</span>
            </div>
            <div className="bg-black/30 rounded px-3 py-2">
              <span className="text-gray-400">MLP:</span>
              <span className="text-white ml-2">5120</span>
            </div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-4">
        <button
          onClick={playAttentionAnimation}
          disabled={isPlaying}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 rounded-lg transition-colors"
        >
          <Play size={18} />
          Play Attention Flow
        </button>
        <button
          onClick={resetAnimation}
          className="flex items-center gap-2 px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <RotateCcw size={18} />
          Reset
        </button>
      </div>

      {/* Single Layer Visualization */}
      <div ref={animationRef} className="bg-black/40 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-center text-gray-300 mb-4">
          Single Transformer Layer
        </h3>
        
        <svg viewBox="0 0 800 300" className="w-full h-auto">
          {/* Input Tokens */}
          <text x="80" y="30" textAnchor="middle" fill="#94a3b8" fontSize="12">Input Tokens</text>
          {['[BOS]', 'A', 'cat', 'on', 'mat', '[EOS]'].map((token, i) => (
            <g key={i}>
              <rect x={30 + i * 90} y="40" width="70" height="35" rx="4" fill="#1e40af" fillOpacity="0.5" stroke="#3b82f6" strokeWidth="1" />
              <text x={65 + i * 90} y="62" textAnchor="middle" fill="white" fontSize="11">{token}</text>
            </g>
          ))}

          {/* Self-Attention Block */}
          <rect x="30" y="95" width="520" height="60" rx="8" fill="#3b82f6" fillOpacity="0.2" stroke="#3b82f6" strokeWidth="2" />
          <text x="290" y="115" textAnchor="middle" fill="#60a5fa" fontSize="12" fontWeight="bold">Causal Self-Attention</text>
          
          {/* Attention lines (causal - can only see previous) */}
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <g key={i}>
              {[...Array(i + 1)].map((_, j) => (
                <line
                  key={j}
                  className="attention-line"
                  x1={65 + j * 90}
                  y1="75"
                  x2={65 + i * 90}
                  y2="95"
                  stroke={`rgba(59, 130, 246, ${0.3 + (j === i ? 0.5 : 0)})`}
                  strokeWidth={j === i ? 2 : 1}
                />
              ))}
            </g>
          ))}

          {/* Layer Norm 1 */}
          <rect x="570" y="95" width="80" height="30" rx="4" fill="#10b981" fillOpacity="0.3" stroke="#10b981" strokeWidth="1" />
          <text x="610" y="115" textAnchor="middle" fill="#34d399" fontSize="10">LayerNorm</text>
          
          {/* Residual arrow 1 */}
          <path d="M 550 125 L 560 125" stroke="#60a5fa" strokeWidth="2" markerEnd="url(#arrow)" />

          {/* MLP Block */}
          <rect x="30" y="170" width="520" height="50" rx="8" fill="#8b5cf6" fillOpacity="0.2" stroke="#8b5cf6" strokeWidth="2" />
          <text x="290" y="190" textAnchor="middle" fill="#a78bfa" fontSize="12" fontWeight="bold">Feed-Forward MLP</text>
          
          {/* MLP nodes */}
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <circle
              key={i}
              className="mlp-node"
              cx={65 + i * 90}
              cy="205"
              r="8"
              fill="#8b5cf6"
            />
          ))}

          {/* Layer Norm 2 */}
          <rect x="570" y="175" width="80" height="30" rx="4" fill="#10b981" fillOpacity="0.3" stroke="#10b981" strokeWidth="1" />
          <text x="610" y="195" textAnchor="middle" fill="#34d399" fontSize="10">LayerNorm</text>

          {/* Output Tokens */}
          <text x="80" y="250" textAnchor="middle" fill="#94a3b8" fontSize="12">Output Tokens</text>
          {['[BOS]', 'A', 'cat', 'on', 'mat', '[EOS]'].map((token, i) => (
            <g key={i} className="output-token">
              <rect x={30 + i * 90} y="260" width="70" height="35" rx="4" fill="#7c3aed" fillOpacity="0.5" stroke="#8b5cf6" strokeWidth="1" />
              <text x={65 + i * 90} y="282" textAnchor="middle" fill="white" fontSize="11">{token}</text>
            </g>
          ))}

          {/* Connections down */}
          <line x1="290" y1="155" x2="290" y2="170" stroke="#60a5fa" strokeWidth="2" strokeDasharray="4,2" />
          <line x1="290" y1="220" x2="290" y2="260" stroke="#a78bfa" strokeWidth="2" strokeDasharray="4,2" />

          <defs>
            <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#60a5fa" />
            </marker>
          </defs>
        </svg>
      </div>

      {/* Causal Attention Explanation */}
      <div className="bg-yellow-500/10 rounded-xl p-6 border border-yellow-500/30">
        <h3 className="font-semibold text-yellow-400 mb-3">🎯 Why Causal (Unidirectional) Attention?</h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-300">
          <div>
            <p className="mb-2">
              <strong>Causal mask:</strong> Each token can only "see" tokens that came before it.
            </p>
            <div className="bg-black/40 rounded p-3 font-mono text-xs">
              <div>Token 0: sees [0]</div>
              <div>Token 1: sees [0, 1]</div>
              <div>Token 2: sees [0, 1, 2]</div>
              <div>...</div>
            </div>
          </div>
          <div>
            <p className="mb-2">
              <strong>This is how CLIP was trained</strong> - as a text model that could also be used for generation.
            </p>
            <p>
              The [EOS] token at the end sees ALL previous tokens, making it a good summary of the whole text.
              This becomes the <span className="text-yellow-400">pooled embedding</span>.
            </p>
          </div>
        </div>
      </div>

      {/* Layer Components */}
      <div className="grid md:grid-cols-4 gap-4">
        {layers.map((layer, i) => (
          <div
            key={i}
            className="bg-black/30 rounded-lg p-4 border border-white/10 cursor-pointer hover:border-white/30 transition-colors"
            onMouseEnter={() => setActiveLayer(i)}
          >
            <div 
              className="w-full h-2 rounded mb-3"
              style={{ backgroundColor: layer.color }}
            />
            <div className="font-semibold text-white text-sm mb-1">{layer.name}</div>
            <div className="text-xs text-gray-400">{layer.description}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default TransformerPanel;
