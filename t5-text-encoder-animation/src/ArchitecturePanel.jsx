import React, { useState, useEffect, useRef } from 'react';
import { Play, RotateCcw, Layers, ArrowRight } from 'lucide-react';
import gsap from 'gsap';

function ArchitecturePanel() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [activeBlock, setActiveBlock] = useState(null);
  const animationRef = useRef(null);
  const timelineRef = useRef(null);

  const blockComponents = [
    { name: 'Self-Attention', color: '#10b981', desc: 'Bidirectional attention over all tokens' },
    { name: 'LayerNorm', color: '#06b6d4', desc: 'Pre-LN normalization' },
    { name: 'Feed-Forward', color: '#8b5cf6', desc: 'MLP with GEGLU activation' },
    { name: 'Residual', color: '#f59e0b', desc: 'Skip connections for gradient flow' },
  ];

  useEffect(() => {
    if (!animationRef.current) return;

    const ctx = gsap.context(() => {
      gsap.set('.arch-block', { opacity: 0, x: -20 });
      gsap.set('.arch-arrow', { scaleX: 0, transformOrigin: 'left' });
      gsap.set('.output-box', { opacity: 0, scale: 0.8 });
    }, animationRef);

    return () => ctx.revert();
  }, []);

  const playAnimation = () => {
    if (!animationRef.current) return;
    setIsPlaying(true);

    const ctx = gsap.context(() => {
      timelineRef.current = gsap.timeline({
        onComplete: () => setIsPlaying(false)
      });

      // Blocks appear
      timelineRef.current.to('.arch-block', {
        opacity: 1,
        x: 0,
        stagger: 0.1,
        duration: 0.4,
        ease: 'power2.out'
      });

      // Arrows
      timelineRef.current.to('.arch-arrow', {
        scaleX: 1,
        stagger: 0.1,
        duration: 0.3
      }, '-=0.3');

      // Output
      timelineRef.current.to('.output-box', {
        opacity: 1,
        scale: 1,
        duration: 0.4,
        ease: 'back.out(1.7)'
      });

    }, animationRef);
  };

  const resetAnimation = () => {
    if (timelineRef.current) timelineRef.current.kill();
    
    const ctx = gsap.context(() => {
      gsap.set('.arch-block', { opacity: 0, x: -20 });
      gsap.set('.arch-arrow', { scaleX: 0 });
      gsap.set('.output-box', { opacity: 0, scale: 0.8 });
    }, animationRef);

    setIsPlaying(false);
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-emerald-400 mb-2">T5-XXL Encoder Architecture</h2>
        <p className="text-gray-300 max-w-3xl mx-auto">
          T5's encoder uses 24 transformer blocks with pre-layer normalization and GEGLU activations.
          Each block has bidirectional self-attention allowing full context understanding.
        </p>
      </div>

      {/* Architecture Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-black/30 rounded-lg p-4 text-center">
          <div className="text-xl font-bold text-emerald-400">24</div>
          <div className="text-sm text-gray-400">Encoder Layers</div>
        </div>
        <div className="bg-black/30 rounded-lg p-4 text-center">
          <div className="text-xl font-bold text-teal-400">4096</div>
          <div className="text-sm text-gray-400">Hidden Dim</div>
        </div>
        <div className="bg-black/30 rounded-lg p-4 text-center">
          <div className="text-xl font-bold text-cyan-400">64</div>
          <div className="text-sm text-gray-400">Attention Heads</div>
        </div>
        <div className="bg-black/30 rounded-lg p-4 text-center">
          <div className="text-xl font-bold text-blue-400">10240</div>
          <div className="text-sm text-gray-400">FFN Dim</div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-4">
        <button
          onClick={playAnimation}
          disabled={isPlaying}
          className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 rounded-lg transition-colors"
        >
          <Play size={18} />
          Animate Architecture
        </button>
        <button
          onClick={resetAnimation}
          className="flex items-center gap-2 px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <RotateCcw size={18} />
          Reset
        </button>
      </div>

      {/* Architecture Visualization */}
      <div ref={animationRef} className="bg-black/40 rounded-xl p-6">
        <svg viewBox="0 0 850 350" className="w-full h-auto">
          {/* Input */}
          <g className="arch-block">
            <rect x="20" y="140" width="80" height="60" rx="6" fill="#10b981" fillOpacity="0.2" stroke="#10b981" strokeWidth="2" />
            <text x="60" y="165" textAnchor="middle" fill="#34d399" fontSize="10">Input</text>
            <text x="60" y="180" textAnchor="middle" fill="white" fontSize="9">Embeddings</text>
          </g>

          {/* Arrow */}
          <line className="arch-arrow" x1="105" y1="170" x2="145" y2="170" stroke="#34d399" strokeWidth="2" />

          {/* Encoder Block (detailed) */}
          <g className="arch-block">
            <rect x="150" y="60" width="280" height="220" rx="8" fill="#1e293b" stroke="#475569" strokeWidth="2" />
            <text x="290" y="85" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">Encoder Block × 24</text>
            
            {/* Pre-LN */}
            <rect x="170" y="100" width="100" height="30" rx="4" fill="#06b6d4" fillOpacity="0.3" stroke="#06b6d4" />
            <text x="220" y="120" textAnchor="middle" fill="#22d3ee" fontSize="9">LayerNorm</text>
            
            {/* Self-Attention */}
            <rect x="170" y="140" width="240" height="40" rx="4" fill="#10b981" fillOpacity="0.2" stroke="#10b981" strokeWidth="2" />
            <text x="290" y="165" textAnchor="middle" fill="#34d399" fontSize="10">Bidirectional Self-Attention</text>
            
            {/* Add & Norm */}
            <circle cx="395" y="115" r="15" fill="#f59e0b" fillOpacity="0.3" stroke="#f59e0b" />
            <text x="395" y="120" textAnchor="middle" fill="#fbbf24" fontSize="12">+</text>
            
            {/* Pre-LN 2 */}
            <rect x="170" y="195" width="100" height="30" rx="4" fill="#06b6d4" fillOpacity="0.3" stroke="#06b6d4" />
            <text x="220" y="215" textAnchor="middle" fill="#22d3ee" fontSize="9">LayerNorm</text>
            
            {/* FFN */}
            <rect x="170" y="235" width="240" height="35" rx="4" fill="#8b5cf6" fillOpacity="0.2" stroke="#8b5cf6" strokeWidth="2" />
            <text x="290" y="258" textAnchor="middle" fill="#a78bfa" fontSize="10">Feed-Forward (GEGLU)</text>
            
            {/* Add */}
            <circle cx="395" y="210" r="15" fill="#f59e0b" fillOpacity="0.3" stroke="#f59e0b" />
            <text x="395" y="215" textAnchor="middle" fill="#fbbf24" fontSize="12">+</text>

            {/* Residual connections */}
            <path d="M 150 170 L 140 170 L 140 115 L 380 115" stroke="#f59e0b" strokeWidth="1" fill="none" strokeDasharray="3,2" />
            <path d="M 150 250 L 140 250 L 140 210 L 380 210" stroke="#f59e0b" strokeWidth="1" fill="none" strokeDasharray="3,2" />
          </g>

          {/* Arrow */}
          <line className="arch-arrow" x1="435" y1="170" x2="475" y2="170" stroke="#a78bfa" strokeWidth="2" />

          {/* Final LayerNorm */}
          <g className="arch-block">
            <rect x="480" y="140" width="100" height="60" rx="6" fill="#06b6d4" fillOpacity="0.3" stroke="#06b6d4" strokeWidth="2" />
            <text x="530" y="165" textAnchor="middle" fill="#22d3ee" fontSize="10">Final</text>
            <text x="530" y="180" textAnchor="middle" fill="#22d3ee" fontSize="10">LayerNorm</text>
          </g>

          {/* Arrow */}
          <line className="arch-arrow" x1="585" y1="170" x2="625" y2="170" stroke="#22d3ee" strokeWidth="2" />

          {/* Output */}
          <g className="output-box">
            <rect x="630" y="120" width="120" height="100" rx="8" fill="#f59e0b" fillOpacity="0.2" stroke="#f59e0b" strokeWidth="2" />
            <text x="690" y="150" textAnchor="middle" fill="#fbbf24" fontSize="11" fontWeight="bold">Output</text>
            <text x="690" y="170" textAnchor="middle" fill="white" fontSize="10">[B, seq, 4096]</text>
            <text x="690" y="195" textAnchor="middle" fill="#94a3b8" fontSize="8">Contextualized</text>
            <text x="690" y="207" textAnchor="middle" fill="#94a3b8" fontSize="8">Embeddings</text>
          </g>

          {/* Legend */}
          <g transform="translate(20, 300)">
            {blockComponents.map((comp, i) => (
              <g key={i} transform={`translate(${i * 200}, 0)`}>
                <rect x="0" y="0" width="12" height="12" rx="2" fill={comp.color} fillOpacity="0.5" />
                <text x="18" y="10" fill={comp.color} fontSize="9">{comp.name}</text>
              </g>
            ))}
          </g>
        </svg>
      </div>

      {/* Block Components */}
      <div className="grid md:grid-cols-2 gap-4">
        {blockComponents.map((comp, i) => (
          <div
            key={i}
            className={`bg-black/30 rounded-lg p-4 border cursor-pointer transition-all ${
              activeBlock === i ? 'border-white/50 bg-white/5' : 'border-white/10 hover:border-white/30'
            }`}
            onClick={() => setActiveBlock(activeBlock === i ? null : i)}
          >
            <div className="flex items-center gap-3 mb-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: comp.color }} />
              <span className="font-semibold text-white">{comp.name}</span>
            </div>
            <p className="text-sm text-gray-400">{comp.desc}</p>
          </div>
        ))}
      </div>

      {/* GEGLU Explanation */}
      <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-xl p-6 border border-purple-500/30">
        <h3 className="font-semibold text-purple-400 mb-3">🔧 GEGLU Activation</h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div className="text-gray-300">
            <p className="mb-2">
              T5 uses <strong>GEGLU</strong> (Gaussian Error Gated Linear Unit) instead of ReLU:
            </p>
            <div className="bg-black/40 rounded p-3 font-mono text-xs">
              GEGLU(x) = GELU(xW₁) ⊙ (xW₂)
            </div>
          </div>
          <div className="text-gray-300">
            <p className="mb-2">Benefits:</p>
            <ul className="list-disc list-inside space-y-1 text-gray-400">
              <li>Better gradient flow</li>
              <li>Smoother activations</li>
              <li>Improved training stability</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Pre-LN vs Post-LN */}
      <div className="bg-teal-500/10 rounded-xl p-6 border border-teal-500/30">
        <h3 className="font-semibold text-teal-400 mb-3">📐 Pre-Layer Normalization</h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-300">
          <div className="bg-black/30 rounded p-4">
            <div className="font-semibold text-gray-400 mb-2">Post-LN (original Transformer)</div>
            <div className="font-mono text-xs">
              y = LayerNorm(x + Attention(x))
            </div>
            <p className="text-xs text-gray-500 mt-2">Norm after residual - harder to train</p>
          </div>
          <div className="bg-black/30 rounded p-4">
            <div className="font-semibold text-teal-400 mb-2">Pre-LN (T5)</div>
            <div className="font-mono text-xs">
              y = x + Attention(LayerNorm(x))
            </div>
            <p className="text-xs text-gray-500 mt-2">Norm before attention - more stable training</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ArchitecturePanel;
