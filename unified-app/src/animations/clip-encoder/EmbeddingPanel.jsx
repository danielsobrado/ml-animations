import React, { useState, useRef, useEffect } from 'react';
import { Zap, ArrowRight, Play, RotateCcw, Circle, Square } from 'lucide-react';
import gsap from 'gsap';

function EmbeddingPanel() {
  const [showPooled, setShowPooled] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);
  const animationRef = useRef(null);
  const timelineRef = useRef(null);

  useEffect(() => {
    return () => {
      if (timelineRef.current) {
        timelineRef.current.kill();
      }
    };
  }, []);

  const playEmbeddingAnimation = () => {
    if (!animationRef.current) return;
    setIsAnimating(true);

    const ctx = gsap.context(() => {
      timelineRef.current = gsap.timeline({
        onComplete: () => {
          setIsAnimating(false);
          setShowPooled(true);
        }
      });

      // Token embeddings appear
      timelineRef.current.fromTo('.token-embed',
        { opacity: 0, y: -20 },
        { opacity: 1, y: 0, stagger: 0.1, duration: 0.3 }
      );

      // Position embeddings add
      timelineRef.current.fromTo('.pos-embed',
        { opacity: 0, scale: 0 },
        { opacity: 1, scale: 1, stagger: 0.1, duration: 0.2 }
      );

      // Combine
      timelineRef.current.to('.combined-embed', {
        opacity: 1,
        duration: 0.5
      });

      // Flow through transformer
      timelineRef.current.fromTo('.transformer-flow',
        { scaleY: 0, transformOrigin: 'top' },
        { scaleY: 1, duration: 0.8, ease: 'power2.inOut' }
      );

      // Final outputs
      timelineRef.current.fromTo('.final-embed',
        { opacity: 0, x: 20 },
        { opacity: 1, x: 0, stagger: 0.1, duration: 0.3 }
      );

      // Pooled highlight
      timelineRef.current.to('.pooled-highlight', {
        opacity: 1,
        scale: 1.1,
        duration: 0.4,
        ease: 'back.out(1.7)'
      });

    }, animationRef);
  };

  const resetAnimation = () => {
    if (timelineRef.current) {
      timelineRef.current.kill();
    }
    
    const ctx = gsap.context(() => {
      gsap.set('.token-embed', { opacity: 0, y: -20 });
      gsap.set('.pos-embed', { opacity: 0, scale: 0 });
      gsap.set('.combined-embed', { opacity: 0 });
      gsap.set('.transformer-flow', { scaleY: 0 });
      gsap.set('.final-embed', { opacity: 0, x: 20 });
      gsap.set('.pooled-highlight', { opacity: 0, scale: 1 });
    }, animationRef);

    setIsAnimating(false);
    setShowPooled(false);
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-blue-600 dark:text-blue-400 mb-2">CLIP Embeddings & Outputs</h2>
        <p className="text-gray-700 dark:text-gray-300 max-w-3xl mx-auto">
          CLIP produces <strong>two types of outputs</strong>: sequence embeddings (one per token) 
          and a pooled embedding (single vector from [EOS] token). SD3 uses both!
        </p>
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-4">
        <button
          onClick={playEmbeddingAnimation}
          disabled={isAnimating}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 rounded-lg transition-colors"
        >
          <Play size={18} />
          Show Embedding Flow
        </button>
        <button
          onClick={resetAnimation}
          className="flex items-center gap-2 px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <RotateCcw size={18} />
          Reset
        </button>
      </div>

      {/* Animation */}
      <div ref={animationRef} className="bg-black/40 rounded-xl p-6">
        <svg viewBox="0 0 900 400" className="w-full h-auto">
          {/* Stage 1: Token IDs */}
          <text x="100" y="30" textAnchor="middle" fill="#94a3b8" fontSize="12">Token IDs</text>
          {[49406, 320, 2368, 525, 4044, 49407].map((id, i) => (
            <g key={i}>
              <rect x={30 + i * 70} y="40" width="60" height="30" rx="4" fill="#1e3a5f" stroke="#3b82f6" strokeWidth="1" />
              <text x={60 + i * 70} y="60" textAnchor="middle" fill="#60a5fa" fontSize="10">{id}</text>
            </g>
          ))}

          {/* Stage 2: Token Embeddings */}
          <text x="100" y="100" textAnchor="middle" fill="#94a3b8" fontSize="12">Token Embeds</text>
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <g key={i} className="token-embed" style={{ opacity: 0 }}>
              <rect x={30 + i * 70} y="110" width="60" height="30" rx="4" fill="#3b82f6" fillOpacity="0.3" stroke="#3b82f6" strokeWidth="1" />
              <text x={60 + i * 70} y="130" textAnchor="middle" fill="white" fontSize="9">768-d</text>
            </g>
          ))}

          {/* Plus signs */}
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <text key={i} x={60 + i * 70} y="155" textAnchor="middle" fill="#60a5fa" fontSize="16">+</text>
          ))}

          {/* Stage 3: Position Embeddings */}
          <text x="100" y="175" textAnchor="middle" fill="#94a3b8" fontSize="12">Position Embeds</text>
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <g key={i} className="pos-embed" style={{ opacity: 0 }}>
              <rect x={30 + i * 70} y="185" width="60" height="30" rx="4" fill="#10b981" fillOpacity="0.3" stroke="#10b981" strokeWidth="1" />
              <text x={60 + i * 70} y="205" textAnchor="middle" fill="white" fontSize="9">pos_{i}</text>
            </g>
          ))}

          {/* Stage 4: Combined */}
          <g className="combined-embed" style={{ opacity: 0 }}>
            <text x="100" y="250" textAnchor="middle" fill="#94a3b8" fontSize="12">Combined</text>
            <rect x="30" y="260" width="420" height="30" rx="4" fill="#8b5cf6" fillOpacity="0.2" stroke="#8b5cf6" strokeWidth="2" />
            <text x="240" y="280" textAnchor="middle" fill="#a78bfa" fontSize="11">[batch, 77, 768] - Ready for Transformer</text>
          </g>

          {/* Transformer Processing */}
          <g className="transformer-flow" style={{ opacity: 1 }}>
            <rect x="500" y="40" width="150" height="260" rx="8" fill="#ec4899" fillOpacity="0.1" stroke="#ec4899" strokeWidth="2" />
            <text x="575" y="65" textAnchor="middle" fill="#f472b6" fontSize="12" fontWeight="bold">Transformer</text>
            <text x="575" y="85" textAnchor="middle" fill="#f472b6" fontSize="10">12 Layers</text>
            
            {[...Array(6)].map((_, i) => (
              <rect key={i} x="520" y={100 + i * 30} width="110" height="22" rx="2" fill="#ec4899" fillOpacity={0.2 + i * 0.1} />
            ))}
          </g>

          {/* Connection arrow */}
          <path d="M 450 275 L 490 275 L 490 170 L 500 170" stroke="#60a5fa" strokeWidth="2" fill="none" strokeDasharray="4,2" />

          {/* Final Outputs */}
          <text x="750" y="100" textAnchor="middle" fill="#94a3b8" fontSize="12">Outputs</text>
          
          {/* Sequence output */}
          <g className="final-embed" style={{ opacity: 0 }}>
            <rect x="680" y="120" width="140" height="50" rx="8" fill="#3b82f6" fillOpacity="0.2" stroke="#3b82f6" strokeWidth="2" />
            <text x="750" y="145" textAnchor="middle" fill="#60a5fa" fontSize="11" fontWeight="bold">Sequence</text>
            <text x="750" y="162" textAnchor="middle" fill="white" fontSize="10">[1, 77, 768]</text>
          </g>

          {/* Pooled output */}
          <g className="final-embed pooled-highlight" style={{ opacity: 0 }}>
            <rect x="680" y="190" width="140" height="50" rx="8" fill="#f59e0b" fillOpacity="0.2" stroke="#f59e0b" strokeWidth="2" />
            <text x="750" y="215" textAnchor="middle" fill="#fbbf24" fontSize="11" fontWeight="bold">Pooled</text>
            <text x="750" y="232" textAnchor="middle" fill="white" fontSize="10">[1, 768]</text>
          </g>

          {/* Connection from transformer */}
          <path d="M 650 170 L 680 145" stroke="#3b82f6" strokeWidth="2" />
          <path d="M 650 170 L 680 215" stroke="#f59e0b" strokeWidth="2" />

          {/* Legend */}
          <g transform="translate(680, 280)">
            <text x="0" y="0" fill="#94a3b8" fontSize="10">Used in SD3:</text>
            <rect x="0" y="10" width="12" height="12" fill="#3b82f6" rx="2" />
            <text x="18" y="20" fill="#60a5fa" fontSize="9">→ Joint Attention</text>
            <rect x="0" y="30" width="12" height="12" fill="#f59e0b" rx="2" />
            <text x="18" y="40" fill="#fbbf24" fontSize="9">→ DiT Conditioning</text>
          </g>
        </svg>
      </div>

      {/* Output Explanation */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-blue-500/10 rounded-xl p-6 border border-blue-500/30">
          <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-3 flex items-center gap-2">
            <Square size={18} />
            Sequence Embeddings
          </h3>
          <div className="space-y-3 text-sm text-gray-700 dark:text-gray-300">
            <p>
              <strong>Shape:</strong> <code className="text-blue-600 dark:text-blue-400">[batch, 77, 768]</code>
            </p>
            <p>
              One 768-dimensional vector for each of the 77 token positions.
              These capture the meaning of each token in context.
            </p>
            <div className="bg-black/40 rounded p-3 mt-2">
              <span className="text-blue-600 dark:text-blue-400 font-semibold">Used for:</span> Joint attention with image tokens.
              The text sequence gets concatenated with image patch tokens.
            </div>
          </div>
        </div>

        <div className="bg-yellow-500/10 rounded-xl p-6 border border-yellow-500/30">
          <h3 className="font-semibold text-yellow-400 mb-3 flex items-center gap-2">
            <Circle size={18} />
            Pooled Embedding
          </h3>
          <div className="space-y-3 text-sm text-gray-700 dark:text-gray-300">
            <p>
              <strong>Shape:</strong> <code className="text-yellow-400">[batch, 768]</code>
            </p>
            <p>
              The hidden state at the [EOS] token position. This single vector 
              summarizes the entire text prompt.
            </p>
            <div className="bg-black/40 rounded p-3 mt-2">
              <span className="text-yellow-400 font-semibold">Used for:</span> AdaLN conditioning in DiT blocks.
              Combined with timestep embedding to modulate transformer layers.
            </div>
          </div>
        </div>
      </div>

      {/* How SD3 Combines CLIP Outputs */}
      <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-xl p-6 border border-purple-500/30">
        <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-4">How SD3 Combines Both CLIPs</h3>
        
        <div className="space-y-4">
          <div className="bg-black/30 rounded-lg p-4">
            <h4 className="text-blue-600 dark:text-blue-400 font-semibold mb-2">Sequence Embeddings (for Joint Attention)</h4>
            <div className="font-mono text-sm text-gray-700 dark:text-gray-300">
              CLIP-L: [1, 77, 768]<br />
              CLIP-G: [1, 77, 1280]<br />
              <span className="text-purple-600 dark:text-purple-400">Concatenated → [1, 77, 2048]</span>
            </div>
          </div>

          <div className="bg-black/30 rounded-lg p-4">
            <h4 className="text-yellow-400 font-semibold mb-2">Pooled Embeddings (for Conditioning)</h4>
            <div className="font-mono text-sm text-gray-700 dark:text-gray-300">
              CLIP-L pooled: [1, 768]<br />
              CLIP-G pooled: [1, 1280]<br />
              <span className="text-yellow-400">Concatenated → [1, 2048]</span><br />
              <span className="text-gray-700 dark:text-gray-500">Then projected to match DiT hidden size</span>
            </div>
          </div>
        </div>
      </div>

      {/* Dimension Summary */}
      <div className="bg-black/30 rounded-xl p-6">
        <h3 className="font-semibold text-gray-700 dark:text-gray-300 mb-4">📊 Dimension Summary</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left py-2 px-3 text-gray-800 dark:text-gray-400">Stage</th>
                <th className="text-left py-2 px-3 text-gray-800 dark:text-gray-400">Shape</th>
                <th className="text-left py-2 px-3 text-gray-800 dark:text-gray-400">Description</th>
              </tr>
            </thead>
            <tbody className="text-gray-700 dark:text-gray-300">
              <tr className="border-b border-white/5">
                <td className="py-2 px-3">Token IDs</td>
                <td className="py-2 px-3 font-mono text-blue-600 dark:text-blue-400">[B, 77]</td>
                <td className="py-2 px-3">Integer token IDs (0-49407)</td>
              </tr>
              <tr className="border-b border-white/5">
                <td className="py-2 px-3">Token + Pos Embed</td>
                <td className="py-2 px-3 font-mono text-purple-600 dark:text-purple-400">[B, 77, 768]</td>
                <td className="py-2 px-3">Combined embeddings</td>
              </tr>
              <tr className="border-b border-white/5">
                <td className="py-2 px-3">After Transformer</td>
                <td className="py-2 px-3 font-mono text-pink-600 dark:text-pink-400">[B, 77, 768]</td>
                <td className="py-2 px-3">Contextualized embeddings</td>
              </tr>
              <tr className="border-b border-white/5">
                <td className="py-2 px-3">Pooled</td>
                <td className="py-2 px-3 font-mono text-yellow-400">[B, 768]</td>
                <td className="py-2 px-3">EOS token hidden state</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default EmbeddingPanel;
