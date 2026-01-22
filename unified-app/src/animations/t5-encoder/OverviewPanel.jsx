import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, FileText, Zap, Clock, HardDrive } from 'lucide-react';
import gsap from 'gsap';

function OverviewPanel() {
  const [isPlaying, setIsPlaying] = useState(false);
  const animationRef = useRef(null);
  const timelineRef = useRef(null);

  useEffect(() => {
    if (!animationRef.current) return;

    const ctx = gsap.context(() => {
      timelineRef.current = gsap.timeline({ paused: true });

      // Text input
      timelineRef.current.fromTo('.t5-input',
        { opacity: 0, x: -30 },
        { opacity: 1, x: 0, duration: 0.5 }
      );

      // Tokenization
      timelineRef.current.fromTo('.t5-tokens',
        { opacity: 0, scale: 0.8 },
        { opacity: 1, scale: 1, duration: 0.4 }
      );

      // Encoder layers
      timelineRef.current.to('.encoder-stack', { opacity: 1, duration: 0.3 });
      timelineRef.current.fromTo('.encoder-layer',
        { opacity: 0.3, scaleX: 0.8 },
        { opacity: 1, scaleX: 1, stagger: 0.05, duration: 0.15 }
      );

      // Output embeddings
      timelineRef.current.fromTo('.t5-output',
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.4 }
      );

    }, animationRef);

    return () => ctx.revert();
  }, []);

  const handlePlayPause = () => {
    if (!timelineRef.current) return;
    
    if (isPlaying) {
      timelineRef.current.pause();
    } else {
      if (timelineRef.current.progress() === 1) {
        timelineRef.current.restart();
      } else {
        timelineRef.current.play();
      }
    }
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    if (!timelineRef.current) return;
    timelineRef.current.restart().pause();
    setIsPlaying(false);
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-emerald-600 dark:text-emerald-400 mb-2">What is T5?</h2>
        <p className="text-gray-700 dark:text-gray-300 max-w-3xl mx-auto">
          <strong>T5 (Text-to-Text Transfer Transformer)</strong> is Google's powerful language model 
          that treats all NLP tasks as text-to-text problems. SD3 uses the <strong>encoder part only</strong> 
          of T5-XXL for deep language understanding.
        </p>
      </div>

      {/* Key Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-emerald-500/10 rounded-xl p-4 text-center border border-emerald-500/30">
          <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">4.7B</div>
          <div className="text-sm text-gray-800 dark:text-gray-400">Parameters</div>
        </div>
        <div className="bg-teal-500/10 rounded-xl p-4 text-center border border-teal-500/30">
          <div className="text-2xl font-bold text-teal-600 dark:text-teal-400">24</div>
          <div className="text-sm text-gray-800 dark:text-gray-400">Encoder Layers</div>
        </div>
        <div className="bg-cyan-500/10 rounded-xl p-4 text-center border border-cyan-500/30">
          <div className="text-2xl font-bold text-cyan-600 dark:text-cyan-400">4096</div>
          <div className="text-sm text-gray-800 dark:text-gray-400">Hidden Dimension</div>
        </div>
        <div className="bg-blue-500/10 rounded-xl p-4 text-center border border-blue-500/30">
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">256+</div>
          <div className="text-sm text-gray-800 dark:text-gray-400">Max Tokens</div>
        </div>
      </div>

      {/* T5 vs Full Model */}
      <div className="bg-gradient-to-r from-emerald-500/10 to-teal-500/10 rounded-xl p-6 border border-emerald-500/30">
        <h3 className="text-lg font-semibold text-emerald-600 dark:text-emerald-400 mb-3">🔍 SD3 Uses Encoder Only</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-black/30 rounded-lg p-4">
            <div className="font-semibold text-emerald-600 dark:text-emerald-400 mb-2">Full T5 (Encoder-Decoder)</div>
            <div className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <p>• <strong>Encoder:</strong> Understands input text</p>
              <p>• <strong>Decoder:</strong> Generates output text</p>
              <p>• Used for: translation, summarization, Q&A</p>
            </div>
          </div>
          <div className="bg-black/30 rounded-lg p-4">
            <div className="font-semibold text-teal-600 dark:text-teal-400 mb-2">T5 in SD3 (Encoder Only)</div>
            <div className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <p>• Only encoder part is used</p>
              <p>• Decoder is discarded entirely</p>
              <p>• Purpose: Create rich text embeddings</p>
            </div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-4">
        <button
          onClick={handlePlayPause}
          className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 rounded-lg transition-colors"
        >
          {isPlaying ? <Pause size={18} /> : <Play size={18} />}
          {isPlaying ? 'Pause' : 'Play Animation'}
        </button>
        <button
          onClick={handleReset}
          className="flex items-center gap-2 px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <RotateCcw size={18} />
          Reset
        </button>
      </div>

      {/* Animation */}
      <div ref={animationRef} className="bg-black/40 rounded-xl p-6 min-h-[350px]">
        <svg viewBox="0 0 800 300" className="w-full h-auto">
          {/* Input */}
          <g className="t5-input" style={{ opacity: 0 }}>
            <rect x="20" y="120" width="180" height="60" rx="8" fill="#10b981" fillOpacity="0.2" stroke="#10b981" strokeWidth="2" />
            <text x="110" y="145" textAnchor="middle" fill="#34d399" fontSize="10">Long Detailed Prompt</text>
            <text x="110" y="165" textAnchor="middle" fill="white" fontSize="9">"A majestic lion standing</text>
            <text x="110" y="177" textAnchor="middle" fill="white" fontSize="9">on a cliff at sunset..."</text>
          </g>

          {/* Arrow */}
          <path d="M 210 150 L 250 150" stroke="#34d399" strokeWidth="2" markerEnd="url(#t5arrow)" />

          {/* Tokenization */}
          <g className="t5-tokens" style={{ opacity: 0 }}>
            <rect x="260" y="110" width="120" height="80" rx="8" fill="#06b6d4" fillOpacity="0.2" stroke="#06b6d4" strokeWidth="2" />
            <text x="320" y="135" textAnchor="middle" fill="#22d3ee" fontSize="10">SentencePiece</text>
            <text x="320" y="155" textAnchor="middle" fill="white" fontSize="9">Tokenizer</text>
            <text x="320" y="175" textAnchor="middle" fill="#94a3b8" fontSize="8">32,000 vocab</text>
          </g>

          {/* Arrow */}
          <path d="M 390 150 L 430 150" stroke="#22d3ee" strokeWidth="2" markerEnd="url(#t5arrow)" />

          {/* Encoder Stack */}
          <g className="encoder-stack" style={{ opacity: 0 }}>
            <text x="530" y="50" textAnchor="middle" fill="#a78bfa" fontSize="11" fontWeight="bold">T5 Encoder (24 Layers)</text>
            {[...Array(12)].map((_, i) => (
              <rect 
                key={i} 
                className="encoder-layer"
                x="450" 
                y={60 + i * 18} 
                width="160" 
                height="15" 
                rx="2" 
                fill="#8b5cf6" 
                fillOpacity={0.2 + i * 0.05}
                stroke="#8b5cf6"
                strokeWidth="1"
              />
            ))}
            <text x="530" y="290" textAnchor="middle" fill="#94a3b8" fontSize="9">Bidirectional Self-Attention</text>
          </g>

          {/* Output */}
          <g className="t5-output" style={{ opacity: 0 }}>
            <rect x="650" y="100" width="130" height="100" rx="8" fill="#f59e0b" fillOpacity="0.2" stroke="#f59e0b" strokeWidth="2" />
            <text x="715" y="130" textAnchor="middle" fill="#fbbf24" fontSize="10" fontWeight="bold">Embeddings</text>
            <text x="715" y="150" textAnchor="middle" fill="white" fontSize="9">[B, seq_len, 4096]</text>
            <text x="715" y="175" textAnchor="middle" fill="#94a3b8" fontSize="8">Rich contextual</text>
            <text x="715" y="187" textAnchor="middle" fill="#94a3b8" fontSize="8">representations</text>
          </g>

          {/* Arrow to output */}
          <path d="M 620 150 L 650 150" stroke="#a78bfa" strokeWidth="2" markerEnd="url(#t5arrow)" />

          <defs>
            <marker id="t5arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#34d399" />
            </marker>
          </defs>
        </svg>
      </div>

      {/* Why T5 in SD3 */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-emerald-500/10 rounded-lg p-4 border border-emerald-500/30">
          <FileText className="text-emerald-600 dark:text-emerald-400 mb-2" size={24} />
          <h4 className="font-semibold text-emerald-600 dark:text-emerald-400 mb-1">Long Context</h4>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            256+ tokens vs CLIP's 77. Handle detailed descriptions and complex prompts.
          </p>
        </div>
        <div className="bg-teal-500/10 rounded-lg p-4 border border-teal-500/30">
          <Zap className="text-teal-600 dark:text-teal-400 mb-2" size={24} />
          <h4 className="font-semibold text-teal-600 dark:text-teal-400 mb-1">Deep Understanding</h4>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            Bidirectional attention captures context from both directions for better comprehension.
          </p>
        </div>
        <div className="bg-cyan-500/10 rounded-lg p-4 border border-cyan-500/30">
          <HardDrive className="text-cyan-600 dark:text-cyan-400 mb-2" size={24} />
          <h4 className="font-semibold text-cyan-600 dark:text-cyan-400 mb-1">Optional Component</h4>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            T5 is optional in SD3. Skip it for faster inference with simpler prompts.
          </p>
        </div>
      </div>

      {/* Resource Warning */}
      <div className="bg-yellow-500/10 rounded-xl p-4 border border-yellow-500/30">
        <div className="flex items-center gap-2 text-yellow-400 font-semibold mb-2">
          <Clock size={18} />
          Resource Requirements
        </div>
        <div className="grid md:grid-cols-3 gap-4 text-sm text-gray-700 dark:text-gray-300">
          <div>
            <span className="text-yellow-400">Memory:</span> ~8-10GB VRAM for T5-XXL
          </div>
          <div>
            <span className="text-yellow-400">Speed:</span> 100-500ms encoding time
          </div>
          <div>
            <span className="text-yellow-400">Disk:</span> ~10GB model weights
          </div>
        </div>
      </div>
    </div>
  );
}

export default OverviewPanel;
