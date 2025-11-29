import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Type, Hash, Cpu, ArrowRight } from 'lucide-react';
import gsap from 'gsap';

function OverviewPanel() {
  const [isPlaying, setIsPlaying] = useState(false);
  const animationRef = useRef(null);
  const timelineRef = useRef(null);

  useEffect(() => {
    if (!animationRef.current) return;

    const ctx = gsap.context(() => {
      timelineRef.current = gsap.timeline({ paused: true });

      // Text appears
      timelineRef.current.fromTo('.text-box',
        { opacity: 0, x: -30 },
        { opacity: 1, x: 0, duration: 0.5 }
      );

      // Tokenizer processing
      timelineRef.current.to('.tokenizer-box', { opacity: 1, duration: 0.3 });
      timelineRef.current.to('.tokenizer-gear', {
        rotation: 360,
        duration: 1,
        ease: 'linear'
      });

      // Tokens appear
      timelineRef.current.fromTo('.token-chip',
        { opacity: 0, scale: 0 },
        { opacity: 1, scale: 1, stagger: 0.1, duration: 0.3, ease: 'back.out(1.7)' }
      );

      // Token IDs appear
      timelineRef.current.fromTo('.token-id',
        { opacity: 0, y: 10 },
        { opacity: 1, y: 0, stagger: 0.1, duration: 0.2 }
      );

      // Embeddings
      timelineRef.current.to('.embed-box', { opacity: 1, duration: 0.4 });

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
        <h2 className="text-2xl font-bold text-orange-400 mb-2">Why Tokenization?</h2>
        <p className="text-gray-300 max-w-3xl mx-auto">
          Neural networks can't process text directly - they work with numbers.
          <strong> Tokenization</strong> converts human-readable text into sequences of integers 
          that can be embedded into high-dimensional vectors.
        </p>
      </div>

      {/* The Process */}
      <div className="grid md:grid-cols-4 gap-4">
        <div className="bg-blue-500/10 rounded-xl p-4 text-center border border-blue-500/30">
          <Type className="mx-auto text-blue-400 mb-2" size={32} />
          <div className="font-semibold text-blue-400">Text</div>
          <div className="text-xs text-gray-400 mt-1">"A cat on a mat"</div>
        </div>
        <div className="bg-orange-500/10 rounded-xl p-4 text-center border border-orange-500/30">
          <Hash className="mx-auto text-orange-400 mb-2" size={32} />
          <div className="font-semibold text-orange-400">Tokenize</div>
          <div className="text-xs text-gray-400 mt-1">Split into subwords</div>
        </div>
        <div className="bg-purple-500/10 rounded-xl p-4 text-center border border-purple-500/30">
          <div className="mx-auto w-8 h-8 rounded bg-purple-400/50 flex items-center justify-center text-white font-mono text-sm mb-2">ID</div>
          <div className="font-semibold text-purple-400">Encode</div>
          <div className="text-xs text-gray-400 mt-1">Map to integers</div>
        </div>
        <div className="bg-green-500/10 rounded-xl p-4 text-center border border-green-500/30">
          <Cpu className="mx-auto text-green-400 mb-2" size={32} />
          <div className="font-semibold text-green-400">Embed</div>
          <div className="text-xs text-gray-400 mt-1">768/4096-dim vectors</div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-4">
        <button
          onClick={handlePlayPause}
          className="flex items-center gap-2 px-4 py-2 bg-orange-600 hover:bg-orange-700 rounded-lg transition-colors"
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
      <div ref={animationRef} className="bg-black/40 rounded-xl p-6 min-h-[300px]">
        <svg viewBox="0 0 800 250" className="w-full h-auto">
          {/* Text Input */}
          <g className="text-box" style={{ opacity: 0 }}>
            <rect x="20" y="90" width="150" height="60" rx="8" fill="#3b82f6" fillOpacity="0.2" stroke="#3b82f6" strokeWidth="2" />
            <text x="95" y="115" textAnchor="middle" fill="#60a5fa" fontSize="10">Text Prompt</text>
            <text x="95" y="135" textAnchor="middle" fill="white" fontSize="11">"A cute cat"</text>
          </g>

          {/* Arrow */}
          <path d="M 175 120 L 210 120" stroke="#60a5fa" strokeWidth="2" markerEnd="url(#tokenArrow)" />

          {/* Tokenizer */}
          <g className="tokenizer-box" style={{ opacity: 0 }}>
            <rect x="220" y="80" width="100" height="80" rx="8" fill="#f97316" fillOpacity="0.2" stroke="#f97316" strokeWidth="2" />
            <text x="270" y="105" textAnchor="middle" fill="#fb923c" fontSize="10">Tokenizer</text>
            <circle className="tokenizer-gear" cx="270" cy="130" r="15" fill="none" stroke="#fb923c" strokeWidth="2" strokeDasharray="5,3" />
          </g>

          {/* Arrow */}
          <path d="M 325 120 L 360 120" stroke="#fb923c" strokeWidth="2" markerEnd="url(#tokenArrow)" />

          {/* Tokens */}
          {['[BOS]', 'A', 'cute', 'cat', '[EOS]'].map((token, i) => (
            <g key={i}>
              <g className="token-chip" style={{ opacity: 0 }}>
                <rect x={370 + i * 55} y="90" width="50" height="30" rx="4" fill="#8b5cf6" fillOpacity="0.3" stroke="#8b5cf6" strokeWidth="1" />
                <text x={395 + i * 55} y="110" textAnchor="middle" fill="white" fontSize="9">{token}</text>
              </g>
              <text className="token-id" x={395 + i * 55} y="140" textAnchor="middle" fill="#fbbf24" fontSize="9" style={{ opacity: 0 }}>
                {[49406, 320, 2242, 2368, 49407][i]}
              </text>
            </g>
          ))}

          {/* Arrow to embeddings */}
          <path d="M 650 120 L 690 120" stroke="#a78bfa" strokeWidth="2" markerEnd="url(#tokenArrow)" />

          {/* Embeddings */}
          <g className="embed-box" style={{ opacity: 0 }}>
            <rect x="700" y="80" width="80" height="80" rx="8" fill="#10b981" fillOpacity="0.2" stroke="#10b981" strokeWidth="2" />
            <text x="740" y="110" textAnchor="middle" fill="#34d399" fontSize="10">Embeddings</text>
            <text x="740" y="130" textAnchor="middle" fill="white" fontSize="9">[5, 768]</text>
            <text x="740" y="150" textAnchor="middle" fill="#94a3b8" fontSize="8">vectors</text>
          </g>

          <defs>
            <marker id="tokenArrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8" />
            </marker>
          </defs>
        </svg>
      </div>

      {/* SD3 Tokenizers */}
      <div className="bg-gradient-to-r from-orange-500/10 to-red-500/10 rounded-xl p-6 border border-orange-500/30">
        <h3 className="text-lg font-semibold text-orange-400 mb-4">SD3's Three Tokenizers</h3>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-black/30 rounded-lg p-4">
            <div className="font-semibold text-blue-400 mb-2">CLIP-L Tokenizer</div>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>• BPE algorithm</li>
              <li>• ~49,000 vocabulary</li>
              <li>• Max 77 tokens</li>
              <li>• [BOS]/[EOS] markers</li>
            </ul>
          </div>
          <div className="bg-black/30 rounded-lg p-4">
            <div className="font-semibold text-purple-400 mb-2">CLIP-G Tokenizer</div>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>• BPE algorithm</li>
              <li>• ~49,000 vocabulary</li>
              <li>• Max 77 tokens</li>
              <li>• Same as CLIP-L</li>
            </ul>
          </div>
          <div className="bg-black/30 rounded-lg p-4">
            <div className="font-semibold text-green-400 mb-2">T5 Tokenizer</div>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>• SentencePiece</li>
              <li>• ~32,000 vocabulary</li>
              <li>• Max 256+ tokens</li>
              <li>• ▁ word markers</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Key Concepts */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-black/30 rounded-lg p-4 border border-white/10">
          <h4 className="font-semibold text-white mb-2">🎯 Why Subword Tokenization?</h4>
          <p className="text-sm text-gray-300">
            Word-level tokenization can't handle new words. Character-level is too fine-grained.
            <strong> Subword tokenization</strong> balances both: common words stay whole, 
            rare words split into meaningful pieces.
          </p>
        </div>
        <div className="bg-black/30 rounded-lg p-4 border border-white/10">
          <h4 className="font-semibold text-white mb-2">📊 Token vs Word</h4>
          <p className="text-sm text-gray-300">
            "photorealistic" → ["photo", "real", "istic"] (3 tokens)<br/>
            "cat" → ["cat"] (1 token)<br/>
            "a" → ["a"] (1 token)<br/>
            Token count ≠ word count!
          </p>
        </div>
      </div>
    </div>
  );
}

export default OverviewPanel;
