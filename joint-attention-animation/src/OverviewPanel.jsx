import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, ArrowRight } from 'lucide-react';
import gsap from 'gsap';

export default function OverviewPanel() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [step, setStep] = useState(0);
  const animationRef = useRef(null);

  const steps = [
    { title: "Separate Modalities", desc: "Image tokens and text tokens exist in different spaces" },
    { title: "Cross-Attention Limitation", desc: "Traditional approach: text conditions image via cross-attention" },
    { title: "Joint Attention Solution", desc: "MM-DiT: concatenate all tokens, apply unified self-attention" },
    { title: "Bidirectional Flow", desc: "Image and text mutually inform each other" },
  ];

  useEffect(() => {
    if (isPlaying) {
      animationRef.current = gsap.to({}, {
        duration: 2,
        repeat: -1,
        onRepeat: () => {
          setStep((s) => (s + 1) % steps.length);
        },
      });
    } else if (animationRef.current) {
      animationRef.current.kill();
    }
    return () => {
      if (animationRef.current) animationRef.current.kill();
    };
  }, [isPlaying]);

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          Why <span className="text-violet-400">Joint Attention</span>?
        </h2>
        <p className="text-gray-400">
          How MM-DiT revolutionizes text-image interaction in diffusion models
        </p>
      </div>

      {/* Main Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <div className="flex justify-center mb-6 gap-4">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 flex items-center gap-2"
          >
            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button
            onClick={() => { setStep(0); setIsPlaying(false); }}
            className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 flex items-center gap-2"
          >
            <RotateCcw size={18} />
            Reset
          </button>
        </div>

        {/* Step indicator */}
        <div className="mb-4 text-center">
          <span className="text-violet-400 font-bold">Step {step + 1}:</span>{' '}
          <span className="text-gray-300">{steps[step].title}</span>
        </div>

        {/* Animation Area */}
        <div className="relative h-80 bg-black/30 rounded-xl overflow-hidden">
          {step === 0 && (
            <div className="absolute inset-0 flex items-center justify-around p-8">
              {/* Image tokens */}
              <div className="text-center">
                <div className="grid grid-cols-4 gap-1 mb-3">
                  {[...Array(16)].map((_, i) => (
                    <div
                      key={i}
                      className="w-8 h-8 rounded bg-gradient-to-br from-blue-500 to-cyan-500 opacity-80"
                    />
                  ))}
                </div>
                <p className="text-blue-400 font-bold">Image Tokens</p>
                <p className="text-xs text-gray-400">4096 tokens (64×64)</p>
              </div>
              
              {/* Separator */}
              <div className="h-40 w-px bg-white/20" />
              
              {/* Text tokens */}
              <div className="text-center">
                <div className="flex flex-col gap-1 mb-3">
                  {['a', 'cat', 'on', 'beach'].map((word, i) => (
                    <div
                      key={i}
                      className="px-3 py-2 rounded bg-gradient-to-br from-orange-500 to-amber-500"
                    >
                      {word}
                    </div>
                  ))}
                </div>
                <p className="text-orange-400 font-bold">Text Tokens</p>
                <p className="text-xs text-gray-400">77-256 tokens</p>
              </div>
            </div>
          )}

          {step === 1 && (
            <div className="absolute inset-0 flex items-center justify-center p-8">
              <div className="flex items-center gap-8">
                {/* Image pathway */}
                <div className="text-center">
                  <div className="w-24 h-24 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center mb-2">
                    <span className="text-2xl">🖼️</span>
                  </div>
                  <p className="text-sm text-blue-400">Self-Attn</p>
                </div>
                
                <div className="flex flex-col items-center">
                  <div className="w-16 h-px bg-orange-400" />
                  <span className="text-xs text-orange-400 my-1">Cross-Attn</span>
                  <div className="w-16 h-px bg-orange-400" />
                </div>
                
                {/* Text conditioning */}
                <div className="text-center">
                  <div className="w-24 h-24 rounded-xl bg-gradient-to-br from-orange-500 to-amber-500 flex items-center justify-center mb-2">
                    <span className="text-2xl">📝</span>
                  </div>
                  <p className="text-sm text-orange-400">Text Enc</p>
                </div>
              </div>
              <div className="absolute bottom-4 left-0 right-0 text-center">
                <p className="text-red-400 text-sm">⚠️ One-way flow: Text → Image only</p>
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="absolute inset-0 flex flex-col items-center justify-center p-8">
              <p className="text-sm text-gray-400 mb-4">Concatenate into unified sequence</p>
              <div className="flex items-center gap-2">
                {/* Combined tokens */}
                {[...Array(4)].map((_, i) => (
                  <div
                    key={`img-${i}`}
                    className="w-10 h-10 rounded bg-gradient-to-br from-blue-500 to-cyan-500"
                  />
                ))}
                <span className="text-gray-400 mx-2">...</span>
                {['a', 'cat', '...'].map((word, i) => (
                  <div
                    key={`txt-${i}`}
                    className="px-2 py-2 rounded bg-gradient-to-br from-orange-500 to-amber-500 text-sm"
                  >
                    {word}
                  </div>
                ))}
              </div>
              <ArrowRight className="my-4 text-violet-400" />
              <div className="w-full max-w-md h-24 rounded-xl bg-gradient-to-r from-violet-600/50 to-fuchsia-600/50 border border-violet-400 flex items-center justify-center">
                <span className="text-violet-200 font-bold">Full Self-Attention</span>
              </div>
              <p className="text-sm text-green-400 mt-4">✓ All tokens can attend to all tokens</p>
            </div>
          )}

          {step === 3 && (
            <div className="absolute inset-0 flex items-center justify-center p-8">
              <div className="relative w-80 h-60">
                {/* Bidirectional arrows */}
                <svg className="absolute inset-0" viewBox="0 0 320 240">
                  <defs>
                    <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                      <path d="M0,0 L0,6 L9,3 z" fill="#a855f7" />
                    </marker>
                  </defs>
                  {/* Image to Text */}
                  <path d="M80,80 Q160,40 240,80" fill="none" stroke="#3b82f6" strokeWidth="2" markerEnd="url(#arrow)" />
                  {/* Text to Image */}
                  <path d="M240,160 Q160,200 80,160" fill="none" stroke="#f97316" strokeWidth="2" markerEnd="url(#arrow)" />
                </svg>
                
                {/* Image block */}
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-28 h-28 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                  <span className="text-3xl">🖼️</span>
                </div>
                
                {/* Text block */}
                <div className="absolute right-0 top-1/2 -translate-y-1/2 w-28 h-28 rounded-xl bg-gradient-to-br from-orange-500 to-amber-500 flex items-center justify-center">
                  <span className="text-3xl">📝</span>
                </div>
                
                {/* Labels */}
                <div className="absolute top-0 left-1/2 -translate-x-1/2 text-blue-400 text-sm">
                  Image attends to Text
                </div>
                <div className="absolute bottom-0 left-1/2 -translate-x-1/2 text-orange-400 text-sm">
                  Text attends to Image
                </div>
              </div>
            </div>
          )}
        </div>

        <p className="text-center text-gray-400 mt-4 text-sm">{steps[step].desc}</p>
      </div>

      {/* Key Comparison */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-red-900/20 rounded-xl p-5 border border-red-500/30">
          <h3 className="font-bold text-red-400 mb-3">❌ Cross-Attention (SD1.5/SDXL)</h3>
          <ul className="text-sm text-gray-300 space-y-2">
            <li>• Text tokens are read-only keys/values</li>
            <li>• Text cannot "see" the image state</li>
            <li>• Separate streams with limited interaction</li>
            <li>• More architectural complexity</li>
          </ul>
        </div>
        
        <div className="bg-green-900/20 rounded-xl p-5 border border-green-500/30">
          <h3 className="font-bold text-green-400 mb-3">✓ Joint Attention (SD3/MM-DiT)</h3>
          <ul className="text-sm text-gray-300 space-y-2">
            <li>• All tokens are equal participants</li>
            <li>• Text tokens update based on image</li>
            <li>• Unified transformer architecture</li>
            <li>• Simpler, more scalable design</li>
          </ul>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-black/30 rounded-xl p-4 text-center border border-white/10">
          <p className="text-2xl font-bold text-violet-400">4096+</p>
          <p className="text-xs text-gray-400">Image tokens (1024px)</p>
        </div>
        <div className="bg-black/30 rounded-xl p-4 text-center border border-white/10">
          <p className="text-2xl font-bold text-orange-400">77-256</p>
          <p className="text-xs text-gray-400">Text tokens</p>
        </div>
        <div className="bg-black/30 rounded-xl p-4 text-center border border-white/10">
          <p className="text-2xl font-bold text-fuchsia-400">O(N²)</p>
          <p className="text-xs text-gray-400">Attention complexity</p>
        </div>
      </div>
    </div>
  );
}
