import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Type, Image, Link2 } from 'lucide-react';
import gsap from 'gsap';

function OverviewPanel() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [step, setStep] = useState(0);
  const animationRef = useRef(null);
  const timelineRef = useRef(null);

  const steps = [
    { title: "Text Input", description: "Text prompt enters CLIP" },
    { title: "Tokenization", description: "BPE tokenizer splits text into tokens" },
    { title: "Embedding", description: "Tokens converted to vectors" },
    { title: "Transformer", description: "12 transformer layers process tokens" },
    { title: "Pooled Output", description: "Final [EOS] token becomes pooled embedding" },
  ];

  useEffect(() => {
    if (!animationRef.current) return;

    const ctx = gsap.context(() => {
      timelineRef.current = gsap.timeline({ paused: true });

      // Step 0: Text input appears
      timelineRef.current.fromTo('.text-input-box',
        { opacity: 0, scale: 0.8 },
        { opacity: 1, scale: 1, duration: 0.5 }
      );

      // Step 1: Tokens appear
      timelineRef.current.to('.token-flow', { opacity: 1, duration: 0.3 }, '+=0.3');
      timelineRef.current.fromTo('.token-box',
        { opacity: 0, y: -20 },
        { opacity: 1, y: 0, stagger: 0.1, duration: 0.3 }
      );

      // Step 2: Embedding vectors
      timelineRef.current.to('.embedding-layer', { opacity: 1, duration: 0.3 }, '+=0.3');
      timelineRef.current.fromTo('.embed-arrow',
        { scaleY: 0 },
        { scaleY: 1, duration: 0.3 }
      );

      // Step 3: Transformer processing
      timelineRef.current.to('.transformer-stack', { opacity: 1, duration: 0.3 }, '+=0.2');
      timelineRef.current.fromTo('.transformer-layer',
        { opacity: 0.3, x: -10 },
        { opacity: 1, x: 0, stagger: 0.08, duration: 0.2 }
      );

      // Step 4: Pooled output
      timelineRef.current.to('.pooled-output', { opacity: 1, duration: 0.3 }, '+=0.2');
      timelineRef.current.fromTo('.pooled-vector',
        { scale: 0 },
        { scale: 1, duration: 0.4, ease: 'back.out(1.7)' }
      );

      // Update step counter
      timelineRef.current.eventCallback('onUpdate', () => {
        const progress = timelineRef.current.progress();
        const newStep = Math.min(Math.floor(progress * steps.length), steps.length - 1);
        setStep(newStep);
      });

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
    setStep(0);
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-blue-600 dark:text-blue-400 mb-2">What is CLIP?</h2>
        <p className="text-gray-700 dark:text-gray-300 max-w-3xl mx-auto">
          <strong>CLIP (Contrastive Language-Image Pre-training)</strong> was trained by OpenAI to understand 
          the relationship between text and images. SD3 uses CLIP's text encoder (specifically CLIP-L/14 and CLIP-G/14)
          to convert your text prompt into embeddings the diffusion model can understand.
        </p>
      </div>

      {/* Why Two CLIPs? */}
      <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-xl p-6 border border-blue-500/30">
        <h3 className="text-lg font-semibold text-blue-600 dark:text-blue-400 mb-3 flex items-center gap-2">
          <Link2 size={20} />
          Why Does SD3 Use Two CLIP Models?
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-black/30 rounded-lg p-4">
            <div className="font-semibold text-purple-600 dark:text-purple-400 mb-2">CLIP-L/14 (Large)</div>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 77 max tokens</li>
              <li>• 768-dim embeddings</li>
              <li>• 12 transformer layers</li>
              <li>• Faster, more general understanding</li>
            </ul>
          </div>
          <div className="bg-black/30 rounded-lg p-4">
            <div className="font-semibold text-pink-600 dark:text-pink-400 mb-2">CLIP-G/14 (Giant)</div>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 77 max tokens</li>
              <li>• 1280-dim embeddings</li>
              <li>• 32 transformer layers</li>
              <li>• More detailed, nuanced understanding</li>
            </ul>
          </div>
        </div>
        <p className="text-sm text-gray-800 dark:text-gray-400 mt-3">
          By combining both, SD3 gets complementary representations for better prompt understanding.
        </p>
      </div>

      {/* Animation Controls */}
      <div className="flex justify-center gap-4">
        <button
          onClick={handlePlayPause}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
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

      {/* Step Indicator */}
      <div className="bg-black/30 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-gray-800 dark:text-gray-400">Step {step + 1} of {steps.length}</span>
          <span className="text-blue-600 dark:text-blue-400 font-semibold">{steps[step].title}</span>
        </div>
        <p className="text-gray-700 dark:text-gray-300">{steps[step].description}</p>
      </div>

      {/* Architecture Animation */}
      <div ref={animationRef} className="bg-black/40 rounded-xl p-6 min-h-[400px] relative overflow-hidden">
        <svg viewBox="0 0 800 350" className="w-full h-auto">
          {/* Text Input */}
          <g className="text-input-box">
            <rect x="50" y="140" width="120" height="60" rx="8" fill="#3b82f6" fillOpacity="0.2" stroke="#3b82f6" strokeWidth="2" />
            <text x="110" y="165" textAnchor="middle" fill="#60a5fa" fontSize="10">Text Prompt</text>
            <text x="110" y="185" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">"A cat on a mat"</text>
          </g>

          {/* Token Flow Arrow */}
          <g className="token-flow" style={{ opacity: 0 }}>
            <path d="M 175 170 L 210 170" stroke="#60a5fa" strokeWidth="2" markerEnd="url(#arrowhead)" />
          </g>

          {/* Tokens */}
          <g className="tokens">
            {['[BOS]', 'A', 'cat', 'on', 'a', 'mat', '[EOS]'].map((token, i) => (
              <g key={i} className="token-box" style={{ opacity: 0 }}>
                <rect x={220 + i * 50} y="150" width="45" height="40" rx="4" fill="#8b5cf6" fillOpacity="0.3" stroke="#8b5cf6" strokeWidth="1" />
                <text x={242 + i * 50} y="175" textAnchor="middle" fill="white" fontSize="10">{token}</text>
              </g>
            ))}
          </g>

          {/* Embedding Layer */}
          <g className="embedding-layer" style={{ opacity: 0 }}>
            <rect x="220" y="210" width="350" height="30" rx="4" fill="#10b981" fillOpacity="0.2" stroke="#10b981" strokeWidth="2" />
            <text x="395" y="230" textAnchor="middle" fill="#34d399" fontSize="11">Token + Position Embeddings</text>
          </g>

          {/* Embed Arrows */}
          {[0, 1, 2, 3, 4, 5, 6].map((i) => (
            <line key={i} className="embed-arrow" x1={242 + i * 50} y1="195" x2={242 + i * 50} y2="210" stroke="#10b981" strokeWidth="2" style={{ transformOrigin: `${242 + i * 50}px 195px` }} />
          ))}

          {/* Transformer Stack */}
          <g className="transformer-stack" style={{ opacity: 0 }}>
            <text x="650" y="120" textAnchor="middle" fill="#f472b6" fontSize="12" fontWeight="bold">Transformer Layers</text>
            {[...Array(12)].map((_, i) => (
              <g key={i} className="transformer-layer">
                <rect x="600" y={130 + i * 17} width="100" height="15" rx="2" fill="#ec4899" fillOpacity={0.3 + (i * 0.05)} stroke="#ec4899" strokeWidth="1" />
                <text x="650" y={142 + i * 17} textAnchor="middle" fill="white" fontSize="8">Layer {i + 1}</text>
              </g>
            ))}
          </g>

          {/* Connection to transformer */}
          <path d="M 570 225 L 590 225 L 590 200 L 600 200" stroke="#60a5fa" strokeWidth="2" fill="none" strokeDasharray="4,4" />

          {/* Pooled Output */}
          <g className="pooled-output" style={{ opacity: 0 }}>
            <text x="650" y="340" textAnchor="middle" fill="#fbbf24" fontSize="10">Pooled Embedding</text>
            <g className="pooled-vector">
              <circle cx="650" cy="320" r="15" fill="#f59e0b" fillOpacity="0.3" stroke="#f59e0b" strokeWidth="2" />
              <text x="650" y="325" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold">768d</text>
            </g>
            <line x1="650" y1="335" x2="650" y2="335" stroke="#f59e0b" strokeWidth="2" />
          </g>

          {/* Arrow marker */}
          <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#60a5fa" />
            </marker>
          </defs>
        </svg>
      </div>

      {/* Key Concepts */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-blue-500/10 rounded-lg p-4 border border-blue-500/30">
          <Type className="text-blue-600 dark:text-blue-400 mb-2" size={24} />
          <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-1">Text Only</h4>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            CLIP's text encoder is separate from the vision encoder. SD3 only uses the text side.
          </p>
        </div>
        <div className="bg-purple-500/10 rounded-lg p-4 border border-purple-500/30">
          <Image className="text-purple-600 dark:text-purple-400 mb-2" size={24} />
          <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-1">Contrastive Training</h4>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            Trained on 400M image-text pairs to align text and image representations.
          </p>
        </div>
        <div className="bg-pink-500/10 rounded-lg p-4 border border-pink-500/30">
          <Link2 className="text-pink-600 dark:text-pink-400 mb-2" size={24} />
          <h4 className="font-semibold text-pink-600 dark:text-pink-400 mb-1">Pooled + Sequence</h4>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            Outputs both token sequence (for joint attention) and pooled embedding (for conditioning).
          </p>
        </div>
      </div>
    </div>
  );
}

export default OverviewPanel;
