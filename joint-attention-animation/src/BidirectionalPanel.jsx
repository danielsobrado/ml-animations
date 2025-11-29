import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, ArrowRight, ArrowLeft, RefreshCw } from 'lucide-react';
import gsap from 'gsap';

export default function BidirectionalPanel() {
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const animRef = useRef(null);

  const steps = [
    {
      title: "Initial State",
      desc: "Image shows noise, text has semantic meaning",
      imgLabel: "Noisy latent",
      txtLabel: "\"A cat on beach\"",
    },
    {
      title: "Text → Image (Q from Img)",
      desc: "Image queries text: 'What should I look like?'",
      direction: "txt-to-img",
    },
    {
      title: "Image regions attend to 'cat'",
      desc: "Center regions get high attention to 'cat' token",
      highlight: "cat",
    },
    {
      title: "Image regions attend to 'beach'",
      desc: "Background regions attend to 'beach' token",
      highlight: "beach",
    },
    {
      title: "Image → Text (Q from Text)",
      desc: "Text queries image: 'What's actually forming?'",
      direction: "img-to-txt",
    },
    {
      title: "Text tokens update",
      desc: "Text representations refine based on image state",
      txtUpdate: true,
    },
    {
      title: "Mutual Refinement",
      desc: "Both modalities have richer representations",
      final: true,
    },
  ];

  useEffect(() => {
    if (isPlaying) {
      animRef.current = setInterval(() => {
        setStep(s => (s + 1) % steps.length);
      }, 2000);
    } else if (animRef.current) {
      clearInterval(animRef.current);
    }
    return () => {
      if (animRef.current) clearInterval(animRef.current);
    };
  }, [isPlaying]);

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="text-violet-400">Bidirectional</span> Fusion
        </h2>
        <p className="text-gray-400">
          How image and text mutually refine each other through joint attention
        </p>
      </div>

      {/* Main Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        {/* Controls */}
        <div className="flex justify-center gap-4 mb-6">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 flex items-center gap-2"
          >
            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
            {isPlaying ? 'Pause' : 'Auto Play'}
          </button>
          <button
            onClick={() => setStep((step - 1 + steps.length) % steps.length)}
            className="px-3 py-2 rounded-lg bg-white/10 hover:bg-white/20"
          >
            <ArrowLeft size={18} />
          </button>
          <button
            onClick={() => setStep((step + 1) % steps.length)}
            className="px-3 py-2 rounded-lg bg-white/10 hover:bg-white/20"
          >
            <ArrowRight size={18} />
          </button>
        </div>

        {/* Step indicator */}
        <div className="mb-4 text-center">
          <span className="text-violet-400 font-bold">Step {step + 1}/{steps.length}:</span>{' '}
          <span className="text-white">{steps[step].title}</span>
        </div>

        {/* Animation Area */}
        <div className="relative h-80 bg-black/30 rounded-xl overflow-hidden">
          <div className="absolute inset-0 flex items-center justify-center">
            {/* Image representation */}
            <div className="relative">
              <div className={`grid grid-cols-3 gap-1 p-4 rounded-xl border-2 transition-all duration-500 ${
                steps[step].direction === 'txt-to-img' ? 'border-orange-400 shadow-lg shadow-orange-400/30' :
                steps[step].direction === 'img-to-txt' ? 'border-blue-400 shadow-lg shadow-blue-400/30' :
                steps[step].final ? 'border-violet-400 shadow-lg shadow-violet-400/30' :
                'border-white/20'
              }`}>
                {[...Array(9)].map((_, i) => {
                  let opacity = step === 0 ? 0.3 : 0.3 + step * 0.1;
                  let highlight = false;
                  
                  // Center cells for "cat"
                  if (steps[step].highlight === 'cat' && [1, 4, 7].includes(i)) {
                    highlight = true;
                  }
                  // Edge cells for "beach"
                  if (steps[step].highlight === 'beach' && [0, 2, 3, 5, 6, 8].includes(i)) {
                    highlight = true;
                  }
                  
                  return (
                    <div
                      key={i}
                      className={`w-12 h-12 rounded transition-all duration-300 ${
                        highlight ? 'ring-2 ring-orange-400 scale-110' : ''
                      } ${
                        steps[step].final ? 'bg-gradient-to-br from-blue-400 to-cyan-400' :
                        `bg-blue-500`
                      }`}
                      style={{ opacity: highlight ? 1 : opacity }}
                    />
                  );
                })}
              </div>
              <p className="text-center text-blue-400 text-sm mt-2">
                Image Tokens
              </p>
            </div>

            {/* Arrows */}
            <div className="mx-8 flex flex-col items-center gap-4">
              {/* Text to Image arrow */}
              <div className={`flex items-center transition-all duration-500 ${
                steps[step].direction === 'txt-to-img' ? 'opacity-100 scale-110' : 'opacity-30'
              }`}>
                <ArrowLeft size={32} className="text-orange-400" />
                <span className="text-xs text-orange-400 mx-2">Text→Img</span>
              </div>
              
              <RefreshCw className={`transition-all duration-500 ${
                steps[step].final ? 'text-violet-400 animate-spin' : 'text-gray-600'
              }`} size={24} />
              
              {/* Image to Text arrow */}
              <div className={`flex items-center transition-all duration-500 ${
                steps[step].direction === 'img-to-txt' ? 'opacity-100 scale-110' : 'opacity-30'
              }`}>
                <span className="text-xs text-blue-400 mx-2">Img→Text</span>
                <ArrowRight size={32} className="text-blue-400" />
              </div>
            </div>

            {/* Text representation */}
            <div className="relative">
              <div className={`flex flex-col gap-2 p-4 rounded-xl border-2 transition-all duration-500 ${
                steps[step].direction === 'img-to-txt' ? 'border-blue-400 shadow-lg shadow-blue-400/30' :
                steps[step].txtUpdate ? 'border-green-400 shadow-lg shadow-green-400/30' :
                steps[step].final ? 'border-violet-400 shadow-lg shadow-violet-400/30' :
                'border-white/20'
              }`}>
                {['A', 'cat', 'on', 'beach'].map((word, i) => {
                  const isHighlighted = steps[step].highlight === word.toLowerCase();
                  const isUpdated = steps[step].txtUpdate;
                  
                  return (
                    <div
                      key={i}
                      className={`px-4 py-2 rounded transition-all duration-300 ${
                        isHighlighted ? 'bg-orange-500 ring-2 ring-orange-300 scale-110' :
                        isUpdated ? 'bg-gradient-to-r from-orange-500 to-green-500' :
                        steps[step].final ? 'bg-gradient-to-r from-orange-400 to-amber-400' :
                        'bg-orange-500'
                      }`}
                    >
                      {word}
                    </div>
                  );
                })}
              </div>
              <p className="text-center text-orange-400 text-sm mt-2">
                Text Tokens
              </p>
            </div>
          </div>
        </div>

        <p className="text-center text-gray-400 mt-4">{steps[step].desc}</p>
      </div>

      {/* Mathematical View */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">The Math Behind Bidirectional Flow</h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-blue-900/20 rounded-xl p-5 border border-blue-500/30">
            <h4 className="font-bold text-blue-400 mb-3">Image Updates from Text</h4>
            <div className="font-mono text-sm space-y-2">
              <p>Q_img = W_q · X_img</p>
              <p>K_all = W_k · [X_img; X_txt]</p>
              <p>V_all = W_v · [X_img; X_txt]</p>
              <p className="text-violet-400">A_img = softmax(Q_img · K_all^T / √d)</p>
              <p className="text-green-400">X_img' = A_img · V_all</p>
            </div>
            <p className="text-xs text-gray-400 mt-3">
              Image tokens query the joint sequence, receiving information from text
            </p>
          </div>

          <div className="bg-orange-900/20 rounded-xl p-5 border border-orange-500/30">
            <h4 className="font-bold text-orange-400 mb-3">Text Updates from Image</h4>
            <div className="font-mono text-sm space-y-2">
              <p>Q_txt = W_q · X_txt</p>
              <p>K_all = W_k · [X_img; X_txt]</p>
              <p>V_all = W_v · [X_img; X_txt]</p>
              <p className="text-violet-400">A_txt = softmax(Q_txt · K_all^T / √d)</p>
              <p className="text-green-400">X_txt' = A_txt · V_all</p>
            </div>
            <p className="text-xs text-gray-400 mt-3">
              Text tokens query the joint sequence, seeing what's forming in the image
            </p>
          </div>
        </div>
      </div>

      {/* Why This Matters */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-gradient-to-br from-green-900/30 to-emerald-800/20 rounded-xl p-5 border border-green-500/30">
          <h3 className="font-bold text-green-300 mb-3">✓ Benefits of Bidirectional</h3>
          <ul className="text-sm text-gray-300 space-y-2">
            <li>• <strong>Text awareness:</strong> Text knows what's forming in image</li>
            <li>• <strong>Context adaptation:</strong> "cat" means different things in different contexts</li>
            <li>• <strong>Error correction:</strong> Text can guide if image drifts off-topic</li>
            <li>• <strong>Compositional:</strong> Better handling of multiple objects</li>
          </ul>
        </div>

        <div className="bg-gradient-to-br from-violet-900/30 to-purple-800/20 rounded-xl p-5 border border-violet-500/30">
          <h3 className="font-bold text-violet-300 mb-3">🔄 Per-Layer Refinement</h3>
          <p className="text-sm text-gray-300 mb-3">
            This bidirectional exchange happens at <strong>every transformer layer</strong>:
          </p>
          <div className="text-xs text-gray-400 space-y-1">
            <p>Layer 1: Basic shapes & text parsing</p>
            <p>Layer 12: Object-level understanding</p>
            <p>Layer 24: Fine details & consistency</p>
            <p>Layer 38: Final refinement</p>
          </div>
        </div>
      </div>

      {/* Comparison */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">Cross-Attention vs Joint Attention</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/20">
                <th className="py-3 px-4 text-left text-gray-400">Aspect</th>
                <th className="py-3 px-4 text-left text-red-400">Cross-Attention</th>
                <th className="py-3 px-4 text-left text-green-400">Joint Attention</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/10">
              <tr>
                <td className="py-3 px-4">Text Token Updates</td>
                <td className="py-3 px-4">❌ Frozen</td>
                <td className="py-3 px-4">✓ Updated each layer</td>
              </tr>
              <tr>
                <td className="py-3 px-4">Information Flow</td>
                <td className="py-3 px-4">One-way (Text→Image)</td>
                <td className="py-3 px-4">Bidirectional</td>
              </tr>
              <tr>
                <td className="py-3 px-4">Attention Patterns</td>
                <td className="py-3 px-4">2 types (self, cross)</td>
                <td className="py-3 px-4">4 types (unified)</td>
              </tr>
              <tr>
                <td className="py-3 px-4">Complexity</td>
                <td className="py-3 px-4">Separate modules</td>
                <td className="py-3 px-4">Single unified attention</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
