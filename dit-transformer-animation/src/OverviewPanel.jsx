import React, { useState } from 'react';
import { ArrowRight, Layers } from 'lucide-react';

export default function OverviewPanel() {
  const [hoveredBlock, setHoveredBlock] = useState(null);

  const blocks = [
    { id: 'patchify', label: 'Patchify', color: 'from-blue-500 to-cyan-500', desc: 'Convert image to sequence of patch tokens' },
    { id: 'pos', label: '+Pos Embed', color: 'from-cyan-500 to-teal-500', desc: 'Add 2D sinusoidal positional encoding' },
    { id: 'blocks', label: 'DiT Blocks ×N', color: 'from-pink-500 to-rose-500', desc: 'N transformer blocks with AdaLN conditioning' },
    { id: 'final', label: 'Final Layer', color: 'from-orange-500 to-amber-500', desc: 'AdaLN + Linear to predict noise' },
    { id: 'unpatch', label: 'Unpatchify', color: 'from-violet-500 to-purple-500', desc: 'Reshape back to image dimensions' },
  ];

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          What is <span className="text-pink-400">DiT</span>?
        </h2>
        <p className="text-gray-400">
          Diffusion Transformer: Using transformers instead of U-Net for noise prediction
        </p>
      </div>

      {/* Architecture Diagram */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-6 text-center">DiT Architecture Overview</h3>
        
        <div className="flex flex-col md:flex-row items-center justify-center gap-4">
          {/* Input */}
          <div className="text-center">
            <div className="w-20 h-20 rounded-xl bg-gradient-to-br from-gray-600 to-gray-700 flex items-center justify-center mb-2">
              <span className="text-2xl">🖼️</span>
            </div>
            <p className="text-xs text-gray-400">Noisy Latent</p>
            <p className="text-xs text-gray-500">z_t</p>
          </div>

          <ArrowRight className="text-gray-500" />

          {/* Main blocks */}
          {blocks.map((block, i) => (
            <React.Fragment key={block.id}>
              <div
                className="relative"
                onMouseEnter={() => setHoveredBlock(block.id)}
                onMouseLeave={() => setHoveredBlock(null)}
              >
                <div className={`w-20 h-20 rounded-xl bg-gradient-to-br ${block.color} flex items-center justify-center cursor-pointer transition-transform hover:scale-110`}>
                  {block.id === 'blocks' ? (
                    <Layers size={24} />
                  ) : (
                    <span className="text-xs font-bold text-center px-1">{block.label}</span>
                  )}
                </div>
                <p className="text-xs text-gray-400 text-center mt-2">{block.label}</p>
                
                {hoveredBlock === block.id && (
                  <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 p-3 bg-black/90 rounded-lg border border-white/20 w-48 z-10">
                    <p className="text-xs text-gray-300">{block.desc}</p>
                  </div>
                )}
              </div>
              {i < blocks.length - 1 && <ArrowRight className="text-gray-500" />}
            </React.Fragment>
          ))}

          <ArrowRight className="text-gray-500" />

          {/* Output */}
          <div className="text-center">
            <div className="w-20 h-20 rounded-xl bg-gradient-to-br from-green-600 to-emerald-600 flex items-center justify-center mb-2">
              <span className="text-2xl">📊</span>
            </div>
            <p className="text-xs text-gray-400">Predicted</p>
            <p className="text-xs text-gray-500">v or ε</p>
          </div>
        </div>
      </div>

      {/* Key Innovation */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-red-900/20 rounded-xl p-5 border border-red-500/30">
          <h3 className="font-bold text-red-400 mb-3">🏛️ U-Net (SD1.5/SDXL)</h3>
          <ul className="text-sm text-gray-300 space-y-2">
            <li>• Convolutional backbone</li>
            <li>• Encoder-decoder with skip connections</li>
            <li>• Cross-attention for text conditioning</li>
            <li>• Inductive bias for images</li>
            <li>• ~860M parameters (SDXL)</li>
          </ul>
        </div>
        
        <div className="bg-green-900/20 rounded-xl p-5 border border-green-500/30">
          <h3 className="font-bold text-green-400 mb-3">🏗️ DiT (SD3/Flux)</h3>
          <ul className="text-sm text-gray-300 space-y-2">
            <li>• Pure transformer backbone</li>
            <li>• Simple isotropic architecture</li>
            <li>• AdaLN for all conditioning</li>
            <li>• Minimal inductive bias</li>
            <li>• 2B-8B+ parameters (scales easily)</li>
          </ul>
        </div>
      </div>

      {/* Timeline */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">Evolution: U-Net → DiT → MM-DiT</h3>
        <div className="flex flex-col md:flex-row items-start gap-4">
          <div className="flex-1 p-4 bg-gray-800/50 rounded-xl border border-gray-600">
            <p className="font-bold text-gray-400 mb-2">2022</p>
            <p className="text-pink-400 font-bold">DiT Paper</p>
            <p className="text-xs text-gray-400 mt-1">
              "Scalable Diffusion Models with Transformers" - Facebook AI
            </p>
          </div>
          <div className="flex-1 p-4 bg-purple-800/30 rounded-xl border border-purple-500/30">
            <p className="font-bold text-gray-400 mb-2">2024</p>
            <p className="text-purple-400 font-bold">SD3 MM-DiT</p>
            <p className="text-xs text-gray-400 mt-1">
              Multimodal DiT with joint attention for text+image
            </p>
          </div>
          <div className="flex-1 p-4 bg-orange-800/30 rounded-xl border border-orange-500/30">
            <p className="font-bold text-gray-400 mb-2">2024</p>
            <p className="text-orange-400 font-bold">Flux</p>
            <p className="text-xs text-gray-400 mt-1">
              Flow-based DiT with improved efficiency from Black Forest Labs
            </p>
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-black/30 rounded-xl p-4 text-center border border-white/10">
          <p className="text-2xl font-bold text-pink-400">38</p>
          <p className="text-xs text-gray-400">Layers (SD3-Large)</p>
        </div>
        <div className="bg-black/30 rounded-xl p-4 text-center border border-white/10">
          <p className="text-2xl font-bold text-orange-400">24</p>
          <p className="text-xs text-gray-400">Attention Heads</p>
        </div>
        <div className="bg-black/30 rounded-xl p-4 text-center border border-white/10">
          <p className="text-2xl font-bold text-violet-400">1536</p>
          <p className="text-xs text-gray-400">Hidden Dim</p>
        </div>
        <div className="bg-black/30 rounded-xl p-4 text-center border border-white/10">
          <p className="text-2xl font-bold text-cyan-400">2×2</p>
          <p className="text-xs text-gray-400">Patch Size</p>
        </div>
      </div>

      {/* Why Transformers */}
      <div className="bg-gradient-to-r from-pink-900/30 to-orange-900/30 rounded-xl p-6 border border-pink-500/30">
        <h3 className="font-bold text-pink-300 mb-3">💡 Why Replace U-Net with Transformers?</h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-300">
          <div>
            <p className="font-bold text-pink-400 mb-1">Scalability</p>
            <p>Transformers scale predictably. More parameters = better quality. U-Nets hit diminishing returns.</p>
          </div>
          <div>
            <p className="font-bold text-orange-400 mb-1">Simplicity</p>
            <p>One block type repeated N times. No encoder/decoder asymmetry, no skip connections.</p>
          </div>
          <div>
            <p className="font-bold text-violet-400 mb-1">Flexibility</p>
            <p>Easy to add modalities (text, audio, video). Joint attention handles any token type.</p>
          </div>
          <div>
            <p className="font-bold text-cyan-400 mb-1">Hardware Efficiency</p>
            <p>Matrix multiplications are GPU-optimized. Flash attention makes long sequences practical.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
