import React, { useState } from 'react';
import { Clock, MessageSquare, Image, Layers } from 'lucide-react';

export default function ConditioningPanel() {
  const [activeCondition, setActiveCondition] = useState('timestep');

  const conditions = {
    timestep: {
      name: 'Timestep Embedding',
      icon: Clock,
      color: 'from-amber-500 to-orange-500',
      description: 'Encodes the current noise level using sinusoidal positional encoding',
      howItWorks: [
        'Input timestep t ∈ [0, 1000]',
        'Apply sinusoidal embedding (like transformer positional encoding)',
        'Pass through MLP to get hidden representation',
        'Add to class/text embedding',
      ],
      formula: 'emb[2i] = sin(t / 10000^(2i/d))\nemb[2i+1] = cos(t / 10000^(2i/d))',
      whyImportant: 'The model needs to know how noisy the input is to predict the correct amount of noise to remove.',
    },
    text: {
      name: 'Text Conditioning (SD3)',
      icon: MessageSquare,
      color: 'from-blue-500 to-violet-500',
      description: 'Injects text information through both AdaLN and joint attention',
      howItWorks: [
        'Encode text with CLIP-L, CLIP-G, T5',
        'Pooled CLIP embeddings → AdaLN conditioning',
        'Sequence embeddings → joint attention',
        'Both paths provide text guidance',
      ],
      formula: 'c = MLP(t_emb + pooled_clip)\nJoint: [img_tokens; txt_tokens]',
      whyImportant: 'Dual path ensures both global semantics (AdaLN) and fine-grained alignment (attention).',
    },
    class: {
      name: 'Class Conditioning',
      icon: Layers,
      color: 'from-green-500 to-emerald-500',
      description: 'For ImageNet-style class-conditional generation',
      howItWorks: [
        'Class label y ∈ [0, 999]',
        'Look up learned embedding from table',
        'Add to timestep embedding',
        'Inject via AdaLN',
      ],
      formula: 'c = t_emb + class_embed[y]',
      whyImportant: 'Simple but effective for categorical conditioning. Foundation for CFG.',
    },
  };

  const current = conditions[activeCondition];
  const Icon = current.icon;

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="text-pink-600 dark:text-pink-400">Conditioning</span> in DiT
        </h2>
        <p className="text-gray-800 dark:text-gray-400">
          How different signals guide the generation process
        </p>
      </div>

      {/* Condition Selector */}
      <div className="flex justify-center gap-4 flex-wrap">
        {Object.entries(conditions).map(([key, cond]) => {
          const CondIcon = cond.icon;
          return (
            <button
              key={key}
              onClick={() => setActiveCondition(key)}
              className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-all ${
                activeCondition === key
                  ? `bg-gradient-to-r ${cond.color} text-white`
                  : 'bg-white/10 text-gray-800 dark:text-gray-400 hover:bg-white/20'
              }`}
            >
              <CondIcon size={18} />
              {cond.name}
            </button>
          );
        })}
      </div>

      {/* Main Info Card */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <div className="flex items-center gap-3 mb-4">
          <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${current.color} flex items-center justify-center`}>
            <Icon size={24} />
          </div>
          <div>
            <h3 className="text-xl font-bold">{current.name}</h3>
            <p className="text-sm text-gray-800 dark:text-gray-400">{current.description}</p>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* How it works */}
          <div>
            <h4 className="font-bold text-gray-700 dark:text-gray-300 mb-3">How It Works</h4>
            <ol className="space-y-2">
              {current.howItWorks.map((step, i) => (
                <li key={i} className="flex items-start gap-3 text-sm">
                  <span className="w-6 h-6 rounded-full bg-white/10 flex items-center justify-center flex-shrink-0 text-xs">
                    {i + 1}
                  </span>
                  <span className="text-gray-700 dark:text-gray-300">{step}</span>
                </li>
              ))}
            </ol>
          </div>

          {/* Formula & Importance */}
          <div className="space-y-4">
            <div>
              <h4 className="font-bold text-gray-700 dark:text-gray-300 mb-2">Formula</h4>
              <div className="bg-black/50 rounded-lg p-3 font-mono text-sm">
                <pre className="text-pink-300 whitespace-pre-wrap">{current.formula}</pre>
              </div>
            </div>
            <div>
              <h4 className="font-bold text-gray-700 dark:text-gray-300 mb-2">Why Important?</h4>
              <p className="text-sm text-gray-800 dark:text-gray-400">{current.whyImportant}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Conditioning Flow Diagram */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-6 text-center">Conditioning Flow in SD3</h3>
        
        <div className="flex flex-col items-center gap-4">
          {/* Inputs Row */}
          <div className="flex justify-center gap-8 flex-wrap">
            <div className="text-center">
              <div className="w-24 h-16 rounded-xl bg-gradient-to-br from-amber-600 to-orange-600 flex items-center justify-center mb-2">
                <Clock size={24} />
              </div>
              <p className="text-xs text-gray-800 dark:text-gray-400">Timestep t</p>
            </div>
            <div className="text-center">
              <div className="w-24 h-16 rounded-xl bg-gradient-to-br from-blue-600 to-violet-600 flex items-center justify-center mb-2">
                <MessageSquare size={24} />
              </div>
              <p className="text-xs text-gray-800 dark:text-gray-400">CLIP Pooled</p>
            </div>
          </div>

          {/* Merge */}
          <div className="text-2xl text-gray-700 dark:text-gray-500">↓ +</div>

          {/* Combined */}
          <div className="w-48 h-12 rounded-xl bg-gradient-to-r from-pink-600 to-rose-600 flex items-center justify-center">
            <span className="font-bold">c = t_emb + pooled</span>
          </div>

          <div className="text-2xl text-gray-700 dark:text-gray-500">↓</div>

          {/* MLP */}
          <div className="w-32 h-12 rounded-xl bg-gradient-to-r from-purple-600 to-violet-600 flex items-center justify-center">
            <span className="font-bold">MLP</span>
          </div>

          <div className="text-2xl text-gray-700 dark:text-gray-500">↓</div>

          {/* Outputs */}
          <div className="flex gap-4">
            <div className="text-center">
              <div className="w-16 h-16 rounded-lg bg-pink-600 flex items-center justify-center">
                <span className="font-bold">γ</span>
              </div>
              <p className="text-xs text-gray-800 dark:text-gray-400 mt-1">Scale</p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 rounded-lg bg-blue-600 flex items-center justify-center">
                <span className="font-bold">β</span>
              </div>
              <p className="text-xs text-gray-800 dark:text-gray-400 mt-1">Shift</p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 rounded-lg bg-green-600 flex items-center justify-center">
                <span className="font-bold">α</span>
              </div>
              <p className="text-xs text-gray-800 dark:text-gray-400 mt-1">Gate</p>
            </div>
          </div>

          <div className="text-2xl text-gray-700 dark:text-gray-500">↓</div>

          {/* Application */}
          <div className="w-64 h-12 rounded-xl bg-gradient-to-r from-gray-700 to-gray-600 flex items-center justify-center">
            <span className="text-sm">Apply to each DiT block</span>
          </div>
        </div>
      </div>

      {/* CFG */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">Classifier-Free Guidance (CFG)</h3>
        <p className="text-gray-800 dark:text-gray-400 mb-4">
          During inference, DiT uses CFG to strengthen conditioning:
        </p>
        
        <div className="bg-black/50 rounded-lg p-4 font-mono text-sm mb-4">
          <p className="text-gray-700 dark:text-gray-300">
            ε_guided = ε_uncond + <span className="text-pink-600 dark:text-pink-400">scale</span> × (ε_cond - ε_uncond)
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-gray-800/50 rounded-xl p-4">
            <p className="font-bold text-gray-800 dark:text-gray-400 mb-2">scale = 1.0</p>
            <p className="text-sm text-gray-700 dark:text-gray-500">Normal conditioning, no guidance</p>
          </div>
          <div className="bg-pink-800/30 rounded-xl p-4 border border-pink-500/30">
            <p className="font-bold text-pink-600 dark:text-pink-400 mb-2">scale = 5-7</p>
            <p className="text-sm text-gray-700 dark:text-gray-500">Recommended range for SD3</p>
          </div>
          <div className="bg-red-800/30 rounded-xl p-4">
            <p className="font-bold text-red-400 mb-2">scale {'>'}  10</p>
            <p className="text-sm text-gray-700 dark:text-gray-500">Over-saturation, artifacts</p>
          </div>
        </div>
      </div>

      {/* Comparison with U-Net */}
      <div className="bg-gradient-to-r from-pink-900/30 to-orange-900/30 rounded-xl p-6 border border-pink-500/30">
        <h3 className="font-bold text-pink-300 mb-4">DiT vs U-Net Conditioning</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/20">
                <th className="py-2 px-4 text-left text-gray-800 dark:text-gray-400">Aspect</th>
                <th className="py-2 px-4 text-left text-gray-800 dark:text-gray-400">U-Net (SD1.5/SDXL)</th>
                <th className="py-2 px-4 text-left text-pink-600 dark:text-pink-400">DiT (SD3)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/10">
              <tr>
                <td className="py-2 px-4">Timestep</td>
                <td className="py-2 px-4">Added to residual blocks</td>
                <td className="py-2 px-4">AdaLN modulation</td>
              </tr>
              <tr>
                <td className="py-2 px-4">Text (sequence)</td>
                <td className="py-2 px-4">Cross-attention layers</td>
                <td className="py-2 px-4">Joint attention</td>
              </tr>
              <tr>
                <td className="py-2 px-4">Text (global)</td>
                <td className="py-2 px-4">Pooled to timestep MLP</td>
                <td className="py-2 px-4">Pooled to AdaLN</td>
              </tr>
              <tr>
                <td className="py-2 px-4"># Injection Points</td>
                <td className="py-2 px-4">Varies by layer type</td>
                <td className="py-2 px-4">Every block (uniform)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
