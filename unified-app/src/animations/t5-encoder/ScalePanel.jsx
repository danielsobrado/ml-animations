import React, { useState } from 'react';
import { Scale, Zap, HardDrive, Clock, TrendingUp } from 'lucide-react';

function ScalePanel() {
  const [selectedModel, setSelectedModel] = useState('xxl');

  const models = {
    base: {
      name: 'T5-Base',
      params: '220M',
      layers: 12,
      hidden: 768,
      heads: 12,
      ffn: 3072,
      memory: '~1GB',
      speed: 'Very Fast',
      quality: 'Basic',
      sdUse: false,
    },
    large: {
      name: 'T5-Large',
      params: '770M',
      layers: 24,
      hidden: 1024,
      heads: 16,
      ffn: 4096,
      memory: '~3GB',
      speed: 'Fast',
      quality: 'Good',
      sdUse: false,
    },
    xl: {
      name: 'T5-XL',
      params: '3B',
      layers: 24,
      hidden: 2048,
      heads: 32,
      ffn: 5120,
      memory: '~6GB',
      speed: 'Medium',
      quality: 'Great',
      sdUse: false,
    },
    xxl: {
      name: 'T5-XXL',
      params: '11B (4.7B encoder)',
      layers: 24,
      hidden: 4096,
      heads: 64,
      ffn: 10240,
      memory: '~8-10GB',
      speed: 'Slow',
      quality: 'Excellent',
      sdUse: true,
    },
  };

  const selectedModelData = models[selectedModel];

  // Calculate relative bar widths
  const maxParams = 11000; // 11B
  const paramValues = { base: 220, large: 770, xl: 3000, xxl: 11000 };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-emerald-600 dark:text-emerald-400 mb-2">T5 Model Scaling</h2>
        <p className="text-gray-700 dark:text-gray-300 max-w-3xl mx-auto">
          T5 comes in multiple sizes. SD3 uses <strong>T5-XXL</strong> (the largest) 
          for maximum text understanding capability, though only the encoder portion.
        </p>
      </div>

      {/* Model Selector */}
      <div className="flex flex-wrap justify-center gap-2">
        {Object.entries(models).map(([key, model]) => (
          <button
            key={key}
            onClick={() => setSelectedModel(key)}
            className={`px-4 py-2 rounded-lg transition-all ${
              selectedModel === key
                ? 'bg-emerald-600 text-white'
                : 'bg-white/10 text-gray-700 dark:text-gray-300 hover:bg-white/20'
            } ${model.sdUse ? 'ring-2 ring-yellow-500/50' : ''}`}
          >
            {model.name}
            {model.sdUse && <span className="ml-2 text-xs">★ SD3</span>}
          </button>
        ))}
      </div>

      {/* Selected Model Details */}
      <div className="bg-gradient-to-r from-emerald-500/10 to-teal-500/10 rounded-xl p-6 border border-emerald-500/30">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold text-emerald-600 dark:text-emerald-400">{selectedModelData.name}</h3>
          {selectedModelData.sdUse && (
            <span className="px-3 py-1 bg-yellow-500/20 text-yellow-400 rounded-full text-sm">
              Used in SD3
            </span>
          )}
        </div>

        <div className="grid md:grid-cols-4 gap-4 mb-6">
          <div className="bg-black/30 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">{selectedModelData.params}</div>
            <div className="text-sm text-gray-800 dark:text-gray-400">Parameters</div>
          </div>
          <div className="bg-black/30 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-teal-600 dark:text-teal-400">{selectedModelData.layers}</div>
            <div className="text-sm text-gray-800 dark:text-gray-400">Encoder Layers</div>
          </div>
          <div className="bg-black/30 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-cyan-600 dark:text-cyan-400">{selectedModelData.hidden}</div>
            <div className="text-sm text-gray-800 dark:text-gray-400">Hidden Dim</div>
          </div>
          <div className="bg-black/30 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{selectedModelData.heads}</div>
            <div className="text-sm text-gray-800 dark:text-gray-400">Attention Heads</div>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-4">
          <div className="flex items-center gap-3 bg-black/30 rounded-lg p-3">
            <HardDrive className="text-purple-600 dark:text-purple-400" size={20} />
            <div>
              <div className="text-sm text-gray-800 dark:text-gray-400">VRAM</div>
              <div className="text-white font-semibold">{selectedModelData.memory}</div>
            </div>
          </div>
          <div className="flex items-center gap-3 bg-black/30 rounded-lg p-3">
            <Clock className="text-orange-600 dark:text-orange-400" size={20} />
            <div>
              <div className="text-sm text-gray-800 dark:text-gray-400">Speed</div>
              <div className="text-white font-semibold">{selectedModelData.speed}</div>
            </div>
          </div>
          <div className="flex items-center gap-3 bg-black/30 rounded-lg p-3">
            <TrendingUp className="text-green-400" size={20} />
            <div>
              <div className="text-sm text-gray-800 dark:text-gray-400">Quality</div>
              <div className="text-white font-semibold">{selectedModelData.quality}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Scale Visualization */}
      <div className="bg-black/40 rounded-xl p-6">
        <h3 className="font-semibold text-gray-700 dark:text-gray-300 mb-4">📊 Parameter Scale Comparison</h3>
        <div className="space-y-3">
          {Object.entries(models).map(([key, model]) => {
            const width = (paramValues[key] / maxParams) * 100;
            const isSelected = key === selectedModel;
            return (
              <div key={key} className="flex items-center gap-4">
                <div className="w-20 text-sm text-gray-800 dark:text-gray-400">{model.name}</div>
                <div className="flex-1 bg-black/50 rounded-full h-6 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 flex items-center justify-end pr-2 ${
                      isSelected ? 'bg-emerald-500' : 'bg-emerald-500/40'
                    }`}
                    style={{ width: `${Math.max(width, 5)}%` }}
                  >
                    <span className="text-xs text-white font-medium">{model.params}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Why XXL */}
      <div className="bg-yellow-500/10 rounded-xl p-6 border border-yellow-500/30">
        <h3 className="font-semibold text-yellow-400 mb-3 flex items-center gap-2">
          <Scale size={20} />
          Why SD3 Uses T5-XXL
        </h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-700 dark:text-gray-300">
          <div className="space-y-2">
            <p><strong className="text-yellow-400">4096-dim embeddings</strong></p>
            <p>Much richer representations than CLIP's 768/1280. Can capture subtle nuances in prompts.</p>
          </div>
          <div className="space-y-2">
            <p><strong className="text-yellow-400">4.7B encoder parameters</strong></p>
            <p>Massive capacity for language understanding. Better at complex relationships and details.</p>
          </div>
        </div>
      </div>

      {/* Scaling Laws */}
      <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-xl p-6 border border-purple-500/30">
        <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-4">📈 Scaling Laws in Practice</h3>
        <div className="space-y-4 text-sm text-gray-700 dark:text-gray-300">
          <div className="bg-black/30 rounded-lg p-4">
            <div className="font-mono text-xs mb-2">
              Quality ∝ log(Parameters) × log(Data) × log(Compute)
            </div>
            <p className="text-gray-800 dark:text-gray-400">
              T5-XXL follows the scaling laws - more parameters = better understanding, 
              but with diminishing returns and increasing costs.
            </p>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 px-3 text-gray-800 dark:text-gray-400">Metric</th>
                  <th className="text-left py-2 px-3">Base → Large</th>
                  <th className="text-left py-2 px-3">Large → XL</th>
                  <th className="text-left py-2 px-3">XL → XXL</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-white/5">
                  <td className="py-2 px-3 text-gray-800 dark:text-gray-400">Params</td>
                  <td className="py-2 px-3 text-emerald-600 dark:text-emerald-400">3.5×</td>
                  <td className="py-2 px-3 text-emerald-600 dark:text-emerald-400">4×</td>
                  <td className="py-2 px-3 text-emerald-600 dark:text-emerald-400">3.7×</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-2 px-3 text-gray-800 dark:text-gray-400">Hidden Dim</td>
                  <td className="py-2 px-3">768→1024</td>
                  <td className="py-2 px-3">1024→2048</td>
                  <td className="py-2 px-3">2048→4096</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-2 px-3 text-gray-800 dark:text-gray-400">Memory</td>
                  <td className="py-2 px-3">3×</td>
                  <td className="py-2 px-3">2×</td>
                  <td className="py-2 px-3">1.5×</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Practical Tips */}
      <div className="bg-emerald-500/10 rounded-xl p-6 border border-emerald-500/30">
        <h3 className="font-semibold text-emerald-600 dark:text-emerald-400 mb-3">💡 Practical Considerations</h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-700 dark:text-gray-300">
          <div className="bg-black/30 rounded-lg p-4">
            <div className="font-semibold text-white mb-2">When to use T5</div>
            <ul className="list-disc list-inside space-y-1 text-gray-800 dark:text-gray-400">
              <li>Complex, detailed prompts</li>
              <li>Spatial relationships</li>
              <li>Negations and qualifiers</li>
              <li>Long descriptions (100+ tokens)</li>
            </ul>
          </div>
          <div className="bg-black/30 rounded-lg p-4">
            <div className="font-semibold text-white mb-2">When to skip T5</div>
            <ul className="list-disc list-inside space-y-1 text-gray-800 dark:text-gray-400">
              <li>Simple prompts</li>
              <li>Fast generation needed</li>
              <li>Limited VRAM (&lt;12GB)</li>
              <li>Batch processing</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ScalePanel;
