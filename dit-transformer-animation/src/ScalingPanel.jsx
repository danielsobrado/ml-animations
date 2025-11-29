import React, { useState } from 'react';
import { TrendingUp } from 'lucide-react';

export default function ScalingPanel() {
  const [selectedModel, setSelectedModel] = useState('XL/2');

  const models = {
    'S/2': { params: 33, gflops: 1.4, fid: 68.4, depth: 12, width: 384, heads: 6 },
    'B/2': { params: 130, gflops: 5.6, fid: 43.5, depth: 12, width: 768, heads: 12 },
    'L/2': { params: 458, gflops: 20, fid: 23.3, depth: 24, width: 1024, heads: 16 },
    'XL/2': { params: 675, gflops: 29, fid: 9.62, depth: 28, width: 1152, heads: 16 },
  };

  const selected = models[selectedModel];

  // Calculate relative bar widths
  const maxParams = 675;
  const maxGflops = 29;
  const maxFid = 70;

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="text-pink-400">Scaling</span> Laws
        </h2>
        <p className="text-gray-400">
          How DiT performance improves with model size
        </p>
      </div>

      {/* Key Insight */}
      <div className="bg-gradient-to-r from-pink-900/30 to-orange-900/30 rounded-xl p-6 border border-pink-500/30">
        <h3 className="font-bold text-pink-300 mb-2 flex items-center gap-2">
          <TrendingUp size={20} />
          The Key Finding
        </h3>
        <p className="text-gray-300">
          DiT follows <strong>predictable scaling laws</strong> similar to LLMs. Doubling compute consistently 
          improves FID score. This is why SD3 and Flux use massive DiT models (2B-12B parameters).
        </p>
      </div>

      {/* Model Selector */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">DiT Model Variants</h3>
        
        <div className="flex justify-center gap-3 mb-6 flex-wrap">
          {Object.keys(models).map((name) => (
            <button
              key={name}
              onClick={() => setSelectedModel(name)}
              className={`px-4 py-2 rounded-lg transition-all ${
                selectedModel === name
                  ? 'bg-gradient-to-r from-pink-600 to-orange-600 text-white'
                  : 'bg-white/10 text-gray-400 hover:bg-white/20'
              }`}
            >
              DiT-{name}
            </button>
          ))}
        </div>

        {/* Selected Model Stats */}
        <div className="grid md:grid-cols-3 gap-4 mb-6">
          <div className="bg-pink-900/30 rounded-xl p-4 text-center border border-pink-500/30">
            <p className="text-3xl font-bold text-pink-400">{selected.params}M</p>
            <p className="text-sm text-gray-400">Parameters</p>
          </div>
          <div className="bg-orange-900/30 rounded-xl p-4 text-center border border-orange-500/30">
            <p className="text-3xl font-bold text-orange-400">{selected.gflops}</p>
            <p className="text-sm text-gray-400">GFLOPs</p>
          </div>
          <div className="bg-green-900/30 rounded-xl p-4 text-center border border-green-500/30">
            <p className="text-3xl font-bold text-green-400">{selected.fid}</p>
            <p className="text-sm text-gray-400">FID (↓ better)</p>
          </div>
        </div>

        {/* Architecture Details */}
        <div className="grid grid-cols-3 gap-4 text-center text-sm">
          <div className="bg-white/5 rounded-lg p-3">
            <p className="text-gray-400">Depth</p>
            <p className="text-lg font-bold">{selected.depth} layers</p>
          </div>
          <div className="bg-white/5 rounded-lg p-3">
            <p className="text-gray-400">Width</p>
            <p className="text-lg font-bold">{selected.width}</p>
          </div>
          <div className="bg-white/5 rounded-lg p-3">
            <p className="text-gray-400">Heads</p>
            <p className="text-lg font-bold">{selected.heads}</p>
          </div>
        </div>
      </div>

      {/* Comparison Chart */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-6">Model Comparison</h3>
        
        <div className="space-y-6">
          {Object.entries(models).map(([name, model]) => (
            <div key={name} className={`transition-all ${selectedModel === name ? 'scale-105' : 'opacity-70'}`}>
              <div className="flex items-center gap-4 mb-2">
                <span className="w-20 font-bold text-gray-300">DiT-{name}</span>
                <div className="flex-1">
                  {/* Parameters bar */}
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs text-gray-500 w-16">Params:</span>
                    <div className="flex-1 bg-white/10 rounded-full h-4">
                      <div 
                        className="bg-gradient-to-r from-pink-500 to-pink-400 h-4 rounded-full flex items-center justify-end pr-2"
                        style={{ width: `${(model.params / maxParams) * 100}%` }}
                      >
                        <span className="text-xs font-bold">{model.params}M</span>
                      </div>
                    </div>
                  </div>
                  {/* FID bar (inverted - lower is better) */}
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-500 w-16">FID:</span>
                    <div className="flex-1 bg-white/10 rounded-full h-4">
                      <div 
                        className="bg-gradient-to-r from-green-500 to-emerald-400 h-4 rounded-full flex items-center justify-end pr-2"
                        style={{ width: `${(1 - model.fid / maxFid) * 100}%` }}
                      >
                        <span className="text-xs font-bold">{model.fid}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Scaling Laws Graph (simplified) */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">FID vs Compute (Log Scale)</h3>
        <div className="h-64 relative bg-black/30 rounded-xl p-4">
          {/* Y axis */}
          <div className="absolute left-0 top-0 bottom-0 w-12 flex flex-col justify-between text-xs text-gray-500">
            <span>70</span>
            <span>35</span>
            <span>10</span>
          </div>
          
          {/* Grid */}
          <div className="absolute left-12 right-4 top-4 bottom-8">
            <div className="h-full border-l border-b border-white/10 relative">
              {/* Data points */}
              {Object.entries(models).map(([name, model], i) => {
                const x = (Math.log(model.gflops) / Math.log(30)) * 100;
                const y = ((maxFid - model.fid) / maxFid) * 100;
                return (
                  <div
                    key={name}
                    className={`absolute w-4 h-4 rounded-full cursor-pointer transition-all ${
                      selectedModel === name ? 'bg-pink-500 scale-150' : 'bg-pink-400'
                    }`}
                    style={{ 
                      left: `${x}%`, 
                      bottom: `${y}%`,
                      transform: 'translate(-50%, 50%)'
                    }}
                    onClick={() => setSelectedModel(name)}
                  >
                    <span className="absolute -top-6 left-1/2 -translate-x-1/2 text-xs whitespace-nowrap">
                      {name}
                    </span>
                  </div>
                );
              })}
              
              {/* Trend line (approximate) */}
              <svg className="absolute inset-0 overflow-visible">
                <path
                  d="M 10,85 Q 40,50 90,10"
                  fill="none"
                  stroke="rgba(236, 72, 153, 0.3)"
                  strokeWidth="2"
                  strokeDasharray="4"
                />
              </svg>
            </div>
          </div>
          
          {/* X axis label */}
          <div className="absolute bottom-0 left-12 right-4 text-center text-xs text-gray-500">
            GFLOPs (log scale)
          </div>
        </div>
        <p className="text-xs text-gray-500 mt-2 text-center">
          Each doubling of compute roughly halves FID error (log-linear relationship)
        </p>
      </div>

      {/* SD3 Model Sizes */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">SD3 Model Family</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/20">
                <th className="py-3 px-4 text-left text-gray-400">Model</th>
                <th className="py-3 px-4 text-left text-gray-400">Parameters</th>
                <th className="py-3 px-4 text-left text-gray-400">Layers</th>
                <th className="py-3 px-4 text-left text-gray-400">Hidden Dim</th>
                <th className="py-3 px-4 text-left text-gray-400">Use Case</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/10">
              <tr>
                <td className="py-3 px-4 text-pink-400">SD3-Medium</td>
                <td className="py-3 px-4">2B</td>
                <td className="py-3 px-4">24</td>
                <td className="py-3 px-4">1536</td>
                <td className="py-3 px-4 text-green-400">Consumer GPUs (8GB+)</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-purple-400">SD3</td>
                <td className="py-3 px-4">3B</td>
                <td className="py-3 px-4">32</td>
                <td className="py-3 px-4">1920</td>
                <td className="py-3 px-4 text-yellow-400">Mid-range (12GB+)</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-orange-400">SD3.5-Large</td>
                <td className="py-3 px-4">8B</td>
                <td className="py-3 px-4">38</td>
                <td className="py-3 px-4">2432</td>
                <td className="py-3 px-4 text-red-400">High-end (24GB+)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Why Scaling Works */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-green-900/20 rounded-xl p-5 border border-green-500/30">
          <h3 className="font-bold text-green-400 mb-3">✓ Why Scaling Works</h3>
          <ul className="text-sm text-gray-300 space-y-2">
            <li>• More parameters = more capacity to model data</li>
            <li>• Transformers parallelize efficiently</li>
            <li>• Uniform architecture (no bottlenecks)</li>
            <li>• Proven by LLM research (GPT, LLaMA)</li>
          </ul>
        </div>
        
        <div className="bg-orange-900/20 rounded-xl p-5 border border-orange-500/30">
          <h3 className="font-bold text-orange-400 mb-3">⚠️ Practical Limits</h3>
          <ul className="text-sm text-gray-300 space-y-2">
            <li>• Training cost grows substantially</li>
            <li>• Inference speed decreases</li>
            <li>• Memory requirements increase</li>
            <li>• Diminishing returns eventually</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
