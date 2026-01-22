import React, { useState } from 'react';
import { ArrowDown, Layers } from 'lucide-react';

export default function BlockStructurePanel() {
  const [selectedBlock, setSelectedBlock] = useState('standard');

  const blockTypes = {
    standard: {
      name: 'Standard ViT Block',
      components: [
        { name: 'LayerNorm', color: 'bg-gray-500', desc: 'Normalize activations' },
        { name: 'Self-Attention', color: 'bg-blue-500', desc: 'Global token mixing' },
        { name: 'LayerNorm', color: 'bg-gray-500', desc: 'Normalize activations' },
        { name: 'MLP', color: 'bg-green-500', desc: 'Per-token feedforward' },
      ],
      residual: true,
    },
    dit: {
      name: 'DiT Block',
      components: [
        { name: 'AdaLN', color: 'bg-pink-500', desc: 'Adaptive norm with conditioning' },
        { name: 'Self-Attention', color: 'bg-blue-500', desc: 'Global token mixing' },
        { name: 'AdaLN', color: 'bg-pink-500', desc: 'Adaptive norm with conditioning' },
        { name: 'MLP', color: 'bg-green-500', desc: 'Per-token feedforward' },
      ],
      residual: true,
      gated: true,
    },
    mmdit: {
      name: 'MM-DiT Block',
      components: [
        { name: 'AdaLN (Img)', color: 'bg-blue-500', desc: 'Norm for image tokens' },
        { name: 'AdaLN (Txt)', color: 'bg-orange-500', desc: 'Norm for text tokens' },
        { name: 'Joint Attention', color: 'bg-violet-500', desc: 'Image+Text attention' },
        { name: 'AdaLN (Img)', color: 'bg-blue-500', desc: 'Norm for image tokens' },
        { name: 'AdaLN (Txt)', color: 'bg-orange-500', desc: 'Norm for text tokens' },
        { name: 'MLP (Img)', color: 'bg-cyan-500', desc: 'Image feedforward' },
        { name: 'MLP (Txt)', color: 'bg-amber-500', desc: 'Text feedforward' },
      ],
      residual: true,
      gated: true,
      separate: true,
    },
  };

  const current = blockTypes[selectedBlock];

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="text-pink-600 dark:text-pink-400">Block</span> Structure
        </h2>
        <p className="text-gray-800 dark:text-gray-400">
          Inside a DiT transformer block: from standard ViT to MM-DiT
        </p>
      </div>

      {/* Block Type Selector */}
      <div className="flex justify-center gap-4">
        {Object.entries(blockTypes).map(([key, block]) => (
          <button
            key={key}
            onClick={() => setSelectedBlock(key)}
            className={`px-4 py-2 rounded-lg transition-all ${
              selectedBlock === key
                ? 'bg-gradient-to-r from-pink-600 to-orange-600 text-white'
                : 'bg-white/10 text-gray-800 dark:text-gray-400 hover:bg-white/20'
            }`}
          >
            {block.name}
          </button>
        ))}
      </div>

      {/* Block Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-6 text-center">{current.name}</h3>
        
        <div className="flex justify-center">
          <div className="relative">
            {/* Input */}
            <div className="text-center mb-4">
              <div className="w-32 h-10 mx-auto rounded-lg bg-gray-700 flex items-center justify-center">
                <span className="text-sm">Input x</span>
              </div>
              <ArrowDown className="mx-auto mt-2 text-gray-700 dark:text-gray-500" />
            </div>

            {/* Residual wrapper */}
            <div className="relative border-2 border-dashed border-pink-500/30 rounded-xl p-4 mx-auto max-w-md">
              {current.residual && (
                <div className="absolute -left-8 top-1/2 -translate-y-1/2 w-6 h-32 border-l-2 border-t-2 border-b-2 border-pink-400/50 rounded-l-lg">
                  <span className="absolute -left-6 top-1/2 -translate-y-1/2 text-pink-600 dark:text-xs">+</span>
                </div>
              )}
              
              {/* Components */}
              {current.separate ? (
                // MM-DiT has parallel streams
                <div className="flex gap-4">
                  <div className="flex-1 space-y-3">
                    <p className="text-xs text-blue-600 dark:text-center mb-2">Image Stream</p>
                    {current.components.filter(c => c.name.includes('Img') || c.name.includes('Joint')).map((comp, i) => (
                      <div key={i} className={`${comp.color} rounded-lg p-3 text-center`}>
                        <span className="text-sm font-medium">{comp.name.replace(' (Img)', '')}</span>
                      </div>
                    ))}
                  </div>
                  <div className="flex-1 space-y-3">
                    <p className="text-xs text-orange-600 dark:text-center mb-2">Text Stream</p>
                    {current.components.filter(c => c.name.includes('Txt') || c.name.includes('Joint')).map((comp, i) => (
                      <div key={i} className={`${comp.color} rounded-lg p-3 text-center`}>
                        <span className="text-sm font-medium">{comp.name.replace(' (Txt)', '')}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                // Standard/DiT single stream
                <div className="space-y-3">
                  {current.components.map((comp, i) => (
                    <div key={i}>
                      <div className={`${comp.color} rounded-lg p-3 text-center`}>
                        <span className="text-sm font-medium">{comp.name}</span>
                      </div>
                      {i < current.components.length - 1 && (
                        <ArrowDown className="mx-auto my-1 text-gray-700 dark:text-gray-500" size={16} />
                      )}
                    </div>
                  ))}
                </div>
              )}
              
              {current.gated && (
                <div className="absolute -right-20 top-1/2 -translate-y-1/2 text-xs text-pink-600 dark:text-pink-400">
                  ×α (gate)
                </div>
              )}
            </div>

            {/* Output */}
            <div className="text-center mt-4">
              <ArrowDown className="mx-auto mb-2 text-gray-700 dark:text-gray-500" />
              <div className="w-32 h-10 mx-auto rounded-lg bg-gradient-to-r from-pink-700 to-orange-700 flex items-center justify-center">
                <span className="text-sm">Output x'</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Component Details */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">Component Details</h3>
        <div className="grid md:grid-cols-2 gap-4">
          {current.components
            .filter((comp, i, arr) => arr.findIndex(c => c.name === comp.name) === i)
            .map((comp, i) => (
              <div key={i} className={`${comp.color}/20 rounded-xl p-4 border border-white/10`}>
                <h4 className="font-bold mb-2">{comp.name}</h4>
                <p className="text-sm text-gray-800 dark:text-gray-400">{comp.desc}</p>
              </div>
            ))}
        </div>
      </div>

      {/* Comparison Table */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">Block Type Comparison</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/20">
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Feature</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Standard ViT</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">DiT</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">MM-DiT</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/10">
              <tr>
                <td className="py-3 px-4">Normalization</td>
                <td className="py-3 px-4">LayerNorm</td>
                <td className="py-3 px-4 text-pink-600 dark:text-pink-400">AdaLN</td>
                <td className="py-3 px-4 text-pink-600 dark:text-pink-400">AdaLN (separate)</td>
              </tr>
              <tr>
                <td className="py-3 px-4">Conditioning</td>
                <td className="py-3 px-4">None (class token)</td>
                <td className="py-3 px-4 text-pink-600 dark:text-pink-400">Via AdaLN</td>
                <td className="py-3 px-4 text-pink-600 dark:text-pink-400">Via AdaLN + Joint Attn</td>
              </tr>
              <tr>
                <td className="py-3 px-4">Attention Type</td>
                <td className="py-3 px-4">Self only</td>
                <td className="py-3 px-4">Self only</td>
                <td className="py-3 px-4 text-violet-400">Joint (Img+Txt)</td>
              </tr>
              <tr>
                <td className="py-3 px-4">Gating</td>
                <td className="py-3 px-4">No</td>
                <td className="py-3 px-4 text-green-400">Yes (α=0 init)</td>
                <td className="py-3 px-4 text-green-400">Yes (α=0 init)</td>
              </tr>
              <tr>
                <td className="py-3 px-4">MLP Streams</td>
                <td className="py-3 px-4">1 (shared)</td>
                <td className="py-3 px-4">1 (shared)</td>
                <td className="py-3 px-4 text-cyan-600 dark:text-cyan-400">2 (separate)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">DiT Block in Code</h3>
        <div className="bg-black/50 rounded-lg p-4 font-mono text-sm overflow-x-auto">
          <pre className="text-gray-700 dark:text-gray-300">{`class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = AdaLayerNorm(hidden_size)  # AdaLN instead of LayerNorm
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = AdaLayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, int(hidden_size * mlp_ratio))
        
        # Gating parameters (initialized to 0)
        self.gate_attn = nn.Parameter(torch.zeros(hidden_size))
        self.gate_mlp = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x, c):
        # c = conditioning embedding (timestep + text pooled)
        
        # Attention with AdaLN and gating
        x = x + self.gate_attn * self.attn(self.norm1(x, c))
        
        # MLP with AdaLN and gating
        x = x + self.gate_mlp * self.mlp(self.norm2(x, c))
        
        return x`}</pre>
        </div>
      </div>
    </div>
  );
}
