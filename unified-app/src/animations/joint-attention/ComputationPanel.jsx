import React, { useState } from 'react';
import { Calculator, Zap, Clock, HardDrive } from 'lucide-react';

export default function ComputationPanel() {
  const [imgSize, setImgSize] = useState(64);
  const [txtLen, setTxtLen] = useState(77);
  const [dModel, setDModel] = useState(1536);
  const [numHeads, setNumHeads] = useState(24);

  // Calculations
  const imgTokens = imgSize * imgSize;
  const totalTokens = imgTokens + txtLen;
  
  // Attention FLOPs: 4 * seq_len^2 * d_model (Q, K, V projections + attention)
  const attentionFlops = 4 * totalTokens * totalTokens * dModel;
  
  // Memory for attention matrix: seq_len^2 * num_heads * 2 bytes (fp16)
  const attentionMemory = totalTokens * totalTokens * numHeads * 2;
  
  // Compare to cross-attention
  const crossAttnFlops = 4 * imgTokens * txtLen * dModel + 4 * imgTokens * imgTokens * dModel;
  
  const formatNumber = (n) => {
    if (n >= 1e12) return (n / 1e12).toFixed(2) + 'T';
    if (n >= 1e9) return (n / 1e9).toFixed(2) + 'G';
    if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(2) + 'K';
    return n.toString();
  };

  const formatBytes = (b) => {
    if (b >= 1e9) return (b / 1e9).toFixed(2) + ' GB';
    if (b >= 1e6) return (b / 1e6).toFixed(2) + ' MB';
    if (b >= 1e3) return (b / 1e3).toFixed(2) + ' KB';
    return b + ' B';
  };

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="text-violet-400">Computational</span> Costs
        </h2>
        <p className="text-gray-800 dark:text-gray-400">
          Understanding the memory and compute requirements of joint attention
        </p>
      </div>

      {/* Calculator */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
          <Calculator size={20} className="text-violet-400" />
          Interactive Calculator
        </h3>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div>
            <label className="text-sm text-gray-800 dark:text-gray-400 block mb-2">
              Image Size: {imgSize}×{imgSize} ({imgTokens} tokens)
            </label>
            <input
              type="range"
              min="16"
              max="128"
              step="16"
              value={imgSize}
              onChange={(e) => setImgSize(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label className="text-sm text-gray-800 dark:text-gray-400 block mb-2">
              Text Length: {txtLen} tokens
            </label>
            <input
              type="range"
              min="32"
              max="256"
              step="1"
              value={txtLen}
              onChange={(e) => setTxtLen(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label className="text-sm text-gray-800 dark:text-gray-400 block mb-2">
              d_model: {dModel}
            </label>
            <select
              value={dModel}
              onChange={(e) => setDModel(parseInt(e.target.value))}
              className="w-full bg-white/10 rounded-lg px-3 py-2 text-white"
            >
              <option value="1024">1024 (Small)</option>
              <option value="1536">1536 (SD3-Medium)</option>
              <option value="3072">3072 (SD3-Large)</option>
            </select>
          </div>
          <div>
            <label className="text-sm text-gray-800 dark:text-gray-400 block mb-2">
              Attention Heads: {numHeads}
            </label>
            <input
              type="range"
              min="8"
              max="48"
              step="8"
              value={numHeads}
              onChange={(e) => setNumHeads(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
        </div>

        {/* Results */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-violet-900/30 rounded-xl p-4 border border-violet-500/30">
            <p className="text-xs text-gray-800 dark:text-gray-400 mb-1">Total Sequence</p>
            <p className="text-2xl font-bold text-violet-400">{formatNumber(totalTokens)}</p>
            <p className="text-xs text-gray-700 dark:text-gray-500">{imgTokens} img + {txtLen} txt</p>
          </div>
          <div className="bg-blue-900/30 rounded-xl p-4 border border-blue-500/30">
            <p className="text-xs text-gray-800 dark:text-gray-400 mb-1">Attention FLOPs</p>
            <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">{formatNumber(attentionFlops)}</p>
            <p className="text-xs text-gray-700 dark:text-gray-500">per layer</p>
          </div>
          <div className="bg-orange-900/30 rounded-xl p-4 border border-orange-500/30">
            <p className="text-xs text-gray-800 dark:text-gray-400 mb-1">Attn Matrix Memory</p>
            <p className="text-2xl font-bold text-orange-600 dark:text-orange-400">{formatBytes(attentionMemory)}</p>
            <p className="text-xs text-gray-700 dark:text-gray-500">FP16, per layer</p>
          </div>
          <div className="bg-green-900/30 rounded-xl p-4 border border-green-500/30">
            <p className="text-xs text-gray-800 dark:text-gray-400 mb-1">vs Cross-Attn</p>
            <p className="text-2xl font-bold text-green-400">
              {((attentionFlops / crossAttnFlops - 1) * 100).toFixed(0)}%
            </p>
            <p className="text-xs text-gray-700 dark:text-gray-500">more compute</p>
          </div>
        </div>
      </div>

      {/* Complexity Analysis */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">Complexity Breakdown</h3>
        
        <div className="space-y-4">
          {/* Self-attention on joint sequence */}
          <div className="bg-violet-900/20 rounded-xl p-4 border border-violet-500/30">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-bold text-violet-400">Joint Self-Attention</h4>
              <span className="font-mono text-violet-300">O((N_img + N_txt)²)</span>
            </div>
            <p className="text-sm text-gray-800 dark:text-gray-400">
              Every token attends to every other token. For 4096 image + 77 text = 4173² ≈ 17.4M attention scores per head.
            </p>
            <div className="mt-2 w-full bg-white/10 rounded-full h-3">
              <div 
                className="bg-gradient-to-r from-violet-500 to-fuchsia-500 h-3 rounded-full"
                style={{ width: '100%' }}
              />
            </div>
          </div>

          {/* Cross attention equivalent */}
          <div className="bg-blue-900/20 rounded-xl p-4 border border-blue-500/30">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-bold text-blue-600 dark:text-blue-400">Cross-Attention (for comparison)</h4>
              <span className="font-mono text-blue-300">O(N_img × N_txt)</span>
            </div>
            <p className="text-sm text-gray-800 dark:text-gray-400">
              Only image queries text. For 4096 × 77 ≈ 315K attention scores - much smaller!
            </p>
            <div className="mt-2 w-full bg-white/10 rounded-full h-3">
              <div 
                className="bg-gradient-to-r from-blue-500 to-cyan-500 h-3 rounded-full"
                style={{ width: `${(imgTokens * txtLen) / (totalTokens * totalTokens) * 100}%` }}
              />
            </div>
          </div>
        </div>

        <div className="mt-6 p-4 bg-yellow-900/20 rounded-xl border border-yellow-500/30">
          <p className="text-sm">
            <strong>⚠️ Why is joint attention worth the cost?</strong><br/>
            The quadratic scaling means joint attention is computationally expensive, but:
            <ul className="mt-2 text-gray-700 dark:text-gray-300 space-y-1">
              <li>• Flash Attention reduces memory from O(N²) to O(N)</li>
              <li>• Modern GPUs handle large matrix ops efficiently</li>
              <li>• The quality gains are significant</li>
              <li>• Text tokens are few compared to image tokens</li>
            </ul>
          </p>
        </div>
      </div>

      {/* Optimization Techniques */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-gradient-to-br from-green-900/30 to-emerald-800/20 rounded-xl p-5 border border-green-500/30">
          <h3 className="font-bold text-green-300 mb-3 flex items-center gap-2">
            <Zap size={18} /> Flash Attention
          </h3>
          <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
            <li>• Fused CUDA kernels</li>
            <li>• O(N) memory instead of O(N²)</li>
            <li>• 2-4× faster training</li>
            <li>• Enables longer sequences</li>
          </ul>
        </div>

        <div className="bg-gradient-to-br from-purple-900/30 to-violet-800/20 rounded-xl p-5 border border-purple-500/30">
          <h3 className="font-bold text-purple-300 mb-3 flex items-center gap-2">
            <Clock size={18} /> xFormers
          </h3>
          <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
            <li>• Memory-efficient attention</li>
            <li>• Automatic kernel selection</li>
            <li>• Works with various GPUs</li>
            <li>• Easy integration</li>
          </ul>
        </div>

        <div className="bg-gradient-to-br from-blue-900/30 to-cyan-800/20 rounded-xl p-5 border border-blue-500/30">
          <h3 className="font-bold text-blue-300 mb-3 flex items-center gap-2">
            <HardDrive size={18} /> Gradient Checkpointing
          </h3>
          <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
            <li>• Trade compute for memory</li>
            <li>• Recompute activations</li>
            <li>• Enable larger batches</li>
            <li>• Essential for training</li>
          </ul>
        </div>
      </div>

      {/* Hardware Requirements Table */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">Resolution vs Memory (Inference, FP16)</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/20">
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Resolution</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Image Tokens</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Total Seq</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Attn Memory</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Total VRAM</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/10">
              <tr>
                <td className="py-3 px-4 text-violet-400">512×512</td>
                <td className="py-3 px-4">1024</td>
                <td className="py-3 px-4">~1100</td>
                <td className="py-3 px-4">~58 MB</td>
                <td className="py-3 px-4 text-green-400">~6 GB</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-violet-400">768×768</td>
                <td className="py-3 px-4">2304</td>
                <td className="py-3 px-4">~2400</td>
                <td className="py-3 px-4">~276 MB</td>
                <td className="py-3 px-4 text-yellow-400">~8 GB</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-violet-400">1024×1024</td>
                <td className="py-3 px-4">4096</td>
                <td className="py-3 px-4">~4200</td>
                <td className="py-3 px-4">~847 MB</td>
                <td className="py-3 px-4 text-orange-600 dark:text-orange-400">~12 GB</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-violet-400">1536×1536</td>
                <td className="py-3 px-4">9216</td>
                <td className="py-3 px-4">~9300</td>
                <td className="py-3 px-4">~4.1 GB</td>
                <td className="py-3 px-4 text-red-400">~20+ GB</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="text-xs text-gray-700 dark:text-gray-500 mt-4">
          Note: Numbers are approximate. Flash Attention can reduce attention memory significantly.
          Total VRAM includes model weights, activations, and optimizer states.
        </p>
      </div>

      {/* Code Snippet */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">Enabling Flash Attention</h3>
        <div className="bg-black/50 rounded-lg p-4 font-mono text-sm overflow-x-auto">
          <pre className="text-gray-700 dark:text-gray-300">{`# In diffusers, enable memory-efficient attention
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
)

# Option 1: Enable xFormers (if installed)
pipe.enable_xformers_memory_efficient_attention()

# Option 2: Enable native PyTorch flash attention
# Requires PyTorch 2.0+ with CUDA
pipe.enable_attention_slicing()  # For lower VRAM

# Option 3: Use scaled_dot_product_attention (automatic in PyTorch 2.0+)
# No explicit call needed - happens automatically`}</pre>
        </div>
      </div>
    </div>
  );
}
