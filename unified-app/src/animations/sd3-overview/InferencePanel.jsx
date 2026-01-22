import React, { useState } from 'react';
import { ArrowRight, Clock, Cpu, Zap, Settings } from 'lucide-react';

export default function InferencePanel() {
  const [resolution, setResolution] = useState('1024');
  const [steps, setSteps] = useState(28);
  const [guidance, setGuidance] = useState(5);

  const resolutions = {
    '512': { size: 512, tokens: 1024, vram: '~6 GB', time: '5s' },
    '768': { size: 768, tokens: 2304, vram: '~8 GB', time: '8s' },
    '1024': { size: 1024, tokens: 4096, vram: '~12 GB', time: '12s' },
    '1536': { size: 1536, tokens: 9216, vram: '~20 GB', time: '25s' },
  };

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          Inference Pipeline: <span className="text-fuchsia-600 dark:text-fuchsia-400">From Prompt to Image</span>
        </h2>
        <p className="text-gray-800 dark:text-gray-400">
          The complete generation pipeline with configurable parameters
        </p>
      </div>

      {/* Pipeline Steps */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-6 text-center">Generation Pipeline</h3>
        
        <div className="flex flex-col md:flex-row items-start justify-between gap-4">
          {/* Step 1 */}
          <div className="flex-1 text-center">
            <div className="w-16 h-16 mx-auto rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center mb-3">
              <span className="text-2xl">📝</span>
            </div>
            <h4 className="font-bold text-blue-600 dark:text-blue-400 mb-2">1. Text Encoding</h4>
            <p className="text-xs text-gray-800 dark:text-gray-400">
              CLIP-L, CLIP-G, T5-XXL<br/>
              encode the prompt
            </p>
            <div className="mt-2 text-xs text-gray-700 dark:text-gray-500">
              ~0.5s
            </div>
          </div>

          <ArrowRight className="text-gray-700 dark:text-gray-500 hidden md:block mt-8" />

          {/* Step 2 */}
          <div className="flex-1 text-center">
            <div className="w-16 h-16 mx-auto rounded-xl bg-gradient-to-br from-gray-500 to-slate-500 flex items-center justify-center mb-3">
              <span className="text-2xl">🎲</span>
            </div>
            <h4 className="font-bold text-gray-800 dark:text-gray-400 mb-2">2. Sample Noise</h4>
            <p className="text-xs text-gray-800 dark:text-gray-400">
              z₀ ~ N(0, I)<br/>
              Random latent init
            </p>
            <div className="mt-2 text-xs text-gray-700 dark:text-gray-500">
              ~0.01s
            </div>
          </div>

          <ArrowRight className="text-gray-700 dark:text-gray-500 hidden md:block mt-8" />

          {/* Step 3 */}
          <div className="flex-1 text-center">
            <div className="w-16 h-16 mx-auto rounded-xl bg-gradient-to-br from-fuchsia-500 to-pink-500 flex items-center justify-center mb-3">
              <span className="text-2xl">🔄</span>
            </div>
            <h4 className="font-bold text-fuchsia-600 dark:text-fuchsia-400 mb-2">3. Denoise Loop</h4>
            <p className="text-xs text-gray-800 dark:text-gray-400">
              MMDiT + Euler<br/>
              {steps} iterations
            </p>
            <div className="mt-2 text-xs text-gray-700 dark:text-gray-500">
              ~{(steps * 0.3).toFixed(1)}s
            </div>
          </div>

          <ArrowRight className="text-gray-700 dark:text-gray-500 hidden md:block mt-8" />

          {/* Step 4 */}
          <div className="flex-1 text-center">
            <div className="w-16 h-16 mx-auto rounded-xl bg-gradient-to-br from-rose-500 to-red-500 flex items-center justify-center mb-3">
              <span className="text-2xl">🖼️</span>
            </div>
            <h4 className="font-bold text-rose-400 mb-2">4. VAE Decode</h4>
            <p className="text-xs text-gray-800 dark:text-gray-400">
              Latent → Pixels<br/>
              {resolution}×{resolution}
            </p>
            <div className="mt-2 text-xs text-gray-700 dark:text-gray-500">
              ~1s
            </div>
          </div>
        </div>
      </div>

      {/* Configuration Panel */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
          <Settings size={20} className="text-fuchsia-600 dark:text-fuchsia-400" />
          Generation Settings
        </h3>

        <div className="grid md:grid-cols-3 gap-6">
          {/* Resolution */}
          <div>
            <label className="text-sm text-gray-800 dark:text-gray-400 block mb-3">Resolution</label>
            <div className="grid grid-cols-2 gap-2">
              {Object.keys(resolutions).map((res) => (
                <button
                  key={res}
                  onClick={() => setResolution(res)}
                  className={`px-3 py-2 rounded-lg text-sm transition-all ${
                    resolution === res
                      ? 'bg-fuchsia-600 text-white'
                      : 'bg-white/10 text-gray-800 dark:text-gray-400 hover:bg-white/20'
                  }`}
                >
                  {res}×{res}
                </button>
              ))}
            </div>
          </div>

          {/* Steps */}
          <div>
            <label className="text-sm text-gray-800 dark:text-gray-400 block mb-3">
              Sampling Steps: {steps}
            </label>
            <input
              type="range"
              min="10"
              max="50"
              value={steps}
              onChange={(e) => setSteps(parseInt(e.target.value))}
              className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-gray-700 dark:text-gray-500 mt-1">
              <span>Fast (10)</span>
              <span>Quality (50)</span>
            </div>
          </div>

          {/* Guidance */}
          <div>
            <label className="text-sm text-gray-800 dark:text-gray-400 block mb-3">
              CFG Scale: {guidance}
            </label>
            <input
              type="range"
              min="1"
              max="15"
              step="0.5"
              value={guidance}
              onChange={(e) => setGuidance(parseFloat(e.target.value))}
              className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-gray-700 dark:text-gray-500 mt-1">
              <span>Natural (1)</span>
              <span>Strong (15)</span>
            </div>
          </div>
        </div>

        {/* Estimated Stats */}
        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white/5 rounded-lg p-3 text-center">
            <p className="text-lg font-bold text-fuchsia-600 dark:text-fuchsia-400">
              {resolutions[resolution].tokens}
            </p>
            <p className="text-xs text-gray-800 dark:text-gray-400">Image Tokens</p>
          </div>
          <div className="bg-white/5 rounded-lg p-3 text-center">
            <p className="text-lg font-bold text-purple-600 dark:text-purple-400">
              {resolutions[resolution].vram}
            </p>
            <p className="text-xs text-gray-800 dark:text-gray-400">Est. VRAM</p>
          </div>
          <div className="bg-white/5 rounded-lg p-3 text-center">
            <p className="text-lg font-bold text-blue-600 dark:text-blue-400">
              {steps * 2}
            </p>
            <p className="text-xs text-gray-800 dark:text-gray-400">NFE (w/ CFG)</p>
          </div>
          <div className="bg-white/5 rounded-lg p-3 text-center">
            <p className="text-lg font-bold text-green-400">
              ~{Math.round(parseFloat(resolutions[resolution].time) * steps / 28)}s
            </p>
            <p className="text-xs text-gray-800 dark:text-gray-400">Est. Time (A100)</p>
          </div>
        </div>
      </div>

      {/* Hardware Requirements */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
          <Cpu size={20} className="text-purple-600 dark:text-purple-400" />
          Hardware Requirements
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/20">
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Model Size</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Parameters</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Min VRAM (FP16)</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Recommended</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/10">
              <tr>
                <td className="py-3 px-4 text-fuchsia-600 dark:text-fuchsia-400">SD3-Medium</td>
                <td className="py-3 px-4">2B</td>
                <td className="py-3 px-4">8 GB</td>
                <td className="py-3 px-4">RTX 3080/4070</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-purple-600 dark:text-purple-400">SD3</td>
                <td className="py-3 px-4">3B</td>
                <td className="py-3 px-4">12 GB</td>
                <td className="py-3 px-4">RTX 4080/A6000</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-blue-600 dark:text-blue-400">SD3.5-Large</td>
                <td className="py-3 px-4">8B</td>
                <td className="py-3 px-4">24 GB</td>
                <td className="py-3 px-4">RTX 4090/A100</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="text-xs text-gray-700 dark:text-gray-500 mt-4">
          Note: Memory usage varies with resolution. Using FP8 quantization or model offloading can reduce requirements.
        </p>
      </div>

      {/* Optimization Tips */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-gradient-to-br from-green-900/30 to-green-800/20 rounded-xl p-5 border border-green-500/30">
          <h3 className="font-bold text-green-300 mb-2 flex items-center gap-2">
            <Zap size={18} /> Speed Tips
          </h3>
          <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
            <li>• Use fewer steps (20-28)</li>
            <li>• Lower resolution first</li>
            <li>• Enable xformers/flash-attn</li>
            <li>• Use torch.compile()</li>
          </ul>
        </div>

        <div className="bg-gradient-to-br from-purple-900/30 to-purple-800/20 rounded-xl p-5 border border-purple-500/30">
          <h3 className="font-bold text-purple-300 mb-2 flex items-center gap-2">
            <Clock size={18} /> Quality Tips
          </h3>
          <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
            <li>• Use 40-50 steps</li>
            <li>• CFG scale 4-7</li>
            <li>• Higher resolution</li>
            <li>• Detailed prompts</li>
          </ul>
        </div>

        <div className="bg-gradient-to-br from-blue-900/30 to-blue-800/20 rounded-xl p-5 border border-blue-500/30">
          <h3 className="font-bold text-blue-300 mb-2 flex items-center gap-2">
            <Cpu size={18} /> Memory Tips
          </h3>
          <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
            <li>• Use FP8 quantization</li>
            <li>• Sequential CPU offload</li>
            <li>• Attention slicing</li>
            <li>• VAE tiling</li>
          </ul>
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">Quick Start (diffusers)</h3>
        <div className="bg-black/50 rounded-lg p-4 font-mono text-sm overflow-x-auto">
          <pre className="text-gray-700 dark:text-gray-300">{`from diffusers import StableDiffusion3Pipeline
import torch

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
).to("cuda")

image = pipe(
    prompt="A cat wearing sunglasses on a beach",
    num_inference_steps=${steps},
    guidance_scale=${guidance},
    height=${resolution},
    width=${resolution},
).images[0]

image.save("output.png")`}</pre>
        </div>
      </div>
    </div>
  );
}
