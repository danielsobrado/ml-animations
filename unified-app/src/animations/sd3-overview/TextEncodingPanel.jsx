import React, { useState } from 'react';
import { Info, ArrowRight, Brain } from 'lucide-react';

export default function TextEncodingPanel() {
  const [selectedEncoder, setSelectedEncoder] = useState('clip-l');

  const encoders = {
    'clip-l': {
      name: 'CLIP-L/14',
      fullName: 'CLIP Large (OpenAI)',
      params: '400M',
      dim: 768,
      maxTokens: 77,
      color: 'from-green-500 to-emerald-500',
      role: 'Visual-language alignment, style understanding',
      description: 'Trained on 400M image-text pairs. Excels at understanding visual concepts and artistic styles.'
    },
    'clip-g': {
      name: 'CLIP-G/14',
      fullName: 'CLIP Giant (OpenCLIP)',
      params: '1.8B',
      dim: 1280,
      maxTokens: 77,
      color: 'from-blue-500 to-cyan-500',
      role: 'Enhanced visual understanding, fine details',
      description: 'Larger CLIP model with better detail understanding. Captures nuanced visual concepts.'
    },
    't5-xxl': {
      name: 'T5-XXL',
      fullName: 'Text-to-Text Transfer Transformer XXL',
      params: '4.7B',
      dim: 4096,
      maxTokens: 512,
      color: 'from-purple-500 to-violet-500',
      role: 'Deep text understanding, complex prompts',
      description: 'Pure text model that understands language deeply. Handles complex, detailed prompts better.'
    }
  };

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          Text Encoding: <span className="text-fuchsia-600 dark:text-fuchsia-400">Triple Encoder Power</span>
        </h2>
        <p className="text-gray-800 dark:text-gray-400">
          How SD3 understands your prompts using three specialized encoders
        </p>
      </div>

      {/* Why Three Encoders */}
      <div className="bg-gradient-to-r from-fuchsia-900/30 to-purple-900/30 rounded-2xl p-6 border border-fuchsia-500/30">
        <div className="flex items-start gap-4">
          <Info className="text-fuchsia-600 dark:text-fuchsia-400 mt-1" size={24} />
          <div>
            <h3 className="font-bold text-lg text-fuchsia-300 mb-2">Why Three Encoders?</h3>
            <p className="text-gray-700 dark:text-gray-300">
              Each encoder brings unique strengths. <strong className="text-green-400">CLIP models</strong> understand 
              how text relates to images (trained on image-text pairs). <strong className="text-purple-600 dark:text-purple-400">T5</strong> provides 
              deep language understanding (trained on massive text corpora). Together, they give SD3 
              unparalleled prompt comprehension.
            </p>
          </div>
        </div>
      </div>

      {/* Encoder Selector */}
      <div className="grid md:grid-cols-3 gap-4">
        {Object.entries(encoders).map(([key, enc]) => (
          <button
            key={key}
            onClick={() => setSelectedEncoder(key)}
            className={`p-4 rounded-xl border transition-all text-left ${
              selectedEncoder === key
                ? 'border-fuchsia-500 bg-fuchsia-500/20'
                : 'border-white/10 bg-black/30 hover:bg-white/5'
            }`}
          >
            <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-lg bg-gradient-to-r ${enc.color} mb-3`}>
              <Brain size={16} />
              <span className="font-bold text-sm">{enc.name}</span>
            </div>
            <p className="text-sm text-gray-800 dark:text-gray-400">{enc.role}</p>
          </button>
        ))}
      </div>

      {/* Selected Encoder Detail */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <div className="flex items-center gap-4 mb-6">
          <div className={`w-16 h-16 rounded-xl bg-gradient-to-br ${encoders[selectedEncoder].color} flex items-center justify-center`}>
            <Brain size={32} />
          </div>
          <div>
            <h3 className="text-xl font-bold">{encoders[selectedEncoder].fullName}</h3>
            <p className="text-gray-800 dark:text-gray-400">{encoders[selectedEncoder].description}</p>
          </div>
        </div>

        <div className="grid md:grid-cols-4 gap-4">
          <div className="bg-white/5 rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-fuchsia-600 dark:text-fuchsia-400">{encoders[selectedEncoder].params}</p>
            <p className="text-sm text-gray-800 dark:text-gray-400">Parameters</p>
          </div>
          <div className="bg-white/5 rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">{encoders[selectedEncoder].dim}</p>
            <p className="text-sm text-gray-800 dark:text-gray-400">Embedding Dim</p>
          </div>
          <div className="bg-white/5 rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">{encoders[selectedEncoder].maxTokens}</p>
            <p className="text-sm text-gray-800 dark:text-gray-400">Max Tokens</p>
          </div>
          <div className="bg-white/5 rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-green-400">
              {selectedEncoder.includes('clip') ? 'CLIP' : 'T2T'}
            </p>
            <p className="text-sm text-gray-800 dark:text-gray-400">Training Type</p>
          </div>
        </div>
      </div>

      {/* Embedding Flow */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-6 text-center">How Embeddings Flow to MMDiT</h3>
        
        <div className="flex flex-col md:flex-row items-center justify-center gap-6">
          {/* Input */}
          <div className="text-center">
            <div className="bg-white/10 rounded-lg px-4 py-3 mb-2">
              <p className="text-sm font-mono">"a cat wearing sunglasses"</p>
            </div>
            <p className="text-xs text-gray-800 dark:text-gray-400">Input Prompt</p>
          </div>

          <ArrowRight className="text-gray-700 dark:text-gray-500 rotate-90 md:rotate-0" />

          {/* Tokenization */}
          <div className="text-center">
            <div className="bg-white/10 rounded-lg px-4 py-3 mb-2">
              <div className="flex gap-1 flex-wrap justify-center">
                {['a', 'cat', 'wearing', 'sun', '##glasses'].map((tok, i) => (
                  <span key={i} className="bg-fuchsia-500/30 px-2 py-1 rounded text-xs">{tok}</span>
                ))}
              </div>
            </div>
            <p className="text-xs text-gray-800 dark:text-gray-400">Tokenized</p>
          </div>

          <ArrowRight className="text-gray-700 dark:text-gray-500 rotate-90 md:rotate-0" />

          {/* Encoders */}
          <div className="flex flex-col gap-2">
            <div className="bg-gradient-to-r from-green-500/30 to-emerald-500/30 rounded-lg px-3 py-2 text-xs">
              CLIP-L → [77 × 768]
            </div>
            <div className="bg-gradient-to-r from-blue-500/30 to-cyan-500/30 rounded-lg px-3 py-2 text-xs">
              CLIP-G → [77 × 1280]
            </div>
            <div className="bg-gradient-to-r from-purple-500/30 to-violet-500/30 rounded-lg px-3 py-2 text-xs">
              T5-XXL → [512 × 4096]
            </div>
          </div>

          <ArrowRight className="text-gray-700 dark:text-gray-500 rotate-90 md:rotate-0" />

          {/* Combined */}
          <div className="text-center">
            <div className="bg-gradient-to-r from-fuchsia-500/30 to-purple-500/30 rounded-lg px-4 py-3 mb-2">
              <p className="text-sm font-mono">Concatenated</p>
              <p className="text-xs text-gray-800 dark:text-gray-400">+ Pooled vectors</p>
            </div>
            <p className="text-xs text-gray-800 dark:text-gray-400">→ MMDiT</p>
          </div>
        </div>
      </div>

      {/* CLIP vs T5 Comparison */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">CLIP vs T5: Complementary Strengths</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/20">
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Aspect</th>
                <th className="py-3 px-4 text-left text-green-400">CLIP (L + G)</th>
                <th className="py-3 px-4 text-left text-purple-600 dark:text-purple-400">T5-XXL</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/10">
              <tr>
                <td className="py-3 px-4 text-gray-700 dark:text-gray-300">Training</td>
                <td className="py-3 px-4">Image-text pairs</td>
                <td className="py-3 px-4">Text-only (C4 corpus)</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-gray-700 dark:text-gray-300">Strength</td>
                <td className="py-3 px-4">Visual concepts, styles</td>
                <td className="py-3 px-4">Language understanding</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-gray-700 dark:text-gray-300">Understands</td>
                <td className="py-3 px-4">"cyberpunk style", "oil painting"</td>
                <td className="py-3 px-4">"the cat to the left of the dog"</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-gray-700 dark:text-gray-300">Context Length</td>
                <td className="py-3 px-4">77 tokens</td>
                <td className="py-3 px-4">512 tokens</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-gray-700 dark:text-gray-300">Used For</td>
                <td className="py-3 px-4">Cross-attention + pooled conditioning</td>
                <td className="py-3 px-4">Cross-attention only</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Pooled vs Sequence */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-gradient-to-br from-green-900/30 to-green-800/20 rounded-xl p-5 border border-green-500/30">
          <h3 className="font-bold text-green-300 mb-2">Pooled Embeddings (CLIP)</h3>
          <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
            A single vector representing the whole prompt. Used for global conditioning 
            through AdaLN layers in MMDiT.
          </p>
          <div className="bg-black/30 rounded-lg p-3 font-mono text-xs">
            pooled = CLIP.encode(text).pooler_output  # [1, 768]
          </div>
        </div>

        <div className="bg-gradient-to-br from-purple-900/30 to-purple-800/20 rounded-xl p-5 border border-purple-500/30">
          <h3 className="font-bold text-purple-300 mb-2">Sequence Embeddings (All)</h3>
          <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
            Per-token embeddings preserving word-level info. Used for cross-attention 
            between image and text tokens.
          </p>
          <div className="bg-black/30 rounded-lg p-3 font-mono text-xs">
            seq = T5.encode(text).last_hidden_state  # [1, N, 4096]
          </div>
        </div>
      </div>
    </div>
  );
}
