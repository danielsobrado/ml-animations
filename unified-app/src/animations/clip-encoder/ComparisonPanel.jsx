import React, { useState } from 'react';
import { GitCompare, Check, X, Zap, Clock, Hash } from 'lucide-react';

function ComparisonPanel() {
  const [selectedFeature, setSelectedFeature] = useState(null);

  const comparisons = [
    {
      feature: 'Max Tokens',
      clip: '77',
      t5: '256+',
      winner: 't5',
      detail: 'T5 can handle much longer prompts, crucial for detailed descriptions.'
    },
    {
      feature: 'Architecture',
      clip: 'Decoder-style (causal)',
      t5: 'Encoder-only (bidirectional)',
      winner: 't5',
      detail: 'T5 uses bidirectional attention - each token sees all others, better for understanding.'
    },
    {
      feature: 'Training Data',
      clip: '400M image-text pairs',
      t5: 'C4 text corpus (750GB)',
      winner: 'both',
      detail: 'CLIP learned visual concepts; T5 learned deep language understanding. Complementary!'
    },
    {
      feature: 'Hidden Dimension',
      clip: '768 (L) / 1280 (G)',
      t5: '4096 (XXL)',
      winner: 't5',
      detail: 'T5-XXL has much larger representations, capturing more nuance.'
    },
    {
      feature: 'Parameters',
      clip: '~400M (L+G combined)',
      t5: '~4.7B (XXL)',
      winner: 'clip',
      detail: 'CLIP is much smaller/faster. T5-XXL is huge but optional in SD3.'
    },
    {
      feature: 'Speed',
      clip: 'Fast',
      t5: 'Slow',
      winner: 'clip',
      detail: 'CLIP encodes in ~10ms; T5-XXL can take 100ms+ depending on prompt length.'
    },
    {
      feature: 'Visual Alignment',
      clip: 'Excellent',
      t5: 'None (pure text)',
      winner: 'clip',
      detail: 'CLIP was trained to align text with images - T5 has no visual grounding.'
    },
  ];

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-blue-600 dark:text-blue-400 mb-2">CLIP vs T5: Why Both?</h2>
        <p className="text-gray-700 dark:text-gray-300 max-w-3xl mx-auto">
          SD3 uses <strong>both CLIP and T5</strong> text encoders. Each has different strengths - 
          together they provide rich, complementary text understanding.
        </p>
      </div>

      {/* Visual Comparison */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* CLIP Card */}
        <div className="bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-xl p-6 border border-blue-500/30">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-xl bg-blue-600 flex items-center justify-center">
              <span className="text-white font-bold text-lg">C</span>
            </div>
            <div>
              <h3 className="font-bold text-blue-600 dark:text-xl">CLIP</h3>
              <p className="text-sm text-gray-800 dark:text-gray-400">Contrastive Language-Image Pre-training</p>
            </div>
          </div>
          
          <div className="space-y-3 text-sm">
            <div className="flex items-center gap-2">
              <Check className="text-green-400" size={16} />
              <span className="text-gray-700 dark:text-gray-300">Visual-aligned embeddings</span>
            </div>
            <div className="flex items-center gap-2">
              <Check className="text-green-400" size={16} />
              <span className="text-gray-700 dark:text-gray-300">Fast inference</span>
            </div>
            <div className="flex items-center gap-2">
              <Check className="text-green-400" size={16} />
              <span className="text-gray-700 dark:text-gray-300">Good compositional understanding</span>
            </div>
            <div className="flex items-center gap-2">
              <X className="text-red-400" size={16} />
              <span className="text-gray-700 dark:text-gray-300">Limited to 77 tokens</span>
            </div>
            <div className="flex items-center gap-2">
              <X className="text-red-400" size={16} />
              <span className="text-gray-700 dark:text-gray-300">Causal attention only</span>
            </div>
          </div>

          <div className="mt-4 p-3 bg-black/30 rounded-lg">
            <div className="text-xs text-gray-800 dark:text-gray-400 mb-1">Best for:</div>
            <div className="text-blue-300">Visual concepts, objects, styles, artistic references</div>
          </div>
        </div>

        {/* T5 Card */}
        <div className="bg-gradient-to-br from-green-500/20 to-teal-500/20 rounded-xl p-6 border border-green-500/30">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-xl bg-green-600 flex items-center justify-center">
              <span className="text-white font-bold text-lg">T5</span>
            </div>
            <div>
              <h3 className="font-bold text-xl">T5-XXL</h3>
              <p className="text-sm text-gray-800 dark:text-gray-400">Text-to-Text Transfer Transformer</p>
            </div>
          </div>
          
          <div className="space-y-3 text-sm">
            <div className="flex items-center gap-2">
              <Check className="text-green-400" size={16} />
              <span className="text-gray-700 dark:text-gray-300">Deep language understanding</span>
            </div>
            <div className="flex items-center gap-2">
              <Check className="text-green-400" size={16} />
              <span className="text-gray-700 dark:text-gray-300">Bidirectional attention</span>
            </div>
            <div className="flex items-center gap-2">
              <Check className="text-green-400" size={16} />
              <span className="text-gray-700 dark:text-gray-300">Long context (256+ tokens)</span>
            </div>
            <div className="flex items-center gap-2">
              <X className="text-red-400" size={16} />
              <span className="text-gray-700 dark:text-gray-300">No visual training</span>
            </div>
            <div className="flex items-center gap-2">
              <X className="text-red-400" size={16} />
              <span className="text-gray-700 dark:text-gray-300">Very slow (4.7B params)</span>
            </div>
          </div>

          <div className="mt-4 p-3 bg-black/30 rounded-lg">
            <div className="text-xs text-gray-800 dark:text-gray-400 mb-1">Best for:</div>
            <div className="text-green-300">Complex descriptions, spatial relations, abstract concepts</div>
          </div>
        </div>
      </div>

      {/* Detailed Comparison Table */}
      <div className="bg-black/30 rounded-xl overflow-hidden">
        <div className="p-4 border-b border-white/10">
          <h3 className="font-semibold text-gray-700 dark:text-gray-300 flex items-center gap-2">
            <GitCompare size={20} />
            Feature Comparison
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/10 bg-black/20">
                <th className="text-left py-3 px-4 text-gray-800 dark:text-gray-400">Feature</th>
                <th className="text-left py-3 px-4 text-blue-600 dark:text-blue-400">CLIP</th>
                <th className="text-left py-3 px-4 text-green-400">T5</th>
                <th className="text-left py-3 px-4 text-gray-800 dark:text-gray-400">Better</th>
              </tr>
            </thead>
            <tbody>
              {comparisons.map((comp, i) => (
                <tr 
                  key={i} 
                  className={`border-b border-white/5 cursor-pointer transition-colors ${
                    selectedFeature === i ? 'bg-white/10' : 'hover:bg-white/5'
                  }`}
                  onClick={() => setSelectedFeature(selectedFeature === i ? null : i)}
                >
                  <td className="py-3 px-4 text-gray-700 dark:text-gray-300 font-medium">{comp.feature}</td>
                  <td className="py-3 px-4 text-gray-700 dark:text-gray-300">{comp.clip}</td>
                  <td className="py-3 px-4 text-gray-700 dark:text-gray-300">{comp.t5}</td>
                  <td className="py-3 px-4">
                    {comp.winner === 'clip' && <span className="text-blue-600 dark:text-blue-400 font-semibold">CLIP</span>}
                    {comp.winner === 't5' && <span className="text-green-400 font-semibold">T5</span>}
                    {comp.winner === 'both' && <span className="text-purple-600 dark:text-purple-400 font-semibold">Both!</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        {selectedFeature !== null && (
          <div className="p-4 bg-yellow-500/10 border-t border-yellow-500/30">
            <div className="text-sm">
              💡 {comparisons[selectedFeature].detail}
            </div>
          </div>
        )}
      </div>

      {/* Architecture Differences */}
      <div className="bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-green-500/10 rounded-xl p-6 border border-purple-500/30">
        <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-4">🔍 Key Architecture Difference: Attention</h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-black/30 rounded-lg p-4">
            <h4 className="text-blue-600 dark:text-blue-400 font-semibold mb-3">CLIP: Causal Attention</h4>
            <div className="font-mono text-xs bg-black/40 rounded p-3 mb-3">
              Token sees: [itself, previous tokens]<br/>
              <span className="text-gray-700 dark:text-gray-500">├─ cat: sees [BOS, a, cat]</span><br/>
              <span className="text-gray-700 dark:text-gray-500">├─ on: sees [BOS, a, cat, on]</span><br/>
              <span className="text-gray-700 dark:text-gray-500">└─ EOS: sees [ALL tokens]</span>
            </div>
            <p className="text-sm text-gray-800 dark:text-gray-400">
              Like GPT - designed for generation. [EOS] aggregates all info.
            </p>
          </div>

          <div className="bg-black/30 rounded-lg p-4">
            <h4 className="text-green-400 font-semibold mb-3">T5: Bidirectional Attention</h4>
            <div className="font-mono text-xs bg-black/40 rounded p-3 mb-3">
              Token sees: [ALL other tokens]<br/>
              <span className="text-gray-700 dark:text-gray-500">├─ cat: sees [a, cat, on, mat]</span><br/>
              <span className="text-gray-700 dark:text-gray-500">├─ on: sees [a, cat, on, mat]</span><br/>
              <span className="text-gray-700 dark:text-gray-500">└─ mat: sees [a, cat, on, mat]</span>
            </div>
            <p className="text-sm text-gray-800 dark:text-gray-400">
              Like BERT - better for understanding context and relationships.
            </p>
          </div>
        </div>
      </div>

      {/* Why SD3 Uses Both */}
      <div className="bg-black/40 rounded-xl p-6">
        <h3 className="font-semibold text-gray-700 dark:text-gray-300 mb-4">🎯 Why SD3 Uses Both Encoders</h3>
        
        <div className="space-y-4">
          <div className="flex items-start gap-4 p-4 bg-blue-500/10 rounded-lg">
            <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center shrink-0">
              <span className="text-white font-bold">1</span>
            </div>
            <div>
              <h4 className="text-blue-600 dark:text-blue-400 font-semibold">Visual Grounding (CLIP)</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                CLIP was trained on images - it knows what "sunset", "cyberpunk", "oil painting" 
                look like. T5 only knows these as words.
              </p>
            </div>
          </div>

          <div className="flex items-start gap-4 p-4 bg-green-500/10 rounded-lg">
            <div className="w-8 h-8 rounded-full bg-green-600 flex items-center justify-center shrink-0">
              <span className="text-white font-bold">2</span>
            </div>
            <div>
              <h4 className="text-green-400 font-semibold">Language Understanding (T5)</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                "A cat to the left of a dog" - T5 better understands spatial relations, 
                counting, negations ("no people"), and complex descriptions.
              </p>
            </div>
          </div>

          <div className="flex items-start gap-4 p-4 bg-purple-500/10 rounded-lg">
            <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center shrink-0">
              <span className="text-white font-bold">3</span>
            </div>
            <div>
              <h4 className="text-purple-600 dark:text-purple-400 font-semibold">Complementary Strengths</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                Different training → different representations. The model can leverage both 
                CLIP's visual knowledge and T5's linguistic depth.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* T5 is Optional Note */}
      <div className="bg-yellow-500/10 rounded-xl p-4 border border-yellow-500/30">
        <div className="flex items-center gap-2 text-yellow-400 font-semibold mb-2">
          <Zap size={18} />
          T5 is Optional!
        </div>
        <p className="text-sm text-gray-700 dark:text-gray-300">
          SD3 can run with just CLIP encoders for faster inference. T5-XXL adds ~8GB VRAM 
          and significant latency. For quick generations, many users skip it. For complex 
          prompts requiring precise understanding, T5 significantly improves results.
        </p>
      </div>
    </div>
  );
}

export default ComparisonPanel;
