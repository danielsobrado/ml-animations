import React, { useState, useEffect, useRef } from 'react';
import { ArrowRight, Zap, Hash, Type, Layers } from 'lucide-react';
import gsap from 'gsap';

function ComparisonPanel() {
  const [activeDemo, setActiveDemo] = useState('clip');
  const containerRef = useRef(null);

  const demoText = "a beautiful sunset over mountains";

  const tokenizations = {
    clip: {
      name: 'CLIP',
      vocabSize: '49,408',
      maxLength: '77',
      special: ['<|startoftext|>', '<|endoftext|>'],
      method: 'Byte-Pair Encoding (BPE)',
      wordMarker: '</w>',
      tokens: ['<|startoftext|>', 'a</w>', 'beautiful</w>', 'sunset</w>', 'over</w>', 'mountains</w>', '<|endoftext|>'],
      ids: [49406, 320, 4044, 7270, 962, 5765, 49407],
      color: 'blue',
      features: [
        'Trained on web image-text pairs',
        'Optimized for visual concepts',
        'Fixed 77 token context',
        'Single encoder output',
        'Good at object/style descriptions',
        'Lowercase processing'
      ]
    },
    t5: {
      name: 'T5 XXL',
      vocabSize: '32,100',
      maxLength: '77-256',
      special: ['<pad>', '</s>'],
      method: 'SentencePiece',
      wordMarker: '▁ (underscore prefix)',
      tokens: ['▁a', '▁beautiful', '▁sun', 'set', '▁over', '▁mountain', 's'],
      ids: [3, 1804, 1135, 2244, 147, 8071, 7],
      color: 'green',
      features: [
        'Trained on C4 text corpus',
        'Better at complex descriptions',
        'Variable context length',
        '4096 tokens max (SD3)',
        'Full NLP understanding',
        'Case sensitive'
      ]
    }
  };

  const active = tokenizations[activeDemo];

  useEffect(() => {
    if (containerRef.current) {
      gsap.fromTo(containerRef.current.querySelectorAll('.token-item'),
        { opacity: 0, y: 10 },
        { opacity: 1, y: 0, duration: 0.3, stagger: 0.05 }
      );
    }
  }, [activeDemo]);

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-orange-600 dark:text-orange-400 mb-2">CLIP vs T5 Tokenization</h2>
        <p className="text-gray-700 dark:text-gray-300 max-w-3xl mx-auto">
          SD3 uses both CLIP and T5 text encoders. Each tokenizes text differently,
          creating complementary text representations.
        </p>
      </div>

      {/* Toggle */}
      <div className="flex justify-center gap-4">
        <button
          onClick={() => setActiveDemo('clip')}
          className={`px-6 py-3 rounded-xl font-semibold transition-all ${
            activeDemo === 'clip'
              ? 'bg-blue-500 text-white'
              : 'bg-black/30 text-gray-800 dark:text-gray-400 hover:text-white'
          }`}
        >
          CLIP Tokenizer
        </button>
        <button
          onClick={() => setActiveDemo('t5')}
          className={`px-6 py-3 rounded-xl font-semibold transition-all ${
            activeDemo === 't5'
              ? 'bg-green-500 text-white'
              : 'bg-black/30 text-gray-800 dark:text-gray-400 hover:text-white'
          }`}
        >
          T5 Tokenizer
        </button>
      </div>

      {/* Demo Input */}
      <div className="bg-black/40 rounded-xl p-4 text-center">
        <div className="text-gray-800 dark:text-sm mb-1">Input Text</div>
        <div className="text-xl text-white font-mono">"{demoText}"</div>
      </div>

      {/* Tokenization Result */}
      <div ref={containerRef} className={`bg-gradient-to-r ${
        activeDemo === 'clip' ? 'from-blue-500/10 to-cyan-500/10 border-blue-500/30' : 'from-green-500/10 to-emerald-500/10 border-green-500/30'
      } rounded-xl p-6 border`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className={`font-semibold ${activeDemo === 'clip' ? 'text-blue-400' : 'text-green-400'}`}>
            {active.name} Tokenization
          </h3>
          <span className="text-gray-800 dark:text-sm">{active.tokens.length} tokens</span>
        </div>
        
        <div className="flex flex-wrap gap-2 mb-4">
          {active.tokens.map((token, i) => (
            <div key={i} className={`token-item flex flex-col items-center px-3 py-2 rounded-lg ${
              activeDemo === 'clip' ? 'bg-blue-500/20' : 'bg-green-500/20'
            }`}>
              <code className={`text-sm ${activeDemo === 'clip' ? 'text-blue-300' : 'text-green-300'}`}>
                {token}
              </code>
              <span className="text-xs text-gray-700 dark:text-gray-500 mt-1">{active.ids[i]}</span>
            </div>
          ))}
        </div>

        <div className="text-xs text-gray-800 dark:text-gray-400 bg-black/30 rounded p-2">
          Word boundary marker: <code className={activeDemo === 'clip' ? 'text-blue-400' : 'text-green-400'}>
            {active.wordMarker}
          </code>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="bg-black/40 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-white/10">
              <th className="text-left py-3 px-4 text-gray-800 dark:text-gray-400">Feature</th>
              <th className="text-left py-3 px-4 text-blue-600 dark:text-blue-400">CLIP</th>
              <th className="text-left py-3 px-4 text-green-400">T5</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-white/5">
              <td className="py-3 px-4 text-gray-700 dark:text-gray-300">Vocab Size</td>
              <td className="py-3 px-4 text-blue-300 font-mono">49,408</td>
              <td className="py-3 px-4 text-green-300 font-mono">32,100</td>
            </tr>
            <tr className="border-b border-white/5">
              <td className="py-3 px-4 text-gray-700 dark:text-gray-300">Max Length</td>
              <td className="py-3 px-4 text-blue-300 font-mono">77</td>
              <td className="py-3 px-4 text-green-300 font-mono">77-256</td>
            </tr>
            <tr className="border-b border-white/5">
              <td className="py-3 px-4 text-gray-700 dark:text-gray-300">Method</td>
              <td className="py-3 px-4 text-blue-300">BPE</td>
              <td className="py-3 px-4 text-green-300">SentencePiece</td>
            </tr>
            <tr className="border-b border-white/5">
              <td className="py-3 px-4 text-gray-700 dark:text-gray-300">Start Token</td>
              <td className="py-3 px-4"><code className="text-blue-600 dark:text-blue-400 bg-blue-500/20 px-2 py-1 rounded">&lt;|startoftext|&gt;</code></td>
              <td className="py-3 px-4"><span className="text-gray-700 dark:text-gray-500">None (implicit)</span></td>
            </tr>
            <tr className="border-b border-white/5">
              <td className="py-3 px-4 text-gray-700 dark:text-gray-300">End Token</td>
              <td className="py-3 px-4"><code className="text-blue-600 dark:text-blue-400 bg-blue-500/20 px-2 py-1 rounded">&lt;|endoftext|&gt;</code></td>
              <td className="py-3 px-4"><code className="text-green-400 bg-green-500/20 px-2 py-1 rounded">&lt;/s&gt;</code></td>
            </tr>
            <tr className="border-b border-white/5">
              <td className="py-3 px-4 text-gray-700 dark:text-gray-300">Case</td>
              <td className="py-3 px-4 text-blue-300">lowercase</td>
              <td className="py-3 px-4 text-green-300">preserved</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Feature Comparison */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-blue-500/10 rounded-xl p-5 border border-blue-500/30">
          <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-4 flex items-center gap-2">
            <Zap size={18} /> CLIP Strengths
          </h3>
          <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
            {tokenizations.clip.features.map((f, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="text-blue-600 dark:text-blue-400 mt-0.5">•</span>
                {f}
              </li>
            ))}
          </ul>
        </div>
        <div className="bg-green-500/10 rounded-xl p-5 border border-green-500/30">
          <h3 className="font-semibold text-green-400 mb-4 flex items-center gap-2">
            <Zap size={18} /> T5 Strengths
          </h3>
          <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
            {tokenizations.t5.features.map((f, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="text-green-400 mt-0.5">•</span>
                {f}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Why Both? */}
      <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-xl p-6 border border-purple-500/30">
        <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-3">Why Use Both?</h3>
        <div className="grid md:grid-cols-3 gap-4 text-sm">
          <div className="bg-black/30 rounded-lg p-4">
            <div className="text-blue-600 dark:text-blue-400 font-medium mb-1">CLIP</div>
            <div className="text-gray-700 dark:text-gray-300">
              Strong visual grounding from image-text pairs. 
              Knows "cyberpunk" = neon, rain, tech aesthetic.
            </div>
          </div>
          <div className="bg-black/30 rounded-lg p-4">
            <div className="text-green-400 font-medium mb-1">T5</div>
            <div className="text-gray-700 dark:text-gray-300">
              Deep language understanding from massive text corpus.
              Parses complex prompts with spatial relationships.
            </div>
          </div>
          <div className="bg-black/30 rounded-lg p-4">
            <div className="text-purple-600 dark:text-purple-400 font-medium mb-1">Together</div>
            <div className="text-gray-700 dark:text-gray-300">
              SD3 concatenates both embeddings. Gets visual vocabulary 
              + linguistic sophistication.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ComparisonPanel;
