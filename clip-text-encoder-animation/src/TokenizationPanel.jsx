import React, { useState, useRef, useEffect } from 'react';
import { Type, Hash, ArrowRight, Play, Pause, RotateCcw } from 'lucide-react';
import gsap from 'gsap';

function TokenizationPanel() {
  const [inputText, setInputText] = useState('A cute cat sitting on a mat');
  const [isPlaying, setIsPlaying] = useState(false);
  const animationRef = useRef(null);
  const timelineRef = useRef(null);

  // Simplified BPE tokenization example
  const tokenizeText = (text) => {
    // Add BOS and EOS tokens
    const bos = { text: '[BOS]', id: 49406, type: 'special' };
    const eos = { text: '[EOS]', id: 49407, type: 'special' };
    
    // Simple word-level tokenization for demo (real BPE is more complex)
    const words = text.toLowerCase().split(/\s+/).filter(w => w);
    const tokens = words.map((word, i) => ({
      text: word,
      id: 1000 + i * 100 + Math.floor(Math.random() * 100), // Fake IDs
      type: 'word'
    }));
    
    // Padding to 77 tokens
    const padding = Array(77 - tokens.length - 2).fill({ text: '[PAD]', id: 0, type: 'padding' });
    
    return [bos, ...tokens, eos, ...padding.slice(0, Math.max(0, 77 - tokens.length - 2))];
  };

  const tokens = tokenizeText(inputText);
  const displayTokens = tokens.slice(0, 15); // Show first 15 for display

  useEffect(() => {
    if (!animationRef.current) return;

    const ctx = gsap.context(() => {
      gsap.set('.token-item', { opacity: 0, y: 20 });
      gsap.set('.id-item', { opacity: 0, scale: 0.5 });
    }, animationRef);

    return () => ctx.revert();
  }, [inputText]);

  const playAnimation = () => {
    if (!animationRef.current) return;

    const ctx = gsap.context(() => {
      timelineRef.current = gsap.timeline();
      
      // Animate tokens appearing
      timelineRef.current.to('.token-item', {
        opacity: 1,
        y: 0,
        stagger: 0.1,
        duration: 0.3,
        ease: 'back.out(1.7)'
      });

      // Animate IDs appearing
      timelineRef.current.to('.id-item', {
        opacity: 1,
        scale: 1,
        stagger: 0.1,
        duration: 0.2
      }, '-=0.5');

      timelineRef.current.eventCallback('onComplete', () => setIsPlaying(false));
    }, animationRef);

    setIsPlaying(true);
  };

  const resetAnimation = () => {
    if (timelineRef.current) {
      timelineRef.current.kill();
    }
    
    const ctx = gsap.context(() => {
      gsap.set('.token-item', { opacity: 0, y: 20 });
      gsap.set('.id-item', { opacity: 0, scale: 0.5 });
    }, animationRef);

    setIsPlaying(false);
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-blue-400 mb-2">CLIP Tokenization (BPE)</h2>
        <p className="text-gray-300 max-w-3xl mx-auto">
          CLIP uses <strong>Byte Pair Encoding (BPE)</strong> with a vocabulary of ~49,000 tokens.
          Text is split into subword units, converted to token IDs, and padded to 77 tokens.
        </p>
      </div>

      {/* Input */}
      <div className="bg-black/30 rounded-xl p-6">
        <label className="block text-sm text-gray-400 mb-2">Enter a prompt to tokenize:</label>
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          className="w-full px-4 py-3 bg-black/50 border border-blue-500/30 rounded-lg text-white focus:outline-none focus:border-blue-500"
          placeholder="Type your prompt here..."
        />
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-4">
        <button
          onClick={playAnimation}
          disabled={isPlaying}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 rounded-lg transition-colors"
        >
          <Play size={18} />
          Animate Tokenization
        </button>
        <button
          onClick={resetAnimation}
          className="flex items-center gap-2 px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <RotateCcw size={18} />
          Reset
        </button>
      </div>

      {/* Tokenization Process */}
      <div ref={animationRef} className="space-y-6">
        {/* Step 1: Text to Tokens */}
        <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-xl p-6 border border-blue-500/30">
          <h3 className="font-semibold text-blue-400 mb-4 flex items-center gap-2">
            <Type size={20} />
            Step 1: BPE Tokenization
          </h3>
          
          <div className="flex items-center gap-4 mb-4">
            <div className="bg-black/40 rounded-lg px-4 py-3 flex-1">
              <span className="text-sm text-gray-400">Input:</span>
              <div className="text-white font-mono mt-1">"{inputText}"</div>
            </div>
            <ArrowRight className="text-blue-400 shrink-0" size={24} />
            <div className="bg-black/40 rounded-lg px-4 py-3">
              <span className="text-sm text-gray-400">Tokens:</span>
              <div className="text-blue-400 font-mono mt-1">{tokens.filter(t => t.type !== 'padding').length}</div>
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            {displayTokens.map((token, i) => (
              <div
                key={i}
                className={`token-item px-3 py-2 rounded-lg text-sm font-mono ${
                  token.type === 'special' 
                    ? 'bg-purple-500/30 text-purple-300 border border-purple-500/50' 
                    : token.type === 'padding'
                    ? 'bg-gray-500/30 text-gray-500 border border-gray-500/30'
                    : 'bg-blue-500/30 text-blue-300 border border-blue-500/50'
                }`}
              >
                {token.text}
              </div>
            ))}
            {tokens.length > 15 && (
              <div className="px-3 py-2 text-gray-500">...+{tokens.length - 15} more</div>
            )}
          </div>
        </div>

        {/* Step 2: Token IDs */}
        <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-xl p-6 border border-purple-500/30">
          <h3 className="font-semibold text-purple-400 mb-4 flex items-center gap-2">
            <Hash size={20} />
            Step 2: Token IDs
          </h3>
          
          <p className="text-gray-300 text-sm mb-4">
            Each token maps to a unique integer ID in CLIP's vocabulary (0-49407):
          </p>

          <div className="flex flex-wrap gap-2">
            {displayTokens.map((token, i) => (
              <div key={i} className="id-item flex flex-col items-center gap-1">
                <div className={`px-3 py-1 rounded text-xs font-mono ${
                  token.type === 'special' 
                    ? 'bg-purple-500/30 text-purple-300' 
                    : token.type === 'padding'
                    ? 'bg-gray-500/30 text-gray-500'
                    : 'bg-blue-500/30 text-blue-300'
                }`}>
                  {token.text}
                </div>
                <div className="text-xs text-gray-400">↓</div>
                <div className="px-2 py-1 bg-black/40 rounded text-xs font-mono text-yellow-400">
                  {token.id}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Step 3: Final Shape */}
        <div className="bg-gradient-to-r from-pink-500/10 to-orange-500/10 rounded-xl p-6 border border-pink-500/30">
          <h3 className="font-semibold text-pink-400 mb-4">Step 3: Fixed Shape Output</h3>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-black/40 rounded-lg p-4">
              <div className="text-sm text-gray-400 mb-2">Input Shape</div>
              <div className="font-mono text-white">
                <span className="text-blue-400">[batch_size, 77]</span>
              </div>
              <div className="text-xs text-gray-500 mt-2">77 token IDs (padded)</div>
            </div>
            <div className="bg-black/40 rounded-lg p-4">
              <div className="text-sm text-gray-400 mb-2">After Embedding</div>
              <div className="font-mono text-white">
                <span className="text-purple-400">[batch_size, 77, 768]</span>
              </div>
              <div className="text-xs text-gray-500 mt-2">CLIP-L: 768-dim vectors</div>
            </div>
          </div>
        </div>
      </div>

      {/* BPE Explanation */}
      <div className="bg-yellow-500/10 rounded-xl p-6 border border-yellow-500/30">
        <h3 className="font-semibold text-yellow-400 mb-3">📚 How BPE Works</h3>
        <div className="text-gray-300 space-y-2 text-sm">
          <p>
            <strong>Byte Pair Encoding</strong> builds a vocabulary by iteratively merging the most frequent character pairs:
          </p>
          <ol className="list-decimal list-inside space-y-1 ml-2">
            <li>Start with individual characters: <code className="text-blue-400">c a t</code></li>
            <li>Find frequent pairs: <code className="text-blue-400">"at"</code> appears often → merge</li>
            <li>Continue merging: <code className="text-blue-400">"cat"</code> → single token</li>
            <li>Rare words split into subwords: <code className="text-purple-400">"unhappiness"</code> → <code className="text-purple-400">un + happiness</code></li>
          </ol>
          <p className="mt-2">
            This balances vocabulary size (~49K) with coverage of rare words.
          </p>
        </div>
      </div>

      {/* Special Tokens */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-purple-500/10 rounded-lg p-4 border border-purple-500/30">
          <div className="font-mono text-purple-400 text-lg mb-2">[BOS]</div>
          <div className="text-sm text-gray-300">
            <strong>Beginning of Sequence</strong> - ID 49406. Marks the start of the text.
          </div>
        </div>
        <div className="bg-purple-500/10 rounded-lg p-4 border border-purple-500/30">
          <div className="font-mono text-purple-400 text-lg mb-2">[EOS]</div>
          <div className="text-sm text-gray-300">
            <strong>End of Sequence</strong> - ID 49407. Its hidden state becomes the pooled embedding.
          </div>
        </div>
        <div className="bg-gray-500/10 rounded-lg p-4 border border-gray-500/30">
          <div className="font-mono text-gray-400 text-lg mb-2">[PAD]</div>
          <div className="text-sm text-gray-300">
            <strong>Padding</strong> - ID 0. Fills remaining positions to reach 77 tokens.
          </div>
        </div>
      </div>
    </div>
  );
}

export default TokenizationPanel;
