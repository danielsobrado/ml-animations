import React, { useState, useRef, useEffect } from 'react';
import { Type, Hash, ArrowRight, Play, RotateCcw, Scissors } from 'lucide-react';
import gsap from 'gsap';

function TokenizationPanel() {
  const [inputText, setInputText] = useState('A majestic lion standing proudly on a cliff');
  const [isPlaying, setIsPlaying] = useState(false);
  const animationRef = useRef(null);
  const timelineRef = useRef(null);

  // Simulate SentencePiece tokenization
  const tokenizeText = (text) => {
    // SentencePiece uses underscore for word boundaries
    const words = text.split(/\s+/);
    const tokens = [];
    
    words.forEach((word, idx) => {
      // Add space marker (▁) before each word except first
      const prefix = idx > 0 ? '▁' : '';
      
      // Simulate subword tokenization
      if (word.length > 6) {
        // Split longer words
        tokens.push({ text: prefix + word.slice(0, 4), type: 'subword' });
        tokens.push({ text: word.slice(4), type: 'subword' });
      } else {
        tokens.push({ text: prefix + word.toLowerCase(), type: 'word' });
      }
    });
    
    // Add EOS token
    tokens.push({ text: '</s>', type: 'special' });
    
    return tokens;
  };

  const tokens = tokenizeText(inputText);
  const displayTokens = tokens.slice(0, 15);

  useEffect(() => {
    if (!animationRef.current) return;

    const ctx = gsap.context(() => {
      gsap.set('.sp-token', { opacity: 0, y: 20 });
      gsap.set('.sp-id', { opacity: 0, scale: 0.5 });
    }, animationRef);

    return () => ctx.revert();
  }, [inputText]);

  const playAnimation = () => {
    if (!animationRef.current) return;
    setIsPlaying(true);

    const ctx = gsap.context(() => {
      timelineRef.current = gsap.timeline({
        onComplete: () => setIsPlaying(false)
      });

      timelineRef.current.to('.sp-token', {
        opacity: 1,
        y: 0,
        stagger: 0.08,
        duration: 0.3,
        ease: 'back.out(1.7)'
      });

      timelineRef.current.to('.sp-id', {
        opacity: 1,
        scale: 1,
        stagger: 0.08,
        duration: 0.2
      }, '-=0.4');

    }, animationRef);
  };

  const resetAnimation = () => {
    if (timelineRef.current) timelineRef.current.kill();
    
    const ctx = gsap.context(() => {
      gsap.set('.sp-token', { opacity: 0, y: 20 });
      gsap.set('.sp-id', { opacity: 0, scale: 0.5 });
    }, animationRef);

    setIsPlaying(false);
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-emerald-400 mb-2">SentencePiece Tokenization</h2>
        <p className="text-gray-300 max-w-3xl mx-auto">
          T5 uses <strong>SentencePiece</strong> with a vocabulary of ~32,000 tokens.
          Unlike CLIP's BPE, SentencePiece treats text as raw bytes and learns subword units directly.
        </p>
      </div>

      {/* Input */}
      <div className="bg-black/30 rounded-xl p-6">
        <label className="block text-sm text-gray-400 mb-2">Enter a prompt to tokenize:</label>
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          className="w-full px-4 py-3 bg-black/50 border border-emerald-500/30 rounded-lg text-white focus:outline-none focus:border-emerald-500"
          placeholder="Type your prompt here..."
        />
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-4">
        <button
          onClick={playAnimation}
          disabled={isPlaying}
          className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 rounded-lg transition-colors"
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

      {/* Tokenization Visualization */}
      <div ref={animationRef} className="space-y-6">
        {/* SentencePiece Process */}
        <div className="bg-gradient-to-r from-emerald-500/10 to-teal-500/10 rounded-xl p-6 border border-emerald-500/30">
          <h3 className="font-semibold text-emerald-400 mb-4 flex items-center gap-2">
            <Scissors size={20} />
            SentencePiece Tokens
          </h3>
          
          <div className="flex items-center gap-4 mb-4">
            <div className="bg-black/40 rounded-lg px-4 py-3 flex-1">
              <span className="text-sm text-gray-400">Input:</span>
              <div className="text-white font-mono mt-1">"{inputText}"</div>
            </div>
            <ArrowRight className="text-emerald-400 shrink-0" size={24} />
            <div className="bg-black/40 rounded-lg px-4 py-3">
              <span className="text-sm text-gray-400">Tokens:</span>
              <div className="text-emerald-400 font-mono mt-1">{tokens.length}</div>
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            {displayTokens.map((token, i) => (
              <div
                key={i}
                className={`sp-token px-3 py-2 rounded-lg text-sm font-mono ${
                  token.type === 'special'
                    ? 'bg-yellow-500/30 text-yellow-300 border border-yellow-500/50'
                    : token.type === 'subword'
                    ? 'bg-teal-500/30 text-teal-300 border border-teal-500/50'
                    : 'bg-emerald-500/30 text-emerald-300 border border-emerald-500/50'
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

        {/* Token IDs */}
        <div className="bg-gradient-to-r from-teal-500/10 to-cyan-500/10 rounded-xl p-6 border border-teal-500/30">
          <h3 className="font-semibold text-teal-400 mb-4 flex items-center gap-2">
            <Hash size={20} />
            Token IDs
          </h3>

          <div className="flex flex-wrap gap-2">
            {displayTokens.map((token, i) => (
              <div key={i} className="sp-id flex flex-col items-center gap-1">
                <div className={`px-3 py-1 rounded text-xs font-mono ${
                  token.type === 'special'
                    ? 'bg-yellow-500/30 text-yellow-300'
                    : token.type === 'subword'
                    ? 'bg-teal-500/30 text-teal-300'
                    : 'bg-emerald-500/30 text-emerald-300'
                }`}>
                  {token.text}
                </div>
                <div className="text-xs text-gray-400">↓</div>
                <div className="px-2 py-1 bg-black/40 rounded text-xs font-mono text-yellow-400">
                  {1000 + i * 127 + Math.floor(Math.random() * 100)}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* SentencePiece vs BPE */}
      <div className="bg-black/40 rounded-xl p-6">
        <h3 className="font-semibold text-gray-300 mb-4">📊 SentencePiece vs BPE (CLIP)</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left py-2 px-3 text-gray-400">Feature</th>
                <th className="text-left py-2 px-3 text-emerald-400">SentencePiece (T5)</th>
                <th className="text-left py-2 px-3 text-blue-400">BPE (CLIP)</th>
              </tr>
            </thead>
            <tbody className="text-gray-300">
              <tr className="border-b border-white/5">
                <td className="py-2 px-3">Vocabulary</td>
                <td className="py-2 px-3">32,000 tokens</td>
                <td className="py-2 px-3">49,000 tokens</td>
              </tr>
              <tr className="border-b border-white/5">
                <td className="py-2 px-3">Space handling</td>
                <td className="py-2 px-3">▁ (underscore) prefix</td>
                <td className="py-2 px-3">Implicit spaces</td>
              </tr>
              <tr className="border-b border-white/5">
                <td className="py-2 px-3">Training</td>
                <td className="py-2 px-3">Unigram language model</td>
                <td className="py-2 px-3">Byte pair merges</td>
              </tr>
              <tr className="border-b border-white/5">
                <td className="py-2 px-3">Special tokens</td>
                <td className="py-2 px-3">&lt;/s&gt;, &lt;pad&gt;, &lt;unk&gt;</td>
                <td className="py-2 px-3">[BOS], [EOS], [PAD]</td>
              </tr>
              <tr className="border-b border-white/5">
                <td className="py-2 px-3">Pre-tokenization</td>
                <td className="py-2 px-3">None (raw bytes)</td>
                <td className="py-2 px-3">Whitespace split</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* The Underscore Marker */}
      <div className="bg-yellow-500/10 rounded-xl p-6 border border-yellow-500/30">
        <h3 className="font-semibold text-yellow-400 mb-3">📝 The ▁ (Underscore) Marker</h3>
        <div className="text-gray-300 text-sm space-y-2">
          <p>
            SentencePiece uses <code className="text-emerald-400">▁</code> (U+2581, lower one eighth block) 
            to mark word boundaries:
          </p>
          <div className="bg-black/40 rounded p-4 font-mono text-sm">
            <div className="text-gray-400"># Input: "hello world"</div>
            <div className="text-emerald-400"># Tokens: ["▁hello", "▁world"]</div>
            <br />
            <div className="text-gray-400"># Input: "unbelievable"</div>
            <div className="text-emerald-400"># Tokens: ["▁un", "believ", "able"]</div>
          </div>
          <p className="mt-2">
            This allows reconstructing the original text perfectly, including spaces, 
            which is important for language models.
          </p>
        </div>
      </div>

      {/* No Padding Needed Note */}
      <div className="bg-emerald-500/10 rounded-xl p-4 border border-emerald-500/30">
        <div className="font-semibold text-emerald-400 mb-2">💡 Dynamic Length in SD3</div>
        <p className="text-sm text-gray-300">
          Unlike CLIP's fixed 77 tokens, T5 in SD3 can handle variable-length sequences.
          The embeddings are projected and padded as needed before entering the DiT.
          This flexibility is why T5 excels at long, detailed prompts.
        </p>
      </div>
    </div>
  );
}

export default TokenizationPanel;
