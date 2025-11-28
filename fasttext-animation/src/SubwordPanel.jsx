import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Puzzle, ChevronRight, Info } from 'lucide-react';

function SubwordPanel() {
  const [word, setWord] = useState('playing');
  const [ngramSize, setNgramSize] = useState(3);

  const generateNgrams = (word, n) => {
    const padded = `<${word}>`;
    const ngrams = [];
    for (let i = 0; i <= padded.length - n; i++) {
      ngrams.push(padded.substring(i, i + n));
    }
    return ngrams;
  };

  const ngrams = generateNgrams(word, ngramSize);

  const exampleWords = ['playing', 'played', 'player', 'playful', 'replay'];

  return (
    <div className="space-y-6">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold text-white mb-2">Subword Embeddings</h2>
        <p className="text-purple-200/70">
          FastText represents words as the sum of their character n-gram vectors
        </p>
      </div>

      {/* Interactive Input */}
      <div className="bg-white/5 rounded-xl p-6">
        <div className="flex flex-col md:flex-row gap-4 items-center justify-center mb-6">
          <div>
            <label className="text-purple-300 text-sm mb-1 block">Enter a word:</label>
            <input
              type="text"
              value={word}
              onChange={(e) => setWord(e.target.value.toLowerCase().replace(/[^a-z]/g, ''))}
              className="bg-white/10 border border-purple-400/30 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-purple-400"
              placeholder="Enter word"
            />
          </div>
          <div>
            <label className="text-purple-300 text-sm mb-1 block">N-gram size:</label>
            <div className="flex gap-2">
              {[2, 3, 4, 5].map((n) => (
                <button
                  key={n}
                  onClick={() => setNgramSize(n)}
                  className={`w-10 h-10 rounded-lg font-semibold transition-all ${
                    ngramSize === n
                      ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
                      : 'bg-white/10 text-purple-300 hover:bg-white/20'
                  }`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Visual Breakdown */}
        <div className="text-center">
          <div className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400 mb-4">
            "{word}"
          </div>
          
          <div className="flex items-center justify-center gap-2 mb-4">
            <span className="text-purple-300">Padded:</span>
            <span className="font-mono text-pink-400">&lt;{word}&gt;</span>
          </div>

          <div className="flex items-center justify-center mb-4">
            <ChevronRight className="w-6 h-6 text-purple-400" />
          </div>

          <div className="flex flex-wrap justify-center gap-2 mb-4">
            <AnimatePresence mode="popLayout">
              {ngrams.map((ngram, i) => (
                <motion.span
                  key={`${ngram}-${i}`}
                  layout
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="ngram-box font-mono text-purple-100"
                >
                  {ngram}
                </motion.span>
              ))}
            </AnimatePresence>
          </div>

          <div className="text-purple-300/70 text-sm">
            Total: {ngrams.length} {ngramSize}-grams
          </div>
        </div>
      </div>

      {/* Formula */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl p-6"
      >
        <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
          <Info className="w-5 h-5 text-purple-400" />
          Word Vector Formula
        </h3>
        <div className="text-center font-mono text-xl text-purple-100 mb-3">
          v<sub>word</sub> = Σ v<sub>ngram</sub>
        </div>
        <p className="text-purple-200/70 text-sm text-center">
          The word vector is the sum of all its n-gram vectors (including the word itself)
        </p>
      </motion.div>

      {/* Example Word Family */}
      <div className="bg-white/5 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 text-center">
          Shared Subwords in Word Family
        </h3>
        <div className="flex flex-wrap justify-center gap-3">
          {exampleWords.map((w, i) => (
            <motion.button
              key={w}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              onClick={() => setWord(w)}
              className={`px-4 py-2 rounded-lg transition-all ${
                word === w
                  ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
                  : 'bg-white/10 text-purple-300 hover:bg-white/20'
              }`}
            >
              {w}
            </motion.button>
          ))}
        </div>
        <p className="text-purple-300/70 text-sm text-center mt-4">
          Click to see how related words share similar n-grams (like "play")
        </p>
      </div>
    </div>
  );
}

export default SubwordPanel;
