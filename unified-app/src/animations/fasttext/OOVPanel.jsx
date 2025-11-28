import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { HelpCircle, Check, X, Sparkles, AlertTriangle } from 'lucide-react';

function OOVPanel() {
  const [testWord, setTestWord] = useState('');
  const [showResult, setShowResult] = useState(false);

  const vocabulary = ['king', 'queen', 'prince', 'princess', 'royal', 'kingdom', 'castle'];
  const oovExamples = [
    { word: 'kingship', known: false, ngrams: ['<ki', 'kin', 'ing', 'ngs', 'gsh', 'shi', 'hip', 'ip>'] },
    { word: 'queenly', known: false, ngrams: ['<qu', 'que', 'uee', 'een', 'enl', 'nly', 'ly>'] },
    { word: 'unkingly', known: false, ngrams: ['<un', 'unk', 'nki', 'kin', 'ing', 'ngl', 'gly', 'ly>'] },
  ];

  const [selectedExample, setSelectedExample] = useState(oovExamples[0]);

  const handleTestWord = () => {
    setShowResult(true);
  };

  return (
    <div className="space-y-6">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold text-white mb-2">Handling Out-of-Vocabulary Words</h2>
        <p className="text-purple-200/70">
          FastText's superpower: generating embeddings for words never seen during training
        </p>
      </div>

      {/* Problem Illustration */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="bg-red-500/10 border border-red-400/30 rounded-xl p-4"
        >
          <div className="flex items-center gap-2 mb-3">
            <X className="w-5 h-5 text-red-400" />
            <h3 className="font-semibold text-red-300">Word2Vec / GloVe</h3>
          </div>
          <p className="text-red-200/70 text-sm mb-3">
            Unknown word? Return a zero vector or random initialization.
          </p>
          <div className="bg-red-500/10 rounded-lg p-3 font-mono text-sm">
            <span className="text-red-300">word2vec</span>
            <span className="text-white">[</span>
            <span className="text-red-400">"kingship"</span>
            <span className="text-white">]</span>
            <span className="text-gray-400"> → </span>
            <span className="text-red-400">KeyError!</span>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="bg-green-500/10 border border-green-400/30 rounded-xl p-4"
        >
          <div className="flex items-center gap-2 mb-3">
            <Check className="w-5 h-5 text-green-400" />
            <h3 className="font-semibold text-green-300">FastText</h3>
          </div>
          <p className="text-green-200/70 text-sm mb-3">
            Unknown word? Sum the n-gram vectors to create an embedding!
          </p>
          <div className="bg-green-500/10 rounded-lg p-3 font-mono text-sm">
            <span className="text-green-300">fasttext</span>
            <span className="text-white">[</span>
            <span className="text-green-400">"kingship"</span>
            <span className="text-white">]</span>
            <span className="text-gray-400"> → </span>
            <span className="text-green-400">[0.3, -0.2, ...]</span>
          </div>
        </motion.div>
      </div>

      {/* Interactive OOV Demo */}
      <div className="bg-white/5 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-purple-400" />
          OOV Word Resolution
        </h3>

        {/* Known Vocabulary */}
        <div className="mb-6">
          <p className="text-purple-300 text-sm mb-2">Training Vocabulary:</p>
          <div className="flex flex-wrap gap-2">
            {vocabulary.map((word) => (
              <span
                key={word}
                className="px-3 py-1 bg-purple-500/20 text-purple-200 rounded-full text-sm"
              >
                {word}
              </span>
            ))}
          </div>
        </div>

        {/* OOV Examples */}
        <div className="mb-6">
          <p className="text-purple-300 text-sm mb-2">Try an OOV word:</p>
          <div className="flex flex-wrap gap-2">
            {oovExamples.map((example) => (
              <motion.button
                key={example.word}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setSelectedExample(example)}
                className={`px-4 py-2 rounded-lg transition-all ${
                  selectedExample.word === example.word
                    ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
                    : 'bg-white/10 text-purple-300 hover:bg-white/20'
                }`}
              >
                {example.word}
                <span className="ml-2 text-xs opacity-70">(OOV)</span>
              </motion.button>
            ))}
          </div>
        </div>

        {/* Resolution Process */}
        <AnimatePresence mode="wait">
          <motion.div
            key={selectedExample.word}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 rounded-xl p-4 border border-purple-400/20"
          >
            <div className="text-center mb-4">
              <div className="flex items-center justify-center gap-2 mb-2">
                <AlertTriangle className="w-4 h-4 text-yellow-400" />
                <span className="text-yellow-300 text-sm">Word not in vocabulary!</span>
              </div>
              <span className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
                "{selectedExample.word}"
              </span>
            </div>

            <div className="flex items-center justify-center gap-2 text-purple-300 mb-4">
              <span>→ Extract n-grams →</span>
            </div>

            <div className="flex flex-wrap justify-center gap-2 mb-4">
              {selectedExample.ngrams.map((ngram, i) => (
                <motion.span
                  key={ngram}
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.1 }}
                  className="ngram-box font-mono text-purple-100 text-sm"
                >
                  {ngram}
                </motion.span>
              ))}
            </div>

            <div className="flex items-center justify-center gap-2 text-purple-300 mb-4">
              <span>→ Sum n-gram vectors →</span>
            </div>

            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.5 }}
              className="text-center"
            >
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-green-500/20 border border-green-400/30 rounded-lg">
                <Check className="w-5 h-5 text-green-400" />
                <span className="text-green-300">Valid embedding generated!</span>
              </div>
            </motion.div>
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Key Benefits */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { title: 'Typos', desc: '"recieve" still works!', icon: '✏️' },
          { title: 'New Words', desc: '"COVID-19" handled', icon: '🆕' },
          { title: 'Morphology', desc: '"un-" prefix captured', icon: '🧩' },
        ].map((item, i) => (
          <motion.div
            key={item.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="bg-white/5 rounded-xl p-4 text-center"
          >
            <div className="text-2xl mb-2">{item.icon}</div>
            <h4 className="font-semibold text-white mb-1">{item.title}</h4>
            <p className="text-purple-300/70 text-sm">{item.desc}</p>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

export default OOVPanel;
