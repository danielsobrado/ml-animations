import React, { useState } from 'react';
import { Lightbulb, ArrowRight, Sparkles, Network } from 'lucide-react';

export default function IntroPanel() {
  const [showAnimation, setShowAnimation] = useState(false);

  const traditionalVsWord2Vec = [
    {
      aspect: 'Representation',
      traditional: 'Sparse one-hot vectors',
      word2vec: 'Dense continuous vectors'
    },
    {
      aspect: 'Dimensions',
      traditional: 'Vocabulary size (10k-1M)',
      word2vec: 'Typically 100-300'
    },
    {
      aspect: 'Similarity',
      traditional: 'All words equally different',
      word2vec: 'Similar words have similar vectors'
    },
    {
      aspect: 'Semantics',
      traditional: 'No semantic meaning',
      word2vec: 'Captures semantic relationships'
    }
  ];

  const famousAnalogies = [
    { a: 'king', b: 'man', c: 'queen', d: 'woman', explanation: 'Gender relationship' },
    { a: 'Paris', b: 'France', c: 'Berlin', d: 'Germany', explanation: 'Capital-country relationship' },
    { a: 'walking', b: 'walked', c: 'swimming', d: 'swam', explanation: 'Tense relationship' },
  ];

  return (
    <div className="space-y-6 pb-20">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="text-blue-400">Word2Vec</span>: Words in Vector Space
        </h2>
        <p className="text-gray-400">
          Revolutionary approach to learning word representations from text
        </p>
      </div>

      {/* The Big Idea */}
      <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-2xl p-6 border border-blue-500/30">
        <h3 className="flex items-center gap-2 text-xl font-bold text-blue-400 mb-4">
          <Lightbulb className="text-yellow-400" />
          The Core Insight
        </h3>
        <div className="text-center text-xl text-gray-300 italic">
          "You shall know a word by the company it keeps"
        </div>
        <p className="text-center text-sm text-gray-500 mt-2">— J.R. Firth (1957)</p>
        <p className="text-gray-300 mt-4 text-center">
          Words that appear in similar contexts tend to have similar meanings.
          Word2Vec learns this by predicting words from their neighbors.
        </p>
      </div>

      {/* What is Word2Vec */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-black/30 rounded-xl p-6 border border-white/10">
          <h3 className="text-lg font-bold text-white mb-4">🎯 What is Word2Vec?</h3>
          <ul className="space-y-3 text-gray-300">
            <li className="flex items-start gap-2">
              <span className="text-green-400">✓</span>
              Neural network that learns word embeddings
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-400">✓</span>
              Maps words to dense vectors (100-300 dimensions)
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-400">✓</span>
              Captures semantic and syntactic relationships
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-400">✓</span>
              Self-supervised: learns from raw text only
            </li>
          </ul>
        </div>

        <div className="bg-black/30 rounded-xl p-6 border border-white/10">
          <h3 className="text-lg font-bold text-white mb-4">📊 Two Architectures</h3>
          <div className="space-y-4">
            <div className="bg-green-900/30 rounded-lg p-4 border border-green-500/30">
              <h4 className="text-green-400 font-medium">Skip-gram</h4>
              <p className="text-sm text-gray-400 mt-1">
                Given center word → predict context words
              </p>
              <div className="text-xs text-gray-500 mt-2">Better for rare words</div>
            </div>
            <div className="bg-yellow-900/30 rounded-lg p-4 border border-yellow-500/30">
              <h4 className="text-yellow-400 font-medium">CBOW (Continuous Bag of Words)</h4>
              <p className="text-sm text-gray-400 mt-1">
                Given context words → predict center word
              </p>
              <div className="text-xs text-gray-500 mt-2">Faster training, good for frequent words</div>
            </div>
          </div>
        </div>
      </div>

      {/* Traditional vs Word2Vec */}
      <div className="bg-black/30 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-bold text-white mb-4">Traditional vs Word2Vec</h3>
        <table className="w-full">
          <thead>
            <tr>
              <th className="text-left p-2 text-gray-400">Aspect</th>
              <th className="text-left p-2 text-red-400">Traditional (One-hot)</th>
              <th className="text-left p-2 text-green-400">Word2Vec</th>
            </tr>
          </thead>
          <tbody>
            {traditionalVsWord2Vec.map((row, i) => (
              <tr key={i} className="border-t border-white/10">
                <td className="p-2 text-gray-300">{row.aspect}</td>
                <td className="p-2 text-sm text-gray-400">{row.traditional}</td>
                <td className="p-2 text-sm text-gray-400">{row.word2vec}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Famous Word Analogies */}
      <div className="bg-gradient-to-r from-purple-900/30 to-pink-900/30 rounded-xl p-6 border border-purple-500/30">
        <h3 className="flex items-center gap-2 text-lg font-bold text-purple-400 mb-4">
          <Sparkles />
          Famous Word Analogies
        </h3>
        <p className="text-gray-400 mb-4 text-sm">
          Word2Vec vectors support arithmetic operations that reveal relationships:
        </p>
        <div className="space-y-4">
          {famousAnalogies.map((analogy, i) => (
            <div key={i} className="bg-black/30 rounded-lg p-4 animate-fadeIn" style={{ animationDelay: `${i * 100}ms` }}>
              <div className="flex flex-wrap items-center gap-2 font-mono text-lg">
                <span className="text-blue-400">{analogy.a}</span>
                <span className="text-gray-500">-</span>
                <span className="text-red-400">{analogy.b}</span>
                <span className="text-gray-500">+</span>
                <span className="text-red-400">{analogy.d}</span>
                <span className="text-gray-500">≈</span>
                <span className="text-green-400">{analogy.c}</span>
              </div>
              <p className="text-xs text-gray-500 mt-2">{analogy.explanation}</p>
            </div>
          ))}
        </div>
        <div className="mt-4 bg-black/40 rounded-lg p-3">
          <p className="text-sm text-gray-300">
            <strong>The magic:</strong> vector("king") - vector("man") + vector("woman") ≈ vector("queen")
          </p>
        </div>
      </div>

      {/* Visual: One-hot vs Dense */}
      <div className="bg-black/30 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-bold text-white mb-4">📐 One-Hot vs Dense Embeddings</h3>
        <div className="grid md:grid-cols-2 gap-6">
          {/* One-hot */}
          <div className="space-y-3">
            <h4 className="text-red-400 font-medium">One-Hot (Sparse)</h4>
            <div className="space-y-2 font-mono text-sm">
              <div className="flex items-center gap-2">
                <span className="text-gray-400 w-12">cat:</span>
                <span className="text-red-400">[1, 0, 0, 0, 0, ...]</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-gray-400 w-12">dog:</span>
                <span className="text-red-400">[0, 1, 0, 0, 0, ...]</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-gray-400 w-12">kitten:</span>
                <span className="text-red-400">[0, 0, 1, 0, 0, ...]</span>
              </div>
            </div>
            <p className="text-xs text-gray-500">All words equally different (similarity = 0)</p>
          </div>
          
          {/* Dense */}
          <div className="space-y-3">
            <h4 className="text-green-400 font-medium">Word2Vec (Dense)</h4>
            <div className="space-y-2 font-mono text-sm">
              <div className="flex items-center gap-2">
                <span className="text-gray-400 w-12">cat:</span>
                <span className="text-green-400">[0.2, -0.4, 0.7, ...]</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-gray-400 w-12">dog:</span>
                <span className="text-green-400">[0.3, -0.3, 0.6, ...]</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-gray-400 w-12">kitten:</span>
                <span className="text-green-400">[0.25, -0.35, 0.72, ...]</span>
              </div>
            </div>
            <p className="text-xs text-gray-500">Similar words have similar vectors!</p>
          </div>
        </div>
      </div>

      {/* Timeline */}
      <div className="bg-black/30 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-bold text-white mb-4">📜 History & Impact</h3>
        <div className="flex flex-wrap gap-4">
          {[
            { year: '2013', event: 'Word2Vec published by Mikolov et al. (Google)' },
            { year: '2014', event: 'GloVe released by Stanford' },
            { year: '2017', event: 'ELMo: contextualized embeddings' },
            { year: '2018', event: 'BERT: transformer-based embeddings' },
          ].map((item, i) => (
            <div key={i} className="flex-1 min-w-[200px] bg-white/5 rounded-lg p-3">
              <div className="text-blue-400 font-bold">{item.year}</div>
              <div className="text-sm text-gray-400">{item.event}</div>
            </div>
          ))}
        </div>
      </div>

      {/* What You'll Learn */}
      <div className="bg-gradient-to-r from-green-900/20 to-cyan-900/20 rounded-xl p-6 border border-green-500/30">
        <h3 className="text-lg font-bold text-green-400 mb-4">🎯 What You'll Learn</h3>
        <div className="grid md:grid-cols-2 gap-3">
          {[
            'Skip-gram architecture and training',
            'CBOW (Continuous Bag of Words) model',
            'Negative sampling optimization',
            'How embeddings capture semantics',
            'Word arithmetic and analogies',
            'Training your own Word2Vec model'
          ].map((item, i) => (
            <div key={i} className="flex items-center gap-2 text-gray-300">
              <span className="text-green-400">✓</span>
              {item}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
