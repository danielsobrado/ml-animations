import React, { useState } from 'react';
import { GitCompare, CheckCircle, XCircle, Zap, Target } from 'lucide-react';

export default function ComparisonPanel() {
  const [selectedAspect, setSelectedAspect] = useState('training');

  const aspects = {
    training: {
      title: 'Training Approach',
      glove: {
        items: [
          'Pre-computes co-occurrence matrix',
          'Weighted least squares optimization',
          'Uses all co-occurrence statistics at once',
          'Trains on observed co-occurrences (X_ij > 0)'
        ]
      },
      word2vec: {
        items: [
          'Trains with sliding context window',
          'Stochastic gradient descent',
          'Samples mini-batches from corpus',
          'Uses negative sampling for efficiency'
        ]
      }
    },
    objective: {
      title: 'Learning Objective',
      glove: {
        items: [
          'Minimize squared error on log counts',
          'J = Σ f(X_ij)(w·w̃ + b - log X)²',
          'Global co-occurrence statistics',
          'Matrix factorization perspective'
        ]
      },
      word2vec: {
        items: [
          'Maximize probability of context words',
          'Skip-gram: P(context | center)',
          'CBOW: P(center | context)',
          'Local context prediction task'
        ]
      }
    },
    efficiency: {
      title: 'Computational Efficiency',
      glove: {
        items: [
          'One-time matrix construction',
          'Training time: O(|X|) non-zero entries',
          'Parallelizes well on non-zero entries',
          'Memory: stores sparse co-occurrence matrix'
        ]
      },
      word2vec: {
        items: [
          'Multiple passes over corpus',
          'Training time: O(corpus size × epochs)',
          'Parallelizes with async SGD',
          'Memory: stores word vectors only'
        ]
      }
    },
    performance: {
      title: 'Performance Characteristics',
      glove: {
        items: [
          'Better on word analogy tasks',
          'Captures global statistics well',
          'More interpretable objective',
          'Faster training on large corpora'
        ]
      },
      word2vec: {
        items: [
          'Better on similarity tasks',
          'Captures local context patterns',
          'More flexible architecture',
          'Online learning possible'
        ]
      }
    }
  };

  const similarities = [
    'Both produce dense word vectors (50-300 dimensions)',
    'Both capture semantic relationships',
    'Both support word arithmetic (king - man + woman ≈ queen)',
    'Both use context window concept',
    'Both pre-trained models widely available',
    'Both foundational for downstream NLP tasks'
  ];

  return (
    <div className="space-y-6 pb-20">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="gradient-text">GloVe vs Word2Vec:</span> A Comparison
        </h2>
        <p className="text-gray-400">
          Understanding the differences between two influential embedding methods
        </p>
      </div>

      {/* High-level Summary */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-gradient-to-br from-violet-900/30 to-violet-800/20 rounded-xl p-6 border border-violet-500/30">
          <div className="flex items-center gap-3 mb-4">
            <span className="text-3xl">🔮</span>
            <div>
              <h3 className="text-xl font-bold text-violet-400">GloVe</h3>
              <p className="text-sm text-gray-400">Global Vectors</p>
            </div>
          </div>
          <p className="text-gray-300 text-sm">
            <strong className="text-violet-400">Count-based method</strong> that learns from the 
            global word-word co-occurrence matrix. Combines the benefits of global matrix 
            factorization and local context window methods.
          </p>
          <div className="mt-4 text-xs text-gray-500">
            Stanford NLP Group, 2014
          </div>
        </div>

        <div className="bg-gradient-to-br from-cyan-900/30 to-cyan-800/20 rounded-xl p-6 border border-cyan-500/30">
          <div className="flex items-center gap-3 mb-4">
            <span className="text-3xl">🎯</span>
            <div>
              <h3 className="text-xl font-bold text-cyan-400">Word2Vec</h3>
              <p className="text-sm text-gray-400">Skip-gram / CBOW</p>
            </div>
          </div>
          <p className="text-gray-300 text-sm">
            <strong className="text-cyan-400">Prediction-based method</strong> that learns word 
            embeddings by predicting context words (Skip-gram) or center words (CBOW) using 
            a neural network trained with SGD.
          </p>
          <div className="mt-4 text-xs text-gray-500">
            Google, Mikolov et al., 2013
          </div>
        </div>
      </div>

      {/* Aspect Selector */}
      <div className="flex flex-wrap justify-center gap-2">
        {Object.keys(aspects).map((key) => (
          <button
            key={key}
            onClick={() => setSelectedAspect(key)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              selectedAspect === key
                ? 'bg-violet-600 text-white'
                : 'bg-white/10 text-gray-400 hover:bg-white/20'
            }`}
          >
            {aspects[key].title}
          </button>
        ))}
      </div>

      {/* Detailed Comparison */}
      <div className="bg-black/30 rounded-xl p-6 border border-white/10">
        <h4 className="text-xl font-bold text-white mb-6 text-center">
          {aspects[selectedAspect].title}
        </h4>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-violet-900/20 rounded-lg p-4 border border-violet-500/30">
            <h5 className="text-violet-400 font-bold mb-4 flex items-center gap-2">
              🔮 GloVe
            </h5>
            <ul className="space-y-3">
              {aspects[selectedAspect].glove.items.map((item, i) => (
                <li key={i} className="flex items-start gap-2 text-gray-300 text-sm">
                  <CheckCircle size={16} className="text-violet-400 mt-0.5 flex-shrink-0" />
                  {item}
                </li>
              ))}
            </ul>
          </div>
          
          <div className="bg-cyan-900/20 rounded-lg p-4 border border-cyan-500/30">
            <h5 className="text-cyan-400 font-bold mb-4 flex items-center gap-2">
              🎯 Word2Vec
            </h5>
            <ul className="space-y-3">
              {aspects[selectedAspect].word2vec.items.map((item, i) => (
                <li key={i} className="flex items-start gap-2 text-gray-300 text-sm">
                  <CheckCircle size={16} className="text-cyan-400 mt-0.5 flex-shrink-0" />
                  {item}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Side-by-side Table */}
      <div className="bg-black/30 rounded-xl p-6 border border-white/10 overflow-x-auto">
        <h4 className="text-lg font-bold text-violet-400 mb-4">📊 Quick Comparison Table</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-white/20">
              <th className="text-left py-2 px-3 text-gray-400">Aspect</th>
              <th className="py-2 px-3 text-violet-400">GloVe</th>
              <th className="py-2 px-3 text-cyan-400">Word2Vec</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/10">
            <tr>
              <td className="py-3 px-3 text-gray-300">Approach</td>
              <td className="py-3 px-3 text-center text-gray-400">Count-based</td>
              <td className="py-3 px-3 text-center text-gray-400">Prediction-based</td>
            </tr>
            <tr>
              <td className="py-3 px-3 text-gray-300">Statistics</td>
              <td className="py-3 px-3 text-center text-gray-400">Global</td>
              <td className="py-3 px-3 text-center text-gray-400">Local</td>
            </tr>
            <tr>
              <td className="py-3 px-3 text-gray-300">Optimization</td>
              <td className="py-3 px-3 text-center text-gray-400">Weighted Least Squares</td>
              <td className="py-3 px-3 text-center text-gray-400">SGD + Negative Sampling</td>
            </tr>
            <tr>
              <td className="py-3 px-3 text-gray-300">Pre-processing</td>
              <td className="py-3 px-3 text-center text-gray-400">Build co-occurrence matrix</td>
              <td className="py-3 px-3 text-center text-gray-400">None (online)</td>
            </tr>
            <tr>
              <td className="py-3 px-3 text-gray-300">Memory Usage</td>
              <td className="py-3 px-3 text-center text-gray-400">Higher (stores matrix)</td>
              <td className="py-3 px-3 text-center text-gray-400">Lower</td>
            </tr>
            <tr>
              <td className="py-3 px-3 text-gray-300">Training Speed</td>
              <td className="py-3 px-3 text-center text-gray-400">Faster on large corpora</td>
              <td className="py-3 px-3 text-center text-gray-400">Good for medium corpora</td>
            </tr>
            <tr>
              <td className="py-3 px-3 text-gray-300">Incremental Learning</td>
              <td className="py-3 px-3 text-center text-red-400">❌ No</td>
              <td className="py-3 px-3 text-center text-green-400">✅ Yes</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Similarities */}
      <div className="bg-gradient-to-r from-green-900/20 to-cyan-900/20 rounded-xl p-6 border border-green-500/30">
        <h4 className="flex items-center gap-2 text-lg font-bold text-green-400 mb-4">
          <Zap size={20} />
          What They Have in Common
        </h4>
        <div className="grid md:grid-cols-2 gap-3">
          {similarities.map((item, i) => (
            <div key={i} className="flex items-start gap-2 text-gray-300 text-sm">
              <CheckCircle size={16} className="text-green-400 mt-0.5 flex-shrink-0" />
              {item}
            </div>
          ))}
        </div>
      </div>

      {/* When to Use */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-violet-900/20 rounded-xl p-6 border border-violet-500/30">
          <h4 className="flex items-center gap-2 text-lg font-bold text-violet-400 mb-4">
            <Target size={20} />
            Use GloVe When:
          </h4>
          <ul className="space-y-2">
            {[
              'You have a large, fixed corpus',
              'Word analogy tasks are important',
              'Training time is a concern',
              'Global statistics matter for your task',
              'Using pre-trained embeddings'
            ].map((item, i) => (
              <li key={i} className="flex items-start gap-2 text-gray-300 text-sm">
                <CheckCircle size={16} className="text-violet-400 mt-0.5 flex-shrink-0" />
                {item}
              </li>
            ))}
          </ul>
        </div>
        
        <div className="bg-cyan-900/20 rounded-xl p-6 border border-cyan-500/30">
          <h4 className="flex items-center gap-2 text-lg font-bold text-cyan-400 mb-4">
            <Target size={20} />
            Use Word2Vec When:
          </h4>
          <ul className="space-y-2">
            {[
              'You need online/incremental learning',
              'Corpus keeps growing over time',
              'Memory is limited',
              'Word similarity is the main task',
              'You want more control over training'
            ].map((item, i) => (
              <li key={i} className="flex items-start gap-2 text-gray-300 text-sm">
                <CheckCircle size={16} className="text-cyan-400 mt-0.5 flex-shrink-0" />
                {item}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* The Truth */}
      <div className="bg-black/30 rounded-xl p-6 border border-white/10 text-center">
        <h4 className="text-lg font-bold text-white mb-3">🎓 The Practical Truth</h4>
        <p className="text-gray-300 max-w-2xl mx-auto">
          In practice, both methods produce comparable results on most tasks. 
          The choice often comes down to <strong className="text-violet-400">availability of pre-trained vectors</strong> or 
          <strong className="text-cyan-400"> specific task requirements</strong>. 
          Today, both are often superseded by contextual embeddings (BERT, GPT) for state-of-the-art results.
        </p>
      </div>
    </div>
  );
}
