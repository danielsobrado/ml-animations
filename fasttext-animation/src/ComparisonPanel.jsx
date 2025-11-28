import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { GitCompare, Check, X, Minus } from 'lucide-react';

function ComparisonPanel() {
  const [selectedFeature, setSelectedFeature] = useState(null);

  const models = [
    { name: 'Word2Vec', year: 2013, color: 'blue' },
    { name: 'GloVe', year: 2014, color: 'green' },
    { name: 'FastText', year: 2016, color: 'purple' },
  ];

  const features = [
    {
      name: 'Training Method',
      word2vec: 'Predictive (Skip-gram/CBOW)',
      glove: 'Count-based + Predictive',
      fasttext: 'Predictive with subwords',
      winner: 'fasttext'
    },
    {
      name: 'OOV Handling',
      word2vec: 'None',
      glove: 'None',
      fasttext: 'N-gram composition',
      winner: 'fasttext'
    },
    {
      name: 'Morphology',
      word2vec: 'Not captured',
      glove: 'Not captured',
      fasttext: 'Captured via n-grams',
      winner: 'fasttext'
    },
    {
      name: 'Training Speed',
      word2vec: 'Fast',
      glove: 'Moderate',
      fasttext: 'Fast',
      winner: 'tie'
    },
    {
      name: 'Memory Usage',
      word2vec: 'Low',
      glove: 'Moderate',
      fasttext: 'Higher (stores n-grams)',
      winner: 'word2vec'
    },
    {
      name: 'Semantic Analogies',
      word2vec: 'Good',
      glove: 'Good',
      fasttext: 'Good',
      winner: 'tie'
    },
    {
      name: 'Rare Words',
      word2vec: 'Poor',
      glove: 'Poor',
      fasttext: 'Better (subword info)',
      winner: 'fasttext'
    },
    {
      name: 'Syntactic Tasks',
      word2vec: 'Good',
      glove: 'Moderate',
      fasttext: 'Excellent',
      winner: 'fasttext'
    },
  ];

  const getIcon = (model, feature) => {
    if (feature.winner === 'tie') return <Minus className="w-4 h-4 text-gray-400" />;
    if (feature.winner === model) return <Check className="w-4 h-4 text-green-400" />;
    return null;
  };

  return (
    <div className="space-y-6">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold text-white mb-2">Model Comparison</h2>
        <p className="text-purple-200/70">
          How FastText compares to Word2Vec and GloVe
        </p>
      </div>

      {/* Comparison Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr>
              <th className="text-left p-3 text-purple-300">Feature</th>
              {models.map((model) => (
                <th key={model.name} className="text-center p-3">
                  <div className={`text-${model.color}-400 font-semibold`}>{model.name}</div>
                  <div className="text-xs text-gray-400">{model.year}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {features.map((feature, i) => (
              <motion.tr
                key={feature.name}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.05 }}
                onClick={() => setSelectedFeature(selectedFeature === i ? null : i)}
                className={`border-t border-white/10 cursor-pointer transition-colors ${
                  selectedFeature === i ? 'bg-white/10' : 'hover:bg-white/5'
                }`}
              >
                <td className="p-3 text-white font-medium">{feature.name}</td>
                <td className="p-3 text-center">
                  <div className="flex items-center justify-center gap-2">
                    <span className="text-blue-300/80 text-sm">{feature.word2vec}</span>
                    {getIcon('word2vec', feature)}
                  </div>
                </td>
                <td className="p-3 text-center">
                  <div className="flex items-center justify-center gap-2">
                    <span className="text-green-300/80 text-sm">{feature.glove}</span>
                    {getIcon('glove', feature)}
                  </div>
                </td>
                <td className="p-3 text-center">
                  <div className="flex items-center justify-center gap-2">
                    <span className="text-purple-300/80 text-sm">{feature.fasttext}</span>
                    {getIcon('fasttext', feature)}
                  </div>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Visual Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-blue-500/10 border border-blue-400/30 rounded-xl p-4"
        >
          <h3 className="font-semibold text-blue-300 mb-2">Word2Vec</h3>
          <p className="text-blue-200/70 text-sm mb-3">
            Pioneer of neural embeddings. Simple, fast, effective.
          </p>
          <div className="text-xs text-blue-300/60">
            Best for: Simple applications, speed-critical systems
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-green-500/10 border border-green-400/30 rounded-xl p-4"
        >
          <h3 className="font-semibold text-green-300 mb-2">GloVe</h3>
          <p className="text-green-200/70 text-sm mb-3">
            Combines count statistics with prediction. Global context.
          </p>
          <div className="text-xs text-green-300/60">
            Best for: When you have co-occurrence statistics available
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-purple-500/10 border border-purple-400/30 rounded-xl p-4 pulse-glow"
        >
          <h3 className="font-semibold text-purple-300 mb-2">FastText</h3>
          <p className="text-purple-200/70 text-sm mb-3">
            Morphologically aware. Handles OOV words gracefully.
          </p>
          <div className="text-xs text-purple-300/60">
            Best for: Morphologically rich languages, handling rare/OOV words
          </div>
        </motion.div>
      </div>

      {/* When to Use Each */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl p-6"
      >
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <GitCompare className="w-5 h-5 text-purple-400" />
          When to Choose FastText
        </h3>
        <ul className="space-y-2 text-purple-200/80">
          <li className="flex items-start gap-2">
            <Check className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
            <span>Working with morphologically rich languages (German, Turkish, Finnish)</span>
          </li>
          <li className="flex items-start gap-2">
            <Check className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
            <span>Need to handle typos, misspellings, or user-generated content</span>
          </li>
          <li className="flex items-start gap-2">
            <Check className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
            <span>Limited training data with many rare words</span>
          </li>
          <li className="flex items-start gap-2">
            <Check className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
            <span>Text classification tasks (FastText has a built-in classifier)</span>
          </li>
        </ul>
      </motion.div>
    </div>
  );
}

export default ComparisonPanel;
