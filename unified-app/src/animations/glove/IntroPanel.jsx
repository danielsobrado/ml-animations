import React from 'react';
import { Lightbulb, BookOpen, Zap, Target, ArrowRight, CheckCircle } from 'lucide-react';

export default function IntroPanel() {
  return (
    <div className="space-y-8 pb-20">
      {/* Title Section */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="gradient-text">GloVe:</span> Global Vectors for Word Representation
        </h2>
        <p className="text-gray-400 max-w-2xl mx-auto">
          Learning word vectors from global word-word co-occurrence statistics
        </p>
      </div>

      {/* Key Insight */}
      <div className="bg-gradient-to-r from-violet-900/30 to-cyan-900/30 rounded-2xl p-6 border border-violet-500/30">
        <h3 className="flex items-center gap-2 text-xl font-bold text-violet-400 mb-4">
          <Lightbulb size={24} />
          The Key Insight
        </h3>
        <div className="text-center">
          <p className="text-xl text-gray-200 italic mb-2">
            "The meaning of a word is captured by the ratio of its co-occurrence probabilities"
          </p>
          <p className="text-sm text-gray-400">— Pennington, Socher, Manning (2014)</p>
        </div>
        <p className="text-gray-300 mt-4 text-center">
          GloVe combines the best of both worlds: global matrix factorization (like LSA) and local context window methods (like Word2Vec).
        </p>
      </div>

      {/* What is GloVe */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-black/30 rounded-xl p-6 border border-white/10">
          <h3 className="flex items-center gap-2 text-lg font-bold text-violet-400 mb-4">
            <Target size={20} />
            What is GloVe?
          </h3>
          <ul className="space-y-3">
            {[
              'Unsupervised learning algorithm for word embeddings',
              'Uses global word-word co-occurrence matrix',
              'Captures semantic relationships through vector arithmetic',
              'Released by Stanford NLP Group in 2014'
            ].map((item, i) => (
              <li key={i} className="flex items-start gap-2 text-gray-300">
                <CheckCircle size={16} className="text-green-400 mt-1 flex-shrink-0" />
                {item}
              </li>
            ))}
          </ul>
        </div>

        <div className="bg-black/30 rounded-xl p-6 border border-white/10">
          <h3 className="flex items-center gap-2 text-lg font-bold text-cyan-400 mb-4">
            <Zap size={20} />
            Why GloVe?
          </h3>
          <ul className="space-y-3">
            {[
              'Leverages global corpus statistics (not just local)',
              'Efficient training through weighted least squares',
              'Pre-trained vectors available for immediate use',
              'State-of-the-art on word analogy tasks'
            ].map((item, i) => (
              <li key={i} className="flex items-start gap-2 text-gray-300">
                <CheckCircle size={16} className="text-cyan-400 mt-1 flex-shrink-0" />
                {item}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Co-occurrence Intuition */}
      <div className="bg-black/30 rounded-xl p-6 border border-white/10">
        <h3 className="flex items-center gap-2 text-lg font-bold text-violet-400 mb-4">
          <BookOpen size={20} />
          The Co-occurrence Intuition
        </h3>
        
        <div className="bg-gradient-to-r from-violet-900/20 to-cyan-900/20 rounded-lg p-4 mb-4">
          <p className="text-gray-300 text-center">
            Consider words <span className="text-violet-400 font-mono">"ice"</span> and <span className="text-cyan-400 font-mono">"steam"</span>. 
            Their relationship to other words reveals their meaning:
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/20">
                <th className="text-left py-2 px-3 text-gray-400">Probability Ratio</th>
                <th className="py-2 px-3 text-cyan-400 font-mono">k = solid</th>
                <th className="py-2 px-3 text-violet-400 font-mono">k = gas</th>
                <th className="py-2 px-3 text-yellow-400 font-mono">k = water</th>
                <th className="py-2 px-3 text-green-400 font-mono">k = fashion</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-white/10">
                <td className="py-2 px-3 text-gray-300 font-mono">P(k | ice)</td>
                <td className="py-2 px-3 text-gray-400 text-center">1.9 × 10⁻⁴</td>
                <td className="py-2 px-3 text-gray-400 text-center">6.6 × 10⁻⁵</td>
                <td className="py-2 px-3 text-gray-400 text-center">3.0 × 10⁻³</td>
                <td className="py-2 px-3 text-gray-400 text-center">1.7 × 10⁻⁵</td>
              </tr>
              <tr className="border-b border-white/10">
                <td className="py-2 px-3 text-gray-300 font-mono">P(k | steam)</td>
                <td className="py-2 px-3 text-gray-400 text-center">2.2 × 10⁻⁵</td>
                <td className="py-2 px-3 text-gray-400 text-center">7.8 × 10⁻⁴</td>
                <td className="py-2 px-3 text-gray-400 text-center">2.2 × 10⁻³</td>
                <td className="py-2 px-3 text-gray-400 text-center">1.8 × 10⁻⁵</td>
              </tr>
              <tr className="bg-violet-900/20">
                <td className="py-2 px-3 text-white font-bold">P(k|ice) / P(k|steam)</td>
                <td className="py-2 px-3 text-cyan-400 font-bold text-center">8.9</td>
                <td className="py-2 px-3 text-violet-400 font-bold text-center">0.085</td>
                <td className="py-2 px-3 text-yellow-400 font-bold text-center">1.36</td>
                <td className="py-2 px-3 text-green-400 font-bold text-center">0.96</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-4">
          <div className="bg-cyan-900/20 rounded-lg p-2 text-center">
            <p className="text-cyan-400 font-bold">≫ 1</p>
            <p className="text-xs text-gray-400">Related to ice</p>
          </div>
          <div className="bg-violet-900/20 rounded-lg p-2 text-center">
            <p className="text-violet-400 font-bold">≪ 1</p>
            <p className="text-xs text-gray-400">Related to steam</p>
          </div>
          <div className="bg-yellow-900/20 rounded-lg p-2 text-center">
            <p className="text-yellow-400 font-bold">≈ 1</p>
            <p className="text-xs text-gray-400">Related to both</p>
          </div>
          <div className="bg-green-900/20 rounded-lg p-2 text-center">
            <p className="text-green-400 font-bold">≈ 1</p>
            <p className="text-xs text-gray-400">Related to neither</p>
          </div>
        </div>
      </div>

      {/* The GloVe Process */}
      <div className="bg-black/30 rounded-xl p-6 border border-white/10">
        <h3 className="flex items-center gap-2 text-lg font-bold text-violet-400 mb-6">
          <Zap size={20} />
          The GloVe Training Process
        </h3>
        
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          {[
            { step: '1', title: 'Build Co-occurrence', desc: 'Count word pairs in context windows' },
            { step: '2', title: 'Compute Probabilities', desc: 'P(j|i) = X_ij / X_i' },
            { step: '3', title: 'Train Vectors', desc: 'Minimize weighted least squares' },
            { step: '4', title: 'Word Vectors', desc: 'W + W̃ for final embeddings' }
          ].map((item, i, arr) => (
            <React.Fragment key={i}>
              <div className="flex flex-col items-center text-center">
                <div className="w-12 h-12 rounded-full bg-violet-600 flex items-center justify-center text-white font-bold mb-2">
                  {item.step}
                </div>
                <h4 className="font-medium text-white">{item.title}</h4>
                <p className="text-xs text-gray-400 max-w-[120px]">{item.desc}</p>
              </div>
              {i < arr.length - 1 && (
                <ArrowRight className="text-gray-600 hidden md:block" size={24} />
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Pre-trained Models */}
      <div className="bg-gradient-to-r from-violet-900/20 to-cyan-900/20 rounded-xl p-6 border border-violet-500/30">
        <h3 className="text-lg font-bold text-violet-400 mb-4">📦 Pre-trained GloVe Models</h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-black/30 rounded-lg p-4">
            <p className="text-cyan-400 font-medium mb-1">Wikipedia + Gigaword</p>
            <p className="text-sm text-gray-400">6B tokens, 400K vocab</p>
            <p className="text-xs text-gray-500">50d, 100d, 200d, 300d vectors</p>
          </div>
          <div className="bg-black/30 rounded-lg p-4">
            <p className="text-cyan-400 font-medium mb-1">Common Crawl (42B)</p>
            <p className="text-sm text-gray-400">42B tokens, 1.9M vocab</p>
            <p className="text-xs text-gray-500">300d vectors</p>
          </div>
          <div className="bg-black/30 rounded-lg p-4">
            <p className="text-cyan-400 font-medium mb-1">Common Crawl (840B)</p>
            <p className="text-sm text-gray-400">840B tokens, 2.2M vocab</p>
            <p className="text-xs text-gray-500">300d vectors</p>
          </div>
          <div className="bg-black/30 rounded-lg p-4">
            <p className="text-cyan-400 font-medium mb-1">Twitter (27B)</p>
            <p className="text-sm text-gray-400">27B tokens, 1.2M vocab</p>
            <p className="text-xs text-gray-500">25d, 50d, 100d, 200d vectors</p>
          </div>
        </div>
      </div>

      {/* What You'll Learn */}
      <div className="bg-black/30 rounded-xl p-6 border border-white/10">
        <h3 className="flex items-center gap-2 text-lg font-bold text-violet-400 mb-4">
          <Target size={20} />
          What You'll Learn
        </h3>
        <div className="grid md:grid-cols-2 gap-3">
          {[
            'How to build a co-occurrence matrix',
            'The GloVe objective function',
            'Weighted least squares optimization',
            'GloVe vs Word2Vec comparison',
            'Loading and using pre-trained GloVe',
            'Word analogies with GloVe vectors'
          ].map((item, i) => (
            <div key={i} className="flex items-center gap-2 text-gray-300">
              <CheckCircle size={16} className="text-green-400 flex-shrink-0" />
              {item}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
