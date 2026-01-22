import React, { useState } from 'react';
import { ArrowRight, Check, X, Scale, Target } from 'lucide-react';

export default function ComparisonPanel() {
  const [activeComparison, setActiveComparison] = useState('matrix');

  const documents = [
    "The cat sat on the mat",
    "The dog chased the cat",
    "A cat and a dog played"
  ];

  // Calculate both BoW and TF-IDF
  const calculateVectors = () => {
    const N = documents.length;
    const tokenized = documents.map(doc => 
      doc.toLowerCase().replace(/[^a-z\s]/g, '').split(/\s+/).filter(w => w)
    );
    const vocab = [...new Set(tokenized.flat())].sort();

    // BoW (raw counts)
    const bow = tokenized.map(tokens => {
      return vocab.map(word => tokens.filter(t => t === word).length);
    });

    // TF-IDF
    const df = {};
    vocab.forEach(word => {
      df[word] = tokenized.filter(tokens => tokens.includes(word)).length;
    });

    const tfidf = tokenized.map(tokens => {
      const counts = {};
      tokens.forEach(t => counts[t] = (counts[t] || 0) + 1);
      return vocab.map(word => {
        const tf = (counts[word] || 0) / tokens.length;
        const idf = Math.log(N / df[word]);
        return tf * idf;
      });
    });

    return { vocab, bow, tfidf };
  };

  const data = calculateVectors();

  const comparisons = [
    {
      aspect: 'Basic Concept',
      bow: 'Counts raw word frequencies',
      tfidf: 'Weights by importance across corpus'
    },
    {
      aspect: 'Common Words',
      bow: 'Treated same as rare words',
      tfidf: 'Penalized (low IDF)'
    },
    {
      aspect: 'Rare Words',
      bow: 'No special treatment',
      tfidf: 'Boosted (high IDF)'
    },
    {
      aspect: 'Document Length',
      bow: 'Longer docs = higher counts',
      tfidf: 'Normalized by doc length'
    },
    {
      aspect: 'Best For',
      bow: 'Simple classification tasks',
      tfidf: 'Search, information retrieval'
    },
    {
      aspect: 'Computation',
      bow: 'Very fast, simple',
      tfidf: 'Slightly more complex'
    }
  ];

  const pros = {
    bow: [
      'Simple and intuitive',
      'Fast to compute',
      'Works well for many tasks',
      'Easy to understand and explain',
      'Good baseline model'
    ],
    tfidf: [
      'Highlights discriminative words',
      'Reduces impact of stop words',
      'Better for search/retrieval',
      'Normalized vector lengths',
      'Captures document specificity'
    ]
  };

  const cons = {
    bow: [
      'Ignores word importance',
      'Stop words dominate',
      'No semantic understanding',
      'Loses word order',
      'High-dimensional sparse vectors'
    ],
    tfidf: [
      'Still loses word order',
      'No semantic understanding',
      'IDF can be unstable with small corpus',
      'Computationally heavier',
      'High-dimensional sparse vectors'
    ]
  };

  return (
    <div className="space-y-6 pb-20">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="text-green-400">BoW</span> vs <span className="text-yellow-400">TF-IDF</span>
        </h2>
        <p className="text-gray-800 dark:text-gray-400">
          When to use each approach for text vectorization
        </p>
      </div>

      {/* Tab Selector */}
      <div className="flex justify-center">
        <div className="bg-black/30 rounded-xl p-1 flex gap-1">
          {[
            { id: 'matrix', label: 'Vector Comparison' },
            { id: 'table', label: 'Feature Comparison' },
            { id: 'proscons', label: 'Pros & Cons' },
            { id: 'usecases', label: 'Use Cases' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveComparison(tab.id)}
              className={`px-4 py-2 rounded-lg text-sm transition-all ${
                activeComparison === tab.id
                  ? 'bg-white/20 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Vector Comparison */}
      {activeComparison === 'matrix' && (
        <div className="space-y-6">
          {/* Documents */}
          <div className="bg-black/30 rounded-xl p-4 border border-white/10">
            <h4 className="text-sm text-gray-800 dark:text-gray-400 mb-3">Sample Documents:</h4>
            {documents.map((doc, i) => (
              <p key={i} className="text-gray-700 dark:text-gray-300 font-mono text-sm">
                Doc {i + 1}: "{doc}"
              </p>
            ))}
          </div>

          {/* Side by Side Matrices */}
          <div className="grid md:grid-cols-2 gap-4">
            {/* BoW Matrix */}
            <div className="bg-green-900/20 rounded-xl p-4 border border-green-500/30">
              <h4 className="font-bold text-green-400 mb-3">Bag of Words (Raw Counts)</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr>
                      <th className="px-1 py-1 text-left text-gray-800 dark:text-gray-400">Doc</th>
                      {data.vocab.map((word, i) => (
                        <th key={i} className="px-1 py-1 text-center text-green-300">{word}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.bow.map((row, i) => (
                      <tr key={i}>
                        <td className="px-1 py-1 text-gray-800 dark:text-gray-400">{i + 1}</td>
                        {row.map((val, j) => (
                          <td key={j} className="px-1 py-1 text-center">
                            <span className={val > 0 ? 'text-green-400 font-bold' : 'text-gray-600'}>
                              {val}
                            </span>
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* TF-IDF Matrix */}
            <div className="bg-yellow-900/20 rounded-xl p-4 border border-yellow-500/30">
              <h4 className="font-bold text-yellow-400 mb-3">TF-IDF (Weighted)</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr>
                      <th className="px-1 py-1 text-left text-gray-800 dark:text-gray-400">Doc</th>
                      {data.vocab.map((word, i) => (
                        <th key={i} className="px-1 py-1 text-center text-yellow-300">{word}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.tfidf.map((row, i) => (
                      <tr key={i}>
                        <td className="px-1 py-1 text-gray-800 dark:text-gray-400">{i + 1}</td>
                        {row.map((val, j) => (
                          <td key={j} className="px-1 py-1 text-center">
                            <span className={val > 0 ? 'text-yellow-400 font-bold' : 'text-gray-600'}>
                              {val.toFixed(2)}
                            </span>
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Key Insight */}
          <div className="bg-blue-900/20 rounded-xl p-4 border border-blue-500/30">
            <h4 className="font-bold text-blue-600 dark:text-blue-400 mb-2">💡 Key Insight</h4>
            <p className="text-gray-700 dark:text-sm">
              Notice how "the" has <span className="text-green-400">high counts in BoW</span> but 
              <span className="text-yellow-400"> zero weight in TF-IDF</span>. This is because "the" 
              appears in all documents, making it non-discriminative. TF-IDF automatically down-weights 
              common words while boosting rare, meaningful terms.
            </p>
          </div>
        </div>
      )}

      {/* Feature Comparison Table */}
      {activeComparison === 'table' && (
        <div className="bg-black/30 rounded-xl p-6 border border-white/10">
          <table className="w-full">
            <thead>
              <tr>
                <th className="text-left p-3 text-gray-800 dark:text-gray-400">Aspect</th>
                <th className="text-left p-3 text-green-400">Bag of Words</th>
                <th className="text-left p-3 text-yellow-400">TF-IDF</th>
              </tr>
            </thead>
            <tbody>
              {comparisons.map((row, i) => (
                <tr key={i} className="border-t border-white/10">
                  <td className="p-3 text-gray-700 dark:text-gray-300 font-medium">{row.aspect}</td>
                  <td className="p-3 text-sm text-gray-800 dark:text-gray-400">{row.bow}</td>
                  <td className="p-3 text-sm text-gray-800 dark:text-gray-400">{row.tfidf}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Pros and Cons */}
      {activeComparison === 'proscons' && (
        <div className="grid md:grid-cols-2 gap-6">
          {/* BoW */}
          <div className="space-y-4">
            <h3 className="text-xl font-bold text-center">Bag of Words</h3>
            
            <div className="bg-green-900/20 rounded-xl p-4 border border-green-500/30">
              <h4 className="flex items-center gap-2 text-green-400 font-medium mb-3">
                <Check size={18} /> Advantages
              </h4>
              <ul className="space-y-2">
                {pros.bow.map((item, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-300">
                    <span className="text-green-400 mt-1">✓</span>
                    {item}
                  </li>
                ))}
              </ul>
            </div>

            <div className="bg-red-900/20 rounded-xl p-4 border border-red-500/30">
              <h4 className="flex items-center gap-2 text-red-400 font-medium mb-3">
                <X size={18} /> Disadvantages
              </h4>
              <ul className="space-y-2">
                {cons.bow.map((item, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-300">
                    <span className="text-red-400 mt-1">✗</span>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* TF-IDF */}
          <div className="space-y-4">
            <h3 className="text-xl font-bold text-center">TF-IDF</h3>
            
            <div className="bg-green-900/20 rounded-xl p-4 border border-green-500/30">
              <h4 className="flex items-center gap-2 text-green-400 font-medium mb-3">
                <Check size={18} /> Advantages
              </h4>
              <ul className="space-y-2">
                {pros.tfidf.map((item, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-300">
                    <span className="text-green-400 mt-1">✓</span>
                    {item}
                  </li>
                ))}
              </ul>
            </div>

            <div className="bg-red-900/20 rounded-xl p-4 border border-red-500/30">
              <h4 className="flex items-center gap-2 text-red-400 font-medium mb-3">
                <X size={18} /> Disadvantages
              </h4>
              <ul className="space-y-2">
                {cons.tfidf.map((item, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-300">
                    <span className="text-red-400 mt-1">✗</span>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Use Cases */}
      {activeComparison === 'usecases' && (
        <div className="space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            {/* BoW Use Cases */}
            <div className="bg-green-900/20 rounded-xl p-6 border border-green-500/30">
              <h3 className="text-xl font-bold text-green-400 mb-4">Use BoW When:</h3>
              <div className="space-y-3">
                {[
                  { title: 'Quick Baseline', desc: 'Need a fast first model for comparison' },
                  { title: 'Small Vocabulary', desc: 'Domain-specific text with limited terms' },
                  { title: 'Document Classification', desc: 'Simple spam/sentiment classification' },
                  { title: 'Memory Constrained', desc: 'Need the simplest representation' },
                  { title: 'Interpretability', desc: 'Need easy-to-explain features' }
                ].map((item, i) => (
                  <div key={i} className="bg-black/30 rounded-lg p-3">
                    <p className="text-green-400 font-medium">{item.title}</p>
                    <p className="text-xs text-gray-800 dark:text-gray-400">{item.desc}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* TF-IDF Use Cases */}
            <div className="bg-yellow-900/20 rounded-xl p-6 border border-yellow-500/30">
              <h3 className="text-xl font-bold text-yellow-400 mb-4">Use TF-IDF When:</h3>
              <div className="space-y-3">
                {[
                  { title: 'Search & Retrieval', desc: 'Finding relevant documents to queries' },
                  { title: 'Keyword Extraction', desc: 'Identifying important terms in documents' },
                  { title: 'Document Similarity', desc: 'Comparing documents meaningfully' },
                  { title: 'Text Summarization', desc: 'Selecting key sentences/phrases' },
                  { title: 'Topic Modeling', desc: 'Pre-processing for LDA/LSA' }
                ].map((item, i) => (
                  <div key={i} className="bg-black/30 rounded-lg p-3">
                    <p className="text-yellow-400 font-medium">{item.title}</p>
                    <p className="text-xs text-gray-800 dark:text-gray-400">{item.desc}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Decision Flowchart */}
          <div className="bg-blue-900/20 rounded-xl p-6 border border-blue-500/30">
            <h4 className="font-bold text-blue-600 dark:text-blue-400 mb-4 text-center">Quick Decision Guide</h4>
            <div className="flex flex-col items-center gap-4">
              <div className="bg-blue-900/40 rounded-lg p-4 text-center">
                <p className="text-blue-300">Do you need to distinguish document-specific terms?</p>
              </div>
              <div className="flex gap-8">
                <div className="flex flex-col items-center gap-2">
                  <ArrowRight className="rotate-90 text-green-400" />
                  <span className="text-sm">No</span>
                  <div className="bg-green-900/40 rounded-lg p-3 text-center">
                    <p className="text-green-400 font-bold">Use BoW</p>
                  </div>
                </div>
                <div className="flex flex-col items-center gap-2">
                  <ArrowRight className="rotate-90 text-yellow-400" />
                  <span className="text-sm">Yes</span>
                  <div className="bg-yellow-900/40 rounded-lg p-3 text-center">
                    <p className="text-yellow-400 font-bold">Use TF-IDF</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Neither is Perfect */}
          <div className="bg-purple-900/20 rounded-xl p-6 border border-purple-500/30">
            <h4 className="font-bold text-purple-600 dark:text-purple-400 mb-3">🤔 Neither is Perfect</h4>
            <p className="text-gray-700 dark:text-sm mb-3">
              Both BoW and TF-IDF have fundamental limitations:
            </p>
            <ul className="text-sm text-gray-800 dark:text-gray-400 space-y-2">
              <li>• <span className="text-red-400">No word order:</span> "dog bites man" = "man bites dog"</li>
              <li>• <span className="text-red-400">No semantics:</span> "happy" and "joyful" are completely different vectors</li>
              <li>• <span className="text-red-400">High dimensionality:</span> Vocabulary size can be huge</li>
            </ul>
            <div className="mt-4 bg-black/30 rounded-lg p-3">
              <p className="text-sm">
                <strong>Modern alternatives:</strong> Word2Vec, GloVe, FastText, BERT embeddings capture semantic meaning
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
