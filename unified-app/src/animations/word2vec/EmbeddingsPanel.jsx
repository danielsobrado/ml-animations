import React, { useState } from 'react';
import { Sparkles, Search, Calculator, ArrowRight, Shuffle } from 'lucide-react';

export default function EmbeddingsPanel() {
  const [selectedPair, setSelectedPair] = useState(0);
  const [operation, setOperation] = useState('similarity');

  // Simulated word embeddings (2D for visualization)
  const embeddings = {
    king: [0.8, 0.9],
    queen: [0.85, 0.3],
    man: [0.2, 0.8],
    woman: [0.25, 0.2],
    prince: [0.6, 0.85],
    princess: [0.65, 0.25],
    cat: [-0.7, 0.4],
    dog: [-0.6, 0.5],
    kitten: [-0.75, 0.35],
    puppy: [-0.55, 0.55],
    paris: [-0.3, -0.7],
    france: [-0.4, -0.8],
    berlin: [0.1, -0.75],
    germany: [0.2, -0.85],
  };

  const wordPairs = [
    { words: ['king', 'queen'], relationship: 'gender' },
    { words: ['man', 'woman'], relationship: 'gender' },
    { words: ['cat', 'dog'], relationship: 'animal type' },
    { words: ['paris', 'france'], relationship: 'capital-country' },
  ];

  const analogies = [
    { a: 'king', b: 'man', c: 'queen', d: 'woman', explanation: 'king - man + woman ≈ queen' },
    { a: 'paris', b: 'france', c: 'berlin', d: 'germany', explanation: 'paris - france + germany ≈ berlin' },
    { a: 'prince', b: 'man', c: 'princess', d: 'woman', explanation: 'prince - man + woman ≈ princess' },
  ];

  const cosineSimilarity = (a, b) => {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (normA * normB);
  };

  // Similar words finder
  const findSimilar = (word, n = 5) => {
    if (!embeddings[word]) return [];
    const wordVec = embeddings[word];
    const similarities = Object.entries(embeddings)
      .filter(([w]) => w !== word)
      .map(([w, vec]) => ({ word: w, similarity: cosineSimilarity(wordVec, vec) }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, n);
    return similarities;
  };

  return (
    <div className="space-y-6 pb-20">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="text-purple-400">Word Embeddings</span>: Semantic Space
        </h2>
        <p className="text-gray-400">
          Exploring what Word2Vec learns about language
        </p>
      </div>

      {/* Operation Selector */}
      <div className="flex justify-center">
        <div className="bg-black/30 rounded-xl p-1 flex gap-1">
          {[
            { id: 'similarity', label: 'Similarity', icon: Search },
            { id: 'analogy', label: 'Word Analogies', icon: Calculator },
            { id: 'visualization', label: '2D Visualization', icon: Sparkles },
            { id: 'properties', label: 'Properties', icon: Shuffle },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setOperation(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-all ${
                operation === tab.id
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <tab.icon size={16} />
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Similarity View */}
      {operation === 'similarity' && (
        <div className="space-y-6">
          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <h3 className="text-lg font-bold text-white mb-4">Similar Words</h3>
            <p className="text-gray-400 text-sm mb-4">
              Words with similar meaning have vectors that point in similar directions.
            </p>
            
            <div className="grid md:grid-cols-2 gap-6">
              {['king', 'cat', 'paris', 'man'].map((word) => (
                <div key={word} className="bg-white/5 rounded-xl p-4">
                  <h4 className="text-purple-400 font-mono mb-3">Similar to "{word}":</h4>
                  <div className="space-y-2">
                    {findSimilar(word).map((item, i) => (
                      <div key={i} className="flex items-center justify-between">
                        <span className="font-mono text-gray-300">{item.word}</span>
                        <div className="flex items-center gap-2">
                          <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-purple-500 rounded-full"
                              style={{ width: `${(item.similarity + 1) * 50}%` }}
                            />
                          </div>
                          <span className="text-xs text-gray-500 w-12">
                            {item.similarity.toFixed(2)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Cosine Similarity Explanation */}
          <div className="bg-black/40 rounded-xl p-6 border border-white/10">
            <h4 className="font-bold text-white mb-4">📐 Cosine Similarity</h4>
            <div className="bg-black/50 rounded-lg p-4 font-mono text-center mb-4">
              <p className="text-purple-300">
                cos(θ) = (A · B) / (||A|| × ||B||)
              </p>
            </div>
            <div className="grid md:grid-cols-3 gap-4 text-center">
              <div className="bg-green-900/20 rounded-lg p-3">
                <p className="text-green-400 text-xl font-bold">1.0</p>
                <p className="text-xs text-gray-400">Identical direction</p>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <p className="text-gray-400 text-xl font-bold">0.0</p>
                <p className="text-xs text-gray-400">Perpendicular</p>
              </div>
              <div className="bg-red-900/20 rounded-lg p-3">
                <p className="text-red-400 text-xl font-bold">-1.0</p>
                <p className="text-xs text-gray-400">Opposite direction</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Analogy View */}
      {operation === 'analogy' && (
        <div className="space-y-6">
          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <h3 className="text-lg font-bold text-white mb-4">Word Arithmetic</h3>
            <p className="text-gray-400 text-sm mb-6">
              The famous "king - man + woman = queen" and more!
            </p>
            
            <div className="space-y-6">
              {analogies.map((analogy, i) => (
                <div 
                  key={i} 
                  className="bg-gradient-to-r from-purple-900/30 to-pink-900/30 rounded-xl p-6 border border-purple-500/30"
                >
                  <div className="flex flex-wrap items-center justify-center gap-3 text-xl font-mono mb-4">
                    <span className="px-3 py-2 bg-blue-500/30 rounded-lg text-blue-300">{analogy.a}</span>
                    <span className="text-gray-500">-</span>
                    <span className="px-3 py-2 bg-red-500/30 rounded-lg text-red-300">{analogy.b}</span>
                    <span className="text-gray-500">+</span>
                    <span className="px-3 py-2 bg-red-500/30 rounded-lg text-red-300">{analogy.d}</span>
                    <span className="text-gray-500">≈</span>
                    <span className="px-3 py-2 bg-green-500/30 rounded-lg text-green-300">{analogy.c}</span>
                  </div>
                  <p className="text-center text-gray-400 text-sm">{analogy.explanation}</p>
                </div>
              ))}
            </div>
          </div>

          {/* How it works */}
          <div className="bg-black/40 rounded-xl p-6 border border-white/10">
            <h4 className="font-bold text-white mb-4">🎯 How Word Arithmetic Works</h4>
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <span className="text-purple-400 font-bold">1.</span>
                <p className="text-gray-300">
                  <strong className="text-purple-300">Capture relationships:</strong> The vector 
                  difference "king - man" captures the concept of "royalty" or "ruling".
                </p>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-purple-400 font-bold">2.</span>
                <p className="text-gray-300">
                  <strong className="text-purple-300">Transfer relationships:</strong> Adding this 
                  difference to "woman" gives us the female equivalent of "king".
                </p>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-purple-400 font-bold">3.</span>
                <p className="text-gray-300">
                  <strong className="text-purple-300">Find nearest:</strong> We find the word whose 
                  vector is closest to the result: "queen".
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 2D Visualization */}
      {operation === 'visualization' && (
        <div className="space-y-6">
          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <h3 className="text-lg font-bold text-white mb-4">2D Embedding Space</h3>
            <p className="text-gray-400 text-sm mb-4">
              Real embeddings have 100-300 dimensions. This is a 2D projection for visualization.
            </p>
            
            {/* Simple 2D plot */}
            <div className="relative bg-black/50 rounded-xl p-4" style={{ height: '400px' }}>
              {/* Axes */}
              <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gray-700" />
              <div className="absolute top-1/2 left-0 right-0 h-px bg-gray-700" />
              
              {/* Words */}
              {Object.entries(embeddings).map(([word, [x, y]], i) => {
                const left = `${50 + x * 40}%`;
                const top = `${50 - y * 40}%`;
                const colors = {
                  king: 'bg-blue-500', queen: 'bg-pink-500',
                  man: 'bg-blue-400', woman: 'bg-pink-400',
                  prince: 'bg-blue-300', princess: 'bg-pink-300',
                  cat: 'bg-orange-500', dog: 'bg-brown-500',
                  kitten: 'bg-orange-400', puppy: 'bg-yellow-600',
                  paris: 'bg-green-500', france: 'bg-green-600',
                  berlin: 'bg-purple-500', germany: 'bg-purple-600',
                };
                return (
                  <div
                    key={word}
                    className="absolute transform -translate-x-1/2 -translate-y-1/2 group"
                    style={{ left, top }}
                  >
                    <div className={`w-3 h-3 rounded-full ${colors[word] || 'bg-gray-400'} cursor-pointer hover:scale-150 transition-transform`} />
                    <span className="absolute left-4 text-xs text-gray-300 whitespace-nowrap opacity-70 group-hover:opacity-100">
                      {word}
                    </span>
                  </div>
                );
              })}
              
              {/* Legend */}
              <div className="absolute bottom-2 right-2 bg-black/70 rounded p-2 text-xs">
                <div className="flex items-center gap-2"><div className="w-2 h-2 bg-blue-500 rounded-full" /> Royalty (male)</div>
                <div className="flex items-center gap-2"><div className="w-2 h-2 bg-pink-500 rounded-full" /> Royalty (female)</div>
                <div className="flex items-center gap-2"><div className="w-2 h-2 bg-orange-500 rounded-full" /> Animals</div>
                <div className="flex items-center gap-2"><div className="w-2 h-2 bg-green-500 rounded-full" /> Places</div>
              </div>
            </div>
          </div>

          <div className="bg-purple-900/20 rounded-xl p-4 border border-purple-500/30">
            <p className="text-purple-300 text-sm">
              💡 Notice how related words cluster together: royalty words in one area, 
              animals in another, places in another. Gender differences are captured as 
              parallel directions!
            </p>
          </div>
        </div>
      )}

      {/* Properties View */}
      {operation === 'properties' && (
        <div className="space-y-6">
          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <h3 className="text-lg font-bold text-white mb-6">What Word2Vec Learns</h3>
            
            <div className="grid md:grid-cols-2 gap-6">
              {/* Syntactic */}
              <div className="bg-green-900/20 rounded-xl p-4 border border-green-500/30">
                <h4 className="text-green-400 font-bold mb-3">Syntactic Relationships</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Plural:</span>
                    <span className="font-mono text-gray-300">car → cars</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Tense:</span>
                    <span className="font-mono text-gray-300">walk → walked</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Comparative:</span>
                    <span className="font-mono text-gray-300">big → bigger</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Superlative:</span>
                    <span className="font-mono text-gray-300">big → biggest</span>
                  </div>
                </div>
              </div>

              {/* Semantic */}
              <div className="bg-purple-900/20 rounded-xl p-4 border border-purple-500/30">
                <h4 className="text-purple-400 font-bold mb-3">Semantic Relationships</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Capital-Country:</span>
                    <span className="font-mono text-gray-300">Paris → France</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Gender:</span>
                    <span className="font-mono text-gray-300">king → queen</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Opposites:</span>
                    <span className="font-mono text-gray-300">good ↔ bad</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Hypernym:</span>
                    <span className="font-mono text-gray-300">dog → animal</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Limitations */}
          <div className="bg-red-900/20 rounded-xl p-6 border border-red-500/30">
            <h4 className="text-red-400 font-bold mb-4">⚠️ Limitations of Word2Vec</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-black/30 rounded-lg p-3">
                <h5 className="text-white font-medium mb-1">No Context Awareness</h5>
                <p className="text-xs text-gray-400">
                  "bank" has the same vector whether it means riverbank or financial bank
                </p>
              </div>
              <div className="bg-black/30 rounded-lg p-3">
                <h5 className="text-white font-medium mb-1">OOV Problem</h5>
                <p className="text-xs text-gray-400">
                  Can't handle words not seen during training (out-of-vocabulary)
                </p>
              </div>
              <div className="bg-black/30 rounded-lg p-3">
                <h5 className="text-white font-medium mb-1">Bias</h5>
                <p className="text-xs text-gray-400">
                  Learns biases present in training data (gender, racial, etc.)
                </p>
              </div>
              <div className="bg-black/30 rounded-lg p-3">
                <h5 className="text-white font-medium mb-1">Static Vectors</h5>
                <p className="text-xs text-gray-400">
                  Word meaning doesn't change based on surrounding context
                </p>
              </div>
            </div>
          </div>

          {/* Evolution */}
          <div className="bg-blue-900/20 rounded-xl p-6 border border-blue-500/30">
            <h4 className="text-blue-400 font-bold mb-4">📈 What Came Next</h4>
            <div className="flex flex-wrap gap-3">
              <div className="px-4 py-2 bg-black/30 rounded-lg">
                <p className="text-white font-medium">GloVe (2014)</p>
                <p className="text-xs text-gray-400">Global co-occurrence statistics</p>
              </div>
              <ArrowRight className="text-gray-500 self-center" />
              <div className="px-4 py-2 bg-black/30 rounded-lg">
                <p className="text-white font-medium">FastText (2016)</p>
                <p className="text-xs text-gray-400">Subword embeddings</p>
              </div>
              <ArrowRight className="text-gray-500 self-center" />
              <div className="px-4 py-2 bg-black/30 rounded-lg">
                <p className="text-white font-medium">ELMo (2018)</p>
                <p className="text-xs text-gray-400">Contextualized embeddings</p>
              </div>
              <ArrowRight className="text-gray-500 self-center" />
              <div className="px-4 py-2 bg-purple-900/50 rounded-lg border border-purple-500">
                <p className="text-purple-300 font-medium">BERT (2018+)</p>
                <p className="text-xs text-gray-400">Transformer-based</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
