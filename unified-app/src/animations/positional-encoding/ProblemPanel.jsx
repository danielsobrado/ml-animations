import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function ProblemPanel() {
    const [usePositions, setUsePositions] = useState(false);

    const sentence1 = ['Dog', 'bites', 'man'];
    const sentence2 = ['Man', 'bites', 'dog'];

    // Mock embeddings (same words have same vectors)
    const embeddings = {
        'Dog': [0.8, 0.2, 0.3],
        'bites': [0.1, 0.9, 0.4],
        'man': [0.7, 0.3, 0.6],
        'Man': [0.7, 0.3, 0.6] // Same as 'man'
    };

    // Simple positional encoding (just for demo)
    const getPositionalEncoding = (pos) => {
        return [pos * 0.1, pos * 0.05, pos * 0.02];
    };

    const addVectors = (v1, v2) => v1.map((val, i) => val + v2[i]);

    const getFinalEmbedding = (word, pos) => {
        if (usePositions) {
            return addVectors(embeddings[word], getPositionalEncoding(pos));
        }
        return embeddings[word];
    };

    const areSentencesSame = () => {
        if (!usePositions) return true; // Without positions, they're identical
        return false; // With positions, they're different
    };

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-cyan-400 mb-4">The Problem</h2>
                <p className="text-lg text-slate-300 leading-relaxed">
                    Transformers process all words in parallel. Without position information, they can't tell the difference between:
                </p>
            </div>

            {/* Sentences */}
            <div className="grid md:grid-cols-2 gap-8 w-full max-w-5xl mb-8">
                {[sentence1, sentence2].map((sentence, sentenceIdx) => (
                    <div key={sentenceIdx} className="bg-slate-800 p-6 rounded-xl border-2 border-slate-700">
                        <div className="text-center mb-4">
                            <h3 className="text-xl font-bold text-white mb-2">
                                {sentenceIdx === 0 ? '🐕 Sentence A' : '👨 Sentence B'}
                            </h3>
                            <p className="text-2xl font-bold text-cyan-300">
                                "{sentence.join(' ')}"
                            </p>
                            <p className="text-sm text-slate-400 mt-2">
                                {sentenceIdx === 0 ? 'Normal event' : 'Very unusual!'}
                            </p>
                        </div>

                        {/* Word Embeddings */}
                        <div className="space-y-3">
                            {sentence.map((word, wordIdx) => {
                                const finalEmb = getFinalEmbedding(word, wordIdx);
                                return (
                                    <motion.div
                                        key={wordIdx}
                                        layout
                                        className="bg-slate-900 p-3 rounded-lg border border-slate-600"
                                    >
                                        <div className="flex items-center justify-between mb-2">
                                            <span className="font-bold text-white">{word}</span>
                                            {usePositions && (
                                                <span className="text-xs bg-cyan-600 px-2 py-1 rounded-full">
                                                    pos: {wordIdx}
                                                </span>
                                            )}
                                        </div>
                                        <div className="text-xs font-mono text-slate-400">
                                            [{finalEmb.map(v => v.toFixed(2)).join(', ')}]
                                        </div>
                                    </motion.div>
                                );
                            })}
                        </div>
                    </div>
                ))}
            </div>

            {/* Toggle Control */}
            <div className="bg-slate-800 p-6 rounded-xl border-2 border-cyan-500/50 w-full max-w-2xl">
                <div className="flex items-center justify-between mb-4">
                    <div>
                        <h3 className="font-bold text-white text-lg">Positional Encoding</h3>
                        <p className="text-sm text-slate-400">Toggle to see the difference</p>
                    </div>
                    <button
                        onClick={() => setUsePositions(!usePositions)}
                        className={`relative w-20 h-10 rounded-full transition-colors ${usePositions ? 'bg-cyan-500' : 'bg-slate-600'
                            }`}
                    >
                        <motion.div
                            className="absolute top-1 left-1 w-8 h-8 bg-white rounded-full shadow-lg"
                            animate={{ x: usePositions ? 40 : 0 }}
                            transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                        />
                    </button>
                </div>

                <AnimatePresence mode="wait">
                    {areSentencesSame() ? (
                        <motion.div
                            key="same"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="bg-red-900/30 border-2 border-red-500 p-4 rounded-lg"
                        >
                            <p className="text-red-300 font-bold text-center">
                                ⚠️ Problem: Both sentences have IDENTICAL embeddings!
                                <br />
                                <span className="text-sm">The model can't tell them apart.</span>
                            </p>
                        </motion.div>
                    ) : (
                        <motion.div
                            key="different"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="bg-green-900/30 border-2 border-green-500 p-4 rounded-lg"
                        >
                            <p className="text-green-300 font-bold text-center">
                                ✅ Solution: Embeddings are now DIFFERENT!
                                <br />
                                <span className="text-sm">Position information is encoded in the vectors.</span>
                            </p>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
}
