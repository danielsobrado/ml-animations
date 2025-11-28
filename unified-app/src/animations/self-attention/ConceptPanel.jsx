import React, { useState } from 'react';
import { motion } from 'framer-motion';

export default function ConceptPanel() {
    const [queryPos, setQueryPos] = useState(0); // 0, 1, 2 (Positions of books)

    const books = [
        { id: 0, title: "Cooking 101", key: "Food", content: "Recipes for delicious meals.", color: "bg-orange-500" },
        { id: 1, title: "Astronomy", key: "Space", content: "Stars, planets, and galaxies.", color: "bg-blue-500" },
        { id: 2, title: "History of Rome", key: "History", content: "The rise and fall of an empire.", color: "bg-red-500" }
    ];

    const query = { text: "I want to learn about planets.", vector: "Space" };

    // Simple "similarity" check (string match for this analogy)
    const getScore = (bookKey) => {
        if (bookKey === query.vector) return 1.0;
        return 0.1;
    };

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-fuchsia-400 mb-4">The Library Analogy</h2>
                <p className="text-lg text-slate-300 leading-relaxed">
                    <strong>Query (Q)</strong>: What you're looking for.
                    <br />
                    <strong>Key (K)</strong>: The label on the book spine (metadata).
                    <br />
                    <strong>Value (V)</strong>: The actual content inside the book.
                </p>
            </div>

            <div className="grid md:grid-cols-2 gap-12 w-full max-w-5xl items-center">
                {/* The Query (User) */}
                <div className="flex flex-col items-center">
                    <div className="bg-slate-800 p-6 rounded-xl border-2 border-fuchsia-500 mb-8 w-64 text-center">
                        <h3 className="text-fuchsia-400 font-bold mb-2">QUERY (Q)</h3>
                        <p className="text-white italic">"{query.text}"</p>
                        <div className="mt-4 bg-slate-900 p-2 rounded text-xs font-mono text-slate-400">
                            Looking for: <span className="text-blue-400 font-bold">{query.vector}</span>
                        </div>
                    </div>

                    {/* Drag Control (Simulated) */}
                    <div className="flex gap-2">
                        {books.map((book, i) => (
                            <button
                                key={i}
                                onClick={() => setQueryPos(i)}
                                className={`w-12 h-12 rounded-full font-bold transition-all ${queryPos === i
                                        ? 'bg-fuchsia-600 text-white scale-110 ring-4 ring-fuchsia-500/30'
                                        : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                                    }`}
                            >
                                {i + 1}
                            </button>
                        ))}
                    </div>
                    <p className="text-xs text-slate-500 mt-2">Move Query to compare with Keys</p>
                </div>

                {/* The Keys (Bookshelf) */}
                <div className="space-y-4">
                    {books.map((book, i) => {
                        const score = getScore(book.key);
                        const isMatch = score > 0.5;
                        const isHovered = queryPos === i;

                        return (
                            <motion.div
                                key={book.id}
                                className={`relative p-4 rounded-xl border-2 transition-all ${isHovered ? 'scale-105 z-10' : 'scale-100 opacity-80'
                                    } ${isMatch && isHovered ? 'border-green-500 bg-green-900/20' : 'border-slate-700 bg-slate-800'}`}
                                animate={{ x: isHovered ? -20 : 0 }}
                            >
                                <div className="flex justify-between items-center">
                                    <div className="flex items-center gap-4">
                                        {/* Book Spine (Key) */}
                                        <div className={`w-12 h-16 ${book.color} rounded flex items-center justify-center text-white font-bold text-xs shadow-lg`}>
                                            KEY
                                        </div>
                                        <div>
                                            <h4 className="font-bold text-white">{book.title}</h4>
                                            <p className="text-xs text-slate-400">Key: {book.key}</p>
                                        </div>
                                    </div>

                                    {/* Score */}
                                    {isHovered && (
                                        <div className="text-right">
                                            <div className="text-xs text-slate-400">Match Score</div>
                                            <div className={`text-2xl font-bold ${isMatch ? 'text-green-400' : 'text-slate-600'}`}>
                                                {(score * 100).toFixed(0)}%
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* Value (Content) */}
                                {isMatch && isHovered && (
                                    <motion.div
                                        initial={{ opacity: 0, height: 0 }}
                                        animate={{ opacity: 1, height: 'auto' }}
                                        className="mt-4 p-3 bg-slate-900 rounded border border-slate-600"
                                    >
                                        <div className="text-xs text-slate-500 mb-1 uppercase tracking-wider">Value (Content)</div>
                                        <p className="text-slate-200 text-sm">{book.content}</p>
                                    </motion.div>
                                )}
                            </motion.div>
                        );
                    })}
                </div>
            </div>

            {/* Summary */}
            <div className="mt-12 bg-slate-800 p-6 rounded-xl border border-slate-700 max-w-3xl text-center">
                <h3 className="font-bold text-white mb-2">How Attention Works</h3>
                <p className="text-slate-300 text-sm">
                    1. Compute <strong>Dot Product</strong> between Query and all Keys (Similarity Score).
                    <br />
                    2. Apply <strong>Softmax</strong> to normalize scores (Probabilities).
                    <br />
                    3. Multiply scores by <strong>Values</strong> to get the final result (Weighted Sum).
                </p>
            </div>
        </div>
    );
}
