import React, { useState, useEffect } from 'react';
import { ArrowRight, Lightbulb, Hash, Layers, Sparkles } from 'lucide-react';

// Sample word embeddings (simplified 2D for visualization)
const SAMPLE_WORDS = {
    'king': { embedding: [0.8, 0.9], category: 'royalty' },
    'queen': { embedding: [0.75, 0.85], category: 'royalty' },
    'man': { embedding: [0.6, 0.7], category: 'gender' },
    'woman': { embedding: [0.55, 0.65], category: 'gender' },
    'cat': { embedding: [-0.3, 0.4], category: 'animal' },
    'dog': { embedding: [-0.25, 0.45], category: 'animal' },
    'car': { embedding: [0.2, -0.6], category: 'vehicle' },
    'truck': { embedding: [0.25, -0.55], category: 'vehicle' },
    'happy': { embedding: [-0.7, 0.2], category: 'emotion' },
    'sad': { embedding: [-0.75, 0.15], category: 'emotion' },
};

const CATEGORY_COLORS = {
    'royalty': 'bg-purple-100 border-purple-400 text-purple-800',
    'gender': 'bg-blue-100 border-blue-400 text-blue-800',
    'animal': 'bg-green-100 border-green-400 text-green-800',
    'vehicle': 'bg-orange-100 border-orange-400 text-orange-800',
    'emotion': 'bg-pink-100 border-pink-400 text-pink-800',
};

export default function ConceptPanel() {
    const [selectedWord, setSelectedWord] = useState('king');
    const [animatingIdx, setAnimatingIdx] = useState(-1);

    useEffect(() => {
        const words = Object.keys(SAMPLE_WORDS);
        words.forEach((_, i) => {
            setTimeout(() => setAnimatingIdx(i), i * 150);
        });
    }, []);

    const selectedData = SAMPLE_WORDS[selectedWord];

    return (
        <div className="p-8 h-full">
            <div className="max-w-4xl mx-auto">
                {/* Introduction */}
                <div className="text-center mb-8">
                    <h2 className="text-3xl font-bold text-indigo-900 mb-4">What are Embeddings?</h2>
                    <p className="text-lg text-slate-700 leading-relaxed max-w-2xl mx-auto">
                        Embeddings are <strong>dense numerical representations</strong> of data (words, sentences, images) 
                        where similar items are close together in a high-dimensional vector space.
                    </p>
                </div>

                {/* The Big Idea */}
                <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl p-6 mb-8 border border-indigo-100">
                    <div className="flex items-center justify-center gap-6 flex-wrap">
                        <div className="text-center">
                            <div className="text-4xl mb-2">👑</div>
                            <div className="font-bold text-slate-800">"King"</div>
                        </div>
                        <ArrowRight className="text-indigo-400" size={32} />
                        <div className="text-center">
                            <div className="font-mono text-sm bg-white p-3 rounded-lg border shadow-sm">
                                [0.23, -0.45, 0.78, ...]
                                <div className="text-xs text-slate-500 mt-1">768 dimensions</div>
                            </div>
                        </div>
                        <ArrowRight className="text-indigo-400" size={32} />
                        <div className="text-center">
                            <div className="text-4xl mb-2">📊</div>
                            <div className="font-bold text-slate-800">Vector Space</div>
                        </div>
                    </div>
                </div>

                {/* Why Embeddings Matter */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                    <div className="bg-blue-50 p-5 rounded-xl border-2 border-blue-100">
                        <div className="bg-blue-100 w-10 h-10 rounded-full flex items-center justify-center mb-3 text-blue-600">
                            <Hash size={20} />
                        </div>
                        <h3 className="font-bold text-blue-900 mb-2">Numerical Representation</h3>
                        <p className="text-blue-800 text-sm">
                            Computers can't understand words directly. Embeddings convert meaning into numbers 
                            that neural networks can process.
                        </p>
                    </div>

                    <div className="bg-green-50 p-5 rounded-xl border-2 border-green-100">
                        <div className="bg-green-100 w-10 h-10 rounded-full flex items-center justify-center mb-3 text-green-600">
                            <Layers size={20} />
                        </div>
                        <h3 className="font-bold text-green-900 mb-2">Semantic Meaning</h3>
                        <p className="text-green-800 text-sm">
                            Similar words have similar embeddings. "King" and "Queen" are closer together 
                            than "King" and "Car".
                        </p>
                    </div>

                    <div className="bg-purple-50 p-5 rounded-xl border-2 border-purple-100">
                        <div className="bg-purple-100 w-10 h-10 rounded-full flex items-center justify-center mb-3 text-purple-600">
                            <Sparkles size={20} />
                        </div>
                        <h3 className="font-bold text-purple-900 mb-2">Learned Features</h3>
                        <p className="text-purple-800 text-sm">
                            Embeddings capture relationships: King - Man + Woman ≈ Queen! 
                            The model learns these patterns from data.
                        </p>
                    </div>
                </div>

                {/* Interactive Word Explorer */}
                <div className="bg-slate-50 rounded-xl p-6 mb-8">
                    <h3 className="text-xl font-bold text-slate-800 mb-4">Explore Word Embeddings</h3>
                    
                    <div className="flex flex-wrap gap-2 mb-4">
                        {Object.entries(SAMPLE_WORDS).map(([word, data], i) => (
                            <button
                                key={word}
                                onClick={() => setSelectedWord(word)}
                                className={`px-4 py-2 rounded-lg font-medium border-2 transition-all duration-300 ${
                                    CATEGORY_COLORS[data.category]
                                } ${selectedWord === word ? 'ring-2 ring-indigo-500 scale-110' : ''} ${
                                    i <= animatingIdx ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
                                }`}
                            >
                                {word}
                            </button>
                        ))}
                    </div>

                    {/* Selected Word Details */}
                    <div className="bg-white rounded-lg p-4 border">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div>
                                <h4 className="text-sm font-medium text-slate-600 mb-1">Word</h4>
                                <div className="text-2xl font-bold text-slate-800">{selectedWord}</div>
                            </div>
                            <div>
                                <h4 className="text-sm font-medium text-slate-600 mb-1">Category</h4>
                                <span className={`px-3 py-1 rounded-full text-sm font-medium ${CATEGORY_COLORS[selectedData.category]}`}>
                                    {selectedData.category}
                                </span>
                            </div>
                            <div>
                                <h4 className="text-sm font-medium text-slate-600 mb-1">Embedding (2D simplified)</h4>
                                <div className="font-mono text-sm bg-slate-100 px-3 py-2 rounded">
                                    [{selectedData.embedding[0].toFixed(2)}, {selectedData.embedding[1].toFixed(2)}]
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* The Famous Equation */}
                <div className="bg-gradient-to-r from-amber-50 to-orange-50 p-6 rounded-xl border border-amber-200 mb-8">
                    <h3 className="text-xl font-bold text-amber-900 mb-4 text-center">The Famous Word2Vec Equation</h3>
                    <div className="flex items-center justify-center gap-3 flex-wrap text-xl font-mono">
                        <span className="bg-purple-100 px-3 py-1 rounded">King</span>
                        <span>-</span>
                        <span className="bg-blue-100 px-3 py-1 rounded">Man</span>
                        <span>+</span>
                        <span className="bg-blue-100 px-3 py-1 rounded">Woman</span>
                        <span>≈</span>
                        <span className="bg-purple-100 px-3 py-1 rounded">Queen</span>
                    </div>
                    <p className="text-amber-800 text-center mt-4 text-sm">
                        Embeddings capture semantic relationships! Subtracting "Man" and adding "Woman" 
                        transforms the "King" vector to be close to "Queen".
                    </p>
                </div>

                {/* Key Insight */}
                <div className="bg-emerald-50 p-4 rounded-xl border border-emerald-200 flex items-start gap-3">
                    <Lightbulb className="text-emerald-600 flex-shrink-0 mt-1" size={24} />
                    <div>
                        <h4 className="font-bold text-emerald-900 mb-1">Key Insight</h4>
                        <p className="text-emerald-800 text-sm">
                            Real embeddings have hundreds or thousands of dimensions (e.g., BERT uses 768, GPT-3 uses 12,288). 
                            Each dimension captures some aspect of meaning. The examples here use 2D for visualization.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
