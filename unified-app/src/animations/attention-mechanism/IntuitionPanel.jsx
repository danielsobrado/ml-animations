import React, { useState, useEffect, useRef } from 'react';
import { Search, MessageSquare, ArrowRight, Lightbulb, Play, Pause, RotateCcw } from 'lucide-react';

export default function IntuitionPanel() {
    const [scenario, setScenario] = useState('library');
    const [isAnimating, setIsAnimating] = useState(false);
    const [highlightedWord, setHighlightedWord] = useState(null);
    const [attentionWeights, setAttentionWeights] = useState({});

    const scenarios = {
        library: {
            title: '📚 Library Search',
            description: 'You walk into a library looking for books about "machine learning"',
            query: 'machine learning',
            items: [
                { id: 1, text: 'Neural Networks', relevance: 0.9 },
                { id: 2, text: 'Python Basics', relevance: 0.4 },
                { id: 3, text: 'Deep Learning', relevance: 0.95 },
                { id: 4, text: 'Cooking Recipes', relevance: 0.05 },
                { id: 5, text: 'AI Fundamentals', relevance: 0.85 },
                { id: 6, text: 'Romance Novels', relevance: 0.02 },
            ],
            analogy: 'Your QUERY is what you\'re looking for. Each book\'s TITLE is a KEY. The book\'s CONTENT is the VALUE.',
        },
        translation: {
            title: '🌍 Translation',
            description: 'Translating "The cat sat on the mat" to French',
            query: 'the',
            items: [
                { id: 1, text: 'The', relevance: 0.7 },
                { id: 2, text: 'cat', relevance: 0.1 },
                { id: 3, text: 'sat', relevance: 0.05 },
                { id: 4, text: 'on', relevance: 0.05 },
                { id: 5, text: 'the', relevance: 0.05 },
                { id: 6, text: 'mat', relevance: 0.05 },
            ],
            analogy: 'When generating "Le", the model attends strongly to "The" because they\'re directly related.',
        },
        conversation: {
            title: '💬 Conversation',
            description: '"The movie was great but the ending was terrible"',
            query: 'sentiment',
            items: [
                { id: 1, text: 'movie', relevance: 0.3 },
                { id: 2, text: 'great', relevance: 0.4 },
                { id: 3, text: 'but', relevance: 0.5 },
                { id: 4, text: 'ending', relevance: 0.4 },
                { id: 5, text: 'terrible', relevance: 0.9 },
            ],
            analogy: 'The model pays attention to "terrible" and "but" to understand the overall sentiment is mixed/negative.',
        },
    };

    const currentScenario = scenarios[scenario];

    useEffect(() => {
        if (isAnimating) {
            let idx = 0;
            const weights = {};
            const interval = setInterval(() => {
                if (idx < currentScenario.items.length) {
                    const item = currentScenario.items[idx];
                    weights[item.id] = item.relevance;
                    setAttentionWeights({ ...weights });
                    setHighlightedWord(item.id);
                    idx++;
                } else {
                    setIsAnimating(false);
                }
            }, 600);
            return () => clearInterval(interval);
        }
    }, [isAnimating, scenario]);

    const startAnimation = () => {
        setAttentionWeights({});
        setHighlightedWord(null);
        setIsAnimating(true);
    };

    const getColorIntensity = (relevance) => {
        if (relevance >= 0.8) return 'bg-green-500 text-white shadow-lg shadow-green-500/50';
        if (relevance >= 0.5) return 'bg-yellow-400 text-slate-800';
        if (relevance >= 0.2) return 'bg-orange-300 text-slate-800';
        return 'bg-slate-600 dark:bg-slate-600 text-slate-800 dark:text-slate-300';
    };

    return (
        <div className="p-6">
            <div className="max-w-5xl mx-auto">
                {/* Introduction */}
                <div className="text-center mb-8">
                    <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-4">
                        What is Attention? <span className="text-gradient">The Core Intuition</span>
                    </h2>
                    <p className="text-lg text-slate-800 dark:text-slate-300 max-w-2xl mx-auto leading-relaxed">
                        Attention is like a <strong className="text-amber-500">spotlight</strong> that helps the model 
                        focus on the most relevant parts of the input when producing each part of the output.
                    </p>
                </div>

                {/* Key Analogy Box */}
                <div className="bg-gradient-to-r from-amber-500/20 to-orange-500/20 rounded-2xl p-6 mb-8 border border-amber-500/30">
                    <div className="flex items-start gap-4">
                        <div className="bg-amber-500 p-3 rounded-xl">
                            <Lightbulb className="text-white" size={24} />
                        </div>
                        <div>
                            <h3 className="text-xl font-bold text-amber-500 mb-2">The Database Analogy</h3>
                            <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
                                Think of Attention like a <strong>fuzzy database lookup</strong>:
                            </p>
                            <ul className="mt-3 space-y-2 text-slate-700 dark:text-slate-300">
                                <li className="flex items-center gap-2">
                                    <span className="bg-blue-500 px-2 py-1 rounded text-xs font-bold text-white">Query (Q)</span>
                                    <span>What you're looking for</span>
                                </li>
                                <li className="flex items-center gap-2">
                                    <span className="bg-green-500 px-2 py-1 rounded text-xs font-bold text-white">Key (K)</span>
                                    <span>Labels/titles of all available items</span>
                                </li>
                                <li className="flex items-center gap-2">
                                    <span className="bg-purple-500 px-2 py-1 rounded text-xs font-bold text-white">Value (V)</span>
                                    <span>The actual content to retrieve</span>
                                </li>
                            </ul>
                            <p className="mt-3 text-slate-700 dark:text-sm italic">
                                Unlike a hard lookup that returns one exact match, attention returns a <strong className="text-slate-900 dark:text-white">weighted combination</strong> of all values!
                            </p>
                        </div>
                    </div>
                </div>

                {/* Scenario Selector */}
                <div className="flex flex-wrap justify-center gap-3 mb-6">
                    {Object.entries(scenarios).map(([key, s]) => (
                        <button
                            key={key}
                            onClick={() => {
                                setScenario(key);
                                setAttentionWeights({});
                                setHighlightedWord(null);
                                setIsAnimating(false);
                            }}
                            className={`px-4 py-2 rounded-lg font-medium transition-all ${
                                scenario === key
                                    ? 'bg-gradient-to-r from-blue-500 to-purple-500 text-white'
                                    : 'bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-600'
                            }`}
                        >
                            {s.title}
                        </button>
                    ))}
                </div>

                {/* Interactive Demo */}
                <div className="card p-6">
                    <div className="flex flex-wrap justify-between items-center gap-4 mb-6">
                        <div>
                            <h3 className="text-xl font-bold text-slate-900 dark:text-white">{currentScenario.title}</h3>
                            <p className="text-slate-800 dark:text-slate-400">{currentScenario.description}</p>
                        </div>
                        <button
                            onClick={startAnimation}
                            disabled={isAnimating}
                            className="btn-primary flex items-center gap-2"
                        >
                            {isAnimating ? <Pause size={18} /> : <Play size={18} />}
                            {isAnimating ? 'Computing...' : 'Run Attention'}
                        </button>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        {/* Query Side */}
                        <div className="space-y-4">
                            <div className="flex items-center gap-2 text-blue-500">
                                <Search size={20} />
                                <span className="font-bold">Query: "{currentScenario.query}"</span>
                            </div>
                            <div className="bg-blue-500/10 dark:bg-blue-500/20 p-4 rounded-xl border border-blue-500/30">
                                <p className="text-blue-700 dark:text-sm">
                                    The query determines what we're looking for. 
                                    It will be compared against all keys to compute attention scores.
                                </p>
                            </div>
                        </div>

                        {/* Keys/Values Side */}
                        <div className="space-y-4">
                            <div className="flex items-center gap-2 text-green-500">
                                <MessageSquare size={20} />
                                <span className="font-bold">Keys & Values</span>
                            </div>
                            <div className="space-y-2">
                                {currentScenario.items.map((item) => (
                                    <div 
                                        key={item.id}
                                        className={`flex items-center justify-between p-3 rounded-lg transition-all duration-300 ${
                                            highlightedWord === item.id ? 'ring-2 ring-blue-500' : ''
                                        } ${
                                            attentionWeights[item.id] !== undefined
                                                ? getColorIntensity(attentionWeights[item.id])
                                                : 'bg-slate-100 dark:bg-slate-700'
                                        }`}
                                    >
                                        <span className="font-medium">{item.text}</span>
                                        {attentionWeights[item.id] !== undefined && (
                                            <span className="text-sm font-mono">
                                                {(attentionWeights[item.id] * 100).toFixed(0)}%
                                            </span>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Result */}
                    {Object.keys(attentionWeights).length === currentScenario.items.length && (
                        <div className="mt-6 p-4 bg-purple-500/10 dark:bg-purple-500/20 rounded-xl border border-purple-500/30 animate-fade-in">
                            <div className="flex items-start gap-3">
                                <div className="bg-purple-500 p-2 rounded-lg">
                                    <ArrowRight className="text-white" size={20} />
                                </div>
                                <div>
                                    <h4 className="font-bold text-purple-600 dark:text-purple-400 mb-1">Result: Weighted Combination</h4>
                                    <p className="text-slate-700 dark:text-sm">
                                        {currentScenario.analogy}
                                    </p>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Key Equations Preview */}
                <div className="mt-8 grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div className="card p-4 text-center">
                        <div className="text-3xl mb-2">🎯</div>
                        <h4 className="font-bold text-slate-900 dark:text-white mb-1">Step 1</h4>
                        <p className="text-slate-800 dark:text-sm">Compute Q·K similarity scores</p>
                    </div>
                    <div className="card p-4 text-center">
                        <div className="text-3xl mb-2">📊</div>
                        <h4 className="font-bold text-slate-900 dark:text-white mb-1">Step 2</h4>
                        <p className="text-slate-800 dark:text-sm">Apply softmax for weights</p>
                    </div>
                    <div className="card p-4 text-center">
                        <div className="text-3xl mb-2">✨</div>
                        <h4 className="font-bold text-slate-900 dark:text-white mb-1">Step 3</h4>
                        <p className="text-slate-800 dark:text-sm">Weighted sum of Values</p>
                    </div>
                </div>

                {/* Why Attention Matters */}
                <div className="mt-8 bg-gradient-to-r from-green-500/10 to-emerald-500/10 rounded-2xl p-6 border border-green-500/20">
                    <h3 className="text-xl font-bold text-green-600 dark:text-green-400 mb-4">💡 Why is Attention Revolutionary?</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-2">
                            <h4 className="font-medium text-slate-900 dark:text-white">Before Attention (RNNs):</h4>
                            <ul className="text-slate-800 dark:text-sm space-y-1">
                                <li>• Information flows sequentially</li>
                                <li>• Long-range dependencies get "forgotten"</li>
                                <li>• Fixed-size hidden state bottleneck</li>
                                <li>• Can't parallelize training</li>
                            </ul>
                        </div>
                        <div className="space-y-2">
                            <h4 className="font-medium text-slate-900 dark:text-white">With Attention:</h4>
                            <ul className="text-green-600 dark:text-sm space-y-1">
                                <li>✓ Direct access to any position</li>
                                <li>✓ No distance limit for dependencies</li>
                                <li>✓ Dynamic, input-dependent connections</li>
                                <li>✓ Fully parallelizable!</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
