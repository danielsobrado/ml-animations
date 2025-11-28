import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, ChevronRight, Lightbulb, ArrowRight } from 'lucide-react';

export default function QKVPanel() {
    const [step, setStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [selectedWord, setSelectedWord] = useState(0);

    // Example sentence
    const sentence = ['The', 'cat', 'sat', 'on', 'mat'];
    const embedDim = 4;

    // Simulated embeddings (simplified)
    const embeddings = [
        [0.2, 0.8, -0.3, 0.5],   // The
        [0.9, -0.2, 0.7, 0.1],   // cat
        [-0.4, 0.6, 0.2, -0.8],  // sat
        [0.1, -0.5, 0.9, 0.3],   // on
        [0.7, 0.4, -0.6, 0.2],   // mat
    ];

    // Weight matrices (simplified 4x4)
    const Wq = [[0.5, 0.2, -0.1, 0.3], [0.1, 0.6, 0.2, -0.2], [-0.3, 0.1, 0.7, 0.4], [0.2, -0.1, 0.3, 0.5]];
    const Wk = [[0.4, -0.2, 0.3, 0.1], [0.2, 0.5, -0.1, 0.3], [0.1, 0.3, 0.6, -0.2], [-0.1, 0.2, 0.2, 0.4]];
    const Wv = [[0.3, 0.1, -0.2, 0.4], [-0.1, 0.4, 0.3, 0.1], [0.2, -0.1, 0.5, 0.2], [0.1, 0.3, 0.1, 0.3]];

    // Matrix multiplication helper
    const matMul = (vec, mat) => {
        return mat[0].map((_, colIdx) => 
            vec.reduce((sum, val, rowIdx) => sum + val * mat[rowIdx][colIdx], 0)
        );
    };

    // Compute Q, K, V for a word
    const computeQKV = (wordIdx) => {
        const embed = embeddings[wordIdx];
        return {
            Q: matMul(embed, Wq).map(v => v.toFixed(2)),
            K: matMul(embed, Wk).map(v => v.toFixed(2)),
            V: matMul(embed, Wv).map(v => v.toFixed(2)),
        };
    };

    const steps = [
        {
            title: 'Word Embeddings',
            desc: 'Each word is represented as a dense vector capturing its meaning.',
            highlight: 'embedding',
        },
        {
            title: 'Query Projection (Q = X · Wq)',
            desc: '"What am I looking for?" - Transforms embedding into a query vector.',
            highlight: 'query',
        },
        {
            title: 'Key Projection (K = X · Wk)',
            desc: '"What do I contain?" - Transforms embedding into a key for matching.',
            highlight: 'key',
        },
        {
            title: 'Value Projection (V = X · Wv)',
            desc: '"What information do I provide?" - The actual content to retrieve.',
            highlight: 'value',
        },
        {
            title: 'All Three Together',
            desc: 'Each word position now has Q, K, and V vectors derived from its embedding.',
            highlight: 'all',
        },
    ];

    useEffect(() => {
        if (isPlaying && step < steps.length - 1) {
            const timer = setTimeout(() => setStep(s => s + 1), 2500);
            return () => clearTimeout(timer);
        } else if (step >= steps.length - 1) {
            setIsPlaying(false);
        }
    }, [isPlaying, step]);

    const reset = () => {
        setStep(0);
        setIsPlaying(false);
    };

    const currentQKV = computeQKV(selectedWord);

    return (
        <div className="p-6 min-h-screen">
            <div className="max-w-6xl mx-auto">
                {/* Header */}
                <div className="text-center mb-6">
                    <h2 className="text-3xl font-bold text-white mb-2">
                        Query, Key, Value: <span className="gradient-text">The Three Projections</span>
                    </h2>
                    <p className="text-slate-400 max-w-2xl mx-auto">
                        Each input embedding is transformed into three different vectors, 
                        each serving a unique purpose in the attention mechanism.
                    </p>
                </div>

                {/* Controls */}
                <div className="flex justify-center gap-4 mb-6">
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
                    >
                        {isPlaying ? <Pause size={18} /> : <Play size={18} />}
                        {isPlaying ? 'Pause' : 'Play Animation'}
                    </button>
                    <button
                        onClick={reset}
                        className="flex items-center gap-2 px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-500"
                    >
                        <RotateCcw size={18} />
                        Reset
                    </button>
                </div>

                {/* Step Progress */}
                <div className="flex justify-center gap-2 mb-6">
                    {steps.map((s, i) => (
                        <button
                            key={i}
                            onClick={() => setStep(i)}
                            className={`px-3 py-1 rounded-full text-sm transition-all ${
                                i === step
                                    ? 'bg-gradient-to-r from-blue-500 to-purple-500 text-white'
                                    : i < step
                                        ? 'bg-green-500/30 text-green-400'
                                        : 'bg-slate-700 text-slate-400'
                            }`}
                        >
                            {i + 1}
                        </button>
                    ))}
                </div>

                {/* Current Step */}
                <div className="bg-blue-500/20 rounded-xl p-4 mb-6 border border-blue-500/30 text-center">
                    <h3 className="font-bold text-blue-400">{steps[step].title}</h3>
                    <p className="text-slate-300 text-sm">{steps[step].desc}</p>
                </div>

                {/* Word Selector */}
                <div className="flex justify-center gap-2 mb-6">
                    <span className="text-slate-400 mr-2">Select word:</span>
                    {sentence.map((word, i) => (
                        <button
                            key={i}
                            onClick={() => setSelectedWord(i)}
                            className={`px-4 py-2 rounded-lg font-medium transition-all ${
                                selectedWord === i
                                    ? 'bg-white text-slate-800'
                                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                            }`}
                        >
                            {word}
                        </button>
                    ))}
                </div>

                {/* Main Visualization */}
                <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                    <div className="flex items-center justify-center gap-8">
                        {/* Input Embedding */}
                        <div className={`transition-all duration-500 ${
                            step >= 0 ? 'opacity-100 scale-100' : 'opacity-30 scale-95'
                        }`}>
                            <div className="text-center mb-2">
                                <span className="text-slate-400 text-sm">Input Embedding</span>
                                <div className="text-white font-bold">"{sentence[selectedWord]}"</div>
                            </div>
                            <div className="bg-slate-700 rounded-lg p-3">
                                <div className="grid grid-cols-4 gap-1">
                                    {embeddings[selectedWord].map((val, i) => (
                                        <div 
                                            key={i}
                                            className={`w-12 h-12 rounded flex items-center justify-center font-mono text-sm transition-all ${
                                                step >= 0 ? 'bg-slate-600 text-white' : 'bg-slate-800 text-slate-500'
                                            }`}
                                        >
                                            {val.toFixed(1)}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Arrow */}
                        <div className="flex flex-col items-center gap-2">
                            <ArrowRight className="text-slate-500" size={32} />
                            <span className="text-slate-500 text-xs">Linear Projections</span>
                        </div>

                        {/* Q, K, V Outputs */}
                        <div className="flex flex-col gap-4">
                            {/* Query */}
                            <div className={`transition-all duration-500 ${
                                step >= 1 && (step === 1 || step === 4) ? 'opacity-100 scale-100 ring-2 ring-blue-500' : 
                                step > 1 ? 'opacity-100 scale-100' : 'opacity-30 scale-95'
                            }`}>
                                <div className="flex items-center gap-3">
                                    <div className="bg-blue-500 px-3 py-1 rounded text-white font-bold text-sm">Q</div>
                                    <div className="bg-blue-500/20 rounded-lg p-2 border border-blue-500/30">
                                        <div className="flex gap-1">
                                            {currentQKV.Q.map((val, i) => (
                                                <div 
                                                    key={i}
                                                    className="w-10 h-10 bg-blue-500/30 rounded flex items-center justify-center font-mono text-xs text-blue-300"
                                                >
                                                    {val}
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                    <span className="text-blue-400 text-xs">Query</span>
                                </div>
                            </div>

                            {/* Key */}
                            <div className={`transition-all duration-500 ${
                                step >= 2 && (step === 2 || step === 4) ? 'opacity-100 scale-100 ring-2 ring-green-500' : 
                                step > 2 ? 'opacity-100 scale-100' : 'opacity-30 scale-95'
                            }`}>
                                <div className="flex items-center gap-3">
                                    <div className="bg-green-500 px-3 py-1 rounded text-white font-bold text-sm">K</div>
                                    <div className="bg-green-500/20 rounded-lg p-2 border border-green-500/30">
                                        <div className="flex gap-1">
                                            {currentQKV.K.map((val, i) => (
                                                <div 
                                                    key={i}
                                                    className="w-10 h-10 bg-green-500/30 rounded flex items-center justify-center font-mono text-xs text-green-300"
                                                >
                                                    {val}
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                    <span className="text-green-400 text-xs">Key</span>
                                </div>
                            </div>

                            {/* Value */}
                            <div className={`transition-all duration-500 ${
                                step >= 3 && (step === 3 || step === 4) ? 'opacity-100 scale-100 ring-2 ring-purple-500' : 
                                step > 3 ? 'opacity-100 scale-100' : 'opacity-30 scale-95'
                            }`}>
                                <div className="flex items-center gap-3">
                                    <div className="bg-purple-500 px-3 py-1 rounded text-white font-bold text-sm">V</div>
                                    <div className="bg-purple-500/20 rounded-lg p-2 border border-purple-500/30">
                                        <div className="flex gap-1">
                                            {currentQKV.V.map((val, i) => (
                                                <div 
                                                    key={i}
                                                    className="w-10 h-10 bg-purple-500/30 rounded flex items-center justify-center font-mono text-xs text-purple-300"
                                                >
                                                    {val}
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                    <span className="text-purple-400 text-xs">Value</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Weight Matrices Visualization */}
                <div className="mt-8 grid grid-cols-3 gap-4">
                    {[
                        { name: 'Wq', color: 'blue', matrix: Wq, desc: 'Query weights' },
                        { name: 'Wk', color: 'green', matrix: Wk, desc: 'Key weights' },
                        { name: 'Wv', color: 'purple', matrix: Wv, desc: 'Value weights' },
                    ].map(({ name, color, matrix, desc }) => (
                        <div 
                            key={name}
                            className={`bg-slate-800/50 rounded-xl p-4 border border-slate-700 transition-all ${
                                (step === 1 && name === 'Wq') ||
                                (step === 2 && name === 'Wk') ||
                                (step === 3 && name === 'Wv') ||
                                step === 4
                                    ? `ring-2 ring-${color}-500`
                                    : ''
                            }`}
                        >
                            <div className="flex items-center justify-between mb-2">
                                <span className={`bg-${color}-500 px-2 py-1 rounded text-white text-xs font-bold`}>
                                    {name}
                                </span>
                                <span className="text-slate-500 text-xs">{desc}</span>
                            </div>
                            <div className="grid grid-cols-4 gap-0.5">
                                {matrix.flat().map((val, i) => (
                                    <div 
                                        key={i}
                                        className={`w-8 h-8 rounded flex items-center justify-center font-mono text-xs
                                            ${val >= 0 ? 'bg-green-900/50 text-green-400' : 'bg-red-900/50 text-red-400'}
                                        `}
                                    >
                                        {val.toFixed(1)}
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>

                {/* Key Insights */}
                <div className="mt-8 bg-amber-500/10 rounded-xl p-6 border border-amber-500/30">
                    <h3 className="font-bold text-amber-400 mb-4 flex items-center gap-2">
                        <Lightbulb size={20} />
                        Key Insights
                    </h3>
                    <div className="grid grid-cols-2 gap-6 text-sm">
                        <div>
                            <h4 className="text-white font-medium mb-2">Why Three Different Projections?</h4>
                            <ul className="text-slate-400 space-y-1">
                                <li>• <strong className="text-blue-400">Q</strong>: Encodes what this position wants to find</li>
                                <li>• <strong className="text-green-400">K</strong>: Encodes what this position offers for matching</li>
                                <li>• <strong className="text-purple-400">V</strong>: Encodes the actual information to pass along</li>
                            </ul>
                        </div>
                        <div>
                            <h4 className="text-white font-medium mb-2">Learnable Parameters</h4>
                            <ul className="text-slate-400 space-y-1">
                                <li>• Wq, Wk, Wv are learned during training</li>
                                <li>• They learn to create useful Q/K/V representations</li>
                                <li>• Same words can have different attention in different contexts</li>
                            </ul>
                        </div>
                    </div>
                </div>

                {/* Formula */}
                <div className="mt-6 bg-slate-900 rounded-xl p-6 text-center">
                    <div className="text-slate-400 text-sm mb-2">The Q, K, V Transformations</div>
                    <div className="text-white font-mono text-lg space-y-2">
                        <div><span className="text-blue-400">Q</span> = X · W<sub>Q</sub></div>
                        <div><span className="text-green-400">K</span> = X · W<sub>K</sub></div>
                        <div><span className="text-purple-400">V</span> = X · W<sub>V</sub></div>
                    </div>
                    <div className="text-slate-500 text-xs mt-3">
                        Where X is the input embedding and W are learnable weight matrices
                    </div>
                </div>
            </div>
        </div>
    );
}
