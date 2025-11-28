import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, ChevronRight, Lightbulb, ArrowDown } from 'lucide-react';

export default function ScaledDotProductPanel() {
    const [step, setStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [queryIdx, setQueryIdx] = useState(1); // Focus on "cat"

    const sentence = ['The', 'cat', 'sat', 'on', 'mat'];
    const n = sentence.length;
    const dk = 4; // dimension

    // Simulated Q, K, V matrices (each word has a 4-dim vector)
    const Q = [
        [0.5, 0.3, -0.2, 0.4],
        [0.8, -0.1, 0.6, 0.2],   // cat - our query
        [-0.3, 0.5, 0.1, -0.6],
        [0.1, -0.4, 0.7, 0.3],
        [0.6, 0.2, -0.5, 0.1],
    ];

    const K = [
        [0.4, 0.2, -0.3, 0.5],
        [0.7, 0.0, 0.5, 0.1],
        [-0.2, 0.6, 0.2, -0.7],
        [0.2, -0.3, 0.8, 0.2],
        [0.5, 0.3, -0.4, 0.2],
    ];

    const V = [
        [0.3, 0.4, -0.1, 0.6],
        [0.9, -0.2, 0.4, 0.3],
        [-0.1, 0.7, 0.3, -0.5],
        [0.2, -0.1, 0.9, 0.4],
        [0.4, 0.5, -0.3, 0.2],
    ];

    // Compute dot product
    const dotProduct = (a, b) => a.reduce((sum, val, i) => sum + val * b[i], 0);

    // Compute attention scores for query position
    const computeScores = (qIdx) => {
        const q = Q[qIdx];
        return K.map(k => dotProduct(q, k));
    };

    // Scale scores
    const scaleScores = (scores) => scores.map(s => s / Math.sqrt(dk));

    // Softmax
    const softmax = (scores) => {
        const maxScore = Math.max(...scores);
        const expScores = scores.map(s => Math.exp(s - maxScore));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        return expScores.map(e => e / sumExp);
    };

    // Compute weighted sum of values
    const weightedSum = (weights, values) => {
        const result = new Array(values[0].length).fill(0);
        weights.forEach((w, i) => {
            values[i].forEach((v, j) => {
                result[j] += w * v;
            });
        });
        return result;
    };

    const rawScores = computeScores(queryIdx);
    const scaledScores = scaleScores(rawScores);
    const attentionWeights = softmax(scaledScores);
    const output = weightedSum(attentionWeights, V);

    const steps = [
        {
            title: 'Step 1: Compute Q·K Scores',
            formula: 'score(i,j) = Q_i · K_j^T',
            desc: 'Take the dot product between the query vector and each key vector.',
        },
        {
            title: 'Step 2: Scale by √d_k',
            formula: 'scaled_score = score / √d_k',
            desc: 'Divide by √d_k to prevent dot products from getting too large (prevents vanishing gradients in softmax).',
        },
        {
            title: 'Step 3: Apply Softmax',
            formula: 'attention_weight = softmax(scaled_score)',
            desc: 'Convert scores to probabilities that sum to 1. Higher scores → higher weights.',
        },
        {
            title: 'Step 4: Weighted Sum of Values',
            formula: 'output = Σ (weight_i × V_i)',
            desc: 'Multiply each value vector by its attention weight and sum them all.',
        },
        {
            title: 'Final Output',
            formula: 'Attention(Q,K,V) = softmax(QK^T/√d_k)V',
            desc: 'The output is a weighted combination of all values, based on query-key similarity!',
        },
    ];

    useEffect(() => {
        if (isPlaying && step < steps.length - 1) {
            const timer = setTimeout(() => setStep(s => s + 1), 3000);
            return () => clearTimeout(timer);
        } else if (step >= steps.length - 1) {
            setIsPlaying(false);
        }
    }, [isPlaying, step]);

    const reset = () => {
        setStep(0);
        setIsPlaying(false);
    };

    const getWeightColor = (weight) => {
        if (weight >= 0.3) return 'bg-green-500 text-white';
        if (weight >= 0.15) return 'bg-yellow-500 text-slate-800';
        return 'bg-slate-600 text-slate-300';
    };

    return (
        <div className="p-6 min-h-screen">
            <div className="max-w-6xl mx-auto">
                {/* Header */}
                <div className="text-center mb-6">
                    <h2 className="text-3xl font-bold text-white mb-2">
                        Scaled Dot-Product Attention: <span className="gradient-text">Step by Step</span>
                    </h2>
                    <p className="text-slate-400 max-w-2xl mx-auto">
                        The heart of the attention mechanism - how queries find and retrieve relevant information.
                    </p>
                </div>

                {/* Controls */}
                <div className="flex justify-center gap-4 mb-6">
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className="flex items-center gap-2 px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600"
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
                                    ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
                                    : i < step
                                        ? 'bg-green-500/30 text-green-400'
                                        : 'bg-slate-700 text-slate-400'
                            }`}
                        >
                            {i + 1}
                        </button>
                    ))}
                </div>

                {/* Current Step Info */}
                <div className="bg-purple-500/20 rounded-xl p-4 mb-6 border border-purple-500/30 text-center">
                    <h3 className="font-bold text-purple-400 text-lg">{steps[step].title}</h3>
                    <div className="font-mono text-white my-2 text-lg">{steps[step].formula}</div>
                    <p className="text-slate-300 text-sm">{steps[step].desc}</p>
                </div>

                {/* Query Selector */}
                <div className="flex justify-center items-center gap-2 mb-6">
                    <span className="text-slate-400">Computing attention for:</span>
                    {sentence.map((word, i) => (
                        <button
                            key={i}
                            onClick={() => setQueryIdx(i)}
                            className={`px-3 py-1 rounded-lg font-medium transition-all ${
                                queryIdx === i
                                    ? 'bg-blue-500 text-white'
                                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                            }`}
                        >
                            {word}
                        </button>
                    ))}
                </div>

                {/* Main Visualization */}
                <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                    <div className="grid grid-cols-5 gap-4">
                        {/* Step 1: Raw Scores */}
                        <div className={`space-y-2 transition-all ${step >= 0 ? 'opacity-100' : 'opacity-30'}`}>
                            <h4 className="text-center text-slate-400 text-sm font-medium">Q·K Scores</h4>
                            {sentence.map((word, i) => (
                                <div key={i} className="flex items-center gap-2">
                                    <span className={`text-xs ${i === queryIdx ? 'text-blue-400 font-bold' : 'text-slate-500'}`}>
                                        {word}
                                    </span>
                                    <div className={`flex-1 h-8 rounded flex items-center justify-center font-mono text-sm ${
                                        step >= 0 ? 'bg-slate-600 text-white' : 'bg-slate-700 text-slate-500'
                                    }`}>
                                        {rawScores[i].toFixed(2)}
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Arrow */}
                        <div className="flex items-center justify-center">
                            <div className={`text-center ${step >= 1 ? 'text-white' : 'text-slate-600'}`}>
                                <ChevronRight size={24} />
                                <div className="text-xs">÷√{dk}</div>
                            </div>
                        </div>

                        {/* Step 2: Scaled Scores */}
                        <div className={`space-y-2 transition-all ${step >= 1 ? 'opacity-100' : 'opacity-30'}`}>
                            <h4 className="text-center text-slate-400 text-sm font-medium">Scaled</h4>
                            {sentence.map((word, i) => (
                                <div key={i} className="flex items-center gap-2">
                                    <div className={`flex-1 h-8 rounded flex items-center justify-center font-mono text-sm ${
                                        step >= 1 ? 'bg-orange-500/30 text-orange-300' : 'bg-slate-700 text-slate-500'
                                    }`}>
                                        {scaledScores[i].toFixed(2)}
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Arrow */}
                        <div className="flex items-center justify-center">
                            <div className={`text-center ${step >= 2 ? 'text-white' : 'text-slate-600'}`}>
                                <ChevronRight size={24} />
                                <div className="text-xs">softmax</div>
                            </div>
                        </div>

                        {/* Step 3: Attention Weights */}
                        <div className={`space-y-2 transition-all ${step >= 2 ? 'opacity-100' : 'opacity-30'}`}>
                            <h4 className="text-center text-slate-400 text-sm font-medium">Weights</h4>
                            {sentence.map((word, i) => (
                                <div key={i} className="flex items-center gap-2">
                                    <div className={`flex-1 h-8 rounded flex items-center justify-center font-mono text-sm transition-all ${
                                        step >= 2 ? getWeightColor(attentionWeights[i]) : 'bg-slate-700 text-slate-500'
                                    }`}>
                                        {(attentionWeights[i] * 100).toFixed(1)}%
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Step 4: Value Aggregation */}
                    {step >= 3 && (
                        <div className="mt-8 pt-6 border-t border-slate-600 animate-fadeIn">
                            <h4 className="text-center text-white font-bold mb-4">
                                Weighted Sum of Values
                            </h4>
                            <div className="flex items-center justify-center gap-4 flex-wrap">
                                {sentence.map((word, i) => (
                                    <div key={i} className="text-center">
                                        <div className={`text-xs mb-1 ${getWeightColor(attentionWeights[i])} px-2 py-0.5 rounded`}>
                                            {(attentionWeights[i] * 100).toFixed(0)}%
                                        </div>
                                        <div className="text-slate-400 text-xs">×</div>
                                        <div className="bg-purple-500/30 px-2 py-1 rounded text-purple-300 text-xs">
                                            V<sub>{word}</sub>
                                        </div>
                                    </div>
                                ))}
                            </div>
                            
                            <div className="flex justify-center mt-4">
                                <ArrowDown className="text-slate-400" size={24} />
                            </div>

                            <div className="flex justify-center mt-2">
                                <div className="bg-gradient-to-r from-purple-500 to-pink-500 px-6 py-3 rounded-xl">
                                    <div className="text-white text-xs mb-1 text-center">Output for "{sentence[queryIdx]}"</div>
                                    <div className="flex gap-2">
                                        {output.map((val, i) => (
                                            <div key={i} className="w-12 h-10 bg-white/20 rounded flex items-center justify-center font-mono text-white text-sm">
                                                {val.toFixed(2)}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Attention Heatmap */}
                <div className="mt-8 bg-slate-800/50 rounded-xl p-6 border border-slate-700">
                    <h4 className="text-white font-bold mb-4 text-center">Full Attention Matrix</h4>
                    <p className="text-slate-400 text-sm text-center mb-4">
                        Each row shows how much one word attends to all other words
                    </p>
                    
                    <div className="overflow-x-auto">
                        <div className="inline-block">
                            <div className="flex gap-1 mb-1 ml-16">
                                {sentence.map((word, i) => (
                                    <div key={i} className="w-14 text-center text-slate-400 text-xs">
                                        {word}
                                    </div>
                                ))}
                            </div>
                            {sentence.map((word, qIdx) => {
                                const scores = scaleScores(computeScores(qIdx));
                                const weights = softmax(scores);
                                return (
                                    <div key={qIdx} className="flex items-center gap-1 mb-1">
                                        <div className={`w-14 text-right pr-2 text-xs ${qIdx === queryIdx ? 'text-blue-400 font-bold' : 'text-slate-400'}`}>
                                            {word}
                                        </div>
                                        {weights.map((w, kIdx) => (
                                            <div
                                                key={kIdx}
                                                className={`w-14 h-10 rounded flex items-center justify-center font-mono text-xs transition-all ${
                                                    qIdx === queryIdx && step >= 2 ? getWeightColor(w) : 
                                                    `bg-opacity-${Math.round(w * 100)} bg-green-500`
                                                }`}
                                                style={{
                                                    backgroundColor: qIdx === queryIdx && step >= 2 
                                                        ? undefined 
                                                        : `rgba(34, 197, 94, ${w})`,
                                                    color: w > 0.25 ? 'white' : '#94a3b8'
                                                }}
                                            >
                                                {(w * 100).toFixed(0)}%
                                            </div>
                                        ))}
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </div>

                {/* Why Scale? */}
                <div className="mt-8 bg-amber-500/10 rounded-xl p-6 border border-amber-500/30">
                    <h4 className="font-bold text-amber-400 mb-3 flex items-center gap-2">
                        <Lightbulb size={20} />
                        Why Scale by √d_k?
                    </h4>
                    <p className="text-slate-300 text-sm mb-3">
                        When the dimension d<sub>k</sub> is large, dot products can become very large in magnitude. 
                        This pushes the softmax into regions with extremely small gradients (saturation).
                    </p>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                        <div className="bg-red-500/20 p-3 rounded-lg border border-red-500/30">
                            <div className="text-red-400 font-medium mb-1">Without Scaling</div>
                            <p className="text-slate-400">Dot products grow as O(d_k). Softmax saturates → gradients vanish.</p>
                        </div>
                        <div className="bg-green-500/20 p-3 rounded-lg border border-green-500/30">
                            <div className="text-green-400 font-medium mb-1">With Scaling</div>
                            <p className="text-slate-400">Division by √d_k keeps variance ~1. Healthy gradients throughout!</p>
                        </div>
                    </div>
                </div>

                {/* Complete Formula */}
                <div className="mt-6 bg-slate-900 rounded-xl p-6 text-center">
                    <div className="text-slate-400 text-sm mb-2">The Complete Scaled Dot-Product Attention</div>
                    <div className="text-2xl font-mono text-white">
                        Attention(Q,K,V) = softmax(<span className="text-blue-400">QK<sup>T</sup></span>/<span className="text-orange-400">√d<sub>k</sub></span>)<span className="text-purple-400">V</span>
                    </div>
                </div>
            </div>

            <style jsx>{`
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(20px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                .animate-fadeIn {
                    animation: fadeIn 0.5s ease-out forwards;
                }
            `}</style>
        </div>
    );
}
