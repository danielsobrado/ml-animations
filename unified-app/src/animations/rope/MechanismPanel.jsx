import React, { useState } from 'react';
import { motion } from 'framer-motion';

export default function MechanismPanel() {
    const [selectedPair, setSelectedPair] = useState(0);

    // Dimension pairs for RoPE (typically d/2 pairs)
    const pairs = [
        { i: 0, freq: 1.0, label: 'Fast rotation' },
        { i: 1, freq: 0.5, label: 'Medium rotation' },
        { i: 2, freq: 0.25, label: 'Slow rotation' },
        { i: 3, freq: 0.125, label: 'Very slow rotation' },
    ];

    return (
        <div className="p-8 h-full overflow-y-auto">
            <div className="max-w-5xl mx-auto">
                {/* Header */}
                <div className="text-center mb-12">
                    <h2 className="text-3xl font-bold text-violet-600 dark:text-violet-400 mb-4">
                        RoPE Mathematics
                    </h2>
                    <p className="text-slate-600 dark:text-slate-300 max-w-2xl mx-auto">
                        RoPE applies rotation matrices to pairs of dimensions, each with a different frequency.
                        This creates a unique "fingerprint" for each position.
                    </p>
                </div>

                {/* Rotation Matrix Formula */}
                <div className="bg-slate-900 rounded-2xl p-8 mb-8 text-white">
                    <h3 className="text-lg font-bold text-violet-400 mb-4">The Rotation Matrix</h3>
                    <div className="flex flex-col md:flex-row items-center justify-center gap-8">
                        <div className="font-mono text-center">
                            <div className="text-sm text-slate-400 mb-2">For dimension pair (2i, 2i+1):</div>
                            <div className="text-xl">
                                R<sub>θ,m</sub> =
                            </div>
                        </div>
                        <div className="bg-slate-800 p-4 rounded-lg font-mono">
                            <div className="grid grid-cols-2 gap-2 text-center">
                                <div className="px-4 py-2 bg-violet-900/50 rounded">cos(mθ<sub>i</sub>)</div>
                                <div className="px-4 py-2 bg-fuchsia-900/50 rounded">-sin(mθ<sub>i</sub>)</div>
                                <div className="px-4 py-2 bg-fuchsia-900/50 rounded">sin(mθ<sub>i</sub>)</div>
                                <div className="px-4 py-2 bg-violet-900/50 rounded">cos(mθ<sub>i</sub>)</div>
                            </div>
                        </div>
                        <div className="text-center">
                            <div className="text-sm text-slate-400 mb-2">Where:</div>
                            <div className="font-mono text-violet-300">
                                θ<sub>i</sub> = 10000<sup>-2i/d</sup>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Multi-frequency Visualization */}
                <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700 mb-8">
                    <h3 className="font-bold text-slate-700 dark:text-slate-200 mb-6">
                        Different Frequencies for Different Dimension Pairs
                    </h3>

                    <div className="grid md:grid-cols-4 gap-4 mb-6">
                        {pairs.map((pair, idx) => (
                            <button
                                key={idx}
                                onClick={() => setSelectedPair(idx)}
                                className={`p-4 rounded-xl border-2 transition-all ${selectedPair === idx
                                        ? 'border-violet-500 bg-violet-50 dark:bg-violet-900/20'
                                        : 'border-slate-200 dark:border-slate-700 hover:border-violet-300'
                                    }`}
                            >
                                <div className="text-sm font-mono text-slate-500">Pair {idx}</div>
                                <div className="text-lg font-bold text-violet-600 dark:text-violet-400">
                                    d<sub>{2 * idx}</sub>, d<sub>{2 * idx + 1}</sub>
                                </div>
                                <div className="text-xs text-slate-400 mt-1">{pair.label}</div>
                            </button>
                        ))}
                    </div>

                    {/* Wave visualization */}
                    <div className="bg-slate-100 dark:bg-slate-900 rounded-xl p-4">
                        <svg viewBox="0 0 400 100" className="w-full h-24">
                            {/* Background grid */}
                            {[0, 1, 2, 3, 4, 5, 6, 7].map(i => (
                                <line
                                    key={i}
                                    x1={i * 50} y1="0"
                                    x2={i * 50} y2="100"
                                    stroke="currentColor"
                                    strokeOpacity="0.1"
                                />
                            ))}
                            <line x1="0" y1="50" x2="400" y2="50" stroke="currentColor" strokeOpacity="0.2" />

                            {/* Wave for selected frequency */}
                            <motion.path
                                d={Array.from({ length: 81 }, (_, i) => {
                                    const x = i * 5;
                                    const y = 50 - 35 * Math.sin(i * 0.2 * pairs[selectedPair].freq * Math.PI);
                                    return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                                }).join(' ')}
                                fill="none"
                                stroke="url(#waveGradient)"
                                strokeWidth="3"
                                initial={{ pathLength: 0 }}
                                animate={{ pathLength: 1 }}
                                key={selectedPair}
                            />

                            {/* Position markers */}
                            {[0, 1, 2, 3, 4, 5, 6, 7].map(pos => {
                                const y = 50 - 35 * Math.sin(pos * 10 * 0.2 * pairs[selectedPair].freq * Math.PI);
                                return (
                                    <g key={pos}>
                                        <circle cx={pos * 50} cy={y} r="4" fill="#8b5cf6" />
                                        <text x={pos * 50} y="95" textAnchor="middle" className="text-xs fill-slate-500">
                                            pos={pos}
                                        </text>
                                    </g>
                                );
                            })}

                            <defs>
                                <linearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                    <stop offset="0%" stopColor="#8b5cf6" />
                                    <stop offset="100%" stopColor="#d946ef" />
                                </linearGradient>
                            </defs>
                        </svg>
                        <div className="text-center text-sm text-slate-500 mt-2">
                            Frequency θ<sub>{selectedPair}</sub> = {pairs[selectedPair].freq.toFixed(3)} (relative)
                        </div>
                    </div>
                </div>

                {/* Implementation insight */}
                <div className="grid md:grid-cols-2 gap-6">
                    <div className="bg-slate-100 dark:bg-slate-800 rounded-xl p-6">
                        <h4 className="font-bold text-slate-700 dark:text-slate-200 mb-3">Efficient Implementation</h4>
                        <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
                            Instead of matrix multiplication, RoPE can be computed element-wise:
                        </p>
                        <div className="bg-white dark:bg-slate-900 p-3 rounded-lg font-mono text-xs overflow-x-auto">
                            <div className="text-violet-600">x'[2i] = x[2i]·cos(mθ) - x[2i+1]·sin(mθ)</div>
                            <div className="text-fuchsia-600">x'[2i+1] = x[2i]·sin(mθ) + x[2i+1]·cos(mθ)</div>
                        </div>
                    </div>

                    <div className="bg-gradient-to-r from-violet-500 to-fuchsia-500 rounded-xl p-6 text-white">
                        <h4 className="font-bold mb-3">Used In</h4>
                        <div className="flex flex-wrap gap-2">
                            {['LLaMA', 'LLaMA 2', 'LLaMA 3', 'Mistral', 'Mixtral', 'PaLM', 'Gemma', 'Falcon'].map(model => (
                                <span key={model} className="px-3 py-1 bg-white/20 rounded-full text-sm">
                                    {model}
                                </span>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
