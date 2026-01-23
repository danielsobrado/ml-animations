import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { RotateCcw, Eye } from 'lucide-react';

export default function PlaygroundPanel() {
    const [queryPos, setQueryPos] = useState(2);
    const [keyPos, setKeyPos] = useState(5);
    const [dimension, setDimension] = useState(64);
    const [showRelative, setShowRelative] = useState(true);

    const baseTheta = 10000;

    // Calculate rotation angles for different dimension pairs
    const rotations = useMemo(() => {
        const numPairs = dimension / 2;
        return Array.from({ length: Math.min(numPairs, 8) }, (_, i) => {
            const theta = Math.pow(baseTheta, -2 * i / dimension);
            const queryAngle = queryPos * theta;
            const keyAngle = keyPos * theta;
            const relativeAngle = (queryPos - keyPos) * theta;
            return {
                pair: i,
                theta,
                queryAngle: (queryAngle * 180 / Math.PI) % 360,
                keyAngle: (keyAngle * 180 / Math.PI) % 360,
                relativeAngle: (relativeAngle * 180 / Math.PI) % 360,
            };
        });
    }, [queryPos, keyPos, dimension]);

    const relativePosition = queryPos - keyPos;

    return (
        <div className="p-8 h-full overflow-y-auto">
            <div className="max-w-5xl mx-auto">
                {/* Header */}
                <div className="text-center mb-8">
                    <h2 className="text-3xl font-bold text-violet-600 dark:text-violet-400 mb-4">
                        RoPE Rotation Explorer
                    </h2>
                    <p className="text-slate-600 dark:text-slate-300">
                        See how query and key positions translate to rotations, and how relative position emerges.
                    </p>
                </div>

                {/* Controls */}
                <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700 mb-8">
                    <div className="grid md:grid-cols-3 gap-6">
                        <div>
                            <label className="flex justify-between text-sm font-medium mb-2 dark:text-slate-300">
                                <span>Query Position (m)</span>
                                <span className="font-mono text-violet-600">{queryPos}</span>
                            </label>
                            <input
                                type="range" min="0" max="15"
                                value={queryPos}
                                onChange={(e) => setQueryPos(parseInt(e.target.value))}
                                className="w-full accent-violet-500"
                            />
                        </div>
                        <div>
                            <label className="flex justify-between text-sm font-medium mb-2 dark:text-slate-300">
                                <span>Key Position (n)</span>
                                <span className="font-mono text-fuchsia-600">{keyPos}</span>
                            </label>
                            <input
                                type="range" min="0" max="15"
                                value={keyPos}
                                onChange={(e) => setKeyPos(parseInt(e.target.value))}
                                className="w-full accent-fuchsia-500"
                            />
                        </div>
                        <div>
                            <label className="flex justify-between text-sm font-medium mb-2 dark:text-slate-300">
                                <span>Embedding Dim</span>
                                <span className="font-mono text-slate-600">{dimension}</span>
                            </label>
                            <input
                                type="range" min="16" max="128" step="16"
                                value={dimension}
                                onChange={(e) => setDimension(parseInt(e.target.value))}
                                className="w-full accent-slate-500"
                            />
                        </div>
                    </div>

                    {/* Relative Position Display */}
                    <div className="mt-6 flex items-center justify-center gap-4">
                        <div className="bg-violet-100 dark:bg-violet-900/30 px-4 py-2 rounded-lg">
                            <span className="text-sm text-violet-600 dark:text-violet-400">m = {queryPos}</span>
                        </div>
                        <span className="text-slate-400">-</span>
                        <div className="bg-fuchsia-100 dark:bg-fuchsia-900/30 px-4 py-2 rounded-lg">
                            <span className="text-sm text-fuchsia-600 dark:text-fuchsia-400">n = {keyPos}</span>
                        </div>
                        <span className="text-slate-400">=</span>
                        <div className={`px-4 py-2 rounded-lg font-bold ${relativePosition > 0
                                ? 'bg-green-100 dark:bg-green-900/30 text-green-600'
                                : relativePosition < 0
                                    ? 'bg-red-100 dark:bg-red-900/30 text-red-600'
                                    : 'bg-slate-100 dark:bg-slate-700 text-slate-600'
                            }`}>
                            Δ = {relativePosition}
                        </div>
                    </div>
                </div>

                {/* Toggle View */}
                <div className="flex justify-center mb-6">
                    <button
                        onClick={() => setShowRelative(!showRelative)}
                        className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${showRelative
                                ? 'bg-violet-500 text-white'
                                : 'bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
                            }`}
                    >
                        <Eye size={18} />
                        {showRelative ? 'Showing Relative Rotation' : 'Showing Absolute Rotations'}
                    </button>
                </div>

                {/* Rotation Visualization for Each Dimension Pair */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                    {rotations.map((rot, idx) => (
                        <div
                            key={idx}
                            className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700"
                        >
                            <div className="text-xs text-slate-500 mb-2 text-center">
                                Pair {rot.pair}: d<sub>{rot.pair * 2}</sub>, d<sub>{rot.pair * 2 + 1}</sub>
                            </div>

                            {/* Mini rotation visual */}
                            <div className="relative w-full aspect-square max-w-[100px] mx-auto">
                                <svg viewBox="-55 -55 110 110" className="w-full h-full">
                                    {/* Circle */}
                                    <circle cx="0" cy="0" r="40" fill="none" stroke="currentColor" strokeOpacity="0.1" />

                                    {showRelative ? (
                                        /* Relative rotation only */
                                        <motion.g
                                            animate={{ rotate: rot.relativeAngle }}
                                            transition={{ type: 'spring', stiffness: 80 }}
                                        >
                                            <line x1="0" y1="0" x2="40" y2="0" stroke="#10b981" strokeWidth="3" />
                                            <circle cx="40" cy="0" r="4" fill="#10b981" />
                                        </motion.g>
                                    ) : (
                                        <>
                                            {/* Query rotation */}
                                            <motion.g
                                                animate={{ rotate: rot.queryAngle }}
                                                transition={{ type: 'spring', stiffness: 80 }}
                                            >
                                                <line x1="0" y1="0" x2="40" y2="0" stroke="#8b5cf6" strokeWidth="2" />
                                                <circle cx="40" cy="0" r="3" fill="#8b5cf6" />
                                            </motion.g>

                                            {/* Key rotation */}
                                            <motion.g
                                                animate={{ rotate: rot.keyAngle }}
                                                transition={{ type: 'spring', stiffness: 80 }}
                                            >
                                                <line x1="0" y1="0" x2="35" y2="0" stroke="#d946ef" strokeWidth="2" strokeDasharray="4,2" />
                                                <circle cx="35" cy="0" r="3" fill="#d946ef" />
                                            </motion.g>
                                        </>
                                    )}
                                </svg>
                            </div>

                            <div className="text-center mt-2">
                                {showRelative ? (
                                    <div className="text-xs font-mono text-green-600">
                                        Δθ = {rot.relativeAngle.toFixed(1)}°
                                    </div>
                                ) : (
                                    <div className="text-xs font-mono">
                                        <span className="text-violet-600">Q: {rot.queryAngle.toFixed(1)}°</span>
                                        <span className="text-slate-400 mx-1">|</span>
                                        <span className="text-fuchsia-600">K: {rot.keyAngle.toFixed(1)}°</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                </div>

                {/* Key Insight */}
                <div className="bg-gradient-to-r from-violet-500 to-fuchsia-500 rounded-2xl p-6 text-white">
                    <div className="flex items-start gap-4">
                        <RotateCcw size={32} className="flex-shrink-0 mt-1" />
                        <div>
                            <h4 className="font-bold text-lg mb-2">The Key Insight</h4>
                            <p className="text-violet-100">
                                Notice how each dimension pair rotates at a different speed. Low-indexed pairs rotate
                                fast (capturing local patterns), while high-indexed pairs rotate slowly (capturing
                                global/long-range patterns). This is similar to how different frequencies in
                                sinusoidal encoding work, but with the added benefit of <strong>naturally encoding
                                    relative positions</strong> through the dot product.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
