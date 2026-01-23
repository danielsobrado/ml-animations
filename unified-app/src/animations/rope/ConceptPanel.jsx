import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { RotateCcw, ArrowRight } from 'lucide-react';

export default function ConceptPanel() {
    const [position, setPosition] = useState(0);
    const [isAnimating, setIsAnimating] = useState(true);

    // Auto-increment position for animation
    useEffect(() => {
        if (isAnimating) {
            const interval = setInterval(() => {
                setPosition(p => (p + 1) % 8);
            }, 1500);
            return () => clearInterval(interval);
        }
    }, [isAnimating]);

    const baseAngle = 45; // Base rotation per position (degrees)

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-4xl w-full">
                {/* Header */}
                <div className="text-center mb-12">
                    <h2 className="text-3xl font-bold text-violet-600 dark:text-violet-400 mb-4">
                        The Key Insight: Position as Rotation
                    </h2>
                    <p className="text-lg text-slate-700 dark:text-slate-300 leading-relaxed max-w-2xl mx-auto">
                        Instead of <em>adding</em> position vectors (like sinusoidal encoding),
                        RoPE <strong>rotates</strong> embeddings based on their position in the sequence.
                    </p>
                </div>

                {/* Animation Controls */}
                <div className="flex justify-center gap-4 mb-8">
                    <button
                        onClick={() => setIsAnimating(!isAnimating)}
                        className={`px-4 py-2 rounded-lg font-medium ${isAnimating
                                ? 'bg-violet-500 text-white'
                                : 'bg-slate-200 dark:bg-slate-700'
                            }`}
                    >
                        {isAnimating ? 'Pause' : 'Play'}
                    </button>
                    <div className="flex items-center gap-2 px-4 py-2 bg-slate-100 dark:bg-slate-800 rounded-lg">
                        <span className="text-slate-500">Position:</span>
                        <span className="font-bold text-violet-600 text-xl">{position}</span>
                    </div>
                </div>

                {/* Main Visualization */}
                <div className="grid md:grid-cols-2 gap-8 mb-12">
                    {/* 2D Rotation Visualization */}
                    <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700">
                        <h3 className="font-bold text-slate-700 dark:text-slate-200 mb-4 flex items-center gap-2">
                            <RotateCcw className="text-violet-500" />
                            2D Vector Rotation
                        </h3>

                        <div className="relative w-full aspect-square max-w-[300px] mx-auto">
                            {/* Coordinate system */}
                            <svg viewBox="-150 -150 300 300" className="w-full h-full">
                                {/* Grid circles */}
                                <circle cx="0" cy="0" r="100" fill="none" stroke="currentColor" strokeOpacity="0.1" />
                                <circle cx="0" cy="0" r="50" fill="none" stroke="currentColor" strokeOpacity="0.1" />

                                {/* Axes */}
                                <line x1="-130" y1="0" x2="130" y2="0" stroke="currentColor" strokeOpacity="0.2" />
                                <line x1="0" y1="-130" x2="0" y2="130" stroke="currentColor" strokeOpacity="0.2" />

                                {/* Original vector (grey) */}
                                <line
                                    x1="0" y1="0"
                                    x2="100" y2="0"
                                    stroke="#94a3b8"
                                    strokeWidth="2"
                                    strokeDasharray="5,5"
                                />
                                <circle cx="100" cy="0" r="4" fill="#94a3b8" />

                                {/* Rotated vector */}
                                <motion.g
                                    animate={{ rotate: position * baseAngle }}
                                    transition={{ type: 'spring', stiffness: 100 }}
                                >
                                    <line
                                        x1="0" y1="0"
                                        x2="100" y2="0"
                                        stroke="url(#gradient)"
                                        strokeWidth="3"
                                    />
                                    <circle cx="100" cy="0" r="6" fill="#8b5cf6" />
                                </motion.g>

                                {/* Angle arc */}
                                <motion.path
                                    d={`M 30 0 A 30 30 0 ${position * baseAngle > 180 ? 1 : 0} 0 ${30 * Math.cos(-position * baseAngle * Math.PI / 180)} ${30 * Math.sin(-position * baseAngle * Math.PI / 180)}`}
                                    fill="none"
                                    stroke="#8b5cf6"
                                    strokeWidth="2"
                                    animate={{
                                        d: `M 30 0 A 30 30 0 ${position * baseAngle > 180 ? 1 : 0} 0 ${30 * Math.cos(-position * baseAngle * Math.PI / 180)} ${30 * Math.sin(-position * baseAngle * Math.PI / 180)}`
                                    }}
                                />

                                <defs>
                                    <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                        <stop offset="0%" stopColor="#8b5cf6" />
                                        <stop offset="100%" stopColor="#d946ef" />
                                    </linearGradient>
                                </defs>
                            </svg>
                        </div>

                        <div className="text-center mt-4">
                            <div className="text-sm text-slate-500">Rotation angle:</div>
                            <div className="font-mono text-xl text-violet-600">
                                {position} × θ = {position * baseAngle}°
                            </div>
                        </div>
                    </div>

                    {/* Explanation */}
                    <div className="space-y-6">
                        <div className="bg-violet-50 dark:bg-violet-900/20 rounded-xl p-6 border border-violet-200 dark:border-violet-800">
                            <h4 className="font-bold text-violet-600 dark:text-violet-400 mb-3">Why Rotation?</h4>
                            <ul className="space-y-2 text-slate-600 dark:text-slate-300 text-sm">
                                <li className="flex items-start gap-2">
                                    <span className="text-violet-500 mt-1">•</span>
                                    <span>Rotation <strong>preserves vector magnitude</strong> - no information loss</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-violet-500 mt-1">•</span>
                                    <span>The dot product Q·K naturally encodes <strong>relative position</strong></span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-violet-500 mt-1">•</span>
                                    <span>Works well with <strong>any sequence length</strong> (extrapolation)</span>
                                </li>
                            </ul>
                        </div>

                        <div className="bg-slate-100 dark:bg-slate-800 rounded-xl p-6">
                            <h4 className="font-bold text-slate-700 dark:text-slate-200 mb-3">The Magic of Relative Position</h4>
                            <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
                                When query at position <strong>m</strong> attends to key at position <strong>n</strong>:
                            </p>
                            <div className="bg-white dark:bg-slate-900 p-4 rounded-lg font-mono text-sm text-center">
                                <div className="text-violet-600">R(m)q · R(n)k = q · R(n-m)k</div>
                                <div className="text-xs text-slate-500 mt-2">
                                    The attention only depends on (n - m), not absolute positions!
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Comparison */}
                <div className="grid md:grid-cols-2 gap-6">
                    <div className="bg-slate-100 dark:bg-slate-800 rounded-xl p-6">
                        <h4 className="font-bold text-slate-600 dark:text-slate-300 mb-2">Sinusoidal (Original)</h4>
                        <p className="text-sm text-slate-500">
                            Adds position vectors: <code className="bg-slate-200 dark:bg-slate-700 px-1 rounded">x + PE(pos)</code>
                        </p>
                        <div className="mt-2 text-xs text-slate-400">
                            ❌ Absolute position only<br />
                            ❌ Limited extrapolation
                        </div>
                    </div>
                    <div className="bg-gradient-to-r from-violet-500 to-fuchsia-500 rounded-xl p-6 text-white">
                        <h4 className="font-bold mb-2">RoPE (Modern)</h4>
                        <p className="text-sm text-violet-100">
                            Rotates vectors: <code className="bg-white/20 px-1 rounded">R(pos) × x</code>
                        </p>
                        <div className="mt-2 text-xs text-violet-200">
                            ✓ Relative position encoded naturally<br />
                            ✓ Better length generalization
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
