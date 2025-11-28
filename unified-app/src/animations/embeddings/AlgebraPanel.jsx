import React, { useState } from 'react';
import { motion } from 'framer-motion';

export default function AlgebraPanel() {
    // Simplified 2D coordinates for the analogy
    const king = { x: 100, y: 300 };
    const man = { x: 50, y: 100 };
    const woman = { x: 250, y: 100 };
    const queen = { x: 300, y: 300 }; // Target

    // Current state of the operation
    // Start with King
    // Step 1: Subtract Man (King - Man)
    // Step 2: Add Woman (King - Man + Woman)
    const [step, setStep] = useState(0);

    const getResultVector = () => {
        let x = king.x;
        let y = king.y;

        if (step >= 1) {
            x -= man.x;
            y -= man.y;
        }
        if (step >= 2) {
            x += woman.x;
            y += woman.y;
        }
        return { x, y };
    };

    const result = getResultVector();

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-cyan-400 mb-4">Word Algebra</h2>
                <p className="text-lg text-slate-300 leading-relaxed">
                    Embeddings capture meaning as direction.
                    <br />
                    <span className="font-mono bg-slate-700 px-2 py-1 rounded text-cyan-300">King - Man + Woman ≈ Queen</span>
                </p>
            </div>

            <div className="flex gap-4 mb-8">
                <button
                    onClick={() => setStep(0)}
                    className={`px-6 py-2 rounded-lg font-bold transition-all ${step === 0 ? 'bg-cyan-600 text-white' : 'bg-slate-700 text-slate-400'}`}
                >
                    1. Start: King
                </button>
                <button
                    onClick={() => setStep(1)}
                    className={`px-6 py-2 rounded-lg font-bold transition-all ${step === 1 ? 'bg-cyan-600 text-white' : 'bg-slate-700 text-slate-400'}`}
                >
                    2. Subtract: Man
                </button>
                <button
                    onClick={() => setStep(2)}
                    className={`px-6 py-2 rounded-lg font-bold transition-all ${step === 2 ? 'bg-cyan-600 text-white' : 'bg-slate-700 text-slate-400'}`}
                >
                    3. Add: Woman
                </button>
            </div>

            <div className="w-full max-w-4xl h-[500px] bg-slate-900 rounded-xl border-2 border-slate-700 relative overflow-hidden shadow-inner">
                <svg className="w-full h-full" viewBox="0 0 600 400">
                    <defs>
                        <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                            <polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8" />
                        </marker>
                        <marker id="arrow-active" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                            <polygon points="0 0, 10 3.5, 0 7" fill="#22d3ee" />
                        </marker>
                    </defs>

                    {/* Grid */}
                    <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
                        <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#1e293b" strokeWidth="1" />
                    </pattern>
                    <rect width="100%" height="100%" fill="url(#grid)" />

                    {/* Reference Vectors (Faded) */}
                    <line x1="0" y1="400" x2={king.x} y2={400 - king.y} stroke="#334155" strokeWidth="2" strokeDasharray="5 5" />
                    <text x={king.x} y={400 - king.y - 10} fill="#64748b" textAnchor="middle" fontSize="12">King</text>

                    <line x1="0" y1="400" x2={man.x} y2={400 - man.y} stroke="#334155" strokeWidth="2" strokeDasharray="5 5" />
                    <text x={man.x} y={400 - man.y - 10} fill="#64748b" textAnchor="middle" fontSize="12">Man</text>

                    <line x1="0" y1="400" x2={woman.x} y2={400 - woman.y} stroke="#334155" strokeWidth="2" strokeDasharray="5 5" />
                    <text x={woman.x} y={400 - woman.y - 10} fill="#64748b" textAnchor="middle" fontSize="12">Woman</text>

                    <circle cx={queen.x} cy={400 - queen.y} r="5" fill="#a855f7" />
                    <text x={queen.x} y={400 - queen.y - 15} fill="#a855f7" textAnchor="middle" fontWeight="bold">Queen (Target)</text>

                    {/* Active Calculation Path */}
                    <motion.g
                        initial={false}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.5 }}
                    >
                        {/* Step 0: King Vector */}
                        <line
                            x1="0" y1="400"
                            x2={king.x} y2={400 - king.y}
                            stroke={step >= 0 ? "#22d3ee" : "#334155"}
                            strokeWidth="4"
                            markerEnd="url(#arrow-active)"
                        />

                        {/* Step 1: - Man Vector (Tail at King) */}
                        {step >= 1 && (
                            <line
                                x1={king.x} y1={400 - king.y}
                                x2={king.x - man.x} y2={400 - king.y + man.y}
                                stroke="#f472b6"
                                strokeWidth="4"
                                markerEnd="url(#arrow-active)"
                            />
                        )}

                        {/* Step 2: + Woman Vector (Tail at (King-Man)) */}
                        {step >= 2 && (
                            <line
                                x1={king.x - man.x} y1={400 - king.y + man.y}
                                x2={result.x} y2={400 - result.y}
                                stroke="#a855f7"
                                strokeWidth="4"
                                markerEnd="url(#arrow-active)"
                            />
                        )}
                    </motion.g>

                    {/* Result Point */}
                    <motion.circle
                        animate={{ cx: result.x, cy: 400 - result.y }}
                        r="8"
                        fill="#fbbf24"
                        stroke="#fff"
                        strokeWidth="2"
                    />
                    <motion.text
                        animate={{ x: result.x, y: 400 - result.y + 25 }}
                        fill="#fbbf24"
                        textAnchor="middle"
                        fontWeight="bold"
                    >
                        Result
                    </motion.text>

                </svg>
            </div>
        </div>
    );
}
