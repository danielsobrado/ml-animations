import React, { useState } from 'react';
import { motion } from 'framer-motion';

export default function DotProductPanel() {
    const [v1, setV1] = useState({ x: 150, y: 100 });
    const [v2, setV2] = useState({ x: 100, y: 150 });

    // Calculate dot product
    const dotProduct = v1.x * v2.x + v1.y * v2.y;

    // Calculate magnitudes
    const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
    const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);

    // Calculate cosine similarity
    const cosineSim = dotProduct / (mag1 * mag2);

    // Calculate angle in degrees
    const angleRad = Math.acos(Math.max(-1, Math.min(1, cosineSim)));
    const angleDeg = (angleRad * 180 / Math.PI).toFixed(1);

    // Projection of v1 onto v2
    const projScalar = dotProduct / (mag2 * mag2);
    const proj = { x: projScalar * v2.x, y: projScalar * v2.y };

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-cyan-400 mb-4">The Dot Product</h2>
                <p className="text-lg text-slate-300 leading-relaxed">
                    Cosine Similarity is just the Dot Product, normalized by magnitude.
                    <br />
                    <span className="font-mono bg-slate-800 px-3 py-1 rounded text-purple-300">
                        cos(θ) = (A·B) / (|A||B|)
                    </span>
                </p>
            </div>

            <div className="flex flex-col lg:flex-row gap-8 w-full max-w-6xl">
                {/* Visualization */}
                <div className="flex-1 h-[500px] bg-slate-800 rounded-xl border-2 border-slate-700 relative">
                    <svg className="w-full h-full" viewBox="-250 -250 500 500">
                        <defs>
                            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                                <polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8" />
                            </marker>
                            <marker id="arrowhead-cyan" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                                <polygon points="0 0, 10 3.5, 0 7" fill="#22d3ee" />
                            </marker>
                            <marker id="arrowhead-pink" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                                <polygon points="0 0, 10 3.5, 0 7" fill="#ec4899" />
                            </marker>
                        </defs>

                        {/* Grid */}
                        <line x1="-250" y1="0" x2="250" y2="0" stroke="#334155" strokeWidth="1" strokeDasharray="5 5" />
                        <line x1="0" y1="-250" x2="0" y2="250" stroke="#334155" strokeWidth="1" strokeDasharray="5 5" />

                        {/* Vector 2 (Pink) */}
                        <motion.line
                            x1="0" y1="0"
                            animate={{ x2: v2.x, y2: -v2.y }}
                            stroke="#ec4899"
                            strokeWidth="4"
                            markerEnd="url(#arrowhead-pink)"
                            transition={{ type: "spring", stiffness: 100 }}
                        />
                        <text x={v2.x + 15} y={-v2.y - 10} fill="#ec4899" fontWeight="bold">Vector B</text>

                        {/* Projection (Dashed) */}
                        <motion.line
                            x1="0" y1="0"
                            animate={{ x2: proj.x, y2: -proj.y }}
                            stroke="#fbbf24"
                            strokeWidth="3"
                            strokeDasharray="5 5"
                            transition={{ type: "spring", stiffness: 100 }}
                        />

                        {/* Projection Drop Line */}
                        <motion.line
                            animate={{
                                x1: v1.x, y1: -v1.y,
                                x2: proj.x, y2: -proj.y
                            }}
                            stroke="#64748b"
                            strokeWidth="1"
                            strokeDasharray="2 2"
                            transition={{ type: "spring", stiffness: 100 }}
                        />

                        {/* Vector 1 (Cyan) */}
                        <motion.line
                            x1="0" y1="0"
                            animate={{ x2: v1.x, y2: -v1.y }}
                            stroke="#22d3ee"
                            strokeWidth="4"
                            markerEnd="url(#arrowhead-cyan)"
                            transition={{ type: "spring", stiffness: 100 }}
                        />
                        <text x={v1.x + 15} y={-v1.y + 5} fill="#22d3ee" fontWeight="bold">Vector A</text>

                        {/* Angle Arc */}
                        <path
                            d={`M 30 0 A 30 30 0 0 0 ${30 * Math.cos(-angleRad * Math.PI / 180)} ${30 * Math.sin(-angleRad * Math.PI / 180)}`}
                            fill="none"
                            stroke="#a855f7"
                            strokeWidth="2"
                        />
                        <text x="40" y="-20" fill="#a855f7" fontSize="14">{angleDeg}°</text>
                    </svg>

                    {/* Instructions */}
                    <div className="absolute bottom-4 left-4 bg-slate-900/80 backdrop-blur-sm px-4 py-2 rounded-lg text-xs text-slate-400">
                        Drag the sliders to change the vectors →
                    </div>
                </div>

                {/* Controls & Formula */}
                <div className="w-full lg:w-96 flex flex-col gap-6">
                    {/* Vector Controls */}
                    <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                        <h3 className="font-bold text-slate-400 uppercase text-xs mb-4">Vector A (Cyan)</h3>
                        <div className="space-y-4">
                            <div>
                                <label className="flex justify-between text-sm mb-2">
                                    X: <span className="text-cyan-400 font-mono">{v1.x}</span>
                                </label>
                                <input
                                    type="range" min="-200" max="200" step="10"
                                    value={v1.x}
                                    onChange={(e) => setV1({ ...v1, x: Number(e.target.value) })}
                                    className="w-full accent-cyan-400"
                                />
                            </div>
                            <div>
                                <label className="flex justify-between text-sm mb-2">
                                    Y: <span className="text-cyan-400 font-mono">{v1.y}</span>
                                </label>
                                <input
                                    type="range" min="-200" max="200" step="10"
                                    value={v1.y}
                                    onChange={(e) => setV1({ ...v1, y: Number(e.target.value) })}
                                    className="w-full accent-cyan-400"
                                />
                            </div>
                        </div>
                    </div>

                    <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                        <h3 className="font-bold text-slate-400 uppercase text-xs mb-4">Vector B (Pink)</h3>
                        <div className="space-y-4">
                            <div>
                                <label className="flex justify-between text-sm mb-2">
                                    X: <span className="text-pink-400 font-mono">{v2.x}</span>
                                </label>
                                <input
                                    type="range" min="-200" max="200" step="10"
                                    value={v2.x}
                                    onChange={(e) => setV2({ ...v2, x: Number(e.target.value) })}
                                    className="w-full accent-pink-400"
                                />
                            </div>
                            <div>
                                <label className="flex justify-between text-sm mb-2">
                                    Y: <span className="text-pink-400 font-mono">{v2.y}</span>
                                </label>
                                <input
                                    type="range" min="-200" max="200" step="10"
                                    value={v2.y}
                                    onChange={(e) => setV2({ ...v2, y: Number(e.target.value) })}
                                    className="w-full accent-pink-400"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Results */}
                    <div className="bg-gradient-to-br from-purple-900/50 to-pink-900/50 p-6 rounded-xl border border-purple-700">
                        <div className="text-center">
                            <h3 className="text-sm uppercase tracking-wider text-purple-300 mb-3">Cosine Similarity</h3>
                            <div className="text-5xl font-mono font-bold text-purple-200 mb-2">
                                {cosineSim.toFixed(3)}
                            </div>
                            <div className="text-xs text-slate-400 space-y-1">
                                <p>Dot Product: <span className="text-white font-mono">{dotProduct.toFixed(0)}</span></p>
                                <p>|A| × |B|: <span className="text-white font-mono">{(mag1 * mag2).toFixed(0)}</span></p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
