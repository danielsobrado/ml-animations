import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function IterativePanel({ nodes, links }) {
    const [ranks, setRanks] = useState({});
    const [iteration, setIteration] = useState(0);
    const damping = 0.85;

    // Initialize ranks
    useEffect(() => {
        const initialRanks = {};
        nodes.forEach(n => initialRanks[n.id] = 1 / nodes.length);
        setRanks(initialRanks);
        setIteration(0);
    }, [nodes]);

    const step = () => {
        const newRanks = {};
        nodes.forEach(n => newRanks[n.id] = (1 - damping) / nodes.length);

        // Distribute rank from sources
        nodes.forEach(source => {
            const outgoing = links.filter(l => l.source === source.id);
            if (outgoing.length > 0) {
                const share = (ranks[source.id] * damping) / outgoing.length;
                outgoing.forEach(link => {
                    newRanks[link.target] += share;
                });
            } else {
                // Sink node: distribute evenly to all (or self-loop logic)
                // Simplified: just lost (or distributed to all as random jump)
                const share = (ranks[source.id] * damping) / nodes.length;
                nodes.forEach(target => {
                    newRanks[target.id] += share;
                });
            }
        });

        setRanks(newRanks);
        setIteration(prev => prev + 1);
    };

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-indigo-900 mb-4">The Power Method</h2>
                <p className="text-lg text-slate-700 leading-relaxed">
                    The math behind the magic. In each step, a page gives its "Rank Juice" to its neighbors.
                    <br />
                    <span className="font-mono text-sm bg-slate-100 p-1 rounded">PR(A) = (1-d) + d * Σ (PR(T) / C(T))</span>
                </p>
            </div>

            <div className="flex gap-4 mb-8">
                <button
                    onClick={step}
                    className="px-8 py-3 bg-indigo-600 text-white rounded-xl font-bold text-xl hover:bg-indigo-700 shadow-lg transition-all transform hover:scale-105"
                >
                    🔄 Run 1 Iteration
                </button>
                <button
                    onClick={() => {
                        const initialRanks = {};
                        nodes.forEach(n => initialRanks[n.id] = 1 / nodes.length);
                        setRanks(initialRanks);
                        setIteration(0);
                    }}
                    className="px-6 py-3 bg-slate-200 text-slate-700 rounded-xl font-bold hover:bg-slate-300"
                >
                    Reset
                </button>
            </div>

            <div className="w-full max-w-4xl h-[500px] bg-slate-50 rounded-xl border-2 border-slate-200 relative overflow-hidden">
                <svg className="w-full h-full">
                    <defs>
                        <marker id="arrowhead-iter" markerWidth="10" markerHeight="7" refX="28" refY="3.5" orient="auto">
                            <polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8" />
                        </marker>
                    </defs>

                    {/* Links with Flow Animation (Simplified) */}
                    {links.map((link, i) => {
                        const source = nodes.find(n => n.id === link.source);
                        const target = nodes.find(n => n.id === link.target);
                        if (!source || !target) return null;
                        return (
                            <g key={i}>
                                <line
                                    x1={source.x} y1={source.y}
                                    x2={target.x} y2={target.y}
                                    stroke="#e2e8f0" strokeWidth="4"
                                    markerEnd="url(#arrowhead-iter)"
                                />
                                {/* Flow Particle */}
                                <circle r="3" fill="#6366f1">
                                    <animateMotion
                                        dur="1s"
                                        repeatCount="indefinite"
                                        path={`M${source.x},${source.y} L${target.x},${target.y}`}
                                    />
                                </circle>
                            </g>
                        );
                    })}

                    {/* Nodes with Rank Bars */}
                    {nodes.map(node => {
                        const rank = ranks[node.id] || 0;
                        const barHeight = rank * 200; // Scale factor

                        return (
                            <g key={node.id} transform={`translate(${node.x},${node.y})`}>
                                {/* Rank Bar */}
                                <rect
                                    x="-10"
                                    y={-25 - barHeight}
                                    width="20"
                                    height={barHeight}
                                    fill="#4f46e5"
                                    rx="4"
                                    className="transition-all duration-500"
                                />
                                <text x="0" y={-30 - barHeight} textAnchor="middle" className="text-xs font-bold text-indigo-600">
                                    {rank.toFixed(3)}
                                </text>

                                <circle r="20" fill="white" stroke="#4f46e5" strokeWidth="3" />
                                <text dy="5" textAnchor="middle" className="font-bold text-indigo-900 pointer-events-none">{node.id}</text>
                            </g>
                        );
                    })}
                </svg>

                <div className="absolute top-4 right-4 bg-white/90 p-4 rounded-lg shadow border border-slate-200">
                    <p className="font-bold text-slate-500 text-xs uppercase mb-2">Iteration: {iteration}</p>
                </div>
            </div>
        </div>
    );
}
