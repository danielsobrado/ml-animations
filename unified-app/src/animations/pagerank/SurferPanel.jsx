import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

export default function SurferPanel({ nodes, links }) {
    const [currentNodeId, setCurrentNodeId] = useState(nodes[0]?.id || null);
    const [visits, setVisits] = useState({});
    const [totalSteps, setTotalSteps] = useState(0);
    const [isRunning, setIsRunning] = useState(false);

    // Initialize visits
    useEffect(() => {
        const initialVisits = {};
        nodes.forEach(n => initialVisits[n.id] = 0);
        setVisits(initialVisits);
        if (nodes.length > 0 && !currentNodeId) setCurrentNodeId(nodes[0].id);
    }, [nodes]);

    useEffect(() => {
        let interval;
        if (isRunning && currentNodeId) {
            interval = setInterval(() => {
                // Find outgoing links
                const outgoing = links.filter(l => l.source === currentNodeId);

                let nextNodeId;
                // Damping factor logic (15% chance to jump randomly)
                if (Math.random() < 0.15 || outgoing.length === 0) {
                    // Jump to random node
                    const randomIdx = Math.floor(Math.random() * nodes.length);
                    nextNodeId = nodes[randomIdx].id;
                } else {
                    // Follow link
                    const randomIdx = Math.floor(Math.random() * outgoing.length);
                    nextNodeId = outgoing[randomIdx].target;
                }

                setCurrentNodeId(nextNodeId);
                setVisits(prev => ({ ...prev, [nextNodeId]: (prev[nextNodeId] || 0) + 1 }));
                setTotalSteps(prev => prev + 1);

            }, 100); // Fast steps
        }
        return () => clearInterval(interval);
    }, [isRunning, currentNodeId, links, nodes]);

    const maxVisits = Math.max(...Object.values(visits), 1);

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-indigo-900 mb-4">The Random Surfer</h2>
                <p className="text-lg text-slate-700 leading-relaxed">
                    Imagine a surfer clicking random links forever.
                    <br />
                    <strong>PageRank</strong> is simply the probability that the surfer is on a specific page at any moment.
                </p>
            </div>

            <div className="flex gap-4 mb-8">
                <button
                    onClick={() => setIsRunning(!isRunning)}
                    className={`px-8 py-3 rounded-xl font-bold text-xl shadow-lg transition-all ${isRunning ? 'bg-red-500 text-white hover:bg-red-600' : 'bg-green-600 text-white hover:bg-green-700'}`}
                >
                    {isRunning ? '🛑 Stop Surfing' : '🏄 Start Surfing'}
                </button>
                <button
                    onClick={() => {
                        const resetVisits = {};
                        nodes.forEach(n => resetVisits[n.id] = 0);
                        setVisits(resetVisits);
                        setTotalSteps(0);
                    }}
                    className="px-6 py-3 bg-slate-200 text-slate-700 rounded-xl font-bold hover:bg-slate-300"
                >
                    Reset Stats
                </button>
            </div>

            <div className="w-full max-w-4xl h-[500px] bg-slate-50 rounded-xl border-2 border-slate-200 relative overflow-hidden">
                <svg className="w-full h-full">
                    <defs>
                        <marker id="arrowhead-surfer" markerWidth="10" markerHeight="7" refX="28" refY="3.5" orient="auto">
                            <polygon points="0 0, 10 3.5, 0 7" fill="#cbd5e1" />
                        </marker>
                    </defs>

                    {/* Links */}
                    {links.map((link, i) => {
                        const source = nodes.find(n => n.id === link.source);
                        const target = nodes.find(n => n.id === link.target);
                        if (!source || !target) return null;
                        return (
                            <line
                                key={i}
                                x1={source.x} y1={source.y}
                                x2={target.x} y2={target.y}
                                stroke="#cbd5e1" strokeWidth="2"
                                markerEnd="url(#arrowhead-surfer)"
                            />
                        );
                    })}

                    {/* Nodes with Heatmap Size */}
                    {nodes.map(node => {
                        const count = visits[node.id] || 0;
                        const percentage = totalSteps > 0 ? (count / totalSteps) * 100 : 0;
                        // Scale radius based on visits (20 to 60)
                        const r = 20 + (count / maxVisits) * 40;

                        return (
                            <g key={node.id} transform={`translate(${node.x},${node.y})`}>
                                <circle
                                    r={r}
                                    fill={currentNodeId === node.id ? '#f59e0b' : 'white'}
                                    stroke={currentNodeId === node.id ? '#d97706' : '#4f46e5'}
                                    strokeWidth={currentNodeId === node.id ? 4 : 2}
                                    className="transition-all duration-300"
                                />
                                <text dy="5" textAnchor="middle" className="font-bold text-indigo-900 pointer-events-none text-sm">
                                    {node.id}
                                </text>
                                <text dy="25" textAnchor="middle" className="text-xs font-mono text-slate-500">
                                    {percentage.toFixed(1)}%
                                </text>
                            </g>
                        );
                    })}
                </svg>

                <div className="absolute top-4 right-4 bg-white/90 p-4 rounded-lg shadow border border-slate-200">
                    <p className="font-bold text-slate-500 text-xs uppercase mb-2">Total Steps: {totalSteps}</p>
                    {nodes.map(n => (
                        <div key={n.id} className="flex justify-between gap-4 text-sm">
                            <span className="font-bold">{n.id}:</span>
                            <span className="font-mono">{(visits[n.id] || 0)} visits</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
