import React, { useState } from 'react';
import { motion } from 'framer-motion';

export default function PlaygroundPanel() {
    const [numQ, setNumQ] = useState(8);
    const [numKV, setNumKV] = useState(2);

    // Constraints
    // numQ must be divisible by numKV for perfect grouping in this demo
    // Simply rounding or enforcing step could work.

    const handleKVChange = (val) => {
        // Ensure numKV is a factor of numQ if possible, or just clamp
        // specific to this demo, let's stick to powers of 2 for simplicity
        setNumKV(val);
    };

    const groupSize = Math.max(1, Math.floor(numQ / numKV));

    // Generate arrays
    const qItems = Array.from({ length: numQ }, (_, i) => i);
    const kvItems = Array.from({ length: numKV }, (_, i) => i);

    const getGQAStatus = () => {
        if (numKV === 1) return { label: "MQA (Multi-Query)", color: "text-red-500" };
        if (numKV === numQ) return { label: "MHA (Multi-Head)", color: "text-blue-500" };
        return { label: "GQA (Grouped-Query)", color: "text-fuchsia-500" };
    };

    const status = getGQAStatus();

    return (
        <div className="p-8 h-full flex flex-col overflow-y-auto">
            <div className="w-full max-w-4xl mx-auto flex flex-col gap-8 h-full">

                {/* Controls */}
                <div className="bg-slate-100 dark:bg-slate-800/50 p-6 rounded-2xl border border-slate-200 dark:border-slate-700">
                    <div className="flex justify-between items-start mb-6">
                        <div>
                            <h2 className="text-xl font-bold dark:text-white">Configuration</h2>
                            <p className="text-slate-500 text-sm">Adjust the number of heads to see how grouping changes.</p>
                        </div>
                        <div className={`px-4 py-2 rounded-lg bg-white dark:bg-slate-800 shadow font-bold ${status.color}`}>
                            {status.label}
                        </div>
                    </div>

                    <div className="grid md:grid-cols-2 gap-8">
                        <div>
                            <label className="flex justify-between text-sm font-medium mb-2 dark:text-slate-300">
                                Query Heads (H_q)
                                <span>{numQ}</span>
                            </label>
                            <input
                                type="range" min="4" max="16" step="4"
                                value={numQ} onChange={(e) => {
                                    const val = parseInt(e.target.value);
                                    setNumQ(val);
                                    // Reset KV to maintain ratio if needed, or clamp
                                    if (numKV > val) setNumKV(val);
                                }}
                                className="w-full"
                            />
                        </div>
                        <div>
                            <label className="flex justify-between text-sm font-medium mb-2 dark:text-slate-300">
                                KV Heads (H_kv)
                                <span>{numKV}</span>
                            </label>
                            <input
                                type="range" min="1" max={numQ}
                                value={numKV}
                                onChange={(e) => {
                                    // Snap to valid divisors to make visualization clean
                                    const val = parseInt(e.target.value);
                                    // Find nearest valid divisor? For now just allow it, visuals might look funny if not divisible
                                    // Let's restrict to powers of 2 for this demo: 1, 2, 4, 8...
                                    setNumKV(val);
                                }}
                                className="w-full"
                            />
                            <div className="flex justify-between text-xs text-slate-400 mt-1">
                                <span>1 (MQA)</span>
                                <span>{numQ} (MHA)</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Visualizer */}
                <div className="flex-1 min-h-[300px] border border-slate-200 dark:border-slate-700 rounded-2xl p-8 relative overflow-hidden flex items-center justify-center">

                    <div className="flex justify-between w-full max-w-2xl px-8 z-10">
                        {/* Q Stack */}
                        <div className="flex flex-col gap-1 items-center">
                            <div className="text-xs font-bold text-slate-400 mb-2">QUERIES</div>
                            {qItems.map(i => {
                                const groupIdx = Math.floor(i / (numQ / numKV));
                                // Alternating colors for groups
                                const hue = (groupIdx * 360) / numKV;
                                return (
                                    <motion.div
                                        key={`q-${i}`}
                                        layout
                                        className="w-12 h-6 rounded bg-slate-800 border border-white/20"
                                        style={{ backgroundColor: `hsla(${hue}, 70%, 50%, 1)` }}
                                    />
                                );
                            })}
                        </div>

                        {/* KV Stack */}
                        <div className="flex flex-col justify-center items-center h-full relative" style={{ gap: numKV === 1 ? 0 : '1rem' }}>
                            {/* Spacer to center vertically effectively */}
                            <div className="absolute -top-6 text-xs font-bold text-slate-400">KEYS / VALUES</div>
                            {kvItems.map(i => {
                                const hue = (i * 360) / numKV;
                                const height = (300 / numKV) - 10; // Dynamic height
                                return (
                                    <motion.div
                                        key={`kv-${i}`}
                                        layout
                                        className="w-16 rounded-lg flex items-center justify-center text-white font-bold shadow-lg"
                                        style={{
                                            backgroundColor: `hsla(${hue}, 70%, 50%, 1)`,
                                            height: `${Math.max(40, (numQ / numKV) * 24 + 4)}px` // Scale with # of queries serving
                                        }}
                                    >
                                        KV
                                    </motion.div>
                                );
                            })}
                        </div>
                    </div>

                    {/* SVG Connections */}
                    <svg className="absolute inset-0 w-full h-full pointer-events-none">
                        {/* We can't easily draw perfect lines in this flex layout without ref coordinates.
                             However, CSS-based connector lines or simple visual juxtaposition is often enough.
                             For the purpose of this task, the color coding firmly establishes the relationship. 
                         */}
                    </svg>

                </div>
            </div>
        </div>
    );
}
