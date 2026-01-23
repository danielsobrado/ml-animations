import React from 'react';
import { motion } from 'framer-motion';

export default function MechanismPanel() {
    // Config
    const numQueryHeads = 8;
    const numKVHeads = 2;
    const groupSize = numQueryHeads / numKVHeads; // 4

    // Data for visualization
    const queryHeads = Array.from({ length: numQueryHeads }, (_, i) => ({ id: i }));
    const kvHeads = Array.from({ length: numKVHeads }, (_, i) => ({ id: i }));

    return (
        <div className="p-8 h-full overflow-y-auto">
            <div className="max-w-4xl mx-auto">
                <div className="text-center mb-12">
                    <h2 className="text-3xl font-bold text-fuchsia-600 dark:text-fuchsia-400 mb-4">GQA Mechanism</h2>
                    <p className="text-slate-600 dark:text-slate-300">
                        In Grouped-Query Attention, we divide the Query heads into <strong>{numKVHeads} groups</strong>.
                        All {groupSize} queries in a group attend to the <strong>same Key/Value head</strong>.
                    </p>
                </div>

                <div className="flex justify-between items-center gap-12 relative min-h-[400px]">
                    {/* Column: Query Heads */}
                    <div className="flex flex-col gap-2 w-1/3">
                        <h3 className="text-center font-bold mb-4 text-purple-600 dark:text-purple-400 flex items-center justify-center gap-2">
                            <span>Query Heads</span>
                            <span className="text-xs bg-purple-100 dark:bg-purple-900/30 px-2 py-1 rounded">Num = {numQueryHeads}</span>
                        </h3>
                        {queryHeads.map((q, i) => {
                            const groupIndex = Math.floor(i / groupSize);
                            const color = groupIndex === 0 ? "bg-orange-500" : "bg-blue-500";
                            return (
                                <motion.div
                                    key={i}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: i * 0.05 }}
                                    className={`relative h-12 rounded-lg flex items-center justify-center text-white font-mono text-sm shadow-sm ${color} border border-white/20`}
                                >
                                    Q{i}

                                    {/* Line to KV */}
                                    <ConnectorLine
                                        startX="100%"
                                        startY="50%"
                                        endX="150%" // Relative to parent, will need calculation or SVG overlay for absolute precision. 
                                    // Simplified: SVG overlay is better for connections.
                                    // Using absolute positioning for visual simplicity in this demo structure.
                                    />
                                </motion.div>
                            );
                        })}
                    </div>

                    {/* SVG Connector Overlay */}
                    <div className="absolute inset-0 pointer-events-none z-0">
                        <svg className="w-full h-full">
                            {queryHeads.map((q, i) => {
                                const groupIndex = Math.floor(i / groupSize);
                                // These calculations are approximate based on the layout
                                const startY = 64 + 8 + (i * 56) + 24; // Header + gap + (index * itemHeight+gap) + halfItem
                                const endY = 64 + 8 + (groupIndex * (56 * 4)) + (56 * 2); // Center of the KV block

                                const color = groupIndex === 0 ? "#f97316" : "#3b82f6";

                                return (
                                    <motion.path
                                        key={i}
                                        d={`M ${300} ${startY} C ${450} ${startY}, ${450} ${endY}, ${600} ${endY}`}
                                        fill="none"
                                        stroke={color}
                                        strokeWidth="2"
                                        strokeOpacity="0.4"
                                        initial={{ pathLength: 0 }}
                                        animate={{ pathLength: 1 }}
                                        transition={{ duration: 1, delay: 0.5 + (i * 0.1) }}
                                    />
                                );
                            })}
                        </svg>
                    </div>


                    {/* Column: KV Heads */}
                    <div className="flex flex-col gap-4 w-1/3 z-10">
                        <h3 className="text-center font-bold mb-4 text-indigo-600 dark:text-indigo-400 flex items-center justify-center gap-2">
                            <span>KV Heads</span>
                            <span className="text-xs bg-indigo-100 dark:bg-indigo-900/30 px-2 py-1 rounded">Num = {numKVHeads}</span>
                        </h3>
                        {kvHeads.map((kv, i) => {
                            const color = i === 0 ? "bg-orange-600" : "bg-blue-600";
                            // Height needs to match the group of Qs roughly
                            const heightClass = `h-[216px]`; // 4 * 56 - 8 (gap) roughly

                            return (
                                <motion.div
                                    key={i}
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: 1 + (i * 0.2) }}
                                    className={`${heightClass} rounded-2xl flex flex-col items-center justify-center text-white font-bold shadow-lg ${color} relative`}
                                >
                                    <div className="text-2xl mb-2">KV{i}</div>
                                    <div className="text-xs opacity-75 font-mono">Shared Memory</div>
                                </motion.div>
                            );
                        })}
                    </div>
                </div>

                <div className="mt-16 grid grid-cols-3 gap-6 text-center">
                    <MetricCard label="Params" value="Lower" desc="Fewer KV heads" color="text-green-500" />
                    <MetricCard label="Speed" value="Faster" desc="Less memory bandwidth" color="text-green-500" />
                    <MetricCard label="Quality" value="High" desc="Close to MHA" color="text-blue-500" />
                </div>
            </div>
        </div>
    );
}

// Helper for metrics
function MetricCard({ label, value, desc, color }) {
    return (
        <div className="bg-white dark:bg-slate-800 p-4 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700">
            <div className="text-sm text-slate-500 uppercase tracking-wider mb-1">{label}</div>
            <div className={`text-2xl font-bold ${color} mb-1`}>{value}</div>
            <div className="text-xs text-slate-400">{desc}</div>
        </div>
    );
}

// Dummy component to fix reference error in map if needed, though SVG is handled separately
function ConnectorLine() { return null; }
