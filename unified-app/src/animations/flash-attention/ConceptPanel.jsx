import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, Zap, HardDrive, Cpu } from 'lucide-react';

export default function ConceptPanel() {
    const [showComparison, setShowComparison] = useState(false);

    const seqLen = 8;
    const matrixSize = seqLen * seqLen;

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-5xl w-full">
                {/* Header */}
                <div className="text-center mb-12">
                    <h2 className="text-3xl font-bold text-amber-600 dark:text-amber-400 mb-4">
                        The Problem: Memory Bandwidth Bottleneck
                    </h2>
                    <p className="text-lg text-slate-700 dark:text-slate-300 leading-relaxed max-w-3xl mx-auto">
                        Standard attention computes the <strong>full N×N attention matrix</strong> and stores it in
                        slow HBM (High Bandwidth Memory). Flash Attention uses <strong>tiling</strong> to keep
                        computations in fast SRAM, avoiding the memory bottleneck.
                    </p>
                </div>

                {/* Memory Hierarchy Visualization */}
                <div className="grid md:grid-cols-2 gap-8 mb-12">
                    {/* HBM - Slow */}
                    <div className="bg-red-50 dark:bg-red-900/20 border-2 border-red-200 dark:border-red-800 rounded-2xl p-6">
                        <div className="flex items-center gap-3 mb-4">
                            <HardDrive className="text-red-500" size={28} />
                            <div>
                                <h3 className="text-xl font-bold text-red-600 dark:text-red-400">HBM (GPU Memory)</h3>
                                <p className="text-sm text-red-500">~1.5 TB/s bandwidth</p>
                            </div>
                        </div>
                        <div className="bg-red-100 dark:bg-red-900/40 rounded-xl p-4 mb-4">
                            <div className="text-center">
                                <div className="text-4xl font-bold text-red-600 dark:text-red-400 mb-1">40-80 GB</div>
                                <div className="text-sm text-red-500">Large but SLOW to access</div>
                            </div>
                        </div>
                        <p className="text-sm text-slate-600 dark:text-slate-400">
                            Standard attention stores the full N×N matrix here.
                            Reading/writing this data is the <strong>main bottleneck</strong>!
                        </p>
                    </div>

                    {/* SRAM - Fast */}
                    <div className="bg-emerald-50 dark:bg-emerald-900/20 border-2 border-emerald-200 dark:border-emerald-800 rounded-2xl p-6">
                        <div className="flex items-center gap-3 mb-4">
                            <Cpu className="text-emerald-500" size={28} />
                            <div>
                                <h3 className="text-xl font-bold text-emerald-600 dark:text-emerald-400">SRAM (On-chip)</h3>
                                <p className="text-sm text-emerald-500">~19 TB/s bandwidth</p>
                            </div>
                        </div>
                        <div className="bg-emerald-100 dark:bg-emerald-900/40 rounded-xl p-4 mb-4">
                            <div className="text-center">
                                <div className="text-4xl font-bold text-emerald-600 dark:text-emerald-400 mb-1">20 MB</div>
                                <div className="text-sm text-emerald-500">Small but 10x FASTER</div>
                            </div>
                        </div>
                        <p className="text-sm text-slate-600 dark:text-slate-400">
                            Flash Attention keeps computations here using <strong>tiling</strong>.
                            Only small blocks are loaded at a time!
                        </p>
                    </div>
                </div>

                {/* Toggle Comparison */}
                <div className="flex justify-center mb-8">
                    <button
                        onClick={() => setShowComparison(!showComparison)}
                        className="px-6 py-3 bg-amber-500 text-white rounded-xl font-medium hover:bg-amber-600 transition-colors shadow-lg"
                    >
                        {showComparison ? 'Hide' : 'Show'} Attention Matrix Comparison
                    </button>
                </div>

                {/* Attention Matrix Comparison */}
                {showComparison && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="grid md:grid-cols-2 gap-8 mb-8"
                    >
                        {/* Standard Attention */}
                        <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700">
                            <div className="flex items-center gap-2 mb-4">
                                <AlertTriangle className="text-red-500" size={20} />
                                <h4 className="font-bold text-slate-700 dark:text-slate-200">Standard Attention</h4>
                            </div>
                            <div className="flex justify-center mb-4">
                                <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${seqLen}, 1fr)` }}>
                                    {Array.from({ length: matrixSize }).map((_, i) => (
                                        <motion.div
                                            key={i}
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            transition={{ delay: i * 0.01 }}
                                            className="w-6 h-6 bg-red-400 dark:bg-red-600 rounded-sm"
                                        />
                                    ))}
                                </div>
                            </div>
                            <div className="text-center text-sm text-slate-600 dark:text-slate-400">
                                Full {seqLen}×{seqLen} = <span className="font-bold text-red-600">{matrixSize}</span> elements in HBM
                            </div>
                        </div>

                        {/* Flash Attention */}
                        <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700">
                            <div className="flex items-center gap-2 mb-4">
                                <CheckCircle className="text-emerald-500" size={20} />
                                <h4 className="font-bold text-slate-700 dark:text-slate-200">Flash Attention (Tiled)</h4>
                            </div>
                            <div className="flex justify-center mb-4">
                                <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${seqLen / 2}, 1fr)` }}>
                                    {Array.from({ length: 4 }).map((_, blockIdx) => (
                                        <motion.div
                                            key={blockIdx}
                                            initial={{ opacity: 0, scale: 0.8 }}
                                            animate={{ opacity: 1, scale: 1 }}
                                            transition={{ delay: blockIdx * 0.2 }}
                                            className="border-2 border-emerald-400 dark:border-emerald-600 rounded p-0.5"
                                        >
                                            <div className="grid grid-cols-2 gap-0.5">
                                                {Array.from({ length: 4 }).map((_, i) => (
                                                    <div
                                                        key={i}
                                                        className="w-5 h-5 bg-emerald-400 dark:bg-emerald-600 rounded-sm"
                                                    />
                                                ))}
                                            </div>
                                        </motion.div>
                                    ))}
                                </div>
                            </div>
                            <div className="text-center text-sm text-slate-600 dark:text-slate-400">
                                Only <span className="font-bold text-emerald-600">4</span> elements in SRAM at a time!
                            </div>
                        </div>
                    </motion.div>
                )}

                {/* Key Insight */}
                <motion.div
                    className="bg-gradient-to-r from-amber-500 to-orange-500 rounded-2xl p-6 text-white"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <div className="flex items-center gap-4">
                        <Zap size={36} />
                        <div>
                            <h4 className="font-bold text-xl mb-2">The Key Insight</h4>
                            <p className="text-amber-100">
                                GPUs spend more time <strong>moving data</strong> than computing.
                                Flash Attention reduces HBM reads/writes from O(N²) to O(N),
                                achieving <strong>2-4x speedup</strong> and <strong>5-20x memory reduction</strong>
                                while computing <em>exact</em> attention!
                            </p>
                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
