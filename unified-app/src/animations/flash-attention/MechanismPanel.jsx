import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowRight, Cpu, HardDrive, RefreshCw } from 'lucide-react';

export default function MechanismPanel() {
    const [currentBlockRow, setCurrentBlockRow] = useState(0);
    const [currentBlockCol, setCurrentBlockCol] = useState(0);
    const [isAnimating, setIsAnimating] = useState(false);
    const [processedBlocks, setProcessedBlocks] = useState(new Set());

    const seqLen = 8;
    const blockSize = 2;
    const numBlocks = seqLen / blockSize;

    // Animation loop
    useEffect(() => {
        if (!isAnimating) return;

        const timer = setInterval(() => {
            setCurrentBlockCol(col => {
                if (col + 1 >= numBlocks) {
                    setCurrentBlockRow(row => {
                        if (row + 1 >= numBlocks) {
                            setIsAnimating(false);
                            return 0;
                        }
                        return row + 1;
                    });
                    return 0;
                }
                return col + 1;
            });
        }, 600);

        return () => clearInterval(timer);
    }, [isAnimating, numBlocks]);

    // Track processed blocks
    useEffect(() => {
        const blockKey = `${currentBlockRow}-${currentBlockCol}`;
        setProcessedBlocks(prev => new Set([...prev, blockKey]));
    }, [currentBlockRow, currentBlockCol]);

    const reset = () => {
        setCurrentBlockRow(0);
        setCurrentBlockCol(0);
        setIsAnimating(false);
        setProcessedBlocks(new Set());
    };

    const isBlockActive = (row, col) => row === currentBlockRow && col === currentBlockCol;
    const isBlockProcessed = (row, col) => processedBlocks.has(`${row}-${col}`);

    return (
        <div className="p-8 h-full overflow-y-auto">
            <div className="max-w-5xl mx-auto">
                {/* Header */}
                <div className="text-center mb-8">
                    <h2 className="text-3xl font-bold text-amber-600 dark:text-amber-400 mb-4">
                        How Tiling Works
                    </h2>
                    <p className="text-slate-600 dark:text-slate-300 max-w-2xl mx-auto">
                        Flash Attention processes the attention matrix in <strong>small tiles</strong> that fit in SRAM.
                        Each tile is computed, combined with running statistics, and the results are accumulated
                        without ever materializing the full attention matrix.
                    </p>
                </div>

                {/* Controls */}
                <div className="flex justify-center gap-4 mb-8">
                    <button
                        onClick={() => setIsAnimating(!isAnimating)}
                        className={`px-6 py-3 rounded-xl font-medium transition-colors shadow-lg ${isAnimating
                                ? 'bg-red-500 hover:bg-red-600 text-white'
                                : 'bg-amber-500 hover:bg-amber-600 text-white'
                            }`}
                    >
                        {isAnimating ? 'Pause' : 'Start'} Animation
                    </button>
                    <button
                        onClick={reset}
                        className="px-6 py-3 bg-slate-200 dark:bg-slate-700 rounded-xl font-medium hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors flex items-center gap-2"
                    >
                        <RefreshCw size={18} />
                        Reset
                    </button>
                </div>

                <div className="grid lg:grid-cols-2 gap-8">
                    {/* Left: Matrix with Tiles */}
                    <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700">
                        <h3 className="font-bold text-slate-700 dark:text-slate-200 mb-4 text-center">
                            Attention Matrix ({seqLen}×{seqLen}) with Tiles
                        </h3>

                        {/* Q and K labels */}
                        <div className="flex justify-center mb-4">
                            <div className="relative">
                                {/* K label */}
                                <div className="absolute -top-6 left-0 right-0 text-center text-sm font-medium text-blue-600 dark:text-blue-400">
                                    K (Keys) →
                                </div>

                                {/* Q label */}
                                <div className="absolute -left-8 top-0 bottom-0 flex items-center">
                                    <span className="text-sm font-medium text-purple-600 dark:text-purple-400 transform -rotate-90">
                                        Q (Queries)
                                    </span>
                                </div>

                                {/* Matrix Grid */}
                                <div
                                    className="grid gap-1 ml-4"
                                    style={{ gridTemplateColumns: `repeat(${numBlocks}, 1fr)` }}
                                >
                                    {Array.from({ length: numBlocks }).map((_, blockRow) => (
                                        Array.from({ length: numBlocks }).map((_, blockCol) => {
                                            const active = isBlockActive(blockRow, blockCol);
                                            const processed = isBlockProcessed(blockRow, blockCol) && !active;

                                            return (
                                                <motion.div
                                                    key={`${blockRow}-${blockCol}`}
                                                    animate={{
                                                        scale: active ? 1.1 : 1,
                                                        boxShadow: active ? '0 0 20px rgba(245, 158, 11, 0.5)' : 'none'
                                                    }}
                                                    className={`
                                                        w-16 h-16 rounded-lg border-2 p-1
                                                        flex items-center justify-center
                                                        ${active
                                                            ? 'border-amber-500 bg-amber-100 dark:bg-amber-900/40'
                                                            : processed
                                                                ? 'border-emerald-400 bg-emerald-50 dark:bg-emerald-900/20'
                                                                : 'border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-700/50'
                                                        }
                                                    `}
                                                >
                                                    {/* Mini block visualization */}
                                                    <div className="grid grid-cols-2 gap-0.5">
                                                        {Array.from({ length: blockSize * blockSize }).map((_, i) => (
                                                            <div
                                                                key={i}
                                                                className={`w-3 h-3 rounded-sm ${active
                                                                        ? 'bg-amber-500'
                                                                        : processed
                                                                            ? 'bg-emerald-400'
                                                                            : 'bg-slate-300 dark:bg-slate-500'
                                                                    }`}
                                                            />
                                                        ))}
                                                    </div>
                                                </motion.div>
                                            );
                                        })
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Legend */}
                        <div className="flex justify-center gap-6 mt-4 text-sm">
                            <div className="flex items-center gap-2">
                                <div className="w-4 h-4 bg-amber-500 rounded"></div>
                                <span className="text-slate-600 dark:text-slate-400">Current Tile</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-4 h-4 bg-emerald-400 rounded"></div>
                                <span className="text-slate-600 dark:text-slate-400">Processed</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-4 h-4 bg-slate-300 dark:bg-slate-500 rounded"></div>
                                <span className="text-slate-600 dark:text-slate-400">Pending</span>
                            </div>
                        </div>
                    </div>

                    {/* Right: Memory Flow */}
                    <div className="space-y-6">
                        {/* Current Operation */}
                        <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700">
                            <h3 className="font-bold text-slate-700 dark:text-slate-200 mb-4">Current Operation</h3>

                            <AnimatePresence mode="wait">
                                <motion.div
                                    key={`${currentBlockRow}-${currentBlockCol}`}
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -20 }}
                                    className="space-y-3"
                                >
                                    <div className="flex items-center gap-3">
                                        <div className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 rounded font-mono text-sm">
                                            Q[{currentBlockRow * blockSize}:{(currentBlockRow + 1) * blockSize}]
                                        </div>
                                        <span className="text-slate-400">×</span>
                                        <div className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded font-mono text-sm">
                                            K[{currentBlockCol * blockSize}:{(currentBlockCol + 1) * blockSize}]
                                        </div>
                                    </div>
                                    <div className="text-sm text-slate-500">
                                        Processing tile ({currentBlockRow}, {currentBlockCol}) →
                                        Computing {blockSize}×{blockSize} = {blockSize * blockSize} attention scores
                                    </div>
                                </motion.div>
                            </AnimatePresence>
                        </div>

                        {/* Memory Flow Visualization */}
                        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-6 text-white">
                            <h3 className="font-bold mb-4">Memory Data Flow</h3>

                            <div className="flex items-center justify-between gap-4">
                                {/* HBM */}
                                <div className="text-center">
                                    <HardDrive size={32} className="mx-auto text-red-400 mb-2" />
                                    <div className="text-sm font-medium">HBM</div>
                                    <div className="text-xs text-slate-400">Q, K, V blocks</div>
                                </div>

                                {/* Arrow */}
                                <motion.div
                                    animate={{ x: [0, 10, 0] }}
                                    transition={{ repeat: Infinity, duration: 1 }}
                                >
                                    <ArrowRight size={24} className="text-amber-400" />
                                </motion.div>

                                {/* SRAM */}
                                <motion.div
                                    className="text-center"
                                    animate={{
                                        scale: isAnimating ? [1, 1.1, 1] : 1,
                                    }}
                                    transition={{ repeat: Infinity, duration: 0.6 }}
                                >
                                    <Cpu size={32} className="mx-auto text-emerald-400 mb-2" />
                                    <div className="text-sm font-medium">SRAM</div>
                                    <div className="text-xs text-slate-400">Tile compute</div>
                                </motion.div>

                                {/* Arrow */}
                                <motion.div
                                    animate={{ x: [0, 10, 0] }}
                                    transition={{ repeat: Infinity, duration: 1, delay: 0.5 }}
                                >
                                    <ArrowRight size={24} className="text-amber-400" />
                                </motion.div>

                                {/* Output */}
                                <div className="text-center">
                                    <HardDrive size={32} className="mx-auto text-blue-400 mb-2" />
                                    <div className="text-sm font-medium">HBM</div>
                                    <div className="text-xs text-slate-400">Output O</div>
                                </div>
                            </div>
                        </div>

                        {/* Online Softmax Info */}
                        <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-2xl p-6">
                            <h3 className="font-bold text-amber-700 dark:text-amber-400 mb-2">
                                🔑 The "Online Softmax" Trick
                            </h3>
                            <p className="text-sm text-slate-600 dark:text-slate-400">
                                Flash Attention uses <strong>running statistics</strong> (max and sum) to compute
                                the correct softmax across tiles without seeing all values at once.
                                Each tile updates these accumulators, producing the exact same result as
                                standard attention!
                            </p>
                        </div>
                    </div>
                </div>

                {/* Progress Bar */}
                <div className="mt-8 bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
                    <div className="flex justify-between text-sm mb-2">
                        <span className="text-slate-600 dark:text-slate-400">Tiles Processed</span>
                        <span className="font-mono text-amber-600">
                            {processedBlocks.size} / {numBlocks * numBlocks}
                        </span>
                    </div>
                    <div className="h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                        <motion.div
                            className="h-full bg-gradient-to-r from-amber-500 to-orange-500"
                            animate={{ width: `${(processedBlocks.size / (numBlocks * numBlocks)) * 100}%` }}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
}
