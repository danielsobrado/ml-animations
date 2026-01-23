import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, RotateCcw, Zap, HardDrive, Cpu } from 'lucide-react';

export default function PlaygroundPanel() {
    const [isPlaying, setIsPlaying] = useState(false);
    const [seqLength, setSeqLength] = useState(16);
    const [blockSize, setBlockSize] = useState(4);
    const [currentTile, setCurrentTile] = useState({ row: 0, col: 0 });
    const [processedTiles, setProcessedTiles] = useState([]);

    const numBlocksPerDim = Math.ceil(seqLength / blockSize);
    const totalTiles = numBlocksPerDim * numBlocksPerDim;

    // Memory calculations
    const standardMemory = (seqLength * seqLength * 4) / 1024; // KB, storing full attention matrix
    const flashMemory = (blockSize * blockSize * 4) / 1024; // KB, only one tile in SRAM
    const memorySavings = ((standardMemory - flashMemory) / standardMemory * 100).toFixed(1);

    // Reset when parameters change
    useEffect(() => {
        reset();
    }, [seqLength, blockSize]);

    // Animation loop
    useEffect(() => {
        if (!isPlaying) return;

        const timer = setInterval(() => {
            setCurrentTile(prev => {
                const newCol = prev.col + 1;
                if (newCol >= numBlocksPerDim) {
                    const newRow = prev.row + 1;
                    if (newRow >= numBlocksPerDim) {
                        setIsPlaying(false);
                        return { row: 0, col: 0 };
                    }
                    return { row: newRow, col: 0 };
                }
                return { ...prev, col: newCol };
            });
        }, 200);

        return () => clearInterval(timer);
    }, [isPlaying, numBlocksPerDim]);

    // Track processed tiles
    useEffect(() => {
        if (isPlaying || processedTiles.length > 0) {
            const key = `${currentTile.row}-${currentTile.col}`;
            if (!processedTiles.includes(key)) {
                setProcessedTiles(prev => [...prev, key]);
            }
        }
    }, [currentTile, isPlaying]);

    const reset = useCallback(() => {
        setIsPlaying(false);
        setCurrentTile({ row: 0, col: 0 });
        setProcessedTiles([]);
    }, []);

    const stepForward = () => {
        setCurrentTile(prev => {
            const newCol = prev.col + 1;
            if (newCol >= numBlocksPerDim) {
                const newRow = prev.row + 1;
                if (newRow >= numBlocksPerDim) {
                    return prev;
                }
                return { row: newRow, col: 0 };
            }
            return { ...prev, col: newCol };
        });
        const key = `${currentTile.row}-${currentTile.col}`;
        if (!processedTiles.includes(key)) {
            setProcessedTiles(prev => [...prev, key]);
        }
    };

    const getTileStatus = (row, col) => {
        const key = `${row}-${col}`;
        if (row === currentTile.row && col === currentTile.col) return 'active';
        if (processedTiles.includes(key)) return 'processed';
        return 'pending';
    };

    return (
        <div className="p-8 h-full overflow-y-auto">
            <div className="max-w-6xl mx-auto">
                {/* Header */}
                <div className="text-center mb-8">
                    <h2 className="text-3xl font-bold text-amber-600 dark:text-amber-400 mb-4">
                        Flash Attention Tiling Simulator
                    </h2>
                    <p className="text-slate-600 dark:text-slate-300">
                        Adjust sequence length and block size to see how Flash Attention tiles the computation.
                    </p>
                </div>

                <div className="grid lg:grid-cols-3 gap-8">
                    {/* Controls */}
                    <div className="space-y-6">
                        <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700">
                            <h3 className="font-bold text-slate-700 dark:text-slate-200 mb-4">Configuration</h3>

                            <div className="space-y-5">
                                <div>
                                    <label className="flex justify-between text-sm mb-2 dark:text-slate-300">
                                        <span>Sequence Length (N)</span>
                                        <span className="font-mono text-amber-600">{seqLength}</span>
                                    </label>
                                    <input
                                        type="range" min="8" max="32" step="4"
                                        value={seqLength}
                                        onChange={(e) => setSeqLength(parseInt(e.target.value))}
                                        className="w-full accent-amber-500"
                                    />
                                </div>

                                <div>
                                    <label className="flex justify-between text-sm mb-2 dark:text-slate-300">
                                        <span>Block Size (B)</span>
                                        <span className="font-mono text-amber-600">{blockSize}</span>
                                    </label>
                                    <input
                                        type="range" min="2" max="8" step="2"
                                        value={blockSize}
                                        onChange={(e) => setBlockSize(parseInt(e.target.value))}
                                        className="w-full accent-amber-500"
                                    />
                                </div>
                            </div>

                            <div className="mt-4 p-3 bg-slate-100 dark:bg-slate-700/50 rounded-lg">
                                <div className="grid grid-cols-2 gap-2 text-sm">
                                    <div className="text-slate-500">Matrix Size:</div>
                                    <div className="font-mono">{seqLength}×{seqLength}</div>
                                    <div className="text-slate-500">Tiles per dim:</div>
                                    <div className="font-mono">{numBlocksPerDim}</div>
                                    <div className="text-slate-500">Total Tiles:</div>
                                    <div className="font-mono text-amber-600 font-bold">{totalTiles}</div>
                                </div>
                            </div>
                        </div>

                        {/* Playback Controls */}
                        <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700">
                            <div className="flex items-center justify-center gap-3">
                                <button
                                    onClick={reset}
                                    className="p-3 rounded-full bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
                                >
                                    <RotateCcw size={20} />
                                </button>
                                <button
                                    onClick={() => setIsPlaying(!isPlaying)}
                                    className="p-4 rounded-full bg-amber-500 text-white hover:bg-amber-600 transition-colors shadow-lg"
                                >
                                    {isPlaying ? <Pause size={24} /> : <Play size={24} />}
                                </button>
                                <button
                                    onClick={stepForward}
                                    disabled={processedTiles.length >= totalTiles}
                                    className="px-4 py-2 rounded-lg bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 disabled:opacity-50 font-medium text-sm"
                                >
                                    Step
                                </button>
                            </div>
                            <div className="text-center mt-4 text-slate-500 text-sm">
                                Tile {Math.min(processedTiles.length, totalTiles)} / {totalTiles}
                            </div>
                        </div>

                        {/* Memory Comparison */}
                        <div className="bg-gradient-to-br from-amber-500 to-orange-600 rounded-2xl p-6 text-white">
                            <h3 className="font-bold mb-4 flex items-center gap-2">
                                <Zap size={20} />
                                Memory Comparison
                            </h3>

                            <div className="space-y-4">
                                <div>
                                    <div className="flex justify-between text-sm mb-1">
                                        <span className="flex items-center gap-1">
                                            <HardDrive size={14} />
                                            Standard Attention
                                        </span>
                                        <span className="font-mono">{standardMemory.toFixed(1)} KB</span>
                                    </div>
                                    <div className="h-3 bg-white/20 rounded-full overflow-hidden">
                                        <div className="h-full bg-red-400 w-full"></div>
                                    </div>
                                </div>

                                <div>
                                    <div className="flex justify-between text-sm mb-1">
                                        <span className="flex items-center gap-1">
                                            <Cpu size={14} />
                                            Flash Attention
                                        </span>
                                        <span className="font-mono">{flashMemory.toFixed(2)} KB</span>
                                    </div>
                                    <div className="h-3 bg-white/20 rounded-full overflow-hidden">
                                        <motion.div
                                            className="h-full bg-emerald-400"
                                            animate={{ width: `${(flashMemory / standardMemory) * 100}%` }}
                                        />
                                    </div>
                                </div>

                                <div className="text-center pt-2 border-t border-white/20">
                                    <span className="text-2xl font-bold">{memorySavings}%</span>
                                    <span className="text-sm ml-2">memory reduction</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Tiled Matrix Visualization */}
                    <div className="lg:col-span-2 bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700">
                        <h3 className="font-bold text-slate-700 dark:text-slate-200 mb-6 text-center">
                            Attention Matrix Tiling
                        </h3>

                        <div className="flex justify-center">
                            <div className="relative">
                                {/* Labels */}
                                <div className="absolute -top-8 left-0 right-0 text-center text-sm font-medium text-blue-600 dark:text-blue-400">
                                    Keys (K)
                                </div>
                                <div className="absolute -left-10 top-0 bottom-0 flex items-center">
                                    <span className="text-sm font-medium text-purple-600 dark:text-purple-400 transform -rotate-90 whitespace-nowrap">
                                        Queries (Q)
                                    </span>
                                </div>

                                {/* Matrix Grid */}
                                <div
                                    className="grid gap-1 ml-4"
                                    style={{
                                        gridTemplateColumns: `repeat(${numBlocksPerDim}, 1fr)`,
                                        maxWidth: '400px'
                                    }}
                                >
                                    {Array.from({ length: numBlocksPerDim }).map((_, row) => (
                                        Array.from({ length: numBlocksPerDim }).map((_, col) => {
                                            const status = getTileStatus(row, col);
                                            const tileSize = Math.max(20, Math.min(60, 400 / numBlocksPerDim - 4));

                                            return (
                                                <motion.div
                                                    key={`${row}-${col}`}
                                                    animate={{
                                                        scale: status === 'active' ? 1.15 : 1,
                                                        boxShadow: status === 'active'
                                                            ? '0 0 25px rgba(245, 158, 11, 0.6)'
                                                            : 'none'
                                                    }}
                                                    transition={{ type: 'spring', stiffness: 300 }}
                                                    style={{ width: tileSize, height: tileSize }}
                                                    className={`
                                                        rounded-md border-2 flex items-center justify-center
                                                        text-xs font-mono transition-colors
                                                        ${status === 'active'
                                                            ? 'border-amber-500 bg-amber-400 text-white'
                                                            : status === 'processed'
                                                                ? 'border-emerald-400 bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300'
                                                                : 'border-slate-300 dark:border-slate-600 bg-slate-100 dark:bg-slate-700 text-slate-400'
                                                        }
                                                    `}
                                                >
                                                    {tileSize > 30 && `${row},${col}`}
                                                </motion.div>
                                            );
                                        })
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Legend */}
                        <div className="flex justify-center gap-8 mt-6 text-sm">
                            <div className="flex items-center gap-2">
                                <div className="w-5 h-5 bg-amber-400 border-2 border-amber-500 rounded"></div>
                                <span className="text-slate-600 dark:text-slate-400">Computing in SRAM</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-5 h-5 bg-emerald-100 dark:bg-emerald-900/40 border-2 border-emerald-400 rounded"></div>
                                <span className="text-slate-600 dark:text-slate-400">Completed</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-5 h-5 bg-slate-100 dark:bg-slate-700 border-2 border-slate-300 dark:border-slate-600 rounded"></div>
                                <span className="text-slate-600 dark:text-slate-400">Pending</span>
                            </div>
                        </div>

                        {/* Progress Info */}
                        <div className="mt-6 p-4 bg-slate-100 dark:bg-slate-700/50 rounded-xl">
                            <div className="flex justify-between items-center mb-2">
                                <span className="text-sm text-slate-600 dark:text-slate-400">Processing Progress</span>
                                <span className="font-mono text-amber-600">
                                    {((processedTiles.length / totalTiles) * 100).toFixed(0)}%
                                </span>
                            </div>
                            <div className="h-2 bg-slate-200 dark:bg-slate-600 rounded-full overflow-hidden">
                                <motion.div
                                    className="h-full bg-gradient-to-r from-amber-500 to-orange-500"
                                    animate={{ width: `${(processedTiles.length / totalTiles) * 100}%` }}
                                />
                            </div>
                        </div>

                        {/* Key Insight */}
                        <div className="mt-6 p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-xl">
                            <p className="text-sm text-slate-700 dark:text-slate-300">
                                <strong className="text-amber-600">Key Insight:</strong> Each tile only needs
                                <span className="font-mono mx-1">{blockSize}×{blockSize} = {blockSize * blockSize}</span>
                                elements in SRAM, regardless of total sequence length. This is why Flash Attention
                                scales to long sequences without running out of memory!
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
