import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, RotateCcw, Zap } from 'lucide-react';

export default function PlaygroundPanel() {
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentStep, setCurrentStep] = useState(0);
    const [seqLength, setSeqLength] = useState(16);
    const [numHeads, setNumHeads] = useState(8);
    const [headDim, setHeadDim] = useState(64);
    const [numLayers, setNumLayers] = useState(12);

    const generatedTokens = Array.from({ length: currentStep }, (_, i) => `T${i}`);

    // Auto-play effect
    useEffect(() => {
        if (isPlaying && currentStep < seqLength) {
            const timer = setTimeout(() => {
                setCurrentStep(s => s + 1);
            }, 300);
            return () => clearTimeout(timer);
        } else if (currentStep >= seqLength) {
            setIsPlaying(false);
        }
    }, [isPlaying, currentStep, seqLength]);

    const reset = () => {
        setCurrentStep(0);
        setIsPlaying(false);
    };

    // Calculate memory in MB
    const calculateMemory = (tokens) => {
        // Memory = 2 (K+V) * tokens * num_heads * head_dim * num_layers * 4 bytes (float32)
        const bytes = 2 * tokens * numHeads * headDim * numLayers * 4;
        return bytes / (1024 * 1024); // Convert to MB
    };

    const currentMemory = calculateMemory(currentStep);
    const maxMemory = calculateMemory(seqLength);
    const memoryPercentage = (currentMemory / maxMemory) * 100 || 0;

    return (
        <div className="p-8 h-full overflow-y-auto">
            <div className="max-w-5xl mx-auto">
                {/* Header */}
                <div className="text-center mb-8">
                    <h2 className="text-3xl font-bold text-emerald-600 dark:text-emerald-400 mb-4">
                        KV Cache Memory Simulator
                    </h2>
                    <p className="text-slate-600 dark:text-slate-300">
                        Watch the KV Cache grow as tokens are generated. Adjust model parameters to see memory impact.
                    </p>
                </div>

                <div className="grid lg:grid-cols-2 gap-8">
                    {/* Controls */}
                    <div className="space-y-6">
                        <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700">
                            <h3 className="font-bold text-slate-700 dark:text-slate-200 mb-4">Model Configuration</h3>

                            <div className="space-y-4">
                                <div>
                                    <label className="flex justify-between text-sm mb-2 dark:text-slate-300">
                                        <span>Sequence Length</span>
                                        <span className="font-mono text-emerald-600">{seqLength}</span>
                                    </label>
                                    <input
                                        type="range" min="8" max="64" step="8"
                                        value={seqLength}
                                        onChange={(e) => {
                                            setSeqLength(parseInt(e.target.value));
                                            setCurrentStep(0);
                                        }}
                                        className="w-full accent-emerald-500"
                                    />
                                </div>

                                <div>
                                    <label className="flex justify-between text-sm mb-2 dark:text-slate-300">
                                        <span>Number of Heads</span>
                                        <span className="font-mono text-emerald-600">{numHeads}</span>
                                    </label>
                                    <input
                                        type="range" min="4" max="32" step="4"
                                        value={numHeads}
                                        onChange={(e) => setNumHeads(parseInt(e.target.value))}
                                        className="w-full accent-emerald-500"
                                    />
                                </div>

                                <div>
                                    <label className="flex justify-between text-sm mb-2 dark:text-slate-300">
                                        <span>Head Dimension</span>
                                        <span className="font-mono text-emerald-600">{headDim}</span>
                                    </label>
                                    <input
                                        type="range" min="32" max="128" step="32"
                                        value={headDim}
                                        onChange={(e) => setHeadDim(parseInt(e.target.value))}
                                        className="w-full accent-emerald-500"
                                    />
                                </div>

                                <div>
                                    <label className="flex justify-between text-sm mb-2 dark:text-slate-300">
                                        <span>Number of Layers</span>
                                        <span className="font-mono text-emerald-600">{numLayers}</span>
                                    </label>
                                    <input
                                        type="range" min="6" max="48" step="6"
                                        value={numLayers}
                                        onChange={(e) => setNumLayers(parseInt(e.target.value))}
                                        className="w-full accent-emerald-500"
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Playback Controls */}
                        <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700">
                            <div className="flex items-center justify-center gap-4">
                                <button
                                    onClick={reset}
                                    className="p-3 rounded-full bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
                                >
                                    <RotateCcw size={20} />
                                </button>
                                <button
                                    onClick={() => setIsPlaying(!isPlaying)}
                                    disabled={currentStep >= seqLength}
                                    className="p-4 rounded-full bg-emerald-500 text-white hover:bg-emerald-600 disabled:opacity-50 transition-colors shadow-lg"
                                >
                                    {isPlaying ? <Pause size={24} /> : <Play size={24} />}
                                </button>
                                <button
                                    onClick={() => setCurrentStep(s => Math.min(s + 1, seqLength))}
                                    disabled={currentStep >= seqLength}
                                    className="px-4 py-2 rounded-lg bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 disabled:opacity-50 font-medium"
                                >
                                    Step +1
                                </button>
                            </div>
                            <div className="text-center mt-4 text-slate-500 text-sm">
                                Token {currentStep} / {seqLength}
                            </div>
                        </div>
                    </div>

                    {/* Visualization */}
                    <div className="space-y-6">
                        {/* Memory Bar */}
                        <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700">
                            <h3 className="font-bold text-slate-700 dark:text-slate-200 mb-4 flex items-center gap-2">
                                <Zap className="text-yellow-500" />
                                Memory Usage
                            </h3>

                            <div className="mb-4">
                                <div className="h-8 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                                    <motion.div
                                        className="h-full bg-gradient-to-r from-emerald-500 to-teal-500"
                                        initial={{ width: 0 }}
                                        animate={{ width: `${memoryPercentage}%` }}
                                        transition={{ type: 'spring', stiffness: 100 }}
                                    />
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4 text-center">
                                <div className="bg-slate-100 dark:bg-slate-700/50 rounded-lg p-3">
                                    <div className="text-2xl font-bold text-emerald-600">{currentMemory.toFixed(2)} MB</div>
                                    <div className="text-xs text-slate-500">Current Usage</div>
                                </div>
                                <div className="bg-slate-100 dark:bg-slate-700/50 rounded-lg p-3">
                                    <div className="text-2xl font-bold text-slate-600 dark:text-slate-300">{maxMemory.toFixed(2)} MB</div>
                                    <div className="text-xs text-slate-500">Max at {seqLength} tokens</div>
                                </div>
                            </div>
                        </div>

                        {/* Token Grid */}
                        <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700">
                            <h3 className="font-bold text-slate-700 dark:text-slate-200 mb-4">Generated Tokens in Cache</h3>

                            <div className="flex flex-wrap gap-2 min-h-[100px]">
                                <AnimatePresence>
                                    {generatedTokens.map((token, i) => (
                                        <motion.div
                                            key={i}
                                            initial={{ scale: 0, opacity: 0 }}
                                            animate={{ scale: 1, opacity: 1 }}
                                            exit={{ scale: 0, opacity: 0 }}
                                            className="w-10 h-10 rounded-lg bg-gradient-to-br from-emerald-400 to-teal-500 text-white flex items-center justify-center text-xs font-mono shadow-sm"
                                        >
                                            {i}
                                        </motion.div>
                                    ))}
                                </AnimatePresence>
                                {currentStep === 0 && (
                                    <div className="text-slate-400 text-sm italic">
                                        Press Play to start generating...
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Connection to GQA */}
                        <div className="bg-gradient-to-r from-purple-500 to-fuchsia-500 rounded-2xl p-6 text-white">
                            <h4 className="font-bold mb-2">Why does this matter?</h4>
                            <p className="text-purple-100 text-sm">
                                As sequences get longer, the KV Cache becomes the main memory bottleneck.
                                <strong> Grouped-Query Attention (GQA)</strong> reduces this by sharing KV heads across multiple query heads,
                                cutting memory usage while preserving quality!
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
