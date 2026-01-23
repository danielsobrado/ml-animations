import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, RotateCcw, ChevronRight, Layers, Plus, ArrowRight, Scale } from 'lucide-react';

// Vector visualization component
function VectorBar({ values, label, color, scale = 1, showNorm = false }) {
    const norm = Math.sqrt(values.reduce((sum, v) => sum + v * v, 0));
    const maxVal = Math.max(...values.map(Math.abs)) || 1;

    return (
        <div className="flex items-center gap-3">
            <div className="text-xs font-mono w-16 text-right text-slate-600 dark:text-slate-400">
                {label}
            </div>
            <div className="flex gap-1">
                {values.map((v, i) => (
                    <motion.div
                        key={i}
                        className="w-8 h-8 rounded flex items-center justify-center text-xs font-mono text-white"
                        style={{
                            backgroundColor: color,
                            opacity: 0.4 + Math.abs(v) / maxVal * 0.6
                        }}
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: scale, opacity: 1 }}
                        transition={{ delay: i * 0.05 }}
                    >
                        {v.toFixed(1)}
                    </motion.div>
                ))}
            </div>
            {showNorm && (
                <div className="text-xs text-slate-500 dark:text-slate-400 font-mono">
                    ||v|| = {norm.toFixed(2)}
                </div>
            )}
        </div>
    );
}

export default function MechanismPanel() {
    const [step, setStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [layerIndex, setLayerIndex] = useState(0);

    // Sample vectors for demonstration
    const initialVector = [0.5, -0.3, 0.8, -0.2];
    const layerOutputs = [
        [0.2, 0.4, -0.1, 0.3],   // Attention output
        [-0.1, 0.2, 0.3, -0.1],  // FFN output
    ];

    const steps = [
        {
            id: 'input',
            title: 'Input Token Vector',
            description: 'The token embedding enters the residual stream',
        },
        {
            id: 'attention',
            title: 'Attention Layer',
            description: 'Self-attention computes relationships and produces an update vector',
        },
        {
            id: 'add-attention',
            title: 'Residual Addition #1',
            description: 'The attention output is ADDED to the stream (not replaced!)',
        },
        {
            id: 'layernorm1',
            title: 'Layer Normalization',
            description: 'LayerNorm rescales the vector to unit norm, preserving direction',
        },
        {
            id: 'ffn',
            title: 'Feed-Forward Network',
            description: 'MLP processes each position independently',
        },
        {
            id: 'add-ffn',
            title: 'Residual Addition #2',
            description: 'FFN output is added to the stream',
        },
        {
            id: 'layernorm2',
            title: 'Final Layer Normalization',
            description: 'Ready for the next transformer block!',
        },
    ];

    // Compute current vectors based on step
    const computeVectors = () => {
        let residual = [...initialVector];
        let attnOut = layerOutputs[0];
        let ffnOut = layerOutputs[1];

        const afterAttn = residual.map((v, i) => v + attnOut[i]);
        const normAfterAttn = normalizeVector(afterAttn);
        const afterFFN = normAfterAttn.map((v, i) => v + ffnOut[i]);
        const normAfterFFN = normalizeVector(afterFFN);

        return {
            initial: residual,
            attnOutput: attnOut,
            afterAttn,
            normAfterAttn,
            ffnOutput: ffnOut,
            afterFFN,
            normAfterFFN,
        };
    };

    const normalizeVector = (vec) => {
        const norm = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
        return vec.map(v => v / norm);
    };

    const vectors = computeVectors();

    useEffect(() => {
        if (isPlaying) {
            const timer = setTimeout(() => {
                if (step < steps.length - 1) {
                    setStep(step + 1);
                } else {
                    setIsPlaying(false);
                }
            }, 2000);
            return () => clearTimeout(timer);
        }
    }, [isPlaying, step]);

    const reset = () => {
        setStep(0);
        setIsPlaying(false);
    };

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-5xl w-full">
                {/* Header */}
                <div className="text-center mb-8">
                    <h2 className="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-4">
                        Vector Flow Through a Transformer Block
                    </h2>
                    <p className="text-lg text-slate-700 dark:text-slate-300">
                        Watch how vectors flow and accumulate through residual connections
                    </p>
                </div>

                {/* Controls */}
                <div className="flex justify-center gap-4 mb-8">
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
                    >
                        {isPlaying ? <Pause size={18} /> : <Play size={18} />}
                        {isPlaying ? 'Pause' : 'Play'}
                    </button>
                    <button
                        onClick={reset}
                        className="flex items-center gap-2 px-4 py-2 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
                    >
                        <RotateCcw size={18} />
                        Reset
                    </button>
                    <button
                        onClick={() => step < steps.length - 1 && setStep(step + 1)}
                        disabled={step >= steps.length - 1}
                        className="flex items-center gap-2 px-4 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 transition-colors disabled:opacity-50"
                    >
                        Next Step
                        <ChevronRight size={18} />
                    </button>
                </div>

                {/* Step Progress */}
                <div className="flex justify-center mb-8">
                    <div className="flex gap-2">
                        {steps.map((s, i) => (
                            <button
                                key={s.id}
                                onClick={() => setStep(i)}
                                className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-all ${
                                    i === step
                                        ? 'bg-blue-500 text-white scale-110'
                                        : i < step
                                            ? 'bg-blue-200 dark:bg-blue-800 text-blue-700 dark:text-blue-300'
                                            : 'bg-slate-200 dark:bg-slate-700 text-slate-500'
                                }`}
                            >
                                {i + 1}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Current Step Info */}
                <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-4 mb-8">
                    <h3 className="font-bold text-blue-700 dark:text-blue-300 text-lg mb-1">
                        Step {step + 1}: {steps[step].title}
                    </h3>
                    <p className="text-slate-600 dark:text-slate-400">{steps[step].description}</p>
                </div>

                {/* Main Visualization */}
                <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700">
                    {/* Transformer Block Diagram */}
                    <div className="flex flex-col gap-6">
                        {/* Residual Stream */}
                        <div className="relative">
                            {/* Input */}
                            <motion.div
                                className={`p-4 rounded-xl border-2 ${
                                    step === 0
                                        ? 'border-cyan-500 bg-cyan-50 dark:bg-cyan-900/20'
                                        : 'border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-900/50'
                                }`}
                                animate={{ scale: step === 0 ? 1.02 : 1 }}
                            >
                                <div className="text-sm font-semibold text-cyan-700 dark:text-cyan-300 mb-2">
                                    Residual Stream (Input)
                                </div>
                                <VectorBar
                                    values={vectors.initial}
                                    label="x"
                                    color="#06b6d4"
                                    showNorm={true}
                                />
                            </motion.div>

                            <div className="flex justify-center my-4">
                                <ArrowRight className="text-slate-400 rotate-90" size={24} />
                            </div>

                            {/* Attention Block */}
                            <motion.div
                                className={`p-4 rounded-xl border-2 ${
                                    step === 1 || step === 2
                                        ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                                        : 'border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-900/50'
                                }`}
                                animate={{ scale: step === 1 || step === 2 ? 1.02 : 1 }}
                            >
                                <div className="flex items-center gap-2 mb-3">
                                    <Layers className="text-purple-500" size={20} />
                                    <div className="text-sm font-semibold text-purple-700 dark:text-purple-300">
                                        Self-Attention
                                    </div>
                                </div>

                                {step >= 1 && (
                                    <motion.div
                                        initial={{ opacity: 0, y: -10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                    >
                                        <VectorBar
                                            values={vectors.attnOutput}
                                            label="Attn(x)"
                                            color="#a855f7"
                                        />
                                    </motion.div>
                                )}

                                {step >= 2 && (
                                    <motion.div
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        className="mt-4 p-3 bg-emerald-100 dark:bg-emerald-900/30 rounded-lg"
                                    >
                                        <div className="flex items-center gap-2 mb-2">
                                            <Plus className="text-emerald-500" size={16} />
                                            <span className="text-sm font-medium text-emerald-700 dark:text-emerald-300">
                                                Residual Addition: x + Attn(x)
                                            </span>
                                        </div>
                                        <VectorBar
                                            values={vectors.afterAttn}
                                            label="x + Attn"
                                            color="#10b981"
                                            showNorm={true}
                                        />
                                    </motion.div>
                                )}
                            </motion.div>

                            {step >= 3 && (
                                <>
                                    <div className="flex justify-center my-4">
                                        <ArrowRight className="text-slate-400 rotate-90" size={24} />
                                    </div>

                                    {/* LayerNorm 1 */}
                                    <motion.div
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        className={`p-4 rounded-xl border-2 ${
                                            step === 3
                                                ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/20'
                                                : 'border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-900/50'
                                        }`}
                                    >
                                        <div className="flex items-center gap-2 mb-3">
                                            <Scale className="text-amber-500" size={20} />
                                            <div className="text-sm font-semibold text-amber-700 dark:text-amber-300">
                                                Layer Normalization
                                            </div>
                                        </div>
                                        <div className="text-xs text-slate-500 dark:text-slate-400 mb-2">
                                            Rescales to unit norm while preserving direction
                                        </div>
                                        <VectorBar
                                            values={vectors.normAfterAttn}
                                            label="LN(x)"
                                            color="#f59e0b"
                                            showNorm={true}
                                        />
                                    </motion.div>
                                </>
                            )}

                            {step >= 4 && (
                                <>
                                    <div className="flex justify-center my-4">
                                        <ArrowRight className="text-slate-400 rotate-90" size={24} />
                                    </div>

                                    {/* FFN Block */}
                                    <motion.div
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        className={`p-4 rounded-xl border-2 ${
                                            step === 4 || step === 5
                                                ? 'border-orange-500 bg-orange-50 dark:bg-orange-900/20'
                                                : 'border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-900/50'
                                        }`}
                                    >
                                        <div className="flex items-center gap-2 mb-3">
                                            <Layers className="text-orange-500" size={20} />
                                            <div className="text-sm font-semibold text-orange-700 dark:text-orange-300">
                                                Feed-Forward Network (MLP)
                                            </div>
                                        </div>

                                        <VectorBar
                                            values={vectors.ffnOutput}
                                            label="FFN(x)"
                                            color="#f97316"
                                        />

                                        {step >= 5 && (
                                            <motion.div
                                                initial={{ opacity: 0 }}
                                                animate={{ opacity: 1 }}
                                                className="mt-4 p-3 bg-emerald-100 dark:bg-emerald-900/30 rounded-lg"
                                            >
                                                <div className="flex items-center gap-2 mb-2">
                                                    <Plus className="text-emerald-500" size={16} />
                                                    <span className="text-sm font-medium text-emerald-700 dark:text-emerald-300">
                                                        Residual Addition: x + FFN(x)
                                                    </span>
                                                </div>
                                                <VectorBar
                                                    values={vectors.afterFFN}
                                                    label="x + FFN"
                                                    color="#10b981"
                                                    showNorm={true}
                                                />
                                            </motion.div>
                                        )}
                                    </motion.div>
                                </>
                            )}

                            {step >= 6 && (
                                <>
                                    <div className="flex justify-center my-4">
                                        <ArrowRight className="text-slate-400 rotate-90" size={24} />
                                    </div>

                                    {/* Final LayerNorm */}
                                    <motion.div
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        className="p-4 rounded-xl border-2 border-indigo-500 bg-indigo-50 dark:bg-indigo-900/20"
                                    >
                                        <div className="flex items-center gap-2 mb-3">
                                            <Scale className="text-indigo-500" size={20} />
                                            <div className="text-sm font-semibold text-indigo-700 dark:text-indigo-300">
                                                Output (Ready for Next Block)
                                            </div>
                                        </div>
                                        <VectorBar
                                            values={vectors.normAfterFFN}
                                            label="Output"
                                            color="#6366f1"
                                            showNorm={true}
                                        />
                                    </motion.div>
                                </>
                            )}
                        </div>
                    </div>
                </div>

                {/* Key Insight */}
                <motion.div
                    className="mt-8 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-2xl p-6 text-white"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                >
                    <h4 className="font-bold text-xl mb-3">The Mathematics of "Add"</h4>
                    <div className="grid md:grid-cols-2 gap-4">
                        <div className="bg-white/10 rounded-lg p-4">
                            <div className="font-mono text-lg mb-2">
                                x<sub>new</sub> = x + f(x)
                            </div>
                            <p className="text-sm text-blue-100">
                                Vector addition means the original information (x) is always preserved.
                                The layer can only add new features, never remove old ones.
                            </p>
                        </div>
                        <div className="bg-white/10 rounded-lg p-4">
                            <div className="font-mono text-lg mb-2">
                                LayerNorm(x) = x / ||x||
                            </div>
                            <p className="text-sm text-blue-100">
                                Normalization keeps vectors at consistent scale without changing their
                                direction. This stabilizes training in very deep networks.
                            </p>
                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
