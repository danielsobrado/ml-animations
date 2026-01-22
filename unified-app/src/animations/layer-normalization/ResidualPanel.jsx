import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, Plus } from 'lucide-react';

export default function ResidualPanel() {
    const [useResidual, setUseResidual] = useState(true);
    const [useNorm, setUseNorm] = useState(true);

    // Simulate gradient flow
    const gradientStrength = useResidual ? 1.0 : 0.1;

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-violet-400 mb-4">Residual Connections</h2>
                <p className="text-lg text-slate-700 dark:text-slate-300 leading-relaxed">
                    The <strong>"Add & Norm"</strong> pattern used in every Transformer layer.
                    <br />
                    Residuals allow gradients to flow directly through the network.
                </p>
            </div>

            {/* Controls */}
            <div className="grid md:grid-cols-2 gap-6 w-full max-w-4xl mb-8">
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                    <div className="flex items-center justify-between mb-2">
                        <div>
                            <h3 className="font-bold text-white">Residual Connection</h3>
                            <p className="text-xs text-slate-800 dark:text-slate-400">Skip connection (Add)</p>
                        </div>
                        <button
                            onClick={() => setUseResidual(!useResidual)}
                            className={`relative w-16 h-8 rounded-full transition-colors ${useResidual ? 'bg-green-500' : 'bg-slate-600'
                                }`}
                        >
                            <motion.div
                                className="absolute top-1 left-1 w-6 h-6 bg-white rounded-full shadow-lg"
                                animate={{ x: useResidual ? 32 : 0 }}
                                transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                            />
                        </button>
                    </div>
                </div>

                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                    <div className="flex items-center justify-between mb-2">
                        <div>
                            <h3 className="font-bold text-white">Layer Normalization</h3>
                            <p className="text-xs text-slate-800 dark:text-slate-400">Normalize (Norm)</p>
                        </div>
                        <button
                            onClick={() => setUseNorm(!useNorm)}
                            className={`relative w-16 h-8 rounded-full transition-colors ${useNorm ? 'bg-violet-500' : 'bg-slate-600'
                                }`}
                        >
                            <motion.div
                                className="absolute top-1 left-1 w-6 h-6 bg-white rounded-full shadow-lg"
                                animate={{ x: useNorm ? 32 : 0 }}
                                transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                            />
                        </button>
                    </div>
                </div>
            </div>

            {/* Flow Diagram */}
            <div className="bg-slate-800 p-8 rounded-xl border border-slate-700 w-full max-w-4xl">
                <h3 className="font-bold text-white mb-6 text-center">Data Flow Through Transformer Layer</h3>

                <div className="flex flex-col items-center gap-6">
                    {/* Input */}
                    <div className="bg-cyan-600 px-8 py-4 rounded-xl text-white font-bold text-lg shadow-lg">
                        Input (x)
                    </div>

                    <ArrowRight className="rotate-90 text-slate-800 dark:text-slate-400" size={32} />

                    {/* Sublayer (e.g., Attention or FFN) */}
                    <div className="bg-purple-600 px-8 py-4 rounded-xl text-white font-bold text-lg shadow-lg">
                        Sublayer (Attention / FFN)
                    </div>

                    <ArrowRight className="rotate-90 text-slate-800 dark:text-slate-400" size={32} />

                    {/* Add (Residual) */}
                    {useResidual ? (
                        <div className="relative">
                            <div className="bg-green-600 px-8 py-4 rounded-xl text-white font-bold text-lg shadow-lg flex items-center gap-3">
                                <Plus size={24} />
                                Add (x + sublayer)
                            </div>

                            {/* Skip Connection Arrow */}
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                className="absolute -left-32 top-1/2 transform -translate-y-1/2"
                            >
                                <div className="flex items-center gap-2">
                                    <div className="text-sm font-bold">Skip</div>
                                    <ArrowRight className="text-green-400" size={24} />
                                </div>
                            </motion.div>
                        </div>
                    ) : (
                        <div className="bg-slate-600 px-8 py-4 rounded-xl text-slate-700 dark:text-slate-300 font-bold text-lg opacity-50">
                            (No Residual)
                        </div>
                    )}

                    <ArrowRight className="rotate-90 text-slate-800 dark:text-slate-400" size={32} />

                    {/* Norm */}
                    {useNorm ? (
                        <div className="bg-violet-600 px-8 py-4 rounded-xl text-white font-bold text-lg shadow-lg">
                            Layer Norm
                        </div>
                    ) : (
                        <div className="bg-slate-600 px-8 py-4 rounded-xl text-slate-700 dark:text-slate-300 font-bold text-lg opacity-50">
                            (No Normalization)
                        </div>
                    )}

                    <ArrowRight className="rotate-90 text-slate-800 dark:text-slate-400" size={32} />

                    {/* Output */}
                    <div className="bg-cyan-600 px-8 py-4 rounded-xl text-white font-bold text-lg shadow-lg">
                        Output
                    </div>
                </div>

                {/* Gradient Flow Indicator */}
                <div className="mt-8 pt-6 border-t border-slate-700">
                    <h4 className="font-bold text-white mb-4 text-center">Gradient Flow (Backward Pass)</h4>
                    <div className="flex items-center justify-center gap-4">
                        <div className="text-slate-800 dark:text-slate-400">Weak</div>
                        <div className="flex-1 h-8 bg-slate-700 rounded-full overflow-hidden">
                            <motion.div
                                className="h-full bg-gradient-to-r from-red-500 to-green-500"
                                initial={{ width: '0%' }}
                                animate={{ width: `${gradientStrength * 100}%` }}
                                transition={{ duration: 0.5 }}
                            />
                        </div>
                        <div className="text-slate-800 dark:text-slate-400">Strong</div>
                    </div>
                    <p className="text-sm text-slate-800 dark:text-slate-400 mt-4 text-center">
                        {useResidual
                            ? '✅ Residual connections allow gradients to flow directly!'
                            : '⚠️ Without residuals, gradients vanish in deep networks.'}
                    </p>
                </div>
            </div>

            {/* Formula */}
            <div className="bg-slate-800 p-6 rounded-xl border border-violet-500/50 w-full max-w-4xl mt-8">
                <h4 className="font-bold text-white mb-3 text-center">The Transformer Formula</h4>
                <div className="font-mono text-center text-lg">
                    <span className="text-violet-300">output = LayerNorm(</span>
                    <span className="text-cyan-300">x</span>
                    <span className="text-green-300"> + </span>
                    <span className="text-purple-300">Sublayer(x)</span>
                    <span className="text-violet-300">)</span>
                </div>
                <p className="text-xs text-slate-800 dark:text-slate-400 mt-3 text-center">
                    This pattern appears twice in every Transformer layer: once for attention, once for FFN.
                </p>
            </div>
        </div>
    );
}
