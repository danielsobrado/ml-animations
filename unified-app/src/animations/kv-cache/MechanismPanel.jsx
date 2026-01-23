import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Database, ArrowRight, Layers } from 'lucide-react';

export default function MechanismPanel() {
    const [currentToken, setCurrentToken] = useState(3);

    const tokens = ["The", "cat", "sat", "on", "the", "mat"];
    const dimK = 64;  // Dimension of K/V vectors (simplified)

    return (
        <div className="p-8 h-full overflow-y-auto">
            <div className="max-w-5xl mx-auto">
                {/* Header */}
                <div className="text-center mb-12">
                    <h2 className="text-3xl font-bold text-emerald-600 dark:text-emerald-400 mb-4">
                        How KV Cache Works
                    </h2>
                    <p className="text-slate-600 dark:text-slate-300 max-w-2xl mx-auto">
                        The cache stores Key and Value matrices from previous tokens.
                        When generating a new token, we only compute K and V for that token,
                        then <strong>append</strong> them to the cache.
                    </p>
                </div>

                {/* Token Slider */}
                <div className="bg-slate-100 dark:bg-slate-800 rounded-xl p-6 mb-8">
                    <label className="flex justify-between text-sm font-medium mb-3 dark:text-slate-300">
                        <span>Current Generation Step</span>
                        <span className="text-emerald-600">Token {currentToken + 1}: "{tokens[currentToken]}"</span>
                    </label>
                    <input
                        type="range"
                        min="0"
                        max={tokens.length - 1}
                        value={currentToken}
                        onChange={(e) => setCurrentToken(parseInt(e.target.value))}
                        className="w-full accent-emerald-500"
                    />
                    <div className="flex justify-between text-xs text-slate-500 mt-1">
                        <span>Step 1</span>
                        <span>Step {tokens.length}</span>
                    </div>
                </div>

                {/* Main Visualization */}
                <div className="grid lg:grid-cols-3 gap-6 mb-8">
                    {/* Input Token */}
                    <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700">
                        <h3 className="font-bold text-slate-700 dark:text-slate-200 mb-4 flex items-center gap-2">
                            <span className="w-6 h-6 bg-emerald-500 text-white rounded-full flex items-center justify-center text-sm">1</span>
                            New Token Input
                        </h3>
                        <div className="flex flex-col items-center gap-4">
                            <motion.div
                                key={currentToken}
                                initial={{ scale: 0.8, opacity: 0 }}
                                animate={{ scale: 1, opacity: 1 }}
                                className="px-6 py-4 bg-emerald-500 text-white rounded-xl font-mono text-xl font-bold shadow-lg"
                            >
                                {tokens[currentToken]}
                            </motion.div>
                            <ArrowRight className="text-slate-400" />
                            <div className="text-center">
                                <div className="text-xs text-slate-500 mb-1">Compute</div>
                                <div className="flex gap-2">
                                    <div className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded font-mono text-sm">
                                        K<sub>new</sub>
                                    </div>
                                    <div className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 rounded font-mono text-sm">
                                        V<sub>new</sub>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* KV Cache */}
                    <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700">
                        <h3 className="font-bold text-slate-700 dark:text-slate-200 mb-4 flex items-center gap-2">
                            <span className="w-6 h-6 bg-emerald-500 text-white rounded-full flex items-center justify-center text-sm">2</span>
                            <Database size={18} className="text-emerald-500" />
                            KV Cache (Memory)
                        </h3>

                        {/* K Cache */}
                        <div className="mb-4">
                            <div className="text-xs font-medium text-blue-600 dark:text-blue-400 mb-2">Key Cache</div>
                            <div className="flex gap-1">
                                {tokens.slice(0, currentToken + 1).map((_, i) => (
                                    <motion.div
                                        key={`k-${i}`}
                                        initial={{ width: 0, opacity: 0 }}
                                        animate={{ width: 'auto', opacity: 1 }}
                                        transition={{ delay: i * 0.1 }}
                                        className={`h-8 w-8 rounded flex items-center justify-center text-xs font-mono ${i === currentToken
                                                ? 'bg-blue-500 text-white ring-2 ring-blue-300'
                                                : 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                                            }`}
                                    >
                                        K{i}
                                    </motion.div>
                                ))}
                            </div>
                        </div>

                        {/* V Cache */}
                        <div>
                            <div className="text-xs font-medium text-purple-600 dark:text-purple-400 mb-2">Value Cache</div>
                            <div className="flex gap-1">
                                {tokens.slice(0, currentToken + 1).map((_, i) => (
                                    <motion.div
                                        key={`v-${i}`}
                                        initial={{ width: 0, opacity: 0 }}
                                        animate={{ width: 'auto', opacity: 1 }}
                                        transition={{ delay: i * 0.1 }}
                                        className={`h-8 w-8 rounded flex items-center justify-center text-xs font-mono ${i === currentToken
                                                ? 'bg-purple-500 text-white ring-2 ring-purple-300'
                                                : 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400'
                                            }`}
                                    >
                                        V{i}
                                    </motion.div>
                                ))}
                            </div>
                        </div>

                        {/* Memory Size */}
                        <div className="mt-4 p-3 bg-slate-100 dark:bg-slate-700/50 rounded-lg">
                            <div className="text-xs text-slate-500 mb-1">Cache Size</div>
                            <div className="font-mono text-sm">
                                2 × {currentToken + 1} × {dimK} = <span className="text-emerald-600 font-bold">{2 * (currentToken + 1) * dimK}</span> floats
                            </div>
                        </div>
                    </div>

                    {/* Attention Computation */}
                    <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700">
                        <h3 className="font-bold text-slate-700 dark:text-slate-200 mb-4 flex items-center gap-2">
                            <span className="w-6 h-6 bg-emerald-500 text-white rounded-full flex items-center justify-center text-sm">3</span>
                            Attention Scores
                        </h3>

                        <div className="space-y-2">
                            <div className="text-xs text-slate-500 mb-2">Q<sub>new</sub> attends to all cached Keys:</div>
                            {tokens.slice(0, currentToken + 1).map((token, i) => {
                                const score = i === currentToken ? 0.4 : 0.6 / currentToken;
                                return (
                                    <motion.div
                                        key={i}
                                        initial={{ x: -20, opacity: 0 }}
                                        animate={{ x: 0, opacity: 1 }}
                                        transition={{ delay: i * 0.05 }}
                                        className="flex items-center gap-2"
                                    >
                                        <span className="font-mono text-xs w-12 text-slate-500">{token}</span>
                                        <div className="flex-1 bg-slate-100 dark:bg-slate-700 rounded h-4 overflow-hidden">
                                            <motion.div
                                                initial={{ width: 0 }}
                                                animate={{ width: `${score * 100}%` }}
                                                transition={{ delay: 0.3 + i * 0.05 }}
                                                className={`h-full ${i === currentToken ? 'bg-emerald-500' : 'bg-blue-400'}`}
                                            />
                                        </div>
                                        <span className="text-xs font-mono w-10 text-right text-slate-500">
                                            {(score * 100).toFixed(0)}%
                                        </span>
                                    </motion.div>
                                );
                            })}
                        </div>
                    </div>
                </div>

                {/* Formula */}
                <div className="bg-gradient-to-r from-slate-800 to-slate-900 rounded-2xl p-6 text-white">
                    <div className="flex items-center gap-4">
                        <Layers size={32} className="text-emerald-400" />
                        <div>
                            <h4 className="font-bold text-lg mb-2">Memory Formula per Layer</h4>
                            <code className="text-emerald-300 font-mono">
                                Memory = 2 × seq_len × num_heads × head_dim × sizeof(float)
                            </code>
                            <p className="text-slate-400 text-sm mt-2">
                                For a 7B model with 32 layers: each token adds ~1MB to the cache!
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
