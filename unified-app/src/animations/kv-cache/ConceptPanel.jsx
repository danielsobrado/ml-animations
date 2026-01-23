import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, CheckCircle, Zap, Database } from 'lucide-react';

export default function ConceptPanel() {
    const [step, setStep] = useState(0);

    const tokens = ["The", "cat", "sat", "on", "the", "mat"];

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-4xl w-full">
                {/* Header */}
                <div className="text-center mb-12">
                    <h2 className="text-3xl font-bold text-emerald-600 dark:text-emerald-400 mb-4">
                        The Problem: Redundant Computation
                    </h2>
                    <p className="text-lg text-slate-700 dark:text-slate-300 leading-relaxed max-w-2xl mx-auto">
                        When generating text <strong>token by token</strong>, without caching we recompute
                        the Keys and Values for <em>every previous token</em> at each step.
                    </p>
                </div>

                {/* Step Control */}
                <div className="flex justify-center gap-4 mb-8">
                    <button
                        onClick={() => setStep(Math.max(0, step - 1))}
                        disabled={step === 0}
                        className="px-4 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg disabled:opacity-50 font-medium"
                    >
                        ← Previous
                    </button>
                    <div className="flex items-center gap-2 px-4">
                        <span className="text-slate-500 dark:text-slate-400">Token:</span>
                        <span className="font-bold text-emerald-600 dark:text-emerald-400 text-xl">{step + 1} / {tokens.length}</span>
                    </div>
                    <button
                        onClick={() => setStep(Math.min(tokens.length - 1, step + 1))}
                        disabled={step === tokens.length - 1}
                        className="px-4 py-2 bg-emerald-600 text-white rounded-lg disabled:opacity-50 font-medium hover:bg-emerald-700"
                    >
                        Next →
                    </button>
                </div>

                {/* Comparison Grid */}
                <div className="grid md:grid-cols-2 gap-8">
                    {/* Without Cache */}
                    <div className="bg-red-50 dark:bg-red-900/20 border-2 border-red-200 dark:border-red-800 rounded-2xl p-6">
                        <div className="flex items-center gap-3 mb-4">
                            <AlertTriangle className="text-red-500" size={24} />
                            <h3 className="text-xl font-bold text-red-600 dark:text-red-400">Without KV Cache</h3>
                        </div>
                        <p className="text-sm text-slate-600 dark:text-slate-400 mb-6">
                            Recomputes K and V for all tokens every step. O(n²) complexity!
                        </p>

                        <div className="flex flex-wrap gap-2 mb-4">
                            {tokens.slice(0, step + 1).map((token, i) => (
                                <motion.div
                                    key={i}
                                    initial={{ scale: 0 }}
                                    animate={{ scale: 1 }}
                                    className={`px-3 py-2 rounded-lg font-mono text-sm ${i === step
                                            ? 'bg-emerald-500 text-white ring-2 ring-emerald-300'
                                            : 'bg-red-200 dark:bg-red-800 text-red-800 dark:text-red-200'
                                        }`}
                                >
                                    {token}
                                    {i < step && (
                                        <span className="ml-1 text-xs">🔄</span>
                                    )}
                                </motion.div>
                            ))}
                        </div>

                        <div className="bg-red-100 dark:bg-red-900/40 rounded-lg p-4">
                            <div className="text-sm font-medium text-red-700 dark:text-red-300 mb-2">
                                Computations for step {step + 1}:
                            </div>
                            <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                                {step + 1} × (K + V) = {(step + 1) * 2} ops
                            </div>
                            <div className="text-xs text-red-500 mt-2">
                                Total so far: {((step + 1) * (step + 2))} operations
                            </div>
                        </div>
                    </div>

                    {/* With Cache */}
                    <div className="bg-emerald-50 dark:bg-emerald-900/20 border-2 border-emerald-200 dark:border-emerald-800 rounded-2xl p-6">
                        <div className="flex items-center gap-3 mb-4">
                            <CheckCircle className="text-emerald-500" size={24} />
                            <h3 className="text-xl font-bold text-emerald-600 dark:text-emerald-400">With KV Cache</h3>
                        </div>
                        <p className="text-sm text-slate-600 dark:text-slate-400 mb-6">
                            Stores K and V, only computes for the new token. O(n) complexity!
                        </p>

                        <div className="flex flex-wrap gap-2 mb-4">
                            {tokens.slice(0, step + 1).map((token, i) => (
                                <motion.div
                                    key={i}
                                    initial={{ scale: 0 }}
                                    animate={{ scale: 1 }}
                                    className={`px-3 py-2 rounded-lg font-mono text-sm ${i === step
                                            ? 'bg-emerald-500 text-white ring-2 ring-emerald-300'
                                            : 'bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 border border-emerald-300 dark:border-emerald-700'
                                        }`}
                                >
                                    {token}
                                    {i < step && (
                                        <span className="ml-1 text-xs">💾</span>
                                    )}
                                </motion.div>
                            ))}
                        </div>

                        <div className="bg-emerald-100 dark:bg-emerald-900/40 rounded-lg p-4">
                            <div className="text-sm font-medium text-emerald-700 dark:text-emerald-300 mb-2">
                                Computations for step {step + 1}:
                            </div>
                            <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">
                                1 × (K + V) = 2 ops
                            </div>
                            <div className="text-xs text-emerald-500 mt-2">
                                Total so far: {(step + 1) * 2} operations
                            </div>
                        </div>
                    </div>
                </div>

                {/* Summary Card */}
                <motion.div
                    className="mt-8 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-2xl p-6 text-white"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <div className="flex items-center gap-4">
                        <Zap size={32} />
                        <div>
                            <h4 className="font-bold text-lg">Savings at step {step + 1}</h4>
                            <p className="text-emerald-100">
                                KV Cache saves <strong>{((step + 1) * (step + 2)) - ((step + 1) * 2)}</strong> operations
                                ({Math.round((1 - ((step + 1) * 2) / ((step + 1) * (step + 2))) * 100)}% reduction)
                            </p>
                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
