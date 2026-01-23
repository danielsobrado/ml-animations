import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowDown, Plus, Layers, AlertTriangle, CheckCircle, Lightbulb } from 'lucide-react';

export default function ConceptPanel() {
    const [showComparison, setShowComparison] = useState(false);
    const [animationStep, setAnimationStep] = useState(0);

    useEffect(() => {
        const timer = setInterval(() => {
            setAnimationStep(prev => (prev + 1) % 4);
        }, 2000);
        return () => clearInterval(timer);
    }, []);

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-5xl w-full">
                {/* Header */}
                <div className="text-center mb-12">
                    <h2 className="text-3xl font-bold text-cyan-600 dark:text-cyan-400 mb-4">
                        The Residual Stream: A Highway for Information
                    </h2>
                    <p className="text-lg text-slate-700 dark:text-slate-300 leading-relaxed max-w-3xl mx-auto">
                        In transformers, each token has a <strong>residual stream</strong> - a vector that flows through
                        all layers. Instead of overwriting this vector, each layer <strong>adds</strong> to it.
                        This is the secret to why deep networks work!
                    </p>
                </div>

                {/* The Two Approaches */}
                <div className="grid md:grid-cols-2 gap-8 mb-12">
                    {/* Without Residuals */}
                    <div className="bg-red-50 dark:bg-red-900/20 border-2 border-red-200 dark:border-red-800 rounded-2xl p-6">
                        <div className="flex items-center gap-3 mb-4">
                            <AlertTriangle className="text-red-500" size={28} />
                            <div>
                                <h3 className="text-xl font-bold text-red-600 dark:text-red-400">Without Residuals</h3>
                                <p className="text-sm text-red-500">Information gets overwritten</p>
                            </div>
                        </div>

                        <div className="flex flex-col items-center gap-2 mb-4">
                            {/* Layer visualization */}
                            {['Input', 'Layer 1', 'Layer 2', 'Layer 3'].map((layer, i) => (
                                <React.Fragment key={layer}>
                                    <motion.div
                                        className={`w-full py-3 rounded-lg text-center font-medium ${
                                            i === 0
                                                ? 'bg-blue-400 dark:bg-blue-600 text-white'
                                                : animationStep >= i
                                                    ? 'bg-red-400 dark:bg-red-600 text-white'
                                                    : 'bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300'
                                        }`}
                                        animate={{
                                            opacity: animationStep >= i || i === 0 ? 1 : 0.5,
                                            scale: animationStep === i && i > 0 ? 1.02 : 1
                                        }}
                                    >
                                        {layer}
                                        {animationStep >= i && i > 0 && (
                                            <span className="ml-2 text-xs opacity-80">
                                                (overwrites previous)
                                            </span>
                                        )}
                                    </motion.div>
                                    {i < 3 && (
                                        <motion.div
                                            animate={{
                                                opacity: animationStep > i ? 1 : 0.3,
                                                y: animationStep === i ? [0, 5, 0] : 0
                                            }}
                                            transition={{ duration: 0.5 }}
                                        >
                                            <ArrowDown className="text-red-400" size={20} />
                                        </motion.div>
                                    )}
                                </React.Fragment>
                            ))}
                        </div>

                        <div className="bg-red-100 dark:bg-red-900/40 rounded-lg p-3 text-sm text-red-700 dark:text-red-300">
                            <strong>Problem:</strong> Early information is lost! The model "forgets" what it learned
                            in earlier layers.
                        </div>
                    </div>

                    {/* With Residuals */}
                    <div className="bg-emerald-50 dark:bg-emerald-900/20 border-2 border-emerald-200 dark:border-emerald-800 rounded-2xl p-6">
                        <div className="flex items-center gap-3 mb-4">
                            <CheckCircle className="text-emerald-500" size={28} />
                            <div>
                                <h3 className="text-xl font-bold text-emerald-600 dark:text-emerald-400">With Residual Stream</h3>
                                <p className="text-sm text-emerald-500">Information accumulates</p>
                            </div>
                        </div>

                        <div className="flex flex-col items-center gap-2 mb-4">
                            {/* Stream visualization */}
                            {['Input', 'Layer 1', 'Layer 2', 'Layer 3'].map((layer, i) => (
                                <React.Fragment key={layer}>
                                    <div className="w-full flex items-center gap-2">
                                        {/* Main stream */}
                                        <motion.div
                                            className="flex-1 py-3 rounded-lg text-center font-medium bg-gradient-to-r from-cyan-400 to-blue-500 text-white"
                                            animate={{
                                                boxShadow: animationStep >= i
                                                    ? `0 0 ${10 + i * 5}px rgba(34, 211, 238, ${0.3 + i * 0.1})`
                                                    : 'none'
                                            }}
                                        >
                                            {i === 0 ? 'x' : `x + Σ layers`}
                                        </motion.div>

                                        {/* Addition from layer */}
                                        {i > 0 && (
                                            <motion.div
                                                className="flex items-center gap-1"
                                                initial={{ opacity: 0, x: 20 }}
                                                animate={{
                                                    opacity: animationStep >= i ? 1 : 0.3,
                                                    x: 0
                                                }}
                                            >
                                                <Plus className="text-emerald-500" size={16} />
                                                <span className="text-xs font-mono bg-emerald-200 dark:bg-emerald-800 px-2 py-1 rounded text-emerald-700 dark:text-emerald-300">
                                                    f{i}(x)
                                                </span>
                                            </motion.div>
                                        )}
                                    </div>
                                    {i < 3 && (
                                        <motion.div
                                            animate={{
                                                opacity: animationStep > i ? 1 : 0.3,
                                                y: animationStep === i ? [0, 5, 0] : 0
                                            }}
                                        >
                                            <ArrowDown className="text-emerald-400" size={20} />
                                        </motion.div>
                                    )}
                                </React.Fragment>
                            ))}
                        </div>

                        <div className="bg-emerald-100 dark:bg-emerald-900/40 rounded-lg p-3 text-sm text-emerald-700 dark:text-emerald-300">
                            <strong>Solution:</strong> Each layer adds its contribution. Original information is
                            preserved throughout!
                        </div>
                    </div>
                </div>

                {/* The Formula */}
                <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700 mb-8">
                    <h3 className="text-xl font-bold text-slate-700 dark:text-slate-200 mb-4 flex items-center gap-2">
                        <Layers className="text-cyan-500" size={24} />
                        The Residual Formula
                    </h3>
                    <div className="grid md:grid-cols-2 gap-6">
                        <div className="bg-slate-50 dark:bg-slate-900/50 rounded-xl p-4">
                            <div className="text-center font-mono text-lg mb-2">
                                <span className="text-cyan-600 dark:text-cyan-400">x</span>
                                <span className="text-slate-500 mx-2">=</span>
                                <span className="text-cyan-600 dark:text-cyan-400">x</span>
                                <span className="text-emerald-500 mx-2">+</span>
                                <span className="text-indigo-500">f(x)</span>
                            </div>
                            <p className="text-sm text-slate-600 dark:text-slate-400 text-center">
                                Output = Input + Layer's Contribution
                            </p>
                        </div>
                        <div className="bg-slate-50 dark:bg-slate-900/50 rounded-xl p-4">
                            <div className="text-center font-mono text-lg mb-2">
                                <span className="text-cyan-600 dark:text-cyan-400">x<sub>out</sub></span>
                                <span className="text-slate-500 mx-2">=</span>
                                <span className="text-cyan-600 dark:text-cyan-400">x<sub>in</sub></span>
                                <span className="text-slate-500 mx-1">+</span>
                                <span className="text-purple-500">Attn(x)</span>
                                <span className="text-slate-500 mx-1">+</span>
                                <span className="text-orange-500">FFN(x)</span>
                            </div>
                            <p className="text-sm text-slate-600 dark:text-slate-400 text-center">
                                In Transformers: Add Attention + MLP outputs
                            </p>
                        </div>
                    </div>
                </div>

                {/* Why This Matters */}
                <motion.div
                    className="bg-gradient-to-r from-cyan-500 to-blue-500 rounded-2xl p-6 text-white"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <div className="flex items-start gap-4">
                        <Lightbulb size={36} className="flex-shrink-0 mt-1" />
                        <div>
                            <h4 className="font-bold text-xl mb-3">Why This Design is Brilliant</h4>
                            <ul className="space-y-2 text-cyan-100">
                                <li className="flex items-start gap-2">
                                    <CheckCircle size={18} className="flex-shrink-0 mt-0.5" />
                                    <span>
                                        <strong>No Overwriting:</strong> Information from layer 1 can directly influence layer 96.
                                        The model never "forgets" early features.
                                    </span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <CheckCircle size={18} className="flex-shrink-0 mt-0.5" />
                                    <span>
                                        <strong>Gradient Highway:</strong> Gradients can flow directly back through the residual
                                        connection, solving the vanishing gradient problem.
                                    </span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <CheckCircle size={18} className="flex-shrink-0 mt-0.5" />
                                    <span>
                                        <strong>Depth Works:</strong> This is why we can train 96-layer GPT-4 or 80-layer Llama models.
                                        Without residuals, 6 layers would be the limit!
                                    </span>
                                </li>
                            </ul>
                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
