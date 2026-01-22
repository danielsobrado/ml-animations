import React, { useState, useEffect } from 'react';
import { Brain, Database, Zap, AlertTriangle, CheckCircle2, ArrowRight, Lightbulb } from 'lucide-react';

export default function ConceptPanel() {
    const [comparisonMode, setComparisonMode] = useState('full');
    const [animStep, setAnimStep] = useState(0);

    useEffect(() => {
        const timer = setInterval(() => {
            setAnimStep(s => (s + 1) % 4);
        }, 1500);
        return () => clearInterval(timer);
    }, []);

    return (
        <div className="p-8 h-full">
            <div className="max-w-4xl mx-auto">
                {/* Introduction */}
                <div className="text-center mb-8">
                    <h2 className="text-3xl font-bold text-purple-900 mb-4">Why Fine-Tune?</h2>
                    <p className="text-lg text-slate-700 leading-relaxed max-w-2xl mx-auto">
                        Pre-trained LLMs are powerful but general. <strong>Fine-tuning</strong> adapts them 
                        to specific tasks, domains, or styles - making them experts at what YOU need.
                    </p>
                </div>

                {/* The Problem */}
                <div className="bg-red-50 rounded-xl p-6 mb-8 border border-red-200">
                    <div className="flex items-start gap-4">
                        <AlertTriangle className="text-red-500 flex-shrink-0 mt-1" size={28} />
                        <div>
                            <h3 className="text-xl font-bold text-red-900 mb-2">The Challenge of Full Fine-Tuning</h3>
                            <ul className="text-red-800 space-y-2">
                                <li>• <strong>Massive Memory:</strong> LLaMA-7B needs ~28GB just for weights, plus optimizer states = 100GB+</li>
                                <li>• <strong>Slow Training:</strong> Updating billions of parameters takes days/weeks</li>
                                <li>• <strong>Expensive Storage:</strong> Each fine-tuned model = full copy of all weights</li>
                                <li>• <strong>Catastrophic Forgetting:</strong> May lose general capabilities</li>
                            </ul>
                        </div>
                    </div>
                </div>

                {/* PEFT Solution */}
                <div className="bg-green-50 rounded-xl p-6 mb-8 border border-green-200">
                    <div className="flex items-start gap-4">
                        <Zap className="text-green-500 flex-shrink-0 mt-1" size={28} />
                        <div>
                            <h3 className="text-xl font-bold text-green-900 mb-2">PEFT to the Rescue!</h3>
                            <p className="text-green-800 mb-4">
                                <strong>Parameter-Efficient Fine-Tuning</strong> updates only a tiny fraction of 
                                parameters while keeping the rest frozen. Same results, fraction of the cost!
                            </p>
                            <div className="grid grid-cols-4 gap-3 text-center">
                                {[
                                    { name: 'LoRA', desc: 'Low-rank adapters' },
                                    { name: 'QLoRA', desc: '4-bit + LoRA' },
                                    { name: 'Prefix', desc: 'Learned prompts' },
                                    { name: 'Adapter', desc: 'Bottleneck layers' },
                                ].map((method, i) => (
                                    <div 
                                        key={method.name}
                                        className={`bg-white p-3 rounded-lg border transition-all ${
                                            animStep === i ? 'ring-2 ring-green-500 scale-105' : ''
                                        }`}
                                    >
                                        <div className="font-bold text-green-700">{method.name}</div>
                                        <div className="text-xs text-green-600">{method.desc}</div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Comparison */}
                <div className="bg-slate-50 rounded-xl p-6 mb-8">
                    <h3 className="text-xl font-bold text-slate-800 mb-4 text-center">Compare Approaches</h3>
                    
                    <div className="flex justify-center gap-4 mb-6">
                        <button
                            onClick={() => setComparisonMode('full')}
                            className={`px-4 py-2 rounded-lg font-medium transition-all ${
                                comparisonMode === 'full'
                                    ? 'bg-red-500 text-white'
                                    : 'bg-white border hover:bg-slate-100'
                            }`}
                        >
                            Full Fine-Tuning
                        </button>
                        <button
                            onClick={() => setComparisonMode('peft')}
                            className={`px-4 py-2 rounded-lg font-medium transition-all ${
                                comparisonMode === 'peft'
                                    ? 'bg-green-500 text-white'
                                    : 'bg-white border hover:bg-slate-100'
                            }`}
                        >
                            PEFT (LoRA)
                        </button>
                    </div>

                    <div className="bg-white rounded-lg p-4 border">
                        {/* Model Visualization */}
                        <div className="flex items-center justify-center gap-4 mb-6">
                            <div className="text-center">
                                <div className="text-4xl mb-2">🧠</div>
                                <div className="font-bold text-slate-800">LLaMA-7B</div>
                            </div>
                            <ArrowRight className="text-slate-800 dark:text-slate-400" size={32} />
                            <div className="relative">
                                <div className="grid grid-cols-8 gap-1">
                                    {Array.from({ length: 32 }).map((_, i) => (
                                        <div
                                            key={i}
                                            className={`w-6 h-6 rounded transition-all ${
                                                comparisonMode === 'full'
                                                    ? 'bg-red-400'
                                                    : i % 8 === 0
                                                        ? 'bg-green-400 animate-pulse'
                                                        : 'bg-slate-200'
                                            }`}
                                        />
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Stats Comparison */}
                        <div className="grid grid-cols-2 gap-4">
                            <div className={`p-4 rounded-lg ${comparisonMode === 'full' ? 'bg-red-50 border-red-200' : 'bg-slate-50'} border`}>
                                <h4 className="font-bold text-slate-800 mb-3">Full Fine-Tuning</h4>
                                <div className="space-y-2 text-sm">
                                    <div className="flex justify-between">
                                        <span>Trainable Parameters:</span>
                                        <span className="font-mono text-red-600">7B (100%)</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>GPU Memory:</span>
                                        <span className="font-mono text-red-600">~100GB</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Storage per Model:</span>
                                        <span className="font-mono text-red-600">~14GB</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Training Time:</span>
                                        <span className="font-mono text-red-600">Days</span>
                                    </div>
                                </div>
                            </div>
                            <div className={`p-4 rounded-lg ${comparisonMode === 'peft' ? 'bg-green-50 border-green-200' : 'bg-slate-50'} border`}>
                                <h4 className="font-bold text-slate-800 mb-3">LoRA Fine-Tuning</h4>
                                <div className="space-y-2 text-sm">
                                    <div className="flex justify-between">
                                        <span>Trainable Parameters:</span>
                                        <span className="font-mono text-green-600">~4M (0.06%)</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>GPU Memory:</span>
                                        <span className="font-mono text-green-600">~16GB</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Storage per Model:</span>
                                        <span className="font-mono text-green-600">~8MB</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Training Time:</span>
                                        <span className="font-mono text-green-600">Hours</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Use Cases */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
                    {[
                        { icon: '💬', title: 'Chatbots', desc: 'Custom personas' },
                        { icon: '📊', title: 'Domain', desc: 'Medical, legal, etc.' },
                        { icon: '🌐', title: 'Languages', desc: 'Low-resource langs' },
                        { icon: '🎨', title: 'Style', desc: 'Tone & format' },
                    ].map((use, i) => (
                        <div key={i} className="bg-white rounded-lg p-4 border text-center hover:shadow-md transition-shadow">
                            <div className="text-3xl mb-2">{use.icon}</div>
                            <div className="font-bold text-slate-800">{use.title}</div>
                            <div className="text-xs text-slate-800 dark:text-slate-600">{use.desc}</div>
                        </div>
                    ))}
                </div>

                {/* Key Insight */}
                <div className="bg-amber-50 p-4 rounded-xl border border-amber-200 flex items-start gap-3">
                    <Lightbulb className="text-amber-600 flex-shrink-0 mt-1" size={24} />
                    <div>
                        <h4 className="font-bold text-amber-900 mb-1">Key Insight</h4>
                        <p className="text-sm">
                            PEFT methods work because neural networks are <strong>over-parameterized</strong>. 
                            The adaptation needed for a new task often lies in a low-dimensional subspace, 
                            so we don't need to update ALL the weights!
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
