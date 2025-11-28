import React, { useState } from 'react';
import { Zap, ChevronRight, Check, X, ArrowRight, Brain, MessageSquare, Search, Image } from 'lucide-react';

export default function VariantsPanel() {
    const [selectedVariant, setSelectedVariant] = useState('encoder-only');

    const variants = {
        'encoder-only': {
            name: 'Encoder-Only',
            icon: Brain,
            color: 'blue',
            examples: ['BERT', 'RoBERTa', 'ALBERT', 'DistilBERT', 'ELECTRA'],
            useCase: 'Understanding / Classification',
            description: 'Bidirectional attention - each position can see all other positions. Great for understanding tasks.',
            tasks: ['Text Classification', 'Named Entity Recognition', 'Question Answering', 'Sentiment Analysis'],
            attention: 'Full bidirectional self-attention',
            training: 'Masked Language Modeling (MLM)',
            diagram: {
                encoder: true,
                decoder: false
            }
        },
        'decoder-only': {
            name: 'Decoder-Only',
            icon: MessageSquare,
            color: 'purple',
            examples: ['GPT-1/2/3/4', 'LLaMA', 'Claude', 'PaLM', 'Mistral', 'Falcon'],
            useCase: 'Generation / Completion',
            description: 'Causal (left-to-right) attention only. Designed for autoregressive text generation.',
            tasks: ['Text Generation', 'Chat/Dialogue', 'Code Generation', 'Creative Writing'],
            attention: 'Causal self-attention (masked)',
            training: 'Next Token Prediction',
            diagram: {
                encoder: false,
                decoder: true
            }
        },
        'encoder-decoder': {
            name: 'Encoder-Decoder',
            icon: ArrowRight,
            color: 'green',
            examples: ['T5', 'BART', 'mT5', 'FLAN-T5', 'mBART'],
            useCase: 'Sequence-to-Sequence',
            description: 'Full transformer as originally proposed. Best for tasks with distinct input and output sequences.',
            tasks: ['Translation', 'Summarization', 'Text-to-Text', 'Data-to-Text'],
            attention: 'Bidirectional (enc) + Causal (dec) + Cross',
            training: 'Seq2Seq / Denoising',
            diagram: {
                encoder: true,
                decoder: true
            }
        }
    };

    const current = variants[selectedVariant];
    const colorClasses = {
        blue: 'bg-blue-500/20 border-blue-500/50 text-blue-400',
        purple: 'bg-purple-500/20 border-purple-500/50 text-purple-400',
        green: 'bg-green-500/20 border-green-500/50 text-green-400'
    };

    return (
        <div className="p-6 min-h-screen">
            <div className="max-w-6xl mx-auto">
                {/* Header */}
                <div className="text-center mb-8">
                    <h2 className="text-3xl font-bold text-white mb-2">
                        Transformer Variants: <span className="gradient-text">The Family Tree</span>
                    </h2>
                    <p className="text-slate-400">
                        Three architectural patterns that dominate modern NLP
                    </p>
                </div>

                {/* Variant Selector */}
                <div className="flex justify-center gap-4 mb-8">
                    {Object.entries(variants).map(([key, variant]) => (
                        <button
                            key={key}
                            onClick={() => setSelectedVariant(key)}
                            className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all ${
                                selectedVariant === key
                                    ? `${colorClasses[variant.color]} border-2`
                                    : 'bg-slate-700/50 text-slate-400 hover:bg-slate-700 border-2 border-transparent'
                            }`}
                        >
                            <variant.icon size={20} />
                            {variant.name}
                        </button>
                    ))}
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Architecture Diagram */}
                    <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                        <h3 className="text-white font-bold mb-4 text-center">{current.name} Architecture</h3>
                        
                        <div className="flex justify-center items-end gap-8 h-64">
                            {/* Encoder */}
                            <div className="flex flex-col items-center">
                                {current.diagram.encoder ? (
                                    <div className="w-24 h-40 bg-gradient-to-t from-green-600 to-green-400 rounded-lg flex flex-col items-center justify-center p-2">
                                        <div className="text-white font-bold text-sm">ENCODER</div>
                                        <div className="text-white/70 text-xs mt-2 text-center">
                                            Bidirectional Attention
                                        </div>
                                    </div>
                                ) : (
                                    <div className="w-24 h-40 bg-slate-700/30 rounded-lg flex flex-col items-center justify-center p-2 border-2 border-dashed border-slate-600">
                                        <X className="text-slate-500" size={32} />
                                        <div className="text-slate-500 text-xs mt-2">Not Used</div>
                                    </div>
                                )}
                                <div className="text-slate-400 text-xs mt-2">Encoder</div>
                            </div>

                            {/* Arrow */}
                            {current.diagram.encoder && current.diagram.decoder && (
                                <div className="flex flex-col items-center justify-center h-40">
                                    <ArrowRight className="text-yellow-400" size={32} />
                                    <div className="text-yellow-400 text-xs mt-1">K, V</div>
                                </div>
                            )}

                            {/* Decoder */}
                            <div className="flex flex-col items-center">
                                {current.diagram.decoder ? (
                                    <div className="w-24 h-48 bg-gradient-to-t from-purple-600 to-purple-400 rounded-lg flex flex-col items-center justify-center p-2">
                                        <div className="text-white font-bold text-sm">DECODER</div>
                                        <div className="text-white/70 text-xs mt-2 text-center">
                                            Causal Attention
                                        </div>
                                        {current.diagram.encoder && (
                                            <div className="text-yellow-300 text-xs mt-1 text-center">
                                                + Cross-Attn
                                            </div>
                                        )}
                                    </div>
                                ) : (
                                    <div className="w-24 h-48 bg-slate-700/30 rounded-lg flex flex-col items-center justify-center p-2 border-2 border-dashed border-slate-600">
                                        <X className="text-slate-500" size={32} />
                                        <div className="text-slate-500 text-xs mt-2">Not Used</div>
                                    </div>
                                )}
                                <div className="text-slate-400 text-xs mt-2">Decoder</div>
                            </div>
                        </div>

                        {/* Description */}
                        <div className="mt-6 p-4 bg-slate-700/30 rounded-lg">
                            <p className="text-slate-300 text-sm">{current.description}</p>
                        </div>
                    </div>

                    {/* Details */}
                    <div className="space-y-4">
                        {/* Use Case */}
                        <div className={`rounded-xl p-4 border ${colorClasses[current.color]}`}>
                            <h4 className="font-bold mb-2">Primary Use Case</h4>
                            <p className="text-white text-lg">{current.useCase}</p>
                        </div>

                        {/* Attention Type */}
                        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                            <h4 className="text-white font-bold mb-2">Attention Pattern</h4>
                            <p className="text-slate-300">{current.attention}</p>
                        </div>

                        {/* Training Objective */}
                        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                            <h4 className="text-white font-bold mb-2">Training Objective</h4>
                            <p className="text-slate-300">{current.training}</p>
                        </div>

                        {/* Tasks */}
                        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                            <h4 className="text-white font-bold mb-2">Common Tasks</h4>
                            <div className="flex flex-wrap gap-2">
                                {current.tasks.map((task, i) => (
                                    <span key={i} className="bg-slate-700 px-3 py-1 rounded-full text-sm text-slate-300">
                                        {task}
                                    </span>
                                ))}
                            </div>
                        </div>

                        {/* Examples */}
                        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                            <h4 className="text-white font-bold mb-2">Popular Models</h4>
                            <div className="flex flex-wrap gap-2">
                                {current.examples.map((model, i) => (
                                    <span 
                                        key={i} 
                                        className={`px-3 py-1 rounded-full text-sm font-medium ${colorClasses[current.color]}`}
                                    >
                                        {model}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Comparison Table */}
                <div className="mt-8 bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                    <h3 className="text-white font-bold mb-4">📊 Quick Comparison</h3>
                    
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-slate-700">
                                    <th className="text-left text-slate-400 py-3 px-4">Feature</th>
                                    <th className="text-center text-blue-400 py-3 px-4">Encoder-Only</th>
                                    <th className="text-center text-purple-400 py-3 px-4">Decoder-Only</th>
                                    <th className="text-center text-green-400 py-3 px-4">Encoder-Decoder</th>
                                </tr>
                            </thead>
                            <tbody className="text-slate-300">
                                <tr className="border-b border-slate-700/50">
                                    <td className="py-3 px-4 text-slate-400">Bidirectional?</td>
                                    <td className="py-3 px-4 text-center"><Check className="inline text-green-400" size={18} /></td>
                                    <td className="py-3 px-4 text-center"><X className="inline text-red-400" size={18} /></td>
                                    <td className="py-3 px-4 text-center"><span className="text-yellow-400">Partial</span></td>
                                </tr>
                                <tr className="border-b border-slate-700/50">
                                    <td className="py-3 px-4 text-slate-400">Autoregressive?</td>
                                    <td className="py-3 px-4 text-center"><X className="inline text-red-400" size={18} /></td>
                                    <td className="py-3 px-4 text-center"><Check className="inline text-green-400" size={18} /></td>
                                    <td className="py-3 px-4 text-center"><Check className="inline text-green-400" size={18} /></td>
                                </tr>
                                <tr className="border-b border-slate-700/50">
                                    <td className="py-3 px-4 text-slate-400">Best for Generation?</td>
                                    <td className="py-3 px-4 text-center"><X className="inline text-red-400" size={18} /></td>
                                    <td className="py-3 px-4 text-center"><Check className="inline text-green-400" size={18} /></td>
                                    <td className="py-3 px-4 text-center"><Check className="inline text-green-400" size={18} /></td>
                                </tr>
                                <tr className="border-b border-slate-700/50">
                                    <td className="py-3 px-4 text-slate-400">Best for Understanding?</td>
                                    <td className="py-3 px-4 text-center"><Check className="inline text-green-400" size={18} /></td>
                                    <td className="py-3 px-4 text-center"><span className="text-yellow-400">OK</span></td>
                                    <td className="py-3 px-4 text-center"><Check className="inline text-green-400" size={18} /></td>
                                </tr>
                                <tr>
                                    <td className="py-3 px-4 text-slate-400">Cross-Attention?</td>
                                    <td className="py-3 px-4 text-center"><X className="inline text-red-400" size={18} /></td>
                                    <td className="py-3 px-4 text-center"><X className="inline text-red-400" size={18} /></td>
                                    <td className="py-3 px-4 text-center"><Check className="inline text-green-400" size={18} /></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Modern Trends */}
                <div className="mt-8 bg-gradient-to-r from-indigo-500/10 to-purple-500/10 rounded-2xl p-6 border border-indigo-500/30">
                    <h3 className="text-indigo-400 font-bold mb-4 flex items-center gap-2">
                        <Zap size={20} />
                        Modern Trends (2023-2024)
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="bg-slate-800/50 p-4 rounded-lg">
                            <h4 className="text-white font-medium mb-2">🦙 Decoder-Only Dominance</h4>
                            <p className="text-slate-400 text-sm">
                                GPT, LLaMA, Mistral - decoder-only models dominate due to simplicity and scaling properties. 
                                They can be prompted to do encoder tasks too!
                            </p>
                        </div>
                        <div className="bg-slate-800/50 p-4 rounded-lg">
                            <h4 className="text-white font-medium mb-2">🔀 Mixture of Experts (MoE)</h4>
                            <p className="text-slate-400 text-sm">
                                Models like Mixtral use sparse MoE layers - only some "experts" activate per token. 
                                More parameters, same compute!
                            </p>
                        </div>
                        <div className="bg-slate-800/50 p-4 rounded-lg">
                            <h4 className="text-white font-medium mb-2">📏 Longer Context</h4>
                            <p className="text-slate-400 text-sm">
                                Techniques like RoPE, ALiBi, and sparse attention enable 100K+ token contexts. 
                                Original transformer: only 512 tokens!
                            </p>
                        </div>
                        <div className="bg-slate-800/50 p-4 rounded-lg">
                            <h4 className="text-white font-medium mb-2">🖼️ Multimodal</h4>
                            <p className="text-slate-400 text-sm">
                                GPT-4V, Gemini, LLaVA - transformers now process images, audio, and text together. 
                                Same architecture, different encoders!
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
