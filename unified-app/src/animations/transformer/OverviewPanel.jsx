import React, { useState } from 'react';
import { Play, Pause, RotateCcw, ArrowRight, ArrowDown, Layers, Eye, Plus, Lightbulb } from 'lucide-react';

export default function OverviewPanel() {
    const [hoveredComponent, setHoveredComponent] = useState(null);
    const [showDetails, setShowDetails] = useState(false);

    const components = {
        input_embedding: {
            name: 'Input Embedding',
            description: 'Converts input tokens to dense vectors (d_model dimensions)',
            color: 'bg-blue-500',
            details: 'Each token ID is mapped to a learnable vector. For vocab size V and dimension d, this is a V×d matrix.'
        },
        positional_encoding: {
            name: 'Positional Encoding',
            description: 'Adds position information using sinusoidal functions',
            color: 'bg-cyan-500',
            details: 'PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(pos/10000^(2i/d))'
        },
        encoder_stack: {
            name: 'Encoder Stack (N×)',
            description: 'N identical layers processing input in parallel',
            color: 'bg-green-500',
            details: 'Each encoder has: Multi-Head Self-Attention → Add & Norm → Feed Forward → Add & Norm'
        },
        decoder_stack: {
            name: 'Decoder Stack (N×)',
            description: 'N identical layers generating output autoregressively',
            color: 'bg-purple-500',
            details: 'Each decoder has: Masked Self-Attention → Cross-Attention (to encoder) → Feed Forward'
        },
        output_embedding: {
            name: 'Output Embedding',
            description: 'Same as input embedding (often shared weights)',
            color: 'bg-pink-500',
            details: 'Output tokens are embedded, then processed through the decoder stack.'
        },
        linear_softmax: {
            name: 'Linear + Softmax',
            description: 'Projects to vocabulary size, converts to probabilities',
            color: 'bg-red-500',
            details: 'Linear layer: d_model → vocab_size, then softmax for probability distribution.'
        }
    };

    return (
        <div className="p-6 min-h-screen">
            <div className="max-w-6xl mx-auto">
                {/* Header */}
                <div className="text-center mb-8">
                    <h2 className="text-3xl font-bold text-white mb-2">
                        The Transformer: <span className="gradient-text">Complete Architecture</span>
                    </h2>
                    <p className="text-slate-400">
                        A sequence-to-sequence model built entirely on attention mechanisms
                    </p>
                </div>

                {/* Main Architecture Diagram */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                    {/* Interactive Diagram */}
                    <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                        <h3 className="text-white font-bold mb-4 text-center">Interactive Architecture</h3>
                        <p className="text-slate-400 text-sm text-center mb-4">Hover over each component to learn more</p>
                        
                        <div className="relative flex justify-center gap-8">
                            {/* Encoder Side */}
                            <div className="flex flex-col items-center gap-3">
                                <div className="text-slate-400 text-sm font-medium mb-2">ENCODER</div>
                                
                                {/* Encoder Stack */}
                                <div 
                                    className={`relative w-32 h-40 rounded-lg border-2 border-dashed border-green-500/50 p-2 cursor-pointer transition-all ${
                                        hoveredComponent === 'encoder_stack' ? 'bg-green-500/20 scale-105' : 'hover:bg-green-500/10'
                                    }`}
                                    onMouseEnter={() => setHoveredComponent('encoder_stack')}
                                    onMouseLeave={() => setHoveredComponent(null)}
                                >
                                    <div className="absolute -top-3 -right-3 bg-green-500 text-white text-xs px-2 py-0.5 rounded-full">N×</div>
                                    <div className="h-full flex flex-col justify-around">
                                        <div className="bg-green-600/50 rounded p-1 text-xs text-center text-white">Multi-Head Attention</div>
                                        <div className="bg-green-500/50 rounded p-1 text-xs text-center text-white">Add & Norm</div>
                                        <div className="bg-green-600/50 rounded p-1 text-xs text-center text-white">Feed Forward</div>
                                        <div className="bg-green-500/50 rounded p-1 text-xs text-center text-white">Add & Norm</div>
                                    </div>
                                </div>

                                <ArrowDown className="text-slate-500" size={20} />

                                {/* Positional Encoding */}
                                <div 
                                    className={`w-32 p-2 rounded-lg cursor-pointer transition-all ${
                                        hoveredComponent === 'positional_encoding' 
                                            ? 'bg-cyan-500 scale-105' 
                                            : 'bg-cyan-500/70 hover:bg-cyan-500/90'
                                    }`}
                                    onMouseEnter={() => setHoveredComponent('positional_encoding')}
                                    onMouseLeave={() => setHoveredComponent(null)}
                                >
                                    <div className="flex items-center justify-center gap-1">
                                        <Plus size={12} className="text-white" />
                                        <span className="text-xs text-white font-medium">Positional</span>
                                    </div>
                                </div>

                                <ArrowDown className="text-slate-500" size={20} />

                                {/* Input Embedding */}
                                <div 
                                    className={`w-32 p-3 rounded-lg cursor-pointer transition-all ${
                                        hoveredComponent === 'input_embedding' 
                                            ? 'bg-blue-500 scale-105' 
                                            : 'bg-blue-500/70 hover:bg-blue-500/90'
                                    }`}
                                    onMouseEnter={() => setHoveredComponent('input_embedding')}
                                    onMouseLeave={() => setHoveredComponent(null)}
                                >
                                    <div className="text-xs text-white text-center font-medium">Input Embedding</div>
                                </div>

                                <ArrowDown className="text-slate-500" size={20} />
                                <div className="text-slate-400 text-xs">Inputs</div>
                            </div>

                            {/* Cross Attention Arrow */}
                            <div className="flex items-center">
                                <div className="flex flex-col items-center">
                                    <ArrowRight className="text-yellow-400" size={32} />
                                    <span className="text-yellow-400 text-xs">K, V</span>
                                </div>
                            </div>

                            {/* Decoder Side */}
                            <div className="flex flex-col items-center gap-3">
                                <div className="text-slate-400 text-sm font-medium mb-2">DECODER</div>
                                
                                {/* Output */}
                                <div 
                                    className={`w-32 p-2 rounded-lg cursor-pointer transition-all ${
                                        hoveredComponent === 'linear_softmax' 
                                            ? 'bg-red-500 scale-105' 
                                            : 'bg-red-500/70 hover:bg-red-500/90'
                                    }`}
                                    onMouseEnter={() => setHoveredComponent('linear_softmax')}
                                    onMouseLeave={() => setHoveredComponent(null)}
                                >
                                    <div className="text-xs text-white text-center font-medium">Linear + Softmax</div>
                                </div>

                                <ArrowDown className="text-slate-500 rotate-180" size={20} />

                                {/* Decoder Stack */}
                                <div 
                                    className={`relative w-32 h-48 rounded-lg border-2 border-dashed border-purple-500/50 p-2 cursor-pointer transition-all ${
                                        hoveredComponent === 'decoder_stack' ? 'bg-purple-500/20 scale-105' : 'hover:bg-purple-500/10'
                                    }`}
                                    onMouseEnter={() => setHoveredComponent('decoder_stack')}
                                    onMouseLeave={() => setHoveredComponent(null)}
                                >
                                    <div className="absolute -top-3 -right-3 bg-purple-500 text-white text-xs px-2 py-0.5 rounded-full">N×</div>
                                    <div className="h-full flex flex-col justify-around">
                                        <div className="bg-purple-600/50 rounded p-1 text-xs text-center text-white">Masked Self-Attn</div>
                                        <div className="bg-purple-500/50 rounded p-1 text-xs text-center text-white">Add & Norm</div>
                                        <div className="bg-yellow-500/50 rounded p-1 text-xs text-center text-white">Cross-Attention</div>
                                        <div className="bg-purple-500/50 rounded p-1 text-xs text-center text-white">Add & Norm</div>
                                        <div className="bg-purple-600/50 rounded p-1 text-xs text-center text-white">Feed Forward</div>
                                    </div>
                                </div>

                                <ArrowDown className="text-slate-500" size={20} />

                                {/* Output Positional */}
                                <div 
                                    className={`w-32 p-2 rounded-lg cursor-pointer transition-all ${
                                        hoveredComponent === 'positional_encoding' 
                                            ? 'bg-cyan-500 scale-105' 
                                            : 'bg-cyan-500/70 hover:bg-cyan-500/90'
                                    }`}
                                    onMouseEnter={() => setHoveredComponent('positional_encoding')}
                                    onMouseLeave={() => setHoveredComponent(null)}
                                >
                                    <div className="flex items-center justify-center gap-1">
                                        <Plus size={12} className="text-white" />
                                        <span className="text-xs text-white font-medium">Positional</span>
                                    </div>
                                </div>

                                <ArrowDown className="text-slate-500" size={20} />

                                {/* Output Embedding */}
                                <div 
                                    className={`w-32 p-3 rounded-lg cursor-pointer transition-all ${
                                        hoveredComponent === 'output_embedding' 
                                            ? 'bg-pink-500 scale-105' 
                                            : 'bg-pink-500/70 hover:bg-pink-500/90'
                                    }`}
                                    onMouseEnter={() => setHoveredComponent('output_embedding')}
                                    onMouseLeave={() => setHoveredComponent(null)}
                                >
                                    <div className="text-xs text-white text-center font-medium">Output Embedding</div>
                                </div>

                                <ArrowDown className="text-slate-500" size={20} />
                                <div className="text-slate-400 text-xs">Outputs (shifted)</div>
                            </div>
                        </div>
                    </div>

                    {/* Component Details */}
                    <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                        <h3 className="text-white font-bold mb-4">Component Details</h3>
                        
                        {hoveredComponent ? (
                            <div className="space-y-4">
                                <div className={`${components[hoveredComponent].color} text-white px-4 py-2 rounded-lg font-bold`}>
                                    {components[hoveredComponent].name}
                                </div>
                                <p className="text-slate-300">
                                    {components[hoveredComponent].description}
                                </p>
                                <div className="bg-slate-700/50 p-4 rounded-lg">
                                    <p className="text-slate-400 text-sm font-mono">
                                        {components[hoveredComponent].details}
                                    </p>
                                </div>
                            </div>
                        ) : (
                            <div className="text-slate-500 text-center py-8">
                                <Eye size={48} className="mx-auto mb-4 opacity-50" />
                                <p>Hover over a component to see details</p>
                            </div>
                        )}

                        {/* Key Numbers */}
                        <div className="mt-6 grid grid-cols-2 gap-3">
                            <div className="bg-slate-700/50 p-3 rounded-lg text-center">
                                <div className="text-2xl font-bold text-blue-400">512</div>
                                <div className="text-xs text-slate-400">d_model</div>
                            </div>
                            <div className="bg-slate-700/50 p-3 rounded-lg text-center">
                                <div className="text-2xl font-bold text-green-400">8</div>
                                <div className="text-xs text-slate-400">Attention Heads</div>
                            </div>
                            <div className="bg-slate-700/50 p-3 rounded-lg text-center">
                                <div className="text-2xl font-bold text-purple-400">6</div>
                                <div className="text-xs text-slate-400">Encoder/Decoder Layers</div>
                            </div>
                            <div className="bg-slate-700/50 p-3 rounded-lg text-center">
                                <div className="text-2xl font-bold text-pink-400">2048</div>
                                <div className="text-xs text-slate-400">FFN Hidden Dim</div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Key Innovations */}
                <div className="bg-gradient-to-r from-amber-500/10 to-orange-500/10 rounded-2xl p-6 border border-amber-500/30 mb-8">
                    <h3 className="text-amber-400 font-bold mb-4 flex items-center gap-2">
                        <Lightbulb size={20} />
                        Why Transformers Changed Everything
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="bg-slate-800/50 p-4 rounded-lg">
                            <h4 className="text-white font-medium mb-2">🚀 Parallelization</h4>
                            <p className="text-slate-400 text-sm">
                                Unlike RNNs, transformers process all positions simultaneously. Training on GPUs is massively faster.
                            </p>
                        </div>
                        <div className="bg-slate-800/50 p-4 rounded-lg">
                            <h4 className="text-white font-medium mb-2">🔗 Long-Range Dependencies</h4>
                            <p className="text-slate-400 text-sm">
                                Any position can attend to any other position directly. No information bottleneck!
                            </p>
                        </div>
                        <div className="bg-slate-800/50 p-4 rounded-lg">
                            <h4 className="text-white font-medium mb-2">📈 Scalability</h4>
                            <p className="text-slate-400 text-sm">
                                The architecture scales beautifully - from BERT (110M) to GPT-4 (1.7T+ estimated).
                            </p>
                        </div>
                    </div>
                </div>

                {/* The Original Paper */}
                <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                    <h3 className="text-white font-bold mb-4">📜 The Original Paper (2017)</h3>
                    <div className="flex flex-col md:flex-row gap-6">
                        <div className="flex-1">
                            <p className="text-slate-300 mb-4">
                                <strong className="text-blue-400">"Attention Is All You Need"</strong> by Vaswani et al. 
                                introduced the Transformer architecture, eliminating recurrence entirely.
                            </p>
                            <div className="space-y-2 text-sm">
                                <div className="flex items-center gap-2">
                                    <span className="text-green-400">✓</span>
                                    <span className="text-slate-400">New SOTA on WMT translation tasks</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="text-green-400">✓</span>
                                    <span className="text-slate-400">3.5 days training on 8 GPUs (vs. weeks for RNNs)</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="text-green-400">✓</span>
                                    <span className="text-slate-400">Foundation for BERT, GPT, T5, and all modern LLMs</span>
                                </div>
                            </div>
                        </div>
                        <div className="bg-slate-700/50 p-4 rounded-lg font-mono text-xs">
                            <div className="text-slate-500">// Original hyperparameters</div>
                            <div className="text-slate-300">d_model = <span className="text-blue-400">512</span></div>
                            <div className="text-slate-300">d_ff = <span className="text-green-400">2048</span></div>
                            <div className="text-slate-300">h = <span className="text-purple-400">8</span> <span className="text-slate-500">// heads</span></div>
                            <div className="text-slate-300">N = <span className="text-pink-400">6</span> <span className="text-slate-500">// layers</span></div>
                            <div className="text-slate-300">d_k = d_v = <span className="text-yellow-400">64</span></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
