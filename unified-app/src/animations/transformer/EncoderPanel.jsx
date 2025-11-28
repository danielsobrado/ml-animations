import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, ArrowDown, Layers, Eye, Plus, ChevronRight } from 'lucide-react';

export default function EncoderPanel() {
    const [currentStep, setCurrentStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [selectedLayer, setSelectedLayer] = useState(0);

    const steps = [
        {
            title: 'Input + Positional Encoding',
            description: 'Input embeddings are added to positional encodings to give the model position information.',
            highlight: 'input',
            formula: 'X = Embedding(tokens) + PE(positions)'
        },
        {
            title: 'Multi-Head Self-Attention',
            description: 'Each position attends to all positions. Multiple heads capture different relationship types.',
            highlight: 'attention',
            formula: 'Attention(Q,K,V) = softmax(QK^T/√d_k)V'
        },
        {
            title: 'Residual Connection + LayerNorm',
            description: 'Add the original input (residual) then normalize. This helps gradient flow and training stability.',
            highlight: 'residual1',
            formula: 'Output = LayerNorm(X + Attention(X))'
        },
        {
            title: 'Feed-Forward Network',
            description: 'Two linear transformations with ReLU activation. Applied identically to each position.',
            highlight: 'ffn',
            formula: 'FFN(x) = max(0, xW₁ + b₁)W₂ + b₂'
        },
        {
            title: 'Residual Connection + LayerNorm',
            description: 'Another residual connection and layer normalization after FFN.',
            highlight: 'residual2',
            formula: 'Output = LayerNorm(X + FFN(X))'
        },
        {
            title: 'Stack N Times',
            description: 'The entire encoder layer is repeated N times (typically 6 or 12). Output goes to decoder.',
            highlight: 'stack',
            formula: 'Encoder = EncoderLayer^N(Input)'
        }
    ];

    useEffect(() => {
        let interval;
        if (isPlaying && currentStep < steps.length - 1) {
            interval = setInterval(() => {
                setCurrentStep(prev => prev + 1);
            }, 3000);
        } else if (currentStep >= steps.length - 1) {
            setIsPlaying(false);
        }
        return () => clearInterval(interval);
    }, [isPlaying, currentStep]);

    const handleReset = () => {
        setCurrentStep(0);
        setIsPlaying(false);
    };

    const getHighlightClass = (component) => {
        const highlight = steps[currentStep]?.highlight;
        if (highlight === component) {
            return 'ring-2 ring-yellow-400 bg-opacity-100 scale-105';
        }
        return 'bg-opacity-70';
    };

    return (
        <div className="p-6 min-h-screen">
            <div className="max-w-6xl mx-auto">
                {/* Header */}
                <div className="text-center mb-6">
                    <h2 className="text-3xl font-bold text-white mb-2">
                        The Encoder: <span className="gradient-text">Processing Input</span>
                    </h2>
                    <p className="text-slate-400">
                        The encoder reads and understands the input sequence
                    </p>
                </div>

                {/* Controls */}
                <div className="flex justify-center gap-4 mb-6">
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className="flex items-center gap-2 px-6 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-all"
                    >
                        {isPlaying ? <Pause size={20} /> : <Play size={20} />}
                        {isPlaying ? 'Pause' : 'Play Animation'}
                    </button>
                    <button
                        onClick={handleReset}
                        className="flex items-center gap-2 px-4 py-2 bg-slate-600 hover:bg-slate-500 text-white rounded-lg transition-all"
                    >
                        <RotateCcw size={20} />
                        Reset
                    </button>
                </div>

                {/* Step indicators */}
                <div className="flex justify-center gap-2 mb-8">
                    {steps.map((_, i) => (
                        <button
                            key={i}
                            onClick={() => setCurrentStep(i)}
                            className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-all ${
                                currentStep === i
                                    ? 'bg-blue-500 text-white'
                                    : currentStep > i
                                        ? 'bg-green-500 text-white'
                                        : 'bg-slate-700 text-slate-400'
                            }`}
                        >
                            {i + 1}
                        </button>
                    ))}
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Encoder Visualization */}
                    <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                        <h3 className="text-white font-bold mb-4 text-center">Single Encoder Layer</h3>
                        
                        <div className="flex flex-col items-center gap-4">
                            {/* Output */}
                            <div className={`w-48 p-3 rounded-lg bg-green-500 text-white text-center text-sm font-medium transition-all ${getHighlightClass('stack')}`}>
                                To Next Layer / Decoder
                            </div>
                            
                            <ArrowDown className="text-slate-500" />

                            {/* Residual 2 */}
                            <div className={`w-48 p-2 rounded-lg bg-emerald-500 text-white text-center text-xs transition-all ${getHighlightClass('residual2')}`}>
                                <div className="flex items-center justify-center gap-2">
                                    <Plus size={14} /> Add & Norm
                                </div>
                            </div>

                            <ArrowDown className="text-slate-500" />

                            {/* FFN */}
                            <div className={`w-48 p-4 rounded-lg bg-orange-500 text-white text-center transition-all ${getHighlightClass('ffn')}`}>
                                <div className="font-medium">Feed Forward</div>
                                <div className="text-xs mt-1 opacity-80">
                                    Linear → ReLU → Linear
                                </div>
                            </div>

                            {/* Residual Connection */}
                            <div className="relative w-full flex justify-center">
                                <ArrowDown className="text-slate-500" />
                                <div className="absolute left-4 top-0 h-full w-px bg-yellow-400/50"></div>
                                <div className="absolute left-4 top-1/2 w-16 h-px bg-yellow-400/50"></div>
                            </div>

                            {/* Residual 1 */}
                            <div className={`w-48 p-2 rounded-lg bg-emerald-500 text-white text-center text-xs transition-all ${getHighlightClass('residual1')}`}>
                                <div className="flex items-center justify-center gap-2">
                                    <Plus size={14} /> Add & Norm
                                </div>
                            </div>

                            <ArrowDown className="text-slate-500" />

                            {/* Multi-Head Attention */}
                            <div className={`w-48 p-4 rounded-lg bg-purple-500 text-white text-center transition-all ${getHighlightClass('attention')}`}>
                                <div className="flex items-center justify-center gap-2">
                                    <Eye size={18} />
                                    <span className="font-medium">Multi-Head Attention</span>
                                </div>
                                <div className="text-xs mt-2 opacity-80">
                                    Q, K, V from same input
                                </div>
                            </div>

                            {/* Residual Connection */}
                            <div className="relative w-full flex justify-center">
                                <ArrowDown className="text-slate-500" />
                                <div className="absolute left-4 top-0 h-full w-px bg-yellow-400/50"></div>
                                <div className="absolute left-4 top-1/2 w-16 h-px bg-yellow-400/50"></div>
                            </div>

                            {/* Input */}
                            <div className={`w-48 p-3 rounded-lg bg-blue-500 text-white text-center transition-all ${getHighlightClass('input')}`}>
                                <div className="font-medium">Input + Positional</div>
                                <div className="text-xs mt-1 opacity-80">Embeddings</div>
                            </div>
                        </div>

                        {/* Layer selector */}
                        <div className="mt-6 text-center">
                            <div className="text-slate-400 text-sm mb-2">Encoder Layers (click to select)</div>
                            <div className="flex justify-center gap-2">
                                {[0, 1, 2, 3, 4, 5].map(i => (
                                    <button
                                        key={i}
                                        onClick={() => setSelectedLayer(i)}
                                        className={`w-8 h-8 rounded text-sm font-medium transition-all ${
                                            selectedLayer === i
                                                ? 'bg-green-500 text-white'
                                                : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                                        }`}
                                    >
                                        {i + 1}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Step Description */}
                    <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                        <div className="bg-blue-500/20 rounded-xl p-4 mb-6">
                            <h3 className="text-blue-400 font-bold text-lg mb-2">
                                Step {currentStep + 1}: {steps[currentStep].title}
                            </h3>
                            <p className="text-slate-300">
                                {steps[currentStep].description}
                            </p>
                        </div>

                        {/* Formula */}
                        <div className="bg-slate-700/50 p-4 rounded-lg mb-6">
                            <div className="text-slate-400 text-sm mb-2">Formula:</div>
                            <div className="text-white font-mono text-center text-lg">
                                {steps[currentStep].formula}
                            </div>
                        </div>

                        {/* Self-Attention Deep Dive */}
                        <div className="space-y-4">
                            <h4 className="text-white font-bold">🔍 Encoder Self-Attention</h4>
                            
                            <div className="grid grid-cols-3 gap-3">
                                <div className="bg-blue-500/20 p-3 rounded-lg text-center">
                                    <div className="text-blue-400 font-bold">Q</div>
                                    <div className="text-slate-400 text-xs">From input</div>
                                </div>
                                <div className="bg-green-500/20 p-3 rounded-lg text-center">
                                    <div className="text-green-400 font-bold">K</div>
                                    <div className="text-slate-400 text-xs">From input</div>
                                </div>
                                <div className="bg-purple-500/20 p-3 rounded-lg text-center">
                                    <div className="text-purple-400 font-bold">V</div>
                                    <div className="text-slate-400 text-xs">From input</div>
                                </div>
                            </div>

                            <div className="text-slate-400 text-sm">
                                In encoder self-attention, all of Q, K, V come from the same input sequence. 
                                This allows every position to attend to every other position in the input.
                            </div>

                            <div className="bg-amber-500/10 p-4 rounded-lg border border-amber-500/30">
                                <h5 className="text-amber-400 font-medium mb-2">💡 Why Residual Connections?</h5>
                                <ul className="text-slate-400 text-sm space-y-1">
                                    <li>• Enable training very deep networks</li>
                                    <li>• Gradients flow directly through skip connections</li>
                                    <li>• Model can learn identity mapping easily</li>
                                    <li>• Inspired by ResNet (He et al., 2015)</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Dimension Flow */}
                <div className="mt-8 bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                    <h3 className="text-white font-bold mb-4">📐 Dimension Flow Through Encoder</h3>
                    
                    <div className="flex items-center justify-around flex-wrap gap-4 text-center">
                        <div>
                            <div className="bg-blue-500/20 px-4 py-2 rounded-lg mb-2">
                                <span className="text-blue-400 font-mono">[seq_len, d_model]</span>
                            </div>
                            <div className="text-slate-400 text-xs">Input</div>
                        </div>
                        <ChevronRight className="text-slate-500" />
                        <div>
                            <div className="bg-purple-500/20 px-4 py-2 rounded-lg mb-2">
                                <span className="text-purple-400 font-mono">[seq_len, d_model]</span>
                            </div>
                            <div className="text-slate-400 text-xs">After Attention</div>
                        </div>
                        <ChevronRight className="text-slate-500" />
                        <div>
                            <div className="bg-orange-500/20 px-4 py-2 rounded-lg mb-2">
                                <span className="text-orange-400 font-mono">[seq_len, d_ff]</span>
                            </div>
                            <div className="text-slate-400 text-xs">FFN Hidden</div>
                        </div>
                        <ChevronRight className="text-slate-500" />
                        <div>
                            <div className="bg-green-500/20 px-4 py-2 rounded-lg mb-2">
                                <span className="text-green-400 font-mono">[seq_len, d_model]</span>
                            </div>
                            <div className="text-slate-400 text-xs">Output</div>
                        </div>
                    </div>

                    <div className="mt-4 text-center text-slate-400 text-sm">
                        d_model = <span className="text-blue-400">512</span>, d_ff = <span className="text-orange-400">2048</span> (4× d_model typically)
                    </div>
                </div>
            </div>
        </div>
    );
}
