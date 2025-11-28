import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, ArrowDown, ArrowRight, Eye, Plus, Lock, Unlock, ChevronRight, AlertTriangle } from 'lucide-react';

export default function DecoderPanel() {
    const [currentStep, setCurrentStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [showMasking, setShowMasking] = useState(true);

    const steps = [
        {
            title: 'Output + Positional Encoding',
            description: 'Previously generated tokens are embedded and get positional encoding (shifted right by 1).',
            highlight: 'output_input',
            formula: 'X = Embedding(outputs_shifted) + PE(positions)'
        },
        {
            title: 'Masked Self-Attention',
            description: 'Each position can only attend to earlier positions. Future tokens are masked to prevent cheating!',
            highlight: 'masked_attention',
            formula: 'MaskedAttn = softmax(QK^T/√d_k + Mask)V'
        },
        {
            title: 'Add & Norm (First)',
            description: 'Residual connection and layer normalization after masked attention.',
            highlight: 'residual1',
            formula: 'X = LayerNorm(X + MaskedAttn(X))'
        },
        {
            title: 'Cross-Attention (Encoder-Decoder)',
            description: 'Queries from decoder, Keys & Values from encoder. This is how decoder "reads" the input!',
            highlight: 'cross_attention',
            formula: 'CrossAttn(Q_dec, K_enc, V_enc)'
        },
        {
            title: 'Add & Norm (Second)',
            description: 'Residual connection and layer normalization after cross-attention.',
            highlight: 'residual2',
            formula: 'X = LayerNorm(X + CrossAttn(X))'
        },
        {
            title: 'Feed-Forward Network',
            description: 'Same FFN structure as encoder - position-wise, two linear layers with ReLU.',
            highlight: 'ffn',
            formula: 'FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂'
        },
        {
            title: 'Linear + Softmax Output',
            description: 'Final projection to vocabulary size, softmax gives probability of next token.',
            highlight: 'output',
            formula: 'P(next_token) = softmax(Linear(output))'
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
            return 'ring-2 ring-yellow-400 scale-105 shadow-lg shadow-yellow-400/20';
        }
        return '';
    };

    // Attention mask visualization
    const maskMatrix = [
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
    ];

    return (
        <div className="p-6 min-h-screen">
            <div className="max-w-6xl mx-auto">
                {/* Header */}
                <div className="text-center mb-6">
                    <h2 className="text-3xl font-bold text-white mb-2">
                        The Decoder: <span className="gradient-text">Generating Output</span>
                    </h2>
                    <p className="text-slate-400">
                        The decoder generates output one token at a time, autoregressively
                    </p>
                </div>

                {/* Controls */}
                <div className="flex justify-center gap-4 mb-6">
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className="flex items-center gap-2 px-6 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-all"
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
                                    ? 'bg-purple-500 text-white'
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
                    {/* Decoder Visualization */}
                    <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                        <h3 className="text-white font-bold mb-4 text-center">Single Decoder Layer</h3>
                        
                        <div className="flex gap-8">
                            {/* Encoder Output (for reference) */}
                            <div className="flex flex-col items-center">
                                <div className="text-slate-400 text-xs mb-2">From Encoder</div>
                                <div className="w-16 h-32 bg-green-500/30 rounded-lg border border-green-500/50 flex items-center justify-center">
                                    <span className="text-green-400 text-xs writing-mode-vertical transform -rotate-90 whitespace-nowrap">K, V</span>
                                </div>
                            </div>

                            {/* Main Decoder */}
                            <div className="flex flex-col items-center gap-3 flex-1">
                                {/* Output */}
                                <div className={`w-40 p-3 rounded-lg bg-red-500 text-white text-center text-sm font-medium transition-all ${getHighlightClass('output')}`}>
                                    Linear + Softmax
                                </div>
                                
                                <ArrowDown className="text-slate-500" />

                                {/* Add & Norm 3 */}
                                <div className="w-40 p-2 rounded-lg bg-emerald-500 text-white text-center text-xs">
                                    <div className="flex items-center justify-center gap-2">
                                        <Plus size={12} /> Add & Norm
                                    </div>
                                </div>

                                <ArrowDown className="text-slate-500" />

                                {/* FFN */}
                                <div className={`w-40 p-3 rounded-lg bg-orange-500 text-white text-center transition-all ${getHighlightClass('ffn')}`}>
                                    <div className="text-sm font-medium">Feed Forward</div>
                                </div>

                                <ArrowDown className="text-slate-500" />

                                {/* Add & Norm 2 */}
                                <div className={`w-40 p-2 rounded-lg bg-emerald-500 text-white text-center text-xs transition-all ${getHighlightClass('residual2')}`}>
                                    <div className="flex items-center justify-center gap-2">
                                        <Plus size={12} /> Add & Norm
                                    </div>
                                </div>

                                <ArrowDown className="text-slate-500" />

                                {/* Cross-Attention */}
                                <div className={`w-40 p-3 rounded-lg bg-yellow-500 text-slate-900 text-center transition-all ${getHighlightClass('cross_attention')}`}>
                                    <div className="flex items-center justify-center gap-2">
                                        <Eye size={16} />
                                        <span className="text-sm font-medium">Cross-Attention</span>
                                    </div>
                                    <div className="text-xs mt-1">Q from dec, K,V from enc</div>
                                </div>

                                <ArrowDown className="text-slate-500" />

                                {/* Add & Norm 1 */}
                                <div className={`w-40 p-2 rounded-lg bg-emerald-500 text-white text-center text-xs transition-all ${getHighlightClass('residual1')}`}>
                                    <div className="flex items-center justify-center gap-2">
                                        <Plus size={12} /> Add & Norm
                                    </div>
                                </div>

                                <ArrowDown className="text-slate-500" />

                                {/* Masked Self-Attention */}
                                <div className={`w-40 p-3 rounded-lg bg-purple-500 text-white text-center transition-all ${getHighlightClass('masked_attention')}`}>
                                    <div className="flex items-center justify-center gap-2">
                                        <Lock size={16} />
                                        <span className="text-sm font-medium">Masked Self-Attn</span>
                                    </div>
                                    <div className="text-xs mt-1 opacity-80">Can't see future!</div>
                                </div>

                                <ArrowDown className="text-slate-500" />

                                {/* Input */}
                                <div className={`w-40 p-3 rounded-lg bg-pink-500 text-white text-center transition-all ${getHighlightClass('output_input')}`}>
                                    <div className="text-sm font-medium">Output Embedding</div>
                                    <div className="text-xs opacity-80">+ Positional</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Step Description + Masking */}
                    <div className="space-y-6">
                        {/* Current Step */}
                        <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                            <div className="bg-purple-500/20 rounded-xl p-4 mb-4">
                                <h3 className="text-purple-400 font-bold text-lg mb-2">
                                    Step {currentStep + 1}: {steps[currentStep].title}
                                </h3>
                                <p className="text-slate-300">
                                    {steps[currentStep].description}
                                </p>
                            </div>

                            <div className="bg-slate-700/50 p-3 rounded-lg">
                                <div className="text-slate-400 text-sm mb-1">Formula:</div>
                                <div className="text-white font-mono text-center">
                                    {steps[currentStep].formula}
                                </div>
                            </div>
                        </div>

                        {/* Causal Masking Visualization */}
                        <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-white font-bold flex items-center gap-2">
                                    <Lock size={18} className="text-purple-400" />
                                    Causal (Look-Ahead) Mask
                                </h3>
                                <button
                                    onClick={() => setShowMasking(!showMasking)}
                                    className="text-sm text-slate-400 hover:text-white"
                                >
                                    {showMasking ? 'Hide' : 'Show'} Details
                                </button>
                            </div>

                            {showMasking && (
                                <>
                                    <p className="text-slate-400 text-sm mb-4">
                                        Prevents positions from attending to subsequent positions. 
                                        Position i can only attend to positions 0...i.
                                    </p>

                                    <div className="flex justify-center mb-4">
                                        <div className="bg-slate-700/50 p-4 rounded-lg">
                                            <div className="flex mb-2">
                                                <div className="w-8"></div>
                                                {['I', 'am', 'a', 'cat', '.']}
                                            </div>
                                            <div className="grid gap-1">
                                                {maskMatrix.map((row, i) => (
                                                    <div key={i} className="flex items-center gap-1">
                                                        <div className="w-8 text-xs text-slate-400 text-right pr-2">
                                                            {['I', 'am', 'a', 'cat', '.'][i]}
                                                        </div>
                                                        {row.map((val, j) => (
                                                            <div
                                                                key={j}
                                                                className={`w-8 h-8 rounded flex items-center justify-center text-xs font-medium ${
                                                                    val === 1
                                                                        ? 'bg-green-500/50 text-green-300'
                                                                        : 'bg-red-500/30 text-red-400'
                                                                }`}
                                                            >
                                                                {val === 1 ? '✓' : '✗'}
                                                            </div>
                                                        ))}
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="flex items-center justify-center gap-4 text-sm">
                                        <div className="flex items-center gap-2">
                                            <div className="w-4 h-4 rounded bg-green-500/50"></div>
                                            <span className="text-slate-400">Can attend</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <div className="w-4 h-4 rounded bg-red-500/30"></div>
                                            <span className="text-slate-400">Masked (-∞)</span>
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>

                        {/* Cross-Attention Info */}
                        <div className="bg-amber-500/10 rounded-2xl p-4 border border-amber-500/30">
                            <h4 className="text-amber-400 font-bold mb-2 flex items-center gap-2">
                                <AlertTriangle size={16} />
                                Key Difference: Cross-Attention
                            </h4>
                            <div className="grid grid-cols-3 gap-2 mb-2">
                                <div className="bg-purple-500/20 p-2 rounded text-center">
                                    <div className="text-purple-400 font-bold text-sm">Q</div>
                                    <div className="text-slate-400 text-xs">From Decoder</div>
                                </div>
                                <div className="bg-green-500/20 p-2 rounded text-center">
                                    <div className="text-green-400 font-bold text-sm">K</div>
                                    <div className="text-slate-400 text-xs">From Encoder</div>
                                </div>
                                <div className="bg-green-500/20 p-2 rounded text-center">
                                    <div className="text-green-400 font-bold text-sm">V</div>
                                    <div className="text-slate-400 text-xs">From Encoder</div>
                                </div>
                            </div>
                            <p className="text-slate-400 text-xs">
                                This is how the decoder "reads" the encoder's understanding of the input!
                            </p>
                        </div>
                    </div>
                </div>

                {/* Autoregressive Generation */}
                <div className="mt-8 bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                    <h3 className="text-white font-bold mb-4">🔄 Autoregressive Generation</h3>
                    
                    <div className="space-y-4">
                        <p className="text-slate-400">
                            During inference, the decoder generates one token at a time, feeding each output back as input:
                        </p>

                        <div className="flex flex-wrap items-center justify-center gap-2">
                            <div className="bg-slate-700 px-3 py-2 rounded text-slate-300 text-sm">&lt;BOS&gt;</div>
                            <ChevronRight className="text-slate-500" size={16} />
                            <div className="bg-blue-500/30 px-3 py-2 rounded text-blue-300 text-sm">The</div>
                            <ChevronRight className="text-slate-500" size={16} />
                            <div className="bg-blue-500/30 px-3 py-2 rounded text-blue-300 text-sm">cat</div>
                            <ChevronRight className="text-slate-500" size={16} />
                            <div className="bg-blue-500/30 px-3 py-2 rounded text-blue-300 text-sm">sat</div>
                            <ChevronRight className="text-slate-500" size={16} />
                            <div className="bg-green-500/30 px-3 py-2 rounded text-green-300 text-sm border border-green-500">on</div>
                            <ChevronRight className="text-slate-500" size={16} />
                            <div className="bg-slate-700/50 px-3 py-2 rounded text-slate-500 text-sm">?</div>
                        </div>

                        <p className="text-slate-500 text-sm text-center">
                            At each step, the model predicts the probability distribution for the next token
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
