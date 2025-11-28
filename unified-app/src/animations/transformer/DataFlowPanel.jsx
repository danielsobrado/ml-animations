import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, ArrowDown, ArrowRight, Zap, Layers } from 'lucide-react';

export default function DataFlowPanel() {
    const [currentStep, setCurrentStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [activeToken, setActiveToken] = useState(0);

    const inputTokens = ['The', 'cat', 'sat'];
    const outputTokens = ['Le', 'chat', 's\'est', 'assis'];

    const steps = [
        {
            title: 'Input Tokenization',
            description: 'The input sentence is split into tokens and converted to IDs.',
            phase: 'input'
        },
        {
            title: 'Input Embedding',
            description: 'Each token ID is mapped to a dense vector of size d_model (512).',
            phase: 'embed'
        },
        {
            title: 'Add Positional Encoding',
            description: 'Positional information is added to embeddings using sinusoidal functions.',
            phase: 'positional'
        },
        {
            title: 'Encoder Processing',
            description: 'Input passes through N encoder layers (self-attention + FFN). Each position attends to all positions.',
            phase: 'encoder'
        },
        {
            title: 'Encoder Output',
            description: 'Final encoder representations are passed to decoder via cross-attention.',
            phase: 'encoder_out'
        },
        {
            title: 'Decoder Input (Shifted)',
            description: 'Previous outputs are embedded and shifted right. <BOS> token starts generation.',
            phase: 'decoder_input'
        },
        {
            title: 'Masked Self-Attention',
            description: 'Decoder attends to previous positions only (causal mask prevents looking ahead).',
            phase: 'masked_attn'
        },
        {
            title: 'Cross-Attention',
            description: 'Decoder queries attend to encoder keys/values. This is where translation "happens"!',
            phase: 'cross_attn'
        },
        {
            title: 'Output Projection',
            description: 'Linear layer + softmax produces probability distribution over vocabulary.',
            phase: 'output'
        },
        {
            title: 'Next Token Selection',
            description: 'Select highest probability token (greedy) or sample from distribution. Feed back as input.',
            phase: 'select'
        }
    ];

    useEffect(() => {
        let interval;
        if (isPlaying && currentStep < steps.length - 1) {
            interval = setInterval(() => {
                setCurrentStep(prev => prev + 1);
            }, 2500);
        } else if (currentStep >= steps.length - 1) {
            setIsPlaying(false);
        }
        return () => clearInterval(interval);
    }, [isPlaying, currentStep]);

    const getPhaseClass = (phase) => {
        const currentPhase = steps[currentStep]?.phase;
        if (currentPhase === phase) {
            return 'ring-2 ring-yellow-400 shadow-lg shadow-yellow-400/20 scale-105';
        }
        return '';
    };

    return (
        <div className="p-6 min-h-screen">
            <div className="max-w-6xl mx-auto">
                {/* Header */}
                <div className="text-center mb-6">
                    <h2 className="text-3xl font-bold text-white mb-2">
                        Data Flow: <span className="gradient-text">End-to-End Journey</span>
                    </h2>
                    <p className="text-slate-400">
                        Watch how data transforms as it flows through the entire Transformer
                    </p>
                </div>

                {/* Controls */}
                <div className="flex justify-center gap-4 mb-6">
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className="flex items-center gap-2 px-6 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg transition-all"
                    >
                        {isPlaying ? <Pause size={20} /> : <Play size={20} />}
                        {isPlaying ? 'Pause' : 'Play Flow'}
                    </button>
                    <button
                        onClick={() => { setCurrentStep(0); setIsPlaying(false); }}
                        className="flex items-center gap-2 px-4 py-2 bg-slate-600 hover:bg-slate-500 text-white rounded-lg transition-all"
                    >
                        <RotateCcw size={20} />
                        Reset
                    </button>
                </div>

                {/* Progress */}
                <div className="flex justify-center gap-1 mb-8">
                    {steps.map((_, i) => (
                        <button
                            key={i}
                            onClick={() => setCurrentStep(i)}
                            className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium transition-all ${
                                currentStep === i
                                    ? 'bg-green-500 text-white scale-110'
                                    : currentStep > i
                                        ? 'bg-green-500/50 text-white'
                                        : 'bg-slate-700 text-slate-400'
                            }`}
                        >
                            {i + 1}
                        </button>
                    ))}
                </div>

                {/* Current Step Info */}
                <div className="bg-green-500/20 rounded-xl p-4 mb-6 border border-green-500/30">
                    <h3 className="text-green-400 font-bold text-lg mb-1">
                        Step {currentStep + 1}: {steps[currentStep].title}
                    </h3>
                    <p className="text-slate-300">{steps[currentStep].description}</p>
                </div>

                {/* Main Flow Diagram */}
                <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700 mb-8">
                    <div className="flex flex-col items-center gap-4">
                        {/* Input Section */}
                        <div className="w-full">
                            <div className="text-slate-400 text-sm text-center mb-2">INPUT (English)</div>
                            <div className={`flex justify-center gap-2 p-3 rounded-lg transition-all ${getPhaseClass('input')}`}>
                                {inputTokens.map((token, i) => (
                                    <div
                                        key={i}
                                        className="bg-blue-500/80 px-4 py-2 rounded-lg text-white font-medium"
                                    >
                                        {token}
                                    </div>
                                ))}
                            </div>
                        </div>

                        <ArrowDown className={`text-slate-500 transition-all ${steps[currentStep]?.phase === 'embed' ? 'text-yellow-400 scale-125' : ''}`} />

                        {/* Embedding Layer */}
                        <div className={`w-64 p-3 rounded-lg bg-indigo-500/70 text-white text-center transition-all ${getPhaseClass('embed')}`}>
                            <div className="font-medium">Embedding Layer</div>
                            <div className="text-xs opacity-80">token → d_model vector</div>
                        </div>

                        <ArrowDown className={`text-slate-500 transition-all ${steps[currentStep]?.phase === 'positional' ? 'text-yellow-400 scale-125' : ''}`} />

                        {/* Positional Encoding */}
                        <div className={`w-64 p-3 rounded-lg bg-cyan-500/70 text-white text-center transition-all ${getPhaseClass('positional')}`}>
                            <div className="font-medium">+ Positional Encoding</div>
                            <div className="text-xs opacity-80">sin/cos functions</div>
                        </div>

                        <ArrowDown className="text-slate-500" />

                        {/* Encoder Stack */}
                        <div className={`w-72 p-4 rounded-lg border-2 border-dashed border-green-500/50 transition-all ${getPhaseClass('encoder')}`}>
                            <div className="text-green-400 font-bold text-center mb-2">ENCODER ×6</div>
                            <div className="space-y-2">
                                <div className="bg-green-600/50 p-2 rounded text-white text-xs text-center">Self-Attention</div>
                                <div className="bg-green-500/50 p-1 rounded text-white text-xs text-center">Add & Norm</div>
                                <div className="bg-green-600/50 p-2 rounded text-white text-xs text-center">FFN</div>
                                <div className="bg-green-500/50 p-1 rounded text-white text-xs text-center">Add & Norm</div>
                            </div>
                        </div>

                        {/* Split to Encoder Output and Decoder */}
                        <div className="flex items-start gap-8 w-full justify-center">
                            {/* Encoder Output */}
                            <div className="flex flex-col items-center">
                                <ArrowDown className="text-slate-500 mb-2" />
                                <div className={`p-3 rounded-lg bg-green-500 text-white text-center transition-all ${getPhaseClass('encoder_out')}`}>
                                    <div className="text-sm font-medium">Encoder Output</div>
                                    <div className="text-xs opacity-80">[seq_len, d_model]</div>
                                </div>
                                <ArrowRight className="text-yellow-400 mt-2" size={32} />
                            </div>

                            {/* Decoder Section */}
                            <div className="flex flex-col items-center">
                                <div className="text-slate-400 text-xs mb-2">OUTPUT (French) - shifted</div>
                                <div className={`flex gap-1 p-2 rounded-lg transition-all ${getPhaseClass('decoder_input')}`}>
                                    <div className="bg-pink-500/80 px-2 py-1 rounded text-white text-xs">&lt;s&gt;</div>
                                    {outputTokens.slice(0, -1).map((token, i) => (
                                        <div key={i} className="bg-pink-500/80 px-2 py-1 rounded text-white text-xs">
                                            {token}
                                        </div>
                                    ))}
                                </div>
                                
                                <ArrowDown className="text-slate-500 my-2" />

                                {/* Decoder Stack */}
                                <div className={`w-64 p-4 rounded-lg border-2 border-dashed border-purple-500/50 transition-all ${
                                    getPhaseClass('masked_attn') || getPhaseClass('cross_attn')
                                }`}>
                                    <div className="text-purple-400 font-bold text-center mb-2">DECODER ×6</div>
                                    <div className="space-y-2">
                                        <div className={`bg-purple-600/50 p-2 rounded text-white text-xs text-center ${
                                            steps[currentStep]?.phase === 'masked_attn' ? 'ring-2 ring-yellow-400' : ''
                                        }`}>
                                            Masked Self-Attn
                                        </div>
                                        <div className="bg-purple-500/50 p-1 rounded text-white text-xs text-center">Add & Norm</div>
                                        <div className={`bg-yellow-500/50 p-2 rounded text-slate-900 text-xs text-center ${
                                            steps[currentStep]?.phase === 'cross_attn' ? 'ring-2 ring-yellow-400' : ''
                                        }`}>
                                            Cross-Attention ← K,V
                                        </div>
                                        <div className="bg-purple-500/50 p-1 rounded text-white text-xs text-center">Add & Norm</div>
                                        <div className="bg-purple-600/50 p-2 rounded text-white text-xs text-center">FFN</div>
                                    </div>
                                </div>

                                <ArrowDown className="text-slate-500 my-2" />

                                {/* Output */}
                                <div className={`w-48 p-3 rounded-lg bg-red-500 text-white text-center transition-all ${getPhaseClass('output')}`}>
                                    <div className="font-medium">Linear + Softmax</div>
                                    <div className="text-xs opacity-80">→ vocab_size probs</div>
                                </div>

                                <ArrowDown className="text-slate-500 my-2" />

                                {/* Selected Token */}
                                <div className={`p-3 rounded-lg bg-emerald-500 text-white text-center transition-all ${getPhaseClass('select')}`}>
                                    <div className="text-sm font-medium">Selected: "assis"</div>
                                    <div className="text-xs opacity-80">argmax or sample</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Tensor Shapes */}
                <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                    <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                        <Layers size={20} />
                        Tensor Shape Journey
                    </h3>
                    
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-slate-700">
                                    <th className="text-left text-slate-400 py-2 px-3">Stage</th>
                                    <th className="text-left text-slate-400 py-2 px-3">Shape</th>
                                    <th className="text-left text-slate-400 py-2 px-3">Example</th>
                                </tr>
                            </thead>
                            <tbody className="text-slate-300">
                                <tr className={`border-b border-slate-700/50 ${currentStep === 0 ? 'bg-green-500/10' : ''}`}>
                                    <td className="py-2 px-3">Input tokens</td>
                                    <td className="py-2 px-3 font-mono text-blue-400">[batch, seq_len]</td>
                                    <td className="py-2 px-3">[32, 128]</td>
                                </tr>
                                <tr className={`border-b border-slate-700/50 ${currentStep === 1 ? 'bg-green-500/10' : ''}`}>
                                    <td className="py-2 px-3">After embedding</td>
                                    <td className="py-2 px-3 font-mono text-indigo-400">[batch, seq_len, d_model]</td>
                                    <td className="py-2 px-3">[32, 128, 512]</td>
                                </tr>
                                <tr className={`border-b border-slate-700/50 ${currentStep === 3 ? 'bg-green-500/10' : ''}`}>
                                    <td className="py-2 px-3">Attention Q,K,V</td>
                                    <td className="py-2 px-3 font-mono text-purple-400">[batch, heads, seq, d_k]</td>
                                    <td className="py-2 px-3">[32, 8, 128, 64]</td>
                                </tr>
                                <tr className={`border-b border-slate-700/50 ${currentStep === 3 ? 'bg-green-500/10' : ''}`}>
                                    <td className="py-2 px-3">Attention scores</td>
                                    <td className="py-2 px-3 font-mono text-yellow-400">[batch, heads, seq, seq]</td>
                                    <td className="py-2 px-3">[32, 8, 128, 128]</td>
                                </tr>
                                <tr className={`border-b border-slate-700/50 ${currentStep === 3 ? 'bg-green-500/10' : ''}`}>
                                    <td className="py-2 px-3">FFN hidden</td>
                                    <td className="py-2 px-3 font-mono text-orange-400">[batch, seq, d_ff]</td>
                                    <td className="py-2 px-3">[32, 128, 2048]</td>
                                </tr>
                                <tr className={`border-b border-slate-700/50 ${currentStep === 4 ? 'bg-green-500/10' : ''}`}>
                                    <td className="py-2 px-3">Encoder output</td>
                                    <td className="py-2 px-3 font-mono text-green-400">[batch, seq_len, d_model]</td>
                                    <td className="py-2 px-3">[32, 128, 512]</td>
                                </tr>
                                <tr className={currentStep === 8 ? 'bg-green-500/10' : ''}>
                                    <td className="py-2 px-3">Final logits</td>
                                    <td className="py-2 px-3 font-mono text-red-400">[batch, seq_len, vocab_size]</td>
                                    <td className="py-2 px-3">[32, 128, 32000]</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
}
