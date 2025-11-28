import React, { useState } from 'react';
import { Image, Type, ArrowRight, Layers, Box, Lightbulb, Eye, Brain } from 'lucide-react';

const ARCHITECTURES = [
    {
        id: 'early-fusion',
        name: 'Early Fusion',
        description: 'Combine modalities at the input level before processing',
        pros: ['Rich cross-modal interactions', 'End-to-end learning'],
        cons: ['Computationally expensive', 'Hard to scale'],
        models: ['Flamingo', 'BLIP-2']
    },
    {
        id: 'late-fusion',
        name: 'Late Fusion',
        description: 'Process each modality separately, combine at decision level',
        pros: ['Modular design', 'Easier to train'],
        cons: ['Limited cross-modal reasoning', 'Miss early interactions'],
        models: ['CLIP + GPT']
    },
    {
        id: 'cross-attention',
        name: 'Cross-Attention',
        description: 'Use attention mechanisms to fuse modalities at multiple layers',
        pros: ['Flexible fusion', 'Strong performance'],
        cons: ['Complex architecture', 'Requires careful design'],
        models: ['GPT-4V', 'Gemini', 'LLaVA']
    }
];

export default function ArchitecturePanel() {
    const [selectedArch, setSelectedArch] = useState('cross-attention');
    const [hoveredComponent, setHoveredComponent] = useState(null);

    const arch = ARCHITECTURES.find(a => a.id === selectedArch);

    return (
        <div className="p-8 h-full">
            <div className="max-w-5xl mx-auto">
                <div className="text-center mb-6">
                    <h2 className="text-3xl font-bold text-indigo-900 mb-2">Multimodal Architecture</h2>
                    <p className="text-slate-600">
                        How different modalities are combined in modern LLMs
                    </p>
                </div>

                {/* Architecture Selection */}
                <div className="flex justify-center gap-4 mb-8">
                    {ARCHITECTURES.map(a => (
                        <button
                            key={a.id}
                            onClick={() => setSelectedArch(a.id)}
                            className={`px-4 py-2 rounded-lg font-medium transition-all ${
                                selectedArch === a.id
                                    ? 'bg-indigo-600 text-white shadow-lg'
                                    : 'bg-white border hover:bg-slate-50'
                            }`}
                        >
                            {a.name}
                        </button>
                    ))}
                </div>

                {/* Main Diagram - Cross Attention (default) */}
                <div className="bg-slate-50 rounded-xl p-6 mb-6">
                    <h3 className="text-lg font-bold text-slate-800 mb-4 text-center">{arch.name} Architecture</h3>
                    
                    {selectedArch === 'cross-attention' && (
                        <div className="flex flex-col items-center gap-4">
                            {/* Input Layer */}
                            <div className="flex gap-8 items-end">
                                {/* Image Input */}
                                <div 
                                    className={`flex flex-col items-center transition-all ${
                                        hoveredComponent === 'vision' ? 'scale-110' : ''
                                    }`}
                                    onMouseEnter={() => setHoveredComponent('vision')}
                                    onMouseLeave={() => setHoveredComponent(null)}
                                >
                                    <div className="w-24 h-24 bg-green-100 rounded-lg border-2 border-green-300 flex items-center justify-center mb-2">
                                        <span className="text-4xl">🖼️</span>
                                    </div>
                                    <span className="text-sm font-medium text-green-700">Image</span>
                                </div>
                                
                                {/* Text Input */}
                                <div 
                                    className={`flex flex-col items-center transition-all ${
                                        hoveredComponent === 'text' ? 'scale-110' : ''
                                    }`}
                                    onMouseEnter={() => setHoveredComponent('text')}
                                    onMouseLeave={() => setHoveredComponent(null)}
                                >
                                    <div className="w-24 h-24 bg-blue-100 rounded-lg border-2 border-blue-300 flex items-center justify-center mb-2">
                                        <Type className="text-blue-600" size={40} />
                                    </div>
                                    <span className="text-sm font-medium text-blue-700">Text</span>
                                </div>
                            </div>

                            {/* Arrows down */}
                            <div className="flex gap-8">
                                <div className="text-slate-400">↓</div>
                                <div className="text-slate-400">↓</div>
                            </div>

                            {/* Encoders */}
                            <div className="flex gap-8">
                                <div 
                                    className={`bg-green-200 rounded-lg p-4 border-2 border-green-400 transition-all ${
                                        hoveredComponent === 'vision-encoder' ? 'scale-105 shadow-lg' : ''
                                    }`}
                                    onMouseEnter={() => setHoveredComponent('vision-encoder')}
                                    onMouseLeave={() => setHoveredComponent(null)}
                                >
                                    <Eye className="mx-auto text-green-700 mb-1" size={24} />
                                    <div className="text-sm font-bold text-green-800">Vision Encoder</div>
                                    <div className="text-xs text-green-600">ViT / CLIP</div>
                                </div>
                                
                                <div 
                                    className={`bg-blue-200 rounded-lg p-4 border-2 border-blue-400 transition-all ${
                                        hoveredComponent === 'text-encoder' ? 'scale-105 shadow-lg' : ''
                                    }`}
                                    onMouseEnter={() => setHoveredComponent('text-encoder')}
                                    onMouseLeave={() => setHoveredComponent(null)}
                                >
                                    <Type className="mx-auto text-blue-700 mb-1" size={24} />
                                    <div className="text-sm font-bold text-blue-800">Text Tokenizer</div>
                                    <div className="text-xs text-blue-600">BPE Tokens</div>
                                </div>
                            </div>

                            {/* Visual Tokens */}
                            <div className="flex gap-8 items-center">
                                <div className="bg-green-100 rounded-lg px-3 py-2 border border-green-300">
                                    <div className="text-xs text-green-600 text-center">Visual Tokens</div>
                                    <div className="flex gap-1 mt-1">
                                        {[1,2,3,4,5].map(i => (
                                            <div key={i} className="w-4 h-4 bg-green-400 rounded text-[8px] text-white flex items-center justify-center">{i}</div>
                                        ))}
                                    </div>
                                </div>
                                <div className="bg-blue-100 rounded-lg px-3 py-2 border border-blue-300">
                                    <div className="text-xs text-blue-600 text-center">Text Tokens</div>
                                    <div className="flex gap-1 mt-1">
                                        {['What', 'is', 'this', '?'].map((t, i) => (
                                            <div key={i} className="px-1 h-4 bg-blue-400 rounded text-[8px] text-white flex items-center justify-center">{t}</div>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            {/* Projection & Combination */}
                            <div className="text-slate-400">↓ Projection Layer ↓</div>

                            <div 
                                className={`bg-purple-100 rounded-xl p-4 border-2 border-purple-300 transition-all ${
                                    hoveredComponent === 'combined' ? 'scale-105 shadow-lg' : ''
                                }`}
                                onMouseEnter={() => setHoveredComponent('combined')}
                                onMouseLeave={() => setHoveredComponent(null)}
                            >
                                <div className="text-sm font-bold text-purple-800 text-center mb-2">Combined Token Sequence</div>
                                <div className="flex gap-1 justify-center">
                                    {[1,2,3,4,5].map(i => (
                                        <div key={`v${i}`} className="w-6 h-6 bg-green-400 rounded text-xs text-white flex items-center justify-center">V{i}</div>
                                    ))}
                                    {['What', 'is', 'this', '?'].map((t, i) => (
                                        <div key={`t${i}`} className="w-8 h-6 bg-blue-400 rounded text-xs text-white flex items-center justify-center">{t}</div>
                                    ))}
                                </div>
                            </div>

                            {/* LLM */}
                            <div className="text-slate-400">↓</div>

                            <div 
                                className={`bg-indigo-200 rounded-xl p-6 border-2 border-indigo-400 transition-all ${
                                    hoveredComponent === 'llm' ? 'scale-105 shadow-lg' : ''
                                }`}
                                onMouseEnter={() => setHoveredComponent('llm')}
                                onMouseLeave={() => setHoveredComponent(null)}
                            >
                                <Brain className="mx-auto text-indigo-700 mb-2" size={32} />
                                <div className="text-lg font-bold text-indigo-800 text-center">Language Model</div>
                                <div className="text-sm text-indigo-600 text-center">Self-Attention + Cross-Attention</div>
                            </div>

                            {/* Output */}
                            <div className="text-slate-400">↓</div>

                            <div className="bg-amber-100 rounded-lg p-3 border border-amber-300">
                                <div className="text-sm font-medium text-amber-800">"A cat sitting on a windowsill"</div>
                            </div>
                        </div>
                    )}

                    {selectedArch === 'early-fusion' && (
                        <div className="flex flex-col items-center gap-4">
                            <div className="flex gap-4">
                                <div className="w-20 h-20 bg-green-100 rounded-lg border-2 border-green-300 flex items-center justify-center">🖼️</div>
                                <div className="w-20 h-20 bg-blue-100 rounded-lg border-2 border-blue-300 flex items-center justify-center"><Type size={32} /></div>
                            </div>
                            <div className="text-slate-400">↓ Concatenate at Input ↓</div>
                            <div className="bg-purple-200 rounded-xl p-4 border-2 border-purple-400 w-64">
                                <div className="text-sm font-bold text-purple-800 text-center">Fused Input</div>
                                <div className="flex gap-1 justify-center mt-2">
                                    <div className="w-8 h-8 bg-green-400 rounded"></div>
                                    <div className="w-8 h-8 bg-blue-400 rounded"></div>
                                    <div className="w-8 h-8 bg-green-400 rounded"></div>
                                    <div className="w-8 h-8 bg-blue-400 rounded"></div>
                                </div>
                            </div>
                            <div className="text-slate-400">↓</div>
                            <div className="bg-indigo-200 rounded-xl p-4 border-2 border-indigo-400">
                                <Brain className="mx-auto text-indigo-700" size={32} />
                                <div className="font-bold text-indigo-800">Single Transformer</div>
                            </div>
                            <div className="text-slate-400">↓</div>
                            <div className="bg-amber-100 rounded-lg p-3 border border-amber-300">
                                <div className="text-sm font-medium text-amber-800">Output</div>
                            </div>
                        </div>
                    )}

                    {selectedArch === 'late-fusion' && (
                        <div className="flex flex-col items-center gap-4">
                            <div className="flex gap-16">
                                <div className="flex flex-col items-center">
                                    <div className="w-20 h-20 bg-green-100 rounded-lg border-2 border-green-300 flex items-center justify-center mb-2">🖼️</div>
                                    <div className="text-slate-400">↓</div>
                                    <div className="bg-green-200 rounded-lg p-3 border border-green-400">
                                        <Eye className="mx-auto text-green-700" size={24} />
                                        <div className="text-xs font-bold">Vision Model</div>
                                    </div>
                                    <div className="text-slate-400">↓</div>
                                    <div className="bg-green-100 rounded p-2 text-xs">Features</div>
                                </div>
                                <div className="flex flex-col items-center">
                                    <div className="w-20 h-20 bg-blue-100 rounded-lg border-2 border-blue-300 flex items-center justify-center mb-2"><Type size={32} /></div>
                                    <div className="text-slate-400">↓</div>
                                    <div className="bg-blue-200 rounded-lg p-3 border border-blue-400">
                                        <Brain className="mx-auto text-blue-700" size={24} />
                                        <div className="text-xs font-bold">Language Model</div>
                                    </div>
                                    <div className="text-slate-400">↓</div>
                                    <div className="bg-blue-100 rounded p-2 text-xs">Features</div>
                                </div>
                            </div>
                            <div className="text-slate-400">↘ ↙</div>
                            <div className="bg-purple-200 rounded-xl p-4 border-2 border-purple-400">
                                <Layers className="mx-auto text-purple-700 mb-1" size={24} />
                                <div className="font-bold text-purple-800">Fusion Layer</div>
                            </div>
                            <div className="text-slate-400">↓</div>
                            <div className="bg-amber-100 rounded-lg p-3 border border-amber-300">
                                <div className="text-sm font-medium text-amber-800">Final Output</div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Architecture Details */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-green-50 rounded-xl p-4 border border-green-200">
                        <h4 className="font-bold text-green-800 mb-2">✓ Advantages</h4>
                        <ul className="text-green-700 text-sm space-y-1">
                            {arch.pros.map((pro, i) => (
                                <li key={i}>• {pro}</li>
                            ))}
                        </ul>
                    </div>
                    <div className="bg-red-50 rounded-xl p-4 border border-red-200">
                        <h4 className="font-bold text-red-800 mb-2">✗ Challenges</h4>
                        <ul className="text-red-700 text-sm space-y-1">
                            {arch.cons.map((con, i) => (
                                <li key={i}>• {con}</li>
                            ))}
                        </ul>
                    </div>
                </div>

                {/* Models using this architecture */}
                <div className="mt-6 bg-slate-50 rounded-xl p-4">
                    <h4 className="font-bold text-slate-800 mb-2">Models using {arch.name}:</h4>
                    <div className="flex gap-2">
                        {arch.models.map(model => (
                            <span key={model} className="px-3 py-1 bg-indigo-100 text-indigo-800 rounded-full text-sm font-medium">
                                {model}
                            </span>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
