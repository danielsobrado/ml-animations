import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, ChevronRight, ChevronLeft, Lightbulb } from 'lucide-react';
import gsap from 'gsap';

const FLOW_STEPS = [
    {
        id: 'input',
        title: '1. Input Reception',
        description: 'The model receives an image and a text prompt simultaneously',
        visual: 'input',
        details: 'Image: 224×224 pixels | Text: "What animal is in this image?"'
    },
    {
        id: 'vision-encode',
        title: '2. Vision Encoding',
        description: 'The Vision Transformer (ViT) splits the image into patches and encodes them',
        visual: 'vision',
        details: '16×16 patches → 196 patch embeddings of 768 dimensions each'
    },
    {
        id: 'text-encode',
        title: '3. Text Tokenization',
        description: 'Text is tokenized into subword tokens and embedded',
        visual: 'text',
        details: '"What animal is in this image?" → 8 tokens → 8 embeddings'
    },
    {
        id: 'projection',
        title: '4. Modality Projection',
        description: 'Visual tokens are projected into the same space as text tokens',
        visual: 'projection',
        details: 'Linear projection: 768-dim → LLM embedding dimension'
    },
    {
        id: 'combine',
        title: '5. Sequence Combination',
        description: 'Visual and text tokens are concatenated into a single sequence',
        visual: 'combine',
        details: '[IMG_1, IMG_2, ..., IMG_196, What, animal, is, in, this, image, ?]'
    },
    {
        id: 'attention',
        title: '6. Cross-Modal Attention',
        description: 'The LLM processes the sequence, allowing text to attend to visual features',
        visual: 'attention',
        details: 'Self-attention lets "animal" attend to visual patches containing the cat'
    },
    {
        id: 'generate',
        title: '7. Response Generation',
        description: 'The model generates tokens autoregressively based on the multimodal context',
        visual: 'generate',
        details: 'Output: "This image shows a cat sitting on a windowsill."'
    }
];

export default function FlowPanel() {
    const [currentStep, setCurrentStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const visualRef = useRef(null);

    useEffect(() => {
        if (!isPlaying) return;
        
        const timer = setInterval(() => {
            setCurrentStep(prev => {
                if (prev >= FLOW_STEPS.length - 1) {
                    setIsPlaying(false);
                    return prev;
                }
                return prev + 1;
            });
        }, 3000);
        
        return () => clearInterval(timer);
    }, [isPlaying]);

    useEffect(() => {
        if (visualRef.current) {
            gsap.fromTo(
                visualRef.current,
                { opacity: 0, y: 20 },
                { opacity: 1, y: 0, duration: 0.5 }
            );
        }
    }, [currentStep]);

    const step = FLOW_STEPS[currentStep];

    const renderVisual = () => {
        switch (step.visual) {
            case 'input':
                return (
                    <div className="flex items-center justify-center gap-8">
                        <div className="text-center">
                            <div className="w-32 h-32 bg-green-100 rounded-lg border-2 border-green-400 flex items-center justify-center text-5xl shadow-lg">
                                🐱
                            </div>
                            <div className="mt-2 text-sm text-green-700 font-medium">Image Input</div>
                        </div>
                        <div className="text-3xl text-slate-300">+</div>
                        <div className="text-center">
                            <div className="bg-blue-100 rounded-lg border-2 border-blue-400 p-4 shadow-lg">
                                <div className="text-blue-800 font-medium">"What animal is in this image?"</div>
                            </div>
                            <div className="mt-2 text-sm text-blue-700 font-medium">Text Input</div>
                        </div>
                    </div>
                );
            
            case 'vision':
                return (
                    <div className="text-center">
                        <div className="inline-grid grid-cols-4 gap-1 p-4 bg-green-50 rounded-lg border border-green-200">
                            {Array(16).fill(0).map((_, i) => (
                                <div 
                                    key={i} 
                                    className="w-12 h-12 bg-green-200 rounded flex items-center justify-center text-green-700 font-mono text-xs border border-green-300"
                                    style={{ animationDelay: `${i * 50}ms` }}
                                >
                                    P{i+1}
                                </div>
                            ))}
                        </div>
                        <div className="mt-3 text-sm text-green-700">Image split into 16 patches → encoded by ViT</div>
                    </div>
                );
            
            case 'text':
                return (
                    <div className="text-center">
                        <div className="flex justify-center gap-2 flex-wrap">
                            {['What', 'animal', 'is', 'in', 'this', 'image', '?'].map((token, i) => (
                                <div 
                                    key={i}
                                    className="px-3 py-2 bg-blue-200 rounded-lg border border-blue-400 font-mono text-blue-800"
                                >
                                    {token}
                                </div>
                            ))}
                        </div>
                        <div className="mt-3 text-sm text-blue-700">Text tokenized into subwords</div>
                    </div>
                );
            
            case 'projection':
                return (
                    <div className="flex items-center justify-center gap-4">
                        <div className="text-center">
                            <div className="bg-green-100 rounded-lg p-3 border border-green-300">
                                <div className="font-mono text-xs text-green-700">[0.23, -0.45, ...]</div>
                                <div className="text-sm text-green-600 mt-1">Visual Embedding</div>
                            </div>
                        </div>
                        <div className="flex flex-col items-center">
                            <div className="text-2xl">→</div>
                            <div className="bg-purple-100 rounded px-2 py-1 text-xs text-purple-700">Linear Proj</div>
                            <div className="text-2xl">→</div>
                        </div>
                        <div className="text-center">
                            <div className="bg-purple-100 rounded-lg p-3 border border-purple-300">
                                <div className="font-mono text-xs text-purple-700">[0.12, 0.89, ...]</div>
                                <div className="text-sm text-purple-600 mt-1">LLM Space</div>
                            </div>
                        </div>
                    </div>
                );
            
            case 'combine':
                return (
                    <div className="text-center">
                        <div className="flex justify-center gap-1 flex-wrap max-w-lg mx-auto">
                            {Array(8).fill(0).map((_, i) => (
                                <div key={`v${i}`} className="w-8 h-8 bg-green-400 rounded text-xs text-white flex items-center justify-center">
                                    V{i+1}
                                </div>
                            ))}
                            <div className="w-2"></div>
                            {['What', 'animal', 'is', 'in', 'this', 'image', '?'].map((t, i) => (
                                <div key={`t${i}`} className="h-8 px-2 bg-blue-400 rounded text-xs text-white flex items-center justify-center">
                                    {t}
                                </div>
                            ))}
                        </div>
                        <div className="mt-3 text-sm text-purple-700">Combined sequence of 15 tokens</div>
                    </div>
                );
            
            case 'attention':
                return (
                    <div className="text-center">
                        <div className="inline-block bg-slate-50 p-4 rounded-lg">
                            <div className="flex justify-center gap-2 mb-4">
                                {['V1', 'V2', 'V3', 'animal', '?'].map((t, i) => (
                                    <div 
                                        key={i}
                                        className={`w-12 h-12 rounded flex items-center justify-center text-xs ${
                                            t.startsWith('V') ? 'bg-green-200' : 'bg-blue-200'
                                        }`}
                                    >
                                        {t}
                                    </div>
                                ))}
                            </div>
                            {/* Attention lines */}
                            <svg className="w-full h-16" viewBox="0 0 260 60">
                                <defs>
                                    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
                                        <polygon points="0 0, 10 3.5, 0 7" fill="#8b5cf6" />
                                    </marker>
                                </defs>
                                <path d="M130,10 Q70,40 30,50" stroke="#8b5cf6" strokeWidth="2" fill="none" strokeDasharray="4" markerEnd="url(#arrowhead)" />
                                <path d="M130,10 Q100,35 70,50" stroke="#8b5cf6" strokeWidth="2" fill="none" strokeDasharray="4" markerEnd="url(#arrowhead)" />
                                <path d="M130,10 Q130,35 130,50" stroke="#8b5cf6" strokeWidth="2" fill="none" strokeDasharray="4" markerEnd="url(#arrowhead)" />
                            </svg>
                            <div className="text-sm text-purple-700">"animal" attends to visual patches containing 🐱</div>
                        </div>
                    </div>
                );
            
            case 'generate':
                return (
                    <div className="text-center">
                        <div className="bg-amber-50 rounded-lg p-6 border-2 border-amber-300 inline-block">
                            <div className="text-lg font-medium text-amber-800 mb-2">Generated Response:</div>
                            <div className="text-xl text-amber-900 font-serif">
                                "This image shows a <strong>cat</strong> sitting on a windowsill."
                            </div>
                        </div>
                        <div className="mt-3 flex justify-center gap-2">
                            {['This', 'image', 'shows', 'a', 'cat', '...'].map((t, i) => (
                                <div 
                                    key={i}
                                    className="px-2 py-1 bg-amber-200 rounded text-xs text-amber-800"
                                    style={{ opacity: 1 - i * 0.1 }}
                                >
                                    {t}
                                </div>
                            ))}
                        </div>
                    </div>
                );
            
            default:
                return null;
        }
    };

    return (
        <div className="p-8 h-full">
            <div className="max-w-4xl mx-auto">
                <div className="text-center mb-6">
                    <h2 className="text-3xl font-bold text-indigo-900 mb-2">Data Flow Animation</h2>
                    <p className="text-slate-600">
                        Watch how data flows through a multimodal LLM step by step
                    </p>
                </div>

                {/* Progress Bar */}
                <div className="mb-6">
                    <div className="flex justify-between mb-2">
                        {FLOW_STEPS.map((s, i) => (
                            <button
                                key={s.id}
                                onClick={() => setCurrentStep(i)}
                                className={`w-8 h-8 rounded-full font-bold text-sm transition-all ${
                                    i === currentStep 
                                        ? 'bg-indigo-600 text-white scale-125' 
                                        : i < currentStep
                                            ? 'bg-indigo-200 text-indigo-700'
                                            : 'bg-slate-200 text-slate-500'
                                }`}
                            >
                                {i + 1}
                            </button>
                        ))}
                    </div>
                    <div className="h-2 bg-slate-200 rounded-full">
                        <div 
                            className="h-full bg-indigo-600 rounded-full transition-all duration-300"
                            style={{ width: `${(currentStep / (FLOW_STEPS.length - 1)) * 100}%` }}
                        />
                    </div>
                </div>

                {/* Controls */}
                <div className="flex items-center justify-center gap-4 mb-6">
                    <button
                        onClick={() => setCurrentStep(0)}
                        className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200"
                    >
                        <RotateCcw size={20} />
                    </button>
                    <button
                        onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
                        disabled={currentStep === 0}
                        className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200 disabled:opacity-50"
                    >
                        <ChevronLeft size={20} />
                    </button>
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className={`px-6 py-2 rounded-lg font-bold ${
                            isPlaying ? 'bg-red-500 text-white' : 'bg-indigo-600 text-white'
                        }`}
                    >
                        {isPlaying ? <Pause size={20} /> : <Play size={20} />}
                    </button>
                    <button
                        onClick={() => setCurrentStep(Math.min(FLOW_STEPS.length - 1, currentStep + 1))}
                        disabled={currentStep === FLOW_STEPS.length - 1}
                        className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200 disabled:opacity-50"
                    >
                        <ChevronRight size={20} />
                    </button>
                </div>

                {/* Current Step Info */}
                <div className="bg-indigo-50 rounded-xl p-4 mb-6 border border-indigo-200">
                    <h3 className="font-bold text-indigo-900 text-xl">{step.title}</h3>
                    <p className="text-indigo-800 mt-1">{step.description}</p>
                    <p className="text-sm text-indigo-600 mt-2 font-mono">{step.details}</p>
                </div>

                {/* Visualization */}
                <div 
                    ref={visualRef}
                    className="bg-white rounded-xl p-8 border min-h-[200px] flex items-center justify-center"
                >
                    {renderVisual()}
                </div>

                {/* Key Insight */}
                <div className="mt-6 bg-amber-50 p-4 rounded-xl border border-amber-200">
                    <div className="flex items-start gap-3">
                        <Lightbulb className="text-amber-600 flex-shrink-0 mt-1" size={20} />
                        <div>
                            <h4 className="font-bold text-amber-900 mb-1">Key Concept</h4>
                            <p className="text-amber-800 text-sm">
                                The magic happens at the <strong>attention layer</strong> - text tokens can "look at" 
                                relevant visual patches. When processing "animal", the model attends strongly to the 
                                patches containing the cat, enabling accurate visual understanding.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
