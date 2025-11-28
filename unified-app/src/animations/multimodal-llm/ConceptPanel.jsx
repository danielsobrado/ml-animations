import React, { useState, useEffect } from 'react';
import { Image, Type, Music, Video, MessageSquare, ArrowRight, Lightbulb, Sparkles } from 'lucide-react';

const MODALITIES = [
    { id: 'text', icon: Type, label: 'Text', color: 'blue', examples: ['Questions', 'Descriptions', 'Code'] },
    { id: 'image', icon: Image, label: 'Vision', color: 'green', examples: ['Photos', 'Diagrams', 'Screenshots'] },
    { id: 'audio', icon: Music, label: 'Audio', color: 'purple', examples: ['Speech', 'Music', 'Sounds'] },
    { id: 'video', icon: Video, label: 'Video', color: 'orange', examples: ['Clips', 'Animations', 'Streams'] },
];

const EXAMPLE_TASKS = [
    {
        inputs: ['image', 'text'],
        question: "What's in this image?",
        description: "Image captioning: Model sees an image and generates a text description",
        output: "A cat sitting on a windowsill looking outside"
    },
    {
        inputs: ['image', 'text'],
        question: "How many apples are in the basket?",
        description: "Visual Question Answering (VQA): Combine vision + reasoning",
        output: "There are 7 red apples in the basket"
    },
    {
        inputs: ['text'],
        question: "Generate an image of a sunset over mountains",
        description: "Text-to-Image: Understanding text to create visuals",
        output: "🖼️ [Generated Image]"
    },
    {
        inputs: ['audio', 'text'],
        question: "Transcribe and summarize this meeting",
        description: "Audio understanding: Speech recognition + summarization",
        output: "Summary: Q3 results discussed, 15% growth..."
    },
];

export default function ConceptPanel() {
    const [selectedExample, setSelectedExample] = useState(0);
    const [animStep, setAnimStep] = useState(0);

    useEffect(() => {
        const timer = setInterval(() => {
            setAnimStep(s => (s + 1) % 4);
        }, 1000);
        return () => clearInterval(timer);
    }, []);

    const example = EXAMPLE_TASKS[selectedExample];

    return (
        <div className="p-8 h-full">
            <div className="max-w-4xl mx-auto">
                {/* Introduction */}
                <div className="text-center mb-8">
                    <h2 className="text-3xl font-bold text-indigo-900 mb-4">What are Multimodal LLMs?</h2>
                    <p className="text-lg text-slate-700 leading-relaxed max-w-2xl mx-auto">
                        Multimodal LLMs can understand and generate content across <strong>multiple types of data</strong> - 
                        text, images, audio, and video - all in one unified model.
                    </p>
                </div>

                {/* Modalities Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                    {MODALITIES.map((mod, i) => {
                        const Icon = mod.icon;
                        const colorClasses = {
                            blue: 'bg-blue-50 border-blue-200 text-blue-600',
                            green: 'bg-green-50 border-green-200 text-green-600',
                            purple: 'bg-purple-50 border-purple-200 text-purple-600',
                            orange: 'bg-orange-50 border-orange-200 text-orange-600',
                        };
                        return (
                            <div 
                                key={mod.id}
                                className={`p-4 rounded-xl border-2 text-center transition-all duration-500 ${
                                    colorClasses[mod.color]
                                } ${animStep === i ? 'scale-110 shadow-lg' : ''}`}
                            >
                                <Icon className="mx-auto mb-2" size={32} />
                                <h3 className="font-bold">{mod.label}</h3>
                                <div className="text-xs mt-2 text-slate-600">
                                    {mod.examples.join(' • ')}
                                </div>
                            </div>
                        );
                    })}
                </div>

                {/* The Big Picture */}
                <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl p-6 mb-8 border border-indigo-100">
                    <h3 className="text-xl font-bold text-indigo-900 mb-4 text-center">
                        The Key Insight: Shared Representation Space
                    </h3>
                    <div className="flex items-center justify-center gap-4 flex-wrap">
                        <div className="flex flex-col items-center">
                            <div className="text-3xl mb-1">🖼️</div>
                            <span className="text-sm">Image</span>
                        </div>
                        <div className="flex flex-col items-center">
                            <div className="text-3xl mb-1">📝</div>
                            <span className="text-sm">Text</span>
                        </div>
                        <div className="flex flex-col items-center">
                            <div className="text-3xl mb-1">🎵</div>
                            <span className="text-sm">Audio</span>
                        </div>
                        <ArrowRight className="text-indigo-400 mx-4" size={32} />
                        <div className="bg-white rounded-xl p-4 shadow-md border">
                            <div className="text-center">
                                <Sparkles className="mx-auto text-indigo-600 mb-2" size={32} />
                                <div className="font-bold text-indigo-900">Unified</div>
                                <div className="font-bold text-indigo-900">Embedding Space</div>
                                <div className="font-mono text-xs text-slate-500 mt-1">[0.23, -0.45, ...]</div>
                            </div>
                        </div>
                        <ArrowRight className="text-indigo-400 mx-4" size={32} />
                        <div className="flex flex-col items-center">
                            <MessageSquare className="text-green-600 mb-1" size={32} />
                            <span className="text-sm font-bold">Understanding</span>
                        </div>
                    </div>
                    <p className="text-center text-slate-600 mt-4 text-sm">
                        Different modalities are encoded into the <strong>same vector space</strong>, 
                        allowing the model to reason across them.
                    </p>
                </div>

                {/* Example Tasks */}
                <div className="bg-slate-50 rounded-xl p-6 mb-8">
                    <h3 className="text-xl font-bold text-slate-800 mb-4">Example Tasks</h3>
                    
                    <div className="flex flex-wrap gap-2 mb-4">
                        {EXAMPLE_TASKS.map((ex, i) => (
                            <button
                                key={i}
                                onClick={() => setSelectedExample(i)}
                                className={`px-4 py-2 rounded-lg font-medium transition-all ${
                                    selectedExample === i
                                        ? 'bg-indigo-600 text-white'
                                        : 'bg-white border hover:bg-slate-100'
                                }`}
                            >
                                {ex.inputs.map(inp => 
                                    inp === 'image' ? '🖼️' : inp === 'audio' ? '🎵' : '📝'
                                ).join('+')} {ex.question.slice(0, 20)}...
                            </button>
                        ))}
                    </div>

                    <div className="bg-white rounded-lg p-4 border">
                        <div className="flex items-start gap-4 mb-4">
                            <div className="flex gap-2">
                                {example.inputs.map(inp => (
                                    <div 
                                        key={inp}
                                        className={`w-12 h-12 rounded-lg flex items-center justify-center text-2xl ${
                                            inp === 'image' ? 'bg-green-100' : 
                                            inp === 'audio' ? 'bg-purple-100' : 'bg-blue-100'
                                        }`}
                                    >
                                        {inp === 'image' ? '🖼️' : inp === 'audio' ? '🎵' : '📝'}
                                    </div>
                                ))}
                            </div>
                            <div className="flex-1">
                                <div className="text-sm text-slate-500 mb-1">Input:</div>
                                <div className="font-medium text-slate-800">"{example.question}"</div>
                            </div>
                        </div>
                        
                        <div className="text-sm text-slate-600 bg-slate-50 p-3 rounded mb-4">
                            {example.description}
                        </div>
                        
                        <div className="flex items-start gap-4">
                            <div className="w-12 h-12 rounded-lg bg-indigo-100 flex items-center justify-center">
                                <MessageSquare className="text-indigo-600" size={24} />
                            </div>
                            <div className="flex-1">
                                <div className="text-sm text-slate-500 mb-1">Output:</div>
                                <div className="font-medium text-green-700">{example.output}</div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Famous Models */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
                    {[
                        { name: 'GPT-4V', company: 'OpenAI', capabilities: 'Text + Vision' },
                        { name: 'Gemini', company: 'Google', capabilities: 'Text + Vision + Audio' },
                        { name: 'Claude 3', company: 'Anthropic', capabilities: 'Text + Vision' },
                        { name: 'LLaVA', company: 'Open Source', capabilities: 'Text + Vision' },
                    ].map(model => (
                        <div key={model.name} className="bg-white rounded-lg p-3 border text-center">
                            <div className="font-bold text-slate-800">{model.name}</div>
                            <div className="text-xs text-slate-500">{model.company}</div>
                            <div className="text-xs text-indigo-600 mt-1">{model.capabilities}</div>
                        </div>
                    ))}
                </div>

                {/* Key Insight */}
                <div className="bg-amber-50 p-4 rounded-xl border border-amber-200 flex items-start gap-3">
                    <Lightbulb className="text-amber-600 flex-shrink-0 mt-1" size={24} />
                    <div>
                        <h4 className="font-bold text-amber-900 mb-1">Key Insight</h4>
                        <p className="text-amber-800 text-sm">
                            The magic of multimodal LLMs is <strong>alignment</strong> - training different encoders 
                            (vision, audio) to map their inputs into the same space as text. This allows 
                            the language model to "understand" images and sounds as if they were words.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
