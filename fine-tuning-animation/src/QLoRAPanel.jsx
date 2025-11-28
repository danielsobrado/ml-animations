import React, { useState, useEffect } from 'react';
import { Zap, ArrowRight, Play, Pause, RotateCcw, Lightbulb, ArrowDown, CheckCircle2 } from 'lucide-react';

export default function QLoRAPanel() {
    const [step, setStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);

    const steps = [
        {
            title: 'Step 1: Load Model in 4-bit',
            desc: 'Quantize the pretrained weights from FP16/BF16 to 4-bit (NF4) format',
            detail: 'Reduces memory from ~14GB to ~3.5GB for a 7B model!',
        },
        {
            title: 'Step 2: Add LoRA Adapters',
            desc: 'Attach low-rank adapter matrices to the quantized model',
            detail: 'LoRA adapters remain in higher precision (FP16/BF16)',
        },
        {
            title: 'Step 3: Train Only LoRA',
            desc: 'Compute gradients only for the small LoRA matrices',
            detail: 'Gradients use double quantization for memory efficiency',
        },
        {
            title: 'Step 4: Inference',
            desc: 'Use quantized base + trained LoRA for predictions',
            detail: 'Can also merge LoRA into quantized weights',
        },
    ];

    useEffect(() => {
        if (isPlaying && step < steps.length - 1) {
            const timer = setTimeout(() => setStep(s => s + 1), 3000);
            return () => clearTimeout(timer);
        } else if (step >= steps.length - 1) {
            setIsPlaying(false);
        }
    }, [isPlaying, step]);

    const reset = () => {
        setStep(0);
        setIsPlaying(false);
    };

    // Memory comparison data
    const memoryData = [
        { method: 'Full FP32', memory: 28, color: 'bg-red-500' },
        { method: 'Full FP16', memory: 14, color: 'bg-orange-500' },
        { method: 'LoRA FP16', memory: 10, color: 'bg-yellow-500' },
        { method: 'QLoRA 8-bit', memory: 6, color: 'bg-blue-500' },
        { method: 'QLoRA 4-bit', memory: 3.5, color: 'bg-green-500' },
    ];

    return (
        <div className="p-6 h-full overflow-y-auto">
            <div className="max-w-5xl mx-auto">
                {/* Header */}
                <div className="text-center mb-6">
                    <h2 className="text-2xl font-bold text-purple-900 mb-2">QLoRA: Quantized LoRA</h2>
                    <p className="text-slate-600">Combine 4-bit quantization with LoRA for extreme memory efficiency</p>
                </div>

                {/* Controls */}
                <div className="flex justify-center gap-4 mb-6">
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className="flex items-center gap-2 px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600"
                    >
                        {isPlaying ? <Pause size={18} /> : <Play size={18} />}
                        {isPlaying ? 'Pause' : 'Play'}
                    </button>
                    <button
                        onClick={reset}
                        className="flex items-center gap-2 px-4 py-2 bg-slate-200 text-slate-700 rounded-lg hover:bg-slate-300"
                    >
                        <RotateCcw size={18} />
                        Reset
                    </button>
                </div>

                {/* Step Progress */}
                <div className="flex justify-between items-center mb-8 px-4">
                    {steps.map((s, i) => (
                        <React.Fragment key={i}>
                            <div 
                                className={`flex flex-col items-center cursor-pointer transition-all ${
                                    i <= step ? 'opacity-100' : 'opacity-40'
                                }`}
                                onClick={() => setStep(i)}
                            >
                                <div className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 transition-all ${
                                    i < step ? 'bg-green-500 text-white' :
                                    i === step ? 'bg-purple-500 text-white animate-pulse' :
                                    'bg-slate-200'
                                }`}>
                                    {i < step ? <CheckCircle2 size={20} /> : i + 1}
                                </div>
                                <span className="text-xs font-medium text-center max-w-24">{s.title.split(': ')[1]}</span>
                            </div>
                            {i < steps.length - 1 && (
                                <div className={`flex-1 h-1 mx-2 rounded ${i < step ? 'bg-green-500' : 'bg-slate-200'}`} />
                            )}
                        </React.Fragment>
                    ))}
                </div>

                <div className="grid grid-cols-2 gap-6 mb-6">
                    {/* Current Step Visualization */}
                    <div className="bg-white rounded-xl p-6 border shadow-sm">
                        <h3 className="font-bold text-purple-800 mb-4">{steps[step].title}</h3>
                        
                        <div className="min-h-48 flex items-center justify-center">
                            {step === 0 && (
                                <div className="text-center animate-fadeIn">
                                    <div className="flex items-center justify-center gap-4 mb-4">
                                        <div className="bg-blue-100 p-4 rounded-lg border-2 border-blue-300">
                                            <div className="text-sm font-medium text-blue-800">FP16 Weights</div>
                                            <div className="text-2xl font-bold text-blue-600">14 GB</div>
                                        </div>
                                        <ArrowRight className="text-slate-400" size={32} />
                                        <div className="bg-green-100 p-4 rounded-lg border-2 border-green-300">
                                            <div className="text-sm font-medium text-green-800">4-bit (NF4)</div>
                                            <div className="text-2xl font-bold text-green-600">3.5 GB</div>
                                        </div>
                                    </div>
                                    <div className="text-sm text-slate-600">
                                        NF4 = NormalFloat 4-bit (optimal for normally distributed weights)
                                    </div>
                                </div>
                            )}

                            {step === 1 && (
                                <div className="animate-fadeIn">
                                    <div className="relative">
                                        <div className="bg-green-100 rounded-lg p-6 border-2 border-green-300">
                                            <div className="text-center font-medium text-green-800 mb-2">4-bit Base Model</div>
                                            <div className="grid grid-cols-4 gap-1">
                                                {Array.from({ length: 16 }).map((_, i) => (
                                                    <div key={i} className="w-8 h-8 bg-green-300 rounded" />
                                                ))}
                                            </div>
                                        </div>
                                        {/* LoRA adapters */}
                                        <div className="absolute -right-4 top-1/2 -translate-y-1/2 flex flex-col gap-2">
                                            <div className="bg-purple-400 text-white px-2 py-1 rounded text-xs">LoRA A</div>
                                            <div className="bg-orange-400 text-white px-2 py-1 rounded text-xs">LoRA B</div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {step === 2 && (
                                <div className="animate-fadeIn space-y-4">
                                    <div className="flex items-center gap-4">
                                        <div className="bg-slate-100 p-3 rounded-lg flex-1">
                                            <div className="text-xs text-slate-500 mb-1">Forward Pass</div>
                                            <ArrowRight className="text-green-500 mx-auto" />
                                        </div>
                                        <div className="bg-slate-100 p-3 rounded-lg flex-1">
                                            <div className="text-xs text-slate-500 mb-1">Compute Loss</div>
                                            <div className="text-2xl text-center">📊</div>
                                        </div>
                                        <div className="bg-purple-100 p-3 rounded-lg flex-1 border-2 border-purple-300">
                                            <div className="text-xs text-purple-600 mb-1">Update LoRA</div>
                                            <ArrowDown className="text-purple-500 mx-auto" />
                                        </div>
                                    </div>
                                    <div className="text-center text-sm text-slate-600">
                                        Base model weights stay frozen & quantized
                                    </div>
                                </div>
                            )}

                            {step === 3 && (
                                <div className="animate-fadeIn text-center">
                                    <div className="text-4xl mb-4">🚀</div>
                                    <div className="text-lg font-medium text-green-700 mb-2">Ready for Inference!</div>
                                    <div className="text-sm text-slate-600">
                                        ~4GB total memory for a fine-tuned 7B model
                                    </div>
                                </div>
                            )}
                        </div>

                        <div className="mt-4 p-3 bg-slate-50 rounded-lg text-sm text-slate-700">
                            {steps[step].detail}
                        </div>
                    </div>

                    {/* Memory Comparison */}
                    <div className="bg-white rounded-xl p-6 border shadow-sm">
                        <h3 className="font-bold text-slate-800 mb-4">Memory Comparison (7B Model)</h3>
                        <div className="space-y-3">
                            {memoryData.map((item, i) => (
                                <div key={i} className="flex items-center gap-3">
                                    <div className="w-24 text-sm text-slate-600">{item.method}</div>
                                    <div className="flex-1 bg-slate-100 rounded-full h-6 overflow-hidden">
                                        <div 
                                            className={`${item.color} h-full rounded-full transition-all duration-500 flex items-center justify-end pr-2`}
                                            style={{ width: `${(item.memory / 28) * 100}%` }}
                                        >
                                            <span className="text-xs text-white font-medium">{item.memory} GB</span>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                        <div className="mt-4 p-3 bg-green-50 rounded-lg border border-green-200">
                            <div className="flex items-center gap-2 text-green-800">
                                <Zap size={18} />
                                <span className="font-medium">QLoRA enables fine-tuning on consumer GPUs!</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Key Innovations */}
                <div className="bg-slate-800 rounded-xl p-6 mb-6">
                    <h3 className="font-bold text-white mb-4 text-center">QLoRA Key Innovations</h3>
                    <div className="grid grid-cols-3 gap-4">
                        <div className="bg-slate-700 p-4 rounded-lg">
                            <div className="text-purple-400 font-bold mb-2">4-bit NormalFloat</div>
                            <p className="text-slate-300 text-sm">
                                Information-theoretically optimal data type for normally distributed weights
                            </p>
                        </div>
                        <div className="bg-slate-700 p-4 rounded-lg">
                            <div className="text-orange-400 font-bold mb-2">Double Quantization</div>
                            <p className="text-slate-300 text-sm">
                                Quantize the quantization constants to save additional memory
                            </p>
                        </div>
                        <div className="bg-slate-700 p-4 rounded-lg">
                            <div className="text-green-400 font-bold mb-2">Paged Optimizers</div>
                            <p className="text-slate-300 text-sm">
                                Use NVIDIA unified memory to avoid OOM during gradient checkpointing
                            </p>
                        </div>
                    </div>
                </div>

                {/* Practical Example */}
                <div className="bg-purple-50 rounded-xl p-6 border border-purple-200 mb-6">
                    <h4 className="font-bold text-purple-900 mb-3">🔧 Quick Start Example</h4>
                    <pre className="bg-slate-900 text-green-400 p-4 rounded-lg text-sm overflow-x-auto">
{`from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=bnb_config
)

# Add LoRA adapters
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)`}
                    </pre>
                </div>

                {/* Tips */}
                <div className="bg-amber-50 p-4 rounded-xl border border-amber-200">
                    <h4 className="font-bold text-amber-900 mb-2 flex items-center gap-2">
                        <Lightbulb size={18} />
                        When to Use QLoRA
                    </h4>
                    <div className="grid grid-cols-2 gap-4 text-sm text-amber-800">
                        <div>
                            <span className="font-medium text-green-700">✅ Good for:</span>
                            <ul className="list-disc ml-4 mt-1">
                                <li>Limited GPU memory (12-24 GB)</li>
                                <li>Fine-tuning large models (7B-70B)</li>
                                <li>Rapid experimentation</li>
                            </ul>
                        </div>
                        <div>
                            <span className="font-medium text-red-700">⚠️ Consider alternatives:</span>
                            <ul className="list-disc ml-4 mt-1">
                                <li>When you need maximum accuracy</li>
                                <li>Production inference (may want full precision)</li>
                                <li>When you have abundant compute</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            <style jsx>{`
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                .animate-fadeIn {
                    animation: fadeIn 0.4s ease-out forwards;
                }
            `}</style>
        </div>
    );
}
