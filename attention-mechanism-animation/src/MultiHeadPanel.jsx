import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Eye, Layers, Lightbulb, ArrowRight, ArrowDown, Merge } from 'lucide-react';

export default function MultiHeadPanel() {
    const [step, setStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [numHeads, setNumHeads] = useState(4);
    const [activeHead, setActiveHead] = useState(null);

    const headColors = [
        { bg: 'bg-red-500', border: 'border-red-500', text: 'text-red-400', light: 'bg-red-500/20' },
        { bg: 'bg-blue-500', border: 'border-blue-500', text: 'text-blue-400', light: 'bg-blue-500/20' },
        { bg: 'bg-green-500', border: 'border-green-500', text: 'text-green-400', light: 'bg-green-500/20' },
        { bg: 'bg-purple-500', border: 'border-purple-500', text: 'text-purple-400', light: 'bg-purple-500/20' },
        { bg: 'bg-orange-500', border: 'border-orange-500', text: 'text-orange-400', light: 'bg-orange-500/20' },
        { bg: 'bg-pink-500', border: 'border-pink-500', text: 'text-pink-400', light: 'bg-pink-500/20' },
        { bg: 'bg-cyan-500', border: 'border-cyan-500', text: 'text-cyan-400', light: 'bg-cyan-500/20' },
        { bg: 'bg-yellow-500', border: 'border-yellow-500', text: 'text-yellow-400', light: 'bg-yellow-500/20' },
    ];

    // Simulated head focuses
    const headFocuses = [
        { name: 'Syntactic', desc: 'Subject-verb relationships', example: '"The cat" → "sat"' },
        { name: 'Positional', desc: 'Nearby words', example: 'Local context' },
        { name: 'Semantic', desc: 'Related concepts', example: '"cat" → "animal"' },
        { name: 'Long-range', desc: 'Distant dependencies', example: 'Coreference' },
        { name: 'Lexical', desc: 'Word identity', example: 'Self-attention' },
        { name: 'Structural', desc: 'Sentence structure', example: 'Phrases' },
        { name: 'Topical', desc: 'Topic relevance', example: 'Theme words' },
        { name: 'Rare', desc: 'Unusual patterns', example: 'Edge cases' },
    ];

    const steps = [
        {
            title: 'Single Head Limitation',
            desc: 'One attention head can only focus on one type of relationship at a time.',
        },
        {
            title: 'Multiple Heads in Parallel',
            desc: 'Run multiple attention operations simultaneously with different learned projections.',
        },
        {
            title: 'Different Perspectives',
            desc: 'Each head learns to focus on different aspects: syntax, semantics, position, etc.',
        },
        {
            title: 'Concatenate Outputs',
            desc: 'Stack all head outputs together into one long vector.',
        },
        {
            title: 'Final Linear Projection',
            desc: 'Project concatenated output back to model dimension with W_O.',
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

    return (
        <div className="p-6 min-h-screen">
            <div className="max-w-6xl mx-auto">
                {/* Header */}
                <div className="text-center mb-6">
                    <h2 className="text-3xl font-bold text-white mb-2">
                        Multi-Head Attention: <span className="gradient-text">Multiple Perspectives</span>
                    </h2>
                    <p className="text-slate-400 max-w-2xl mx-auto">
                        Instead of one attention, run multiple in parallel - each learning to focus on different things.
                    </p>
                </div>

                {/* Controls */}
                <div className="flex justify-center items-center gap-4 mb-6">
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className="flex items-center gap-2 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600"
                    >
                        {isPlaying ? <Pause size={18} /> : <Play size={18} />}
                        {isPlaying ? 'Pause' : 'Play Animation'}
                    </button>
                    <button
                        onClick={reset}
                        className="flex items-center gap-2 px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-500"
                    >
                        <RotateCcw size={18} />
                        Reset
                    </button>
                    <div className="flex items-center gap-2 ml-4">
                        <span className="text-slate-400 text-sm">Heads:</span>
                        <select
                            value={numHeads}
                            onChange={(e) => setNumHeads(Number(e.target.value))}
                            className="bg-slate-700 text-white px-3 py-1 rounded-lg"
                        >
                            <option value={2}>2</option>
                            <option value={4}>4</option>
                            <option value={6}>6</option>
                            <option value={8}>8</option>
                        </select>
                    </div>
                </div>

                {/* Step Progress */}
                <div className="flex justify-center gap-2 mb-6">
                    {steps.map((s, i) => (
                        <button
                            key={i}
                            onClick={() => setStep(i)}
                            className={`px-3 py-1 rounded-full text-sm transition-all ${
                                i === step
                                    ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white'
                                    : i < step
                                        ? 'bg-green-500/30 text-green-400'
                                        : 'bg-slate-700 text-slate-400'
                            }`}
                        >
                            {i + 1}
                        </button>
                    ))}
                </div>

                {/* Current Step Info */}
                <div className="bg-green-500/20 rounded-xl p-4 mb-6 border border-green-500/30 text-center">
                    <h3 className="font-bold text-green-400 text-lg">{steps[step].title}</h3>
                    <p className="text-slate-300 text-sm">{steps[step].desc}</p>
                </div>

                {/* Main Visualization */}
                <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                    {/* Step 0: Single Head Limitation */}
                    {step === 0 && (
                        <div className="text-center animate-fadeIn">
                            <div className="inline-block bg-slate-700 rounded-xl p-6">
                                <Eye className="mx-auto text-slate-400 mb-2" size={48} />
                                <h4 className="text-white font-bold mb-2">Single Attention Head</h4>
                                <p className="text-slate-400 text-sm max-w-xs">
                                    Can only capture ONE type of relationship. 
                                    We need to look at input from MULTIPLE perspectives!
                                </p>
                            </div>
                        </div>
                    )}

                    {/* Step 1-2: Multiple Heads */}
                    {step >= 1 && step <= 2 && (
                        <div className="animate-fadeIn">
                            <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${Math.min(numHeads, 4)}, 1fr)` }}>
                                {Array.from({ length: numHeads }).map((_, i) => (
                                    <div
                                        key={i}
                                        className={`rounded-xl p-4 border-2 transition-all cursor-pointer ${
                                            headColors[i].light
                                        } ${headColors[i].border} ${
                                            activeHead === i ? 'scale-105 shadow-lg' : ''
                                        }`}
                                        onMouseEnter={() => setActiveHead(i)}
                                        onMouseLeave={() => setActiveHead(null)}
                                    >
                                        <div className={`${headColors[i].bg} w-10 h-10 rounded-full flex items-center justify-center mx-auto mb-2`}>
                                            <Eye className="text-white" size={20} />
                                        </div>
                                        <h5 className={`font-bold text-center ${headColors[i].text}`}>
                                            Head {i + 1}
                                        </h5>
                                        {step >= 2 && (
                                            <div className="mt-2 text-center">
                                                <div className="text-white text-xs font-medium">
                                                    {headFocuses[i].name}
                                                </div>
                                                <div className="text-slate-400 text-xs mt-1">
                                                    {headFocuses[i].desc}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>

                            {step >= 2 && activeHead !== null && (
                                <div className={`mt-4 p-3 rounded-lg ${headColors[activeHead].light} border ${headColors[activeHead].border}`}>
                                    <div className={`${headColors[activeHead].text} font-bold`}>
                                        Head {activeHead + 1}: {headFocuses[activeHead].name}
                                    </div>
                                    <div className="text-slate-300 text-sm">
                                        Example: {headFocuses[activeHead].example}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Step 3: Concatenation */}
                    {step === 3 && (
                        <div className="animate-fadeIn">
                            <div className="flex items-center justify-center gap-2 mb-6">
                                {Array.from({ length: numHeads }).map((_, i) => (
                                    <React.Fragment key={i}>
                                        <div className={`${headColors[i].bg} px-4 py-6 rounded-lg flex flex-col items-center`}>
                                            <Eye className="text-white mb-1" size={16} />
                                            <span className="text-white text-xs font-bold">H{i + 1}</span>
                                            <span className="text-white/70 text-xs mt-1">d/h</span>
                                        </div>
                                        {i < numHeads - 1 && (
                                            <span className="text-slate-400 text-lg">+</span>
                                        )}
                                    </React.Fragment>
                                ))}
                            </div>

                            <div className="flex justify-center">
                                <ArrowDown className="text-slate-400" size={32} />
                            </div>

                            <div className="flex justify-center mt-4">
                                <div className="bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 p-4 rounded-xl">
                                    <div className="text-white font-bold text-center mb-2">Concatenated</div>
                                    <div className="flex gap-1">
                                        {Array.from({ length: numHeads }).map((_, i) => (
                                            <div key={i} className={`${headColors[i].bg} w-8 h-16 rounded`} />
                                        ))}
                                    </div>
                                    <div className="text-white/70 text-xs text-center mt-2">
                                        {numHeads} × (d/h) = d
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Step 4: Final Projection */}
                    {step === 4 && (
                        <div className="animate-fadeIn">
                            <div className="flex items-center justify-center gap-6">
                                <div className="bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 p-4 rounded-xl">
                                    <div className="text-white text-xs text-center mb-2">Concat</div>
                                    <div className="flex gap-1">
                                        {Array.from({ length: numHeads }).map((_, i) => (
                                            <div key={i} className={`${headColors[i].bg} w-6 h-12 rounded`} />
                                        ))}
                                    </div>
                                </div>

                                <ArrowRight className="text-slate-400" size={32} />

                                <div className="bg-slate-600 p-4 rounded-xl">
                                    <div className="text-slate-300 text-xs text-center mb-2">W<sub>O</sub></div>
                                    <div className="grid grid-cols-4 gap-0.5">
                                        {Array.from({ length: 16 }).map((_, i) => (
                                            <div key={i} className="w-4 h-4 bg-slate-500 rounded-sm" />
                                        ))}
                                    </div>
                                </div>

                                <ArrowRight className="text-slate-400" size={32} />

                                <div className="bg-gradient-to-r from-green-500 to-emerald-500 p-4 rounded-xl">
                                    <div className="text-white text-xs text-center mb-2">Output</div>
                                    <div className="w-24 h-12 bg-white/20 rounded flex items-center justify-center">
                                        <span className="text-white font-mono text-sm">d_model</span>
                                    </div>
                                </div>
                            </div>

                            <div className="mt-6 text-center">
                                <div className="inline-block bg-slate-900 px-6 py-3 rounded-lg">
                                    <span className="text-slate-400">MultiHead(Q,K,V) = </span>
                                    <span className="text-white font-mono">Concat(head₁, ..., head_h)W<sub>O</sub></span>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Dimension Breakdown */}
                <div className="mt-8 bg-slate-800/50 rounded-xl p-6 border border-slate-700">
                    <h4 className="text-white font-bold mb-4 text-center">Dimension Breakdown</h4>
                    <div className="grid grid-cols-4 gap-4 text-center">
                        <div className="bg-slate-700 rounded-lg p-3">
                            <div className="text-2xl font-bold text-white">d<sub>model</sub></div>
                            <div className="text-slate-400 text-sm">512</div>
                            <div className="text-slate-500 text-xs">Model dimension</div>
                        </div>
                        <div className="bg-slate-700 rounded-lg p-3">
                            <div className="text-2xl font-bold text-white">h</div>
                            <div className="text-slate-400 text-sm">{numHeads}</div>
                            <div className="text-slate-500 text-xs">Number of heads</div>
                        </div>
                        <div className="bg-slate-700 rounded-lg p-3">
                            <div className="text-2xl font-bold text-white">d<sub>k</sub></div>
                            <div className="text-slate-400 text-sm">{Math.floor(512 / numHeads)}</div>
                            <div className="text-slate-500 text-xs">Key dimension</div>
                        </div>
                        <div className="bg-slate-700 rounded-lg p-3">
                            <div className="text-2xl font-bold text-white">d<sub>v</sub></div>
                            <div className="text-slate-400 text-sm">{Math.floor(512 / numHeads)}</div>
                            <div className="text-slate-500 text-xs">Value dimension</div>
                        </div>
                    </div>
                    <div className="text-center mt-4 text-slate-400 text-sm">
                        d<sub>model</sub> = h × d<sub>k</sub> → 512 = {numHeads} × {Math.floor(512 / numHeads)}
                    </div>
                </div>

                {/* Head Visualization Example */}
                <div className="mt-8 bg-slate-800/50 rounded-xl p-6 border border-slate-700">
                    <h4 className="text-white font-bold mb-4 text-center">
                        What Different Heads Learn (Example from BERT)
                    </h4>
                    <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-3">
                            <div className="flex items-center gap-3 bg-red-500/20 p-3 rounded-lg border border-red-500/30">
                                <div className="bg-red-500 w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold">1</div>
                                <div>
                                    <div className="text-red-400 font-medium">Direct Object</div>
                                    <div className="text-slate-400 text-xs">"The cat ate [the fish]" - verb → object</div>
                                </div>
                            </div>
                            <div className="flex items-center gap-3 bg-blue-500/20 p-3 rounded-lg border border-blue-500/30">
                                <div className="bg-blue-500 w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold">2</div>
                                <div>
                                    <div className="text-blue-400 font-medium">Possessive Pronouns</div>
                                    <div className="text-slate-400 text-xs">"[His] car" - possessive → noun</div>
                                </div>
                            </div>
                        </div>
                        <div className="space-y-3">
                            <div className="flex items-center gap-3 bg-green-500/20 p-3 rounded-lg border border-green-500/30">
                                <div className="bg-green-500 w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold">3</div>
                                <div>
                                    <div className="text-green-400 font-medium">Coreference</div>
                                    <div className="text-slate-400 text-xs">"The dog... [it] barked" - noun → pronoun</div>
                                </div>
                            </div>
                            <div className="flex items-center gap-3 bg-purple-500/20 p-3 rounded-lg border border-purple-500/30">
                                <div className="bg-purple-500 w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold">4</div>
                                <div>
                                    <div className="text-purple-400 font-medium">Sentence Delimiter</div>
                                    <div className="text-slate-400 text-xs">"Hello. [.]" - period positions</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Key Insights */}
                <div className="mt-8 bg-amber-500/10 rounded-xl p-6 border border-amber-500/30">
                    <h4 className="font-bold text-amber-400 mb-3 flex items-center gap-2">
                        <Lightbulb size={20} />
                        Why Multi-Head Attention Works
                    </h4>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                            <h5 className="text-white font-medium mb-2">Diverse Representations</h5>
                            <ul className="text-slate-400 space-y-1">
                                <li>• Each head has different W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub></li>
                                <li>• Learns different types of relationships</li>
                                <li>• Like an ensemble of attention mechanisms</li>
                            </ul>
                        </div>
                        <div>
                            <h5 className="text-white font-medium mb-2">Computational Efficiency</h5>
                            <ul className="text-slate-400 space-y-1">
                                <li>• Same total compute as single head</li>
                                <li>• d/h per head × h heads = d total</li>
                                <li>• Runs in parallel on GPU</li>
                            </ul>
                        </div>
                    </div>
                </div>

                {/* Full Formula */}
                <div className="mt-6 bg-slate-900 rounded-xl p-6 text-center">
                    <div className="text-slate-400 text-sm mb-2">The Multi-Head Attention Formula</div>
                    <div className="space-y-2 text-white font-mono">
                        <div>
                            MultiHead(Q,K,V) = Concat(head<sub>1</sub>, ..., head<sub>h</sub>)W<sup>O</sup>
                        </div>
                        <div className="text-slate-400 text-sm">where</div>
                        <div>
                            head<sub>i</sub> = Attention(QW<sub>i</sub><sup>Q</sup>, KW<sub>i</sub><sup>K</sup>, VW<sub>i</sub><sup>V</sup>)
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
                    animation: fadeIn 0.5s ease-out forwards;
                }
            `}</style>
        </div>
    );
}
