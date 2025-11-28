import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Eye, Lightbulb, Sparkles } from 'lucide-react';

export default function SelfAttentionPanel() {
    const [selectedWord, setSelectedWord] = useState(2); // "sat"
    const [isAnimating, setIsAnimating] = useState(false);
    const [showWeights, setShowWeights] = useState(true);

    const sentence = ['The', 'cat', 'sat', 'on', 'the', 'mat'];
    
    // Simulated attention weights for each word as query
    // Each row represents which words a query attends to
    const attentionMatrix = [
        [0.35, 0.25, 0.15, 0.10, 0.08, 0.07], // "The" attends to...
        [0.15, 0.30, 0.25, 0.12, 0.08, 0.10], // "cat" attends to...
        [0.10, 0.35, 0.20, 0.15, 0.08, 0.12], // "sat" attends to... (verb focuses on subject)
        [0.08, 0.10, 0.30, 0.22, 0.15, 0.15], // "on" attends to...
        [0.12, 0.08, 0.10, 0.15, 0.35, 0.20], // "the" attends to...
        [0.05, 0.15, 0.15, 0.20, 0.15, 0.30], // "mat" attends to...
    ];

    const getWeightColor = (weight) => {
        if (weight >= 0.3) return { bg: 'bg-green-500', opacity: 1 };
        if (weight >= 0.2) return { bg: 'bg-yellow-500', opacity: 0.9 };
        if (weight >= 0.15) return { bg: 'bg-orange-500', opacity: 0.7 };
        if (weight >= 0.1) return { bg: 'bg-red-500', opacity: 0.5 };
        return { bg: 'bg-slate-500', opacity: 0.3 };
    };

    const getBeamWidth = (weight) => {
        return Math.max(2, weight * 20);
    };

    return (
        <div className="p-6 min-h-screen">
            <div className="max-w-5xl mx-auto">
                {/* Header */}
                <div className="text-center mb-6">
                    <h2 className="text-3xl font-bold text-white mb-2">
                        Self-Attention: <span className="gradient-text">Live Visualization</span>
                    </h2>
                    <p className="text-slate-400">
                        Click any word to see what it "attends to" in the sentence
                    </p>
                </div>

                {/* Interactive Sentence */}
                <div className="bg-slate-800/50 rounded-2xl p-8 border border-slate-700 mb-8">
                    <div className="flex justify-center items-center gap-4 mb-8">
                        {sentence.map((word, i) => (
                            <button
                                key={i}
                                onClick={() => setSelectedWord(i)}
                                className={`relative px-6 py-4 rounded-xl font-bold text-lg transition-all ${
                                    selectedWord === i
                                        ? 'bg-blue-500 text-white scale-110 shadow-lg shadow-blue-500/50'
                                        : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                                }`}
                            >
                                {word}
                                {selectedWord === i && (
                                    <span className="absolute -top-2 -right-2 bg-yellow-400 text-slate-800 text-xs px-2 py-0.5 rounded-full">
                                        Query
                                    </span>
                                )}
                            </button>
                        ))}
                    </div>

                    {/* Attention Visualization */}
                    <div className="relative">
                        <div className="flex justify-center items-center gap-4">
                            {sentence.map((word, i) => {
                                const weight = attentionMatrix[selectedWord][i];
                                const color = getWeightColor(weight);
                                return (
                                    <div key={i} className="text-center">
                                        <div 
                                            className={`px-4 py-3 rounded-lg transition-all duration-500 ${
                                                i === selectedWord 
                                                    ? 'ring-2 ring-blue-400' 
                                                    : ''
                                            }`}
                                            style={{
                                                backgroundColor: `rgba(${
                                                    weight >= 0.3 ? '34, 197, 94' :
                                                    weight >= 0.2 ? '234, 179, 8' :
                                                    weight >= 0.15 ? '249, 115, 22' :
                                                    weight >= 0.1 ? '239, 68, 68' : '100, 116, 139'
                                                }, ${weight * 2})`,
                                            }}
                                        >
                                            <span className="text-white font-medium">{word}</span>
                                        </div>
                                        {showWeights && (
                                            <div className={`mt-2 text-sm font-mono ${
                                                weight >= 0.25 ? 'text-green-400' :
                                                weight >= 0.15 ? 'text-yellow-400' : 'text-slate-500'
                                            }`}>
                                                {(weight * 100).toFixed(0)}%
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>

                        {/* Connection Lines (SVG) */}
                        <svg className="absolute top-0 left-0 w-full h-full pointer-events-none" style={{ zIndex: -1 }}>
                            {sentence.map((_, i) => {
                                if (i === selectedWord) return null;
                                const weight = attentionMatrix[selectedWord][i];
                                const sourceX = (selectedWord + 0.5) * (100 / sentence.length);
                                const targetX = (i + 0.5) * (100 / sentence.length);
                                return (
                                    <line
                                        key={i}
                                        x1={`${sourceX}%`}
                                        y1="50%"
                                        x2={`${targetX}%`}
                                        y2="50%"
                                        stroke={
                                            weight >= 0.3 ? '#22c55e' :
                                            weight >= 0.2 ? '#eab308' :
                                            weight >= 0.15 ? '#f97316' : '#64748b'
                                        }
                                        strokeWidth={getBeamWidth(weight)}
                                        strokeOpacity={weight * 1.5}
                                        className="transition-all duration-500"
                                    />
                                );
                            })}
                        </svg>
                    </div>

                    {/* Toggle */}
                    <div className="flex justify-center mt-6">
                        <button
                            onClick={() => setShowWeights(!showWeights)}
                            className="text-sm text-slate-400 hover:text-white"
                        >
                            {showWeights ? 'Hide' : 'Show'} percentages
                        </button>
                    </div>
                </div>

                {/* Attention Matrix Heatmap */}
                <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700 mb-8">
                    <h3 className="text-white font-bold mb-4 text-center">Full Attention Matrix</h3>
                    <p className="text-slate-400 text-sm text-center mb-4">
                        Rows = Query words, Columns = Key words. Brighter = higher attention.
                    </p>
                    
                    <div className="overflow-x-auto flex justify-center">
                        <div>
                            {/* Header row */}
                            <div className="flex gap-1 mb-1 ml-20">
                                {sentence.map((word, i) => (
                                    <div 
                                        key={i} 
                                        className={`w-14 text-center text-xs ${
                                            selectedWord === i ? 'text-blue-400 font-bold' : 'text-slate-500'
                                        }`}
                                    >
                                        {word}
                                    </div>
                                ))}
                            </div>

                            {/* Matrix rows */}
                            {sentence.map((word, qIdx) => (
                                <div 
                                    key={qIdx} 
                                    className={`flex items-center gap-1 mb-1 ${
                                        selectedWord === qIdx ? 'bg-blue-500/10 rounded-lg' : ''
                                    }`}
                                >
                                    <div className={`w-20 text-right pr-3 text-xs ${
                                        selectedWord === qIdx ? 'text-blue-400 font-bold' : 'text-slate-500'
                                    }`}>
                                        {word}
                                    </div>
                                    {attentionMatrix[qIdx].map((weight, kIdx) => (
                                        <div
                                            key={kIdx}
                                            onClick={() => setSelectedWord(qIdx)}
                                            className={`w-14 h-10 rounded flex items-center justify-center font-mono text-xs cursor-pointer transition-all hover:scale-105 ${
                                                qIdx === selectedWord && kIdx === selectedWord ? 'ring-2 ring-white' : ''
                                            }`}
                                            style={{
                                                backgroundColor: `rgba(34, 197, 94, ${weight * 1.5})`,
                                                color: weight > 0.2 ? 'white' : '#94a3b8'
                                            }}
                                        >
                                            {(weight * 100).toFixed(0)}
                                        </div>
                                    ))}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Interpretation */}
                <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-2xl p-6 border border-purple-500/20 mb-8">
                    <h3 className="text-white font-bold mb-3 flex items-center gap-2">
                        <Eye size={20} className="text-purple-400" />
                        Interpreting the Attention Pattern
                    </h3>
                    
                    <div className="space-y-3 text-sm">
                        <div className="flex items-start gap-3 p-3 bg-slate-800/50 rounded-lg">
                            <span className="bg-blue-500 px-2 py-1 rounded text-white text-xs font-bold">Query</span>
                            <p className="text-slate-300">
                                <strong>"{sentence[selectedWord]}"</strong> is looking at the sentence from its perspective.
                            </p>
                        </div>
                        
                        <div className="flex items-start gap-3 p-3 bg-slate-800/50 rounded-lg">
                            <span className="bg-green-500 px-2 py-1 rounded text-white text-xs font-bold">Strongest</span>
                            <p className="text-slate-300">
                                Attends most to <strong>"{sentence[attentionMatrix[selectedWord].indexOf(Math.max(...attentionMatrix[selectedWord]))]}"</strong> 
                                {' '}({(Math.max(...attentionMatrix[selectedWord]) * 100).toFixed(0)}%)
                                {selectedWord === 2 && ' — The verb "sat" focuses on its subject "cat"!'}
                            </p>
                        </div>

                        <div className="flex items-start gap-3 p-3 bg-slate-800/50 rounded-lg">
                            <span className="bg-slate-500 px-2 py-1 rounded text-white text-xs font-bold">Pattern</span>
                            <p className="text-slate-300">
                                {selectedWord <= 1 
                                    ? 'Articles and subjects often attend to themselves and nearby content words.'
                                    : selectedWord === 2
                                        ? 'Verbs strongly attend to their subjects and nearby prepositions.'
                                        : selectedWord === 3
                                            ? 'Prepositions attend to both the verb they modify and their objects.'
                                            : 'Nouns at the end attend to modifying prepositions and determiners.'
                                }
                            </p>
                        </div>
                    </div>
                </div>

                {/* Real-world Examples */}
                <div className="bg-amber-500/10 rounded-2xl p-6 border border-amber-500/30">
                    <h3 className="text-amber-400 font-bold mb-4 flex items-center gap-2">
                        <Lightbulb size={20} />
                        What Attention Heads Actually Learn
                    </h3>
                    
                    <div className="grid grid-cols-2 gap-4 text-sm">
                        <div className="space-y-2">
                            <h4 className="text-white font-medium">Syntax-focused heads:</h4>
                            <ul className="text-slate-400 space-y-1">
                                <li>• Subject → Verb relationships</li>
                                <li>• Determiner → Noun pairs</li>
                                <li>• Preposition → Object links</li>
                            </ul>
                        </div>
                        <div className="space-y-2">
                            <h4 className="text-white font-medium">Semantic-focused heads:</h4>
                            <ul className="text-slate-400 space-y-1">
                                <li>• Pronoun → Antecedent (coreference)</li>
                                <li>• Rare words → Context clues</li>
                                <li>• Named entities → Related terms</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div className="mt-4 p-3 bg-slate-800/50 rounded-lg">
                        <p className="text-slate-300 text-sm">
                            <strong className="text-amber-400">💡 Key insight:</strong> Different attention heads in real transformers 
                            specialize in different linguistic patterns. Some focus on local syntax, others on long-range 
                            semantic relationships. This is why multi-head attention is so powerful!
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
