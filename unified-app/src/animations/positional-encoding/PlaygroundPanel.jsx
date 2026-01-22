import React, { useState } from 'react';

export default function PlaygroundPanel() {
    const [sentence, setSentence] = useState('The cat sat on the mat');
    const [encodingType, setEncodingType] = useState('sinusoidal');

    const words = sentence.split(' ').filter(w => w.length > 0);

    // Generate different types of encodings
    const generateEncoding = (pos, type, dim = 8) => {
        switch (type) {
            case 'sinusoidal':
                return Array.from({ length: dim }, (_, i) => {
                    const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / dim);
                    return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
                });
            case 'learned':
                // Simulate learned embeddings (random but consistent per position)
                const seed = pos * 12345;
                return Array.from({ length: dim }, (_, i) => {
                    const x = Math.sin(seed + i) * 10000;
                    return (x - Math.floor(x)) * 2 - 1;
                });
            case 'integer':
                // Simple integer encoding (not recommended but shown for comparison)
                return Array.from({ length: dim }, (_, i) => i === 0 ? pos / 10 : 0);
            default:
                return Array(dim).fill(0);
        }
    };

    const getColor = (value) => {
        // Map -1 to 1 to a color gradient
        const normalized = (value + 1) / 2; // 0 to 1
        const r = Math.floor(normalized * 255);
        const b = Math.floor((1 - normalized) * 255);
        return `rgb(${r}, 100, ${b})`;
    };

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-cyan-600 dark:text-cyan-400 mb-4">Encoding Playground</h2>
                <p className="text-lg text-slate-700 dark:text-slate-300 leading-relaxed">
                    Compare different positional encoding strategies.
                </p>
            </div>

            {/* Input */}
            <div className="w-full max-w-4xl mb-8">
                <label className="block text-sm font-bold text-slate-700 dark:text-slate-300 mb-2">Your Sentence:</label>
                <input
                    type="text"
                    value={sentence}
                    onChange={(e) => setSentence(e.target.value)}
                    placeholder="Type a sentence..."
                    className="w-full px-4 py-3 bg-slate-800 border-2 border-slate-700 rounded-xl text-white focus:border-cyan-500 focus:outline-none"
                />
            </div>

            {/* Encoding Type Selector */}
            <div className="flex gap-4 mb-8">
                {[
                    { id: 'sinusoidal', label: 'Sinusoidal (Transformer)', color: 'cyan' },
                    { id: 'learned', label: 'Learned Embeddings', color: 'purple' },
                    { id: 'integer', label: 'Simple Integer', color: 'slate' }
                ].map(type => (
                    <button
                        key={type.id}
                        onClick={() => setEncodingType(type.id)}
                        className={`px-6 py-3 rounded-xl font-bold transition-all ${encodingType === type.id
                                ? `bg-${type.color}-600 text-white shadow-lg scale-105`
                                : 'bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-600'
                            }`}
                    >
                        {type.label}
                    </button>
                ))}
            </div>

            {/* Visualization */}
            <div className="w-full max-w-5xl bg-slate-800 p-6 rounded-xl border border-slate-700">
                <h3 className="font-bold text-white mb-4 text-center">
                    {encodingType === 'sinusoidal' && '🌊 Sinusoidal Encoding (Fixed, Generalizes)'}
                    {encodingType === 'learned' && '🎓 Learned Encoding (Trained, Limited)'}
                    {encodingType === 'integer' && '🔢 Integer Encoding (Too Simple)'}
                </h3>

                <div className="space-y-4">
                    {words.map((word, pos) => {
                        const encoding = generateEncoding(pos, encodingType);
                        return (
                            <div key={pos} className="bg-slate-900 p-4 rounded-lg border border-slate-600">
                                <div className="flex items-center justify-between mb-3">
                                    <div className="flex items-center gap-3">
                                        <span className="text-xs bg-cyan-600 px-2 py-1 rounded-full font-mono">
                                            pos: {pos}
                                        </span>
                                        <span className="font-bold text-white text-lg">{word}</span>
                                    </div>
                                </div>

                                {/* Encoding Heatmap */}
                                <div className="flex gap-1">
                                    {encoding.map((val, i) => (
                                        <div
                                            key={i}
                                            className="flex-1 h-12 rounded transition-all hover:scale-110 cursor-pointer relative group"
                                            style={{ backgroundColor: getColor(val) }}
                                            title={`Dim ${i}: ${val.toFixed(3)}`}
                                        >
                                            <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-black text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                                                {val.toFixed(3)}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        );
                    })}
                </div>

                {/* Legend */}
                <div className="mt-6 pt-6 border-t border-slate-700">
                    <div className="flex items-center justify-center gap-4 text-sm">
                        <div className="flex items-center gap-2">
                            <div className="w-6 h-6 rounded" style={{ backgroundColor: getColor(-1) }}></div>
                            <span className="text-slate-800 dark:text-slate-400">-1.0</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-6 h-6 rounded" style={{ backgroundColor: getColor(0) }}></div>
                            <span className="text-slate-800 dark:text-slate-400">0.0</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-6 h-6 rounded" style={{ backgroundColor: getColor(1) }}></div>
                            <span className="text-slate-800 dark:text-slate-400">+1.0</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Comparison Notes */}
            <div className="grid md:grid-cols-3 gap-4 w-full max-w-5xl mt-8">
                <div className="bg-cyan-900/30 border-2 border-cyan-500 p-4 rounded-lg">
                    <h4 className="font-bold text-cyan-300 mb-2">✅ Sinusoidal</h4>
                    <p className="text-sm text-slate-700 dark:text-slate-300">
                        • Fixed (no training needed)
                        <br />• Generalizes to any length
                        <br />• Unique pattern per position
                    </p>
                </div>
                <div className="bg-purple-900/30 border-2 border-purple-500 p-4 rounded-lg">
                    <h4 className="font-bold text-purple-300 mb-2">⚠️ Learned</h4>
                    <p className="text-sm text-slate-700 dark:text-slate-300">
                        • Requires training
                        <br />• Limited to max length
                        <br />• Can be more flexible
                    </p>
                </div>
                <div className="bg-slate-700/50 border-2 border-slate-500 p-4 rounded-lg">
                    <h4 className="font-bold text-slate-700 dark:text-slate-300 mb-2">❌ Integer</h4>
                    <p className="text-sm text-slate-700 dark:text-slate-300">
                        • Too simple
                        <br />• Doesn't capture patterns
                        <br />• Poor for learning
                    </p>
                </div>
            </div>
        </div>
    );
}
