import React, { useState } from 'react';

export default function PlaygroundPanel() {
    const [sentence, setSentence] = useState("The animal didn't cross the street because it was too tired");
    const [hoveredIndex, setHoveredIndex] = useState(null);

    const words = sentence.split(/\s+/).filter(w => w);

    // Mock attention weights logic (simple heuristic for demo)
    // In a real model, these would come from the trained weights
    const getAttentionWeights = (focusIndex) => {
        if (focusIndex === null) return Array(words.length).fill(0);

        const focusWord = words[focusIndex].toLowerCase();

        return words.map((w, i) => {
            const word = w.toLowerCase();

            // Self-attention always has some weight
            if (i === focusIndex) return 0.2;

            // "it" usually refers to the subject ("animal")
            if (focusWord === 'it' && (word === 'animal' || word === 'street')) {
                return word === 'animal' ? 0.8 : 0.1; // Resolving "it" -> "animal"
            }

            // "tired" usually refers to animate objects
            if (focusWord === 'tired' && word === 'animal') return 0.6;

            // "cross" relates to "street"
            if (focusWord === 'cross' && word === 'street') return 0.5;

            // Default low attention
            return 0.05;
        });
    };

    const weights = getAttentionWeights(hoveredIndex);

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-indigo-600 dark:text-indigo-400 mb-4">Attention Playground</h2>
                <p className="text-lg text-slate-700 dark:text-slate-300 leading-relaxed">
                    Hover over a word to see what it "pays attention" to.
                    <br />
                    <span className="text-sm text-slate-800 dark:text-slate-400">Simulated weights for coreference resolution.</span>
                </p>
            </div>

            {/* Input */}
            <div className="w-full max-w-4xl mb-12">
                <input
                    type="text"
                    value={sentence}
                    onChange={(e) => setSentence(e.target.value)}
                    className="w-full bg-slate-800 border border-slate-600 rounded-xl p-4 text-xl text-center text-white focus:border-indigo-500 outline-none"
                />
            </div>

            {/* Visualization */}
            <div className="relative w-full max-w-5xl h-64 flex items-center justify-center">
                <div className="flex flex-wrap justify-center gap-4 relative z-10">
                    {words.map((word, i) => {
                        const weight = weights[i];
                        const isFocus = i === hoveredIndex;

                        return (
                            <div
                                key={i}
                                onMouseEnter={() => setHoveredIndex(i)}
                                onMouseLeave={() => setHoveredIndex(null)}
                                className={`relative px-4 py-2 rounded-lg cursor-pointer transition-all duration-300 ${isFocus
                                        ? 'bg-indigo-600 text-white scale-110 shadow-lg ring-2 ring-indigo-400'
                                        : 'bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-700'
                                    }`}
                                style={{
                                    // Visualize attention weight as opacity/border when hovering another word
                                    opacity: hoveredIndex !== null && !isFocus ? Math.max(0.3, weight + 0.2) : 1,
                                    borderColor: hoveredIndex !== null && !isFocus ? `rgba(99, 102, 241, ${weight})` : 'transparent',
                                    borderWidth: hoveredIndex !== null && !isFocus ? '2px' : '0px'
                                }}
                            >
                                {word}

                                {/* Attention Score Tooltip */}
                                {hoveredIndex !== null && !isFocus && weight > 0.1 && (
                                    <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-indigo-900 text-xs px-2 py-1 rounded opacity-100 whitespace-nowrap">
                                        {(weight * 100).toFixed(0)}%
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>

                {/* Connection Lines (SVG Overlay) */}
                {hoveredIndex !== null && (
                    <svg className="absolute inset-0 w-full h-full pointer-events-none overflow-visible">
                        <defs>
                            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="28" refY="3.5" orient="auto">
                                <polygon points="0 0, 10 3.5, 0 7" fill="#6366f1" />
                            </marker>
                        </defs>
                        {/* 
              Note: Drawing actual lines between dynamic DOM elements in React without a library like react-archer 
              is complex due to coordinate calculation. 
              For this demo, we visualized attention via opacity/badges above.
              
              In a full production app, we would calculate getBoundingClientRect() for each word 
              and draw SVG paths here.
            */}
                    </svg>
                )}
            </div>

            {/* Explanation Box */}
            {hoveredIndex !== null && (
                <div className="mt-8 p-6 bg-slate-800 rounded-xl border border-indigo-500/50 max-w-2xl animate-fade-in">
                    <h3 className="font-bold text-white mb-2">
                        Focusing on: <span className="text-indigo-600 dark:text-indigo-400">"{words[hoveredIndex]}"</span>
                    </h3>
                    <p className="text-slate-700 dark:text-sm">
                        {words[hoveredIndex].toLowerCase() === 'it' && sentence.includes('animal') ? (
                            <span>
                                The model attends strongly to <strong>"animal"</strong> to resolve what "it" refers to.
                                This is called <em>Coreference Resolution</em>.
                            </span>
                        ) : (
                            <span>
                                The model looks at other words in the sentence to build context for "{words[hoveredIndex]}".
                                Stronger attention means more information flows from that word.
                            </span>
                        )}
                    </p>
                </div>
            )}
        </div>
    );
}
