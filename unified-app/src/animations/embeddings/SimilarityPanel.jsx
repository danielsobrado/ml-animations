import React, { useState } from 'react';

export default function SimilarityPanel() {
    const [angle, setAngle] = useState(45); // Degrees

    // Convert to radians
    const rad = (angle * Math.PI) / 180;

    // Calculate Cosine Similarity
    const cosineSim = Math.cos(rad);

    // Vector coordinates (radius 150)
    const r = 150;
    const v1 = { x: r, y: 0 }; // Fixed on X axis
    const v2 = { x: r * Math.cos(rad), y: -r * Math.sin(rad) }; // Rotated (Y inverted for SVG)

    const getLabel = (sim) => {
        if (sim > 0.9) return "Very Similar (Synonyms)";
        if (sim > 0.5) return "Related";
        if (sim > -0.1 && sim < 0.1) return "Unrelated (Orthogonal)";
        if (sim < -0.5) return "Opposites";
        return "Somewhat Related";
    };

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-cyan-400 mb-4">Similarity Lab</h2>
                <p className="text-lg text-slate-300 leading-relaxed">
                    How do we know if "Cat" is close to "Dog"?
                    <br />
                    We measure the <strong>Cosine of the Angle</strong> between their vectors.
                </p>
            </div>

            <div className="flex flex-col md:flex-row gap-12 items-center w-full max-w-5xl">
                {/* Visualization */}
                <div className="w-[400px] h-[400px] bg-slate-900 rounded-full border-4 border-slate-700 relative flex items-center justify-center shadow-[0_0_50px_rgba(34,211,238,0.1)]">
                    <svg className="w-full h-full overflow-visible" viewBox="-200 -200 400 400">
                        <defs>
                            <marker id="arrow-sim" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                                <polygon points="0 0, 10 3.5, 0 7" fill="#cbd5e1" />
                            </marker>
                        </defs>

                        {/* Center */}
                        <circle cx="0" cy="0" r="5" fill="white" />

                        {/* Vector 1 (Fixed) */}
                        <line x1="0" y1="0" x2={v1.x} y2={v1.y} stroke="#94a3b8" strokeWidth="4" markerEnd="url(#arrow-sim)" />
                        <text x={v1.x + 20} y={v1.y + 5} fill="#94a3b8" fontWeight="bold">Word A</text>

                        {/* Vector 2 (Rotatable) */}
                        <line x1="0" y1="0" x2={v2.x} y2={v2.y} stroke="#22d3ee" strokeWidth="4" markerEnd="url(#arrow-sim)" />
                        <text x={v2.x * 1.2} y={v2.y * 1.2} fill="#22d3ee" fontWeight="bold" textAnchor="middle">Word B</text>

                        {/* Angle Arc */}
                        <path
                            d={`M 50 0 A 50 50 0 ${angle > 180 ? 1 : 0} 0 ${50 * Math.cos(-rad)} ${50 * Math.sin(-rad)}`}
                            fill="none"
                            stroke="#fbbf24"
                            strokeWidth="2"
                            strokeDasharray="4 4"
                        />
                        <text x="60" y="-20" fill="#fbbf24" fontSize="12">{angle}°</text>
                    </svg>
                </div>

                {/* Controls & Stats */}
                <div className="flex-1 w-full max-w-md bg-slate-800 p-8 rounded-2xl border border-slate-700 shadow-xl">
                    <div className="mb-8">
                        <label className="flex justify-between text-sm font-bold text-slate-400 mb-4">
                            Angle: <span className="text-white">{angle}°</span>
                        </label>
                        <input
                            type="range" min="0" max="180" step="1"
                            value={angle}
                            onChange={(e) => setAngle(Number(e.target.value))}
                            className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                        />
                    </div>

                    <div className="text-center">
                        <h3 className="text-sm uppercase tracking-widest text-slate-500 mb-2">Cosine Similarity</h3>
                        <div className={`text-5xl font-mono font-bold mb-4 ${cosineSim > 0.5 ? 'text-green-400' : cosineSim < -0.5 ? 'text-red-400' : 'text-slate-200'}`}>
                            {cosineSim.toFixed(3)}
                        </div>
                        <div className="inline-block px-4 py-2 bg-slate-900 rounded-lg border border-slate-700 text-cyan-300 font-bold">
                            {getLabel(cosineSim)}
                        </div>
                    </div>

                    <div className="mt-8 pt-8 border-t border-slate-700 text-sm text-slate-400 space-y-2">
                        <p><strong>1.0</strong> = Same Direction (0°)</p>
                        <p><strong>0.0</strong> = Unrelated (90°)</p>
                        <p><strong>-1.0</strong> = Opposite Direction (180°)</p>
                    </div>
                </div>
            </div>
        </div>
    );
}
