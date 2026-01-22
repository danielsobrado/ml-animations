import React, { useState } from 'react';

export default function MechanismPanel() {
    // Simplified 2D vectors for visualization
    // Word 1: "Bank" (Financial)
    // Word 2: "River" (Nature)

    const [q1, setQ1] = useState([1.0, 0.2]); // Query for "Bank"
    const [k1, setK1] = useState([1.0, 0.1]); // Key for "Bank"
    const [k2, setK2] = useState([0.1, 1.0]); // Key for "River"
    const [v1, setV1] = useState([2.0, 0.5]); // Value for "Bank"
    const [v2, setV2] = useState([0.5, 2.0]); // Value for "River"

    // Step 1: Dot Products (Scores)
    // Score = Q . K
    const score1 = q1[0] * k1[0] + q1[1] * k1[1];
    const score2 = q1[0] * k2[0] + q1[1] * k2[1];

    // Step 2: Softmax
    const exp1 = Math.exp(score1);
    const exp2 = Math.exp(score2);
    const sumExp = exp1 + exp2;
    const attn1 = exp1 / sumExp;
    const attn2 = exp2 / sumExp;

    // Step 3: Weighted Sum (Output)
    const out0 = attn1 * v1[0] + attn2 * v2[0];
    const out1 = attn1 * v1[1] + attn2 * v2[1];

    const VectorInput = ({ label, vec, setVec, color }) => (
        <div className="flex items-center gap-2 mb-2">
            <span className={`font-bold w-8 ${color}`}>{label}</span>
            <input
                type="number" step="0.1" value={vec[0]}
                onChange={e => setVec([parseFloat(e.target.value), vec[1]])}
                className="w-16 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-sm text-center"
            />
            <input
                type="number" step="0.1" value={vec[1]}
                onChange={e => setVec([vec[0], parseFloat(e.target.value)])}
                className="w-16 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-sm text-center"
            />
        </div>
    );

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-purple-600 dark:text-purple-400 mb-4">The Math Mechanism</h2>
                <div className="bg-slate-800 p-4 rounded-lg font-mono text-sm inline-block">
                    <span className="text-fuchsia-600 dark:text-fuchsia-400">Attention(Q, K, V)</span> =
                    softmax(<span className="text-blue-600 dark:text-blue-400">Q</span> · <span className="text-green-400">Kᵀ</span> / √d) · <span className="text-orange-600 dark:text-orange-400">V</span>
                </div>
            </div>

            <div className="grid lg:grid-cols-3 gap-8 w-full max-w-6xl items-start">
                {/* Inputs */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                    <h3 className="font-bold text-white mb-4 border-b border-slate-700 pb-2">1. Inputs</h3>

                    <div className="mb-6">
                        <h4 className="text-sm text-slate-800 dark:text-slate-400 mb-2">Query (What we focus on)</h4>
                        <VectorInput label="Q1" vec={q1} setVec={setQ1} color="text-blue-600 dark:text-blue-400" />
                    </div>

                    <div className="mb-6">
                        <h4 className="text-sm text-slate-800 dark:text-slate-400 mb-2">Keys (What we match against)</h4>
                        <VectorInput label="K1" vec={k1} setVec={setK1} color="text-green-400" />
                        <VectorInput label="K2" vec={k2} setVec={setK2} color="text-green-400" />
                    </div>

                    <div>
                        <h4 className="text-sm text-slate-800 dark:text-slate-400 mb-2">Values (Content to retrieve)</h4>
                        <VectorInput label="V1" vec={v1} setVec={setV1} color="text-orange-600 dark:text-orange-400" />
                        <VectorInput label="V2" vec={v2} setVec={setV2} color="text-orange-600 dark:text-orange-400" />
                    </div>
                </div>

                {/* Calculations */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                    <h3 className="font-bold text-white mb-4 border-b border-slate-700 pb-2">2. Calculations</h3>

                    <div className="mb-6">
                        <h4 className="text-sm text-slate-800 dark:text-slate-400 mb-2">Dot Products (Scores)</h4>
                        <div className="font-mono text-sm bg-slate-900 p-2 rounded mb-1">
                            Q1 · K1 = {(score1).toFixed(2)}
                        </div>
                        <div className="font-mono text-sm bg-slate-900 p-2 rounded">
                            Q1 · K2 = {(score2).toFixed(2)}
                        </div>
                    </div>

                    <div className="mb-6">
                        <h4 className="text-sm text-slate-800 dark:text-slate-400 mb-2">Softmax (Probabilities)</h4>
                        <div className="relative h-8 bg-slate-900 rounded overflow-hidden flex">
                            <div style={{ width: `${attn1 * 100}%` }} className="bg-green-500/50 flex items-center justify-center text-xs text-white">
                                {(attn1 * 100).toFixed(0)}%
                            </div>
                            <div style={{ width: `${attn2 * 100}%` }} className="bg-slate-700 flex items-center justify-center text-xs text-white">
                                {(attn2 * 100).toFixed(0)}%
                            </div>
                        </div>
                        <div className="flex justify-between text-xs text-slate-700 dark:text-slate-500 mt-1">
                            <span>Attn to K1</span>
                            <span>Attn to K2</span>
                        </div>
                    </div>
                </div>

                {/* Output */}
                <div className="bg-slate-800 p-6 rounded-xl border border-fuchsia-500/50">
                    <h3 className="font-bold text-white mb-4 border-b border-slate-700 pb-2">3. Output</h3>

                    <div className="mb-6">
                        <h4 className="text-sm text-slate-800 dark:text-slate-400 mb-2">Weighted Sum</h4>
                        <div className="font-mono text-xs text-slate-700 dark:text-slate-300 mb-2">
                            {attn1.toFixed(2)} * V1 + {attn2.toFixed(2)} * V2
                        </div>
                        <div className="bg-fuchsia-900/30 border border-fuchsia-500 p-4 rounded-xl text-center">
                            <div className="text-3xl font-mono font-bold text-fuchsia-600 dark:text-fuchsia-400">
                                [{out0.toFixed(2)}, {out1.toFixed(2)}]
                            </div>
                            <p className="text-xs text-slate-800 dark:text-slate-400 mt-2">Context-Aware Embedding</p>
                        </div>
                    </div>

                    <div className="text-xs text-slate-800 dark:text-slate-400">
                        Notice: If Q matches K1 closely, the Output looks very similar to V1.
                        <br />
                        If Q matches K2, Output looks like V2.
                        <br />
                        This is how the model "retrieves" relevant information!
                    </div>
                </div>
            </div>
        </div>
    );
}
