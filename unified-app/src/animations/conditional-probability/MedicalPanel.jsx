import React, { useState } from 'react';

export default function MedicalPanel() {
    const [prevalence, setPrevalence] = useState(0.01); // 1% have disease
    const [sensitivity, setSensitivity] = useState(0.95); // 95% true positive rate
    const [specificity, setSpecificity] = useState(0.90); // 90% true negative rate

    const falsePositiveRate = 1 - specificity;

    // Bayes' Theorem
    const probPosGivenDisease = sensitivity;
    const probPosGivenHealthy = falsePositiveRate;
    const probPos = probPosGivenDisease * prevalence + probPosGivenHealthy * (1 - prevalence);
    const probDiseaseGivenPos = probPos > 0 ? (probPosGivenDisease * prevalence) / probPos : 0;

    // Population of 1000
    const population = 1000;
    const diseased = Math.round(population * prevalence);
    const healthy = population - diseased;
    const truePositives = Math.round(diseased * sensitivity);
    const falseNegatives = diseased - truePositives;
    const falsePositives = Math.round(healthy * falsePositiveRate);
    const trueNegatives = healthy - falsePositives;
    const totalPositives = truePositives + falsePositives;

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-emerald-400 mb-4">Medical Testing</h2>
                <p className="text-lg text-slate-300 leading-relaxed">
                    You test <strong className="text-red-400">positive</strong> for a rare disease.
                    <br />
                    What's the probability you actually <em>have</em> the disease?
                </p>
            </div>

            {/* Controls */}
            <div className="grid md:grid-cols-3 gap-6 w-full max-w-5xl mb-8">
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                    <label className="flex justify-between text-sm font-bold mb-3">
                        Prevalence: <span className="text-purple-400">{(prevalence * 100).toFixed(1)}%</span>
                    </label>
                    <input
                        type="range" min="0.001" max="0.1" step="0.001"
                        value={prevalence}
                        onChange={(e) => setPrevalence(Number(e.target.value))}
                        className="w-full accent-purple-400"
                    />
                    <p className="text-xs text-slate-400 mt-2">% of population with disease</p>
                </div>

                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                    <label className="flex justify-between text-sm font-bold mb-3">
                        Sensitivity: <span className="text-green-400">{(sensitivity * 100).toFixed(0)}%</span>
                    </label>
                    <input
                        type="range" min="0.5" max="1" step="0.01"
                        value={sensitivity}
                        onChange={(e) => setSensitivity(Number(e.target.value))}
                        className="w-full accent-green-400"
                    />
                    <p className="text-xs text-slate-400 mt-2">P(Pos | Disease) - True Positive Rate</p>
                </div>

                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                    <label className="flex justify-between text-sm font-bold mb-3">
                        Specificity: <span className="text-blue-400">{(specificity * 100).toFixed(0)}%</span>
                    </label>
                    <input
                        type="range" min="0.5" max="1" step="0.01"
                        value={specificity}
                        onChange={(e) => setSpecificity(Number(e.target.value))}
                        className="w-full accent-blue-400"
                    />
                    <p className="text-xs text-slate-400 mt-2">P(Neg | Healthy) - True Negative Rate</p>
                </div>
            </div>

            {/* Result */}
            <div className="bg-gradient-to-br from-cyan-900/50 to-emerald-900/50 p-8 rounded-xl border-2 border-cyan-500 w-full max-w-5xl mb-8">
                <h3 className="font-bold text-white mb-4 text-center text-xl">
                    If you test POSITIVE, probability you have the disease:
                </h3>
                <div className="text-center">
                    <div className="text-7xl font-mono font-bold text-cyan-400 mb-2">
                        {(probDiseaseGivenPos * 100).toFixed(1)}%
                    </div>
                    <p className="text-slate-300 text-lg">P(Disease | Positive Test)</p>
                </div>
                {probDiseaseGivenPos < 0.5 && (
                    <div className="mt-6 p-4 bg-yellow-900/30 rounded-lg border border-yellow-700">
                        <p className="text-yellow-300 text-sm text-center">
                            ⚠️ Surprising! Even with a positive test, you're more likely to be healthy!
                            <br />
                            This is because the disease is <strong>rare</strong> (low prevalence).
                        </p>
                    </div>
                )}
            </div>

            {/* Population Visualization */}
            <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 w-full max-w-5xl">
                <h3 className="font-bold text-white mb-4 text-center">
                    Population of {population} People
                </h3>

                <div className="grid grid-cols-2 gap-6 mb-6">
                    <div className="bg-red-900/30 p-4 rounded-lg border border-red-700">
                        <h4 className="font-bold text-red-400 mb-3 text-center">Diseased ({diseased})</h4>
                        <div className="space-y-2">
                            <div className="flex justify-between items-center p-2 bg-slate-900 rounded">
                                <span className="text-sm text-slate-300">True Positives:</span>
                                <span className="text-green-400 font-bold">{truePositives}</span>
                            </div>
                            <div className="flex justify-between items-center p-2 bg-slate-900 rounded">
                                <span className="text-sm text-slate-300">False Negatives:</span>
                                <span className="text-red-400 font-bold">{falseNegatives}</span>
                            </div>
                        </div>
                    </div>

                    <div className="bg-green-900/30 p-4 rounded-lg border border-green-700">
                        <h4 className="font-bold text-green-400 mb-3 text-center">Healthy ({healthy})</h4>
                        <div className="space-y-2">
                            <div className="flex justify-between items-center p-2 bg-slate-900 rounded">
                                <span className="text-sm text-slate-300">False Positives:</span>
                                <span className="text-yellow-400 font-bold">{falsePositives}</span>
                            </div>
                            <div className="flex justify-between items-center p-2 bg-slate-900 rounded">
                                <span className="text-sm text-slate-300">True Negatives:</span>
                                <span className="text-blue-400 font-bold">{trueNegatives}</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="bg-cyan-900/30 p-4 rounded-lg border border-cyan-700">
                    <p className="text-cyan-300 text-center font-bold">
                        Total Positive Tests: {totalPositives} = {truePositives} (true) + {falsePositives} (false)
                        <br />
                        <span className="text-sm text-slate-400 mt-2 block">
                            Of those who test positive, only {truePositives}/{totalPositives} = {(probDiseaseGivenPos * 100).toFixed(1)}% actually have the disease!
                        </span>
                    </p>
                </div>
            </div>
        </div>
    );
}
