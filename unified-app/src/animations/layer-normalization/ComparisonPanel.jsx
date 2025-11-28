import React, { useState } from 'react';

export default function ComparisonPanel() {
    const [batchSize, setBatchSize] = useState(32);

    // Mock data: batch of samples with features
    const generateBatch = (size) => {
        return Array.from({ length: size }, (_, i) => ({
            sample: i,
            features: Array.from({ length: 4 }, () => Math.random() * 2 - 1)
        }));
    };

    const batch = generateBatch(batchSize);

    // Batch Norm: normalize across batch dimension (for each feature)
    const batchNorm = () => {
        const numFeatures = 4;
        const normalized = [];

        for (let f = 0; f < numFeatures; f++) {
            // Get all values for this feature across the batch
            const values = batch.map(s => s.features[f]);
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
            const std = Math.sqrt(variance + 1e-5);

            normalized.push({
                feature: f,
                mean: mean.toFixed(3),
                std: std.toFixed(3),
                normalized: values.map(v => ((v - mean) / std).toFixed(3))
            });
        }

        return normalized;
    };

    // Layer Norm: normalize across feature dimension (for each sample)
    const layerNorm = () => {
        return batch.map(sample => {
            const mean = sample.features.reduce((a, b) => a + b, 0) / sample.features.length;
            const variance = sample.features.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / sample.features.length;
            const std = Math.sqrt(variance + 1e-5);

            return {
                sample: sample.sample,
                mean: mean.toFixed(3),
                std: std.toFixed(3),
                normalized: sample.features.map(f => ((f - mean) / std).toFixed(3))
            };
        });
    };

    const batchNormResult = batchNorm();
    const layerNormResult = layerNorm();

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-violet-400 mb-4">Layer Norm vs Batch Norm</h2>
                <p className="text-lg text-slate-300 leading-relaxed mb-4">
                    The key difference: <strong>which axis to normalize</strong>.
                </p>
                <div className="bg-slate-800 p-4 rounded-lg font-mono text-sm text-left">
                    <p className="text-cyan-300">y = γ * (x - μ) / σ + β</p>
                    <p className="text-slate-400 mt-2">
                        • Batch Norm: μ, σ computed across <strong>batch</strong> (per feature)
                        <br />
                        • Layer Norm: μ, σ computed across <strong>features</strong> (per sample)
                    </p>
                </div>
            </div>

            {/* Batch Size Control */}
            <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 w-full max-w-4xl mb-8">
                <label className="flex justify-between text-sm font-bold mb-3">
                    Batch Size: <span className="text-violet-400">{batchSize}</span>
                </label>
                <input
                    type="range" min="1" max="64" step="1"
                    value={batchSize}
                    onChange={(e) => setBatchSize(Number(e.target.value))}
                    className="w-full accent-violet-400"
                />
                <p className="text-xs text-slate-400 mt-2 text-center">
                    {batchSize === 1 && '⚠️ Batch Norm fails with batch size = 1 (no variance!)'}
                    {batchSize < 8 && batchSize > 1 && '⚠️ Batch Norm unstable with small batches'}
                    {batchSize >= 8 && '✅ Batch Norm works well with larger batches'}
                </p>
            </div>

            {/* Comparison */}
            <div className="grid lg:grid-cols-2 gap-8 w-full max-w-6xl">
                {/* Batch Norm */}
                <div className="bg-slate-800 p-6 rounded-xl border-2 border-cyan-500/50">
                    <h3 className="font-bold text-cyan-400 mb-4 text-center text-xl">Batch Normalization</h3>
                    <p className="text-sm text-slate-400 mb-4 text-center">
                        Normalize across <strong>batch dimension</strong> (↓)
                    </p>

                    <div className="space-y-3 max-h-[400px] overflow-y-auto">
                        {batchNormResult.map((feat, idx) => (
                            <div key={idx} className="bg-slate-900 p-4 rounded-lg border border-slate-600">
                                <div className="flex justify-between items-center mb-2">
                                    <span className="font-bold text-white">Feature {idx}</span>
                                    <div className="text-xs font-mono text-slate-400">
                                        μ: {feat.mean}, σ: {feat.std}
                                    </div>
                                </div>
                                <div className="flex gap-1">
                                    {feat.normalized.slice(0, 8).map((val, i) => (
                                        <div
                                            key={i}
                                            className="flex-1 h-8 rounded flex items-center justify-center text-xs font-mono"
                                            style={{
                                                backgroundColor: `hsl(${180 + parseFloat(val) * 30}, 70%, 50%)`,
                                                color: 'white'
                                            }}
                                            title={`Sample ${i}: ${val}`}
                                        >
                                            {val}
                                        </div>
                                    ))}
                                    {feat.normalized.length > 8 && (
                                        <div className="flex items-center text-xs text-slate-500">
                                            +{feat.normalized.length - 8}
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="mt-4 p-3 bg-cyan-900/30 rounded-lg border border-cyan-700">
                        <p className="text-xs text-cyan-300">
                            ⚠️ <strong>Problem</strong>: Depends on batch size. Fails with batch=1 or small batches.
                        </p>
                    </div>
                </div>

                {/* Layer Norm */}
                <div className="bg-slate-800 p-6 rounded-xl border-2 border-violet-500/50">
                    <h3 className="font-bold text-violet-400 mb-4 text-center text-xl">Layer Normalization</h3>
                    <p className="text-sm text-slate-400 mb-4 text-center">
                        Normalize across <strong>feature dimension</strong> (→)
                    </p>

                    <div className="space-y-3 max-h-[400px] overflow-y-auto">
                        {layerNormResult.slice(0, 8).map((sample, idx) => (
                            <div key={idx} className="bg-slate-900 p-4 rounded-lg border border-slate-600">
                                <div className="flex justify-between items-center mb-2">
                                    <span className="font-bold text-white">Sample {idx}</span>
                                    <div className="text-xs font-mono text-slate-400">
                                        μ: {sample.mean}, σ: {sample.std}
                                    </div>
                                </div>
                                <div className="flex gap-1">
                                    {sample.normalized.map((val, i) => (
                                        <div
                                            key={i}
                                            className="flex-1 h-8 rounded flex items-center justify-center text-xs font-mono"
                                            style={{
                                                backgroundColor: `hsl(${270 + parseFloat(val) * 30}, 70%, 50%)`,
                                                color: 'white'
                                            }}
                                            title={`Feature ${i}: ${val}`}
                                        >
                                            {val}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                        {layerNormResult.length > 8 && (
                            <div className="text-center text-xs text-slate-500">
                                +{layerNormResult.length - 8} more samples
                            </div>
                        )}
                    </div>

                    <div className="mt-4 p-3 bg-violet-900/30 rounded-lg border border-violet-700">
                        <p className="text-xs text-violet-300">
                            ✅ <strong>Advantage</strong>: Works with ANY batch size. Perfect for sequences (Transformers).
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
