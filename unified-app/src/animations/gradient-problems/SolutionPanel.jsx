import React, { useState } from 'react';
import { Plus, ArrowRight } from 'lucide-react';

export default function SolutionPanel() {
    const [useResidual, setUseResidual] = useState(false);
    const [layers, setLayers] = useState(20);
    const [weight, setWeight] = useState(0.5); // Small weights usually cause vanishing

    // Calculate signal strength
    const calculateSignal = () => {
        let signal = 1.0;
        const history = [signal];

        for (let i = 0; i < layers; i++) {
            if (useResidual) {
                // y = x + f(x)
                // gradient = 1 + f'(x)
                // Even if f'(x) is small (e.g. 0.5), total gradient is 1.5!
                // But here we simulate forward pass signal or backward gradient?
                // Let's simulate Gradient.
                // ResNet Gradient: dL/dx = dL/dy * (1 + f'(x))
                // If f'(x) is small (vanishing), the '1' saves us.
                signal = signal * (1 + weight);
            } else {
                // Standard: y = f(x)
                // Gradient: dL/dx = dL/dy * f'(x)
                signal = signal * weight;
            }
            history.push(signal);
        }
        return history;
    };

    const signalHistory = calculateSignal();
    const finalSignal = signalHistory[signalHistory.length - 1];

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-4">The Residual Fix (ResNet)</h2>
                <p className="text-lg text-slate-700 dark:text-slate-300 leading-relaxed">
                    How do we train networks with 100+ layers?
                    <br />
                    <strong>Residual Connections</strong> add the input to the output: <code className="bg-slate-800 px-2 py-1 rounded text-orange-300">y = x + f(x)</code>.
                    <br />
                    This creates a "Superhighway" for gradients to flow backwards without vanishing.
                </p>
            </div>

            <div className="grid md:grid-cols-2 gap-12 w-full max-w-6xl items-start">
                {/* Controls */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                    <h3 className="font-bold text-white mb-6">Architecture</h3>

                    <div className="flex items-center justify-between p-4 bg-slate-900 rounded-lg border border-slate-600 mb-8">
                        <span className="text-slate-700 dark:text-slate-300 font-bold">Use Residual Connections?</span>
                        <button
                            onClick={() => setUseResidual(!useResidual)}
                            className={`w-16 h-8 rounded-full transition-colors relative ${useResidual ? 'bg-orange-500' : 'bg-slate-700'
                                }`}
                        >
                            <div className={`absolute top-1 w-6 h-6 bg-white rounded-full transition-transform ${useResidual ? 'left-9' : 'left-1'
                                }`} />
                        </button>
                    </div>

                    <div className="space-y-6">
                        <div>
                            <div className="flex justify-between items-end mb-2">
                                <label className="text-sm text-slate-800 dark:text-slate-400">Network Depth</label>
                                <span className="font-mono font-bold text-orange-600 dark:text-orange-400">{layers}</span>
                            </div>
                            <input
                                type="range" min="5" max="50" step="1"
                                value={layers}
                                onChange={(e) => setLayers(parseInt(e.target.value))}
                                className="w-full accent-orange-400"
                            />
                        </div>

                        <div>
                            <div className="flex justify-between items-end mb-2">
                                <label className="text-sm text-slate-800 dark:text-slate-400">Layer Gradient (f'(x))</label>
                                <span className="font-mono font-bold text-orange-600 dark:text-orange-400">{weight.toFixed(2)}</span>
                            </div>
                            <input
                                type="range" min="-0.5" max="0.5" step="0.05"
                                value={weight}
                                onChange={(e) => setWeight(parseFloat(e.target.value))}
                                className="w-full accent-orange-400"
                            />
                            <p className="text-xs text-slate-700 dark:text-slate-500 mt-2">
                                Small gradients usually kill the signal.
                            </p>
                        </div>
                    </div>

                    <div className="mt-8">
                        <h4 className="font-bold text-white mb-4">Architecture Diagram</h4>
                        <div className="flex flex-col items-center gap-2 p-4 bg-slate-900 rounded-lg border border-slate-600">
                            {useResidual ? (
                                // ResNet Block Diagram
                                <div className="relative w-32 h-32 flex items-center justify-center">
                                    <div className="absolute left-0 top-1/2 w-full h-1 bg-orange-500/30 -z-10" /> {/* Skip Connection */}
                                    <div className="w-16 h-16 bg-slate-800 border-2 border-orange-500 rounded flex items-center justify-center text-orange-600 dark:text-orange-400 font-bold z-10">
                                        f(x)
                                    </div>
                                    <div className="absolute right-0 top-1/2 transform translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-slate-700 rounded-full flex items-center justify-center border border-slate-500 z-20">
                                        <Plus size={16} />
                                    </div>
                                    <div className="absolute -top-6 text-xs text-orange-600 dark:text-orange-400">x + f(x)</div>
                                </div>
                            ) : (
                                // Plain Block Diagram
                                <div className="relative w-32 h-32 flex items-center justify-center">
                                    <div className="w-16 h-16 bg-slate-800 border-2 border-slate-500 rounded flex items-center justify-center text-slate-800 dark:text-slate-400 font-bold">
                                        f(x)
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Visualization */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 min-h-[400px] flex flex-col">
                    <h3 className="font-bold text-white mb-6">Gradient Strength at Input</h3>

                    <div className="flex-1 flex items-end justify-center gap-1 bg-slate-900 rounded-lg p-4 border border-slate-600 overflow-hidden relative">
                        {/* Bar Chart of Signal History */}
                        {signalHistory.map((s, i) => {
                            const height = Math.min(100, Math.abs(s) * 10); // Scale for visibility
                            return (
                                <div
                                    key={i}
                                    className={`w-2 transition-all duration-300 ${useResidual ? 'bg-orange-500' : 'bg-red-500'}`}
                                    style={{ height: `${height}%` }}
                                    title={`Layer ${i}: ${s.toFixed(4)}`}
                                />
                            )
                        })}

                        {/* Overlay Text */}
                        <div className="absolute top-4 left-4 bg-slate-900/80 p-2 rounded backdrop-blur">
                            <div className="text-xs text-slate-800 dark:text-slate-400">Final Gradient:</div>
                            <div className={`text-2xl font-mono font-bold ${Math.abs(finalSignal) < 0.001 ? 'text-red-500' : 'text-green-400'
                                }`}>
                                {finalSignal.toExponential(2)}
                            </div>
                        </div>
                    </div>

                    <div className="mt-4 text-sm text-slate-800 dark:text-center">
                        {useResidual
                            ? "With Residuals, the gradient flows through the skip connection (the '1') even if f'(x) is small."
                            : "Without Residuals, multiplying by small numbers (0.5) repeatedly kills the signal."
                        }
                    </div>
                </div>
            </div>
        </div>
    );
}
