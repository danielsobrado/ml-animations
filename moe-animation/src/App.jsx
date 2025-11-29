import React, { useState } from 'react';
import RoutingPanel from './RoutingPanel';
import GatingPanel from './GatingPanel';
import LoadBalancingPanel from './LoadBalancingPanel';

export default function App() {
    const [numExperts, setNumExperts] = useState(8);
    const [topK, setTopK] = useState(2);
    const [batchSize, setBatchSize] = useState(10);
    const [expertLoads, setExpertLoads] = useState(Array(8).fill(0));

    const handleGenerate = () => {
        // Simulate load updates
        const newLoads = [...expertLoads];
        for (let i = 0; i < batchSize; i++) {
            // Randomly assign to topK experts
            for (let k = 0; k < topK; k++) {
                const idx = Math.floor(Math.random() * numExperts);
                newLoads[idx]++;
            }
        }
        setExpertLoads(newLoads);
    };

    // Reset loads when expert count changes
    React.useEffect(() => {
        setExpertLoads(Array(numExperts).fill(0));
    }, [numExperts]);

    return (
        <div className="min-h-screen bg-slate-950 p-8 font-sans text-slate-200">
            <header className="max-w-7xl mx-auto mb-8 flex justify-between items-end">
                <div>
                    <h1 className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-neon-blue to-neon-purple mb-2">
                        Mixture of Experts
                    </h1>
                    <p className="text-slate-400 text-lg">
                        Visualizing Sparse Gating & Expert Routing
                    </p>
                </div>
                <div className="flex gap-4">
                    <div className="bg-slate-900 p-4 rounded-lg border border-slate-800">
                        <label className="block text-xs font-bold text-slate-500 uppercase mb-1">Experts</label>
                        <div className="flex gap-2">
                            {[4, 8, 16].map(n => (
                                <button
                                    key={n}
                                    onClick={() => setNumExperts(n)}
                                    className={`px-3 py-1 rounded text-sm font-bold transition-colors ${numExperts === n ? 'bg-neon-blue text-black' : 'bg-slate-800 hover:bg-slate-700'
                                        }`}
                                >
                                    {n}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="bg-slate-900 p-4 rounded-lg border border-slate-800">
                        <label className="block text-xs font-bold text-slate-500 uppercase mb-1">Top-K</label>
                        <div className="flex gap-2">
                            {[1, 2].map(k => (
                                <button
                                    key={k}
                                    onClick={() => setTopK(k)}
                                    className={`px-3 py-1 rounded text-sm font-bold transition-colors ${topK === k ? 'bg-neon-pink text-black' : 'bg-slate-800 hover:bg-slate-700'
                                        }`}
                                >
                                    {k}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Main Visualization */}
                <div className="lg:col-span-2 aspect-video">
                    <RoutingPanel
                        numExperts={numExperts}
                        topK={topK}
                        batchSize={batchSize}
                        onGenerate={handleGenerate}
                    />
                </div>

                {/* Stats / Info Panel */}
                <div className="space-y-6">
                    <div className="bg-slate-900 p-6 rounded-xl border border-slate-800">
                        <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                            <span className="w-2 h-8 bg-neon-green rounded-full"></span>
                            How it Works
                        </h2>
                        <div className="space-y-4 text-sm text-slate-400">
                            <p>
                                <strong className="text-white">1. The Router:</strong> A learned gating network that predicts which experts are best suited for each token.
                            </p>
                            <p>
                                <strong className="text-white">2. Sparse Activation:</strong> Instead of using all parameters (Dense), we only use the Top-{topK} experts. This saves massive compute.
                            </p>
                            <p>
                                <strong className="text-white">3. Load Balancing:</strong> Ideally, tokens are spread evenly. If one expert gets too many, it becomes a bottleneck (Expert Collapse).
                            </p>
                        </div>

                        <div className="mt-6">
                            <GatingPanel numExperts={numExperts} topK={topK} />
                        </div>
                    </div>

                    <LoadBalancingPanel numExperts={numExperts} loads={expertLoads} />
                </div>
            </main>
        </div>
    );
}
