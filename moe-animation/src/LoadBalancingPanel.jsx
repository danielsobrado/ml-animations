import React from 'react';

export default function LoadBalancingPanel({ numExperts, loads }) {
    // loads is an array of numbers representing tokens processed
    const maxLoad = Math.max(...(loads || [1]));

    return (
        <div className="bg-slate-900 p-6 rounded-xl border border-slate-800">
            <h2 className="text-xl font-bold text-white mb-4">Expert Load</h2>
            <div className="space-y-2">
                {Array.from({ length: numExperts }).map((_, i) => {
                    const load = loads ? loads[i] : 0;
                    const percentage = (load / maxLoad) * 100 || 0;

                    return (
                        <div key={i} className="flex items-center gap-2">
                            <span className="text-xs font-mono w-6 text-slate-500">E{i}</span>
                            <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                                <div
                                    className={`h-full transition-all duration-500 ${percentage > 90 ? 'bg-red-500' : 'bg-neon-green'
                                        }`}
                                    style={{ width: `${percentage}%` }}
                                ></div>
                            </div>
                            <span className="text-xs text-slate-500 w-8 text-right">{load}</span>
                        </div>
                    );
                })}
            </div>
            <p className="text-xs text-slate-500 mt-4">
                *Red indicates potential bottleneck (Expert Collapse risk)
            </p>
        </div>
    );
}
