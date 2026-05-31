import React, { useMemo, useState } from 'react';
import { FlaskConical, Layers } from 'lucide-react';

const singularValues = [9, 5, 2.5, 1, 0.4];

export default function LowRankApproximationAnimation() {
  const [rank, setRank] = useState(2);
  const kept = singularValues.slice(0, rank);
  const dropped = singularValues.slice(rank);
  const energy = useMemo(() => {
    const total = singularValues.reduce((sum, value) => sum + value * value, 0);
    const retained = kept.reduce((sum, value) => sum + value * value, 0);
    return Math.round((retained / total) * 100);
  }, [kept]);

  return (
    <div className="min-h-full bg-gray-950 p-4 text-gray-100">
      <div className="mx-auto max-w-7xl">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-2xl font-bold">Low-Rank Approximation</h1>
            <p className="text-sm text-gray-400">Keep the strongest singular components and discard the rest.</p>
          </div>
          <label className="flex items-center gap-3 rounded-md bg-gray-900 px-4 py-2 text-sm">
            rank k
            <input min="1" max="5" type="range" value={rank} onChange={(event) => setRank(Number(event.target.value))} />
            <span className="font-mono text-cyan-200">{rank}</span>
          </label>
        </div>

        <div className="grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
          <section className="rounded-lg border border-gray-800 bg-gray-900 p-5">
            <div className="mb-5 grid gap-3 md:grid-cols-3">
              <div className="rounded-md border border-gray-700 bg-gray-950 p-4">
                <div className="mb-3 text-sm font-semibold text-gray-300">Original A</div>
                <div className="grid grid-cols-5 gap-1">
                  {Array.from({ length: 25 }, (_, i) => (
                    <div key={i} className="aspect-square rounded-sm bg-cyan-300" style={{ opacity: 0.25 + ((i * 17) % 60) / 100 }} />
                  ))}
                </div>
              </div>
              <div className="rounded-md border border-gray-700 bg-gray-950 p-4">
                <div className="mb-3 text-sm font-semibold text-gray-300">Rank-{rank} A_k</div>
                <div className="grid grid-cols-5 gap-1">
                  {Array.from({ length: 25 }, (_, i) => (
                    <div key={i} className="aspect-square rounded-sm bg-emerald-300" style={{ opacity: 0.2 + (((i * rank + 11) % 55) / 100) }} />
                  ))}
                </div>
              </div>
              <div className="rounded-md border border-gray-700 bg-gray-950 p-4">
                <div className="mb-3 text-sm font-semibold text-gray-300">Residual A - A_k</div>
                <div className="grid grid-cols-5 gap-1">
                  {Array.from({ length: 25 }, (_, i) => (
                    <div
                      key={i}
                      className="aspect-square rounded-sm bg-rose-300"
                      style={{ opacity: rank === singularValues.length ? 0 : Math.max(0.08, (6 - rank) / 8 - ((i % 3) * 0.04)) }}
                    />
                  ))}
                </div>
              </div>
            </div>
            <div className="space-y-3">
              {singularValues.map((value, index) => (
                <div key={value} className="grid grid-cols-[80px_1fr_80px] items-center gap-3 text-sm">
                  <span className="font-mono text-gray-400">sigma {index + 1}</span>
                  <div className="h-4 overflow-hidden rounded-full bg-gray-800">
                    <div className={`h-full rounded-full ${index < rank ? 'bg-cyan-300' : 'bg-gray-600'}`} style={{ width: `${value * 10}%` }} />
                  </div>
                  <span className={index < rank ? 'text-cyan-100' : 'text-gray-500'}>{index < rank ? 'kept' : 'dropped'}</span>
                </div>
              ))}
            </div>
          </section>

          <aside className="grid gap-4">
            <div className="rounded-lg border border-cyan-300/30 bg-cyan-300/10 p-5">
              <div className="mb-3 flex items-center gap-2 text-cyan-100">
                <Layers size={20} />
                <h2 className="text-lg font-bold">Eckart-Young</h2>
              </div>
              <p className="rounded-md bg-gray-950 p-3 font-mono text-sm text-cyan-100">A_k = sum from i=1 to k of sigma_i u_i v_i^T</p>
              <p className="mt-3 text-sm text-gray-300">Among all rank-{rank} matrices, truncated SVD gives the closest approximation in Frobenius norm and spectral norm.</p>
              <p className="mt-3 text-sm text-emerald-200">Retained energy: {energy}%</p>
            </div>
            <div className="rounded-lg border border-gray-800 bg-gray-900 p-5">
              <div className="mb-3 flex items-center gap-2">
                <FlaskConical size={20} className="text-rose-300" />
                <h2 className="text-lg font-bold">Practice Lab</h2>
              </div>
              <p className="text-sm text-gray-300">Dropped singular values are the reconstruction error budget. Raising k keeps more signal and less compression.</p>
              <p className="mt-3 rounded-md bg-gray-950 p-3 text-sm text-gray-200">Dropped: {dropped.length ? dropped.join(', ') : 'none'}</p>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}
