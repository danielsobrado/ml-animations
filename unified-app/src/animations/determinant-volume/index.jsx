import React, { useMemo, useState } from 'react';
import { Box, FlaskConical } from 'lucide-react';

const transforms = [
  { id: 'scale', name: 'Positive scale', a: [2, 0], b: [0.4, 1.4] },
  { id: 'flip', name: 'Orientation flip', a: [1.4, 0.2], b: [0.5, -1.2] },
  { id: 'collapse', name: 'Collapse', a: [1.8, 0.8], b: [0.9, 0.4] },
];

function det(a, b) {
  return a[0] * b[1] - a[1] * b[0];
}

function point([x, y]) {
  return [210 + x * 72, 230 - y * 72];
}

export default function DeterminantVolumeAnimation() {
  const [id, setId] = useState('scale');
  const active = transforms.find((item) => item.id === id) || transforms[0];
  const area = useMemo(() => det(active.a, active.b), [active]);
  const points = [[0, 0], active.a, [active.a[0] + active.b[0], active.a[1] + active.b[1]], active.b].map(point);

  return (
    <div className="min-h-full bg-stone-950 p-4 text-stone-100">
      <div className="mx-auto grid max-w-7xl gap-4 lg:grid-cols-[1.1fr_0.9fr]">
        <section className="rounded-lg border border-stone-800 bg-stone-900 p-5">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h1 className="text-2xl font-bold">Determinant as Volume</h1>
              <p className="text-sm text-stone-400">A determinant is signed area or volume scaling.</p>
            </div>
            <div className="flex flex-wrap gap-2">
              {transforms.map((item) => (
                <button
                  key={item.id}
                  onClick={() => setId(item.id)}
                  className={`rounded-md px-3 py-2 text-sm font-semibold ${
                    id === item.id ? 'bg-orange-300 text-stone-950' : 'bg-stone-800 text-stone-300 hover:bg-stone-700'
                  }`}
                >
                  {item.name}
                </button>
              ))}
            </div>
          </div>
          <svg viewBox="0 0 420 360" className="h-[390px] w-full rounded-md bg-stone-950">
            {Array.from({ length: 9 }, (_, i) => 66 + i * 36).map((n) => (
              <g key={n}>
                <line x1={n} y1="34" x2={n} y2="326" stroke="#292524" />
                <line x1="34" y1={n} x2="386" y2={n} stroke="#292524" />
              </g>
            ))}
            <polygon points={points.map((p) => p.join(',')).join(' ')} fill="#fb923c40" stroke="#fdba74" strokeWidth="4" />
            <line x1={points[0][0]} y1={points[0][1]} x2={points[1][0]} y2={points[1][1]} stroke="#38bdf8" strokeWidth="4" />
            <line x1={points[0][0]} y1={points[0][1]} x2={points[3][0]} y2={points[3][1]} stroke="#f472b6" strokeWidth="4" />
            <text x="42" y="42" fill="#fed7aa" fontSize="16">area scale = |det A| = {Math.abs(area).toFixed(2)}</text>
            <text x="42" y="66" fill={area < 0 ? '#fecdd3' : '#bbf7d0'} fontSize="16">sign = {area < 0 ? 'orientation reversed' : area === 0 ? 'collapsed' : 'orientation preserved'}</text>
          </svg>
        </section>

        <aside className="grid gap-4">
          <div className="rounded-lg border border-orange-300/30 bg-orange-300/10 p-5">
            <div className="mb-3 flex items-center gap-2 text-orange-100">
              <Box size={20} />
              <h2 className="text-lg font-bold">Meaning</h2>
            </div>
            <p className="text-sm text-stone-300">The columns of A transform the unit square into a parallelogram. Its signed area is det(A).</p>
            <p className="mt-3 rounded-md bg-stone-950 p-3 font-mono text-sm text-orange-100">det(A) = {area.toFixed(2)}</p>
          </div>
          <div className="rounded-lg border border-stone-800 bg-stone-900 p-5">
            <div className="mb-3 flex items-center gap-2">
              <FlaskConical size={20} className="text-sky-300" />
              <h2 className="text-lg font-bold">Practice Lab</h2>
            </div>
            <p className="text-sm text-stone-300">If det(A) = 0, the unit square collapses to a line or point, so A cannot be invertible.</p>
          </div>
        </aside>
      </div>
    </div>
  );
}
