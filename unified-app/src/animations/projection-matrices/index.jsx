import React, { useMemo, useState } from 'react';
import { CheckCircle2, FlaskConical, Target } from 'lucide-react';

const vectors = [
  { id: 'x1', name: 'x high', point: [280, 76] },
  { id: 'x2', name: 'x low', point: [310, 220] },
  { id: 'x3', name: 'already projected', point: [255.5, 185] },
];

function project([x, y]) {
  const x0 = 80;
  const y0 = 250;
  const dx = 270;
  const dy = -100;
  const t = ((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy);
  return [x0 + t * dx, y0 + t * dy];
}

export default function ProjectionMatricesAnimation() {
  const [id, setId] = useState('x1');
  const [revealed, setRevealed] = useState(false);
  const active = vectors.find((item) => item.id === id) || vectors[0];
  const p = useMemo(() => project(active.point), [active]);

  return (
    <div className="min-h-full bg-slate-950 p-4 text-slate-100">
      <div className="mx-auto grid max-w-7xl gap-4 lg:grid-cols-[1.1fr_0.9fr]">
        <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h1 className="text-2xl font-bold">Projection Matrices</h1>
              <p className="text-sm text-slate-400">P sends every vector to a chosen subspace, and applying P again changes nothing.</p>
            </div>
            <div className="flex flex-wrap gap-2">
              {vectors.map((item) => (
                <button
                  key={item.id}
                  onClick={() => setId(item.id)}
                  className={`rounded-md px-3 py-2 text-sm font-semibold ${
                    id === item.id ? 'bg-teal-300 text-slate-950' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  {item.name}
                </button>
              ))}
            </div>
          </div>
          <svg viewBox="0 0 420 310" className="h-[380px] w-full rounded-md bg-slate-950">
            <line x1="60" y1="257" x2="365" y2="144" stroke="#134e4a" strokeWidth="18" strokeLinecap="round" />
            <line x1="60" y1="257" x2="365" y2="144" stroke="#5eead4" strokeWidth="3" />
            <line x1="65" y1="260" x2={active.point[0]} y2={active.point[1]} stroke="#94a3b8" strokeWidth="2" />
            <line x1={active.point[0]} y1={active.point[1]} x2={p[0]} y2={p[1]} stroke="#fb7185" strokeDasharray="7 6" strokeWidth="4" />
            <line x1="65" y1="260" x2={p[0]} y2={p[1]} stroke="#5eead4" strokeWidth="5" />
            <circle cx={active.point[0]} cy={active.point[1]} r="8" fill="#fb7185" />
            <circle cx={p[0]} cy={p[1]} r="8" fill="#5eead4" />
            <text x={active.point[0] + 10} y={active.point[1] - 8} fill="#fecdd3" fontSize="16">x</text>
            <text x={p[0] + 10} y={p[1] + 22} fill="#ccfbf1" fontSize="16">Px</text>
            <text x="270" y="122" fill="#99f6e4" fontSize="15">target subspace</text>
          </svg>
        </section>

        <aside className="grid gap-4">
          <div className="rounded-lg border border-teal-300/30 bg-teal-300/10 p-5">
            <div className="mb-3 flex items-center gap-2 text-teal-100">
              <Target size={20} />
              <h2 className="text-lg font-bold">Projection Test</h2>
            </div>
            <p className="rounded-md bg-slate-950 p-3 font-mono text-sm text-teal-100">P^2 = P</p>
            <p className="mt-3 text-sm text-slate-300">Once x lands on the subspace, projecting again keeps it there. For orthogonal projection, x - Px is perpendicular to the subspace.</p>
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
            <div className="mb-3 flex items-center gap-2">
              <FlaskConical size={20} className="text-rose-300" />
              <h2 className="text-lg font-bold">Practice Lab</h2>
            </div>
            <p className="text-sm text-slate-300">If P is a projection matrix, what is P(Px)?</p>
            <button onClick={() => setRevealed(true)} className="mt-4 rounded-md bg-slate-100 px-4 py-2 text-sm font-semibold text-slate-950">Reveal</button>
            {revealed && <p className="mt-3 flex items-start gap-2 text-sm text-emerald-200"><CheckCircle2 size={16} className="mt-0.5 shrink-0" /> P(Px) = Px. That is exactly idempotence.</p>}
          </div>
        </aside>
      </div>
    </div>
  );
}
