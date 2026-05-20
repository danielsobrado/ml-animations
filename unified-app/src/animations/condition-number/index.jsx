import React, { useMemo, useState } from 'react';
import { Activity, FlaskConical } from 'lucide-react';

const presets = [
  { id: 'stable', name: 'Well conditioned', major: 120, minor: 82, angle: -18 },
  { id: 'sensitive', name: 'Sensitive', major: 154, minor: 34, angle: -18 },
  { id: 'near-singular', name: 'Near singular', major: 172, minor: 12, angle: -18 },
];

export default function ConditionNumberAnimation() {
  const [presetId, setPresetId] = useState('sensitive');
  const active = presets.find((item) => item.id === presetId) || presets[0];
  const kappa = useMemo(() => (active.major / active.minor).toFixed(1), [active]);

  return (
    <div className="min-h-full bg-neutral-950 p-4 text-neutral-100">
      <div className="mx-auto max-w-7xl">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-2xl font-bold">Condition Number</h1>
            <p className="text-sm text-neutral-400">How much a matrix can amplify input or data error.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            {presets.map((item) => (
              <button
                key={item.id}
                onClick={() => setPresetId(item.id)}
                className={`rounded-md px-3 py-2 text-sm font-semibold ${
                  presetId === item.id ? 'bg-lime-300 text-neutral-950' : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
                }`}
              >
                {item.name}
              </button>
            ))}
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-[1fr_0.9fr]">
          <section className="rounded-lg border border-neutral-800 bg-neutral-900 p-5">
            <svg viewBox="0 0 620 360" className="h-[390px] w-full rounded-md bg-neutral-950">
              <line x1="60" y1="180" x2="270" y2="180" stroke="#525252" />
              <line x1="165" y1="285" x2="165" y2="75" stroke="#525252" />
              <circle cx="165" cy="180" r="82" fill="#84cc1630" stroke="#bef264" strokeWidth="3" />
              <text x="112" y="48" fill="#d9f99d" fontSize="16">unit circle</text>
              <path d="M296 180 L338 180" stroke="#a3a3a3" strokeWidth="3" />
              <path d="M330 170 L342 180 L330 190" fill="none" stroke="#a3a3a3" strokeWidth="3" />
              <line x1="360" y1="180" x2="570" y2="180" stroke="#525252" />
              <line x1="465" y1="285" x2="465" y2="75" stroke="#525252" />
              <ellipse
                cx="465"
                cy="180"
                rx={active.major}
                ry={active.minor}
                transform={`rotate(${active.angle} 465 180)`}
                fill="#22d3ee30"
                stroke="#67e8f9"
                strokeWidth="3"
              />
              <text x="407" y="48" fill="#a5f3fc" fontSize="16">output ellipse</text>
              <text x="394" y="316" fill="#d4d4d4" fontSize="15">stretch spread = {kappa}x</text>
            </svg>
          </section>

          <aside className="grid gap-4">
            <div className="rounded-lg border border-lime-300/30 bg-lime-300/10 p-5">
              <div className="mb-3 flex items-center gap-2 text-lime-100">
                <Activity size={20} />
                <h2 className="text-lg font-bold">Rule</h2>
              </div>
              <p className="rounded-md bg-neutral-950 p-3 font-mono text-sm text-lime-100">kappa(A) = sigma_max / sigma_min = {kappa}</p>
              <p className="mt-3 text-sm text-neutral-300">Large kappa means a tiny perturbation can become a large solution error, especially along the smallest singular direction.</p>
            </div>
            <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-5">
              <div className="mb-3 flex items-center gap-2">
                <FlaskConical size={20} className="text-cyan-300" />
                <h2 className="text-lg font-bold">Practice Lab</h2>
              </div>
              <p className="text-sm text-neutral-300">When sigma_min approaches zero, kappa approaches infinity. The matrix is nearly singular and inverse calculations become fragile.</p>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}
