import React, { useState } from 'react';
import { ArrowRightLeft, CheckCircle2, FlaskConical } from 'lucide-react';

const modes = [
  { id: 'full-column', name: 'Full column rank', formula: 'A+ = (A^T A)^-1 A^T', outcome: 'Least-squares solution is unique.' },
  { id: 'full-row', name: 'Full row rank', formula: 'A+ = A^T (A A^T)^-1', outcome: 'Chooses the minimum-norm exact solution.' },
  { id: 'rank-deficient', name: 'Rank deficient', formula: 'A+ = V Sigma+ U^T', outcome: 'Invert only nonzero singular directions.' },
];

const singulars = [6, 3.5, 1.3, 0];

export default function PseudoinverseAnimation() {
  const [modeId, setModeId] = useState('rank-deficient');
  const [checked, setChecked] = useState(false);
  const mode = modes.find((item) => item.id === modeId) || modes[0];

  return (
    <div className="min-h-full bg-zinc-950 p-4 text-zinc-100">
      <div className="mx-auto max-w-7xl">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-2xl font-bold">Pseudoinverse</h1>
            <p className="text-sm text-zinc-400">The inverse that works for rectangular, singular, and inconsistent systems.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            {modes.map((item) => (
              <button
                key={item.id}
                onClick={() => setModeId(item.id)}
                className={`rounded-md px-3 py-2 text-sm font-semibold ${
                  modeId === item.id ? 'bg-amber-300 text-zinc-950' : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'
                }`}
              >
                {item.name}
              </button>
            ))}
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-[1fr_0.9fr]">
          <section className="rounded-lg border border-zinc-800 bg-zinc-900 p-5">
            <div className="mb-5 grid gap-3 md:grid-cols-5">
              {['b', 'U^T b', 'Sigma+', 'V', 'x+'].map((label, index) => (
                <div key={label} className="flex items-center gap-3">
                  <div className="flex min-h-28 flex-1 items-center justify-center rounded-md border border-zinc-700 bg-zinc-950 p-3 text-center font-mono text-lg text-amber-100">
                    {label}
                  </div>
                  {index < 4 && <ArrowRightLeft className="hidden shrink-0 text-zinc-500 md:block" size={20} />}
                </div>
              ))}
            </div>

            <div className="rounded-md bg-zinc-950 p-4">
              <div className="mb-3 text-sm font-semibold text-zinc-300">Singular directions</div>
              <div className="space-y-3">
                {singulars.map((value, index) => (
                  <div key={index} className="grid grid-cols-[72px_1fr_90px] items-center gap-3 text-sm">
                    <span className="font-mono text-zinc-400">sigma {index + 1}</span>
                    <div className="h-4 overflow-hidden rounded-full bg-zinc-800">
                      <div
                        className={`h-full rounded-full ${value === 0 ? 'bg-rose-400' : 'bg-amber-300'}`}
                        style={{ width: `${Math.max(8, value * 14)}%` }}
                      />
                    </div>
                    <span className={value === 0 ? 'text-rose-200' : 'text-amber-100'}>
                      {value === 0 ? 'skip' : `1/${value}`}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <aside className="grid gap-4">
            <div className="rounded-lg border border-amber-300/30 bg-amber-300/10 p-5">
              <h2 className="mb-3 text-lg font-bold text-amber-100">Active Rule</h2>
              <p className="rounded-md bg-zinc-950 p-3 font-mono text-sm text-amber-100">{mode.formula}</p>
              <p className="mt-3 text-sm text-zinc-300">{mode.outcome}</p>
            </div>
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-5">
              <div className="mb-3 flex items-center gap-2">
                <FlaskConical size={20} className="text-sky-300" />
                <h2 className="text-lg font-bold">Practice Lab</h2>
              </div>
              <p className="text-sm text-zinc-300">A has a zero singular value. Should its reciprocal appear in Sigma+?</p>
              <button
                onClick={() => setChecked(true)}
                className="mt-4 flex items-center gap-2 rounded-md bg-sky-300 px-4 py-2 text-sm font-semibold text-zinc-950"
              >
                <CheckCircle2 size={16} /> Check
              </button>
              {checked && <p className="mt-3 text-sm text-emerald-200">No. Zero singular values stay zero so the solution avoids impossible inverse directions.</p>}
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}
