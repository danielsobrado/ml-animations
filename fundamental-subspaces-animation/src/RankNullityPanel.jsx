import React, { useMemo, useState } from 'react';
import { ArrowRight, Minus, Plus } from 'lucide-react';

const CASES = [
  {
    id: 'full-column',
    label: 'Tall full column rank',
    m: 5,
    n: 3,
    r: 3,
    note: 'No kernel except 0. Every input direction is visible to A.',
  },
  {
    id: 'wide',
    label: 'Wide matrix',
    m: 3,
    n: 5,
    r: 3,
    note: 'Two input directions collapse to 0 because the domain has more dimensions than the image can carry.',
  },
  {
    id: 'rank-deficient',
    label: 'Rank deficient',
    m: 4,
    n: 4,
    r: 2,
    note: 'Only two independent output directions remain; two domain directions are erased.',
  },
];

function DimensionBar({ label, total, segments }) {
  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-sm">
        <span className="font-bold text-slate-900 dark:text-white">{label}</span>
        <span className="font-mono text-slate-500 dark:text-slate-400">{total} dims</span>
      </div>
      <div className="flex h-12 overflow-hidden rounded-lg border border-slate-200 bg-slate-100 dark:border-slate-700 dark:bg-slate-800">
        {segments.map((segment) => (
          <div
            key={segment.label}
            className={`flex min-w-[42px] items-center justify-center px-2 text-xs font-bold text-white ${segment.color}`}
            style={{ flexGrow: segment.value }}
            title={`${segment.label}: ${segment.value}`}
          >
            {segment.label} {segment.value}
          </div>
        ))}
      </div>
    </div>
  );
}

function VectorCloud({ count, active, color, label }) {
  const points = useMemo(() => (
    Array.from({ length: count }, (_, index) => {
      const angle = (index / Math.max(count, 1)) * Math.PI * 2;
      const radius = 30 + (index % 3) * 14;
      return {
        x: 50 + Math.cos(angle) * radius,
        y: 50 + Math.sin(angle) * radius * 0.68,
      };
    })
  ), [count]);

  return (
    <div className="relative h-48 rounded-lg border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-950">
      <p className="absolute left-3 top-3 text-xs font-bold uppercase tracking-wide text-slate-500 dark:text-slate-400">{label}</p>
      {points.map((point, index) => (
        <span
          key={`${label}-${index}`}
          className={`absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full transition-all ${color} ${
            active ? 'opacity-95 shadow-md' : 'opacity-35'
          }`}
          style={{ left: `${point.x}%`, top: `${point.y}%` }}
        />
      ))}
    </div>
  );
}

export default function RankNullityPanel() {
  const [caseIndex, setCaseIndex] = useState(1);
  const current = CASES[caseIndex];
  const nullity = current.n - current.r;
  const leftNullity = current.m - current.r;

  const nextCase = (direction) => {
    setCaseIndex((value) => (value + direction + CASES.length) % CASES.length);
  };

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-5 p-4 text-slate-900 dark:text-slate-100">
      <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-emerald-600 dark:text-emerald-300">Rank-nullity</p>
            <h2 className="mt-1 text-2xl font-bold">Kernel and Image as a Dimension Budget</h2>
            <p className="mt-2 max-w-3xl text-sm text-slate-600 dark:text-slate-400">
              A maps the row-space directions onto the image and sends kernel directions to zero. Rank-nullity is the accounting rule.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => nextCase(-1)}
              className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-slate-100 text-slate-700 transition hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
              title="Previous case"
            >
              <Minus size={18} />
            </button>
            <div className="min-w-[210px] rounded-lg bg-slate-100 px-4 py-2 text-center dark:bg-slate-800">
              <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Example</p>
              <p className="text-sm font-bold text-slate-950 dark:text-white">{current.label}</p>
            </div>
            <button
              onClick={() => nextCase(1)}
              className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-slate-100 text-slate-700 transition hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
              title="Next case"
            >
              <Plus size={18} />
            </button>
          </div>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_auto_1fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-700 dark:bg-slate-900">
          <h3 className="text-lg font-bold">Domain R^n</h3>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">Split into visible directions and erased directions.</p>
          <div className="mt-5 space-y-5">
            <DimensionBar
              label="n = rank(A) + nullity(A)"
              total={current.n}
              segments={[
                { label: 'rank', value: current.r, color: 'bg-sky-500' },
                { label: 'kernel', value: nullity, color: 'bg-amber-500' },
              ].filter((segment) => segment.value > 0)}
            />
            <div className="grid gap-3 md:grid-cols-2">
              <VectorCloud count={current.r} active color="bg-sky-500" label="row space" />
              <VectorCloud count={Math.max(nullity, 1)} active={nullity > 0} color="bg-amber-500" label="kernel" />
            </div>
          </div>
        </div>

        <div className="flex items-center justify-center">
          <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-center shadow-sm dark:border-emerald-800 dark:bg-emerald-950">
            <p className="font-mono text-xl font-bold text-emerald-900 dark:text-emerald-100">A</p>
            <ArrowRight className="mx-auto my-2 text-emerald-700 dark:text-emerald-300" size={28} />
            <p className="text-xs font-semibold text-emerald-700 dark:text-emerald-300">kernel goes to 0</p>
          </div>
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-700 dark:bg-slate-900">
          <h3 className="text-lg font-bold">Codomain R^m</h3>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">The image occupies rank(A) dimensions; the left null space is the unused orthogonal part.</p>
          <div className="mt-5 space-y-5">
            <DimensionBar
              label="m = rank(A) + left nullity"
              total={current.m}
              segments={[
                { label: 'image', value: current.r, color: 'bg-emerald-500' },
                { label: 'left null', value: leftNullity, color: 'bg-fuchsia-500' },
              ].filter((segment) => segment.value > 0)}
            />
            <div className="grid gap-3 md:grid-cols-2">
              <VectorCloud count={current.r} active color="bg-emerald-500" label="image" />
              <VectorCloud count={Math.max(leftNullity, 1)} active={leftNullity > 0} color="bg-fuchsia-500" label="left null" />
            </div>
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
          <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Matrix shape</p>
          <p className="mt-2 font-mono text-lg font-bold">{`A: R^${current.n} -> R^${current.m}`}</p>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
          <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Rank-nullity</p>
          <p className="mt-2 font-mono text-lg font-bold">{current.r} + {nullity} = {current.n}</p>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
          <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Interpretation</p>
          <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">{current.note}</p>
        </div>
      </div>
    </div>
  );
}
