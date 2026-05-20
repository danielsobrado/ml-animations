import React, { useEffect, useMemo, useState } from 'react';
import { Pause, Play, RotateCcw, StepBack, StepForward } from 'lucide-react';

const SUBSPACES = [
  {
    id: 'row',
    title: 'Row Space',
    notation: 'Row(A) = C(A^T)',
    ambient: 'R^n',
    dimension: 'rank r',
    role: 'The directions in the input that A can measure.',
    relation: 'Null(A)^perp',
    side: 'domain',
    color: 'bg-sky-500',
    border: 'border-sky-300',
    text: 'text-sky-800',
  },
  {
    id: 'null',
    title: 'Null Space',
    notation: 'Null(A) = N(A)',
    ambient: 'R^n',
    dimension: 'n - r',
    role: 'All input directions erased by A: Ax = 0.',
    relation: 'Row(A)^perp',
    side: 'domain',
    color: 'bg-amber-500',
    border: 'border-amber-300',
    text: 'text-amber-800',
  },
  {
    id: 'column',
    title: 'Column Space',
    notation: 'Col(A) = C(A)',
    ambient: 'R^m',
    dimension: 'rank r',
    role: 'All outputs reachable as Ax.',
    relation: 'Null(A^T)^perp',
    side: 'codomain',
    color: 'bg-emerald-500',
    border: 'border-emerald-300',
    text: 'text-emerald-800',
  },
  {
    id: 'left-null',
    title: 'Left Null Space',
    notation: 'Null(A^T) = N(A^T)',
    ambient: 'R^m',
    dimension: 'm - r',
    role: 'Output-side tests y with A^T y = 0.',
    relation: 'Col(A)^perp',
    side: 'codomain',
    color: 'bg-fuchsia-500',
    border: 'border-fuchsia-300',
    text: 'text-fuchsia-800',
  },
];

const STEPS = [
  {
    id: 'map',
    label: 'Linear map',
    active: null,
    title: 'A maps inputs to outputs',
    detail: 'For an m x n matrix A, the domain is R^n and the codomain is R^m. The four fundamental subspaces split across those two worlds.',
  },
  {
    id: 'row',
    label: 'Row space',
    active: 'row',
    title: 'Row(A) carries the measurable input directions',
    detail: 'Row(A) is spanned by the rows of A. Every particular solution x_p can be chosen inside Row(A).',
  },
  {
    id: 'null',
    label: 'Null space',
    active: 'null',
    title: 'Null(A) is the solution space of Ax = 0',
    detail: 'Adding any null vector z to x_p does not change the output: A(x_p + z) = Ax_p.',
  },
  {
    id: 'column',
    label: 'Column space',
    active: 'column',
    title: 'Col(A) is the set of reachable right-hand sides',
    detail: 'The system Ax = b is consistent exactly when b is inside Col(A).',
  },
  {
    id: 'left-null',
    label: 'Left null',
    active: 'left-null',
    title: 'Null(A^T) gives compatibility conditions',
    detail: 'A vector y in the left null space is orthogonal to every column of A. Consistency requires y^T b = 0 for all such y.',
  },
  {
    id: 'dimensions',
    label: 'Dimensions',
    active: null,
    title: 'Rank appears twice; the leftover dimensions are nullities',
    detail: 'dim Row(A) = dim Col(A) = r, dim Null(A) = n - r, and dim Null(A^T) = m - r.',
  },
];

function SubspaceCard({ subspace, active }) {
  return (
    <div
      className={`rounded-lg border bg-white p-4 shadow-sm transition-all dark:bg-slate-900 ${
        active ? `${subspace.border} ring-2 ring-offset-2 ring-offset-white dark:ring-offset-slate-950` : 'border-slate-200 dark:border-slate-700'
      }`}
    >
      <div className="mb-3 flex items-center gap-3">
        <span className={`h-3 w-3 rounded-full ${subspace.color}`} />
        <h3 className="text-base font-bold text-slate-900 dark:text-white">{subspace.title}</h3>
      </div>
      <div className="space-y-2 text-sm">
        <p className="font-mono text-slate-700 dark:text-slate-300">{subspace.notation}</p>
        <p className="text-slate-600 dark:text-slate-400">{subspace.role}</p>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="rounded bg-slate-100 px-2 py-1 dark:bg-slate-800">
            <span className="block text-slate-500 dark:text-slate-400">Ambient</span>
            <span className="font-semibold text-slate-900 dark:text-white">{subspace.ambient}</span>
          </div>
          <div className="rounded bg-slate-100 px-2 py-1 dark:bg-slate-800">
            <span className="block text-slate-500 dark:text-slate-400">Dimension</span>
            <span className="font-semibold text-slate-900 dark:text-white">{subspace.dimension}</span>
          </div>
        </div>
        <p className={`text-xs font-semibold ${subspace.text}`}>{subspace.relation}</p>
      </div>
    </div>
  );
}

function Plane({ title, subtitle, side, activeId }) {
  const local = SUBSPACES.filter((space) => space.side === side);

  return (
    <div className="relative min-h-[260px] rounded-lg border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-900">
      <div className="mb-3">
        <h3 className="text-sm font-bold text-slate-900 dark:text-white">{title}</h3>
        <p className="text-xs text-slate-500 dark:text-slate-400">{subtitle}</p>
      </div>

      <div className="relative h-48 overflow-hidden rounded-lg bg-white dark:bg-slate-950">
        <div className="absolute left-1/2 top-4 h-40 w-px -translate-x-1/2 bg-slate-200 dark:bg-slate-700" />
        <div className="absolute left-8 right-8 top-1/2 h-px -translate-y-1/2 bg-slate-200 dark:bg-slate-700" />

        {local.map((space, index) => {
          const active = activeId === space.id;
          const rotation = index === 0 ? '-rotate-12' : 'rotate-[58deg]';
          const position = index === 0 ? 'left-[18%] top-[48%]' : 'left-[46%] top-[48%]';
          return (
            <div key={space.id}>
              <div
                className={`absolute ${position} h-2 w-32 origin-left rounded-full ${space.color} ${rotation} transition-all ${
                  active ? 'opacity-100 shadow-lg scale-110' : 'opacity-45'
                }`}
              />
              <div
                className={`absolute rounded-md border bg-white px-2 py-1 text-xs font-semibold shadow-sm transition-all dark:bg-slate-800 ${
                  index === 0 ? 'left-6 top-7' : 'right-5 bottom-6'
                } ${active ? `${space.border} text-slate-900 dark:text-white` : 'border-slate-200 text-slate-500 dark:border-slate-700 dark:text-slate-400'}`}
              >
                {space.title}
              </div>
            </div>
          );
        })}

        {side === 'domain' ? (
          <>
            <div className="absolute left-[24%] top-[34%] rounded bg-sky-100 px-2 py-1 text-xs font-mono text-sky-800">x_p</div>
            <div className="absolute bottom-[28%] right-[24%] rounded bg-amber-100 px-2 py-1 text-xs font-mono text-amber-800">z</div>
          </>
        ) : (
          <>
            <div className="absolute left-[26%] top-[32%] rounded bg-emerald-100 px-2 py-1 text-xs font-mono text-emerald-800">b</div>
            <div className="absolute bottom-[28%] right-[22%] rounded bg-fuchsia-100 px-2 py-1 text-xs font-mono text-fuchsia-800">y</div>
          </>
        )}
      </div>
    </div>
  );
}

export default function AnimationPanel() {
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const activeStep = STEPS[step];
  const activeId = activeStep.active;

  const equation = useMemo(() => {
    switch (activeStep.id) {
      case 'row':
        return 'Row(A) = Null(A)^perp';
      case 'null':
        return 'A z = 0, so A(x_p + z) = b';
      case 'column':
        return 'Ax = b is consistent iff b in Col(A)';
      case 'left-null':
        return 'y in Null(A^T) implies y^T b = 0';
      case 'dimensions':
        return 'r + (n - r) = n and r + (m - r) = m';
      case 'map':
      default:
        return 'A: R^n -> R^m';
    }
  }, [activeStep.id]);

  useEffect(() => {
    if (!isPlaying) return undefined;
    if (step >= STEPS.length - 1) {
      setIsPlaying(false);
      return undefined;
    }

    const timer = window.setTimeout(() => {
      setStep((current) => Math.min(current + 1, STEPS.length - 1));
    }, 2200);

    return () => window.clearTimeout(timer);
  }, [isPlaying, step]);

  const next = () => setStep((current) => Math.min(current + 1, STEPS.length - 1));
  const previous = () => setStep((current) => Math.max(current - 1, 0));
  const reset = () => {
    setIsPlaying(false);
    setStep(0);
  };

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-5 p-4 text-slate-900 dark:text-slate-100">
      <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-indigo-600 dark:text-indigo-300">Linear algebra</p>
            <h2 className="mt-1 text-2xl font-bold">The Four Fundamental Subspaces</h2>
            <p className="mt-2 max-w-3xl text-sm text-slate-600 dark:text-slate-400">
              Follow how one matrix A organizes input directions, erased directions, reachable outputs, and output-side constraints.
            </p>
          </div>

          <div className="rounded-lg bg-slate-100 px-4 py-3 text-center dark:bg-slate-800">
            <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Current relation</p>
            <p className="mt-1 font-mono text-lg font-bold text-slate-950 dark:text-white">{equation}</p>
          </div>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_auto_1fr]">
        <Plane title="Domain" subtitle="Input space R^n" side="domain" activeId={activeId} />
        <div className="flex min-w-[92px] items-center justify-center">
          <div className="rounded-lg border border-indigo-200 bg-indigo-50 px-4 py-3 text-center shadow-sm dark:border-indigo-800 dark:bg-indigo-950">
            <p className="font-mono text-xl font-bold text-indigo-900 dark:text-indigo-100">A</p>
            <div className="my-2 h-px w-14 bg-indigo-300 dark:bg-indigo-700" />
            <p className="text-xs font-semibold text-indigo-700 dark:text-indigo-300">maps to</p>
          </div>
        </div>
        <Plane title="Codomain" subtitle="Output space R^m" side="codomain" activeId={activeId} />
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <h3 className="text-lg font-bold">{activeStep.title}</h3>
            <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">{activeStep.detail}</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={previous}
              disabled={step === 0}
              className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-slate-100 text-slate-700 transition hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
              title="Previous step"
            >
              <StepBack size={18} />
            </button>
            <button
              onClick={() => setIsPlaying((value) => !value)}
              className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-indigo-600 text-white transition hover:bg-indigo-700"
              title={isPlaying ? 'Pause' : 'Play'}
            >
              {isPlaying ? <Pause size={18} /> : <Play size={18} />}
            </button>
            <button
              onClick={next}
              disabled={step === STEPS.length - 1}
              className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-slate-100 text-slate-700 transition hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
              title="Next step"
            >
              <StepForward size={18} />
            </button>
            <button
              onClick={reset}
              className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-slate-100 text-slate-700 transition hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
              title="Reset"
            >
              <RotateCcw size={18} />
            </button>
          </div>
        </div>

        <div className="grid gap-2 md:grid-cols-6">
          {STEPS.map((item, index) => (
            <button
              key={item.id}
              onClick={() => {
                setIsPlaying(false);
                setStep(index);
              }}
              className={`rounded-lg px-3 py-2 text-left text-xs font-semibold transition ${
                step === index
                  ? 'bg-indigo-600 text-white shadow'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700'
              }`}
            >
              {index + 1}. {item.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-4">
        {SUBSPACES.map((subspace) => (
          <SubspaceCard key={subspace.id} subspace={subspace} active={activeId === subspace.id} />
        ))}
      </div>
    </div>
  );
}
