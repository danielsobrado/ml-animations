import React, { useMemo, useState } from 'react';
import { BarChart3, SlidersHorizontal, TrendingUp } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const ACTIONS = [
  { id: 'left', label: 'Left', preference: 0.4 },
  { id: 'right', label: 'Right', preference: 0.1 },
  { id: 'jump', label: 'Jump', preference: -0.2 },
];

function softmax(items) {
  const max = Math.max(...items);
  const exps = items.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

export default function PolicyGradientsAnimation() {
  const [sampledAction, setSampledAction] = useState('right');
  const [returnValue, setReturnValue] = useState(7);
  const [baseline, setBaseline] = useState(3);
  const [learningRate, setLearningRate] = useState(0.35);
  const advantage = returnValue - baseline;

  const update = useMemo(() => {
    const before = softmax(ACTIONS.map((action) => action.preference));
    const nextPrefs = ACTIONS.map((action) => (
      action.preference + (action.id === sampledAction ? learningRate * advantage : 0)
    ));
    const after = softmax(nextPrefs);
    return ACTIONS.map((action, index) => ({
      ...action,
      before: before[index],
      after: after[index],
      delta: after[index] - before[index],
    }));
  }, [sampledAction, learningRate, advantage]);

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
      <section className="grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-sky-700">
            <SlidersHorizontal size={16} />
            Sampled trajectory signal
          </div>

          <div className="grid gap-2">
            {ACTIONS.map((action) => (
              <button
                key={action.id}
                type="button"
                onClick={() => setSampledAction(action.id)}
                className={`rounded-xl border px-4 py-3 text-left transition ${
                  sampledAction === action.id
                    ? 'border-sky-500 bg-sky-50 text-sky-900'
                    : 'border-slate-200 bg-white text-slate-700 hover:border-slate-300'
                }`}
              >
                <div className="font-semibold">{action.label}</div>
                <div className="text-sm">Sampled action</div>
              </button>
            ))}
          </div>

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="pg-return">
            Return G: {returnValue}
          </label>
          <input
            id="pg-return"
            type="range"
            min="-8"
            max="12"
            step="1"
            value={returnValue}
            onChange={(event) => setReturnValue(Number(event.target.value))}
            className="mt-2 w-full accent-sky-500"
          />

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="pg-baseline">
            Baseline b: {baseline}
          </label>
          <input
            id="pg-baseline"
            type="range"
            min="-4"
            max="10"
            step="1"
            value={baseline}
            onChange={(event) => setBaseline(Number(event.target.value))}
            className="mt-2 w-full accent-sky-500"
          />

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="pg-lr">
            Learning rate: {learningRate.toFixed(2)}
          </label>
          <input
            id="pg-lr"
            type="range"
            min="0.05"
            max="0.8"
            step="0.05"
            value={learningRate}
            onChange={(event) => setLearningRate(Number(event.target.value))}
            className="mt-2 w-full accent-sky-500"
          />
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-sky-700">
            <BarChart3 size={16} />
            Policy probabilities
          </div>
          <div className="space-y-4">
            {update.map((action) => (
              <div key={action.id} className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                <div className="flex items-center justify-between gap-3 text-sm font-semibold text-slate-900">
                  <span>{action.label}</span>
                  <span className={action.delta >= 0 ? 'text-emerald-700' : 'text-rose-700'}>
                    {action.delta >= 0 ? '+' : ''}{(action.delta * 100).toFixed(1)} pp
                  </span>
                </div>
                <div className="mt-3 grid gap-2">
                  <div className="h-2 rounded-full bg-white">
                    <div className="h-2 rounded-full bg-slate-400" style={{ width: `${action.before * 100}%` }} />
                  </div>
                  <div className="h-2 rounded-full bg-white">
                    <div className="h-2 rounded-full bg-sky-500" style={{ width: `${action.after * 100}%` }} />
                  </div>
                </div>
                <div className="mt-2 text-xs text-slate-600">
                  before {(action.before * 100).toFixed(1)}%, after {(action.after * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>

          <div className="mt-5 rounded-xl bg-slate-900 p-4 text-white">
            <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-sky-200">
              <TrendingUp size={14} />
              Advantage signal
            </div>
            <div className="mt-1 text-3xl font-bold">{advantage}</div>
            <p className="mt-2 text-sm text-slate-300">
              Toy update: positive advantage increases the sampled action preference before softmax; negative
              advantage lowers it. Other probabilities move because the distribution is renormalized.
            </p>
          </div>
        </div>
      </section>

      <AssessmentPanel lessonId="policy-gradients" title="Policy gradients check" />
    </div>
  );
}
