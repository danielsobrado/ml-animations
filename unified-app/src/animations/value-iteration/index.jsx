import React, { useMemo, useState } from 'react';
import { Calculator, RotateCcw, StepForward } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const STATES = ['Start', 'Bridge', 'Trap', 'Goal'];
const REWARDS = {
  Start: 0,
  Bridge: 1,
  Trap: -6,
  Goal: 10,
};
const ACTIONS = {
  Start: {
    risk: [{ to: 'Goal', p: 0.35 }, { to: 'Trap', p: 0.65 }],
    safe: [{ to: 'Bridge', p: 1 }],
  },
  Bridge: {
    forward: [{ to: 'Goal', p: 0.75 }, { to: 'Trap', p: 0.25 }],
    reset: [{ to: 'Start', p: 1 }],
  },
  Trap: {
    recover: [{ to: 'Bridge', p: 0.55 }, { to: 'Trap', p: 0.45 }],
  },
  Goal: {
    stay: [{ to: 'Goal', p: 1 }],
  },
};

function backup(state, values, discount) {
  const actionValues = Object.entries(ACTIONS[state]).map(([actionId, transitions]) => {
    const expected = transitions.reduce((sum, transition) => (
      sum + transition.p * (REWARDS[transition.to] + discount * values[transition.to])
    ), 0);
    return { actionId, expected };
  });
  return actionValues.reduce((best, candidate) => (
    candidate.expected > best.expected ? candidate : best
  ), actionValues[0]);
}

function runSweeps(count, discount) {
  let values = Object.fromEntries(STATES.map((state) => [state, 0]));
  const history = [values];

  for (let sweep = 0; sweep < count; sweep += 1) {
    const next = {};
    for (const state of STATES) {
      next[state] = backup(state, values, discount).expected;
    }
    values = next;
    history.push(values);
  }

  return history;
}

export default function ValueIterationAnimation() {
  const [sweeps, setSweeps] = useState(3);
  const [discount, setDiscount] = useState(0.8);
  const history = useMemo(() => runSweeps(sweeps, discount), [sweeps, discount]);
  const values = history[history.length - 1];
  const previous = history[Math.max(0, history.length - 2)];

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
      <section className="grid gap-4 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-indigo-700">
            <Calculator size={16} />
            Bellman planning controls
          </div>

          <label className="block text-sm font-semibold text-slate-700" htmlFor="vi-sweeps">
            Sweeps: {sweeps}
          </label>
          <input
            id="vi-sweeps"
            type="range"
            min="0"
            max="8"
            step="1"
            value={sweeps}
            onChange={(event) => setSweeps(Number(event.target.value))}
            className="mt-2 w-full accent-indigo-500"
          />

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="vi-discount">
            Discount gamma: {discount.toFixed(2)}
          </label>
          <input
            id="vi-discount"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={discount}
            onChange={(event) => setDiscount(Number(event.target.value))}
            className="mt-2 w-full accent-indigo-500"
          />

          <div className="mt-5 rounded-xl bg-slate-900 p-4 text-white">
            <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-indigo-200">
              <StepForward size={14} />
              Bellman backup
            </div>
            <p className="mt-2 text-sm text-slate-300">
              For each state, compare every action by adding immediate reward to discounted next-state value,
              then keep the best action value.
            </p>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-indigo-700">
            <RotateCcw size={16} />
            Value propagation
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            {STATES.map((state) => {
              const best = backup(state, values, discount);
              const delta = values[state] - previous[state];
              const terminal = state === 'Goal';
              return (
                <div
                  key={state}
                  className={`rounded-xl border p-4 ${
                    terminal ? 'border-emerald-200 bg-emerald-50' : 'border-slate-200 bg-slate-50'
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-sm font-semibold text-slate-900">{state}</div>
                      <div className="text-xs text-slate-600">Reward {REWARDS[state]}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-slate-900">{values[state].toFixed(2)}</div>
                      <div className={`text-xs ${delta >= 0 ? 'text-emerald-700' : 'text-rose-700'}`}>
                        {delta >= 0 ? '+' : ''}{delta.toFixed(2)}
                      </div>
                    </div>
                  </div>
                  <div className="mt-3 rounded-lg bg-white px-3 py-2 text-sm text-slate-700">
                    Greedy action: <strong>{best.actionId}</strong>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="mb-4 text-sm font-semibold uppercase tracking-wide text-indigo-700">Known model</div>
        <div className="grid gap-3 md:grid-cols-3">
          {Object.entries(ACTIONS).flatMap(([state, actions]) => (
            Object.entries(actions).map(([actionId, transitions]) => (
              <div key={`${state}-${actionId}`} className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                <div className="text-sm font-semibold text-slate-900">{state}: {actionId}</div>
                <div className="mt-2 space-y-1 text-xs text-slate-600">
                  {transitions.map((transition) => (
                    <div key={`${transition.to}-${transition.p}`}>
                      {Math.round(transition.p * 100)}% to {transition.to}
                    </div>
                  ))}
                </div>
              </div>
            ))
          ))}
        </div>
      </section>

      <AssessmentPanel lessonId="value-iteration" title="Value iteration check" />
    </div>
  );
}
