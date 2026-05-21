import React, { useMemo, useState } from 'react';
import { GitBranch, RefreshCw, StepForward } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const STATES = ['Start', 'Bridge', 'Trap'];
const REWARDS = { Start: 0, Bridge: 1, Trap: -5, Goal: 10 };
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
    recover: [{ to: 'Bridge', p: 0.6 }, { to: 'Trap', p: 0.4 }],
    wait: [{ to: 'Trap', p: 1 }],
  },
};

const INITIAL_POLICY = {
  Start: 'risk',
  Bridge: 'reset',
  Trap: 'wait',
};

function evaluate(policy, discount, depth) {
  let values = { Start: 0, Bridge: 0, Trap: 0, Goal: REWARDS.Goal };

  for (let step = 0; step < depth; step += 1) {
    const next = { Goal: REWARDS.Goal };
    for (const state of STATES) {
      next[state] = ACTIONS[state][policy[state]].reduce((sum, transition) => (
        sum + transition.p * (REWARDS[transition.to] + discount * values[transition.to])
      ), 0);
    }
    values = next;
  }

  return values;
}

function greedyAction(state, values, discount) {
  return Object.entries(ACTIONS[state]).reduce((best, [actionId, transitions]) => {
    const score = transitions.reduce((sum, transition) => (
      sum + transition.p * (REWARDS[transition.to] + discount * values[transition.to])
    ), 0);
    return score > best.score ? { actionId, score } : best;
  }, { actionId: null, score: Number.NEGATIVE_INFINITY });
}

function improve(policy, values, discount) {
  return Object.fromEntries(STATES.map((state) => [
    state,
    greedyAction(state, values, discount).actionId,
  ]));
}

function runPolicyIteration(rounds, depth, discount) {
  let policy = INITIAL_POLICY;
  const snapshots = [];

  for (let round = 0; round <= rounds; round += 1) {
    const values = evaluate(policy, discount, depth);
    const improved = improve(policy, values, discount);
    snapshots.push({ policy, values, improved });
    policy = improved;
  }

  return snapshots;
}

export default function PolicyIterationAnimation() {
  const [rounds, setRounds] = useState(1);
  const [depth, setDepth] = useState(3);
  const [discount, setDiscount] = useState(0.8);
  const snapshots = useMemo(() => (
    runPolicyIteration(rounds, depth, discount)
  ), [rounds, depth, discount]);
  const snapshot = snapshots[snapshots.length - 1];
  const stable = STATES.every((state) => snapshot.policy[state] === snapshot.improved[state]);

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
      <section className="grid gap-4 lg:grid-cols-[0.85fr_1.15fr]">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-emerald-700">
            <RefreshCw size={16} />
            Evaluation then improvement
          </div>

          <label className="block text-sm font-semibold text-slate-700" htmlFor="pi-rounds">
            Improvement rounds: {rounds}
          </label>
          <input
            id="pi-rounds"
            type="range"
            min="0"
            max="5"
            step="1"
            value={rounds}
            onChange={(event) => setRounds(Number(event.target.value))}
            className="mt-2 w-full accent-emerald-500"
          />

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="pi-depth">
            Evaluation depth: {depth}
          </label>
          <input
            id="pi-depth"
            type="range"
            min="1"
            max="8"
            step="1"
            value={depth}
            onChange={(event) => setDepth(Number(event.target.value))}
            className="mt-2 w-full accent-emerald-500"
          />

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="pi-discount">
            Discount gamma: {discount.toFixed(2)}
          </label>
          <input
            id="pi-discount"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={discount}
            onChange={(event) => setDiscount(Number(event.target.value))}
            className="mt-2 w-full accent-emerald-500"
          />

          <div className={`mt-5 rounded-xl p-4 ${stable ? 'bg-emerald-900' : 'bg-slate-900'} text-white`}>
            <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-emerald-200">
              <StepForward size={14} />
              {stable ? 'Policy stable' : 'Policy can improve'}
            </div>
            <p className="mt-2 text-sm text-slate-200">
              Evaluate current actions for each state, then greedily replace any action with a better lookahead.
            </p>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-emerald-700">
            <GitBranch size={16} />
            Policy table
          </div>
          <div className="grid gap-3">
            {STATES.map((state) => {
              const current = snapshot.policy[state];
              const next = snapshot.improved[state];
              const changed = current !== next;
              return (
                <div
                  key={state}
                  className={`rounded-xl border p-4 ${
                    changed ? 'border-amber-200 bg-amber-50' : 'border-emerald-200 bg-emerald-50'
                  }`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <div className="text-sm font-semibold text-slate-900">{state}</div>
                      <div className="text-xs text-slate-600">Value {snapshot.values[state].toFixed(2)}</div>
                    </div>
                    <div className="text-right text-sm">
                      <div>Current: <strong>{current}</strong></div>
                      <div>Improved: <strong>{next}</strong></div>
                    </div>
                  </div>
                  <div className="mt-3 h-2 rounded-full bg-white">
                    <div
                      className={`h-2 rounded-full ${changed ? 'bg-amber-500' : 'bg-emerald-500'}`}
                      style={{ width: `${Math.min(100, Math.max(12, Math.abs(snapshot.values[state]) * 8))}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="mb-4 text-sm font-semibold uppercase tracking-wide text-emerald-700">Action lookaheads</div>
        <div className="grid gap-3 md:grid-cols-3">
          {STATES.map((state) => (
            <div key={state} className="rounded-xl border border-slate-200 bg-slate-50 p-3">
              <div className="text-sm font-semibold text-slate-900">{state}</div>
              <div className="mt-2 space-y-1 text-xs text-slate-600">
                {Object.keys(ACTIONS[state]).map((actionId) => (
                  <div key={actionId}>
                    {actionId}: {greedyAction(state, { ...snapshot.values, Goal: REWARDS.Goal }, discount).actionId === actionId ? 'greedy candidate' : 'alternative'}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      <AssessmentPanel lessonId="policy-iteration" title="Policy iteration check" />
    </div>
  );
}
