import React, { useMemo, useState } from 'react';
import { ArrowRight, RotateCcw, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const ACTIONS = {
  explore: {
    label: 'Explore',
    reward: 2,
    transitions: [
      { state: 'Shortcut', probability: 0.55, reward: 8 },
      { state: 'Loop', probability: 0.30, reward: -2 },
      { state: 'Goal', probability: 0.15, reward: 14 },
    ],
  },
  safe: {
    label: 'Safe route',
    reward: 4,
    transitions: [
      { state: 'Shortcut', probability: 0.15, reward: 8 },
      { state: 'Loop', probability: 0.10, reward: -2 },
      { state: 'Goal', probability: 0.75, reward: 14 },
    ],
  },
};

function expectedNext(action) {
  return action.transitions.reduce((sum, transition) => (
    sum + transition.probability * transition.reward
  ), 0);
}

export default function MdpFormalismAnimation() {
  const [actionId, setActionId] = useState('explore');
  const [discount, setDiscount] = useState(0.8);
  const action = ACTIONS[actionId];
  const value = useMemo(() => (
    action.reward + discount * expectedNext(action)
  ), [action, discount]);

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
      <section className="grid gap-4 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-amber-700">
            <RotateCcw size={16} />
            Markov Decision Process
          </div>
          <div className="grid gap-3 sm:grid-cols-5">
            {['States', 'Actions', 'P(next state)', 'Rewards', 'Discount'].map((label) => (
              <div key={label} className="rounded-xl border border-amber-100 bg-amber-50 p-3">
                <div className="text-xs font-semibold uppercase text-amber-700">{label}</div>
                <div className="mt-1 text-sm text-slate-700">
                  {label === 'States' && 'Where the agent can be'}
                  {label === 'Actions' && 'Choices available now'}
                  {label === 'P(next state)' && 'Environment uncertainty'}
                  {label === 'Rewards' && 'Immediate feedback'}
                  {label === 'Discount' && 'Future reward weight'}
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6 rounded-xl border border-slate-200 bg-slate-50 p-4">
            <div className="mb-3 text-sm font-semibold text-slate-700">Current state: Crossroad</div>
            <div className="flex flex-wrap items-center gap-3">
              <div className="rounded-xl bg-slate-900 px-4 py-3 text-sm font-semibold text-white">Crossroad</div>
              <ArrowRight className="text-slate-400" size={22} />
              {action.transitions.map((transition) => (
                <div key={transition.state} className="min-w-32 rounded-xl border border-slate-200 bg-white p-3">
                  <div className="text-sm font-semibold text-slate-900">{transition.state}</div>
                  <div className="mt-1 h-2 rounded-full bg-slate-100">
                    <div
                      className="h-2 rounded-full bg-amber-500"
                      style={{ width: `${transition.probability * 100}%` }}
                    />
                  </div>
                  <div className="mt-1 text-xs text-slate-600">
                    {Math.round(transition.probability * 100)}% chance, reward {transition.reward}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-amber-700">
            <SlidersHorizontal size={16} />
            Decision controls
          </div>
          <div className="grid gap-2">
            {Object.entries(ACTIONS).map(([id, item]) => (
              <button
                key={id}
                type="button"
                onClick={() => setActionId(id)}
                className={`rounded-xl border px-4 py-3 text-left transition ${
                  actionId === id
                    ? 'border-amber-500 bg-amber-50 text-amber-900'
                    : 'border-slate-200 bg-white text-slate-700 hover:border-slate-300'
                }`}
              >
                <div className="font-semibold">{item.label}</div>
                <div className="text-sm">Immediate reward {item.reward}</div>
              </button>
            ))}
          </div>

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="mdp-discount">
            Discount gamma: {discount.toFixed(2)}
          </label>
          <input
            id="mdp-discount"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={discount}
            onChange={(event) => setDiscount(Number(event.target.value))}
            className="mt-2 w-full accent-amber-500"
          />

          <div className="mt-5 rounded-xl bg-slate-900 p-4 text-white">
            <div className="text-xs uppercase tracking-wide text-amber-200">Expected one-step value</div>
            <div className="mt-1 text-3xl font-bold">{value.toFixed(2)}</div>
            <div className="mt-2 text-sm text-slate-300">
              reward {action.reward} + gamma {discount.toFixed(2)} x expected next reward {expectedNext(action).toFixed(2)}
            </div>
          </div>
        </div>
      </section>

      <AssessmentPanel lessonId="mdp-formalism" title="MDP formalism check" />
    </div>
  );
}
