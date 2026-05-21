import React, { useMemo, useState } from 'react';
import { Activity, AlertTriangle, SlidersHorizontal, Target } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const GOAL_STATE = 6;

function potential(state) {
  return -Math.abs(GOAL_STATE - state);
}

export default function RewardShapingAnimation() {
  const [state, setState] = useState(2);
  const [nextState, setNextState] = useState(3);
  const [gamma, setGamma] = useState(0.9);
  const [shapingWeight, setShapingWeight] = useState(1);
  const [stepPenalty, setStepPenalty] = useState(0);

  const reward = useMemo(() => {
    const reachedGoal = nextState === GOAL_STATE;
    const rawReward = reachedGoal ? 10 : stepPenalty;
    const shapingBonus = shapingWeight * ((gamma * potential(nextState)) - potential(state));
    const totalReward = rawReward + shapingBonus;
    const movedCloser = Math.abs(GOAL_STATE - nextState) < Math.abs(GOAL_STATE - state);

    return {
      rawReward,
      shapingBonus,
      totalReward,
      movedCloser,
      reachedGoal,
      currentPotential: potential(state),
      nextPotential: potential(nextState),
    };
  }, [gamma, nextState, shapingWeight, state, stepPenalty]);

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
      <section className="grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-amber-700">
            <SlidersHorizontal size={16} />
            Transition controls
          </div>

          <label className="block text-sm font-semibold text-slate-700" htmlFor="rs-state">
            Current state: {state}
          </label>
          <input
            id="rs-state"
            type="range"
            min="0"
            max="5"
            step="1"
            value={state}
            onChange={(event) => setState(Number(event.target.value))}
            className="mt-2 w-full accent-amber-500"
          />

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="rs-next-state">
            Next state: {nextState}
          </label>
          <input
            id="rs-next-state"
            type="range"
            min="0"
            max="6"
            step="1"
            value={nextState}
            onChange={(event) => setNextState(Number(event.target.value))}
            className="mt-2 w-full accent-amber-500"
          />

          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            <label className="block text-sm font-semibold text-slate-700" htmlFor="rs-gamma">
              Discount gamma {gamma.toFixed(2)}
              <input
                id="rs-gamma"
                type="range"
                min="0.5"
                max="1"
                step="0.05"
                value={gamma}
                onChange={(event) => setGamma(Number(event.target.value))}
                className="mt-2 w-full accent-amber-500"
              />
            </label>
            <label className="block text-sm font-semibold text-slate-700" htmlFor="rs-weight">
              Shaping weight {shapingWeight.toFixed(1)}
              <input
                id="rs-weight"
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={shapingWeight}
                onChange={(event) => setShapingWeight(Number(event.target.value))}
                className="mt-2 w-full accent-amber-500"
              />
            </label>
          </div>

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="rs-step-penalty">
            Sparse step reward: {stepPenalty}
          </label>
          <input
            id="rs-step-penalty"
            type="range"
            min="-2"
            max="1"
            step="1"
            value={stepPenalty}
            onChange={(event) => setStepPenalty(Number(event.target.value))}
            className="mt-2 w-full accent-amber-500"
          />
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-amber-700">
            <Target size={16} />
            Dense learning signal
          </div>

          <div className="grid grid-cols-7 gap-2">
            {Array.from({ length: GOAL_STATE + 1 }, (_, index) => {
              const isCurrent = index === state;
              const isNext = index === nextState;
              const isGoal = index === GOAL_STATE;
              return (
                <div
                  key={index}
                  className={`flex aspect-square flex-col items-center justify-center rounded-xl border text-sm font-semibold ${
                    isGoal
                      ? 'border-emerald-300 bg-emerald-50 text-emerald-900'
                      : isCurrent || isNext
                        ? 'border-amber-400 bg-amber-50 text-amber-900'
                        : 'border-slate-200 bg-slate-50 text-slate-600'
                  }`}
                >
                  <span>{index}</span>
                  <span className="text-[10px] font-medium">
                    {isGoal ? 'goal' : isCurrent ? 'now' : isNext ? 'next' : ''}
                  </span>
                </div>
              );
            })}
          </div>

          <div className="mt-5 grid gap-4 md:grid-cols-3">
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-xs uppercase tracking-wide text-slate-500">Task reward</div>
              <div className="mt-2 text-3xl font-bold text-slate-900">{reward.rawReward.toFixed(2)}</div>
              <p className="mt-2 text-sm text-slate-600">
                {reward.reachedGoal ? 'Goal reached.' : 'Sparse signal before shaping.'}
              </p>
            </div>
            <div className="rounded-xl border border-amber-200 bg-amber-50 p-4">
              <div className="text-xs uppercase tracking-wide text-amber-700">Potential bonus</div>
              <div className="mt-2 text-3xl font-bold text-slate-900">
                {reward.shapingBonus >= 0 ? '+' : ''}{reward.shapingBonus.toFixed(2)}
              </div>
              <p className="mt-2 text-sm text-slate-700">
                Phi now {reward.currentPotential}, Phi next {reward.nextPotential}.
              </p>
            </div>
            <div className="rounded-xl border border-slate-900 bg-slate-900 p-4 text-white">
              <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-amber-200">
                <Activity size={14} />
                Learning signal
              </div>
              <div className="mt-2 text-3xl font-bold">
                {reward.totalReward >= 0 ? '+' : ''}{reward.totalReward.toFixed(2)}
              </div>
              <p className="mt-2 text-sm text-slate-300">
                {reward.movedCloser ? 'Closer to goal is encouraged.' : 'Moving away loses shaped reward.'}
              </p>
            </div>
          </div>

          <div className="mt-5 flex items-start gap-3 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900">
            <AlertTriangle className="mt-0.5 shrink-0" size={18} />
            <p>
              Reward shaping should guide exploration. If the added reward changes what behavior is optimal, it has changed the task.
            </p>
          </div>
        </div>
      </section>

      <AssessmentPanel lessonId="reward-shaping" title="Reward shaping check" />
    </div>
  );
}
