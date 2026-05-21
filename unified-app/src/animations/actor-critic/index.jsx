import React, { useMemo, useState } from 'react';
import { Brain, SlidersHorizontal, Users } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

export default function ActorCriticAnimation() {
  const [returnValue, setReturnValue] = useState(8);
  const [criticValue, setCriticValue] = useState(4);
  const [actorStep, setActorStep] = useState(0.4);
  const [criticStep, setCriticStep] = useState(0.35);
  const advantage = returnValue - criticValue;

  const update = useMemo(() => {
    const actorSignal = actorStep * advantage;
    const criticSignal = criticStep * (returnValue - criticValue);
    return {
      actorSignal,
      criticSignal,
      nextCritic: criticValue + criticSignal,
      reinforce: actorSignal >= 0,
    };
  }, [returnValue, criticValue, actorStep, criticStep, advantage]);

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
      <section className="grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-violet-700">
            <SlidersHorizontal size={16} />
            Shared trajectory
          </div>

          <label className="block text-sm font-semibold text-slate-700" htmlFor="ac-return">
            Observed return: {returnValue}
          </label>
          <input
            id="ac-return"
            type="range"
            min="-8"
            max="14"
            step="1"
            value={returnValue}
            onChange={(event) => setReturnValue(Number(event.target.value))}
            className="mt-2 w-full accent-violet-500"
          />

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="ac-critic">
            Critic value estimate: {criticValue}
          </label>
          <input
            id="ac-critic"
            type="range"
            min="-8"
            max="14"
            step="1"
            value={criticValue}
            onChange={(event) => setCriticValue(Number(event.target.value))}
            className="mt-2 w-full accent-violet-500"
          />

          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            <label className="block text-sm font-semibold text-slate-700" htmlFor="ac-actor-step">
              Actor step {actorStep.toFixed(2)}
              <input
                id="ac-actor-step"
                type="range"
                min="0.05"
                max="0.8"
                step="0.05"
                value={actorStep}
                onChange={(event) => setActorStep(Number(event.target.value))}
                className="mt-2 w-full accent-violet-500"
              />
            </label>
            <label className="block text-sm font-semibold text-slate-700" htmlFor="ac-critic-step">
              Critic step {criticStep.toFixed(2)}
              <input
                id="ac-critic-step"
                type="range"
                min="0.05"
                max="0.8"
                step="0.05"
                value={criticStep}
                onChange={(event) => setCriticStep(Number(event.target.value))}
                className="mt-2 w-full accent-violet-500"
              />
            </label>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-violet-700">
            <Users size={16} />
            Actor and critic updates
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-xl border border-violet-200 bg-violet-50 p-4">
              <div className="flex items-center gap-2 text-sm font-semibold text-violet-900">
                <Brain size={16} />
                Actor
              </div>
              <div className="mt-4 text-3xl font-bold text-slate-900">
                {update.actorSignal >= 0 ? '+' : ''}{update.actorSignal.toFixed(2)}
              </div>
              <p className="mt-2 text-sm text-slate-700">
                {update.reinforce ? 'Increase' : 'Decrease'} the sampled action log-probability.
              </p>
            </div>
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-sm font-semibold text-slate-900">Critic</div>
              <div className="mt-4 text-3xl font-bold text-slate-900">{update.nextCritic.toFixed(2)}</div>
              <p className="mt-2 text-sm text-slate-700">
                Move value estimate by {update.criticSignal >= 0 ? '+' : ''}{update.criticSignal.toFixed(2)} toward the observed return.
              </p>
            </div>
          </div>

          <div className="mt-5 rounded-xl bg-slate-900 p-4 text-white">
            <div className="text-xs uppercase tracking-wide text-violet-200">Advantage = return - value</div>
            <div className="mt-1 text-3xl font-bold">{advantage}</div>
            <p className="mt-2 text-sm text-slate-300">
              Advantage is the bridge: it trains the actor while the critic learns a better baseline.
            </p>
          </div>
        </div>
      </section>

      <AssessmentPanel lessonId="actor-critic" title="Actor-critic check" />
    </div>
  );
}
