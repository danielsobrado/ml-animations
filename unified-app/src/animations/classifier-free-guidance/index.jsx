import React, { useMemo, useState } from 'react';
import { Activity, SlidersHorizontal, Target, Zap } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export default function ClassifierFreeGuidanceAnimation() {
  const [guidanceScale, setGuidanceScale] = useState(5);
  const [conditionalPrediction, setConditionalPrediction] = useState(0.42);
  const [unconditionalPrediction, setUnconditionalPrediction] = useState(-0.12);

  const guidance = useMemo(() => {
    const promptDirection = conditionalPrediction - unconditionalPrediction;
    const guidedPrediction = unconditionalPrediction + (guidanceScale * promptDirection);
    const promptMatch = clamp(45 + (guidanceScale * 9) + (Math.abs(promptDirection) * 18), 0, 100);
    const diversity = clamp(96 - (guidanceScale * 10), 8, 100);
    const overshoot = clamp(Math.abs(guidedPrediction) - 1.2, 0, 2);
    const artifactRisk = clamp((guidanceScale - 7) * 13 + overshoot * 24, 0, 100);

    return {
      promptDirection,
      guidedPrediction,
      promptMatch,
      diversity,
      artifactRisk,
    };
  }, [conditionalPrediction, guidanceScale, unconditionalPrediction]);

  const bars = [
    { id: 'uncond', label: 'Unconditional eps', value: unconditionalPrediction, color: 'bg-slate-500' },
    { id: 'cond', label: 'Conditional eps', value: conditionalPrediction, color: 'bg-sky-500' },
    { id: 'guided', label: 'Guided eps', value: guidance.guidedPrediction, color: 'bg-fuchsia-500' },
  ];

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
      <section className="grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-fuchsia-700">
            <SlidersHorizontal size={16} />
            Guidance controls
          </div>

          <label className="block text-sm font-semibold text-slate-700" htmlFor="cfg-scale">
            Guidance scale: {guidanceScale.toFixed(1)}
          </label>
          <input
            id="cfg-scale"
            type="range"
            min="0"
            max="12"
            step="0.5"
            value={guidanceScale}
            onChange={(event) => setGuidanceScale(Number(event.target.value))}
            className="mt-2 w-full accent-fuchsia-500"
          />

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="cfg-cond">
            Conditional noise prediction: {conditionalPrediction.toFixed(2)}
          </label>
          <input
            id="cfg-cond"
            type="range"
            min="-1"
            max="1"
            step="0.05"
            value={conditionalPrediction}
            onChange={(event) => setConditionalPrediction(Number(event.target.value))}
            className="mt-2 w-full accent-sky-500"
          />

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="cfg-uncond">
            Unconditional noise prediction: {unconditionalPrediction.toFixed(2)}
          </label>
          <input
            id="cfg-uncond"
            type="range"
            min="-1"
            max="1"
            step="0.05"
            value={unconditionalPrediction}
            onChange={(event) => setUnconditionalPrediction(Number(event.target.value))}
            className="mt-2 w-full accent-slate-500"
          />

          <div className="mt-5 rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700">
            CFG pushes the sampler away from the unconditional prediction and toward the prompt-conditioned prediction.
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-fuchsia-700">
            <Target size={16} />
            Prompt pull vs diversity
          </div>

          <div className="space-y-3 rounded-xl border border-slate-200 p-4">
            {bars.map((bar) => (
              <div key={bar.id}>
                <div className="mb-1 flex items-center justify-between text-sm font-semibold text-slate-700">
                  <span>{bar.label}</span>
                  <span>{bar.value.toFixed(2)}</span>
                </div>
                <div className="h-3 rounded-full bg-slate-100">
                  <div className={`h-full rounded-full ${bar.color}`} style={{ width: `${clamp((bar.value + 2) * 25, 4, 100)}%` }} />
                </div>
              </div>
            ))}
          </div>

          <div className="mt-5 grid gap-4 md:grid-cols-3">
            <div className="rounded-xl border border-sky-200 bg-sky-50 p-4">
              <div className="text-xs uppercase tracking-wide text-sky-700">Prompt match</div>
              <div className="mt-2 text-3xl font-bold text-slate-900">{guidance.promptMatch.toFixed(0)}%</div>
            </div>
            <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4">
              <div className="text-xs uppercase tracking-wide text-emerald-700">Diversity</div>
              <div className="mt-2 text-3xl font-bold text-slate-900">{guidance.diversity.toFixed(0)}%</div>
            </div>
            <div className="rounded-xl border border-slate-900 bg-slate-900 p-4 text-white">
              <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-fuchsia-200">
                <Zap size={14} />
                Artifact risk
              </div>
              <div className="mt-2 text-3xl font-bold">{guidance.artifactRisk.toFixed(0)}%</div>
            </div>
          </div>

          <div className="mt-5 flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
            <Activity className="mt-0.5 shrink-0" size={18} />
            <p>
              Higher guidance usually improves prompt following, but too much scale can reduce variety and exaggerate artifacts.
            </p>
          </div>
        </div>
      </section>

      <AssessmentPanel lessonId="classifier-free-guidance" title="Classifier-free guidance check" />
    </div>
  );
}
