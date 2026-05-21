import React, { useMemo, useState } from 'react';
import { Activity, BarChart3, RefreshCw, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export default function TrainingLoopDynamicsAnimation() {
  const [learningRate, setLearningRate] = useState(0.2);
  const [batchSize, setBatchSize] = useState(32);
  const [steps, setSteps] = useState(8);
  const [validationGap, setValidationGap] = useState(0.25);
  const [curvature, setCurvature] = useState(1.4);

  const dynamics = useMemo(() => {
    const noise = 1 / Math.sqrt(batchSize);
    const stepStrength = learningRate * curvature;
    const stableStep = stepStrength < 0.85;
    const trainLoss = stableStep
      ? clamp(2.6 * Math.exp(-steps * stepStrength * 0.45) + noise * 0.8, 0.05, 3)
      : clamp(1.2 + (stepStrength - 0.85) * steps * 0.45, 0.05, 5);
    const overfitPressure = Math.max(0, steps - 7) * validationGap * 0.08;
    const validationLoss = clamp(trainLoss + validationGap + overfitPressure + noise * 0.35, 0.05, 5);
    const generalizationGap = validationLoss - trainLoss;

    return {
      noise,
      stepStrength,
      trainLoss,
      validationLoss,
      generalizationGap,
      stableStep,
      state: !stableStep ? 'overshooting' : generalizationGap > 0.9 ? 'overfitting' : noise > 0.15 ? 'noisy' : 'healthy',
      history: Array.from({ length: steps }, (_, index) => {
        const progress = index + 1;
        const base = stableStep ? 2.6 * Math.exp(-progress * stepStrength * 0.45) : 1.2 + progress * 0.22;
        return clamp(base + noise * (index % 2 === 0 ? 0.7 : -0.2), 0.05, 5);
      }),
    };
  }, [batchSize, curvature, learningRate, steps, validationGap]);

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
      <section className="grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-indigo-700">
            <SlidersHorizontal size={16} />
            Training controls
          </div>

          <label className="block text-sm font-semibold text-slate-700" htmlFor="tld-lr">
            Learning rate {learningRate.toFixed(2)}
          </label>
          <input
            id="tld-lr"
            type="range"
            min="0.02"
            max="0.8"
            step="0.02"
            value={learningRate}
            onChange={(event) => setLearningRate(Number(event.target.value))}
            className="mt-2 w-full accent-indigo-500"
          />

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="tld-batch">
            Mini-batch size {batchSize}
          </label>
          <input
            id="tld-batch"
            type="range"
            min="4"
            max="256"
            step="4"
            value={batchSize}
            onChange={(event) => setBatchSize(Number(event.target.value))}
            className="mt-2 w-full accent-indigo-500"
          />

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="tld-steps">
            Optimizer steps {steps}
          </label>
          <input
            id="tld-steps"
            type="range"
            min="1"
            max="16"
            step="1"
            value={steps}
            onChange={(event) => setSteps(Number(event.target.value))}
            className="mt-2 w-full accent-indigo-500"
          />

          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            <label className="block text-sm font-semibold text-slate-700" htmlFor="tld-gap">
              Validation difficulty {validationGap.toFixed(2)}
              <input
                id="tld-gap"
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={validationGap}
                onChange={(event) => setValidationGap(Number(event.target.value))}
                className="mt-2 w-full accent-indigo-500"
              />
            </label>
            <label className="block text-sm font-semibold text-slate-700" htmlFor="tld-curvature">
              Loss curvature {curvature.toFixed(1)}
              <input
                id="tld-curvature"
                type="range"
                min="0.5"
                max="3"
                step="0.1"
                value={curvature}
                onChange={(event) => setCurvature(Number(event.target.value))}
                className="mt-2 w-full accent-indigo-500"
              />
            </label>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-indigo-700">
            <RefreshCw size={16} />
            Loop signal
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-xs uppercase tracking-wide text-slate-500">Gradient noise</div>
              <div className="mt-2 text-3xl font-bold text-slate-900">{dynamics.noise.toFixed(2)}</div>
              <p className="mt-2 text-sm text-slate-600">Smaller batches make noisier gradients.</p>
            </div>
            <div className="rounded-xl border border-indigo-200 bg-indigo-50 p-4">
              <div className="text-xs uppercase tracking-wide text-indigo-700">Step strength</div>
              <div className="mt-2 text-3xl font-bold text-slate-900">{dynamics.stepStrength.toFixed(2)}</div>
              <p className="mt-2 text-sm text-slate-700">Learning rate times local curvature.</p>
            </div>
            <div className="rounded-xl border border-slate-900 bg-slate-900 p-4 text-white">
              <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-indigo-200">
                <Activity size={14} />
                Diagnosis
              </div>
              <div className="mt-2 text-2xl font-bold capitalize">{dynamics.state}</div>
              <p className="mt-2 text-sm text-slate-300">
                Train {dynamics.trainLoss.toFixed(2)}, validation {dynamics.validationLoss.toFixed(2)}.
              </p>
            </div>
          </div>

          <div className="mt-5 rounded-xl border border-slate-200 p-4">
            <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-900">
              <BarChart3 size={16} />
              Training loss by step
            </div>
            <div className="flex h-36 items-end gap-2">
              {dynamics.history.map((loss, index) => (
                <div
                  key={`${index}-${loss.toFixed(2)}`}
                  className="flex flex-1 items-end rounded-t bg-indigo-500"
                  style={{ height: `${clamp(loss * 26, 4, 100)}%` }}
                  title={`Step ${index + 1}: ${loss.toFixed(2)}`}
                />
              ))}
            </div>
          </div>

          <p className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
            Training loss alone is not enough. A healthy loop watches update stability, batch noise, and validation behavior together.
          </p>
        </div>
      </section>

      <AssessmentPanel lessonId="training-loop-dynamics" title="Training loop check" />
    </div>
  );
}
