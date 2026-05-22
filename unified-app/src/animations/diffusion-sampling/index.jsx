import React, { useMemo, useState } from 'react';
import { Activity, GitBranch, SlidersHorizontal, Workflow } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const SAMPLERS = {
  ddpm: {
    label: 'DDPM',
    caption: 'Stochastic reverse steps add controlled noise while denoising.',
    eta: 0.35,
    color: 'bg-fuchsia-500',
  },
  ddim: {
    label: 'DDIM',
    caption: 'Deterministic steps follow a straighter path from noise to sample.',
    eta: 0,
    color: 'bg-sky-500',
  },
  flow: {
    label: 'Flow / ODE',
    caption: 'A flow or ODE-style path uses a velocity field to transport noise toward data.',
    eta: 0.08,
    color: 'bg-emerald-500',
  },
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function samplerNoise(mode, progress, step, eta) {
  if (mode === 'ddim') return 1 - progress;
  if (mode === 'flow') return Math.pow(1 - progress, 1.4) + (eta * Math.sin(step * 1.7));
  return (1 - progress) + (eta * Math.sin(step * 2.1));
}

export default function DiffusionSamplingAnimation() {
  const [mode, setMode] = useState('ddpm');
  const [steps, setSteps] = useState(8);
  const [predictionQuality, setPredictionQuality] = useState(0.78);

  const sampler = SAMPLERS[mode];
  const trajectory = useMemo(() => {
    const qualityPenalty = (1 - predictionQuality) * 0.35;
    return Array.from({ length: steps + 1 }, (_, index) => {
      const progress = index / steps;
      const rawNoise = samplerNoise(mode, progress, index, sampler.eta);
      const remainingNoise = clamp(rawNoise + qualityPenalty, 0, 1);
      const sampleClarity = clamp(1 - remainingNoise, 0, 1);
      return {
        index,
        progress,
        remainingNoise,
        sampleClarity,
      };
    });
  }, [mode, predictionQuality, sampler.eta, steps]);

  const final = trajectory[trajectory.length - 1];
  const stochasticity = mode === 'ddim' ? 0 : sampler.eta;

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
      <section className="grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-fuchsia-700">
            <SlidersHorizontal size={16} />
            Sampler controls
          </div>

          <div className="grid gap-2 sm:grid-cols-3">
            {Object.entries(SAMPLERS).map(([id, option]) => (
              <button
                key={id}
                type="button"
                onClick={() => setMode(id)}
                className={`rounded-xl border px-3 py-3 text-left text-sm transition ${
                  mode === id
                    ? 'border-slate-900 bg-slate-900 text-white'
                    : 'border-slate-200 bg-slate-50 text-slate-700 hover:border-fuchsia-300'
                }`}
              >
                <span className="block font-semibold">{option.label}</span>
                <span className={`mt-2 block h-1.5 rounded-full ${option.color}`} />
              </button>
            ))}
          </div>

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="ds-steps">
            Reverse steps: {steps}
          </label>
          <input
            id="ds-steps"
            type="range"
            min="3"
            max="16"
            step="1"
            value={steps}
            onChange={(event) => setSteps(Number(event.target.value))}
            className="mt-2 w-full accent-fuchsia-500"
          />

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="ds-quality">
            Noise prediction quality: {predictionQuality.toFixed(2)}
          </label>
          <input
            id="ds-quality"
            type="range"
            min="0.35"
            max="0.98"
            step="0.01"
            value={predictionQuality}
            onChange={(event) => setPredictionQuality(Number(event.target.value))}
            className="mt-2 w-full accent-fuchsia-500"
          />

          <div className="mt-5 rounded-xl border border-slate-200 bg-slate-50 p-4">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-800">
              <GitBranch size={16} />
              {sampler.label} intuition
            </div>
            <p className="mt-2 text-sm text-slate-600">{sampler.caption}</p>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-fuchsia-700">
            <Workflow size={16} />
            Noise to sample trajectory
          </div>

          <div className="grid gap-3">
            {trajectory.map((point) => (
              <div key={point.index} className="grid grid-cols-[3rem_1fr_4rem] items-center gap-3">
                <div className="text-xs font-semibold text-slate-500">t{point.index}</div>
                <div className="h-4 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className={`h-full rounded-full ${sampler.color}`}
                    style={{ width: `${clamp(point.sampleClarity * 100, 3, 100)}%` }}
                  />
                </div>
                <div className="text-right text-xs font-semibold text-slate-600">
                  {(point.sampleClarity * 100).toFixed(0)}%
                </div>
              </div>
            ))}
          </div>

          <div className="mt-5 grid gap-4 md:grid-cols-3">
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-xs uppercase tracking-wide text-slate-500">Stochasticity</div>
              <div className="mt-2 text-3xl font-bold text-slate-900">{stochasticity.toFixed(2)}</div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-xs uppercase tracking-wide text-slate-500">Final clarity</div>
              <div className="mt-2 text-3xl font-bold text-slate-900">{(final.sampleClarity * 100).toFixed(0)}%</div>
            </div>
            <div className="rounded-xl border border-slate-900 bg-slate-900 p-4 text-white">
              <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-fuchsia-200">
                <Activity size={14} />
                Path shape
              </div>
              <div className="mt-2 text-xl font-bold">{mode === 'flow' ? 'continuous' : mode === 'ddim' ? 'deterministic' : 'stochastic'}</div>
            </div>
          </div>

          <p className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
            In this beginner comparison, samplers reuse a denoising model differently: DDPM keeps randomness,
            DDIM can remove it, and flow/ODE-style sampling follows a smooth transport path.
          </p>
        </div>
      </section>

      <AssessmentPanel lessonId="diffusion-sampling" title="Diffusion sampling check" />
    </div>
  );
}
