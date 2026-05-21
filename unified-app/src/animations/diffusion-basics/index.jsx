import React, { useMemo, useState } from 'react';
import { Activity, SlidersHorizontal, Sparkles, Waves } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export default function DiffusionBasicsAnimation() {
  const [cleanSignal, setCleanSignal] = useState(0.7);
  const [noise, setNoise] = useState(-0.5);
  const [timestep, setTimestep] = useState(0.55);
  const [predictionError, setPredictionError] = useState(0.15);

  const diffusion = useMemo(() => {
    const signalScale = Math.sqrt(1 - timestep);
    const noiseScale = Math.sqrt(timestep);
    const noisySample = (signalScale * cleanSignal) + (noiseScale * noise);
    const predictedNoise = noise + predictionError;
    const denoised = (noisySample - (noiseScale * predictedNoise)) / Math.max(signalScale, 0.1);
    const reconstructionError = Math.abs(cleanSignal - denoised);

    return {
      signalScale,
      noiseScale,
      noisySample,
      predictedNoise,
      denoised,
      reconstructionError,
    };
  }, [cleanSignal, noise, predictionError, timestep]);

  const bars = [
    { id: 'clean', label: 'Clean x0', value: cleanSignal, color: 'bg-emerald-500' },
    { id: 'noise', label: 'Noise eps', value: noise, color: 'bg-rose-500' },
    { id: 'noisy', label: 'Noisy xt', value: diffusion.noisySample, color: 'bg-violet-500' },
    { id: 'denoised', label: 'Estimate x0', value: diffusion.denoised, color: 'bg-sky-500' },
  ];

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
      <section className="grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-fuchsia-700">
            <SlidersHorizontal size={16} />
            Forward and reverse controls
          </div>

          <label className="block text-sm font-semibold text-slate-700" htmlFor="db-clean">
            Clean signal x0: {cleanSignal.toFixed(2)}
          </label>
          <input
            id="db-clean"
            type="range"
            min="-1"
            max="1"
            step="0.05"
            value={cleanSignal}
            onChange={(event) => setCleanSignal(Number(event.target.value))}
            className="mt-2 w-full accent-fuchsia-500"
          />

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="db-noise">
            Noise eps: {noise.toFixed(2)}
          </label>
          <input
            id="db-noise"
            type="range"
            min="-1"
            max="1"
            step="0.05"
            value={noise}
            onChange={(event) => setNoise(Number(event.target.value))}
            className="mt-2 w-full accent-fuchsia-500"
          />

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="db-timestep">
            Noise timestep t: {timestep.toFixed(2)}
          </label>
          <input
            id="db-timestep"
            type="range"
            min="0.05"
            max="0.95"
            step="0.05"
            value={timestep}
            onChange={(event) => setTimestep(Number(event.target.value))}
            className="mt-2 w-full accent-fuchsia-500"
          />

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="db-error">
            Noise prediction error: {predictionError.toFixed(2)}
          </label>
          <input
            id="db-error"
            type="range"
            min="-0.6"
            max="0.6"
            step="0.05"
            value={predictionError}
            onChange={(event) => setPredictionError(Number(event.target.value))}
            className="mt-2 w-full accent-fuchsia-500"
          />
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-fuchsia-700">
            <Waves size={16} />
            Add noise, predict noise, remove noise
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-xs uppercase tracking-wide text-slate-500">Forward noising</div>
              <div className="mt-2 text-3xl font-bold text-slate-900">{diffusion.noisySample.toFixed(2)}</div>
              <p className="mt-2 text-sm text-slate-600">
                xt mixes clean signal and sampled noise.
              </p>
            </div>
            <div className="rounded-xl border border-fuchsia-200 bg-fuchsia-50 p-4">
              <div className="text-xs uppercase tracking-wide text-fuchsia-700">Model predicts eps</div>
              <div className="mt-2 text-3xl font-bold text-slate-900">{diffusion.predictedNoise.toFixed(2)}</div>
              <p className="mt-2 text-sm text-slate-700">Training teaches the network to estimate the noise.</p>
            </div>
            <div className="rounded-xl border border-slate-900 bg-slate-900 p-4 text-white">
              <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-fuchsia-200">
                <Sparkles size={14} />
                Denoised estimate
              </div>
              <div className="mt-2 text-3xl font-bold">{diffusion.denoised.toFixed(2)}</div>
              <p className="mt-2 text-sm text-slate-300">
                Error {diffusion.reconstructionError.toFixed(2)} from clean signal.
              </p>
            </div>
          </div>

          <div className="mt-5 space-y-3 rounded-xl border border-slate-200 p-4">
            {bars.map((bar) => (
              <div key={bar.id}>
                <div className="mb-1 flex items-center justify-between text-sm font-semibold text-slate-700">
                  <span>{bar.label}</span>
                  <span>{bar.value.toFixed(2)}</span>
                </div>
                <div className="h-3 rounded-full bg-slate-100">
                  <div className={`h-full rounded-full ${bar.color}`} style={{ width: `${clamp((bar.value + 1) * 50, 4, 100)}%` }} />
                </div>
              </div>
            ))}
          </div>

          <div className="mt-5 flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
            <Activity className="mt-0.5 shrink-0" size={18} />
            <p>
              The model does not directly memorize clean images; it learns how to remove noise at many timesteps.
            </p>
          </div>
        </div>
      </section>

      <AssessmentPanel lessonId="diffusion-basics" title="Diffusion basics check" />
    </div>
  );
}
