import React, { useMemo, useState } from 'react';
import { Activity, Layers, SlidersHorizontal, ToggleLeft } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

function normalize(value, mean, std) {
  return (value - mean) / Math.max(std, 0.1);
}

export default function DropoutBatchNormAnimation() {
  const [activation, setActivation] = useState(3);
  const [batchMean, setBatchMean] = useState(1);
  const [batchStd, setBatchStd] = useState(2);
  const [gamma, setGamma] = useState(1);
  const [beta, setBeta] = useState(0);
  const [dropoutRate, setDropoutRate] = useState(0.4);
  const [trainingMode, setTrainingMode] = useState(true);

  const flow = useMemo(() => {
    const normalized = normalize(activation, batchMean, batchStd);
    const batchNormOutput = (gamma * normalized) + beta;
    const keepProbability = 1 - dropoutRate;
    const droppedValue = trainingMode ? 0 : batchNormOutput;
    const keptValue = trainingMode ? batchNormOutput / Math.max(keepProbability, 0.05) : batchNormOutput;

    return {
      normalized,
      batchNormOutput,
      keepProbability,
      droppedValue,
      keptValue,
      expectedValue: (dropoutRate * droppedValue) + (keepProbability * keptValue),
    };
  }, [activation, batchMean, batchStd, beta, dropoutRate, gamma, trainingMode]);

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
      <section className="grid gap-4 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-cyan-700">
            <SlidersHorizontal size={16} />
            Layer controls
          </div>

          <label className="block text-sm font-semibold text-slate-700" htmlFor="dbn-activation">
            Incoming activation: {activation.toFixed(1)}
          </label>
          <input
            id="dbn-activation"
            type="range"
            min="-4"
            max="6"
            step="0.5"
            value={activation}
            onChange={(event) => setActivation(Number(event.target.value))}
            className="mt-2 w-full accent-cyan-500"
          />

          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            <label className="block text-sm font-semibold text-slate-700" htmlFor="dbn-mean">
              Batch mean {batchMean.toFixed(1)}
              <input
                id="dbn-mean"
                type="range"
                min="-3"
                max="4"
                step="0.5"
                value={batchMean}
                onChange={(event) => setBatchMean(Number(event.target.value))}
                className="mt-2 w-full accent-cyan-500"
              />
            </label>
            <label className="block text-sm font-semibold text-slate-700" htmlFor="dbn-std">
              Batch std {batchStd.toFixed(1)}
              <input
                id="dbn-std"
                type="range"
                min="0.5"
                max="4"
                step="0.5"
                value={batchStd}
                onChange={(event) => setBatchStd(Number(event.target.value))}
                className="mt-2 w-full accent-cyan-500"
              />
            </label>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            <label className="block text-sm font-semibold text-slate-700" htmlFor="dbn-gamma">
              BatchNorm gamma {gamma.toFixed(1)}
              <input
                id="dbn-gamma"
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={gamma}
                onChange={(event) => setGamma(Number(event.target.value))}
                className="mt-2 w-full accent-cyan-500"
              />
            </label>
            <label className="block text-sm font-semibold text-slate-700" htmlFor="dbn-beta">
              BatchNorm beta {beta.toFixed(1)}
              <input
                id="dbn-beta"
                type="range"
                min="-2"
                max="2"
                step="0.1"
                value={beta}
                onChange={(event) => setBeta(Number(event.target.value))}
                className="mt-2 w-full accent-cyan-500"
              />
            </label>
          </div>

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="dbn-dropout">
            Dropout rate {(dropoutRate * 100).toFixed(0)}%
          </label>
          <input
            id="dbn-dropout"
            type="range"
            min="0"
            max="0.8"
            step="0.1"
            value={dropoutRate}
            onChange={(event) => setDropoutRate(Number(event.target.value))}
            className="mt-2 w-full accent-cyan-500"
          />

          <button
            type="button"
            onClick={() => setTrainingMode((value) => !value)}
            className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl border border-cyan-200 bg-cyan-50 px-4 py-3 text-sm font-semibold text-cyan-900"
          >
            <ToggleLeft size={18} />
            {trainingMode ? 'Training mode' : 'Inference mode'}
          </button>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-cyan-700">
            <Layers size={16} />
            BatchNorm then dropout
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-xs uppercase tracking-wide text-slate-500">Normalize</div>
              <div className="mt-2 text-3xl font-bold text-slate-900">{flow.normalized.toFixed(2)}</div>
              <p className="mt-2 text-sm text-slate-600">Subtract batch mean, divide by batch std.</p>
            </div>
            <div className="rounded-xl border border-cyan-200 bg-cyan-50 p-4">
              <div className="text-xs uppercase tracking-wide text-cyan-700">Scale and shift</div>
              <div className="mt-2 text-3xl font-bold text-slate-900">{flow.batchNormOutput.toFixed(2)}</div>
              <p className="mt-2 text-sm text-slate-700">Gamma and beta restore learnable range.</p>
            </div>
            <div className="rounded-xl border border-slate-900 bg-slate-900 p-4 text-white">
              <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-cyan-200">
                <Activity size={14} />
                Dropout output
              </div>
              <div className="mt-2 text-3xl font-bold">{flow.expectedValue.toFixed(2)}</div>
              <p className="mt-2 text-sm text-slate-300">
                {trainingMode ? 'Expected value after random masking.' : 'No units are masked at inference.'}
              </p>
            </div>
          </div>

          <div className="mt-5 grid gap-4 md:grid-cols-2">
            <div className="rounded-xl border border-slate-200 p-4">
              <div className="text-sm font-semibold text-slate-900">If kept</div>
              <div className="mt-3 h-3 overflow-hidden rounded-full bg-slate-100">
                <div
                  className="h-full rounded-full bg-emerald-500"
                  style={{ width: `${Math.min(100, Math.abs(flow.keptValue) * 18)}%` }}
                />
              </div>
              <div className="mt-2 text-sm text-slate-600">{flow.keptValue.toFixed(2)}</div>
            </div>
            <div className="rounded-xl border border-slate-200 p-4">
              <div className="text-sm font-semibold text-slate-900">If dropped</div>
              <div className="mt-3 h-3 overflow-hidden rounded-full bg-slate-100">
                <div
                  className="h-full rounded-full bg-rose-500"
                  style={{ width: `${Math.min(100, Math.abs(flow.droppedValue) * 18)}%` }}
                />
              </div>
              <div className="mt-2 text-sm text-slate-600">{flow.droppedValue.toFixed(2)}</div>
            </div>
          </div>

          <p className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
            BatchNorm stabilizes activation scale. Dropout regularizes by forcing redundant pathways. They solve different training problems.
          </p>
        </div>
      </section>

      <AssessmentPanel lessonId="dropout-batchnorm" title="Dropout and BatchNorm check" />
    </div>
  );
}
