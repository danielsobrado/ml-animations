import React, { useMemo, useState } from 'react';
import { Activity, AlertTriangle, SlidersHorizontal, Zap } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const METHODS = {
  tiny: {
    label: 'Too small',
    description: 'Weights barely pass signal forward or gradients backward.',
    std: ({ fanIn }) => 0.08 / Math.sqrt(fanIn),
  },
  xavier: {
    label: 'Xavier / Glorot',
    description: 'Balances fan-in and fan-out for tanh-like activations.',
    std: ({ fanIn, fanOut }) => Math.sqrt(2 / (fanIn + fanOut)),
  },
  he: {
    label: 'He',
    description: 'Compensates for ReLU dropping roughly half the activations.',
    std: ({ fanIn }) => Math.sqrt(2 / fanIn),
  },
  huge: {
    label: 'Too large',
    description: 'Weights amplify signal until activations or gradients explode.',
    std: ({ fanIn }) => 3 / Math.sqrt(fanIn),
  },
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export default function InitializationAnimation() {
  const [method, setMethod] = useState('he');
  const [activation, setActivation] = useState('relu');
  const [fanIn, setFanIn] = useState(64);
  const [fanOut, setFanOut] = useState(64);
  const [layers, setLayers] = useState(5);

  const signal = useMemo(() => {
    const std = METHODS[method].std({ fanIn, fanOut });
    const activationFactor = activation === 'relu' ? 0.5 : 0.8;
    const multiplier = fanIn * std * std * activationFactor;
    const variances = Array.from({ length: layers }, (_, index) => {
      const variance = Math.pow(multiplier, index + 1);
      return {
        id: index + 1,
        variance,
        width: `${clamp(variance * 50, 4, 100)}%`,
      };
    });
    const finalVariance = variances.at(-1)?.variance ?? 1;
    const health = finalVariance < 0.25 ? 'vanishing' : finalVariance > 4 ? 'exploding' : 'stable';

    return {
      std,
      multiplier,
      variances,
      finalVariance,
      health,
    };
  }, [activation, fanIn, fanOut, layers, method]);

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
      <section className="grid gap-4 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-emerald-700">
            <SlidersHorizontal size={16} />
            Initialization controls
          </div>

          <div className="grid gap-2">
            {Object.entries(METHODS).map(([id, config]) => (
              <button
                key={id}
                type="button"
                onClick={() => setMethod(id)}
                className={`rounded-xl border px-4 py-3 text-left transition ${
                  method === id
                    ? 'border-emerald-500 bg-emerald-50 text-emerald-900'
                    : 'border-slate-200 bg-white text-slate-700 hover:border-slate-300'
                }`}
              >
                <div className="font-semibold">{config.label}</div>
                <div className="text-sm">{config.description}</div>
              </button>
            ))}
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            <label className="block text-sm font-semibold text-slate-700" htmlFor="init-fan-in">
              Fan-in {fanIn}
              <input
                id="init-fan-in"
                type="range"
                min="16"
                max="256"
                step="16"
                value={fanIn}
                onChange={(event) => setFanIn(Number(event.target.value))}
                className="mt-2 w-full accent-emerald-500"
              />
            </label>
            <label className="block text-sm font-semibold text-slate-700" htmlFor="init-fan-out">
              Fan-out {fanOut}
              <input
                id="init-fan-out"
                type="range"
                min="16"
                max="256"
                step="16"
                value={fanOut}
                onChange={(event) => setFanOut(Number(event.target.value))}
                className="mt-2 w-full accent-emerald-500"
              />
            </label>
          </div>

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="init-layers">
            Layers {layers}
          </label>
          <input
            id="init-layers"
            type="range"
            min="2"
            max="8"
            step="1"
            value={layers}
            onChange={(event) => setLayers(Number(event.target.value))}
            className="mt-2 w-full accent-emerald-500"
          />

          <div className="mt-5 grid grid-cols-2 rounded-xl border border-slate-200 bg-slate-50 p-1 text-sm font-semibold">
            {['relu', 'tanh'].map((option) => (
              <button
                key={option}
                type="button"
                onClick={() => setActivation(option)}
                className={`rounded-lg px-3 py-2 capitalize ${
                  activation === option ? 'bg-white text-emerald-800 shadow-sm' : 'text-slate-600'
                }`}
              >
                {option}
              </button>
            ))}
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-emerald-700">
            <Activity size={16} />
            Signal scale through depth
          </div>

          <div className="rounded-xl bg-slate-900 p-4 text-white">
            <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-emerald-200">
              <Zap size={14} />
              Weight standard deviation
            </div>
            <div className="mt-1 text-3xl font-bold">{signal.std.toFixed(3)}</div>
            <p className="mt-2 text-sm text-slate-300">
              Each layer multiplies variance by about {signal.multiplier.toFixed(2)}.
            </p>
          </div>

          <div className="mt-5 space-y-3">
            {signal.variances.map((layer) => (
              <div key={layer.id}>
                <div className="mb-1 flex items-center justify-between text-sm font-semibold text-slate-700">
                  <span>Layer {layer.id}</span>
                  <span>{layer.variance.toFixed(2)}x</span>
                </div>
                <div className="h-3 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className={`h-full rounded-full ${
                      signal.health === 'stable'
                        ? 'bg-emerald-500'
                        : signal.health === 'vanishing'
                          ? 'bg-sky-500'
                          : 'bg-rose-500'
                    }`}
                    style={{ width: layer.width }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="mt-5 flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
            <AlertTriangle className="mt-0.5 shrink-0" size={18} />
            <p>
              Current setup is {signal.health}. Good initialization keeps activations and gradients in a useful range before training starts.
            </p>
          </div>
        </div>
      </section>

      <AssessmentPanel lessonId="initialization" title="Initialization check" />
    </div>
  );
}
