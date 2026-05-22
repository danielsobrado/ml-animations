import React, { useMemo, useState } from 'react';
import { Activity, GitMerge, Layers, Route, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const COMPONENTS = [
  { id: 'embedding', label: 'Token embedding', base: [0.9, 0.2, 0.1, 0.4], color: '#0f766e' },
  { id: 'attn1', label: 'Attention write 1', base: [0.2, 0.7, -0.1, 0.1], color: '#2563eb' },
  { id: 'mlp1', label: 'MLP write 1', base: [-0.1, 0.2, 0.8, 0.2], color: '#7c3aed' },
  { id: 'attn2', label: 'Attention write 2', base: [0.1, -0.3, 0.1, 0.7], color: '#db2777' },
  { id: 'mlp2', label: 'MLP write 2', base: [0.3, 0.1, 0.4, -0.2], color: '#ea580c' },
];

const FEATURE_LABELS = ['subject', 'relation', 'syntax', 'prediction'];

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const add = (left, right) => left.map((value, index) => value + right[index]);
const scale = (vector, amount) => vector.map((value) => value * amount);
const norm = (vector) => Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
const normalize = (vector) => {
  const magnitude = norm(vector);
  return magnitude === 0 ? vector : vector.map((value) => value / magnitude);
};

function Bar({ value, color }) {
  const width = `${Math.abs(value) * 100}%`;
  return (
    <div className="flex items-center gap-2">
      <div className="w-20 text-xs font-semibold text-slate-500">{value.toFixed(2)}</div>
      <div className="relative h-3 flex-1 rounded-full bg-slate-100">
        <div
          className={`absolute top-0 h-3 rounded-full ${value >= 0 ? 'left-1/2' : 'right-1/2'}`}
          style={{ width, backgroundColor: color }}
        />
        <div className="absolute left-1/2 top-[-2px] h-5 w-px bg-slate-300" />
      </div>
    </div>
  );
}

function Metric({ icon: Icon, label, value, helper }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
        <Icon size={16} />
        {label}
      </div>
      <div className="mt-2 text-2xl font-bold text-slate-950">{value}</div>
      <p className="mt-1 text-sm text-slate-600">{helper}</p>
    </div>
  );
}

export default function ResidualStreamAnimation() {
  const [weights, setWeights] = useState({
    embedding: 1,
    attn1: 0.7,
    mlp1: 0.55,
    attn2: 0.4,
    mlp2: 0.35,
  });
  const [normalizeAfterWrite, setNormalizeAfterWrite] = useState(false);
  const [selectedStep, setSelectedStep] = useState(3);

  const steps = useMemo(() => {
    let stream = [0, 0, 0, 0];
    return COMPONENTS.map((component) => {
      const write = scale(component.base, weights[component.id]);
      const before = stream;
      const summed = add(stream, write);
      stream = normalizeAfterWrite ? normalize(summed) : summed;
      return {
        ...component,
        write,
        before,
        after: stream,
        magnitude: norm(write),
      };
    });
  }, [normalizeAfterWrite, weights]);

  const selected = steps[selectedStep];
  const finalStream = steps[steps.length - 1].after;
  const totalWrite = steps.reduce((sumValue, step) => sumValue + step.magnitude, 0);
  const finalMagnitude = norm(finalStream);
  const dominantFeature = FEATURE_LABELS[finalStream.reduce((best, value, index, list) => (Math.abs(value) > Math.abs(list[best]) ? index : best), 0)];

  return (
    <div className="min-h-full bg-slate-50">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 p-4 md:p-6">
        <header className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
            <div>
              <div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-blue-700">
                <GitMerge size={17} />
                Transformer layer flow
              </div>
              <h1 className="mt-2 text-2xl font-bold text-slate-950 md:text-3xl">Residual stream</h1>
              <p className="mt-2 max-w-3xl text-slate-700">
                A transformer layer usually adds new attention and MLP writes into a running vector. The residual stream
                is the shared workspace where token identity, context, syntax, and prediction features accumulate.
              </p>
            </div>
            <label className="flex items-center gap-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-semibold text-slate-700">
              <input
                type="checkbox"
                checked={normalizeAfterWrite}
                onChange={(event) => setNormalizeAfterWrite(event.target.checked)}
                className="h-4 w-4 accent-blue-700"
              />
              Normalize after each write
            </label>
          </div>
        </header>

        <section className="grid gap-4 lg:grid-cols-[320px_1fr]">
          <aside className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex items-center gap-2 font-semibold text-slate-950">
              <SlidersHorizontal size={18} />
              Write strengths
            </div>
            <div className="mt-5 space-y-4">
              {COMPONENTS.map((component, index) => (
                <label key={component.id} className="block">
                  <div className="mb-2 flex items-center justify-between text-sm font-semibold text-slate-700">
                    <button
                      type="button"
                      onClick={() => setSelectedStep(index)}
                      className={`text-left ${selectedStep === index ? 'text-blue-700' : ''}`}
                    >
                      {component.label}
                    </button>
                    <span>{weights[component.id].toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1.5"
                    step="0.05"
                    value={weights[component.id]}
                    onChange={(event) =>
                      setWeights((current) => ({
                        ...current,
                        [component.id]: Number(event.target.value),
                      }))
                    }
                    className="w-full accent-blue-700"
                  />
                </label>
              ))}
            </div>
          </aside>

          <main className="space-y-4">
            <div className="grid gap-4 md:grid-cols-4">
              <Metric icon={Activity} label="Final magnitude" value={finalMagnitude.toFixed(2)} helper="How large the accumulated stream is." />
              <Metric icon={Route} label="Total write load" value={totalWrite.toFixed(2)} helper="Sum of component write magnitudes." />
              <Metric icon={Layers} label="Dominant feature" value={dominantFeature} helper="Largest final feature dimension." />
              <Metric icon={GitMerge} label="Update rule" value="add" helper="Layer output is added, not substituted." />
            </div>

            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <h2 className="text-lg font-bold text-slate-950">Layer contribution ledger</h2>
              <p className="text-sm text-slate-600">
                Each component writes a vector into the same running stream. Click a row to inspect the before/write/after state.
              </p>
              <div className="mt-4 space-y-2">
                {steps.map((step, index) => (
                  <button
                    key={step.id}
                    type="button"
                    onClick={() => setSelectedStep(index)}
                    className={`grid w-full gap-3 rounded-lg border p-3 text-left md:grid-cols-[160px_1fr] ${
                      selectedStep === index ? 'border-blue-500 bg-blue-50' : 'border-slate-200 bg-white hover:border-blue-300'
                    }`}
                  >
                    <div>
                      <div className="font-bold text-slate-950">{step.label}</div>
                      <div className="text-xs text-slate-500">write magnitude {step.magnitude.toFixed(2)}</div>
                    </div>
                    <div className="grid gap-2 md:grid-cols-4">
                      {FEATURE_LABELS.map((feature, featureIndex) => (
                        <div key={feature} className="rounded-md bg-slate-50 p-2">
                          <div className="mb-1 text-xs font-semibold text-slate-500">{feature}</div>
                          <Bar value={step.after[featureIndex]} color={step.color} />
                        </div>
                      ))}
                    </div>
                  </button>
                ))}
              </div>
            </section>

            <section className="grid gap-4 md:grid-cols-3">
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">Before</h3>
                <div className="mt-3 space-y-2">
                  {selected.before.map((value, index) => (
                    <Bar key={FEATURE_LABELS[index]} value={value} color="#64748b" />
                  ))}
                </div>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">Write: {selected.label}</h3>
                <div className="mt-3 space-y-2">
                  {selected.write.map((value, index) => (
                    <Bar key={FEATURE_LABELS[index]} value={value} color={selected.color} />
                  ))}
                </div>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">After</h3>
                <div className="mt-3 space-y-2">
                  {selected.after.map((value, index) => (
                    <Bar key={FEATURE_LABELS[index]} value={value} color="#1d4ed8" />
                  ))}
                </div>
              </div>
            </section>

            <section className="grid gap-4 md:grid-cols-3">
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">Predict before running</h3>
                <p className="mt-2 text-sm text-slate-700">
                  Increase an early write and watch later components add on top of it. Residual flow preserves a path for
                  previous information instead of forcing every layer to rewrite everything.
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">Failure mode</h3>
                <p className="mt-2 text-sm text-slate-700">
                  Very large writes can dominate the stream. Normalization and learned projections help keep features
                  usable across many layers.
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">Mistake to avoid</h3>
                <p className="mt-2 text-sm text-slate-700">
                  The residual stream is not a separate memory bank. It is the current token representation being updated
                  by attention and MLP blocks.
                </p>
              </div>
            </section>
          </main>
        </section>

        <AssessmentPanel lessonId="residual-stream" />
      </div>
    </div>
  );
}
