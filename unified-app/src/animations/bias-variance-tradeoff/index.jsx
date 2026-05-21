import React, { useMemo, useState } from 'react';
import { Activity, RotateCcw, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const SAMPLE_LEVELS = {
  small: { label: 'Small sample', count: 10, varianceRelief: 0.2, detail: 'Few examples make flexible models unstable.' },
  medium: { label: 'Medium sample', count: 22, varianceRelief: 0.45, detail: 'More examples reduce the penalty for flexibility.' },
  large: { label: 'Large sample', count: 42, varianceRelief: 0.7, detail: 'Many examples make variance easier to control.' },
};

const MODEL_TYPES = {
  simple: { label: 'Simple', complexity: 1, detail: 'High bias: the model misses curved signal.' },
  balanced: { label: 'Balanced', complexity: 2, detail: 'Lower bias without chasing most sample noise.' },
  flexible: { label: 'Flexible', complexity: 3, detail: 'Low bias, but higher variance when data is noisy or scarce.' },
};

function truth(x) {
  return 42 + 28 * Math.sin((x - 8) / 11) + 0.72 * x;
}

function pseudoNoise(index) {
  const raw = Math.sin(index * 12.9898) * 43758.5453;
  return raw - Math.floor(raw);
}

function makePoints(sampleLevel, noise) {
  const { count } = SAMPLE_LEVELS[sampleLevel];
  return Array.from({ length: count }, (_, index) => {
    const x = 4 + (index / Math.max(1, count - 1)) * 92;
    const centeredNoise = pseudoNoise(index + count * 3) - 0.5;
    return {
      id: index,
      x,
      y: truth(x) + centeredNoise * noise * 30,
    };
  });
}

function predict(x, model, noise) {
  if (model === 'simple') {
    return 51 + 0.36 * x;
  }
  if (model === 'balanced') {
    return 42 + 23 * Math.sin((x - 8) / 13) + 0.58 * x;
  }
  return truth(x) + Math.sin(x * 0.55) * noise * 8 + Math.sin(x * 1.3) * noise * 3;
}

function errorProfile(model, sampleLevel, noise) {
  const complexity = MODEL_TYPES[model].complexity;
  const varianceRelief = SAMPLE_LEVELS[sampleLevel].varianceRelief;
  const bias = model === 'simple' ? 34 : model === 'balanced' ? 13 : 6;
  const variance = model === 'simple'
    ? 5 + noise * 2
    : model === 'balanced'
      ? 9 + noise * 8 - varianceRelief * 5
      : 18 + noise * 20 - varianceRelief * 14;
  const irreducible = 8 + noise * 10;
  const train = Math.max(6, bias * 0.55 + variance * 0.22 + irreducible * 0.5 - complexity * 5);
  const validation = Math.max(8, bias + variance + irreducible);
  return {
    bias,
    variance: Math.max(4, variance),
    irreducible,
    train,
    validation,
    gap: Math.max(0, validation - train),
  };
}

function project(point) {
  return {
    cx: 34 + (point.x / 100) * 332,
    cy: 262 - ((point.y - 15) / 95) * 226,
  };
}

function curvePath(model, noise, source = predict) {
  return Array.from({ length: 70 }, (_, index) => {
    const x = (index / 69) * 100;
    const y = source === truth ? truth(x) : predict(x, model, noise);
    const { cx, cy } = project({ x, y });
    return `${index === 0 ? 'M' : 'L'} ${cx.toFixed(1)} ${cy.toFixed(1)}`;
  }).join(' ');
}

function Stat({ label, value, detail }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <p className="text-xs font-black uppercase tracking-wide text-slate-500">{label}</p>
      <strong className="mt-1 block text-2xl font-black text-slate-950">{value}</strong>
      <span className="text-sm text-slate-600">{detail}</span>
    </div>
  );
}

function ErrorBar({ label, value, color }) {
  const width = Math.min(100, Math.max(8, value));
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-sm font-bold text-slate-700">
        <span>{label}</span>
        <span>{value.toFixed(1)}</span>
      </div>
      <div className="h-3 overflow-hidden rounded-full bg-slate-100">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${width}%` }} />
      </div>
    </div>
  );
}

export default function BiasVarianceTradeoffAnimation() {
  const [model, setModel] = useState('balanced');
  const [sampleLevel, setSampleLevel] = useState('medium');
  const [noise, setNoise] = useState(0.45);

  const points = useMemo(() => makePoints(sampleLevel, noise), [sampleLevel, noise]);
  const profile = errorProfile(model, sampleLevel, noise);
  const recommendation = profile.bias > profile.variance
    ? 'The model is underfitting: add useful flexibility or better features.'
    : profile.variance > profile.bias + 10
      ? 'The model is variance-heavy: add data, regularize, simplify, or use averaging.'
      : 'Bias and variance are reasonably balanced for this teaching setup.';

  const reset = () => {
    setModel('balanced');
    setSampleLevel('medium');
    setNoise(0.45);
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Generalization tradeoff</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Bias-Variance Tradeoff</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Bias is error from an overly simple assumption. Variance is error from a model that changes too much with
              the sampled training data. Tune complexity, noise, and sample size to find where validation error is lowest.
            </p>
          </div>
          <button
            type="button"
            onClick={reset}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-bold text-slate-800"
          >
            <RotateCcw size={16} />
            Reset
          </button>
        </div>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <SlidersHorizontal size={16} />
          Tradeoff controls
        </div>
        <div className="grid gap-4 lg:grid-cols-[1.2fr_1.2fr_1fr]">
          <div className="grid gap-2">
            <span className="text-sm font-bold text-slate-700">Model complexity</span>
            <div className="grid grid-cols-3 gap-2">
              {Object.entries(MODEL_TYPES).map(([id, config]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setModel(id)}
                  className={`rounded-lg border px-3 py-2 text-sm font-black transition ${model === id ? 'border-cyan-500 bg-cyan-600 text-white' : 'border-slate-200 bg-slate-50 text-slate-700'}`}
                >
                  {config.label}
                </button>
              ))}
            </div>
          </div>
          <div className="grid gap-2">
            <span className="text-sm font-bold text-slate-700">Training sample</span>
            <div className="grid grid-cols-3 gap-2">
              {Object.entries(SAMPLE_LEVELS).map(([id, config]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setSampleLevel(id)}
                  className={`rounded-lg border px-3 py-2 text-sm font-black transition ${sampleLevel === id ? 'border-emerald-500 bg-emerald-600 text-white' : 'border-slate-200 bg-slate-50 text-slate-700'}`}
                >
                  {config.label}
                </button>
              ))}
            </div>
          </div>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Data noise: {noise.toFixed(2)}
            <input min="0" max="1" step="0.05" type="range" value={noise} onChange={(event) => setNoise(Number(event.target.value))} />
          </label>
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          <p className="rounded-lg bg-slate-50 p-3 text-sm leading-6 text-slate-700">
            <strong className="text-slate-950">{MODEL_TYPES[model].label} model:</strong> {MODEL_TYPES[model].detail}
          </p>
          <p className="rounded-lg bg-slate-50 p-3 text-sm leading-6 text-slate-700">
            <strong className="text-slate-950">{SAMPLE_LEVELS[sampleLevel].label}:</strong> {SAMPLE_LEVELS[sampleLevel].detail}
          </p>
        </div>
      </section>

      <div className="grid gap-3 md:grid-cols-4">
        <Stat label="Bias" value={profile.bias.toFixed(1)} detail="missed signal" />
        <Stat label="Variance" value={profile.variance.toFixed(1)} detail="sample sensitivity" />
        <Stat label="Train error" value={profile.train.toFixed(1)} detail="fit on seen data" />
        <Stat label="Validation error" value={profile.validation.toFixed(1)} detail="generalization estimate" />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <Activity size={16} />
            Signal, sample, and fitted curve
          </h3>
          <svg viewBox="0 0 400 300" className="mt-4 h-auto w-full rounded-lg border border-slate-200 bg-slate-50" role="img" aria-label="Bias variance fitted curve">
            <line x1="34" y1="262" x2="366" y2="262" stroke="#cbd5e1" />
            <line x1="34" y1="36" x2="34" y2="262" stroke="#cbd5e1" />
            <path d={curvePath(model, noise, truth)} fill="none" stroke="#94a3b8" strokeWidth="3" strokeDasharray="6 6" />
            <path d={curvePath(model, noise)} fill="none" stroke="#0891b2" strokeWidth="4" />
            {points.map((point) => {
              const { cx, cy } = project(point);
              return <circle key={point.id} cx={cx} cy={cy} r="5" fill="#f97316" stroke="#fff" strokeWidth="2" />;
            })}
            <text x="200" y="286" textAnchor="middle" className="fill-slate-600 text-xs font-bold">feature value</text>
            <text x="14" y="150" textAnchor="middle" transform="rotate(-90 14 150)" className="fill-slate-600 text-xs font-bold">target</text>
            <text x="268" y="56" className="fill-slate-500 text-xs font-bold">true signal</text>
            <text x="268" y="78" className="fill-cyan-700 text-xs font-bold">model fit</text>
          </svg>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Error decomposition</h3>
          <div className="mt-5 space-y-4">
            <ErrorBar label="Bias component" value={profile.bias} color="bg-amber-500" />
            <ErrorBar label="Variance component" value={profile.variance} color="bg-cyan-500" />
            <ErrorBar label="Irreducible noise" value={profile.irreducible} color="bg-slate-400" />
            <ErrorBar label="Train-validation gap" value={profile.gap} color="bg-rose-500" />
          </div>
          <p className="mt-5 rounded-lg border border-cyan-200 bg-cyan-50 p-4 text-sm leading-6 text-cyan-950">
            {recommendation}
          </p>
        </section>
      </div>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-amber-700">Problem solved</h3>
          <p className="mt-3 text-sm leading-6 text-amber-950">
            Bias-variance explains whether validation error comes from a model that is too rigid or too sensitive to one
            training sample.
          </p>
        </div>
        <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-cyan-700">Mistake to avoid</h3>
          <p className="mt-3 text-sm leading-6 text-cyan-950">
            Do not call every generalization failure overfitting; high bias can make both train and validation error bad.
          </p>
        </div>
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-emerald-700">Understanding check</h3>
          <p className="mt-3 text-sm leading-6 text-emerald-950">
            Switch from simple to flexible, then explain why train error can drop while validation error rises.
          </p>
        </div>
      </section>

      <AssessmentPanel lessonId="bias-variance-tradeoff" title="Bias-Variance Tradeoff check" />
    </div>
  );
}
