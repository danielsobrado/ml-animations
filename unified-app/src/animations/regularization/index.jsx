import React, { useMemo, useState } from 'react';
import { AlertTriangle, BarChart3, RotateCcw, Scale, ShieldCheck, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const FEATURES = [
  { id: 'signalA', label: 'Signal A', base: 2.4, importance: 1.0, useful: true },
  { id: 'signalB', label: 'Signal B', base: -1.8, importance: 0.85, useful: true },
  { id: 'weakSignal', label: 'Weak signal', base: 0.8, importance: 0.45, useful: true },
  { id: 'noiseA', label: 'Noise A', base: 1.35, importance: 0.08, useful: false },
  { id: 'noiseB', label: 'Noise B', base: -1.1, importance: 0.05, useful: false },
  { id: 'noiseC', label: 'Noise C', base: 0.65, importance: 0.04, useful: false },
];

const PENALTIES = {
  none: {
    label: 'None',
    detail: 'Weights only answer the training loss, so noisy features can stay large.',
    l1: 0,
    l2: 0,
  },
  l2: {
    label: 'L2 / ridge',
    detail: 'Shrinks weights smoothly while usually keeping all features active.',
    l1: 0,
    l2: 1,
  },
  l1: {
    label: 'L1 / lasso',
    detail: 'Can drive weak or noisy weights exactly to zero.',
    l1: 1,
    l2: 0,
  },
  elastic: {
    label: 'Elastic net',
    detail: 'Combines sparse selection with smooth shrinkage.',
    l1: 0.55,
    l2: 0.45,
  },
};

function shrinkFeature(feature, penaltyId, lambda) {
  const penalty = PENALTIES[penaltyId];
  if (penaltyId === 'none' || lambda === 0) return { ...feature, weight: feature.base, removed: false };

  const l2Shrink = 1 / (1 + lambda * penalty.l2 * 2.1);
  const afterL2 = feature.base * l2Shrink;
  const l1Cut = lambda * penalty.l1 * (feature.useful ? 0.9 : 1.45);
  const sign = Math.sign(afterL2);
  const magnitude = Math.max(0, Math.abs(afterL2) - l1Cut);
  const weight = sign * magnitude;
  return { ...feature, weight, removed: Math.abs(weight) < 0.04 };
}

function lossProfile(weights, lambda, penaltyId) {
  const penalty = PENALTIES[penaltyId];
  const signalLoss = weights.reduce((sum, feature) => {
    const lostUsefulSignal = feature.useful ? Math.abs(feature.base - feature.weight) * feature.importance * 5.5 : 0;
    const noisyVariance = feature.useful ? 0 : Math.abs(feature.weight) * 6.5;
    return sum + lostUsefulSignal + noisyVariance;
  }, 15);
  const l1Penalty = weights.reduce((sum, feature) => sum + Math.abs(feature.weight), 0) * lambda * penalty.l1 * 2.5;
  const l2Penalty = weights.reduce((sum, feature) => sum + feature.weight ** 2, 0) * lambda * penalty.l2 * 1.25;
  const train = signalLoss + l1Penalty * 0.2 + l2Penalty * 0.2;
  const validation = signalLoss + weights.filter((feature) => !feature.useful && !feature.removed).length * 3.5 + Math.max(0, lambda - 0.55) * 18;
  return {
    dataLoss: signalLoss,
    penaltyLoss: l1Penalty + l2Penalty,
    train,
    validation,
    total: signalLoss + l1Penalty + l2Penalty,
  };
}

function sweepProfile(penaltyId) {
  return Array.from({ length: 11 }, (_, index) => {
    const lambda = index / 10;
    const weights = FEATURES.map((feature) => shrinkFeature(feature, penaltyId, lambda));
    const losses = lossProfile(weights, lambda, penaltyId);
    return { lambda, ...losses };
  });
}

function bestLambda(points) {
  return points.reduce((best, point) => (point.validation < best.validation ? point : best), points[0]);
}

function linePath(points, key) {
  const max = Math.max(...points.flatMap((point) => [point.train, point.validation, point.total]), 1);
  return points.map((point, index) => {
    const x = 28 + index * 30;
    const y = 168 - (point[key] / max) * 130;
    return `${index === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(' ');
}

function percent(value) {
  return `${Math.round(value * 100)}%`;
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

function WeightRow({ feature }) {
  const baseWidth = Math.min(100, Math.abs(feature.base) * 34);
  const weightWidth = Math.min(100, Math.abs(feature.weight) * 34);
  const tone = feature.useful ? 'bg-cyan-600' : 'bg-amber-500';
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
      <div className="flex items-center justify-between gap-3 text-sm">
        <strong className="text-slate-800">{feature.label}</strong>
        <span className={`rounded px-2 py-1 text-xs font-black ${feature.useful ? 'bg-cyan-100 text-cyan-700' : 'bg-amber-100 text-amber-700'}`}>
          {feature.useful ? 'signal' : 'noise'}
        </span>
      </div>
      <div className="mt-3 grid gap-2">
        <div>
          <div className="mb-1 flex justify-between text-xs font-bold text-slate-500">
            <span>unregularized</span>
            <span>{feature.base.toFixed(2)}</span>
          </div>
          <div className="h-2 rounded-full bg-slate-200">
            <div className="h-2 rounded-full bg-slate-400" style={{ width: `${baseWidth}%` }} />
          </div>
        </div>
        <div>
          <div className="mb-1 flex justify-between text-xs font-bold text-slate-500">
            <span>after penalty</span>
            <span>{feature.removed ? '0.00' : feature.weight.toFixed(2)}</span>
          </div>
          <div className="h-3 rounded-full bg-slate-200">
            <div className={`h-3 rounded-full ${feature.removed ? 'bg-slate-300' : tone}`} style={{ width: `${weightWidth}%` }} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default function RegularizationAnimation() {
  const [penaltyId, setPenaltyId] = useState('l2');
  const [lambda, setLambda] = useState(0.35);
  const [showValidationSweep, setShowValidationSweep] = useState(true);

  const weights = useMemo(() => FEATURES.map((feature) => shrinkFeature(feature, penaltyId, lambda)), [penaltyId, lambda]);
  const losses = lossProfile(weights, lambda, penaltyId);
  const sweep = useMemo(() => sweepProfile(penaltyId), [penaltyId]);
  const best = bestLambda(sweep);
  const removedCount = weights.filter((feature) => feature.removed).length;
  const noisyActive = weights.filter((feature) => !feature.useful && !feature.removed).length;
  const usefulMass = weights.filter((feature) => feature.useful).reduce((sum, feature) => sum + Math.abs(feature.weight), 0);
  const baseUsefulMass = FEATURES.filter((feature) => feature.useful).reduce((sum, feature) => sum + Math.abs(feature.base), 0);
  const usefulRetention = usefulMass / baseUsefulMass;
  const diagnosis = lambda < 0.15
    ? 'Too weak: noisy weights remain active and validation can suffer.'
    : lambda > 0.75
      ? 'Too strong: useful signal is being shrunk enough to underfit.'
      : noisyActive <= 1 && usefulRetention > 0.55
        ? 'Balanced: noisy weights are controlled while useful signal remains.'
        : 'Tradeoff zone: compare validation loss before increasing lambda.';

  const reset = () => {
    setPenaltyId('l2');
    setLambda(0.35);
    setShowValidationSweep(true);
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Generalization control</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Regularization</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Regularization adds a complexity penalty to the data loss. The goal is not to make weights small for its
              own sake; it is to keep complexity that earns its place on validation data.
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
          Penalty controls
        </div>
        <div className="grid gap-4 xl:grid-cols-[1.4fr_1fr_1fr]">
          <div className="grid gap-2">
            <span className="text-sm font-bold text-slate-700">Penalty family</span>
            <div className="grid gap-2 sm:grid-cols-2">
              {Object.entries(PENALTIES).map(([id, penalty]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setPenaltyId(id)}
                  className={`rounded-lg border px-3 py-2 text-left text-sm font-black transition ${
                    penaltyId === id ? 'border-cyan-500 bg-cyan-600 text-white' : 'border-slate-200 bg-slate-50 text-slate-700'
                  }`}
                >
                  {penalty.label}
                  <span className={`mt-1 block text-xs font-semibold normal-case leading-4 ${penaltyId === id ? 'text-cyan-50' : 'text-slate-500'}`}>
                    {penalty.detail}
                  </span>
                </button>
              ))}
            </div>
          </div>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Lambda: {lambda.toFixed(2)}
            <input min="0" max="1" step="0.01" type="range" value={lambda} onChange={(event) => setLambda(Number(event.target.value))} />
            <span className="text-xs font-semibold text-slate-500">
              Higher lambda makes the penalty matter more relative to fit on training rows.
            </span>
          </label>
          <label className="flex items-start gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm font-bold text-slate-700">
            <input
              type="checkbox"
              checked={showValidationSweep}
              onChange={(event) => setShowValidationSweep(event.target.checked)}
              className="mt-1"
            />
            <span>
              Show validation sweep
              <small className="mt-1 block font-semibold leading-5 text-slate-500">
                Use validation loss to tune lambda; keep the test set untouched.
              </small>
            </span>
          </label>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.15fr_0.95fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <BarChart3 size={16} />
            Weight shrinkage
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            {weights.map((feature) => (
              <WeightRow key={feature.id} feature={feature} />
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
              <Scale size={16} />
              Loss decomposition
            </div>
            <div className="rounded-lg bg-slate-50 p-4 font-mono text-sm text-slate-800">
              total = data loss + lambda * penalty
              <br />
              {losses.total.toFixed(1)} = {losses.dataLoss.toFixed(1)} + {losses.penaltyLoss.toFixed(1)}
            </div>
            <div className="mt-4 grid gap-3">
              <div>
                <div className="mb-1 flex justify-between text-sm font-bold text-slate-700">
                  <span>Data loss</span>
                  <span>{losses.dataLoss.toFixed(1)}</span>
                </div>
                <div className="h-3 rounded-full bg-slate-100">
                  <div className="h-3 rounded-full bg-sky-600" style={{ width: `${Math.min(100, losses.dataLoss * 2)}%` }} />
                </div>
              </div>
              <div>
                <div className="mb-1 flex justify-between text-sm font-bold text-slate-700">
                  <span>Penalty</span>
                  <span>{losses.penaltyLoss.toFixed(1)}</span>
                </div>
                <div className="h-3 rounded-full bg-slate-100">
                  <div className="h-3 rounded-full bg-emerald-600" style={{ width: `${Math.min(100, losses.penaltyLoss * 4)}%` }} />
                </div>
              </div>
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <div className="mb-3 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
              <BarChart3 size={16} />
              Lambda sweep
            </div>
            <svg viewBox="0 0 360 205" role="img" aria-label="Training and validation loss over lambda" className="h-auto w-full rounded-lg bg-slate-50">
              <rect x="28" y="28" width="300" height="140" rx="8" fill="#ffffff" stroke="#cbd5e1" />
              {[0, 0.5, 1].map((mark) => (
                <line key={mark} x1={28 + mark * 300} x2={28 + mark * 300} y1="28" y2="168" stroke="#e2e8f0" />
              ))}
              <path d={linePath(sweep, 'train')} fill="none" stroke="#0284c7" strokeWidth="4" strokeLinecap="round" />
              {showValidationSweep && <path d={linePath(sweep, 'validation')} fill="none" stroke="#e11d48" strokeWidth="4" strokeLinecap="round" />}
              <line x1={28 + lambda * 300} x2={28 + lambda * 300} y1="24" y2="172" stroke="#0f172a" strokeWidth="3" />
              {showValidationSweep && (
                <circle cx={28 + best.lambda * 300} cy={168 - (best.validation / Math.max(...sweep.flatMap((point) => [point.train, point.validation, point.total]))) * 130} r="7" fill="#10b981" stroke="#ffffff" strokeWidth="3" />
              )}
              <text x="178" y="195" textAnchor="middle" fontSize="12" fontWeight="800" fill="#475569">
                lambda
              </text>
            </svg>
            <div className="mt-3 grid gap-2 text-sm font-bold text-slate-700">
              <span className="inline-flex items-center gap-2"><i className="h-1 w-8 rounded bg-sky-600" />training loss</span>
              <span className="inline-flex items-center gap-2"><i className="h-1 w-8 rounded bg-rose-600" />validation loss</span>
              <span className="inline-flex items-center gap-2"><i className="h-3 w-3 rounded-full bg-emerald-500" />best lambda: {best.lambda.toFixed(1)}</span>
            </div>
          </div>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <Stat label="Zeroed weights" value={removedCount} detail="Weights driven close enough to zero to remove." />
        <Stat label="Useful signal kept" value={percent(usefulRetention)} detail="Remaining magnitude on true signal features." />
        <Stat label="Diagnosis" value={lambda < 0.15 ? 'Weak' : lambda > 0.75 ? 'Strong' : 'Tuned'} detail={diagnosis} />
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-4">
          <p className="text-xs font-black uppercase tracking-wide text-cyan-700">Predict before running</p>
          <p className="mt-2 text-sm leading-6 text-cyan-950">
            Switch from L2 to L1 and predict which noisy weights will disappear first as lambda rises.
          </p>
        </div>
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
          <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-amber-700">
            <AlertTriangle size={14} />
            Failure mode
          </p>
          <p className="mt-2 text-sm leading-6 text-amber-950">
            Stronger regularization is not automatically better. Once useful signal shrinks, validation loss can rise.
          </p>
        </div>
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4">
          <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-emerald-700">
            <ShieldCheck size={14} />
            Practical rule
          </p>
          <p className="mt-2 text-sm leading-6 text-emerald-950">
            Tune lambda on validation data, compare simple baselines, and report final performance on untouched test data.
          </p>
        </div>
      </section>

      <AssessmentPanel lessonId="regularization" />
    </div>
  );
}
