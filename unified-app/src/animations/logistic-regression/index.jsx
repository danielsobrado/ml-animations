import React, { useMemo, useState } from 'react';
import { Activity, AlertTriangle, Gauge, RotateCcw, Sigma, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';
import {
  POINTS,
  PRESETS,
  boundaryLine,
  classifyPoint,
  metricPercent,
  safeRatio,
  scorePoint,
  summarize,
} from './logisticRegressionModel';

function Stat({ label, value, detail }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <p className="text-xs font-black uppercase tracking-wide text-slate-500">{label}</p>
      <strong className="mt-1 block text-2xl font-black text-slate-950">{value}</strong>
      <span className="text-sm text-slate-600">{detail}</span>
    </div>
  );
}

function ConfusionCell({ label, value, tone }) {
  const toneClass = {
    good: 'border-emerald-200 bg-emerald-50 text-emerald-800',
    warn: 'border-amber-200 bg-amber-50 text-amber-800',
    miss: 'border-rose-200 bg-rose-50 text-rose-800',
    quiet: 'border-slate-200 bg-slate-50 text-slate-700',
  }[tone];

  return (
    <div className={`rounded-lg border p-3 text-center ${toneClass}`}>
      <span className="block text-xs font-black uppercase tracking-wide">{label}</span>
      <strong className="mt-1 block text-2xl font-black">{value}</strong>
    </div>
  );
}

export default function LogisticRegressionAnimation() {
  const [weightRisk, setWeightRisk] = useState(PRESETS.balanced.weightRisk);
  const [weightEngagement, setWeightEngagement] = useState(PRESETS.balanced.weightEngagement);
  const [bias, setBias] = useState(PRESETS.balanced.bias);
  const [threshold, setThreshold] = useState(PRESETS.balanced.threshold);
  const [selectedId, setSelectedId] = useState('J');

  const scored = useMemo(
    () => POINTS.map((point) => classifyPoint(scorePoint(point, weightRisk, weightEngagement, bias), threshold)),
    [weightRisk, weightEngagement, bias, threshold],
  );
  const selected = scored.find((point) => point.id === selectedId) ?? scored[0];
  const counts = summarize(scored);
  const accuracy = safeRatio(counts.tp + counts.tn, scored.length);
  const precision = safeRatio(counts.tp, counts.tp + counts.fp);
  const recall = safeRatio(counts.tp, counts.tp + counts.fn);
  const boundary = boundaryLine(weightRisk, weightEngagement, bias, threshold);
  const nearThreshold = scored.filter((point) => Math.abs(point.probability - threshold) <= 0.08);
  const compressed = scored.filter((point) => point.probability > 0.4 && point.probability < 0.6).length;

  const applyPreset = (preset) => {
    setWeightRisk(preset.weightRisk);
    setWeightEngagement(preset.weightEngagement);
    setBias(preset.bias);
    setThreshold(preset.threshold);
  };

  const reset = () => applyPreset(PRESETS.balanced);

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Core classifier</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Logistic Regression</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              The model computes a linear logit, passes it through a sigmoid, then compares the probability with a
              decision threshold. Changing the threshold changes decisions without refitting the weights.
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
          Model controls
        </div>
        <div className="grid gap-4 xl:grid-cols-[1.2fr_1fr_1fr_1fr]">
          <div className="grid gap-2">
            <span className="text-sm font-bold text-slate-700">Preset</span>
            <div className="grid gap-2 sm:grid-cols-3 xl:grid-cols-1">
              {Object.entries(PRESETS).map(([id, preset]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => applyPreset(preset)}
                  className="ds-btn border border-[var(--ds-rule)] bg-[var(--ds-panel)] hover:bg-[var(--ds-accent-w)] hover:border-[var(--ds-accent)] text-[var(--ds-ink)] rounded p-3 text-left flex flex-col items-start transition-all duration-120"
                >
                  {preset.label}
                  <span className="mt-1 block text-xs font-semibold normal-case leading-4 text-[var(--ds-faint)]">{preset.detail}</span>
                </button>
              ))}
            </div>
          </div>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Risk weight: {weightRisk.toFixed(2)}
            <input
              min="-2"
              max="2.5"
              step="0.05"
              type="range"
              value={weightRisk}
              onChange={(event) => setWeightRisk(Number(event.target.value))}
            />
            <span className="text-xs font-semibold text-slate-500">Positive values make higher risk raise the probability.</span>
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Engagement weight: {weightEngagement.toFixed(2)}
            <input
              min="-2"
              max="2"
              step="0.05"
              type="range"
              value={weightEngagement}
              onChange={(event) => setWeightEngagement(Number(event.target.value))}
            />
            <span className="text-xs font-semibold text-slate-500">Negative values make higher engagement protective.</span>
          </label>
          <div className="grid gap-4">
            <label className="grid gap-2 text-sm font-bold text-slate-700">
              Bias: {bias.toFixed(2)}
              <input min="-2" max="2" step="0.05" type="range" value={bias} onChange={(event) => setBias(Number(event.target.value))} />
            </label>
            <label className="grid gap-2 text-sm font-bold text-slate-700">
              Threshold: {threshold.toFixed(2)}
              <input
                min="0.1"
                max="0.9"
                step="0.01"
                type="range"
                value={threshold}
                onChange={(event) => setThreshold(Number(event.target.value))}
              />
            </label>
          </div>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.35fr_0.85fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <Gauge size={16} />
            Decision surface
          </div>
          <svg viewBox="0 0 360 360" role="img" aria-label="Logistic regression decision boundary" className="h-auto w-full rounded-lg bg-slate-50">
            <defs>
              <linearGradient id="logisticBoundaryFill" x1="0" x2="1" y1="0" y2="1">
                <stop offset="0%" stopColor="#e0f2fe" />
                <stop offset="100%" stopColor="#ffe4e6" />
              </linearGradient>
            </defs>
            <rect x="24" y="24" width="312" height="312" rx="10" fill="url(#logisticBoundaryFill)" />
            {[25, 50, 75].map((value) => (
              <g key={value}>
                <line x1={24 + value * 3.12} x2={24 + value * 3.12} y1="24" y2="336" stroke="#cbd5e1" strokeDasharray="4 4" />
                <line x1="24" x2="336" y1={336 - value * 3.12} y2={336 - value * 3.12} stroke="#cbd5e1" strokeDasharray="4 4" />
              </g>
            ))}
            <line x1={boundary.x1} y1={boundary.y1} x2={boundary.x2} y2={boundary.y2} stroke="#0f172a" strokeWidth="4" strokeLinecap="round" />
            {scored.map((point) => {
              const x = 24 + point.risk * 3.12;
              const y = 336 - point.engagement * 3.12;
              const correct = point.y === point.predicted;
              const selectedPoint = point.id === selected.id;
              return (
                <g key={point.id} onClick={() => setSelectedId(point.id)} className="cursor-pointer">
                  <circle
                    cx={x}
                    cy={y}
                    r={selectedPoint ? 11 : 8}
                    fill={point.y ? '#e11d48' : '#0284c7'}
                    stroke={correct ? '#ffffff' : '#f59e0b'}
                    strokeWidth={correct ? 3 : 5}
                  />
                  <text x={x} y={y + 4} textAnchor="middle" fontSize="9" fontWeight="900" fill="#ffffff">
                    {point.id}
                  </text>
                </g>
              );
            })}
            <text x="180" y="352" textAnchor="middle" fontSize="12" fontWeight="800" fill="#475569">
              risk score
            </text>
            <text x="-180" y="14" textAnchor="middle" fontSize="12" fontWeight="800" fill="#475569" transform="rotate(-90)">
              engagement
            </text>
          </svg>
          <div className="mt-3 flex flex-wrap gap-3 text-xs font-bold text-slate-600">
            <span className="inline-flex items-center gap-2"><i className="h-3 w-3 rounded-full bg-sky-600" />actual negative</span>
            <span className="inline-flex items-center gap-2"><i className="h-3 w-3 rounded-full bg-rose-600" />actual positive</span>
            <span className="inline-flex items-center gap-2"><i className="h-3 w-3 rounded-full border-4 border-amber-500 bg-white" />mistake</span>
          </div>
        </div>

        <div className="space-y-4">
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <div className="mb-3 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
              <Sigma size={16} />
              Selected point {selected.id}
            </div>
            <div className="rounded-lg bg-slate-50 p-4 font-mono text-sm text-slate-800">
              z = ({weightRisk.toFixed(2)} * {((selected.risk - 50) / 18).toFixed(2)}) + ({weightEngagement.toFixed(2)} *{' '}
              {((selected.engagement - 50) / 18).toFixed(2)}) + {bias.toFixed(2)}
              <br />
              p = sigmoid({selected.z.toFixed(2)}) = {selected.probability.toFixed(2)}
            </div>
            <div className="mt-4 h-3 rounded-full bg-slate-100">
              <div className="h-3 rounded-full bg-cyan-600" style={{ width: `${selected.probability * 100}%` }} />
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-700">
              At threshold {threshold.toFixed(2)}, this point is predicted <strong>class {selected.predicted}</strong> and
              the true label is <strong>class {selected.y}</strong>.
            </p>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <div className="mb-3 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
              <Activity size={16} />
              Confusion matrix
            </div>
            <div className="grid grid-cols-2 gap-3">
              <ConfusionCell label="TP" value={counts.tp} tone="good" />
              <ConfusionCell label="FP" value={counts.fp} tone="warn" />
              <ConfusionCell label="FN" value={counts.fn} tone="miss" />
              <ConfusionCell label="TN" value={counts.tn} tone="quiet" />
            </div>
          </div>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <Stat label="Accuracy" value={metricPercent(accuracy)} detail="All correct decisions divided by all examples." />
        <Stat label="Precision" value={metricPercent(precision)} detail="How trustworthy the positive predictions are." />
        <Stat label="Recall" value={metricPercent(recall)} detail="How many actual positives were recovered." />
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-4">
          <p className="text-xs font-black uppercase tracking-wide text-cyan-700">Predict before running</p>
          <p className="mt-2 text-sm leading-6 text-cyan-950">
            Raise the threshold and predict whether false positives or false negatives will increase before checking the matrix.
          </p>
        </div>
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
          <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-amber-700">
            <AlertTriangle size={14} />
            Failure mode
          </p>
          <p className="mt-2 text-sm leading-6 text-amber-950">
            {compressed >= 8
              ? 'Many probabilities are compressed near 0.5, so small threshold moves can flip many uncertain cases.'
              : `${nearThreshold.length} cases sit close to the threshold and should be audited before deployment.`}
          </p>
        </div>
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4">
          <p className="text-xs font-black uppercase tracking-wide text-emerald-700">Practical rule</p>
          <p className="mt-2 text-sm leading-6 text-emerald-950">
            Fit weights on training data, tune thresholds on validation data, and reserve the test set for the final estimate.
          </p>
        </div>
      </section>

      <AssessmentPanel lessonId="logistic-regression" />
    </div>
  );
}
