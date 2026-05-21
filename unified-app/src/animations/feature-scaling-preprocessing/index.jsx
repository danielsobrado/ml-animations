import React, { useMemo, useState } from 'react';
import { RotateCcw, Scale, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const BASE_POINTS = [
  { id: 'A', age: 23, income: 38000, split: 'train', label: 'student' },
  { id: 'B', age: 31, income: 52000, split: 'train', label: 'early career' },
  { id: 'C', age: 42, income: 76000, split: 'train', label: 'manager' },
  { id: 'D', age: 53, income: 94000, split: 'train', label: 'senior' },
  { id: 'E', age: 61, income: 112000, split: 'validation', label: 'director' },
  { id: 'F', age: 35, income: 58000, split: 'validation', label: 'analyst' },
];

const OUTLIER = { id: 'G', age: 64, income: 235000, split: 'validation', label: 'outlier' };

const METHODS = {
  raw: {
    label: 'Raw',
    formula: 'x',
    detail: 'No scaling: income units dominate distance and gradient magnitudes.',
  },
  standard: {
    label: 'Standardize',
    formula: '(x - mean) / std',
    detail: 'Centers each feature and measures values in standard deviations.',
  },
  minmax: {
    label: 'Min-max',
    formula: '(x - min) / (max - min)',
    detail: 'Compresses features into a 0-to-1 range using fitted endpoints.',
  },
  robust: {
    label: 'Robust',
    formula: '(x - median) / IQR',
    detail: 'Uses median and interquartile range, so one outlier has less leverage.',
  },
};

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function percentile(values, p) {
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * p;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

function stats(values) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const q1 = percentile(values, 0.25);
  const q3 = percentile(values, 0.75);
  return {
    mean,
    std: Math.sqrt(variance) || 1,
    min: Math.min(...values),
    max: Math.max(...values),
    median: median(values),
    iqr: q3 - q1 || 1,
  };
}

function fitScaler(points, fitOnAllData) {
  const fitPoints = fitOnAllData ? points : points.filter((point) => point.split === 'train');
  return {
    age: stats(fitPoints.map((point) => point.age)),
    income: stats(fitPoints.map((point) => point.income)),
  };
}

function transformValue(value, featureStats, method) {
  if (method === 'standard') return (value - featureStats.mean) / featureStats.std;
  if (method === 'minmax') return (value - featureStats.min) / (featureStats.max - featureStats.min || 1);
  if (method === 'robust') return (value - featureStats.median) / featureStats.iqr;
  return value;
}

function transformPoint(point, scaler, method) {
  return {
    ...point,
    x: transformValue(point.age, scaler.age, method),
    y: transformValue(point.income, scaler.income, method),
  };
}

function distance(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function bounds(points) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
  };
}

function project(point, box) {
  const pad = 34;
  const xRange = box.maxX - box.minX || 1;
  const yRange = box.maxY - box.minY || 1;
  return {
    cx: pad + ((point.x - box.minX) / xRange) * (360 - pad * 2),
    cy: 260 - pad - ((point.y - box.minY) / yRange) * (260 - pad * 2),
  };
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

export default function FeatureScalingPreprocessingAnimation() {
  const [method, setMethod] = useState('standard');
  const [includeOutlier, setIncludeOutlier] = useState(true);
  const [fitOnAllData, setFitOnAllData] = useState(false);
  const [selectedPoint, setSelectedPoint] = useState('F');

  const points = useMemo(
    () => (includeOutlier ? [...BASE_POINTS, OUTLIER] : BASE_POINTS),
    [includeOutlier],
  );
  const scaler = useMemo(() => fitScaler(points, fitOnAllData), [points, fitOnAllData]);
  const transformed = useMemo(
    () => points.map((point) => transformPoint(point, scaler, method)),
    [points, scaler, method],
  );
  const rawPoints = useMemo(
    () => points.map((point) => ({ ...point, x: point.age, y: point.income })),
    [points],
  );
  const selected = transformed.find((point) => point.id === selectedPoint) || transformed[0];
  const anchor = transformed.find((point) => point.id === 'B');
  const rawSelected = rawPoints.find((point) => point.id === selected.id);
  const rawAnchor = rawPoints.find((point) => point.id === 'B');
  const scaledDistance = distance(selected, anchor);
  const rawDistance = distance(rawSelected, rawAnchor);
  const plotBox = bounds(transformed);

  const reset = () => {
    setMethod('standard');
    setIncludeOutlier(true);
    setFitOnAllData(false);
    setSelectedPoint('F');
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Data preparation</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Feature Scaling and Preprocessing</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Preprocessing turns raw columns into model-ready features. Fit the scaler on training data, transform
              validation data with the same parameters, and watch how units and outliers change distances.
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
          Preprocessing controls
        </div>
        <div className="grid gap-4 lg:grid-cols-[1.5fr_1fr_1fr_1fr]">
          <div className="grid gap-2">
            <span className="text-sm font-bold text-slate-700">Scaler</span>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {Object.entries(METHODS).map(([id, config]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setMethod(id)}
                  className={`rounded-lg border px-3 py-2 text-sm font-black transition ${method === id ? 'border-cyan-500 bg-cyan-600 text-white' : 'border-slate-200 bg-slate-50 text-slate-700'}`}
                >
                  {config.label}
                </button>
              ))}
            </div>
          </div>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Compare point
            <select value={selectedPoint} onChange={(event) => setSelectedPoint(event.target.value)} className="rounded-lg border border-slate-300 bg-white px-3 py-2">
              {points.filter((point) => point.id !== 'B').map((point) => (
                <option key={point.id} value={point.id}>{point.id}: {point.label}</option>
              ))}
            </select>
          </label>
          <label className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-bold text-slate-700">
            Include outlier
            <input type="checkbox" checked={includeOutlier} onChange={(event) => setIncludeOutlier(event.target.checked)} />
          </label>
          <label className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-bold text-slate-700">
            Fit on all data
            <input type="checkbox" checked={fitOnAllData} onChange={(event) => setFitOnAllData(event.target.checked)} />
          </label>
        </div>
        <p className="mt-4 rounded-lg bg-slate-50 p-3 text-sm leading-6 text-slate-700">
          <strong className="text-slate-950">{METHODS[method].label}:</strong> {METHODS[method].detail}
          {' '}Formula: <span className="font-mono">{METHODS[method].formula}</span>.
        </p>
      </section>

      <div className="grid gap-3 md:grid-cols-4">
        <Stat label="Raw distance" value={rawDistance.toFixed(0)} detail="age and income units mixed" />
        <Stat label="Scaled distance" value={scaledDistance.toFixed(2)} detail={`B to ${selected.id}`} />
        <Stat label="Income mean" value={`$${Math.round(scaler.income.mean / 1000)}k`} detail={fitOnAllData ? 'fit using train + validation' : 'fit using train only'} />
        <Stat label="Income spread" value={`$${Math.round(scaler.income.std / 1000)}k`} detail="standard deviation used by scaler" />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <Scale size={16} />
            Transformed feature space
          </h3>
          <svg viewBox="0 0 360 260" className="mt-4 h-auto w-full rounded-lg border border-slate-200 bg-slate-50" role="img" aria-label="Scaled feature scatter plot">
            <line x1="34" y1="226" x2="330" y2="226" stroke="#cbd5e1" />
            <line x1="34" y1="34" x2="34" y2="226" stroke="#cbd5e1" />
            {transformed.map((point) => {
              const { cx, cy } = project(point, plotBox);
              const isSelected = point.id === selected.id || point.id === 'B';
              return (
                <g key={point.id}>
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isSelected ? 10 : 7}
                    fill={point.split === 'train' ? '#2563eb' : '#f97316'}
                    stroke={point.id === OUTLIER.id ? '#be123c' : '#ffffff'}
                    strokeWidth="3"
                  />
                  <text x={cx + 12} y={cy + 4} className="fill-slate-700 text-xs font-black">{point.id}</text>
                </g>
              );
            })}
            <text x="175" y="250" textAnchor="middle" className="fill-slate-600 text-xs font-bold">scaled age</text>
            <text x="14" y="140" textAnchor="middle" transform="rotate(-90 14 140)" className="fill-slate-600 text-xs font-bold">scaled income</text>
          </svg>
          <div className="mt-3 flex flex-wrap gap-3 text-xs font-bold text-slate-600">
            <span className="inline-flex items-center gap-2"><span className="h-3 w-3 rounded-full bg-blue-600" /> Train</span>
            <span className="inline-flex items-center gap-2"><span className="h-3 w-3 rounded-full bg-orange-500" /> Validation</span>
          </div>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Fit parameters</h3>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[520px] text-left text-sm">
              <thead className="text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="py-2">Feature</th>
                  <th className="py-2">Mean</th>
                  <th className="py-2">Std</th>
                  <th className="py-2">Min</th>
                  <th className="py-2">Max</th>
                  <th className="py-2">Median</th>
                  <th className="py-2">IQR</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {Object.entries(scaler).map(([feature, values]) => (
                  <tr key={feature}>
                    <td className="py-3 font-black text-slate-950">{feature}</td>
                    <td className="py-3">{feature === 'income' ? `$${Math.round(values.mean / 1000)}k` : values.mean.toFixed(1)}</td>
                    <td className="py-3">{feature === 'income' ? `$${Math.round(values.std / 1000)}k` : values.std.toFixed(1)}</td>
                    <td className="py-3">{feature === 'income' ? `$${Math.round(values.min / 1000)}k` : values.min}</td>
                    <td className="py-3">{feature === 'income' ? `$${Math.round(values.max / 1000)}k` : values.max}</td>
                    <td className="py-3">{feature === 'income' ? `$${Math.round(values.median / 1000)}k` : values.median.toFixed(1)}</td>
                    <td className="py-3">{feature === 'income' ? `$${Math.round(values.iqr / 1000)}k` : values.iqr.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className={`mt-4 rounded-lg border p-4 ${fitOnAllData ? 'border-rose-200 bg-rose-50 text-rose-950' : 'border-emerald-200 bg-emerald-50 text-emerald-950'}`}>
            <strong className="text-sm">{fitOnAllData ? 'Leakage warning' : 'Leakage-safe fit'}</strong>
            <p className="mt-2 text-sm leading-6">
              {fitOnAllData
                ? 'Validation values are shaping preprocessing parameters. That makes validation feedback too optimistic.'
                : 'Fit statistics come from training rows only, then validation rows reuse those training parameters.'}
            </p>
          </div>
        </section>
      </div>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-cyan-700">Problem solved</h3>
          <p className="mt-3 text-sm leading-6 text-cyan-950">
            Scaling puts columns with different units onto comparable ranges so distances, penalties, and gradients are
            not dominated by one large-unit feature.
          </p>
        </div>
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-amber-700">Mistake to avoid</h3>
          <p className="mt-3 text-sm leading-6 text-amber-950">
            Do not fit scalers, imputers, encoders, or feature selectors on validation or test data before evaluation.
          </p>
        </div>
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-emerald-700">Understanding check</h3>
          <p className="mt-3 text-sm leading-6 text-emerald-950">
            Predict whether the outlier should affect min-max scaling or robust scaling more, then toggle it.
          </p>
        </div>
      </section>

      <AssessmentPanel lessonId="feature-scaling-preprocessing" title="Feature Scaling and Preprocessing check" />
    </div>
  );
}
