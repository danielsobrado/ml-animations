import React, { useMemo, useState } from 'react';
import { AlertTriangle, BarChart3, Calculator, RotateCcw, ShieldCheck, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const REGRESSION_POINTS = [
  { id: 'A', x: 0.5, y: 1.1 },
  { id: 'B', x: 1.2, y: 1.8 },
  { id: 'C', x: 2.1, y: 2.7 },
  { id: 'D', x: 3.0, y: 3.1 },
  { id: 'E', x: 3.7, y: 4.6 },
  { id: 'F', x: 4.5, y: 4.8 },
];

const CLASSIFICATION_POINTS = [
  { id: 'A', score: 0.12, y: 0 },
  { id: 'B', score: 0.28, y: 0 },
  { id: 'C', score: 0.42, y: 1 },
  { id: 'D', score: 0.58, y: 0 },
  { id: 'E', score: 0.74, y: 1 },
  { id: 'F', score: 0.88, y: 1 },
];

const MODES = {
  gaussian: {
    label: 'Gaussian regression',
    detail: 'Squared error is negative log-likelihood when residuals are Gaussian with fixed variance.',
  },
  laplace: {
    label: 'Laplace regression',
    detail: 'Absolute error fits a heavier-tailed residual model that is less dominated by outliers.',
  },
  bernoulli: {
    label: 'Bernoulli classification',
    detail: 'Cross-entropy is negative log-likelihood for observed binary labels.',
  },
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function regressionPrediction(point, slope, intercept) {
  return intercept + slope * point.x;
}

function classificationProbability(point, logitScale, bias) {
  return sigmoid((point.score - 0.5) * logitScale + bias);
}

function regressionRows(slope, intercept, outlierOn) {
  return REGRESSION_POINTS.map((point) => {
    const y = outlierOn && point.id === 'E' ? point.y + 2.1 : point.y;
    const prediction = regressionPrediction(point, slope, intercept);
    const residual = y - prediction;
    return {
      ...point,
      y,
      prediction,
      residual,
      squared: residual ** 2,
      absolute: Math.abs(residual),
    };
  });
}

function classificationRows(logitScale, bias, flippedLabel) {
  return CLASSIFICATION_POINTS.map((point) => {
    const y = flippedLabel && point.id === 'D' ? 1 : point.y;
    const probability = clamp(classificationProbability(point, logitScale, bias), 0.001, 0.999);
    const crossEntropy = -(y * Math.log(probability) + (1 - y) * Math.log(1 - probability));
    const accuracyLoss = (probability >= 0.5 ? 1 : 0) === y ? 0 : 1;
    return {
      ...point,
      y,
      probability,
      crossEntropy,
      accuracyLoss,
    };
  });
}

function sum(rows, key) {
  return rows.reduce((total, row) => total + row[key], 0);
}

function curvePath(values, key) {
  const max = Math.max(...values.map((value) => value[key]), 1);
  return values.map((value, index) => {
    const x = 30 + index * 32;
    const y = 166 - (value[key] / max) * 128;
    return `${index === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(' ');
}

function regressionSweep(mode, intercept, outlierOn) {
  return Array.from({ length: 12 }, (_, index) => {
    const slope = 0.45 + index * 0.16;
    const rows = regressionRows(slope, intercept, outlierOn);
    return {
      x: slope,
      squared: sum(rows, 'squared'),
      absolute: sum(rows, 'absolute'),
      active: mode === 'gaussian' ? sum(rows, 'squared') : sum(rows, 'absolute'),
    };
  });
}

function classificationSweep(logitScale, flippedLabel) {
  return Array.from({ length: 12 }, (_, index) => {
    const bias = -2 + index * 0.36;
    const rows = classificationRows(logitScale, bias, flippedLabel);
    return {
      x: bias,
      crossEntropy: sum(rows, 'crossEntropy'),
      accuracyLoss: sum(rows, 'accuracyLoss'),
      active: sum(rows, 'crossEntropy'),
    };
  });
}

function bestCandidate(values) {
  return values.reduce((best, item) => (item.active < best.active ? item : best), values[0]);
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

function LossRow({ label, value, tone }) {
  const toneClass = {
    sky: 'bg-sky-600',
    emerald: 'bg-emerald-600',
    rose: 'bg-rose-600',
    slate: 'bg-slate-500',
  }[tone];
  return (
    <div>
      <div className="mb-1 flex justify-between text-sm font-bold text-slate-700">
        <span>{label}</span>
        <span>{value.toFixed(2)}</span>
      </div>
      <div className="h-3 rounded-full bg-slate-100">
        <div className={`h-3 rounded-full ${toneClass}`} style={{ width: `${Math.min(100, value * 8)}%` }} />
      </div>
    </div>
  );
}

export default function LossFunctionsLikelihoodsAnimation() {
  const [mode, setMode] = useState('gaussian');
  const [slope, setSlope] = useState(0.9);
  const [intercept, setIntercept] = useState(0.7);
  const [logitScale, setLogitScale] = useState(5);
  const [bias, setBias] = useState(-0.1);
  const [outlierOn, setOutlierOn] = useState(false);
  const [flippedLabel, setFlippedLabel] = useState(false);

  const isClassification = mode === 'bernoulli';
  const regression = useMemo(() => regressionRows(slope, intercept, outlierOn), [slope, intercept, outlierOn]);
  const classification = useMemo(() => classificationRows(logitScale, bias, flippedLabel), [logitScale, bias, flippedLabel]);
  const sweep = useMemo(
    () => (isClassification ? classificationSweep(logitScale, flippedLabel) : regressionSweep(mode, intercept, outlierOn)),
    [isClassification, logitScale, flippedLabel, mode, intercept, outlierOn],
  );
  const best = bestCandidate(sweep);

  const squaredLoss = sum(regression, 'squared');
  const absoluteLoss = sum(regression, 'absolute');
  const crossEntropy = sum(classification, 'crossEntropy');
  const accuracyLoss = sum(classification, 'accuracyLoss');
  const activeLoss = mode === 'gaussian' ? squaredLoss : mode === 'laplace' ? absoluteLoss : crossEntropy;
  const activeLabel = mode === 'gaussian' ? 'Squared error' : mode === 'laplace' ? 'Absolute error' : 'Cross-entropy';
  const assumption = mode === 'gaussian'
    ? 'Gaussian residuals'
    : mode === 'laplace'
      ? 'Laplace residuals'
      : 'Bernoulli labels';

  const reset = () => {
    setMode('gaussian');
    setSlope(0.9);
    setIntercept(0.7);
    setLogitScale(5);
    setBias(-0.1);
    setOutlierOn(false);
    setFlippedLabel(false);
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Loss as modeling assumption</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Loss Functions and Likelihoods</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Many training losses are negative log-likelihoods in compact form. Change the target assumption and the
              loss changes what errors it treats as expensive.
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
          Assumption controls
        </div>
        <div className="grid gap-4 xl:grid-cols-[1.4fr_1fr_1fr_1fr]">
          <div className="grid gap-2">
            <span className="text-sm font-bold text-slate-700">Target model</span>
            <div className="grid gap-2">
              {Object.entries(MODES).map(([id, config]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setMode(id)}
                  className={`rounded-lg border px-3 py-2 text-left text-sm font-black transition ${
                    mode === id ? 'border-cyan-500 bg-cyan-600 text-white' : 'border-slate-200 bg-slate-50 text-slate-700'
                  }`}
                >
                  {config.label}
                  <span className={`mt-1 block text-xs font-semibold normal-case leading-4 ${mode === id ? 'text-cyan-50' : 'text-slate-500'}`}>
                    {config.detail}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {isClassification ? (
            <>
              <label className="grid gap-2 text-sm font-bold text-slate-700">
                Logit scale: {logitScale.toFixed(1)}
                <input min="1" max="9" step="0.1" type="range" value={logitScale} onChange={(event) => setLogitScale(Number(event.target.value))} />
                <span className="text-xs font-semibold text-slate-500">Higher scale makes probabilities more confident.</span>
              </label>
              <label className="grid gap-2 text-sm font-bold text-slate-700">
                Bias: {bias.toFixed(2)}
                <input min="-2" max="2" step="0.05" type="range" value={bias} onChange={(event) => setBias(Number(event.target.value))} />
              </label>
              <label className="flex items-start gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm font-bold text-slate-700">
                <input type="checkbox" checked={flippedLabel} onChange={(event) => setFlippedLabel(event.target.checked)} className="mt-1" />
                <span>Flip one ambiguous label<small className="mt-1 block font-semibold leading-5 text-slate-500">Watch confidence become costly when the label disagrees.</small></span>
              </label>
            </>
          ) : (
            <>
              <label className="grid gap-2 text-sm font-bold text-slate-700">
                Slope: {slope.toFixed(2)}
                <input min="0.35" max="1.75" step="0.01" type="range" value={slope} onChange={(event) => setSlope(Number(event.target.value))} />
              </label>
              <label className="grid gap-2 text-sm font-bold text-slate-700">
                Intercept: {intercept.toFixed(2)}
                <input min="-0.2" max="1.8" step="0.01" type="range" value={intercept} onChange={(event) => setIntercept(Number(event.target.value))} />
              </label>
              <label className="flex items-start gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm font-bold text-slate-700">
                <input type="checkbox" checked={outlierOn} onChange={(event) => setOutlierOn(event.target.checked)} className="mt-1" />
                <span>Add outlier<small className="mt-1 block font-semibold leading-5 text-slate-500">Squared error reacts more strongly than absolute error.</small></span>
              </label>
            </>
          )}
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.15fr_0.95fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <BarChart3 size={16} />
            Example-level penalties
          </div>
          <div className="grid gap-3">
            {(isClassification ? classification : regression).map((row) => {
              const value = isClassification ? row.crossEntropy : mode === 'gaussian' ? row.squared : row.absolute;
              const label = isClassification
                ? `${row.id}: y=${row.y}, p=${row.probability.toFixed(2)}`
                : `${row.id}: residual=${row.residual.toFixed(2)}`;
              return <LossRow key={row.id} label={label} value={value} tone={value > 2.5 ? 'rose' : value > 1 ? 'emerald' : 'sky'} />;
            })}
          </div>
        </div>

        <div className="space-y-4">
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <div className="mb-3 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
              <Calculator size={16} />
              Likelihood mapping
            </div>
            <div className="rounded-lg bg-slate-50 p-4 font-mono text-sm text-slate-800">
              {mode === 'gaussian' && '-log p(y|x) = const + residual^2 / 2sigma^2'}
              {mode === 'laplace' && '-log p(y|x) = const + |residual| / b'}
              {mode === 'bernoulli' && '-log p(y|x) = -[y log(p) + (1-y) log(1-p)]'}
              <br />
              active loss = {activeLoss.toFixed(2)}
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-700">
              Current assumption: <strong>{assumption}</strong>. The loss is a compact scoring rule for that assumption.
            </p>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <div className="mb-3 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
              <BarChart3 size={16} />
              Parameter sweep
            </div>
            <svg viewBox="0 0 400 210" role="img" aria-label="Loss across candidate parameter values" className="h-auto w-full rounded-lg bg-slate-50">
              <rect x="30" y="28" width="352" height="138" rx="8" fill="#ffffff" stroke="#cbd5e1" />
              <path d={curvePath(sweep, 'active')} fill="none" stroke="#0f172a" strokeWidth="4" strokeLinecap="round" />
              <line x1={30 + sweep.indexOf(best) * 32} x2={30 + sweep.indexOf(best) * 32} y1="24" y2="170" stroke="#10b981" strokeWidth="4" strokeDasharray="6 6" />
              <text x="206" y="198" textAnchor="middle" fontSize="12" fontWeight="800" fill="#475569">
                candidate parameter
              </text>
            </svg>
            <p className="mt-3 text-sm font-bold text-slate-700">Best sweep candidate: {best.x.toFixed(2)}</p>
          </div>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <Stat label="Active loss" value={activeLoss.toFixed(2)} detail={activeLabel} />
        <Stat label="Assumption" value={assumption} detail="The implied target/noise model." />
        <Stat
          label="Alternative"
          value={isClassification ? accuracyLoss.toFixed(0) : (mode === 'gaussian' ? absoluteLoss : squaredLoss).toFixed(2)}
          detail={isClassification ? '0/1 mistakes hide confidence.' : mode === 'gaussian' ? 'Absolute error under same fit.' : 'Squared error under same fit.'}
        />
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-4">
          <p className="text-xs font-black uppercase tracking-wide text-cyan-700">Predict before running</p>
          <p className="mt-2 text-sm leading-6 text-cyan-950">Add the outlier or flip the label, then predict which loss changes most.</p>
        </div>
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
          <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-amber-700">
            <AlertTriangle size={14} />
            Failure mode
          </p>
          <p className="mt-2 text-sm leading-6 text-amber-950">Do not call a loss arbitrary. It usually encodes which target behavior the model assumes.</p>
        </div>
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4">
          <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-emerald-700">
            <ShieldCheck size={14} />
            Practical rule
          </p>
          <p className="mt-2 text-sm leading-6 text-emerald-950">Choose losses that match the target type, noise pattern, and downstream cost of bad confidence.</p>
        </div>
      </section>

      <AssessmentPanel lessonId="loss-functions-likelihoods" />
    </div>
  );
}
