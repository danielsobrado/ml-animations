import React, { useMemo, useState } from 'react';
import { AlertTriangle, BarChart3, Calculator, RotateCcw, ShieldCheck, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const BERNOULLI_DATASETS = {
  balanced: { label: '6 successes / 4 failures', successes: 6, failures: 4, detail: 'The MLE is near the observed success rate.' },
  rare: { label: '2 successes / 10 failures', successes: 2, failures: 10, detail: 'Rare positives push the MLE downward.' },
  strong: { label: '9 successes / 1 failure', successes: 9, failures: 1, detail: 'Mostly successful observations push the MLE upward.' },
};

const GAUSSIAN_DATASETS = {
  compact: { label: 'Compact measurements', values: [4.7, 5.0, 5.2, 4.9, 5.1, 5.4], sigma: 0.45 },
  shifted: { label: 'Shifted measurements', values: [6.2, 6.4, 6.9, 6.6, 6.8, 7.1], sigma: 0.5 },
  noisy: { label: 'Noisy measurements', values: [3.7, 5.5, 4.3, 6.2, 4.8, 5.9], sigma: 0.9 },
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function bernoulliLogLikelihood(p, successes, failures) {
  const safeP = clamp(p, 0.001, 0.999);
  return successes * Math.log(safeP) + failures * Math.log(1 - safeP);
}

function gaussianLogLikelihood(mu, values, sigma) {
  const variance = sigma ** 2;
  return values.reduce((sum, value) => {
    const residual = value - mu;
    return sum - 0.5 * Math.log(2 * Math.PI * variance) - residual ** 2 / (2 * variance);
  }, 0);
}

function bernoulliCurve(successes, failures) {
  return Array.from({ length: 80 }, (_, index) => {
    const p = 0.02 + (index / 79) * 0.96;
    return { x: p, y: bernoulliLogLikelihood(p, successes, failures) };
  });
}

function gaussianCurve(values, sigma) {
  const min = Math.min(...values) - 1.2;
  const max = Math.max(...values) + 1.2;
  return Array.from({ length: 80 }, (_, index) => {
    const mu = min + (index / 79) * (max - min);
    return { x: mu, y: gaussianLogLikelihood(mu, values, sigma) };
  });
}

function normalizePath(points, width = 320, height = 150) {
  const minX = Math.min(...points.map((point) => point.x));
  const maxX = Math.max(...points.map((point) => point.x));
  const minY = Math.min(...points.map((point) => point.y));
  const maxY = Math.max(...points.map((point) => point.y));
  return points.map((point, index) => {
    const x = 28 + ((point.x - minX) / Math.max(0.001, maxX - minX)) * width;
    const y = 178 - ((point.y - minY) / Math.max(0.001, maxY - minY)) * height;
    return `${index === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(' ');
}

function candidateX(value, min, max, width = 320) {
  return 28 + ((value - min) / Math.max(0.001, max - min)) * width;
}

function likelihoodRatio(candidateLogLikelihood, mleLogLikelihood) {
  return Math.exp(candidateLogLikelihood - mleLogLikelihood);
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

function ObservationStrip({ successes, failures }) {
  const observations = [
    ...Array.from({ length: successes }, (_, index) => ({ id: `s${index}`, y: 1 })),
    ...Array.from({ length: failures }, (_, index) => ({ id: `f${index}`, y: 0 })),
  ];
  return (
    <div className="grid grid-cols-5 gap-2 sm:grid-cols-10">
      {observations.map((item) => (
        <div
          key={item.id}
          className={`rounded-lg border px-2 py-3 text-center text-sm font-black ${
            item.y ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-slate-200 bg-slate-50 text-slate-600'
          }`}
        >
          y={item.y}
        </div>
      ))}
    </div>
  );
}

export default function MaximumLikelihoodEstimationAnimation() {
  const [mode, setMode] = useState('bernoulli');
  const [bernoulliDatasetId, setBernoulliDatasetId] = useState('balanced');
  const [gaussianDatasetId, setGaussianDatasetId] = useState('compact');
  const [candidateP, setCandidateP] = useState(0.45);
  const [candidateMu, setCandidateMu] = useState(5.4);
  const [showNegativeLog, setShowNegativeLog] = useState(true);

  const bernoulli = BERNOULLI_DATASETS[bernoulliDatasetId];
  const gaussian = GAUSSIAN_DATASETS[gaussianDatasetId];
  const bernoulliMle = bernoulli.successes / (bernoulli.successes + bernoulli.failures);
  const gaussianMle = gaussian.values.reduce((sum, value) => sum + value, 0) / gaussian.values.length;

  const curve = useMemo(() => {
    if (mode === 'bernoulli') return bernoulliCurve(bernoulli.successes, bernoulli.failures);
    return gaussianCurve(gaussian.values, gaussian.sigma);
  }, [mode, bernoulli.successes, bernoulli.failures, gaussian.values, gaussian.sigma]);

  const candidate = mode === 'bernoulli' ? candidateP : candidateMu;
  const mle = mode === 'bernoulli' ? bernoulliMle : gaussianMle;
  const candidateLogLikelihood = mode === 'bernoulli'
    ? bernoulliLogLikelihood(candidateP, bernoulli.successes, bernoulli.failures)
    : gaussianLogLikelihood(candidateMu, gaussian.values, gaussian.sigma);
  const mleLogLikelihood = mode === 'bernoulli'
    ? bernoulliLogLikelihood(bernoulliMle, bernoulli.successes, bernoulli.failures)
    : gaussianLogLikelihood(gaussianMle, gaussian.values, gaussian.sigma);
  const relativeLikelihood = likelihoodRatio(candidateLogLikelihood, mleLogLikelihood);
  const xMin = Math.min(...curve.map((point) => point.x));
  const xMax = Math.max(...curve.map((point) => point.x));
  const candidatePosition = candidateX(candidate, xMin, xMax);
  const mlePosition = candidateX(mle, xMin, xMax);
  const negativeLogLikelihood = -candidateLogLikelihood;
  const nearMle = Math.abs(candidate - mle) < (mode === 'bernoulli' ? 0.04 : 0.12);

  const reset = () => {
    setMode('bernoulli');
    setBernoulliDatasetId('balanced');
    setGaussianDatasetId('compact');
    setCandidateP(0.45);
    setCandidateMu(5.4);
    setShowNegativeLog(true);
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Statistical fitting</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Maximum Likelihood Estimation</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              MLE compares candidate parameters by asking how likely the observed data would be if each candidate were
              true. The best parameter maximizes likelihood, or equivalently minimizes negative log-likelihood.
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
          Fitting controls
        </div>
        <div className="grid gap-4 xl:grid-cols-[1fr_1.4fr_1fr_1fr]">
          <div className="grid gap-2">
            <span className="text-sm font-bold text-slate-700">Model family</span>
            <div className="grid gap-2">
              {[
                ['bernoulli', 'Bernoulli probability'],
                ['gaussian', 'Gaussian mean'],
              ].map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setMode(id)}
                  className={`rounded-lg border px-3 py-2 text-left text-sm font-black transition ${
                    mode === id ? 'border-cyan-500 bg-cyan-600 text-white' : 'border-slate-200 bg-slate-50 text-slate-700'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div className="grid gap-2">
            <span className="text-sm font-bold text-slate-700">Observed dataset</span>
            <div className="grid gap-2">
              {Object.entries(mode === 'bernoulli' ? BERNOULLI_DATASETS : GAUSSIAN_DATASETS).map(([id, dataset]) => {
                const active = mode === 'bernoulli' ? bernoulliDatasetId === id : gaussianDatasetId === id;
                return (
                  <button
                    key={id}
                    type="button"
                    onClick={() => (mode === 'bernoulli' ? setBernoulliDatasetId(id) : setGaussianDatasetId(id))}
                    className={`rounded-lg border px-3 py-2 text-left text-sm font-black transition ${
                      active ? 'border-emerald-500 bg-emerald-600 text-white' : 'border-slate-200 bg-slate-50 text-slate-700'
                    }`}
                  >
                    {dataset.label}
                    <span className={`mt-1 block text-xs font-semibold normal-case leading-4 ${active ? 'text-emerald-50' : 'text-slate-500'}`}>
                      {dataset.detail ?? `sigma = ${dataset.sigma}`}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {mode === 'bernoulli' ? (
            <label className="grid gap-2 text-sm font-bold text-slate-700">
              Candidate p: {candidateP.toFixed(2)}
              <input min="0.02" max="0.98" step="0.01" type="range" value={candidateP} onChange={(event) => setCandidateP(Number(event.target.value))} />
              <span className="text-xs font-semibold text-slate-500">Move p and watch the likelihood curve peak near the observed rate.</span>
            </label>
          ) : (
            <label className="grid gap-2 text-sm font-bold text-slate-700">
              Candidate mu: {candidateMu.toFixed(2)}
              <input min="3.2" max="7.4" step="0.05" type="range" value={candidateMu} onChange={(event) => setCandidateMu(Number(event.target.value))} />
              <span className="text-xs font-semibold text-slate-500">Move the mean and watch fit degrade as residuals grow.</span>
            </label>
          )}

          <label className="flex items-start gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm font-bold text-slate-700">
            <input
              type="checkbox"
              checked={showNegativeLog}
              onChange={(event) => setShowNegativeLog(event.target.checked)}
              className="mt-1"
            />
            <span>
              Show negative log-likelihood
              <small className="mt-1 block font-semibold leading-5 text-slate-500">Training losses often minimize this value.</small>
            </span>
          </label>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.2fr_0.9fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <BarChart3 size={16} />
            Likelihood curve
          </div>
          <svg viewBox="0 0 380 220" role="img" aria-label="Likelihood curve over candidate parameter" className="h-auto w-full rounded-lg bg-slate-50">
            <rect x="28" y="28" width="320" height="150" rx="8" fill="#ffffff" stroke="#cbd5e1" />
            {[0, 0.25, 0.5, 0.75, 1].map((mark) => (
              <line key={mark} x1={28 + mark * 320} x2={28 + mark * 320} y1="28" y2="178" stroke="#e2e8f0" />
            ))}
            <path d={normalizePath(curve)} fill="none" stroke="#0f172a" strokeWidth="4" strokeLinecap="round" />
            <line x1={candidatePosition} x2={candidatePosition} y1="24" y2="182" stroke="#0284c7" strokeWidth="4" />
            <line x1={mlePosition} x2={mlePosition} y1="24" y2="182" stroke="#10b981" strokeWidth="4" strokeDasharray="6 6" />
            <text x="188" y="208" textAnchor="middle" fontSize="12" fontWeight="800" fill="#475569">
              candidate parameter
            </text>
          </svg>
          <div className="mt-3 flex flex-wrap gap-3 text-xs font-bold text-slate-600">
            <span className="inline-flex items-center gap-2"><i className="h-1 w-8 rounded bg-slate-900" />log-likelihood</span>
            <span className="inline-flex items-center gap-2"><i className="h-4 w-1 rounded bg-sky-600" />candidate</span>
            <span className="inline-flex items-center gap-2"><i className="h-4 w-1 rounded bg-emerald-500" />MLE</span>
          </div>
        </div>

        <div className="space-y-4">
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <div className="mb-3 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
              <Calculator size={16} />
              Candidate score
            </div>
            <div className="rounded-lg bg-slate-50 p-4 font-mono text-sm text-slate-800">
              {mode === 'bernoulli'
                ? `log L(p=${candidateP.toFixed(2)}) = ${candidateLogLikelihood.toFixed(2)}`
                : `log L(mu=${candidateMu.toFixed(2)}) = ${candidateLogLikelihood.toFixed(2)}`}
              <br />
              {showNegativeLog ? `NLL = ${negativeLogLikelihood.toFixed(2)}` : `relative L = ${relativeLikelihood.toFixed(3)}`}
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-700">
              {nearMle
                ? 'This candidate is close to the maximum-likelihood estimate.'
                : 'This candidate explains the observed data less well than the MLE.'}
            </p>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Observed data</p>
            {mode === 'bernoulli' ? (
              <div className="mt-3">
                <ObservationStrip successes={bernoulli.successes} failures={bernoulli.failures} />
              </div>
            ) : (
              <div className="mt-3 grid gap-2">
                {gaussian.values.map((value, index) => (
                  <div key={`${value}-${index}`} className="grid grid-cols-[44px_1fr_48px] items-center gap-2 text-sm">
                    <span className="font-mono font-black text-slate-500">x{index + 1}</span>
                    <div className="h-3 rounded-full bg-slate-100">
                      <div className="h-3 rounded-full bg-cyan-600" style={{ width: `${((value - 3) / 5) * 100}%` }} />
                    </div>
                    <strong className="font-mono text-slate-700">{value.toFixed(1)}</strong>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <Stat label="Candidate" value={candidate.toFixed(2)} detail={mode === 'bernoulli' ? 'Proposed success probability.' : 'Proposed Gaussian mean.'} />
        <Stat label="MLE" value={mle.toFixed(2)} detail={mode === 'bernoulli' ? 'Observed success fraction.' : 'Observed sample mean.'} />
        <Stat label="Relative likelihood" value={relativeLikelihood.toFixed(3)} detail="Candidate likelihood divided by MLE likelihood." />
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-4">
          <p className="text-xs font-black uppercase tracking-wide text-cyan-700">Predict before running</p>
          <p className="mt-2 text-sm leading-6 text-cyan-950">
            Switch datasets and predict where the likelihood peak moves before reading the MLE tile.
          </p>
        </div>
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
          <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-amber-700">
            <AlertTriangle size={14} />
            Failure mode
          </p>
          <p className="mt-2 text-sm leading-6 text-amber-950">
            Likelihood compares parameters after data is observed. It is not the same thing as a prior probability for the parameter.
          </p>
        </div>
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4">
          <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-emerald-700">
            <ShieldCheck size={14} />
            Practical rule
          </p>
          <p className="mt-2 text-sm leading-6 text-emerald-950">
            Maximize log-likelihood for numerical stability, or minimize negative log-likelihood as a training loss.
          </p>
        </div>
      </section>

      <AssessmentPanel lessonId="maximum-likelihood-estimation" />
    </div>
  );
}
