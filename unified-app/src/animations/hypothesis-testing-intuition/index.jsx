import React, { useMemo, useState } from 'react';
import { AlertTriangle, BarChart3, Calculator, RotateCcw, ShieldCheck, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return sign * y;
}

function normalCdf(x) {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

function zCritical(alpha) {
  if (alpha <= 0.01) return 2.58;
  if (alpha <= 0.05) return 1.96;
  if (alpha <= 0.1) return 1.64;
  return 1.28;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
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

function normalPath(mean, spread, scale = 1) {
  return Array.from({ length: 90 }, (_, index) => {
    const x = -4 + index * (8 / 89);
    const density = Math.exp(-0.5 * ((x - mean) / spread) ** 2) * scale;
    const px = 40 + ((x + 4) / 8) * 390;
    const py = 190 - density * 150;
    return `${index === 0 ? 'M' : 'L'} ${px.toFixed(1)} ${py.toFixed(1)}`;
  }).join(' ');
}

export default function HypothesisTestingIntuitionAnimation() {
  const [effect, setEffect] = useState(7);
  const [noise, setNoise] = useState(18);
  const [sampleSize, setSampleSize] = useState(180);
  const [alphaPct, setAlphaPct] = useState(5);

  const metrics = useMemo(() => {
    const standardError = noise / Math.sqrt(sampleSize);
    const z = standardError === 0 ? 0 : effect / standardError;
    const pValue = Math.min(1, 2 * (1 - normalCdf(Math.abs(z))));
    const alpha = alphaPct / 100;
    const critical = zCritical(alpha);
    const power = 1 - normalCdf(critical - z) + normalCdf(-critical - z);
    const practical = effect >= 10 ? 'large enough to matter' : effect >= 5 ? 'context dependent' : 'tiny effect';
    return {
      standardError,
      z,
      pValue,
      alpha,
      critical,
      power: clamp(power, 0, 1),
      reject: pValue < alpha,
      practical,
    };
  }, [effect, noise, sampleSize, alphaPct]);

  const reset = () => {
    setEffect(7);
    setNoise(18);
    setSampleSize(180);
    setAlphaPct(5);
  };

  const observedX = clamp(235 + metrics.z * 48, 40, 430);
  const criticalRight = 235 + metrics.critical * 48;
  const criticalLeft = 235 - metrics.critical * 48;

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Signal versus noise</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Hypothesis Testing Intuition</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              A hypothesis test asks whether the observed effect is unusual under a no-effect baseline. It measures
              evidence against noise; it does not decide whether the effect is useful.
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
          Test controls
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Observed effect: {effect} pts
            <input type="range" min="0" max="30" value={effect} onChange={(event) => setEffect(Number(event.target.value))} />
            <span className="text-xs font-semibold text-slate-500">The measured lift, gap, or difference.</span>
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Outcome noise: {noise} pts
            <input type="range" min="5" max="35" value={noise} onChange={(event) => setNoise(Number(event.target.value))} />
            <span className="text-xs font-semibold text-slate-500">More variability raises standard error.</span>
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Sample size: {sampleSize}
            <input type="range" min="25" max="1200" step="25" value={sampleSize} onChange={(event) => setSampleSize(Number(event.target.value))} />
            <span className="text-xs font-semibold text-slate-500">Large samples can detect tiny effects.</span>
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Alpha threshold: {alphaPct}%
            <input type="range" min="1" max="20" value={alphaPct} onChange={(event) => setAlphaPct(Number(event.target.value))} />
            <span className="text-xs font-semibold text-slate-500">False-positive tolerance for this test.</span>
          </label>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-4">
        <Stat label="Standard error" value={metrics.standardError.toFixed(2)} detail="Noise divided by sqrt(sample size)." />
        <Stat label="Test statistic" value={metrics.z.toFixed(2)} detail="Observed effect / standard error." />
        <Stat label="p-value" value={`${(metrics.pValue * 100).toFixed(1)}%`} detail={metrics.reject ? 'Below alpha: statistically unusual.' : 'Not below alpha.'} />
        <Stat label="Power estimate" value={`${Math.round(metrics.power * 100)}%`} detail="Chance to reject if this effect is real." />
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <BarChart3 size={16} />
            Null distribution and observed result
          </div>
          <svg viewBox="0 0 470 240" role="img" aria-label="Hypothesis testing distribution" className="h-auto w-full rounded-lg bg-slate-50">
            <rect x="40" y="24" width="390" height="170" rx="8" fill="#ffffff" stroke="#cbd5e1" />
            <path d={normalPath(0, 1, 1)} fill="none" stroke="#0891b2" strokeWidth="4" />
            <path d={normalPath(metrics.z, 1, 0.72)} fill="none" stroke="#10b981" strokeWidth="4" strokeDasharray="8 6" />
            <line x1={criticalLeft} x2={criticalLeft} y1="36" y2="196" stroke="#f97316" strokeWidth="3" strokeDasharray="6 6" />
            <line x1={criticalRight} x2={criticalRight} y1="36" y2="196" stroke="#f97316" strokeWidth="3" strokeDasharray="6 6" />
            <line x1={observedX} x2={observedX} y1="28" y2="202" stroke="#0f172a" strokeWidth="4" />
            <text x="235" y="222" textAnchor="middle" fontSize="12" fontWeight="800" fill="#475569">
              z-score scale: null centered at 0, observed line at current statistic
            </text>
          </svg>
          <p className="mt-3 text-sm leading-6 text-slate-700">
            The orange cutoffs are the rejection boundary for alpha. The black line is the observed statistic. The
            dashed green curve shows where repeated estimates would center if the current effect is real.
          </p>
        </div>

        <div className="space-y-4">
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <div className="mb-3 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
              <Calculator size={16} />
              Decision lens
            </div>
            <div className="rounded-lg bg-slate-950 p-4 font-mono text-sm leading-7 text-cyan-100">
              z = effect / standard error<br />
              z = {effect} / {metrics.standardError.toFixed(2)} = {metrics.z.toFixed(2)}<br />
              reject? {metrics.pValue.toFixed(3)} &lt; {metrics.alpha.toFixed(2)} = {String(metrics.reject)}
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-700">
              Practical size: <strong>{metrics.practical}</strong>. Evidence and usefulness should be reported separately.
            </p>
          </div>
          <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-4">
            <p className="text-xs font-black uppercase tracking-wide text-cyan-700">Predict before running</p>
            <p className="mt-2 text-sm leading-6 text-cyan-950">
              Keep the effect fixed and increase sample size. Predict whether p-value, practical importance, or both change.
            </p>
          </div>
          <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
            <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-amber-700">
              <AlertTriangle size={14} />
              Failure mode
            </p>
            <p className="mt-2 text-sm leading-6 text-amber-950">
              Statistical significance can come from a tiny effect with enough data. It is not automatic product value.
            </p>
          </div>
          <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4">
            <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-emerald-700">
              <ShieldCheck size={14} />
              Practical rule
            </p>
            <p className="mt-2 text-sm leading-6 text-emerald-950">
              Report the effect size, uncertainty interval, p-value, sample size, and decision threshold together.
            </p>
          </div>
        </div>
      </section>

      <AssessmentPanel lessonId="hypothesis-testing-intuition" />
    </div>
  );
}
