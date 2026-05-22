import React, { useMemo, useState } from 'react';
import { AlertTriangle, BarChart3, Calculator, RotateCcw, ShieldCheck, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function zFor(confidence) {
  if (confidence >= 99) return 2.58;
  if (confidence >= 95) return 1.96;
  if (confidence >= 90) return 1.64;
  return 1.28;
}

function deterministicNoise(index) {
  return (Math.sin(index * 12.9898) * 43758.5453) % 1;
}

function pseudoNormal(index) {
  const u1 = clamp(Math.abs(deterministicNoise(index)) || 0.01, 0.01, 0.99);
  const u2 = clamp(Math.abs(deterministicNoise(index + 31)) || 0.01, 0.01, 0.99);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function intervalFor(pHat, n, z) {
  const se = Math.sqrt((pHat * (1 - pHat)) / n);
  const margin = z * se;
  return {
    low: clamp(pHat - margin, 0, 1),
    high: clamp(pHat + margin, 0, 1),
    margin,
  };
}

function makeIntervals(trueRate, sampleSize, confidence, runs) {
  const p = trueRate / 100;
  const z = zFor(confidence);
  const se = Math.sqrt((p * (1 - p)) / sampleSize);
  return Array.from({ length: runs }, (_, index) => {
    const pHat = clamp(p + pseudoNormal(index + 1) * se, 0.001, 0.999);
    const interval = intervalFor(pHat, sampleSize, z);
    return {
      pHat,
      ...interval,
      captures: interval.low <= p && p <= interval.high,
    };
  });
}

function pct(value) {
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

export default function SamplingConfidenceIntervalsAnimation() {
  const [trueRate, setTrueRate] = useState(58);
  const [sampleSize, setSampleSize] = useState(160);
  const [confidence, setConfidence] = useState(95);
  const [runs, setRuns] = useState(40);

  const intervals = useMemo(
    () => makeIntervals(trueRate, sampleSize, confidence, runs),
    [trueRate, sampleSize, confidence, runs],
  );
  const first = intervals[0];
  const coverage = intervals.filter((interval) => interval.captures).length / intervals.length;
  const averageMargin = intervals.reduce((total, interval) => total + interval.margin, 0) / intervals.length;
  const quadrupledMargin = intervalFor(trueRate / 100, sampleSize * 4, zFor(confidence)).margin;

  const reset = () => {
    setTrueRate(58);
    setSampleSize(160);
    setConfidence(95);
    setRuns(40);
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Sampling uncertainty</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Sampling and Confidence Intervals</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              A confidence interval is a repeated-sampling procedure. Some intervals miss the fixed population value,
              but the procedure should capture it at roughly the advertised rate over many comparable samples.
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
          Sampling controls
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            True population rate: {trueRate}%
            <input type="range" min="10" max="90" value={trueRate} onChange={(event) => setTrueRate(Number(event.target.value))} />
            <span className="text-xs font-semibold text-slate-500">Shown only so the simulation can check coverage.</span>
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Sample size: {sampleSize}
            <input type="range" min="25" max="1000" step="25" value={sampleSize} onChange={(event) => setSampleSize(Number(event.target.value))} />
            <span className="text-xs font-semibold text-slate-500">Larger samples reduce standard error.</span>
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Confidence level: {confidence}%
            <input type="range" min="80" max="99" value={confidence} onChange={(event) => setConfidence(Number(event.target.value))} />
            <span className="text-xs font-semibold text-slate-500">Higher confidence uses a wider critical value.</span>
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Repeated samples: {runs}
            <input type="range" min="20" max="80" step="5" value={runs} onChange={(event) => setRuns(Number(event.target.value))} />
            <span className="text-xs font-semibold text-slate-500">More runs make misses visible.</span>
          </label>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-4">
        <Stat label="First interval" value={`${pct(first.low)}-${pct(first.high)}`} detail={`sample estimate ${pct(first.pHat)}`} />
        <Stat label="Average margin" value={`+/- ${(averageMargin * 100).toFixed(1)} pts`} detail="Across simulated samples." />
        <Stat label="Coverage in runs" value={pct(coverage)} detail={`${intervals.filter((interval) => interval.captures).length} of ${runs} intervals capture truth.`} />
        <Stat label="4x sample size" value={`+/- ${(quadrupledMargin * 100).toFixed(1)} pts`} detail="Margin shrinks by about half." />
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <BarChart3 size={16} />
            Repeated interval coverage
          </div>
          <svg viewBox="0 0 520 300" role="img" aria-label="Repeated confidence intervals" className="h-auto w-full rounded-lg bg-slate-50">
            <rect x="48" y="24" width="420" height="236" rx="8" fill="#ffffff" stroke="#cbd5e1" />
            <line x1={48 + (trueRate / 100) * 420} x2={48 + (trueRate / 100) * 420} y1="18" y2="270" stroke="#0f172a" strokeWidth="3" strokeDasharray="6 6" />
            {intervals.slice(0, 40).map((interval, index) => {
              const y = 34 + index * 5.4;
              return (
                <g key={index}>
                  <line
                    x1={48 + interval.low * 420}
                    x2={48 + interval.high * 420}
                    y1={y}
                    y2={y}
                    stroke={interval.captures ? '#0891b2' : '#f97316'}
                    strokeWidth="3"
                    strokeLinecap="round"
                  />
                  <circle cx={48 + interval.pHat * 420} cy={y} r="2.4" fill={interval.captures ? '#0891b2' : '#f97316'} />
                </g>
              );
            })}
            <text x="258" y="288" textAnchor="middle" fontSize="12" fontWeight="800" fill="#475569">
              population rate scale
            </text>
          </svg>
          <p className="mt-3 text-sm leading-6 text-slate-700">
            Blue intervals capture the fixed truth; orange intervals miss. Individual intervals are uncertain, while
            the method has a long-run coverage target.
          </p>
        </div>

        <div className="space-y-4">
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <div className="mb-3 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
              <Calculator size={16} />
              Width formula
            </div>
            <div className="rounded-lg bg-slate-950 p-4 font-mono text-sm leading-7 text-cyan-100">
              p_hat +/- z * sqrt(p_hat(1-p_hat) / n)<br />
              z = {zFor(confidence).toFixed(2)}<br />
              first margin = {(first.margin * 100).toFixed(1)} percentage points
            </div>
          </div>
          <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-4">
            <p className="text-xs font-black uppercase tracking-wide text-cyan-700">Predict before running</p>
            <p className="mt-2 text-sm leading-6 text-cyan-950">
              Quadruple sample size and predict how much the interval margin changes before reading the stat tile.
            </p>
          </div>
          <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
            <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-amber-700">
              <AlertTriangle size={14} />
              Failure mode
            </p>
            <p className="mt-2 text-sm leading-6 text-amber-950">
              A 95% interval does not mean this one fixed interval has a 95% probability of containing the truth.
            </p>
          </div>
          <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4">
            <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-emerald-700">
              <ShieldCheck size={14} />
              Practical rule
            </p>
            <p className="mt-2 text-sm leading-6 text-emerald-950">
              Wider intervals are honest about uncertainty. Narrow intervals require either more data or a lower confidence target.
            </p>
          </div>
        </div>
      </section>

      <AssessmentPanel lessonId="sampling-confidence-intervals" />
    </div>
  );
}
