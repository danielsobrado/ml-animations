import React, { useMemo, useState } from 'react';
import { BarChart3, RotateCcw, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const SCENARIOS = {
  calibrated: {
    label: 'Calibrated',
    detail: 'Predicted probabilities line up with observed frequencies.',
    bins: [
      { confidence: 0.1, observed: 0.09, count: 18 },
      { confidence: 0.3, observed: 0.31, count: 24 },
      { confidence: 0.5, observed: 0.49, count: 28 },
      { confidence: 0.7, observed: 0.72, count: 24 },
      { confidence: 0.9, observed: 0.88, count: 16 },
    ],
  },
  overconfident: {
    label: 'Overconfident',
    detail: 'Scores are too extreme: high-confidence buckets are right less often than promised.',
    bins: [
      { confidence: 0.1, observed: 0.22, count: 18 },
      { confidence: 0.3, observed: 0.38, count: 24 },
      { confidence: 0.5, observed: 0.51, count: 28 },
      { confidence: 0.7, observed: 0.61, count: 24 },
      { confidence: 0.9, observed: 0.74, count: 16 },
    ],
  },
  underconfident: {
    label: 'Underconfident',
    detail: 'Scores are too timid: low and high buckets sit closer to 0.5 than the outcomes justify.',
    bins: [
      { confidence: 0.1, observed: 0.03, count: 18 },
      { confidence: 0.3, observed: 0.18, count: 24 },
      { confidence: 0.5, observed: 0.5, count: 28 },
      { confidence: 0.7, observed: 0.82, count: 24 },
      { confidence: 0.9, observed: 0.96, count: 16 },
    ],
  },
};

function totalCount(bins) {
  return bins.reduce((sum, bin) => sum + bin.count, 0);
}

function expectedCalibrationError(bins) {
  const total = totalCount(bins);
  return bins.reduce((sum, bin) => sum + (bin.count / total) * Math.abs(bin.observed - bin.confidence), 0);
}

function brierScore(bins) {
  const total = totalCount(bins);
  return bins.reduce((sum, bin) => {
    const positives = bin.count * bin.observed;
    const negatives = bin.count - positives;
    return sum + positives * (1 - bin.confidence) ** 2 + negatives * bin.confidence ** 2;
  }, 0) / total;
}

function thresholdStats(bins, threshold) {
  const predictedPositive = bins.filter((bin) => bin.confidence >= threshold);
  const predictedNegative = bins.filter((bin) => bin.confidence < threshold);
  const tp = predictedPositive.reduce((sum, bin) => sum + bin.count * bin.observed, 0);
  const fp = predictedPositive.reduce((sum, bin) => sum + bin.count * (1 - bin.observed), 0);
  const fn = predictedNegative.reduce((sum, bin) => sum + bin.count * bin.observed, 0);
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  return {
    predictedPositive: predictedPositive.reduce((sum, bin) => sum + bin.count, 0),
    precision,
    recall,
  };
}

function project(value) {
  return 260 - value * 220;
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

export default function CalibrationAnimation() {
  const [scenario, setScenario] = useState('overconfident');
  const [threshold, setThreshold] = useState(0.5);
  const bins = SCENARIOS[scenario].bins;
  const ece = useMemo(() => expectedCalibrationError(bins), [bins]);
  const brier = useMemo(() => brierScore(bins), [bins]);
  const stats = useMemo(() => thresholdStats(bins, threshold), [bins, threshold]);

  const reset = () => {
    setScenario('overconfident');
    setThreshold(0.5);
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Probability quality</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Calibration</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Classification metrics ask whether labels are right after a threshold. Calibration asks whether predicted
              probabilities mean what they say: among examples scored 0.8, about 80 percent should be positive.
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
          Calibration controls
        </div>
        <div className="grid gap-4 lg:grid-cols-[1.6fr_1fr]">
          <div className="grid gap-2">
            <span className="text-sm font-bold text-slate-700">Probability behavior</span>
            <div className="grid grid-cols-3 gap-2">
              {Object.entries(SCENARIOS).map(([id, config]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setScenario(id)}
                  className={`rounded-lg border px-3 py-2 text-sm font-black transition ${scenario === id ? 'border-cyan-500 bg-cyan-600 text-white' : 'border-slate-200 bg-slate-50 text-slate-700'}`}
                >
                  {config.label}
                </button>
              ))}
            </div>
          </div>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Decision threshold: {threshold.toFixed(2)}
            <input min="0.1" max="0.9" step="0.1" type="range" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
          </label>
        </div>
        <p className="mt-4 rounded-lg bg-slate-50 p-3 text-sm leading-6 text-slate-700">
          <strong className="text-slate-950">{SCENARIOS[scenario].label}:</strong> {SCENARIOS[scenario].detail}
        </p>
      </section>

      <div className="grid gap-3 md:grid-cols-4">
        <Stat label="ECE" value={`${(ece * 100).toFixed(1)}%`} detail="weighted calibration gap" />
        <Stat label="Brier score" value={brier.toFixed(3)} detail="probability error" />
        <Stat label="Predicted positive" value={stats.predictedPositive} detail={`score >= ${threshold.toFixed(1)}`} />
        <Stat label="Precision / recall" value={`${Math.round(stats.precision * 100)} / ${Math.round(stats.recall * 100)}%`} detail="threshold metrics" />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <BarChart3 size={16} />
            Reliability diagram
          </h3>
          <svg viewBox="0 0 400 300" className="mt-4 h-auto w-full rounded-lg border border-slate-200 bg-slate-50" role="img" aria-label="Calibration reliability diagram">
            <line x1="42" y1="260" x2="370" y2="260" stroke="#cbd5e1" />
            <line x1="42" y1="40" x2="42" y2="260" stroke="#cbd5e1" />
            <line x1="42" y1="260" x2="370" y2="40" stroke="#94a3b8" strokeWidth="3" strokeDasharray="7 7" />
            <line
              x1={42 + threshold * 328}
              y1="260"
              x2={42 + threshold * 328}
              y2="40"
              stroke="#f97316"
              strokeWidth="2"
              strokeDasharray="5 5"
            />
            {bins.map((bin) => {
              const x = 42 + bin.confidence * 328;
              const y = project(bin.observed);
              const idealY = project(bin.confidence);
              return (
                <g key={bin.confidence}>
                  <line x1={x} y1={idealY} x2={x} y2={y} stroke="#f97316" strokeWidth="3" opacity="0.65" />
                  <circle cx={x} cy={y} r={5 + bin.count / 12} fill="#0891b2" stroke="#ffffff" strokeWidth="3" />
                  <text x={x} y="278" textAnchor="middle" className="fill-slate-600 text-xs font-bold">{bin.confidence.toFixed(1)}</text>
                </g>
              );
            })}
            <text x="206" y="294" textAnchor="middle" className="fill-slate-600 text-xs font-bold">predicted probability</text>
            <text x="16" y="150" textAnchor="middle" transform="rotate(-90 16 150)" className="fill-slate-600 text-xs font-bold">observed positive rate</text>
            <text x="240" y="62" className="fill-slate-500 text-xs font-bold">perfect calibration</text>
          </svg>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Bucket table</h3>
          <div className="mt-4 space-y-3">
            {bins.map((bin) => {
              const gap = bin.observed - bin.confidence;
              return (
                <div key={bin.confidence} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <div className="flex items-center justify-between gap-3 text-sm">
                    <strong className="text-slate-950">Score {bin.confidence.toFixed(1)}</strong>
                    <span className={Math.abs(gap) < 0.04 ? 'font-bold text-emerald-700' : 'font-bold text-rose-700'}>
                      observed {(bin.observed * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="mt-2 h-2 overflow-hidden rounded-full bg-white">
                    <div className="h-full rounded-full bg-cyan-500" style={{ width: `${bin.count}%` }} />
                  </div>
                  <p className="mt-2 text-xs font-bold uppercase tracking-wide text-slate-500">{bin.count} examples in bucket</p>
                </div>
              );
            })}
          </div>
        </section>
      </div>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-cyan-700">Problem solved</h3>
          <p className="mt-3 text-sm leading-6 text-cyan-950">
            Calibration tells you whether probabilities can be used for risk ranking, triage, and threshold decisions.
          </p>
        </div>
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-amber-700">Mistake to avoid</h3>
          <p className="mt-3 text-sm leading-6 text-amber-950">
            A sigmoid output is bounded between 0 and 1, but that does not prove the score is calibrated.
          </p>
        </div>
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-emerald-700">Understanding check</h3>
          <p className="mt-3 text-sm leading-6 text-emerald-950">
            Compare overconfident and calibrated modes, then explain which bucket has the largest reliability gap.
          </p>
        </div>
      </section>

      <AssessmentPanel lessonId="calibration" title="Calibration check" />
    </div>
  );
}
