import React, { useMemo, useState } from 'react';
import { BarChart3, RotateCcw, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';
import {
  SCORED_EXAMPLES,
  confusionAt,
  curvePoints,
  metricPercent,
  metrics,
  prPrecisionForPlot,
} from './rocPrCurvesModel';

function plot(points, xKey, yKey) {
  return points
    .map((point) => `${36 + point[xKey] * 288},${324 - point[yKey] * 288}`)
    .join(' ');
}

function Stat({ label, value, detail }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <p className="text-xs font-bold uppercase tracking-wide text-slate-500">{label}</p>
      <strong className="mt-1 block text-2xl font-black text-slate-900">{value}</strong>
      <span className="text-sm text-slate-600">{detail}</span>
    </div>
  );
}

function CurvePanel({ title, xLabel, yLabel, points, xKey, yKey, active, tone }) {
  const activeX = 36 + active[xKey] * 288;
  const activeY = 324 - active[yKey] * 288;

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5">
      <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">{title}</h3>
      <svg viewBox="0 0 360 360" className="mt-4 h-auto w-full rounded-lg border border-slate-200 bg-slate-50">
        <line x1="36" y1="324" x2="324" y2="324" stroke="#cbd5e1" strokeWidth="2" />
        <line x1="36" y1="324" x2="36" y2="36" stroke="#cbd5e1" strokeWidth="2" />
        <line x1="36" y1="324" x2="324" y2="36" stroke="#e2e8f0" strokeDasharray="5 5" />
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line x1={36 + tick * 288} y1="324" x2={36 + tick * 288} y2="329" stroke="#94a3b8" />
            <line x1="31" y1={324 - tick * 288} x2="36" y2={324 - tick * 288} stroke="#94a3b8" />
          </g>
        ))}
        <polyline points={plot(points, xKey, yKey)} fill="none" stroke={tone} strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
        {points.map((point) => (
          <circle key={`${point.threshold}-${xKey}-${yKey}`} cx={36 + point[xKey] * 288} cy={324 - point[yKey] * 288} r="4" fill={tone} opacity="0.45" />
        ))}
        <circle cx={activeX} cy={activeY} r="8" fill="white" stroke={tone} strokeWidth="4" />
        <text x="180" y="350" textAnchor="middle" className="fill-slate-600 text-xs font-bold">{xLabel}</text>
        <text x="16" y="184" textAnchor="middle" transform="rotate(-90 16 184)" className="fill-slate-600 text-xs font-bold">{yLabel}</text>
      </svg>
    </section>
  );
}

export default function RocPrCurvesAnimation() {
  const [threshold, setThreshold] = useState(0.5);
  const counts = useMemo(() => confusionAt(threshold), [threshold]);
  const current = metrics(counts);
  const points = useMemo(curvePoints, []);
  const activePoint = { threshold, ...current, precisionPlot: prPrecisionForPlot(current) };
  const precisionDetail = current.precision === null ? 'no predicted positives' : `${counts.tp} TP, ${counts.fp} FP`;

  const reset = () => setThreshold(0.5);

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-bold uppercase tracking-wide text-slate-500">Threshold sweep evaluation</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">ROC / Precision-Recall Curves</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              ROC and precision-recall curves show how a ranked classifier behaves as the decision threshold moves.
              ROC compares true positive rate with false positive rate; PR focuses on precision and recall for the
              positive class.
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

      <div className="grid gap-3 md:grid-cols-4">
        <Stat label="Threshold" value={threshold.toFixed(2)} detail="score cutoff for positive" />
        <Stat label="Precision" value={metricPercent(current.precision)} detail={precisionDetail} />
        <Stat label="Recall / TPR" value={metricPercent(current.recall)} detail={`${counts.tp} found, ${counts.fn} missed`} />
        <Stat label="FPR" value={metricPercent(current.fpr)} detail={`${counts.fp} false alarms`} />
      </div>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <SlidersHorizontal size={16} />
          Decision threshold
        </div>
        <label className="grid gap-2 text-sm font-bold text-slate-700">
          Predict positive when score {'>='} {threshold.toFixed(2)}
          <input
            min="0"
            max="1"
            step="0.05"
            type="range"
            value={threshold}
            onChange={(event) => setThreshold(Number(event.target.value))}
          />
        </label>
        <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {SCORED_EXAMPLES.map((example) => {
            const predicted = example.score >= threshold;
            const correct = predicted === Boolean(example.label);
            return (
              <div
                key={example.id}
                className={`rounded-lg border p-3 text-sm ${
                  correct ? 'border-emerald-200 bg-emerald-50 text-emerald-950' : 'border-rose-200 bg-rose-50 text-rose-950'
                }`}
              >
                <div className="flex items-center justify-between gap-3">
                  <strong>score {example.score.toFixed(2)}</strong>
                  <span className="rounded bg-white px-2 py-1 text-xs font-black">{example.label ? 'actual +' : 'actual -'}</span>
                </div>
                <p className="mt-2 text-xs font-bold uppercase tracking-wide">
                  predicted {predicted ? '+' : '-'} {correct ? 'correct' : 'mistake'}
                </p>
              </div>
            );
          })}
        </div>
      </section>

      <div className="grid gap-4 xl:grid-cols-2">
        <CurvePanel
          title="ROC curve"
          xLabel="False positive rate"
          yLabel="True positive rate"
          points={points}
          xKey="fpr"
          yKey="tpr"
          active={activePoint}
          tone="#2563eb"
        />
        <CurvePanel
          title="Precision-recall curve"
          xLabel="Recall"
          yLabel="Precision"
          points={[...points].sort((a, b) => a.recall - b.recall)}
          xKey="recall"
          yKey="precisionPlot"
          active={activePoint}
          tone="#dc2626"
        />
      </div>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-blue-200 bg-blue-50 p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-blue-700">
            <BarChart3 size={16} />
            What ROC shows
          </h3>
          <p className="mt-3 text-sm leading-6 text-blue-950">
            ROC asks whether positives rank above negatives across thresholds. It can look strong even when positives
            are rare because true negatives dominate the denominator.
          </p>
        </div>
        <div className="rounded-lg border border-rose-200 bg-rose-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-rose-700">What PR shows</h3>
          <p className="mt-3 text-sm leading-6 text-rose-950">
            PR asks how many predicted positives are real and how many real positives were found. It is often more
            revealing when the positive class is rare.
          </p>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Common mistake</h3>
          <p className="mt-3 text-sm leading-6 text-slate-700">
            A curve summarizes ranking quality, not the operating threshold. Choose the threshold from the cost of
            false positives and false negatives.
          </p>
        </div>
      </section>

      <AssessmentPanel lessonId="roc-pr-curves" title="ROC / Precision-Recall Curves check" />
    </div>
  );
}
