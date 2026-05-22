import React, { useMemo, useState } from 'react';
import {
  BadgeCheck,
  RefreshCw,
  Scale,
  ShieldCheck,
  TrendingUp,
  Users,
} from 'lucide-react';

const BASE = {
  protected: {
    label: 'Protected',
    scores: [0.22, 0.31, 0.37, 0.55, 0.65, 0.71, 0.18, 0.24, 0.46, 0.39, 0.52, 0.68],
    labels: [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
  },
  unprotected: {
    label: 'Unprotected',
    scores: [0.17, 0.39, 0.43, 0.45, 0.51, 0.58, 0.62, 0.66, 0.47, 0.52, 0.59, 0.64],
    labels: [0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1],
  },
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function binaryStats(samples, threshold) {
  const tp = samples.filter((sample) => sample.score >= threshold && sample.label === 1).length;
  const fp = samples.filter((sample) => sample.score >= threshold && sample.label === 0).length;
  const fn = samples.filter((sample) => sample.score < threshold && sample.label === 1).length;
  const tn = samples.filter((sample) => sample.score < threshold && sample.label === 0).length;

  const tpr = tp + fn === 0 ? 0 : tp / (tp + fn);
  const fpr = fp + tn === 0 ? 0 : fp / (fp + tn);
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const selectionRate = (tp + fp) / samples.length;
  return {
    tp,
    fp,
    fn,
    tn,
    tpr,
    fpr,
    precision,
    recall: tpr,
    selectionRate,
    total: samples.length,
    positiveRate: (tp + fn) / samples.length,
  };
}

function buildRows(groupData) {
  return {
    protected: groupData.protected.scores.map((score, index) => ({
      id: index,
      group: 'protected',
      score,
      label: groupData.protected.labels[index],
    })),
    unprotected: groupData.unprotected.scores.map((score, index) => ({
      id: index,
      group: 'unprotected',
      score,
      label: groupData.unprotected.labels[index],
    })),
  };
}

function metricGap(a, b) {
  return a - b;
}

function percent(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function Stat({ label, value, tone = 'slate', detail }) {
  const toneClass = {
    slate: 'border-slate-200 bg-white text-slate-900',
    amber: 'border-amber-200 bg-amber-50 text-amber-950',
    emerald: 'border-emerald-200 bg-emerald-50 text-emerald-950',
    rose: 'border-rose-200 bg-rose-50 text-rose-950',
  }[tone];
  return (
    <div className={`rounded-lg border p-4 ${toneClass}`}>
      <p className="text-xs font-black uppercase tracking-wide text-slate-500">{label}</p>
      <strong className="mt-1 block text-2xl font-black">{value}</strong>
      <span className="mt-2 block text-sm leading-6">{detail}</span>
    </div>
  );
}

function GroupTable({ name, stats }) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4">
      <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">{name}</h3>
      <div className="mt-2 grid grid-cols-4 gap-2 text-xs">
        {Object.entries({
          TP: stats.tp,
          FP: stats.fp,
          FN: stats.fn,
          TN: stats.tn,
        }).map(([key, value]) => (
          <div key={key} className="rounded-lg border border-slate-200 bg-slate-50 p-2">
            <p className="text-slate-500 uppercase">{key}</p>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
      <div className="mt-3 text-sm leading-6 text-slate-700">
        <p>Recall (TPR): <strong>{percent(stats.tpr)}</strong></p>
        <p>FPR: <strong>{percent(stats.fpr)}</strong></p>
        <p>Precision: <strong>{percent(stats.precision)}</strong></p>
        <p>Selection rate: <strong>{percent(stats.selectionRate)}</strong></p>
      </div>
    </section>
  );
}

function recommendationForGap(gap, metricName) {
  if (Math.abs(gap) < 0.04) {
    return `Current ${metricName} gap is small.`;
  }
  return `Reduce by re-tuning thresholds by group before relaxing this constraint in production.`;
}

export default function ModelFairness() {
  const [biasShift, setBiasShift] = useState(0);
  const [thresholdMode, setThresholdMode] = useState('global');
  const [globalThreshold, setGlobalThreshold] = useState(0.5);
  const [protectedThreshold, setProtectedThreshold] = useState(0.5);
  const [unprotectedThreshold, setUnprotectedThreshold] = useState(0.46);
  const [objective, setObjective] = useState('selection-rate');
  const [showCounterfactual, setShowCounterfactual] = useState(true);

  const rows = useMemo(() => buildRows(BASE), []);

  function thresholdsForScores() {
    return {
      protected: clamp(globalThreshold + (thresholdMode === 'global' ? 0 : 0) + 0, 0.02, 0.98),
      unprotected: clamp(globalThreshold + (thresholdMode === 'global' ? 0 : 0), 0.02, 0.98),
    };
  }

  const protectedData = rows.protected.map((row) => ({
    ...row,
    score: clamp(row.score + biasShift * 0.08, 0, 1),
  }));
  const unprotectedData = rows.unprotected.map((row) => ({
    ...row,
    score: clamp(row.score - biasShift * 0.08, 0, 1),
  }));

  const effectiveProtectedThreshold = thresholdMode === 'global'
    ? globalThreshold
    : clamp(protectedThreshold, 0.02, 0.98);
  const effectiveUnprotectedThreshold = thresholdMode === 'global'
    ? globalThreshold
    : clamp(unprotectedThreshold, 0.02, 0.98);

  const protectedStats = binaryStats(
    protectedData,
    effectiveProtectedThreshold,
  );
  const unprotectedStats = binaryStats(
    unprotectedData,
    effectiveUnprotectedThreshold,
  );

  const gaps = {
    selectionRate: metricGap(protectedStats.selectionRate, unprotectedStats.selectionRate),
    fpr: metricGap(protectedStats.fpr, unprotectedStats.fpr),
    tpr: metricGap(protectedStats.tpr, unprotectedStats.tpr),
  };

  const objectiveGap = {
    'selection-rate': gaps.selectionRate,
    'fpr-parity': gaps.fpr,
    'tpr-parity': gaps.tpr,
  }[objective];

  const objectiveAligned = Math.abs(objectiveGap) < 0.04;
  const fairnessTone = objectiveAligned ? 'emerald' : Math.abs(objectiveGap) < 0.08 ? 'amber' : 'rose';

  const counterfactual = useMemo(() => {
    if (!showCounterfactual) return null;
    return {
      protected: clamp(effectiveProtectedThreshold + 0.06, 0.02, 0.98),
      unprotected: clamp(effectiveUnprotectedThreshold - 0.03, 0.02, 0.98),
      note: recommendationForGap(objectiveGap, {
        'selection-rate': 'selection rate',
        'fpr-parity': 'false positive rate',
        'tpr-parity': 'true positive rate',
      }[objective]),
    };
  }, [showCounterfactual, objectiveGap, effectiveProtectedThreshold, effectiveUnprotectedThreshold, objective]);

  const selectedRate = protectedThreshold < 0.3 || unprotectedThreshold < 0.3 ? 'Very permissive' : 'Moderate';
  const selectedPrecision = objectiveAligned
    ? 'Pareto region still requires trade-offs'
    : 'Trade-off present across objectives';

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Responsible ML</p>
            <h1 className="mt-1 text-2xl font-black text-slate-950">Model Fairness</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Compare group metrics and inspect how threshold strategy changes parity, error balance, and selection behavior.
            </p>
          </div>
          <button
            type="button"
            onClick={() => {
              setBiasShift(0);
              setThresholdMode('global');
              setGlobalThreshold(0.5);
              setProtectedThreshold(0.5);
              setUnprotectedThreshold(0.46);
              setObjective('selection-rate');
              setShowCounterfactual(true);
            }}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-bold text-slate-800"
          >
            <RefreshCw size={16} />
            Reset
          </button>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-[1.2fr_1fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-3 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <Scale size={16} />
            Group settings
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm">
              Bias injection
              <span className="ml-2 text-slate-500">{biasShift >= 0 ? `+${biasShift.toFixed(2)}` : biasShift.toFixed(2)}</span>
              <input type="range" min={-0.7} max={0.7} step={0.05} value={biasShift} onChange={(event) => setBiasShift(Number(event.target.value))} className="mt-2 w-full accent-cyan-700" />
              <p className="mt-1 text-xs text-slate-600">Shifts protected/unprotected scores in opposite directions.</p>
            </label>
            <label className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm">
              Threshold mode
              <select value={thresholdMode} onChange={(event) => setThresholdMode(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-2 py-2">
                <option value="global">Global threshold</option>
                <option value="group">Group thresholds</option>
              </select>
            </label>
            <label className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm">
              Threshold strategy
              <span className="ml-2 text-slate-500">{thresholdMode === 'global' ? globalThreshold.toFixed(2) : `${protectedThreshold.toFixed(2)} / ${unprotectedThreshold.toFixed(2)}`}</span>
              <input
                type="range"
                min={0.2}
                max={0.8}
                step={0.02}
                value={globalThreshold}
                onChange={(event) => setGlobalThreshold(Number(event.target.value))}
                className="mt-2 w-full accent-cyan-700"
                disabled={thresholdMode !== 'global'}
              />
              <input
                type="range"
                min={0.2}
                max={0.8}
                step={0.02}
                value={protectedThreshold}
                onChange={(event) => setProtectedThreshold(Number(event.target.value))}
                className="mt-2 w-full accent-cyan-700"
                disabled={thresholdMode === 'global'}
              />
              <input
                type="range"
                min={0.2}
                max={0.8}
                step={0.02}
                value={unprotectedThreshold}
                onChange={(event) => setUnprotectedThreshold(Number(event.target.value))}
                className="mt-2 w-full accent-cyan-700"
                disabled={thresholdMode === 'global'}
              />
            </label>
            <label className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm">
              Fairness objective
              <select value={objective} onChange={(event) => setObjective(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-2 py-2">
                <option value="selection-rate">Selection rate parity</option>
                <option value="fpr-parity">FPR parity</option>
                <option value="tpr-parity">TPR parity</option>
              </select>
            </label>
            <label className="flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm font-bold text-slate-700">
              Show counterfactual tuning hint
              <input type="checkbox" checked={showCounterfactual} onChange={(event) => setShowCounterfactual(event.target.checked)} />
            </label>
          </div>
        </div>

        <div className="grid gap-3">
          <Stat
            label="Objective gap"
            value={percent(objectiveGap)}
            tone={fairnessTone}
            detail={`Current ${objective} imbalance`}
          />
          <Stat
            label="Protected selection"
            value={percent(protectedStats.selectionRate)}
            tone="slate"
            detail={`TPR ${percent(protectedStats.tpr)} · FPR ${percent(protectedStats.fpr)}`}
          />
          <Stat
            label="Unprotected selection"
            value={percent(unprotectedStats.selectionRate)}
            tone="slate"
            detail={`TPR ${percent(unprotectedStats.tpr)} · FPR ${percent(unprotectedStats.fpr)}`}
          />
        </div>
      </section>

      <div className="grid gap-4 xl:grid-cols-2">
        <GroupTable name="Protected group" stats={protectedStats} />
        <GroupTable name="Unprotected group" stats={unprotectedStats} />
      </div>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-3 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <Users size={16} />
          Trade-off check
        </div>
        <div className="grid gap-2 text-sm text-slate-700">
          <p>
            Selection gap: <strong>{percent(gaps.selectionRate)}</strong>, FPR gap: <strong>{percent(gaps.fpr)}</strong>, TPR gap: <strong>{percent(gaps.tpr)}</strong>.
          </p>
          <p>
            Current objective mode is <strong>{objective}</strong> and is currently <strong>{objectiveAligned ? 'better aligned' : 'misaligned'}</strong>.
          </p>
          <p>
            Reminder: improving one fairness constraint can hurt another. Treat these metrics as operating constraints, not optimization maxima.
          </p>
          {counterfactual && (
            <div className={`rounded-lg border p-3 ${objectiveAligned ? 'border-emerald-200 bg-emerald-50' : 'border-amber-200 bg-amber-50'}`}>
              <div className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-slate-600">
                <BadgeCheck size={14} />
                Counterfactual suggestion
              </div>
              <p className="mt-2 text-sm leading-6">
                Try protected threshold <strong>{counterfactual.protected.toFixed(2)}</strong> and unprotected threshold <strong>{counterfactual.unprotected.toFixed(2)}</strong>. {counterfactual.note}
              </p>
            </div>
          )}
        </div>
      </section>

      <section className="rounded-lg border border-cyan-200 bg-cyan-50 p-4 text-sm leading-6 text-cyan-950">
        <div className="flex items-center gap-2 font-black uppercase tracking-wide text-cyan-700">
          <ShieldCheck size={16} />
          Decision guidance
        </div>
        <p className="mt-2">
          {selectedRate} thresholds produce <strong>{selectedPrecision}</strong> for this toy data.
          Use this exercise to force an explicit governance choice for which fairness property you enforce.
        </p>
      </section>

      <section className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm leading-6 text-rose-950">
        <div className="flex items-center gap-2 font-black uppercase tracking-wide text-rose-700">
          <TrendingUp size={16} />
          Common trap
        </div>
        <p className="mt-2">
          If one metric is perfectly matched, another often drifts.
          A safe practice is to document the accepted objective set and monitor all of them each release.
        </p>
      </section>
    </div>
  );
}
