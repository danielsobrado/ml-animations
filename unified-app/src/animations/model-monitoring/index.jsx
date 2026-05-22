import React, { useMemo, useState } from 'react';
import { Activity, AlertTriangle, RefreshCw, ShieldCheck } from 'lucide-react';

const SCENARIOS = {
  'covariate-shift': {
    label: 'Covariate shift',
    drift: 0.11,
    precisionBase: 0.82,
    recallBase: 0.74,
    calibrationBase: 0.08,
    latencyBase: 0.24,
    throughputBase: 1.0,
  },
  'label-shift': {
    label: 'Label shift',
    drift: 0.08,
    precisionBase: 0.84,
    recallBase: 0.68,
    calibrationBase: 0.11,
    latencyBase: 0.22,
    throughputBase: 1.0,
  },
  'serving-volatility': {
    label: 'Serving volatility',
    drift: 0.09,
    precisionBase: 0.86,
    recallBase: 0.76,
    calibrationBase: 0.05,
    latencyBase: 0.34,
    throughputBase: 0.79,
  },
  'concept-drift': {
    label: 'Concept drift',
    drift: 0.14,
    precisionBase: 0.8,
    recallBase: 0.64,
    calibrationBase: 0.12,
    latencyBase: 0.25,
    throughputBase: 0.96,
  },
};

const HOURS = 16;
const BASELINE_VALUES = {
  precision: 0.78,
  recall: 0.73,
  calibration: 0.95,
  latency: 0.25,
  throughput: 1.0,
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function buildSeries(scenario, trend, strictness) {
  return Array.from({ length: HOURS }, (_, index) => {
    const t = index + 1;
    const fatigue = clamp((t - 4) / 12, 0, 1);
    const drift = clamp(
      scenario.drift + fatigue * 0.22 * trend + Math.sin(t / 2) * 0.008 * (1 + trend),
      0.01,
      0.45,
    );
    const precision = clamp(
      scenario.precisionBase - drift * 0.45 - Math.max(0, fatigue - 0.35) * 0.06,
      0.45,
      0.95,
    );
    const recall = clamp(
      scenario.recallBase - drift * 0.38,
      0.44,
      0.95,
    );
    const calibration = clamp(
      scenario.calibrationBase + drift * 0.35 + 0.07 * Math.sin(t / 3),
      0.01,
      0.2,
    );
    const latency = clamp(
      scenario.latencyBase + drift * 1.8 + scenario.throughputBase * 0 + fatigue * 0.10 * strictness,
      0.1,
      1.2,
    );
    const throughput = clamp(
      scenario.throughputBase - fatigue * 0.17 - drift * 0.4,
      0.4,
      1.4,
    );
    return {
      hour: t,
      drift,
      precision,
      recall,
      calibration,
      latency,
      throughput,
    };
  });
}

function scoreToPercent(value, digits = 0) {
  return `${(value * 100).toFixed(digits)}%`;
}

function Stat({ label, value, tone = 'slate', detail }) {
  const toneClass = {
    slate: 'border-slate-200 bg-white text-slate-900',
    emerald: 'border-emerald-200 bg-emerald-50 text-emerald-950',
    amber: 'border-amber-200 bg-amber-50 text-amber-950',
    rose: 'border-rose-200 bg-rose-50 text-rose-950',
  }[tone];
  return (
    <div className={`rounded-lg border p-4 ${toneClass}`}>
      <p className="text-xs font-black uppercase tracking-wide text-slate-500">{label}</p>
      <strong className="mt-1 block text-2xl font-black">{value}</strong>
      {detail ? <span className="mt-2 block text-sm leading-6 text-slate-700">{detail}</span> : null}
    </div>
  );
}

function signalColor(value, low, high) {
  if (value < low) return 'bg-emerald-500';
  if (value < high) return 'bg-amber-500';
  return 'bg-rose-500';
}

function TimelineCard({ title, points, accessor, threshold, unit = '' }) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-3">
      <p className="mb-2 text-xs font-black uppercase tracking-wide text-slate-500">{title}</p>
      <div className="space-y-2">
        {points.map((point) => (
          <div key={`${title}-${point.hour}`} className="grid grid-cols-[2.25rem_1fr_3.2rem] items-center gap-2 text-xs">
            <span className="font-black text-slate-500">{String(point.hour).padStart(2, '0')}h</span>
            <div className="h-3 rounded bg-slate-100">
              <div
                className={`h-3 rounded ${signalColor(point[accessor], threshold.low, threshold.high)}`}
                style={{ width: `${clamp(point[accessor] * (unit === 'ms' ? 220 : 100), 0, 100)}%` }}
              />
            </div>
            <span className="text-right text-slate-700 font-black">
              {unit === 'ms' ? `${point[accessor].toFixed(2)}${unit}` : `${(point[accessor] * 100).toFixed(1)}%`}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

export default function ModelMonitoring() {
  const [scenarioId, setScenarioId] = useState('covariate-shift');
  const [strictness, setStrictness] = useState(100);
  const [trendShock, setTrendShock] = useState(100);
  const [monitorDrift, setMonitorDrift] = useState(true);
  const [monitorPerf, setMonitorPerf] = useState(true);
  const [monitorCal, setMonitorCal] = useState(true);
  const [monitorLatency, setMonitorLatency] = useState(true);
  const [playbookOpen, setPlaybookOpen] = useState('investigate');

  const scenario = SCENARIOS[scenarioId];
  const strictFactor = strictness / 100;
  const trend = trendShock / 100;
  const series = useMemo(() => buildSeries(scenario, trend, strictFactor), [scenario, trend, strictFactor]);
  const latest = series[series.length - 1];
  const recent = series.slice(series.length - 6);
  const avgPrecision = recent.reduce((acc, point) => acc + point.precision, 0) / recent.length;
  const avgRecall = recent.reduce((acc, point) => acc + point.recall, 0) / recent.length;
  const avgCalibration = recent.reduce((acc, point) => acc + point.calibration, 0) / recent.length;
  const avgLatency = recent.reduce((acc, point) => acc + point.latency, 0) / recent.length;
  const avgThroughput = recent.reduce((acc, point) => acc + point.throughput, 0) / recent.length;

  const driftAlert = latest.drift > (monitorDrift ? (0.18 * strictFactor) : 0);
  const precisionAlert = avgPrecision < (monitorPerf ? 0.7 : 2);
  const recallAlert = avgRecall < (monitorPerf ? 0.62 : 2);
  const calibrationAlert = avgCalibration > (monitorCal ? 0.13 : 10);
  const latencyAlert = avgLatency > (monitorLatency ? 0.35 * strictFactor : 10);

  const activeAlerts = [driftAlert, precisionAlert, recallAlert, calibrationAlert, latencyAlert].filter(Boolean).length;
  const healthScore = clamp(100 - activeAlerts * 16 - (latest.drift * 80), 5, 100);

  const recommendations = {
    investigate: [
      'Confirm that data contracts, schemas, and feature stores are unchanged since monitoring started.',
      'Check upstream ingestion for null-rate spikes and parser exceptions.',
      'Pause automated rollout if precision and recall dropped together.',
    ],
    retrain: [
      'Run a distribution comparison on features and labels by source system.',
      'Open a bias audit if a specific segment drives the drift.',
      'Prepare a controlled retraining experiment with holdout and calibration recalibration.',
    ],
    rollback: [
      'Disable the most brittle feature branch and keep conservative defaults.',
      'Roll back to last known-good model version if serving latency is unstable.',
      'Notify product and run post-incident monitoring triage.',
    ],
  }[playbookOpen];

  const reset = () => {
    setScenarioId('covariate-shift');
    setStrictness(100);
    setTrendShock(100);
    setMonitorDrift(true);
    setMonitorPerf(true);
    setMonitorCal(true);
    setMonitorLatency(true);
    setPlaybookOpen('investigate');
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Operations</p>
            <h1 className="mt-1 text-2xl font-black text-slate-950">Model Monitoring</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Track post-deploy behavior over time, tune alert sensitivity, and connect signal changes to concrete actions.
            </p>
          </div>
          <button
            type="button"
            onClick={reset}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-bold text-slate-800"
          >
            <RefreshCw size={16} />
            Reset signals
          </button>
        </div>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-4 grid gap-3 md:grid-cols-[1.3fr_1fr]">
          <label className="text-sm font-bold text-slate-700">
            Scenario
            <select value={scenarioId} onChange={(event) => setScenarioId(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-3 py-2">
              {Object.entries(SCENARIOS).map(([id, item]) => (
                <option key={id} value={id}>{item.label}</option>
              ))}
            </select>
          </label>
          <div className="grid gap-3 text-sm">
            <label className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 font-bold">
              Alert strictness
              <span>{strictness}%</span>
            </label>
            <input type="range" min={60} max={160} value={strictness} onChange={(event) => setStrictness(Number(event.target.value))} className="accent-cyan-700" />
            <label className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 font-bold">
              Drift ramp
              <span>{trendShock}%</span>
            </label>
            <input type="range" min={60} max={170} value={trendShock} onChange={(event) => setTrendShock(Number(event.target.value))} className="accent-cyan-700" />
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-4">
          <label className="flex items-center justify-between gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-3 text-sm font-bold text-slate-700">
            Monitor drift
            <input type="checkbox" checked={monitorDrift} onChange={(event) => setMonitorDrift(event.target.checked)} />
          </label>
          <label className="flex items-center justify-between gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-3 text-sm font-bold text-slate-700">
            Monitor precision/recall
            <input type="checkbox" checked={monitorPerf} onChange={(event) => setMonitorPerf(event.target.checked)} />
          </label>
          <label className="flex items-center justify-between gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-3 text-sm font-bold text-slate-700">
            Monitor calibration
            <input type="checkbox" checked={monitorCal} onChange={(event) => setMonitorCal(event.target.checked)} />
          </label>
          <label className="flex items-center justify-between gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-3 text-sm font-bold text-slate-700">
            Monitor latency
            <input type="checkbox" checked={monitorLatency} onChange={(event) => setMonitorLatency(event.target.checked)} />
          </label>
        </div>
      </section>

      <section className="grid gap-3 md:grid-cols-4">
        <Stat label="Scenario health" value={`${Math.round(healthScore)} / 100`} tone={healthScore >= 75 ? 'emerald' : healthScore >= 52 ? 'amber' : 'rose'} detail="recent 6-hour aggregate score" />
        <Stat label="Precision" value={scoreToPercent(avgPrecision, 1)} tone={precisionAlert ? 'amber' : 'emerald'} detail={`baseline ${(BASELINE_VALUES.precision * 100).toFixed(0)}%`} />
        <Stat label="Recall" value={scoreToPercent(avgRecall, 1)} tone={recallAlert ? 'amber' : 'emerald'} detail={`baseline ${(BASELINE_VALUES.recall * 100).toFixed(0)}%`} />
        <Stat label="False calibration error" value={avgCalibration.toFixed(3)} tone={calibrationAlert ? 'rose' : 'emerald'} detail={`latest ${latest.calibration.toFixed(3)}`} />
      </section>

      <div className="grid gap-4 xl:grid-cols-2">
        <TimelineCard
          title="Input drift"
          points={recent}
          accessor="drift"
          threshold={{ low: 0.08, high: 0.15 }}
        />
        <TimelineCard
          title="Throughput utilization"
          points={recent}
          accessor="throughput"
          threshold={{ low: 0.65, high: 1.0 }}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <TimelineCard
          title="Precision vs alert windows"
          points={recent}
          accessor="precision"
          threshold={{ low: 0.64, high: 0.72 }}
        />
        <TimelineCard
          title="Latency (s)"
          points={recent}
          accessor="latency"
          threshold={{ low: 0.30, high: 0.45 }}
          unit="ms"
        />
      </div>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-3 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <AlertTriangle size={16} />
          Active alerts
        </div>
        <p className="text-sm leading-6 text-slate-700">
          {activeAlerts === 0
            ? 'No blocking alert conditions crossed with current strictness.'
            : `${activeAlerts} alert condition(s) currently active.`}
        </p>
        <div className="mt-3 grid gap-2 sm:grid-cols-3">
          <span className={`rounded-lg border px-3 py-2 text-sm ${driftAlert ? 'border-rose-300 bg-rose-50 text-rose-900' : 'border-slate-200 bg-slate-50 text-slate-700'}`}>
            Drift monitor: {driftAlert ? 'breach' : 'ok'}
          </span>
          <span className={`rounded-lg border px-3 py-2 text-sm ${precisionAlert ? 'border-rose-300 bg-rose-50 text-rose-900' : 'border-slate-200 bg-slate-50 text-slate-700'}`}>
            Precision monitor: {precisionAlert ? 'breach' : 'ok'}
          </span>
          <span className={`rounded-lg border px-3 py-2 text-sm ${latencyAlert ? 'border-rose-300 bg-rose-50 text-rose-900' : 'border-slate-200 bg-slate-50 text-slate-700'}`}>
            Latency monitor: {latencyAlert ? 'breach' : 'ok'}
          </span>
        </div>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <ShieldCheck size={16} />
          Recovery playbook
        </div>
        <div className="grid gap-3 md:grid-cols-3">
          {Object.entries({
            investigate: 'Investigate',
            retrain: 'Retrain/refresh',
            rollback: 'Rollback',
          }).map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => setPlaybookOpen(id)}
              className={`rounded-lg border px-3 py-2 text-sm font-bold ${playbookOpen === id ? 'border-cyan-500 bg-cyan-50 text-cyan-950' : 'border-slate-200 bg-slate-50 text-slate-800'}`}
            >
              {label}
            </button>
          ))}
        </div>
        <ul className="mt-4 list-disc space-y-2 pl-6 text-sm leading-6 text-slate-700">
          {recommendations.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      </section>

      <section className="rounded-lg border border-cyan-200 bg-cyan-50 p-5">
        <div className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-cyan-700">
          <Activity size={16} />
          Operational reminder
        </div>
        <p className="mt-3 text-sm leading-6 text-cyan-950">
          Monitoring is not about one signal. It is about consistency across data, performance, calibration, and serving cost.
          A single spike often reflects contract drift, not model complexity alone.
        </p>
      </section>
    </div>
  );
}
