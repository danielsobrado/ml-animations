import React, { useMemo, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  RotateCcw,
  Search,
  SlidersHorizontal,
} from 'lucide-react';

const SCENARIOS = [
  {
    id: 'segment-weakness',
    title: 'Recall is fine overall, but one segment regresses',
    description:
      'The total metric is stable, yet support quality drops for a specific user population in production.',
    rootCause: 'data',
    evidence: {
      data: 'Traffic mix changed after a rollout: more low-activity users now dominate recent logs.',
      training: 'Training and validation looked stable at the end of model fit.',
      evaluation: 'Aggregate validation and calibration looked healthy; a subgroup has rising FN.',
      serving: 'Postprocessing and scaling stayed constant.',
    },
    slices: [
      { segment: 'High-activity users', support: 180, errorRate: 0.13 },
      { segment: 'Low-activity users', support: 72, errorRate: 0.52 },
      { segment: 'New regions', support: 29, errorRate: 0.58 },
    ],
    metrics: { train: 0.93, valid: 0.91, prod: 0.83, falseNeg: 0.38 },
    interventions: [
      {
        id: 'split-audit',
        target: 'data',
        label: 'Recompute production train/serving diagnostics by segment',
        lift: 0.08,
        collateral: 0,
      },
      {
        id: 'harder-regularization',
        target: 'overfit',
        label: 'Add additional regularization and early stop',
        lift: 0.01,
        collateral: -0.02,
      },
      {
        id: 'threshold-only',
        target: 'threshold',
        label: 'Raise global decision threshold',
        lift: 0.0,
        collateral: -0.06,
      },
    ],
  },
  {
    id: 'leakage-learn',
    title: 'Impossibly high validation with disappointing production',
    description: 'Cross-validation seems excellent, but fresh holdout traffic still crashes performance.',
    rootCause: 'training',
    evidence: {
      data: 'A feature is generated from an external identifier that leaks future activity.',
      training: 'Leak-prone feature is learned with the rest of the pipeline before splitting.',
      evaluation: 'Leakage appears strongest in long-tail campaigns.',
      serving: 'Serving contract still receives the same field and appears overconfident.',
    },
    slices: [
      { segment: 'Campaign A', support: 102, errorRate: 0.11 },
      { segment: 'Campaign B', support: 130, errorRate: 0.47 },
      { segment: 'Campaign C', support: 60, errorRate: 0.49 },
    ],
    metrics: { train: 0.96, valid: 0.95, prod: 0.78, falseNeg: 0.29 },
    interventions: [
      {
        id: 'remove-leaky-feature',
        target: 'training',
        label: 'Remove target-derived feature and rebalance data splits',
        lift: 0.11,
        collateral: -0.01,
      },
      {
        id: 'strict-split',
        target: 'training',
        label: 'Fit all transforms inside each split',
        lift: 0.06,
        collateral: 0,
      },
      {
        id: 'model-size-up',
        target: 'capacity',
        label: 'Increase model capacity and fit longer',
        lift: 0.0,
        collateral: -0.03,
      },
    ],
  },
  {
    id: 'serving-noise',
    title: 'Decision rules flap even while data looks stable',
    description: 'Input stats are stable, but predictions vary with non-material request noise.',
    rootCause: 'serving',
    evidence: {
      data: 'Feature names, units, and source coverage did not materially shift.',
      training: 'Weights are unchanged and checkpoints are frozen.',
      evaluation: 'Calibration is acceptable at periodic checkpoints.',
      serving: 'Feature ordering and preprocessing version changed with backend rollout.',
    },
    slices: [
      { segment: 'High confidence', support: 135, errorRate: 0.22 },
      { segment: 'Mid confidence', support: 198, errorRate: 0.31 },
      { segment: 'Low confidence', support: 76, errorRate: 0.37 },
    ],
    metrics: { train: 0.9, valid: 0.89, prod: 0.8, falseNeg: 0.24 },
    interventions: [
      {
        id: 'pin-preprocessing',
        target: 'serving',
        label: 'Lock feature schema, scaling version, and input ordering',
        lift: 0.07,
        collateral: 0.01,
      },
      {
        id: 'recalibrate',
        target: 'calibration',
        label: 'Run post-hoc recalibration and recalibrate decision boundary',
        lift: 0.03,
        collateral: -0.01,
      },
      {
        id: 'wait-for-volume',
        target: 'serving',
        label: 'Collect more production traffic before tuning',
        lift: 0.01,
        collateral: 0,
      },
    ],
  },
];

const STAGE_ORDER = ['data', 'training', 'evaluation', 'serving'];
const STAGE_LABELS = {
  data: 'Data',
  training: 'Training',
  evaluation: 'Evaluation',
  serving: 'Serving',
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function pct(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function computeMetrics(base, intervention, isAlignedGuess, isAlignedIntervention) {
  const lift = isAlignedGuess && isAlignedIntervention ? intervention.lift : 0;
  const collateral = intervention.collateral || 0;
  return {
    train: clamp(base.train - 0.02 + lift * 0.4, 0.62, 1),
    valid: clamp(base.valid + lift * 0.5, 0.63, 1),
    prod: clamp(base.prod + lift + collateral, 0.45, 1),
    precision: clamp(base.prod * 0.88 + (isAlignedGuess ? 0.03 : 0), 0.45, 1),
    recall: clamp(0.73 + lift * 1.8 + collateral * 1.3, 0.45, 0.98),
    falseNeg: clamp(base.falseNeg - lift * 1.3 - collateral * 0.7, 0.02, 0.95),
    gain: lift,
  };
}

function Stat({ label, value, detail, tone = 'slate' }) {
  const classes = {
    slate: 'border-slate-200 bg-white text-slate-900',
    amber: 'border-amber-200 bg-amber-50 text-amber-950',
    emerald: 'border-emerald-200 bg-emerald-50 text-emerald-950',
    rose: 'border-rose-200 bg-rose-50 text-rose-950',
  }[tone];
  return (
    <div className={`rounded-lg border p-4 ${classes}`}>
      <p className="text-xs font-black uppercase tracking-wide text-slate-500">{label}</p>
      <strong className="mt-1 block text-2xl font-black">{value}</strong>
      <span className="mt-2 block text-sm leading-6 text-slate-700">{detail}</span>
    </div>
  );
}

function StageCard({ label, selected, isSignal }) {
  return (
    <div className={`rounded-lg border p-3 text-left text-sm ${selected ? 'border-cyan-300 bg-cyan-50' : 'border-slate-200 bg-white'}`}>
      <div className="flex items-center justify-between gap-2">
        <span className="font-black text-slate-800">{label}</span>
        {selected && <CheckCircle2 size={16} className="text-cyan-900" />}
      </div>
      <span className="text-slate-600">
        {selected
          ? isSignal
            ? 'This check is the strongest candidate for the current suspicion.'
            : 'Useful context, but not the likely primary fault.'
          : 'Not checked yet'}
      </span>
    </div>
  );
}

function SliceChart({ slices, appliedLift }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Slice-level errors</h3>
      <div className="mt-4 space-y-3">
        {slices.map((slice) => {
          const predicted = clamp(slice.errorRate - appliedLift, 0.02, 0.99);
          return (
            <div key={slice.segment} className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="font-semibold text-slate-700">{slice.segment}</span>
                <span className="text-xs font-black text-slate-500">n={slice.support}</span>
              </div>
              <div className="h-4 rounded bg-slate-100">
                <div className="h-4 rounded bg-rose-500" style={{ width: `${predicted * 100}%` }} />
              </div>
              <div className="flex items-center justify-between text-xs text-slate-500">
                <span>after patch: {pct(predicted)}</span>
                <span>before: {pct(slice.errorRate)}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function ModelDebugging() {
  const [scenarioId, setScenarioId] = useState(SCENARIOS[0].id);
  const [checked, setChecked] = useState({
    data: false,
    training: false,
    evaluation: false,
    serving: false,
  });
  const [rootCauseGuess, setRootCauseGuess] = useState('data');
  const [interventionId, setInterventionId] = useState(SCENARIOS[0].interventions[0].id);

  const scenario = useMemo(
    () => SCENARIOS.find((entry) => entry.id === scenarioId) || SCENARIOS[0],
    [scenarioId],
  );
  const intervention = useMemo(
    () => scenario.interventions.find((entry) => entry.id === interventionId) || scenario.interventions[0],
    [interventionId, scenario.interventions],
  );

  const causeOptions = useMemo(() => STAGE_ORDER.slice(), []);
  const checkedCount = STAGE_ORDER.filter((stage) => checked[stage]).length;
  const alignedGuess = rootCauseGuess === scenario.rootCause;
  const alignedIntervention = intervention.target === scenario.rootCause;
  const localization = alignedGuess ? Math.min(100, Math.round((checkedCount / STAGE_ORDER.length) * 100)) : 0;
  const metrics = useMemo(
    () => computeMetrics(scenario.metrics, intervention, alignedGuess, alignedIntervention),
    [scenario.metrics, intervention, alignedGuess, alignedIntervention],
  );

  const reset = () => {
    setChecked({
      data: false,
      training: false,
      evaluation: false,
      serving: false,
    });
    setRootCauseGuess(scenario.rootCause);
    setInterventionId(scenario.interventions[0].id);
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Reliability diagnostics</p>
            <h1 className="mt-1 text-2xl font-black text-slate-950">Model Debugging</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Follow a practical loop: localize the failure first, then test one hypothesis and one intervention at a time.
            </p>
          </div>
          <button
            type="button"
            onClick={reset}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-bold text-slate-800"
          >
            <RotateCcw size={16} />
            Reset scenario
          </button>
        </div>

        <div className="mt-5 grid gap-2">
          <label className="text-xs font-black uppercase tracking-wide text-slate-600">Choose an incident mode</label>
          <div className="grid gap-2 lg:grid-cols-3">
            {SCENARIOS.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => {
                  setScenarioId(item.id);
                  setChecked({
                    data: false,
                    training: false,
                    evaluation: false,
                    serving: false,
                  });
                  setRootCauseGuess(item.rootCause);
                  setInterventionId(item.interventions[0].id);
                }}
                className={`rounded-lg border px-3 py-2 text-left text-sm font-black transition ${
                  scenarioId === item.id
                    ? 'border-sky-600 bg-sky-600 text-white'
                    : 'border-slate-300 bg-slate-50 text-slate-800'
                }`}
              >
                {item.title}
              </button>
            ))}
          </div>
          <p className="text-sm leading-6 text-slate-700">{scenario.description}</p>
        </div>
      </section>

      <div className="grid gap-3 md:grid-cols-4">
        <Stat label="Checks completed" value={`${checkedCount}/${STAGE_ORDER.length}`} detail="Pipeline stages inspected" tone={checkedCount ? 'emerald' : 'slate'} />
        <Stat label="Root-cause localization" value={`${localization}%`} detail={alignedGuess ? 'Evidence now points to one layer' : 'Need more evidence signals'} tone={alignmentTone(localization)} />
        <Stat label="Production recall estimate" value={pct(metrics.recall)} detail={`False-negative rate ${pct(metrics.falseNeg, 0)}`} tone={metrics.recall >= 0.82 ? 'emerald' : metrics.recall >= 0.76 ? 'amber' : 'rose'} />
        <Stat label="Intervention impact" value={alignedIntervention ? `+${pct(metrics.gain)}` : 'No clear lift'} detail={alignedIntervention ? 'Cause and mitigation align' : 'Likely partial or noisy change'} tone={alignedIntervention ? 'emerald' : 'amber'} />
      </div>

      <section className="grid gap-4 xl:grid-cols-[1fr_1.1fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <Search size={16} />
            Pipeline checklist
          </div>
          <div className="grid gap-2">
            {STAGE_ORDER.map((stage) => (
              <button key={stage} type="button" onClick={() => setChecked((state) => ({ ...state, [stage]: !state[stage] }))} className="text-left">
                <StageCard label={STAGE_LABELS[stage]} selected={checked[stage]} isSignal={scenario.rootCause === stage && alignedGuess} />
              </button>
            ))}
          </div>

          {Object.values(checked).some(Boolean) && (
            <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
              <p className="font-black uppercase text-xs tracking-wide text-slate-500">Evidence snippets</p>
              <ul className="mt-2 space-y-1">
                {STAGE_ORDER.filter((stage) => checked[stage]).map((stage) => (
                  <li key={stage}>- {STAGE_LABELS[stage]}: {scenario.evidence[stage]}</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <SlidersHorizontal size={16} />
            Cause and intervention
          </div>

          <label className="mb-4 block text-sm font-bold text-slate-700">
            Root-cause guess
            <select
              value={rootCauseGuess}
              onChange={(event) => setRootCauseGuess(event.target.value)}
              className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-3 py-2"
            >
              {causeOptions.map((cause) => (
                <option key={cause} value={cause}>
                  {cause}
                </option>
              ))}
            </select>
          </label>

          <div className="grid gap-2">
            {scenario.interventions.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setInterventionId(item.id)}
                className={`rounded-lg border px-3 py-2 text-left text-sm ${
                  interventionId === item.id
                    ? 'border-emerald-600 bg-emerald-50 text-emerald-950'
                    : 'border-slate-300 bg-white text-slate-800'
                }`}
              >
                <strong>{item.label}</strong>
                <p className="mt-1 text-xs text-slate-600">{item.target}</p>
              </button>
            ))}
          </div>

          <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
            <p className="font-black uppercase tracking-wide text-slate-500 text-xs">Debug rule</p>
            <p className="mt-2 leading-6">
              {alignedGuess && alignedIntervention
                ? 'Cause and intervention are aligned. Run one intervention, re-measure all stages, and then re-run checks.'
                : 'Try a different root-cause guess or a targeted intervention before changing modeling complexity.'}
            </p>
          </div>
        </div>
      </section>

      <div className="grid gap-4 xl:grid-cols-2">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <Activity size={16} />
            Before/after metrics
          </div>
          <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-3">
            <div className="grid gap-2 sm:grid-cols-3 text-sm">
              <span>Train: {pct(scenario.metrics.train)} {'->'} {pct(metrics.train)}</span>
              <span>Valid: {pct(scenario.metrics.valid)} {'->'} {pct(metrics.valid)}</span>
              <span>Prod: {pct(scenario.metrics.prod)} {'->'} {pct(metrics.prod)}</span>
            </div>
            <div className="mt-3 text-sm">
              <span>Precision: {pct(metrics.precision)}</span>
              <span className="ml-4">Recall: {pct(metrics.recall)}</span>
              <span className="ml-4">FN: {pct(metrics.falseNeg, 0)}</span>
            </div>
          </div>
        </section>
        <SliceChart slices={scenario.slices} appliedLift={alignedGuess && alignedIntervention ? intervention.lift : 0.02} />
      </div>

      <section className="grid gap-3 xl:grid-cols-3">
        <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-4 text-sm leading-6 text-cyan-900">
          <h3 className="text-sm font-black uppercase tracking-wide text-cyan-700">First move</h3>
          <p className="mt-2">
            Validate data pipeline and serving contracts before touching model weights.
            Most production incidents are not random model failures; they are boundary mismatches.
          </p>
        </div>
        <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm leading-6 text-rose-900">
          <h3 className="text-sm font-black uppercase tracking-wide text-rose-700">Common anti-pattern</h3>
          <p className="mt-2">
            Tuning one global threshold for every segment can appear to fix one metric while hiding subgroup drift.
          </p>
        </div>
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm leading-6 text-emerald-900">
          <h3 className="text-sm font-black uppercase tracking-wide text-emerald-700">Lesson checkpoint</h3>
          <p className="mt-2">
            Consider the run complete when one slice improves and at least two non-target slices remain stable.
          </p>
        </div>
      </section>

      <div className="rounded-lg border border-slate-200 bg-slate-900 p-4 text-sm text-white">
        <div className="flex items-center gap-2 font-black uppercase tracking-wide text-slate-200">
          <AlertTriangle size={16} />
          Quick check
        </div>
        <p className="mt-2">
          Do not skip pipeline checks when symptoms look urgent. Slice first, then stage, then mitigation.
        </p>
      </div>
    </div>
  );
}

function alignmentTone(localization) {
  if (localization >= 100) return 'emerald';
  if (localization >= 50) return 'amber';
  return 'slate';
}
