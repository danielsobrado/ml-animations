import React, { useMemo, useState } from 'react';
import { BarChart3, CheckCircle2, GitBranch, SlidersHorizontal } from 'lucide-react';
import { coreMlLessons } from './lessons';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const sampleScores = [
  { score: 0.94, actual: 1 },
  { score: 0.87, actual: 1 },
  { score: 0.76, actual: 0 },
  { score: 0.68, actual: 1 },
  { score: 0.61, actual: 0 },
  { score: 0.55, actual: 1 },
  { score: 0.47, actual: 0 },
  { score: 0.39, actual: 1 },
  { score: 0.31, actual: 0 },
  { score: 0.22, actual: 0 },
  { score: 0.16, actual: 0 },
  { score: 0.09, actual: 1 },
];

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function metricSummary(threshold) {
  const counts = sampleScores.reduce((acc, item) => {
    const predicted = item.score >= threshold ? 1 : 0;
    if (predicted === 1 && item.actual === 1) acc.tp += 1;
    if (predicted === 1 && item.actual === 0) acc.fp += 1;
    if (predicted === 0 && item.actual === 1) acc.fn += 1;
    if (predicted === 0 && item.actual === 0) acc.tn += 1;
    return acc;
  }, { tp: 0, fp: 0, fn: 0, tn: 0 });

  const precision = counts.tp + counts.fp === 0 ? 0 : counts.tp / (counts.tp + counts.fp);
  const recall = counts.tp + counts.fn === 0 ? 0 : counts.tp / (counts.tp + counts.fn);
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  const accuracy = (counts.tp + counts.tn) / sampleScores.length;

  return { ...counts, precision, recall, f1, accuracy };
}

function linePath(points, width, height) {
  return points.map((point, index) => {
    const x = 20 + (point.x / 10) * (width - 40);
    const y = height - 20 - point.y * (height - 48);
    return `${index === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(' ');
}

function MiniLineChart({ series }) {
  const width = 520;
  const height = 210;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full rounded border border-slate-200 bg-white">
      <line x1="20" x2="500" y1="180" y2="180" stroke="#d7d0bf" />
      <line x1="20" x2="20" y1="24" y2="180" stroke="#d7d0bf" />
      {series.map((item) => (
        <path key={item.label} d={linePath(item.points, width, height)} fill="none" stroke={item.color} strokeWidth="4" strokeLinecap="round" />
      ))}
      {series.map((item, index) => (
        <g key={item.label} transform={`translate(${28 + index * 150} 24)`}>
          <rect width="10" height="10" fill={item.color} rx="2" />
          <text x="16" y="10" fontSize="12" fill="#334155">{item.label}</text>
        </g>
      ))}
    </svg>
  );
}

function StatTile({ label, value, tone = 'slate' }) {
  const toneClass = {
    slate: 'border-slate-200 bg-white text-slate-900',
    blue: 'border-blue-200 bg-blue-50 text-blue-950',
    green: 'border-emerald-200 bg-emerald-50 text-emerald-950',
    orange: 'border-orange-200 bg-orange-50 text-orange-950',
  }[tone];

  return (
    <div className={`rounded-lg border p-3 ${toneClass}`}>
      <div className="text-xs font-semibold uppercase">{label}</div>
      <div className="mt-1 text-2xl font-bold">{value}</div>
    </div>
  );
}

function ControlPanel({ controls, values, onChange }) {
  return (
    <div className="space-y-4 rounded-lg border border-slate-200 bg-white p-4">
      {controls.map((control) => (
        <label key={control.id} className="block">
          <div className="mb-2 flex items-center justify-between gap-4 text-sm font-semibold text-slate-800">
            <span>{control.label}</span>
            <span>{values[control.id]}</span>
          </div>
          <input
            type="range"
            min={control.min}
            max={control.max}
            step={control.step}
            value={values[control.id]}
            onChange={(event) => onChange(control.id, Number(event.target.value))}
            className="w-full accent-blue-700"
          />
        </label>
      ))}
    </div>
  );
}

function TrainSplitVisual({ values }) {
  const validation = values.validation;
  const test = values.test;
  const train = Math.max(0, 100 - validation - test);
  const dots = Array.from({ length: 50 }, (_, index) => {
    const pct = (index + 1) * 2;
    if (pct <= train) return 'train';
    if (pct <= train + validation) return 'validation';
    return 'test';
  });

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <StatTile label="Train" value={`${train}%`} tone="blue" />
        <StatTile label="Validation" value={`${validation}%`} tone="orange" />
        <StatTile label="Test" value={`${test}%`} tone="green" />
      </div>
      <div className="grid grid-cols-10 gap-2 rounded-lg border border-slate-200 bg-slate-50 p-4">
        {dots.map((kind, index) => (
          <div
            key={index}
            className={`h-7 rounded ${kind === 'train' ? 'bg-blue-600' : kind === 'validation' ? 'bg-orange-500' : 'bg-emerald-600'}`}
            title={kind}
          />
        ))}
      </div>
    </div>
  );
}

function OverfittingVisual({ values }) {
  const complexity = values.complexity;
  const points = Array.from({ length: 10 }, (_, index) => {
    const x = index + 1;
    const train = Math.max(0.08, 0.75 - x * 0.06);
    const validation = 0.22 + Math.pow((x - 5.2) / 7, 2) + Math.max(0, x - 6) * 0.045;
    return { x, train, validation: Math.min(validation, 0.92) };
  });
  const chosen = points[complexity - 1];

  return (
    <div className="space-y-4">
      <MiniLineChart
        series={[
          { label: 'Training error', color: '#2563eb', points: points.map((point) => ({ x: point.x, y: point.train })) },
          { label: 'Validation error', color: '#ea580c', points: points.map((point) => ({ x: point.x, y: point.validation })) },
        ]}
      />
      <div className="grid grid-cols-2 gap-3">
        <StatTile label="Train error" value={chosen.train.toFixed(2)} tone="blue" />
        <StatTile label="Validation error" value={chosen.validation.toFixed(2)} tone={complexity > 6 ? 'orange' : 'green'} />
      </div>
    </div>
  );
}

function LogisticVisual({ values }) {
  const xs = [-3, -2, -1, 0, 1, 2, 3];
  const points = xs.map((x) => ({ x, p: sigmoid(values.weight * x + values.bias) }));

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-7 gap-2">
        {points.map((point) => (
          <div key={point.x} className="rounded-lg border border-slate-200 bg-white p-2 text-center">
            <div className="text-xs font-semibold text-slate-500">x={point.x}</div>
            <div className="mt-1 text-lg font-bold text-slate-900">{point.p.toFixed(2)}</div>
            <div className={`mt-2 h-2 rounded ${point.p >= values.threshold ? 'bg-blue-600' : 'bg-slate-300'}`} />
          </div>
        ))}
      </div>
      <MiniLineChart
        series={[
          { label: 'Sigmoid probability', color: '#2563eb', points: points.map((point) => ({ x: point.x + 4, y: point.p })) },
          { label: 'Threshold', color: '#ea580c', points: [{ x: 1, y: values.threshold }, { x: 7, y: values.threshold }] },
        ]}
      />
    </div>
  );
}

function MetricsVisual({ values }) {
  const metrics = metricSummary(values.threshold);
  const pct = (value) => `${Math.round(value * 100)}%`;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <StatTile label="True positives" value={metrics.tp} tone="green" />
        <StatTile label="False positives" value={metrics.fp} tone="orange" />
        <StatTile label="False negatives" value={metrics.fn} tone="orange" />
        <StatTile label="True negatives" value={metrics.tn} tone="blue" />
      </div>
      <div className="grid grid-cols-4 gap-3">
        <StatTile label="Precision" value={pct(metrics.precision)} />
        <StatTile label="Recall" value={pct(metrics.recall)} />
        <StatTile label="F1" value={pct(metrics.f1)} />
        <StatTile label="Accuracy" value={pct(metrics.accuracy)} />
      </div>
    </div>
  );
}

function RegularizationVisual({ values }) {
  const lambda = values.lambda;
  const rawWeights = [2.8, -1.9, 1.2, -0.8, 0.55];
  const weights = rawWeights.map((weight) => weight / (1 + lambda * 0.22));
  const dataLoss = 0.24 + lambda * 0.018;
  const penalty = lambda * weights.reduce((sum, weight) => sum + weight * weight, 0) * 0.015;

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <div className="mb-3 text-sm font-semibold text-slate-800">Coefficient shrinkage</div>
        <div className="space-y-3">
          {weights.map((weight, index) => (
            <div key={index} className="grid grid-cols-[56px_1fr_56px] items-center gap-3 text-sm">
              <span className="font-semibold text-slate-600">w{index + 1}</span>
              <div className="h-3 rounded bg-slate-100">
                <div
                  className={`h-3 rounded ${weight >= 0 ? 'bg-blue-600' : 'bg-orange-500'}`}
                  style={{ width: `${Math.min(100, Math.abs(weight) * 28)}%` }}
                />
              </div>
              <span className="text-right font-semibold text-slate-800">{weight.toFixed(2)}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="grid grid-cols-3 gap-3">
        <StatTile label="Data loss" value={dataLoss.toFixed(2)} tone="blue" />
        <StatTile label="Penalty" value={penalty.toFixed(2)} tone="orange" />
        <StatTile label="Total" value={(dataLoss + penalty).toFixed(2)} tone="green" />
      </div>
    </div>
  );
}

function Playground({ lesson, values }) {
  if (lesson.id === 'train-validation-test-split') return <TrainSplitVisual values={values} />;
  if (lesson.id === 'overfitting') return <OverfittingVisual values={values} />;
  if (lesson.id === 'logistic-regression') return <LogisticVisual values={values} />;
  if (lesson.id === 'classification-metrics') return <MetricsVisual values={values} />;
  return <RegularizationVisual values={values} />;
}

export default function CoreMlLesson({ lessonId }) {
  const lesson = coreMlLessons[lessonId];
  const [activeTab, setActiveTab] = useState('concept');
  const [values, setValues] = useState(() => Object.fromEntries(
    lesson.controls.map((control) => [control.id, control.defaultValue]),
  ));

  const tabs = [
    { id: 'concept', label: 'Concept', icon: GitBranch },
    { id: 'playground', label: 'Playground', icon: SlidersHorizontal },
    { id: 'check', label: 'Check', icon: CheckCircle2 },
  ];

  const mechanismCopy = useMemo(() => lesson.mechanism.join(' '), [lesson.mechanism]);

  return (
    <div className="min-h-full bg-[#fbf8f1] text-slate-900">
      <nav className="sticky top-0 z-10 border-b border-slate-200 bg-[#fbf8f1]/95 backdrop-blur">
        <div className="flex gap-2 overflow-x-auto px-4 py-3">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition ${
                activeTab === tab.id
                  ? 'bg-slate-900 text-white'
                  : 'bg-white text-slate-700 hover:bg-slate-100'
              }`}
            >
              <tab.icon size={17} />
              {tab.label}
            </button>
          ))}
        </div>
      </nav>

      <section className="mx-auto max-w-6xl px-4 py-8">
        <div className="mb-6">
          <div className="text-xs font-bold uppercase text-blue-800">{lesson.eyebrow}</div>
          <h2 className="mt-2 text-3xl font-bold text-slate-950">{lesson.title}</h2>
          <p className="mt-3 max-w-3xl text-lg leading-8 text-slate-700">{lesson.summary}</p>
        </div>

        {activeTab === 'concept' && (
          <div className="grid gap-5 lg:grid-cols-[1.1fr_0.9fr]">
            <div className="rounded-lg border border-slate-200 bg-white p-5">
              <div className="mb-3 text-sm font-bold uppercase text-slate-500">Intuition</div>
              <p className="text-lg leading-8 text-slate-800">{lesson.intuition}</p>
              <div className="mt-5 rounded-lg bg-slate-950 p-4 font-mono text-sm text-slate-100">
                {lesson.equation}
              </div>
            </div>
            <div className="rounded-lg border border-slate-200 bg-white p-5">
              <div className="mb-3 flex items-center gap-2 text-sm font-bold uppercase text-slate-500">
                <BarChart3 size={16} />
                Mechanism
              </div>
              <ol className="space-y-3">
                {lesson.mechanism.map((step, index) => (
                  <li key={step} className="grid grid-cols-[32px_1fr] gap-3 text-slate-800">
                    <span className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-700 text-sm font-bold text-white">{index + 1}</span>
                    <span className="leading-7">{step}</span>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        )}

        {activeTab === 'playground' && (
          <div className="grid gap-5 lg:grid-cols-[320px_1fr]">
            <ControlPanel
              controls={lesson.controls}
              values={values}
              onChange={(id, value) => setValues((current) => ({ ...current, [id]: value }))}
            />
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <Playground lesson={lesson} values={values} />
            </div>
          </div>
        )}

        {activeTab === 'check' && (
          <AssessmentPanel
            lessonId={lesson.id}
            title="Check yourself"
            legacyQuestion={lesson.check}
            legacyAnswer={lesson.answer}
            legacyExplanation={mechanismCopy}
          />
        )}
      </section>
    </div>
  );
}
