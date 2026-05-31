import React, { useMemo, useState } from 'react';
import { AlertTriangle, BarChart3, RotateCcw, ShieldCheck, Shuffle, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';
import { assignByMode, driftGap, positiveRate } from './trainValidationTestSplitModel.js';

const MODES = {
  random: {
    label: 'Random',
    detail: 'Good default when rows are exchangeable and class proportions are balanced.',
  },
  stratified: {
    label: 'Stratified',
    detail: 'Keeps positive and negative labels represented across train, validation, and test.',
  },
  time: {
    label: 'Time ordered',
    detail: 'Keeps future rows out of training when deployment predicts future events.',
  },
};

function Stat({ label, value, detail }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <p className="text-xs font-black uppercase tracking-wide text-slate-500">{label}</p>
      <strong className="mt-1 block text-2xl font-black text-slate-950">{value}</strong>
      <span className="text-sm text-slate-600">{detail}</span>
    </div>
  );
}

function SplitColumn({ label, rows, tone }) {
  const toneClass = {
    train: 'border-blue-200 bg-blue-50',
    validation: 'border-amber-200 bg-amber-50',
    test: 'border-emerald-200 bg-emerald-50',
  }[tone];

  return (
    <div className={`rounded-lg border p-4 ${toneClass}`}>
      <div className="flex items-center justify-between gap-3">
        <strong className="text-sm font-black uppercase tracking-wide text-slate-800">{label}</strong>
        <span className="font-mono text-xs font-black text-slate-600">{rows.length} rows</span>
      </div>
      <div className="mt-4 grid gap-2">
        {rows.map((row) => (
          <div key={row.id} className="grid grid-cols-[32px_1fr_36px] items-center gap-2 rounded border border-white bg-white px-2 py-1 text-xs">
            <span className="font-mono font-black text-slate-700">{row.id}</span>
            <span className="text-slate-600">t{row.time} / seg {row.segment}</span>
            <span className={`rounded px-1.5 py-0.5 text-center font-black ${row.y ? 'bg-rose-100 text-rose-700' : 'bg-slate-100 text-slate-600'}`}>
              y{row.y}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function TrainValidationTestSplitAnimation() {
  const [mode, setMode] = useState('stratified');
  const [validationPercent, setValidationPercent] = useState(0.2);
  const [testPercent, setTestPercent] = useState(0.2);
  const [fitPreprocessingBeforeSplit, setFitPreprocessingBeforeSplit] = useState(false);
  const [usedTestForTuning, setUsedTestForTuning] = useState(false);

  const splits = useMemo(
    () => assignByMode(mode, validationPercent, testPercent),
    [mode, validationPercent, testPercent],
  );

  const trainRate = positiveRate(splits.train);
  const validationRate = positiveRate(splits.validation);
  const testRate = positiveRate(splits.test);
  const validationDrift = driftGap(splits.train, splits.validation);
  const testDrift = driftGap(splits.train, splits.test);
  const classGap = Math.max(Math.abs(trainRate - validationRate), Math.abs(trainRate - testRate));
  const hasWarning = fitPreprocessingBeforeSplit || usedTestForTuning || classGap > 0.35;

  const reset = () => {
    setMode('stratified');
    setValidationPercent(0.2);
    setTestPercent(0.2);
    setFitPreprocessingBeforeSplit(false);
    setUsedTestForTuning(false);
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Evaluation foundations</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Train / Validation / Test Split</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Training data fits parameters, validation data guides development choices, and test data estimates final
              generalization. The boundary matters as much as the ratios.
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
          Split controls
        </div>
        <div className="grid gap-4 xl:grid-cols-[1.4fr_0.8fr_0.8fr_1fr_1fr]">
          <div className="grid gap-2">
            <span className="text-sm font-bold text-slate-700">Split mode</span>
            <div className="grid gap-2 sm:grid-cols-3">
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
                </button>
              ))}
            </div>
          </div>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Validation: {Math.round(validationPercent * 100)}%
            <input
              min="0.1"
              max="0.3"
              step="0.05"
              type="range"
              value={validationPercent}
              onChange={(event) => setValidationPercent(Number(event.target.value))}
            />
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Test: {Math.round(testPercent * 100)}%
            <input
              min="0.1"
              max="0.3"
              step="0.05"
              type="range"
              value={testPercent}
              onChange={(event) => setTestPercent(Number(event.target.value))}
            />
          </label>
          <label className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-bold text-slate-700">
            Fit preprocessing before split
            <input
              type="checkbox"
              checked={fitPreprocessingBeforeSplit}
              onChange={(event) => setFitPreprocessingBeforeSplit(event.target.checked)}
            />
          </label>
          <label className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-bold text-slate-700">
            Test score used for tuning
            <input
              type="checkbox"
              checked={usedTestForTuning}
              onChange={(event) => setUsedTestForTuning(event.target.checked)}
            />
          </label>
        </div>
        <p className="mt-4 rounded-lg bg-slate-50 p-3 text-sm leading-6 text-slate-700">
          <strong className="text-slate-950">{MODES[mode].label}:</strong> {MODES[mode].detail}
        </p>
      </section>

      <div className="grid gap-3 md:grid-cols-4">
        <Stat label="Train rows" value={splits.train.length} detail="fit model parameters" />
        <Stat label="Validation rows" value={splits.validation.length} detail="choose thresholds and hyperparameters" />
        <Stat label="Test rows" value={splits.test.length} detail="final untouched estimate" />
        <Stat label="Largest class gap" value={`${Math.round(classGap * 100)} pts`} detail="split label imbalance" />
      </div>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <Shuffle size={16} />
          Dataset partitions
        </h3>
        <div className="mt-4 grid gap-4 xl:grid-cols-3">
          <SplitColumn label="Train" rows={splits.train} tone="train" />
          <SplitColumn label="Validation" rows={splits.validation} tone="validation" />
          <SplitColumn label="Test" rows={splits.test} tone="test" />
        </div>
      </section>

      <div className="grid gap-4 xl:grid-cols-[1fr_0.9fr]">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <BarChart3 size={16} />
            Split diagnostics
          </h3>
          <div className="mt-4 grid gap-4 md:grid-cols-3">
            {[
              ['Train', trainRate, splits.train],
              ['Validation', validationRate, splits.validation],
              ['Test', testRate, splits.test],
            ].map(([label, rate, rows]) => (
              <div key={label} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                <p className="text-xs font-black uppercase tracking-wide text-slate-500">{label} positive rate</p>
                <div className="mt-3 h-3 overflow-hidden rounded bg-white">
                  <div className="h-full bg-cyan-600" style={{ width: `${rate * 100}%` }} />
                </div>
                <strong className="mt-2 block text-xl font-black text-slate-950">{Math.round(rate * 100)}%</strong>
                <span className="text-sm text-slate-600">{rows.length} rows</span>
              </div>
            ))}
          </div>
          <div className="mt-4 grid gap-3 md:grid-cols-2">
            <Stat label="Validation drift" value={validationDrift.toFixed(1)} detail="feature mean gap vs train" />
            <Stat label="Test drift" value={testDrift.toFixed(1)} detail="feature mean gap vs train" />
          </div>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            {hasWarning ? <AlertTriangle size={16} /> : <ShieldCheck size={16} />}
            Boundary audit
          </h3>
          <div className="mt-4 space-y-4">
            <div className={`rounded-lg border p-4 ${fitPreprocessingBeforeSplit ? 'border-rose-200 bg-rose-50' : 'border-emerald-200 bg-emerald-50'}`}>
              <h4 className={`text-sm font-black uppercase tracking-wide ${fitPreprocessingBeforeSplit ? 'text-rose-700' : 'text-emerald-700'}`}>
                Preprocessing scope
              </h4>
              <p className={`mt-2 text-sm leading-6 ${fitPreprocessingBeforeSplit ? 'text-rose-950' : 'text-emerald-950'}`}>
                {fitPreprocessingBeforeSplit
                  ? 'Global preprocessing uses validation/test statistics before evaluation. Split first, then fit transformations on train only.'
                  : 'Preprocessing is learned from train rows and applied unchanged to validation/test rows.'}
              </p>
            </div>
            <div className={`rounded-lg border p-4 ${usedTestForTuning ? 'border-rose-200 bg-rose-50' : 'border-emerald-200 bg-emerald-50'}`}>
              <h4 className={`text-sm font-black uppercase tracking-wide ${usedTestForTuning ? 'text-rose-700' : 'text-emerald-700'}`}>
                Test set role
              </h4>
              <p className={`mt-2 text-sm leading-6 ${usedTestForTuning ? 'text-rose-950' : 'text-emerald-950'}`}>
                {usedTestForTuning
                  ? 'The test set has become development feedback. The final score is no longer an unbiased final estimate.'
                  : 'The test set remains untouched until the final report.'}
              </p>
            </div>
            <div className={`rounded-lg border p-4 ${classGap > 0.35 ? 'border-amber-200 bg-amber-50' : 'border-slate-200 bg-slate-50'}`}>
              <h4 className={`text-sm font-black uppercase tracking-wide ${classGap > 0.35 ? 'text-amber-700' : 'text-slate-600'}`}>
                Label balance
              </h4>
              <p className={`mt-2 text-sm leading-6 ${classGap > 0.35 ? 'text-amber-950' : 'text-slate-700'}`}>
                {classGap > 0.35
                  ? 'One split has a very different positive rate. Use stratification or collect more rows before trusting the comparison.'
                  : 'The class distribution is close enough for this toy dataset.'}
              </p>
            </div>
          </div>
        </section>
      </div>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Predict before running</h3>
          <p className="mt-3 text-sm leading-6 text-slate-700">
            Switch from random to stratified mode. The largest class gap should usually shrink because labels are
            distributed more evenly.
          </p>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Failure mode</h3>
          <p className="mt-3 text-sm leading-6 text-slate-700">
            Tuning on the test set invalidates it. You can still report the number, but it now measures the process that
            adapted to that set.
          </p>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Practical rule</h3>
          <p className="mt-3 text-sm leading-6 text-slate-700">
            Use validation or cross-validation for iteration, then touch test once after the training recipe is fixed.
          </p>
        </div>
      </section>

      <AssessmentPanel lessonId="train-validation-test-split" />
    </div>
  );
}
