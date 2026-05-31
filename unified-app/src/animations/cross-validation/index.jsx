import React, { useMemo, useState } from 'react';
import { AlertTriangle, GitBranch, RotateCcw, ShieldCheck, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';
import { assignFolds, summarize } from './crossValidationModel.js';

const STRATEGIES = {
  random: {
    label: 'Random rows',
    detail: 'Rows are distributed independently. Fast, but entity duplicates can cross fold boundaries.',
  },
  grouped: {
    label: 'Grouped users',
    detail: 'All rows from the same user stay in the same fold, so entity memory cannot leak across validation.',
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

function FoldStrip({ rows, selectedFold, k }) {
  return (
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
      {Array.from({ length: k }, (_, fold) => {
        const foldRows = rows.filter((row) => row.fold === fold);
        const active = fold === selectedFold;
        return (
          <div
            key={fold}
            className={`rounded-lg border p-3 ${active ? 'border-cyan-500 bg-cyan-50' : 'border-slate-200 bg-slate-50'}`}
          >
            <div className="flex items-center justify-between gap-2">
              <strong className="text-sm font-black text-slate-900">Fold {fold + 1}</strong>
              <span className={`rounded-full px-2 py-1 text-xs font-black ${active ? 'bg-cyan-600 text-white' : 'bg-white text-slate-500'}`}>
                {active ? 'validation' : 'train'}
              </span>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {foldRows.map((row) => (
                <span key={row.id} className="rounded border border-white bg-white px-2 py-1 font-mono text-xs text-slate-700">
                  {row.id}/{row.user}
                </span>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function CrossValidationAnimation() {
  const [k, setK] = useState(5);
  const [strategy, setStrategy] = useState('random');
  const [preprocessingInsideFold, setPreprocessingInsideFold] = useState(true);
  const [selectedFold, setSelectedFold] = useState(0);

  const rows = useMemo(() => assignFolds(k, strategy), [k, strategy]);
  const summary = useMemo(
    () => summarize(rows, preprocessingInsideFold, k),
    [rows, preprocessingInsideFold, k],
  );
  const selected = summary.folds[selectedFold] || summary.folds[0];
  const hasLeakage = selected.leakageUsers.length > 0 || !preprocessingInsideFold;
  const scoreSpread = summary.max - summary.min;

  const reset = () => {
    setK(5);
    setStrategy('random');
    setPreprocessingInsideFold(true);
    setSelectedFold(0);
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Evaluation design</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Cross-Validation and Data Leakage</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Cross-validation rotates which fold acts as validation, then averages the scores. The estimate is useful
              only when every fold keeps validation information out of training and preprocessing.
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
          Fold controls
        </div>
        <div className="grid gap-4 xl:grid-cols-[0.8fr_1.5fr_1fr_1fr]">
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Folds: {k}
            <input
              min="3"
              max="6"
              step="1"
              type="range"
              value={k}
              onChange={(event) => {
                const nextK = Number(event.target.value);
                setK(nextK);
                setSelectedFold((fold) => Math.min(fold, nextK - 1));
              }}
            />
          </label>
          <div className="grid gap-2">
            <span className="text-sm font-bold text-slate-700">Split strategy</span>
            <div className="grid gap-2 sm:grid-cols-2">
              {Object.entries(STRATEGIES).map(([id, option]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setStrategy(id)}
                  className={`rounded-lg border px-3 py-2 text-left text-sm font-black transition ${
                    strategy === id ? 'border-cyan-500 bg-cyan-600 text-white' : 'border-slate-200 bg-slate-50 text-slate-700'
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Inspect fold
            <select
              value={selectedFold}
              onChange={(event) => setSelectedFold(Number(event.target.value))}
              className="rounded-lg border border-slate-300 bg-white px-3 py-2"
            >
              {Array.from({ length: k }, (_, fold) => (
                <option key={fold} value={fold}>Fold {fold + 1}</option>
              ))}
            </select>
          </label>
          <label className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-bold text-slate-700">
            Preprocess inside fold
            <input
              type="checkbox"
              checked={preprocessingInsideFold}
              onChange={(event) => setPreprocessingInsideFold(event.target.checked)}
            />
          </label>
        </div>
        <p className="mt-4 rounded-lg bg-slate-50 p-3 text-sm leading-6 text-slate-700">
          <strong className="text-slate-950">{STRATEGIES[strategy].label}:</strong> {STRATEGIES[strategy].detail}
        </p>
      </section>

      <div className="grid gap-3 md:grid-cols-4">
        <Stat label="Mean CV score" value={`${Math.round(summary.mean * 100)}%`} detail="average across folds" />
        <Stat label="Fold std" value={`${(summary.std * 100).toFixed(1)} pts`} detail="instability across folds" />
        <Stat label="Score range" value={`${Math.round(scoreSpread * 100)} pts`} detail="best fold minus worst fold" />
        <Stat label="Leak-risk folds" value={`${summary.leakedFoldCount}/${k}`} detail="duplicate users cross boundary" />
      </div>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <GitBranch size={16} />
          Rotate validation responsibility
        </h3>
        <div className="mt-4">
          <FoldStrip rows={rows} selectedFold={selectedFold} k={k} />
        </div>
      </section>

      <div className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <AlertTriangle size={16} />
            Selected fold audit
          </h3>
          <div className="mt-4 overflow-hidden rounded-lg border border-slate-200">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-100 text-xs font-black uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="px-3 py-2">Row</th>
                  <th className="px-3 py-2">User</th>
                  <th className="px-3 py-2">Role</th>
                  <th className="px-3 py-2">Audit</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const isValidation = row.fold === selectedFold;
                  const duplicateLeak = isValidation && selected.leakageUsers.includes(row.user);
                  const leak = duplicateLeak || (isValidation && !preprocessingInsideFold);
                  return (
                    <tr key={row.id} className={leak ? 'bg-rose-50 text-rose-950' : 'bg-white text-slate-700'}>
                      <td className="px-3 py-2 font-black">{row.id}</td>
                      <td className="px-3 py-2">{row.user}</td>
                      <td className="px-3 py-2">{isValidation ? 'validation' : 'training'}</td>
                      <td className="px-3 py-2 font-bold">
                        {duplicateLeak
                          ? 'same user in train'
                          : isValidation && !preprocessingInsideFold
                            ? 'global preprocessing'
                            : 'contained'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <ShieldCheck size={16} />
            Interpretation
          </h3>
          <div className="mt-4 space-y-4">
            <div className={`rounded-lg border p-4 ${hasLeakage ? 'border-rose-200 bg-rose-50' : 'border-emerald-200 bg-emerald-50'}`}>
              <h4 className={`text-sm font-black uppercase tracking-wide ${hasLeakage ? 'text-rose-700' : 'text-emerald-700'}`}>
                {hasLeakage ? 'Leakage warning' : 'Fold boundary clean'}
              </h4>
              <p className={`mt-2 text-sm leading-6 ${hasLeakage ? 'text-rose-950' : 'text-emerald-950'}`}>
                {hasLeakage
                  ? 'The selected fold has validation information that can shape training. The reported score may be optimistic.'
                  : 'Validation rows are held out from training and learned preprocessing. This fold gives a cleaner estimate.'}
              </p>
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <h4 className="text-sm font-black uppercase tracking-wide text-slate-600">Score for selected fold</h4>
              <p className="mt-2 text-3xl font-black text-slate-950">{Math.round(selected.score * 100)}%</p>
              <p className="mt-1 text-sm leading-6 text-slate-700">
                {selected.validationSize} validation rows. Use the mean and standard deviation across folds, not the
                best fold, when comparing candidate models.
              </p>
            </div>
            <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-4">
              <h4 className="text-sm font-black uppercase tracking-wide text-cyan-700">Rule</h4>
              <p className="mt-2 text-sm leading-6 text-cyan-950">
                For each fold: split first, fit model and transformations on training folds only, score on the held-out
                fold, then average all held-out scores.
              </p>
            </div>
          </div>
        </section>
      </div>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Predict before running</h3>
          <p className="mt-3 text-sm leading-6 text-slate-700">
            Switch from random rows to grouped users. The duplicate-user leak count should drop because validation users
            no longer appear in training folds.
          </p>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Failure mode</h3>
          <p className="mt-3 text-sm leading-6 text-slate-700">
            Fitting preprocessing globally before cross-validation lets validation statistics influence training folds,
            even when the model itself never sees validation labels.
          </p>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Deployment check</h3>
          <p className="mt-3 text-sm leading-6 text-slate-700">
            Cross-validation estimates model selection risk. Keep a final test set untouched for the last report.
          </p>
        </div>
      </section>

      <AssessmentPanel lessonId="cross-validation" />
    </div>
  );
}
