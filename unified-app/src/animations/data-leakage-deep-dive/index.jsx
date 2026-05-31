import React, { useMemo, useState } from 'react';
import { AlertTriangle, RotateCcw, ShieldCheck, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';
import { LEAKAGE_MODES, LEAKAGE_ROWS, rowIsLeaked, scoreGap } from './dataLeakageDeepDiveModel.js';

function Stat({ label, value, detail }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <p className="text-xs font-black uppercase tracking-wide text-slate-500">{label}</p>
      <strong className="mt-1 block text-2xl font-black text-slate-950">{value}</strong>
      <span className="text-sm text-slate-600">{detail}</span>
    </div>
  );
}

export default function DataLeakageDeepDiveAnimation() {
  const [mode, setMode] = useState('target');
  const [strictPipeline, setStrictPipeline] = useState(false);
  const scores = useMemo(() => scoreGap(mode, strictPipeline), [mode, strictPipeline]);
  const config = LEAKAGE_MODES[mode];

  const reset = () => {
    setMode('target');
    setStrictPipeline(false);
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Evaluation integrity</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Data Leakage Deep Dive</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Leakage happens when validation or test information shapes training. The score can look excellent while
              the deployed model fails because the shortcut disappears outside the dataset.
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
          Leakage controls
        </div>
        <div className="grid gap-4 xl:grid-cols-[1.7fr_0.8fr]">
          <div className="grid gap-2">
            <span className="text-sm font-bold text-slate-700">Leakage mode</span>
            <div className="grid gap-2 sm:grid-cols-3 lg:grid-cols-5">
              {Object.entries(LEAKAGE_MODES).map(([id, option]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setMode(id)}
                  className={`rounded-lg border px-3 py-2 text-sm font-black transition ${mode === id ? 'border-rose-500 bg-rose-600 text-white' : 'border-slate-200 bg-slate-50 text-slate-700'}`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
          <label className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-bold text-slate-700">
            Strict pipeline
            <input type="checkbox" checked={strictPipeline} onChange={(event) => setStrictPipeline(event.target.checked)} />
          </label>
        </div>
      </section>

      <div className="grid gap-3 md:grid-cols-4">
        <Stat label="Reported score" value={`${Math.round(scores.suspicious * 100)}%`} detail={strictPipeline ? 'leakage controlled' : 'suspiciously high'} />
        <Stat label="Honest score" value={`${Math.round(scores.honest * 100)}%`} detail="after leakage audit" />
        <Stat label="Optimism" value={`${Math.round(scores.optimism * 100)} pts`} detail="score inflation" />
        <Stat label="Leak source" value={config.leakedItem} detail="information crossing boundary" />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <AlertTriangle size={16} />
            Boundary audit table
          </h3>
          <div className="mt-4 overflow-hidden rounded-lg border border-slate-200">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-100 text-xs font-black uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="px-3 py-2">Row</th>
                  <th className="px-3 py-2">User</th>
                  <th className="px-3 py-2">Time</th>
                  <th className="px-3 py-2">Split</th>
                  <th className="px-3 py-2">Audit</th>
                </tr>
              </thead>
              <tbody>
                {LEAKAGE_ROWS.map((row) => {
                  const leaked = !strictPipeline && rowIsLeaked(row, mode);
                  return (
                    <tr key={row.id} className={leaked ? 'bg-rose-50 text-rose-950' : 'bg-white text-slate-700'}>
                      <td className="px-3 py-2 font-black">{row.id}</td>
                      <td className="px-3 py-2">{row.user}</td>
                      <td className="px-3 py-2">{row.time}</td>
                      <td className="px-3 py-2">{row.split}</td>
                      <td className="px-3 py-2 font-bold">{leaked ? 'leak risk' : 'contained'}</td>
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
            Diagnosis
          </h3>
          <div className="mt-4 space-y-4">
            <div className="rounded-lg border border-rose-200 bg-rose-50 p-4">
              <h4 className="text-sm font-black uppercase tracking-wide text-rose-700">Leakage path</h4>
              <p className="mt-2 text-sm leading-6 text-rose-950">{config.leak}</p>
            </div>
            <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4">
              <h4 className="text-sm font-black uppercase tracking-wide text-emerald-700">Safer fix</h4>
              <p className="mt-2 text-sm leading-6 text-emerald-950">{config.fix}</p>
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <h4 className="text-sm font-black uppercase tracking-wide text-slate-600">Pipeline rule</h4>
              <p className="mt-2 text-sm leading-6 text-slate-700">
                Split first according to the evaluation question, then fit every learned transformation using training
                data only.
              </p>
            </div>
          </div>
        </section>
      </div>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-cyan-700">Problem solved</h3>
          <p className="mt-3 text-sm leading-6 text-cyan-950">
            Leakage audits explain why a validation score can be invalid even when the model code runs correctly.
          </p>
        </div>
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-amber-700">Mistake to avoid</h3>
          <p className="mt-3 text-sm leading-6 text-amber-950">
            Random splitting is not automatically safe when users repeat, time matters, or features know the future.
          </p>
        </div>
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-emerald-700">Understanding check</h3>
          <p className="mt-3 text-sm leading-6 text-emerald-950">
            Pick a leakage mode, name the boundary being crossed, then choose the split or pipeline fix.
          </p>
        </div>
      </section>

      <AssessmentPanel lessonId="data-leakage-deep-dive" title="Data Leakage Deep Dive check" />
    </div>
  );
}
