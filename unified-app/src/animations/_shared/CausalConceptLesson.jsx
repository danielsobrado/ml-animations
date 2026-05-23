import React, { useMemo, useState } from 'react';
import { AlertTriangle, BarChart3, CheckCircle2, GitBranch, RotateCcw, SlidersHorizontal, Target } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

function Stat({ label, value, detail, tone = 'slate' }) {
  const tones = {
    slate: 'border-slate-200 bg-white text-slate-950',
    cyan: 'border-cyan-200 bg-cyan-50 text-cyan-950',
    emerald: 'border-emerald-200 bg-emerald-50 text-emerald-950',
    amber: 'border-amber-200 bg-amber-50 text-amber-950',
    rose: 'border-rose-200 bg-rose-50 text-rose-950',
  };

  return (
    <div className={`rounded-lg border p-4 ${tones[tone]}`}>
      <p className="text-xs font-black uppercase tracking-wide opacity-70">{label}</p>
      <strong className="mt-1 block text-2xl font-black">{value}</strong>
      <span className="text-sm leading-5 opacity-80">{detail}</span>
    </div>
  );
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export default function CausalConceptLesson({ config }) {
  const [values, setValues] = useState(() => Object.fromEntries(config.controls.map((control) => [control.id, control.defaultValue])));

  const metrics = useMemo(() => {
    const raw = config.compute(values);
    return {
      ...raw,
      bars: raw.bars.map((bar) => ({ ...bar, width: clamp(bar.width, 4, 100) })),
    };
  }, [config, values]);

  const reset = () => setValues(Object.fromEntries(config.controls.map((control) => [control.id, control.defaultValue])));

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">{config.kicker}</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">{config.title}</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">{config.description}</p>
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
          Scenario controls
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {config.controls.map((control) => (
            <label key={control.id} className="grid gap-2 text-sm font-bold text-slate-700">
              {control.label}: {control.format(values[control.id])}
              <input
                type="range"
                min={control.min}
                max={control.max}
                step={control.step}
                value={values[control.id]}
                onChange={(event) => setValues((current) => ({ ...current, [control.id]: Number(event.target.value) }))}
              />
              <span className="text-xs font-semibold text-slate-500">{control.help}</span>
            </label>
          ))}
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-4">
        {metrics.stats.map((stat) => (
          <Stat key={stat.label} {...stat} />
        ))}
      </section>

      <section className="grid gap-6 xl:grid-cols-[1fr_1fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <BarChart3 size={16} />
            Effect readout
          </div>
          <div className="space-y-4">
            {metrics.bars.map((bar) => (
              <div key={bar.label}>
                <div className="mb-1 flex justify-between gap-3 text-sm font-bold text-slate-700">
                  <span>{bar.label}</span>
                  <span>{bar.value}</span>
                </div>
                <div className="h-5 rounded-full bg-slate-100">
                  <div className={`h-5 rounded-full ${bar.color}`} style={{ width: `${bar.width}%` }} />
                </div>
              </div>
            ))}
          </div>
          <div className="mt-5 rounded-lg bg-slate-950 p-4 font-mono text-sm leading-7 text-cyan-100">
            {metrics.formulaLines.map((line) => (
              <React.Fragment key={line}>
                {line}
                <br />
              </React.Fragment>
            ))}
          </div>
          <p className="mt-3 text-sm leading-6 text-slate-700">{metrics.readout}</p>
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <GitBranch size={16} />
            Decision logic
          </div>
          <div className="space-y-3">
            {metrics.steps.map((step, index) => (
              <div key={step.title} className={`rounded-lg border p-4 ${step.pass ? 'border-emerald-200 bg-emerald-50' : 'border-amber-200 bg-amber-50'}`}>
                <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-slate-700">
                  {step.pass ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
                  Step {index + 1}: {step.title}
                </p>
                <p className="mt-2 text-sm leading-6 text-slate-800">{step.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="rounded-lg border border-cyan-200 bg-cyan-50 p-5">
        <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-cyan-900">
          <Target size={14} />
          Practical takeaway
        </p>
        <p className="mt-2 text-sm leading-6 text-cyan-950">{metrics.takeaway}</p>
      </section>

      <AssessmentPanel lessonId={config.lessonId} />
    </div>
  );
}
