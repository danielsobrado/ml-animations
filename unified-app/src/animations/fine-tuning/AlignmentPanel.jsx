import React, { useMemo, useState } from 'react';
import { CheckCircle, GitCompare, SlidersHorizontal } from 'lucide-react';

const METHODS = {
  sft: {
    label: 'Instruction SFT',
    signal: 'prompt -> reference answer',
    loss: '-log p(reference answer | prompt)',
    behavior: 'Teaches the model the response format and task-following style in demonstrations.',
    risk: 'The model imitates demonstrations, but it does not directly learn which of two plausible answers people prefer.',
  },
  dpo: {
    label: 'DPO preference',
    signal: 'chosen answer > rejected answer',
    loss: 'increase chosen log-odds over rejected log-odds',
    behavior: 'Pushes the model toward preferred answers without training a separate reward model.',
    risk: 'Preference data can overfit style preferences and does not magically add missing domain facts.',
  },
  rlhf: {
    label: 'RLHF',
    signal: 'reward model score',
    loss: 'maximize reward while staying near the reference policy',
    behavior: 'Optimizes a policy against a learned reward model with a constraint against drifting too far.',
    risk: 'A flawed reward model can be exploited, producing high reward without better real-world answers.',
  },
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function qualityScore(method, dataQuality, preferenceMargin) {
  const base = { sft: 0.52, dpo: 0.47, rlhf: 0.43 }[method];
  const marginBoost = method === 'sft' ? 0.08 : preferenceMargin * 0.18;
  return clamp(base + dataQuality * 0.34 + marginBoost, 0, 0.96);
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

export default function AlignmentPanel() {
  const [method, setMethod] = useState('sft');
  const [dataQuality, setDataQuality] = useState(0.7);
  const [preferenceMargin, setPreferenceMargin] = useState(0.55);
  const config = METHODS[method];
  const score = useMemo(() => qualityScore(method, dataQuality, preferenceMargin), [method, dataQuality, preferenceMargin]);

  return (
    <div className="space-y-6 p-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <p className="text-xs font-black uppercase tracking-wide text-slate-500">Behavior tuning</p>
        <h2 className="mt-1 text-2xl font-black text-slate-950">Instruction And Preference Fine-Tuning</h2>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
          Fine-tuning methods use different feedback signals. SFT imitates curated demonstrations, while preference
          optimization compares candidate answers and pushes the model toward chosen behavior.
        </p>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <SlidersHorizontal size={16} />
          Method controls
        </div>
        <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
          <div className="grid gap-2 sm:grid-cols-3">
            {Object.entries(METHODS).map(([id, item]) => (
              <button
                key={id}
                type="button"
                onClick={() => setMethod(id)}
                className={`rounded-lg border px-3 py-3 text-sm font-black transition ${
                  method === id ? 'border-purple-600 bg-purple-600 text-white' : 'border-slate-200 bg-white text-slate-700'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="grid gap-2 text-sm font-bold text-slate-700">
              Data quality: {dataQuality.toFixed(2)}
              <input min="0" max="1" step="0.05" type="range" value={dataQuality} onChange={(event) => setDataQuality(Number(event.target.value))} />
            </label>
            <label className="grid gap-2 text-sm font-bold text-slate-700">
              Preference margin: {preferenceMargin.toFixed(2)}
              <input min="0" max="1" step="0.05" type="range" value={preferenceMargin} onChange={(event) => setPreferenceMargin(Number(event.target.value))} />
            </label>
          </div>
        </div>
      </section>

      <div className="grid gap-3 md:grid-cols-4">
        <Stat label="Method" value={config.label} detail="selected tuning stage" />
        <Stat label="Signal" value={config.signal} detail="what the data provides" />
        <Stat label="Quality score" value={`${Math.round(score * 100)}%`} detail="signal quality estimate" />
        <Stat label="Weights" value="updated" detail="adapters or full policy can train" />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_0.9fr]">
        <section className="rounded-lg border border-purple-200 bg-purple-50 p-5 text-purple-950">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-purple-700">
            <GitCompare size={16} />
            Objective
          </h3>
          <p className="mt-4 rounded-lg border border-purple-200 bg-white p-4 font-mono text-sm">{config.loss}</p>
          <p className="mt-4 text-sm leading-6">{config.behavior}</p>
        </section>

        <section className="rounded-lg border border-amber-200 bg-amber-50 p-5 text-amber-950">
          <h3 className="text-sm font-black uppercase tracking-wide text-amber-700">Mistake to avoid</h3>
          <p className="mt-4 text-sm leading-6">{config.risk}</p>
        </section>
      </div>

      <section className="grid gap-4 lg:grid-cols-3">
        {[
          ['Problem solved', 'Tuning bridges a pretrained model and the behavior a product or task actually needs.'],
          ['Core math', 'The target may be a reference answer, a chosen-over-rejected comparison, or a reward-constrained policy update.'],
          ['Understanding check', 'Match the data signal to the method before choosing SFT, DPO, or RLHF.'],
        ].map(([title, body]) => (
          <div key={title} className="rounded-lg border border-emerald-200 bg-emerald-50 p-5 text-emerald-950">
            <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-emerald-700">
              <CheckCircle size={16} />
              {title}
            </h3>
            <p className="mt-3 text-sm leading-6">{body}</p>
          </div>
        ))}
      </section>
    </div>
  );
}
