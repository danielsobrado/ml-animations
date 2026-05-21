import React, { useMemo, useState } from 'react';
import { BookOpen, RotateCcw, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const OBJECTIVES = {
  nextToken: {
    label: 'Next-token prediction',
    stage: 'Pretraining',
    family: 'Decoder-only',
    loss: '-log p(next token | prefix)',
    example: ['The', 'model', 'predicts', '?'],
    target: 'tokens',
    detail: 'The model learns broad language patterns by predicting each next token from earlier tokens.',
    mistake: 'It learns continuation skill, not instruction following by default.',
  },
  masked: {
    label: 'Masked language modeling',
    stage: 'Pretraining',
    family: 'Encoder-only',
    loss: '-log p(masked token | visible context)',
    example: ['The', '[MASK]', 'is', 'blue'],
    target: 'sky',
    detail: 'The model reads both left and right context to reconstruct hidden input tokens.',
    mistake: 'It is not naturally an autoregressive chat generator.',
  },
  sft: {
    label: 'Supervised fine-tuning',
    stage: 'Instruction tuning',
    family: 'Mostly decoder-only',
    loss: '-log p(reference answer | prompt)',
    example: ['Prompt:', 'summarize', 'article'],
    target: 'reference answer',
    detail: 'The model imitates curated demonstrations so prompts map to useful response formats.',
    mistake: 'It copies demonstrations; it does not directly optimize human preference tradeoffs.',
  },
  preference: {
    label: 'Preference optimization',
    stage: 'Alignment',
    family: 'Mostly decoder-only',
    loss: 'prefer chosen response over rejected response',
    example: ['Prompt', 'chosen', 'rejected'],
    target: 'chosen > rejected',
    detail: 'The model is pushed toward responses people or reward models prefer over alternatives.',
    mistake: 'Preference data compares outputs; it is not the same as adding more factual knowledge.',
  },
};

const ORDER = ['nextToken', 'masked', 'sft', 'preference'];

function objectiveScore(objective, dataQuality) {
  const base = {
    nextToken: 0.72,
    masked: 0.68,
    sft: 0.62,
    preference: 0.58,
  }[objective];
  return Math.min(0.96, base + dataQuality * 0.28);
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

export default function LlmTrainingObjectivesAnimation() {
  const [objective, setObjective] = useState('nextToken');
  const [dataQuality, setDataQuality] = useState(0.55);
  const config = OBJECTIVES[objective];
  const score = useMemo(() => objectiveScore(objective, dataQuality), [objective, dataQuality]);

  const reset = () => {
    setObjective('nextToken');
    setDataQuality(0.55);
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Training signals</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">LLM Training Objectives</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              LLM behavior comes from a sequence of objectives. Pretraining teaches prediction, instruction tuning
              teaches prompt-response format, and preference optimization shifts outputs toward chosen responses.
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
          Objective controls
        </div>
        <div className="grid gap-4 lg:grid-cols-[1.7fr_0.8fr]">
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {ORDER.map((id) => (
              <button
                key={id}
                type="button"
                onClick={() => setObjective(id)}
                className={`rounded-lg border px-3 py-3 text-sm font-black transition ${objective === id ? 'border-cyan-500 bg-cyan-600 text-white' : 'border-slate-200 bg-slate-50 text-slate-700'}`}
              >
                {OBJECTIVES[id].label}
              </button>
            ))}
          </div>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Data quality: {dataQuality.toFixed(2)}
            <input min="0" max="1" step="0.05" type="range" value={dataQuality} onChange={(event) => setDataQuality(Number(event.target.value))} />
          </label>
        </div>
      </section>

      <div className="grid gap-3 md:grid-cols-4">
        <Stat label="Stage" value={config.stage} detail="where this objective appears" />
        <Stat label="Architecture fit" value={config.family} detail="natural model family" />
        <Stat label="Training signal" value={config.target} detail="what the loss rewards" />
        <Stat label="Teaching score" value={`${Math.round(score * 100)}%`} detail="quality-sensitive outcome" />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.05fr_0.95fr]">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <BookOpen size={16} />
            Objective example
          </h3>
          <div className="mt-4 flex flex-wrap gap-2">
            {config.example.map((token, index) => (
              <span key={`${token}-${index}`} className="rounded-lg border border-cyan-200 bg-cyan-50 px-3 py-2 text-sm font-black text-cyan-950">
                {token}
              </span>
            ))}
          </div>
          <div className="mt-5 rounded-lg border border-slate-200 bg-slate-50 p-4">
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Loss target</p>
            <p className="mt-2 text-lg font-black text-slate-950">{config.loss}</p>
          </div>
          <div className="mt-5 grid gap-2 sm:grid-cols-4">
            {ORDER.map((id, index) => (
              <div key={id} className={`rounded-lg border p-3 ${objective === id ? 'border-cyan-500 bg-cyan-50' : 'border-slate-200 bg-white'}`}>
                <p className="text-xs font-black uppercase tracking-wide text-slate-500">Step {index + 1}</p>
                <p className="mt-1 text-sm font-black text-slate-950">{OBJECTIVES[id].stage}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-lg border border-cyan-200 bg-cyan-50 p-5 text-cyan-950">
          <h3 className="text-sm font-black uppercase tracking-wide text-cyan-700">What this objective teaches</h3>
          <p className="mt-4 text-sm leading-6">{config.detail}</p>
          <div className="mt-5 rounded-lg border border-amber-200 bg-amber-50 p-4 text-amber-950">
            <h4 className="text-sm font-black uppercase tracking-wide text-amber-700">Mistake to avoid</h4>
            <p className="mt-2 text-sm leading-6">{config.mistake}</p>
          </div>
        </section>
      </div>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-cyan-700">Problem solved</h3>
          <p className="mt-3 text-sm leading-6 text-cyan-950">
            Objectives explain why pretraining, instruction tuning, and preference optimization produce different
            behavior even when the architecture is similar.
          </p>
        </div>
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-amber-700">Core math</h3>
          <p className="mt-3 text-sm leading-6 text-amber-950">
            Most stages still optimize log probabilities, but the target changes from raw tokens to demonstrations or
            chosen-over-rejected comparisons.
          </p>
        </div>
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-emerald-700">Understanding check</h3>
          <p className="mt-3 text-sm leading-6 text-emerald-950">
            Choose which objective teaches factual continuation, prompt following, and preference-shaped answers.
          </p>
        </div>
      </section>

      <AssessmentPanel lessonId="llm-training-objectives" title="LLM Training Objectives check" />
    </div>
  );
}
