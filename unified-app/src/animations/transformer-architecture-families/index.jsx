import React, { useMemo, useState } from 'react';
import { GitBranch, RotateCcw, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const FAMILIES = {
  encoder: {
    label: 'Encoder-only',
    example: 'BERT',
    objective: 'Masked language modeling or representation learning',
    attention: 'Bidirectional self-attention over visible input tokens',
    output: 'Contextual embeddings for classification, search, extraction, and reranking',
    prompt: ['[CLS]', 'the', 'movie', '[MASK]', 'great', '[SEP]'],
    target: ['was'],
    visible: 'all input tokens except masked content',
    color: 'cyan',
  },
  decoder: {
    label: 'Decoder-only',
    example: 'GPT',
    objective: 'Next-token prediction',
    attention: 'Causal self-attention over previous tokens only',
    output: 'One token at a time, appended back into the context',
    prompt: ['The', 'model', 'writes'],
    target: ['the', 'next', 'token'],
    visible: 'past and current tokens only',
    color: 'emerald',
  },
  encoderDecoder: {
    label: 'Encoder-decoder',
    example: 'T5',
    objective: 'Conditional generation from an encoded source sequence',
    attention: 'Encoder bidirectional attention plus decoder causal and cross-attention',
    output: 'Target sequence conditioned on a separate input sequence',
    prompt: ['translate:', 'good', 'morning'],
    target: ['buenos', 'dias'],
    visible: 'source tokens through cross-attention, target prefix through causal attention',
    color: 'violet',
  },
};

const COLORS = {
  cyan: { active: 'border-cyan-500 bg-cyan-600 text-white', soft: 'border-cyan-200 bg-cyan-50 text-cyan-950', line: '#0891b2' },
  emerald: { active: 'border-emerald-500 bg-emerald-600 text-white', soft: 'border-emerald-200 bg-emerald-50 text-emerald-950', line: '#059669' },
  violet: { active: 'border-violet-500 bg-violet-600 text-white', soft: 'border-violet-200 bg-violet-50 text-violet-950', line: '#7c3aed' },
};

function attentionMatrix(family) {
  if (family === 'encoder') {
    return [
      [1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1],
    ];
  }
  return [
    [1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
  ];
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

function TokenRow({ label, tokens, activeColor }) {
  return (
    <div>
      <p className="text-xs font-black uppercase tracking-wide text-slate-500">{label}</p>
      <div className="mt-2 flex flex-wrap gap-2">
        {tokens.map((token, index) => (
          <span key={`${token}-${index}`} className={`rounded-lg border px-3 py-2 text-sm font-black ${activeColor}`}>
            {token}
          </span>
        ))}
      </div>
    </div>
  );
}

export default function TransformerArchitectureFamiliesAnimation() {
  const [family, setFamily] = useState('decoder');
  const config = FAMILIES[family];
  const color = COLORS[config.color];
  const matrix = useMemo(() => attentionMatrix(family === 'encoderDecoder' ? 'decoder' : family), [family]);

  const reset = () => {
    setFamily('decoder');
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Transformer families</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Encoder-Only vs Decoder-Only vs Encoder-Decoder</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              BERT, GPT, and T5 use transformer blocks differently. The key differences are which tokens can attend to
              which other tokens, what objective trains the model, and what kind of output the model is built to produce.
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
          Architecture controls
        </div>
        <div className="grid gap-2 sm:grid-cols-3">
          {Object.entries(FAMILIES).map(([id, option]) => (
            <button
              key={id}
              type="button"
              onClick={() => setFamily(id)}
              className={`rounded-lg border px-3 py-3 text-sm font-black transition ${family === id ? COLORS[option.color].active : 'border-slate-200 bg-slate-50 text-slate-700'}`}
            >
              {option.label}
            </button>
          ))}
        </div>
      </section>

      <div className="grid gap-3 md:grid-cols-4">
        <Stat label="Family" value={config.label} detail={config.example} />
        <Stat label="Objective" value={family === 'decoder' ? 'Next token' : family === 'encoder' ? 'Masked token' : 'Seq2seq'} detail="training signal" />
        <Stat label="Visibility" value={family === 'encoder' ? 'Full' : family === 'decoder' ? 'Causal' : 'Mixed'} detail="attention mask pattern" />
        <Stat label="Output" value={family === 'encoder' ? 'Vectors' : 'Tokens'} detail="natural use case" />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.05fr_0.95fr]">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <GitBranch size={16} />
            Information flow
          </h3>
          <div className="mt-5 grid gap-4">
            <TokenRow label="Input / source" tokens={config.prompt} activeColor={color.soft} />
            {family === 'encoderDecoder' && (
              <div className="rounded-lg border border-violet-200 bg-violet-50 p-4 text-sm leading-6 text-violet-950">
                The encoder first builds source representations. The decoder then uses causal target attention plus
                cross-attention into those source representations.
              </div>
            )}
            <TokenRow label={family === 'encoder' ? 'Predicted masked content' : 'Generated target'} tokens={config.target} activeColor="border-slate-200 bg-slate-50 text-slate-950" />
          </div>

          <div className="mt-6 rounded-lg border border-slate-200 bg-slate-50 p-4">
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Attention mask intuition</p>
            <div className="mt-3 grid w-fit grid-cols-5 gap-1">
              {matrix.flatMap((row, rowIndex) => row.map((enabled, colIndex) => (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className="h-8 w-8 rounded border border-white"
                  style={{ background: enabled ? color.line : '#e2e8f0' }}
                  title={`query ${rowIndex + 1}, key ${colIndex + 1}`}
                />
              )))}
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-700">{config.visible}</p>
          </div>
        </section>

        <section className={`rounded-lg border p-5 ${color.soft}`}>
          <h3 className="text-sm font-black uppercase tracking-wide">Why this family exists</h3>
          <div className="mt-4 space-y-4 text-sm leading-6">
            <p><strong>Example:</strong> {config.example}</p>
            <p><strong>Training objective:</strong> {config.objective}</p>
            <p><strong>Attention rule:</strong> {config.attention}</p>
            <p><strong>Best fit:</strong> {config.output}</p>
          </div>
        </section>
      </div>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-cyan-700">Problem solved</h3>
          <p className="mt-3 text-sm leading-6 text-cyan-950">
            Architecture families explain why a BERT-style model, GPT-style model, and T5-style model are not swapped
            into the same workflow.
          </p>
        </div>
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-amber-700">Mistake to avoid</h3>
          <p className="mt-3 text-sm leading-6 text-amber-950">
            Bidirectional attention is useful for understanding a fixed input, but it cannot directly generate left to
            right without changing the objective and mask.
          </p>
        </div>
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-emerald-700">Understanding check</h3>
          <p className="mt-3 text-sm leading-6 text-emerald-950">
            Pick a task, then choose the family whose attention pattern and output type match the task.
          </p>
        </div>
      </section>

      <AssessmentPanel lessonId="transformer-architecture-families" title="Transformer Architecture Families check" />
    </div>
  );
}
