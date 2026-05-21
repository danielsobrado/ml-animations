import React, { useMemo, useState } from 'react';
import { EyeOff, RotateCcw, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const SELF_TOKENS = ['The', 'model', 'predicts', '[PAD]', '[PAD]'];
const DECODER_TOKENS = ['The', 'cat', 'sat'];
const ENCODER_TOKENS = ['Le', 'chat', 'assis', '[PAD]'];

const MODES = {
  bidirectional: {
    label: 'Bidirectional',
    detail: 'Encoder-style self-attention can read every real token in the sequence.',
  },
  causal: {
    label: 'Causal',
    detail: 'Decoder self-attention hides future tokens so next-token prediction cannot cheat.',
  },
  padding: {
    label: 'Padding',
    detail: 'Padding masks remove artificial [PAD] tokens from attention scores.',
  },
  cross: {
    label: 'Cross-attention',
    detail: 'Decoder queries read encoder keys and values, usually with source padding masked out.',
  },
};

function rawScore(row, col) {
  return 1.25 - Math.abs(row - col) * 0.35 + Math.sin((row + 1) * (col + 2)) * 0.18;
}

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

function buildMask({ mode, selectedQuery, maskPadding }) {
  const queryTokens = mode === 'cross' ? DECODER_TOKENS : SELF_TOKENS;
  const keyTokens = mode === 'cross' ? ENCODER_TOKENS : SELF_TOKENS;
  const cells = [];

  for (let row = 0; row < queryTokens.length; row += 1) {
    for (let col = 0; col < keyTokens.length; col += 1) {
      const keyIsPad = keyTokens[col] === '[PAD]';
      const queryIsPad = queryTokens[row] === '[PAD]';
      let allowed = true;
      let reason = 'Visible context';

      if (mode === 'causal' && col > row) {
        allowed = false;
        reason = 'Future token hidden';
      }
      if ((mode === 'padding' || mode === 'cross' || maskPadding) && keyIsPad) {
        allowed = false;
        reason = 'Padding token removed';
      }
      if (queryIsPad) {
        allowed = false;
        reason = 'Padding query ignored';
      }

      cells.push({
        row,
        col,
        allowed,
        reason,
        score: rawScore(row, col),
      });
    }
  }

  const selectedCells = cells.filter((cell) => cell.row === selectedQuery);
  const allowedCells = selectedCells.filter((cell) => cell.allowed);
  const probabilities = allowedCells.length
    ? softmax(allowedCells.map((cell) => cell.score))
    : [];
  const probabilityByCol = new Map(allowedCells.map((cell, index) => [cell.col, probabilities[index]]));

  const enrichedCells = cells.map((cell) => ({
    ...cell,
    probability: probabilityByCol.get(cell.col) || 0,
    selected: cell.row === selectedQuery,
  }));

  return {
    queryTokens,
    keyTokens,
    cells: enrichedCells,
    selectedCells: enrichedCells.filter((cell) => cell.row === selectedQuery),
    allowedCells: enrichedCells.filter((cell) => cell.row === selectedQuery && cell.allowed),
  };
}

function cellTone(cell) {
  if (!cell.selected) return cell.allowed ? 'border-slate-200 bg-slate-50 text-slate-500' : 'border-slate-100 bg-slate-100 text-slate-400';
  if (cell.allowed) return 'border-emerald-300 bg-emerald-50 text-emerald-950';
  return 'border-rose-200 bg-rose-50 text-rose-950';
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

export default function AttentionMasksAnimation() {
  const [mode, setMode] = useState('causal');
  const [selectedQuery, setSelectedQuery] = useState(2);
  const [maskPadding, setMaskPadding] = useState(true);

  const matrix = useMemo(
    () => buildMask({ mode, selectedQuery, maskPadding }),
    [mode, selectedQuery, maskPadding],
  );
  const allowedForQuery = matrix.allowedCells.length;
  const blockedForQuery = matrix.selectedCells.length - allowedForQuery;
  const selectedToken = matrix.queryTokens[selectedQuery];

  const reset = () => {
    setMode('causal');
    setSelectedQuery(2);
    setMaskPadding(true);
  };

  const handleMode = (nextMode) => {
    setMode(nextMode);
    setSelectedQuery((current) => Math.min(current, (nextMode === 'cross' ? DECODER_TOKENS : SELF_TOKENS).length - 1));
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Transformer visibility rules</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Attention Masks</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Attention masks decide which query-key scores survive before softmax. Switch mask types and inspect how
              future tokens, padding tokens, and encoder memory become visible or hidden.
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
          Mask controls
        </div>
        <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr_1fr]">
          <div className="grid gap-2">
            <span className="text-sm font-bold text-slate-700">Mask type</span>
            <div className="grid gap-2 sm:grid-cols-4">
              {Object.entries(MODES).map(([id, config]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => handleMode(id)}
                  className={`rounded-lg border px-3 py-2 text-sm font-black transition ${mode === id ? 'border-purple-500 bg-purple-600 text-white' : 'border-slate-200 bg-slate-50 text-slate-700'}`}
                >
                  {config.label}
                </button>
              ))}
            </div>
          </div>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Query row: {selectedToken}
            <input
              min="0"
              max={matrix.queryTokens.length - 1}
              step="1"
              type="range"
              value={selectedQuery}
              onChange={(event) => setSelectedQuery(Number(event.target.value))}
            />
          </label>
          <label className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-bold text-slate-700">
            Also mask padding
            <input type="checkbox" checked={maskPadding} onChange={(event) => setMaskPadding(event.target.checked)} />
          </label>
        </div>
        <p className="mt-4 rounded-lg bg-slate-50 p-3 text-sm leading-6 text-slate-700">
          <strong className="text-slate-950">{MODES[mode].label}:</strong> {MODES[mode].detail}
        </p>
      </section>

      <div className="grid gap-3 md:grid-cols-4">
        <Stat label="Query token" value={selectedToken} detail={`row ${selectedQuery + 1}`} />
        <Stat label="Visible keys" value={allowedForQuery} detail="scores kept before softmax" />
        <Stat label="Blocked keys" value={blockedForQuery} detail="scores replaced by -inf" />
        <Stat label="Mask math" value="+ mask" detail="added before softmax" />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <EyeOff size={16} />
            Query-key visibility matrix
          </h3>
          <div className="mt-4 overflow-x-auto">
            <div
              className="grid min-w-[560px] gap-2"
              style={{ gridTemplateColumns: `92px repeat(${matrix.keyTokens.length}, minmax(78px, 1fr))` }}
            >
              <div />
              {matrix.keyTokens.map((token, index) => (
                <div key={`key-${token}-${index}`} className="rounded-lg bg-slate-100 px-2 py-2 text-center text-xs font-black text-slate-600">
                  key {index + 1}<br />{token}
                </div>
              ))}
              {matrix.queryTokens.map((token, row) => (
                <React.Fragment key={`row-${token}-${row}`}>
                  <div className={`rounded-lg px-2 py-3 text-xs font-black ${row === selectedQuery ? 'bg-purple-600 text-white' : 'bg-slate-100 text-slate-600'}`}>
                    query {row + 1}<br />{token}
                  </div>
                  {matrix.cells.filter((cell) => cell.row === row).map((cell) => (
                    <div key={`${cell.row}-${cell.col}`} className={`rounded-lg border p-3 text-center ${cellTone(cell)}`}>
                      <strong className="block text-sm">{cell.allowed ? 'keep' : 'mask'}</strong>
                      <span className="mt-1 block text-xs">{cell.allowed ? cell.probability.toFixed(2) : '-inf'}</span>
                    </div>
                  ))}
                </React.Fragment>
              ))}
            </div>
          </div>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Selected row explanation</h3>
          <p className="mt-3 font-mono text-sm text-slate-950">softmax((QK^T / sqrt(d)) + mask)</p>
          <div className="mt-4 space-y-3">
            {matrix.selectedCells.map((cell) => (
              <article key={cell.col} className={`rounded-lg border p-3 ${cell.allowed ? 'border-emerald-200 bg-emerald-50' : 'border-rose-200 bg-rose-50'}`}>
                <div className="flex items-center justify-between gap-3">
                  <strong className="text-sm text-slate-950">{matrix.keyTokens[cell.col]}</strong>
                  <span className="rounded bg-white px-2 py-1 text-xs font-black text-slate-700">
                    {cell.allowed ? `weight ${cell.probability.toFixed(2)}` : 'masked'}
                  </span>
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-700">{cell.reason}</p>
              </article>
            ))}
          </div>
        </section>
      </div>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-purple-200 bg-purple-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-purple-700">Problem solved</h3>
          <p className="mt-3 text-sm leading-6 text-purple-950">
            Masks enforce the task contract: encoders can read context, decoders cannot read future answers, and padding
            should not behave like real evidence.
          </p>
        </div>
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-amber-700">Mistake to avoid</h3>
          <p className="mt-3 text-sm leading-6 text-amber-950">
            Masked language modeling masks input tokens for prediction; attention masks control which positions can be
            read. They are related but not the same mechanism.
          </p>
        </div>
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-emerald-700">Understanding check</h3>
          <p className="mt-3 text-sm leading-6 text-emerald-950">
            Before switching modes, predict whether the selected query should see future tokens, padding tokens, or
            encoder memory.
          </p>
        </div>
      </section>

      <AssessmentPanel lessonId="attention-masks" title="Attention Masks check" />
    </div>
  );
}
