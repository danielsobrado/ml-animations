import React, { useMemo, useState } from 'react';
import { ArrowRight, Database, RotateCcw, SlidersHorizontal, StepForward } from 'lucide-react';
import { computeSoftmax } from '../../data/softmaxModel';

const BASE_CONTEXT = ['The', 'model', 'writes'];
const VOCABULARY = [
  { token: ' clearly', logit: 3.2 },
  { token: ' code', logit: 2.4 },
  { token: ' about', logit: 1.8 },
  { token: ' because', logit: 1.35 },
  { token: ' ...', logit: 0.55 },
];

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function topPFilter(rows, topP) {
  let cumulative = 0;
  return rows.map((row, index) => {
    const keep = index === 0 || cumulative < topP;
    cumulative += row.probability;
    return { ...row, keptByTopP: keep, cumulative };
  });
}

function chooseToken(rows, strategy, step) {
  if (strategy === 'greedy') return rows[0];
  const keptRows = rows.filter((row) => row.kept);
  const weightedIndex = Math.floor(((step + 1) * 1.618 * 1000) % keptRows.length);
  return keptRows[weightedIndex] || rows[0];
}

function buildDistribution({ generated, temperature, topK, topP, strategy }) {
  const step = generated.length;
  const logits = VOCABULARY.map((item, index) => (
    item.logit + Math.sin((step + 1) * (index + 1)) * 0.35 - generated.filter((token) => token === item.token).length * 0.6
  ));
  const probabilities = computeSoftmax(logits, temperature);
  const rankedRows = VOCABULARY
    .map((item, index) => ({
      ...item,
      logit: logits[index],
      probability: probabilities[index],
    }))
    .sort((a, b) => b.probability - a.probability);
  const nucleusRows = topPFilter(rankedRows, topP);
  const rows = nucleusRows.map((row, index) => ({
    ...row,
    kept: index < topK && row.keptByTopP,
  }));
  const selected = chooseToken(rows, strategy, step);

  return { rows, selected };
}

function Stat({ label, value, detail }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <p className="text-xs font-bold uppercase tracking-wide text-slate-500">{label}</p>
      <strong className="mt-1 block text-2xl font-black text-slate-900">{value}</strong>
      <span className="text-sm text-slate-600">{detail}</span>
    </div>
  );
}

function PipelineStep({ title, children, active }) {
  return (
    <section className={`rounded-lg border p-4 ${active ? 'border-blue-500 bg-blue-50' : 'border-slate-200 bg-white'}`}>
      <h3 className="text-sm font-black uppercase tracking-wide text-slate-700">{title}</h3>
      <div className="mt-3 text-sm leading-6 text-slate-700">{children}</div>
    </section>
  );
}

export default function TransformerTokenGeneration() {
  const [generated, setGenerated] = useState([]);
  const [temperature, setTemperature] = useState(0.9);
  const [topK, setTopK] = useState(3);
  const [topP, setTopP] = useState(0.9);
  const [strategy, setStrategy] = useState('sample');

  const allTokens = [...BASE_CONTEXT, ...generated];
  const distribution = useMemo(
    () => buildDistribution({ generated, temperature, topK, topP, strategy }),
    [generated, temperature, topK, topP, strategy],
  );
  const cacheSavings = Math.max(0, allTokens.length * generated.length - generated.length);

  const generateNext = () => {
    if (generated.length >= 6) return;
    setGenerated((tokens) => [...tokens, distribution.selected.token]);
  };

  const reset = () => setGenerated([]);

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-bold uppercase tracking-wide text-slate-500">Autoregressive decoding</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Transformer Token Generation Loop</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Each step runs the current context through the decoder, converts logits into probabilities,
              filters candidate tokens, samples one token, appends it, then reuses cached keys and values on the next step.
            </p>
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={generateNext}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-900 bg-slate-900 px-4 py-2 text-sm font-bold text-white"
            >
              <StepForward size={16} />
              Next token
            </button>
            <button
              type="button"
              onClick={reset}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-bold text-slate-800"
            >
              <RotateCcw size={16} />
              Reset
            </button>
          </div>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        <Stat label="Generated" value={generated.length} detail="tokens appended" />
        <Stat label="KV cache" value={`${allTokens.length} rows`} detail={`${cacheSavings} recompute units avoided`} />
        <Stat label="Sampling" value={strategy === 'greedy' ? 'Greedy' : 'Sample'} detail={`top-${topK}, top-p ${topP.toFixed(2)}`} />
      </div>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <SlidersHorizontal size={16} />
          Controls
        </div>
        <div className="grid gap-4 lg:grid-cols-4">
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Temperature: {temperature.toFixed(2)}
            <input
              type="range"
              min="0.3"
              max="1.8"
              step="0.05"
              value={temperature}
              onChange={(event) => setTemperature(Number(event.target.value))}
            />
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Top-k: {topK}
            <input
              type="range"
              min="1"
              max="5"
              step="1"
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
            />
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Top-p: {topP.toFixed(2)}
            <input
              type="range"
              min="0.5"
              max="1"
              step="0.05"
              value={topP}
              onChange={(event) => setTopP(Number(event.target.value))}
            />
          </label>
          <div className="grid grid-cols-2 gap-2 self-end">
            {['sample', 'greedy'].map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setStrategy(mode)}
                className={`rounded-lg border px-3 py-2 text-sm font-bold capitalize ${
                  strategy === mode ? 'border-blue-600 bg-blue-600 text-white' : 'border-slate-300 bg-white text-slate-700'
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>
      </section>

      <div className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Current context</h3>
          <div className="mt-4 flex flex-wrap gap-2">
            {allTokens.map((token, index) => (
              <span
                key={`${token}-${index}`}
                className={`rounded-lg border px-3 py-2 font-mono text-sm ${
                  index < BASE_CONTEXT.length
                    ? 'border-slate-300 bg-slate-50 text-slate-800'
                    : 'border-blue-300 bg-blue-50 text-blue-900'
                }`}
              >
                {token}
              </span>
            ))}
          </div>

          <div className="mt-5 grid gap-3 md:grid-cols-5">
            <PipelineStep title="1. Context" active>
              Feed all visible tokens into the decoder.
            </PipelineStep>
            <PipelineStep title="2. Logits" active>
              Project the last hidden state to vocabulary scores.
            </PipelineStep>
            <PipelineStep title="3. Softmax" active>
              Apply temperature, then normalize into probabilities.
            </PipelineStep>
            <PipelineStep title="4. Filter">
              Keep candidates allowed by top-k and top-p.
            </PipelineStep>
            <PipelineStep title="5. Append">
              Add <strong>{distribution.selected.token}</strong> and repeat.
            </PipelineStep>
          </div>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Next-token distribution</h3>
          <div className="mt-4 space-y-3">
            {distribution.rows.map((row) => {
              const selected = row.token === distribution.selected.token;
              return (
                <div key={row.token} className="grid grid-cols-[84px_1fr_64px] items-center gap-3">
                  <span className={`font-mono text-sm ${selected ? 'font-black text-blue-700' : 'text-slate-700'}`}>
                    {row.token}
                  </span>
                  <div className="h-8 rounded bg-slate-100">
                    <div
                      className={`h-8 rounded ${row.kept ? 'bg-blue-500' : 'bg-slate-300'} ${selected ? 'ring-2 ring-blue-900' : ''}`}
                      style={{ width: `${clamp(row.probability * 100, 4, 100)}%` }}
                    />
                  </div>
                  <span className="text-right font-mono text-sm text-slate-700">
                    {(row.probability * 100).toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>
          <p className="mt-4 text-sm leading-6 text-slate-600">
            Gray candidates are filtered out before selection. Greedy always chooses the largest kept probability;
            sampling can choose another kept token.
          </p>
        </section>
      </div>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <Database size={16} />
          KV cache growth
        </div>
        <div className="mt-4 grid gap-2">
          {allTokens.map((token, index) => (
            <div key={`${token}-cache-${index}`} className="grid grid-cols-[96px_1fr_auto] items-center gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
              <span className="font-mono text-sm text-slate-800">{token}</span>
              <div className="flex items-center gap-2 text-sm text-slate-600">
                <span className="rounded border border-slate-300 bg-white px-2 py-1">K</span>
                <ArrowRight size={14} />
                <span className="rounded border border-slate-300 bg-white px-2 py-1">V</span>
              </div>
              <strong className={index === allTokens.length - 1 && generated.length ? 'text-blue-700' : 'text-slate-500'}>
                {index === allTokens.length - 1 && generated.length ? 'new' : 'reused'}
              </strong>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
