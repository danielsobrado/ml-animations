import React, { useMemo, useState } from 'react';
import { GitBranch, RotateCcw, SlidersHorizontal, Thermometer } from 'lucide-react';
import { computeSoftmax } from '../../data/softmaxModel';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const BASE_LOGITS = [
  { token: ' clear', logit: 3.1 },
  { token: ' careful', logit: 2.45 },
  { token: ' creative', logit: 2.05 },
  { token: ' risky', logit: 1.25 },
  { token: ' strange', logit: 0.55 },
  { token: ' broken', logit: -0.15 },
];

const STRATEGIES = {
  greedy: {
    label: 'Greedy',
    detail: 'Always pick the highest-probability token.',
    risk: 'Can become repetitive or bland because alternatives never get a chance.',
  },
  beam: {
    label: 'Beam search',
    detail: 'Keep several high-scoring partial sequences, then expand the best beams.',
    risk: 'Can over-prefer safe generic continuations when diversity matters.',
  },
  temperature: {
    label: 'Temperature',
    detail: 'Rescale logits before softmax to sharpen or flatten the distribution.',
    risk: 'High temperature can make weak tokens too likely; very low temperature acts almost greedy.',
  },
  topK: {
    label: 'Top-k',
    detail: 'Keep only the k most likely candidates, then sample inside that set.',
    risk: 'A fixed k can keep too many bad tokens or remove useful tail options depending on the prompt.',
  },
  topP: {
    label: 'Top-p',
    detail: 'Keep ranked candidates until cumulative probability reaches p, including the token that crosses the threshold.',
    risk: 'The candidate count changes by context, so p controls mass rather than a fixed number of tokens.',
  },
};

function topPFilter(rows, topP) {
  let cumulative = 0;
  return rows.map((row, index) => {
    const keep = index === 0 || cumulative < topP;
    cumulative += row.probability;
    return { ...row, topPKept: keep, cumulative };
  });
}

function buildRows({ temperature, topK, topP }) {
  const probs = computeSoftmax(BASE_LOGITS.map((item) => item.logit), temperature);
  const ranked = BASE_LOGITS
    .map((item, index) => ({ ...item, probability: probs[index] }))
    .sort((a, b) => b.probability - a.probability);

  return topPFilter(ranked, topP).map((row, index) => ({
    ...row,
    topKKept: index < topK,
    stochasticKept: index < topK && row.topPKept,
  }));
}

function beamRows(width) {
  return [
    { path: 'The answer is clear', score: -0.38 },
    { path: 'The answer is careful', score: -0.51 },
    { path: 'The answer is creative', score: -0.83 },
    { path: 'The answer is risky', score: -1.42 },
  ].map((row, index) => ({ ...row, kept: index < width }));
}

function StrategyButton({ id, active, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-lg border px-3 py-3 text-sm font-black transition ${
        active ? 'border-cyan-600 bg-cyan-600 text-white' : 'border-slate-200 bg-white text-slate-700 hover:border-cyan-300'
      }`}
    >
      {STRATEGIES[id].label}
    </button>
  );
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

export default function SamplingStrategiesAnimation() {
  const [strategy, setStrategy] = useState('topP');
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(4);
  const [topP, setTopP] = useState(0.86);
  const [beamWidth, setBeamWidth] = useState(2);

  const rows = useMemo(() => buildRows({ temperature, topK, topP }), [temperature, topK, topP]);
  const beams = useMemo(() => beamRows(beamWidth), [beamWidth]);
  const selected = rows.find((row) => row.stochasticKept) || rows[0];
  const candidateCount = strategy === 'beam' ? beamWidth : rows.filter((row) => row.stochasticKept).length;
  const activeConfig = STRATEGIES[strategy];

  const reset = () => {
    setStrategy('topP');
    setTemperature(0.8);
    setTopK(4);
    setTopP(0.86);
    setBeamWidth(2);
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Inference-time decoding</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Sampling Strategies</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Decoding turns next-token probabilities into an actual continuation. Greedy and beam search favor high-score
              paths, while temperature, top-k, and top-p control how much uncertainty survives before sampling.
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
          Strategy controls
        </div>
        <div className="grid gap-4 xl:grid-cols-[1.2fr_1fr]">
          <div className="grid gap-2 sm:grid-cols-5">
            {Object.keys(STRATEGIES).map((id) => (
              <StrategyButton key={id} id={id} active={strategy === id} onClick={() => setStrategy(id)} />
            ))}
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="grid gap-2 text-sm font-bold text-slate-700">
              Temperature: {temperature.toFixed(2)}
              <input min="0.25" max="1.8" step="0.05" type="range" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} />
            </label>
            <label className="grid gap-2 text-sm font-bold text-slate-700">
              Top-k: {topK}
              <input min="1" max="6" step="1" type="range" value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
            </label>
            <label className="grid gap-2 text-sm font-bold text-slate-700">
              Top-p: {topP.toFixed(2)}
              <input min="0.45" max="1" step="0.05" type="range" value={topP} onChange={(event) => setTopP(Number(event.target.value))} />
            </label>
            <label className="grid gap-2 text-sm font-bold text-slate-700">
              Beam width: {beamWidth}
              <input min="1" max="4" step="1" type="range" value={beamWidth} onChange={(event) => setBeamWidth(Number(event.target.value))} />
            </label>
          </div>
        </div>
      </section>

      <div className="grid gap-3 md:grid-cols-4">
        <Stat label="Strategy" value={activeConfig.label} detail="current decoding rule" />
        <Stat label="Candidates" value={candidateCount} detail="kept for next decision" />
        <Stat label="Most likely" value={rows[0].token.trim()} detail={`${Math.round(rows[0].probability * 100)}% probability`} />
        <Stat label="Selected path" value={strategy === 'beam' ? `${beamWidth} beams` : selected.token.trim()} detail="what remains eligible" />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_0.9fr]">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <Thermometer size={16} />
            Token distribution
          </h3>
          <div className="mt-4 space-y-3">
            {rows.map((row, index) => {
              const kept = strategy === 'greedy' ? index === 0 : strategy === 'topK' ? row.topKKept : strategy === 'topP' ? row.topPKept : row.stochasticKept;
              return (
                <div key={row.token} className="grid gap-2 sm:grid-cols-[90px_1fr_78px] sm:items-center">
                  <span className={`rounded-lg border px-3 py-2 font-mono text-sm ${kept ? 'border-cyan-300 bg-cyan-50 text-cyan-950' : 'border-slate-200 bg-slate-50 text-slate-500'}`}>
                    {row.token}
                  </span>
                  <div className="h-3 overflow-hidden rounded-full bg-slate-100">
                    <div className={kept ? 'h-full bg-cyan-500' : 'h-full bg-slate-300'} style={{ width: `${Math.max(5, row.probability * 100)}%` }} />
                  </div>
                  <span className="text-sm font-black text-slate-700">{Math.round(row.probability * 100)}%</span>
                </div>
              );
            })}
          </div>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <GitBranch size={16} />
            Beam comparison
          </h3>
          <div className="mt-4 space-y-3">
            {beams.map((beam) => (
              <div key={beam.path} className={`rounded-lg border p-3 ${beam.kept ? 'border-emerald-300 bg-emerald-50' : 'border-slate-200 bg-slate-50'}`}>
                <p className="font-mono text-sm text-slate-900">{beam.path}</p>
                <p className="mt-1 text-xs font-black uppercase tracking-wide text-slate-500">log score {beam.score.toFixed(2)}</p>
              </div>
            ))}
          </div>
        </section>
      </div>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-cyan-700">Problem solved</h3>
          <p className="mt-3 text-sm leading-6 text-cyan-950">
            Sampling strategies decide how deterministic, diverse, or conservative an autoregressive model should be at inference time.
          </p>
        </div>
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-amber-700">Core math</h3>
          <p className="mt-3 text-sm leading-6 text-amber-950">
            Temperature rescales logits before softmax; top-k filters by rank; top-p filters by cumulative probability mass; beam search maximizes sequence score.
          </p>
        </div>
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-emerald-700">Understanding check</h3>
          <p className="mt-3 text-sm leading-6 text-emerald-950">
            Predict whether each control increases determinism, diversity, or candidate coverage before moving it.
          </p>
        </div>
      </section>

      <section className="rounded-lg border border-amber-200 bg-amber-50 p-5 text-amber-950">
        <h3 className="text-sm font-black uppercase tracking-wide text-amber-700">Mistake to avoid</h3>
        <p className="mt-3 text-sm leading-6">{activeConfig.risk}</p>
        <p className="mt-3 text-sm leading-6">{activeConfig.detail}</p>
      </section>

      <AssessmentPanel lessonId="sampling-strategies" title="Sampling Strategies check" />
    </div>
  );
}
