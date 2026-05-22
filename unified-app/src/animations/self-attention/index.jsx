import React, { useMemo, useState } from 'react';
import { AlertTriangle, BarChart3, Calculator, RotateCcw, ShieldCheck, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const TOKENS = [
  {
    token: 'The',
    q: [0.2, 0.3, 0.1],
    k: [0.1, 0.2, 0.1],
    v: [0.2, 0.1],
  },
  {
    token: 'animal',
    q: [0.9, 0.1, 0.2],
    k: [0.95, 0.1, 0.2],
    v: [0.9, 0.2],
  },
  {
    token: 'crossed',
    q: [0.2, 0.85, 0.1],
    k: [0.1, 0.9, 0.2],
    v: [0.35, 0.75],
  },
  {
    token: 'street',
    q: [0.15, 0.8, 0.3],
    k: [0.1, 0.7, 0.45],
    v: [0.25, 0.9],
  },
  {
    token: 'tired',
    q: [0.8, 0.15, 0.4],
    k: [0.8, 0.15, 0.35],
    v: [0.85, 0.35],
  },
];

function dot(left, right) {
  return left.reduce((total, value, index) => total + value * right[index], 0);
}

function softmax(values, temperature) {
  const scaled = values.map((value) => value / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

function weightedValue(weights) {
  return TOKENS.reduce(
    (out, token, index) => [out[0] + weights[index] * token.v[0], out[1] + weights[index] * token.v[1]],
    [0, 0],
  );
}

function fmtVector(vector) {
  return `[${vector.map((value) => value.toFixed(2)).join(', ')}]`;
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

function MatrixCell({ value, active, blocked }) {
  const opacity = blocked ? 0.18 : Math.max(0.18, value);
  return (
    <div
      className={`flex aspect-square items-center justify-center rounded text-xs font-black ${
        active ? 'ring-2 ring-slate-950' : ''
      } ${blocked ? 'bg-slate-200 text-slate-400' : 'bg-cyan-600 text-white'}`}
      style={blocked ? undefined : { opacity }}
    >
      {blocked ? '-' : Math.round(value * 100)}
    </div>
  );
}

export default function SelfAttentionAnimation() {
  const [queryIndex, setQueryIndex] = useState(4);
  const [temperature, setTemperature] = useState(1);
  const [causalMask, setCausalMask] = useState(false);
  const [boostToken, setBoostToken] = useState(1);

  const attention = useMemo(() => {
    const query = TOKENS[queryIndex].q.map((value, index) => (index === 0 ? value + boostToken / 10 : value));
    const scale = Math.sqrt(query.length);
    const rawScores = TOKENS.map((token, index) => {
      const blocked = causalMask && index > queryIndex;
      return {
        index,
        token: token.token,
        blocked,
        score: blocked ? Number.NEGATIVE_INFINITY : dot(query, token.k) / scale,
      };
    });
    const visibleScores = rawScores.map((item) => (item.blocked ? -1000 : item.score));
    const weights = softmax(visibleScores, temperature);
    const output = weightedValue(weights);
    const winner = rawScores.reduce((best, item, index) => (weights[index] > weights[best.index] ? { ...item, index } : best), {
      ...rawScores[0],
      index: 0,
    });
    return { query, rawScores, weights, output, winner, scale };
  }, [boostToken, causalMask, queryIndex, temperature]);

  const rows = TOKENS.map((queryToken, rowIndex) => {
    const scores = TOKENS.map((keyToken, colIndex) => {
      if (causalMask && colIndex > rowIndex) return Number.NEGATIVE_INFINITY;
      return dot(queryToken.q, keyToken.k) / Math.sqrt(queryToken.q.length);
    });
    return softmax(scores.map((score) => (score === Number.NEGATIVE_INFINITY ? -1000 : score)), temperature);
  });

  const reset = () => {
    setQueryIndex(4);
    setTemperature(1);
    setCausalMask(false);
    setBoostToken(1);
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Transformer foundation</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Self-Attention</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Self-attention builds a new context vector for each token. A query compares with every key, softmax turns
              scores into weights, and the output is a weighted mix of value vectors.
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
          Attention controls
        </div>
        <div className="grid gap-4 xl:grid-cols-[1.4fr_1fr_1fr_1fr]">
          <div className="grid gap-2">
            <span className="text-sm font-bold text-slate-700">Query token</span>
            <div className="flex flex-wrap gap-2">
              {TOKENS.map((token, index) => (
                <button
                  key={token.token}
                  type="button"
                  onClick={() => setQueryIndex(index)}
                  className={`rounded-lg border px-3 py-2 text-sm font-black ${
                    queryIndex === index ? 'border-cyan-500 bg-cyan-600 text-white' : 'border-slate-200 bg-slate-50 text-slate-700'
                  }`}
                >
                  {token.token}
                </button>
              ))}
            </div>
          </div>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Softmax temperature: {temperature.toFixed(2)}
            <input min="0.35" max="2.5" step="0.05" type="range" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} />
            <span className="text-xs font-semibold text-slate-500">Lower values make attention sharper.</span>
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Animal-query boost: {boostToken.toFixed(1)}
            <input min="-2" max="3" step="0.1" type="range" value={boostToken} onChange={(event) => setBoostToken(Number(event.target.value))} />
            <span className="text-xs font-semibold text-slate-500">Perturbs the selected query vector.</span>
          </label>
          <label className="flex items-start gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm font-bold text-slate-700">
            <input type="checkbox" checked={causalMask} onChange={(event) => setCausalMask(event.target.checked)} className="mt-1" />
            <span>
              Causal mask
              <small className="mt-1 block font-semibold leading-5 text-slate-500">Block future keys before softmax.</small>
            </span>
          </label>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-4">
        <Stat label="Query" value={TOKENS[queryIndex].token} detail={`q = ${fmtVector(attention.query)}`} />
        <Stat label="Scale" value={`sqrt(${TOKENS[0].q.length})`} detail={`Scores divide by ${attention.scale.toFixed(2)}.`} />
        <Stat label="Strongest key" value={attention.winner.token} detail={`${Math.round(attention.weights[attention.winner.index] * 100)}% attention weight.`} />
        <Stat label="Output" value={fmtVector(attention.output)} detail="Weighted value mixture." />
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <BarChart3 size={16} />
            One attention row
          </div>
          <div className="grid gap-3">
            {attention.rawScores.map((item, index) => (
              <div key={item.token}>
                <div className="mb-1 flex items-center justify-between text-sm">
                  <strong className="text-slate-800">{TOKENS[queryIndex].token} reads {item.token}</strong>
                  <span className="font-bold text-slate-500">
                    {item.blocked ? 'masked' : `${Math.round(attention.weights[index] * 100)}%`}
                  </span>
                </div>
                <div className="h-3 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className={`h-full rounded-full ${item.blocked ? 'bg-slate-300' : 'bg-cyan-600'}`}
                    style={{ width: `${item.blocked ? 0 : attention.weights[index] * 100}%` }}
                  />
                </div>
                <p className="mt-1 text-xs font-semibold text-slate-500">
                  score = {item.blocked ? 'blocked before softmax' : item.score.toFixed(2)}
                </p>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <div className="mb-3 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
              <Calculator size={16} />
              Formula trace
            </div>
            <div className="rounded-lg bg-slate-950 p-4 font-mono text-sm leading-7 text-cyan-100">
              scores = Q K^T / sqrt(d_k)<br />
              weights = softmax(scores)<br />
              output = weights V
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-700">
              Scaling prevents high-dimensional dot products from pushing softmax into a nearly one-hot distribution too early.
            </p>
          </div>
          <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-4">
            <p className="text-xs font-black uppercase tracking-wide text-cyan-700">Predict before running</p>
            <p className="mt-2 text-sm leading-6 text-cyan-950">
              Lower the temperature before checking the bars, then predict which value vector will dominate the output.
            </p>
          </div>
          <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
            <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-amber-700">
              <AlertTriangle size={14} />
              Failure mode
            </p>
            <p className="mt-2 text-sm leading-6 text-amber-950">
              Attention weights are routing weights for value mixing. They are not guaranteed human explanations.
            </p>
          </div>
          <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4">
            <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-emerald-700">
              <ShieldCheck size={14} />
              Practical rule
            </p>
            <p className="mt-2 text-sm leading-6 text-emerald-950">
              Check the row, mask, and value vectors together. The largest attention weight only matters through the value it retrieves.
            </p>
          </div>
        </div>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <BarChart3 size={16} />
          Full attention matrix
        </div>
        <div className="grid grid-cols-[96px_repeat(5,minmax(42px,1fr))] gap-2">
          <div />
          {TOKENS.map((token) => (
            <div key={token.token} className="truncate text-center text-xs font-black uppercase text-slate-500">
              {token.token}
            </div>
          ))}
          {rows.map((row, rowIndex) => (
            <React.Fragment key={TOKENS[rowIndex].token}>
              <div className="flex items-center text-sm font-black text-slate-700">{TOKENS[rowIndex].token}</div>
              {row.map((value, colIndex) => (
                <MatrixCell
                  key={`${rowIndex}-${colIndex}`}
                  value={value}
                  active={rowIndex === queryIndex && colIndex === attention.winner.index}
                  blocked={causalMask && colIndex > rowIndex}
                />
              ))}
            </React.Fragment>
          ))}
        </div>
      </section>

      <AssessmentPanel lessonId="self-attention" />
    </div>
  );
}
