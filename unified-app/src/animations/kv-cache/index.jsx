import React, { useMemo, useState } from 'react';
import { AlertTriangle, BarChart3, Calculator, Database, RotateCcw, ShieldCheck, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const TOKENS = ['The', 'model', 'predicts', 'next', 'token', 'carefully', 'today', '.'];

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function formatCount(value) {
  return value >= 1000 ? `${(value / 1000).toFixed(1)}k` : String(Math.round(value));
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

function CacheRow({ token, index, active, evicted }) {
  return (
    <div
      className={`grid grid-cols-[44px_1fr_1fr] items-center gap-2 rounded-lg border p-2 text-sm ${
        evicted
          ? 'border-slate-200 bg-slate-50 text-slate-400'
          : active
            ? 'border-cyan-400 bg-cyan-50 text-slate-900'
            : 'border-slate-200 bg-white text-slate-700'
      }`}
    >
      <span className="font-mono text-xs font-black">t{index}</span>
      <span className="truncate font-black">{token}</span>
      <span className="font-mono text-xs">{evicted ? 'evicted' : `K${index}, V${index}`}</span>
    </div>
  );
}

export default function KVCacheAnimation() {
  const [contextLength, setContextLength] = useState(6);
  const [decodeStep, setDecodeStep] = useState(4);
  const [heads, setHeads] = useState(16);
  const [headDim, setHeadDim] = useState(64);
  const [windowSize, setWindowSize] = useState(8);
  const [useCache, setUseCache] = useState(true);

  const metrics = useMemo(() => {
    const activeTokens = TOKENS.slice(0, contextLength);
    const visibleStart = Math.max(0, decodeStep - windowSize + 1);
    const visibleTokens = activeTokens.map((token, index) => ({
      token,
      index,
      active: index === decodeStep,
      evicted: index < visibleStart,
    }));
    const visibleCount = visibleTokens.filter((token) => !token.evicted && token.index <= decodeStep).length;
    const prefixLength = decodeStep + 1;
    const noCacheProjections = prefixLength * heads * 2;
    const cachedProjections = heads * 2;
    const attentionReads = visibleCount * heads;
    const memoryValues = visibleCount * heads * headDim * 2;
    const saving = noCacheProjections === 0 ? 0 : 1 - cachedProjections / noCacheProjections;
    return {
      activeTokens,
      visibleTokens,
      visibleStart,
      visibleCount,
      prefixLength,
      noCacheProjections,
      cachedProjections,
      attentionReads,
      memoryValues,
      saving,
      currentToken: activeTokens[decodeStep] || activeTokens[activeTokens.length - 1],
    };
  }, [contextLength, decodeStep, headDim, heads, windowSize]);

  const reset = () => {
    setContextLength(6);
    setDecodeStep(4);
    setHeads(16);
    setHeadDim(64);
    setWindowSize(8);
    setUseCache(true);
  };

  const setLength = (value) => {
    const nextLength = Number(value);
    setContextLength(nextLength);
    setDecodeStep((current) => Math.min(current, nextLength - 1));
    setWindowSize((current) => Math.min(Math.max(current, 2), nextLength));
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">LLM inference efficiency</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">KV Cache</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              During autoregressive generation, previous tokens keep the same key and value vectors. A KV cache stores
              those vectors so each new step projects only the new token while still attending over prior context.
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
          Decode controls
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Context length: {contextLength}
            <input min="2" max={TOKENS.length} type="range" value={contextLength} onChange={(event) => setLength(event.target.value)} />
            <span className="text-xs font-semibold text-slate-500">Tokens already in the generation prefix.</span>
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Decode step: {decodeStep + 1}
            <input min="0" max={contextLength - 1} type="range" value={decodeStep} onChange={(event) => setDecodeStep(Number(event.target.value))} />
            <span className="text-xs font-semibold text-slate-500">Current token whose query is being evaluated.</span>
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Attention heads: {heads}
            <input min="4" max="32" step="4" type="range" value={heads} onChange={(event) => setHeads(Number(event.target.value))} />
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Head dimension: {headDim}
            <input min="32" max="128" step="16" type="range" value={headDim} onChange={(event) => setHeadDim(Number(event.target.value))} />
          </label>
          <label className="flex items-start gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm font-bold text-slate-700">
            <input type="checkbox" checked={useCache} onChange={(event) => setUseCache(event.target.checked)} className="mt-1" />
            <span>
              Use KV cache
              <small className="mt-1 block font-semibold leading-5 text-slate-500">Project only the new token at each step.</small>
            </span>
          </label>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-4">
        <Stat label="Current token" value={metrics.currentToken} detail={`Step ${metrics.decodeStep ?? decodeStep + 1} reads prior keys and values.`} />
        <Stat
          label="K/V projections"
          value={formatCount(useCache ? metrics.cachedProjections : metrics.noCacheProjections)}
          detail={useCache ? 'Only new-token K,V are computed.' : 'Entire prefix K,V are recomputed.'}
        />
        <Stat label="Attention reads" value={formatCount(metrics.attentionReads)} detail={`${metrics.visibleCount} visible positions times ${heads} heads.`} />
        <Stat label="Projection savings" value={`${Math.round(metrics.saving * 100)}%`} detail="Compared with recomputing the prefix." />
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <Database size={16} />
            Cache table
          </div>
          <div className="grid gap-2">
            {metrics.visibleTokens.map((item) => (
              <CacheRow key={`${item.token}-${item.index}`} {...item} />
            ))}
          </div>
          <div className="mt-4">
            <label className="grid gap-2 text-sm font-bold text-slate-700">
              Sliding window size: {windowSize}
              <input
                min="2"
                max={contextLength}
                type="range"
                value={windowSize}
                onChange={(event) => setWindowSize(Number(event.target.value))}
              />
              <span className="text-xs font-semibold text-slate-500">
                Smaller windows reduce cache reads but drop older context.
              </span>
            </label>
          </div>
        </div>

        <div className="space-y-4">
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <div className="mb-3 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
              <Calculator size={16} />
              What gets reused
            </div>
            <div className="rounded-lg bg-slate-950 p-4 font-mono text-sm leading-7 text-cyan-100">
              cached per token = K and V for each head<br />
              new step computes Q for current token<br />
              attention reads cached K,V from visible tokens
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-700">
              The cache does not skip attention. It avoids recomputing old key and value projections, then lets the new
              query compare against the cached keys.
            </p>
          </div>
          <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-4">
            <p className="text-xs font-black uppercase tracking-wide text-cyan-700">Predict before running</p>
            <p className="mt-2 text-sm leading-6 text-cyan-950">
              Increase decode step and predict whether cached projection work stays flat or grows with the full prefix.
            </p>
          </div>
          <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
            <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-amber-700">
              <AlertTriangle size={14} />
              Failure mode
            </p>
            <p className="mt-2 text-sm leading-6 text-amber-950">
              KV cache speeds incremental decoding, but memory grows with visible tokens, layers, heads, and head dimension.
            </p>
          </div>
          <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4">
            <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-emerald-700">
              <ShieldCheck size={14} />
              Practical rule
            </p>
            <p className="mt-2 text-sm leading-6 text-emerald-950">
              Use cache for generation, manage memory with batching, grouped-query attention, paged caches, or sliding windows.
            </p>
          </div>
        </div>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <BarChart3 size={16} />
          Recompute versus cache
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <div className="mb-1 flex justify-between text-sm font-bold text-slate-700">
              <span>No cache: recompute prefix K,V</span>
              <span>{formatCount(metrics.noCacheProjections)}</span>
            </div>
            <div className="h-4 rounded-full bg-slate-100">
              <div className="h-4 rounded-full bg-amber-500" style={{ width: '100%' }} />
            </div>
          </div>
          <div>
            <div className="mb-1 flex justify-between text-sm font-bold text-slate-700">
              <span>With cache: project current K,V</span>
              <span>{formatCount(metrics.cachedProjections)}</span>
            </div>
            <div className="h-4 rounded-full bg-slate-100">
              <div
                className="h-4 rounded-full bg-cyan-600"
                style={{ width: `${clamp((metrics.cachedProjections / metrics.noCacheProjections) * 100, 4, 100)}%` }}
              />
            </div>
          </div>
        </div>
      </section>

      <AssessmentPanel lessonId="kv-cache" />
    </div>
  );
}
