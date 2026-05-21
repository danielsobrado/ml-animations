import React, { useMemo, useState } from 'react';
import { Database, GitBranch, RotateCcw, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const METHODS = {
  exact: {
    label: 'Exact search',
    speed: 0.25,
    recall: 0.98,
    detail: 'Compare the query with every vector. Simple and accurate, but expensive at scale.',
  },
  ivf: {
    label: 'IVF clusters',
    speed: 0.68,
    recall: 0.72,
    detail: 'Search only nearby centroid buckets. Faster, but relevant vectors can hide in unprobed clusters.',
  },
  hnsw: {
    label: 'HNSW graph',
    speed: 0.82,
    recall: 0.8,
    detail: 'Walk a navigable neighbor graph from coarse entry points toward nearby candidates.',
  },
};

const POINTS = [
  { id: 'refund policy', x: 18, y: 30, relevant: true },
  { id: 'annual renewal', x: 26, y: 24, relevant: true },
  { id: 'billing admin', x: 37, y: 35 },
  { id: 'password reset', x: 72, y: 26 },
  { id: 'sso setup', x: 78, y: 44 },
  { id: 'invoice export', x: 42, y: 62 },
  { id: 'trial limits', x: 58, y: 74 },
  { id: 'seat count', x: 50, y: 48 },
];

function methodScore(method, searchBreadth, scale) {
  const config = METHODS[method];
  const breadthBoost = searchBreadth * 0.22;
  const recall = Math.min(0.99, config.recall + breadthBoost);
  const latency = Math.round((method === 'exact' ? scale * 0.9 : scale * (1 - config.speed) + searchBreadth * 32));
  return { recall, latency };
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

export default function RagVectorIndexingAnimation() {
  const [method, setMethod] = useState('hnsw');
  const [searchBreadth, setSearchBreadth] = useState(0.55);
  const [scale, setScale] = useState(70);
  const metrics = useMemo(() => methodScore(method, searchBreadth, scale), [method, searchBreadth, scale]);
  const active = METHODS[method];

  const reset = () => {
    setMethod('hnsw');
    setSearchBreadth(0.55);
    setScale(70);
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">RAG retrieval infrastructure</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Vector Indexing And ANN Search</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Vector indexes make embedding search fast by avoiding a full comparison against every chunk. Approximate
              nearest neighbor search trades some recall for lower latency and better scale.
            </p>
          </div>
          <button type="button" onClick={reset} className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-bold text-slate-800">
            <RotateCcw size={16} />
            Reset
          </button>
        </div>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <SlidersHorizontal size={16} />
          Index controls
        </div>
        <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
          <div className="grid gap-2 sm:grid-cols-3">
            {Object.entries(METHODS).map(([id, config]) => (
              <button
                key={id}
                type="button"
                onClick={() => setMethod(id)}
                className={`rounded-lg border px-3 py-3 text-sm font-black transition ${method === id ? 'border-indigo-600 bg-indigo-600 text-white' : 'border-slate-200 bg-white text-slate-700'}`}
              >
                {config.label}
              </button>
            ))}
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="grid gap-2 text-sm font-bold text-slate-700">
              Search breadth: {searchBreadth.toFixed(2)}
              <input min="0" max="1" step="0.05" type="range" value={searchBreadth} onChange={(event) => setSearchBreadth(Number(event.target.value))} />
            </label>
            <label className="grid gap-2 text-sm font-bold text-slate-700">
              Corpus scale: {scale}k chunks
              <input min="10" max="200" step="10" type="range" value={scale} onChange={(event) => setScale(Number(event.target.value))} />
            </label>
          </div>
        </div>
      </section>

      <div className="grid gap-3 md:grid-cols-4">
        <Stat label="Index" value={active.label} detail="selected search structure" />
        <Stat label="Recall" value={`${Math.round(metrics.recall * 100)}%`} detail="chance relevant neighbor is found" />
        <Stat label="Latency" value={`${metrics.latency} ms`} detail="rough search cost" />
        <Stat label="Tradeoff" value={method === 'exact' ? 'accuracy' : 'speed'} detail="dominant behavior" />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_0.9fr]">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <Database size={16} />
            Vector map
          </h3>
          <div className="relative mt-4 h-80 rounded-lg border border-slate-200 bg-slate-50">
            <div className="absolute left-[23%] top-[28%] h-24 w-28 rounded-full border-2 border-dashed border-indigo-300 bg-indigo-100/40" />
            <div className="absolute left-[60%] top-[22%] h-28 w-32 rounded-full border-2 border-dashed border-slate-300 bg-white/50" />
            {POINTS.map((point) => (
              <div
                key={point.id}
                className={`absolute -translate-x-1/2 -translate-y-1/2 rounded-lg border px-2 py-1 text-xs font-black ${point.relevant ? 'border-emerald-400 bg-emerald-100 text-emerald-950' : 'border-slate-300 bg-white text-slate-700'}`}
                style={{ left: `${point.x}%`, top: `${point.y}%` }}
              >
                {point.id}
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-lg border border-indigo-200 bg-indigo-50 p-5 text-indigo-950">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-indigo-700">
            <GitBranch size={16} />
            What changes
          </h3>
          <p className="mt-4 text-sm leading-6">{active.detail}</p>
          <div className="mt-5 rounded-lg border border-amber-200 bg-amber-50 p-4 text-amber-950">
            <h4 className="text-sm font-black uppercase tracking-wide text-amber-700">Mistake to avoid</h4>
            <p className="mt-2 text-sm leading-6">ANN search is not a quality guarantee. If indexing or search breadth misses the right chunk, reranking and generation never see it.</p>
          </div>
        </section>
      </div>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-5 text-cyan-950">
          <h3 className="text-sm font-black uppercase tracking-wide text-cyan-700">Problem solved</h3>
          <p className="mt-3 text-sm leading-6">Vector indexes make semantic search practical when the chunk corpus is too large for exact comparison.</p>
        </div>
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-5 text-amber-950">
          <h3 className="text-sm font-black uppercase tracking-wide text-amber-700">Core math</h3>
          <p className="mt-3 text-sm leading-6">Approximate search reduces candidate comparisons by probing clusters or walking a neighbor graph.</p>
        </div>
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-5 text-emerald-950">
          <h3 className="text-sm font-black uppercase tracking-wide text-emerald-700">Understanding check</h3>
          <p className="mt-3 text-sm leading-6">Increase search breadth and predict whether latency, recall, or both should rise.</p>
        </div>
      </section>

      <AssessmentPanel lessonId="rag-vector-indexing" title="Vector Indexing And ANN Search check" />
    </div>
  );
}
