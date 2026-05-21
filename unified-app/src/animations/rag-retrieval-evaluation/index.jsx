import React, { useMemo, useState } from 'react';
import { Database, RotateCcw, Search, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const QUERY = 'refund annual plan after cancellation';
const QUERY_TERMS = QUERY.split(' ');

const DOCUMENTS = [
  {
    title: 'Billing policy',
    sentences: [
      { id: 'b1', text: 'Monthly plans can be cancelled from the billing page.' },
      { id: 'b2', text: 'Annual plans are eligible for a prorated refund within 30 days of renewal.', relevant: true },
      { id: 'b3', text: 'After 30 days, annual cancellations keep access until the term ends.', relevant: true },
      { id: 'b4', text: 'Invoices are emailed to workspace owners.' },
    ],
  },
  {
    title: 'Support macros',
    sentences: [
      { id: 's1', text: 'Ask the customer for their workspace id before changing billing settings.' },
      { id: 's2', text: 'Escalate refund exceptions to the finance queue.' },
      { id: 's3', text: 'Trial extensions are handled by the success team.' },
    ],
  },
  {
    title: 'Admin setup',
    sentences: [
      { id: 'a1', text: 'Admins can invite users and assign roles from settings.' },
      { id: 'a2', text: 'Enterprise accounts can enforce SSO and domain capture.' },
      { id: 'a3', text: 'Plan upgrades take effect immediately after payment.' },
    ],
  },
];

const RELEVANT_SENTENCE_IDS = new Set(['b2', 'b3']);

function tokenize(text) {
  return text.toLowerCase().replace(/[^a-z0-9 ]/g, ' ').split(/\s+/).filter(Boolean);
}

function buildChunks(chunkSize, overlap) {
  return DOCUMENTS.flatMap((doc) => {
    const step = Math.max(1, chunkSize - overlap);
    const chunks = [];
    for (let start = 0; start < doc.sentences.length; start += step) {
      const sentences = doc.sentences.slice(start, start + chunkSize);
      if (!sentences.length) continue;
      chunks.push({
        id: `${doc.title}-${start}`,
        title: doc.title,
        sentenceIds: sentences.map((sentence) => sentence.id),
        text: sentences.map((sentence) => sentence.text).join(' '),
      });
      if (start + chunkSize >= doc.sentences.length) break;
    }
    return chunks;
  });
}

function lexicalScore(chunk) {
  const terms = tokenize(chunk.text);
  return QUERY_TERMS.reduce((score, term) => score + (terms.includes(term) ? 1 : 0), 0);
}

function rerankBoost(chunk) {
  const text = chunk.text.toLowerCase();
  let boost = 0;
  if (text.includes('prorated refund')) boost += 2.2;
  if (text.includes('annual')) boost += 1.1;
  if (text.includes('cancellations keep access')) boost += 0.9;
  if (text.includes('upgrade')) boost -= 0.6;
  if (text.includes('trial')) boost -= 0.4;
  return boost;
}

function rankChunks(chunks, useReranker) {
  return chunks
    .map((chunk) => {
      const base = lexicalScore(chunk);
      const score = base + (useReranker ? rerankBoost(chunk) : 0);
      const relevantCount = chunk.sentenceIds.filter((id) => RELEVANT_SENTENCE_IDS.has(id)).length;
      return { ...chunk, base, score, relevantCount };
    })
    .sort((a, b) => b.score - a.score || b.relevantCount - a.relevantCount || a.text.length - b.text.length);
}

function log2(value) {
  return Math.log(value) / Math.log(2);
}

function metrics(ranked, topK) {
  const top = ranked.slice(0, topK);
  const recovered = new Set();
  top.forEach((chunk) => {
    chunk.sentenceIds.forEach((id) => {
      if (RELEVANT_SENTENCE_IDS.has(id)) recovered.add(id);
    });
  });
  const firstRelevantIndex = top.findIndex((chunk) => chunk.relevantCount > 0);
  const dcg = top.reduce((sum, chunk, index) => sum + chunk.relevantCount / log2(index + 2), 0);
  const idealRelevance = ranked
    .map((chunk) => chunk.relevantCount)
    .sort((a, b) => b - a)
    .slice(0, topK);
  const ideal = idealRelevance.reduce((sum, relevance, index) => sum + relevance / log2(index + 2), 0);
  return {
    top,
    recall: recovered.size / RELEVANT_SENTENCE_IDS.size,
    mrr: firstRelevantIndex === -1 ? 0 : 1 / (firstRelevantIndex + 1),
    ndcg: ideal === 0 ? 0 : dcg / ideal,
  };
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

export default function RagRetrievalEvaluationAnimation() {
  const [chunkSize, setChunkSize] = useState(2);
  const [overlap, setOverlap] = useState(1);
  const [topK, setTopK] = useState(3);
  const [useReranker, setUseReranker] = useState(true);
  const chunks = useMemo(() => buildChunks(chunkSize, overlap), [chunkSize, overlap]);
  const ranked = useMemo(() => rankChunks(chunks, useReranker), [chunks, useReranker]);
  const result = useMemo(() => metrics(ranked, topK), [ranked, topK]);

  const reset = () => {
    setChunkSize(2);
    setOverlap(1);
    setTopK(3);
    setUseReranker(true);
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-bold uppercase tracking-wide text-slate-500">RAG retrieval quality</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Chunking, Reranking, and Retrieval Evaluation</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              A RAG system can only answer from what retrieval brings into context. Tune chunking, overlap, top-k, and
              reranking, then inspect whether the relevant evidence appears early enough to be useful.
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

      <div className="grid gap-3 md:grid-cols-4">
        <Stat label="Recall@k" value={`${Math.round(result.recall * 100)}%`} detail="relevant evidence recovered" />
        <Stat label="MRR" value={result.mrr.toFixed(2)} detail="first useful result rank" />
        <Stat label="nDCG" value={result.ndcg.toFixed(2)} detail="ranking quality with position" />
        <Stat label="Chunks" value={chunks.length} detail={`${chunkSize} sentence window`} />
      </div>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <SlidersHorizontal size={16} />
          Retrieval controls
        </div>
        <div className="grid gap-4 lg:grid-cols-4">
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Chunk size: {chunkSize} sentence{chunkSize === 1 ? '' : 's'}
            <input min="1" max="3" step="1" type="range" value={chunkSize} onChange={(event) => setChunkSize(Number(event.target.value))} />
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Overlap: {overlap} sentence
            <input min="0" max="1" step="1" type="range" value={overlap} onChange={(event) => setOverlap(Number(event.target.value))} />
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            top-k: {topK}
            <input min="2" max="5" step="1" type="range" value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
          </label>
          <label className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-bold text-slate-700">
            Reranker
            <input type="checkbox" checked={useReranker} onChange={(event) => setUseReranker(event.target.checked)} />
          </label>
        </div>
      </section>

      <div className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <Database size={16} />
            Chunked corpus
          </h3>
          <p className="mt-3 rounded-lg bg-slate-950 p-3 font-mono text-sm text-slate-100">query: {QUERY}</p>
          <div className="mt-4 space-y-3">
            {chunks.map((chunk) => (
              <article key={chunk.id} className={`rounded-lg border p-3 ${chunk.sentenceIds.some((id) => RELEVANT_SENTENCE_IDS.has(id)) ? 'border-emerald-200 bg-emerald-50' : 'border-slate-200 bg-slate-50'}`}>
                <div className="flex items-center justify-between gap-3">
                  <strong className="text-sm text-slate-950">{chunk.title}</strong>
                  <span className="rounded bg-white px-2 py-1 text-xs font-black text-slate-600">{chunk.sentenceIds.join(', ')}</span>
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-700">{chunk.text}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <Search size={16} />
            Ranked retrieval results
          </h3>
          <div className="mt-4 space-y-3">
            {result.top.map((chunk, index) => (
              <article key={chunk.id} className={`rounded-lg border p-4 ${chunk.relevantCount ? 'border-emerald-300 bg-emerald-50' : 'border-rose-200 bg-rose-50'}`}>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <strong className="text-sm text-slate-950">rank {index + 1}: {chunk.title}</strong>
                  <span className="rounded bg-white px-2 py-1 text-xs font-black text-slate-700">
                    score {chunk.score.toFixed(1)} {chunk.relevantCount ? 'relevant' : 'distractor'}
                  </span>
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-700">{chunk.text}</p>
              </article>
            ))}
          </div>
        </section>
      </div>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-blue-200 bg-blue-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-blue-700">Chunking failure</h3>
          <p className="mt-3 text-sm leading-6 text-blue-950">
            Chunks that are too small can split answer evidence apart; chunks that are too large can mix useful text
            with distractors and waste context.
          </p>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Reranking failure</h3>
          <p className="mt-3 text-sm leading-6 text-slate-700">
            A reranker helps reorder candidates after first-pass search, but it cannot recover evidence missing from
            the candidate set.
          </p>
        </div>
        <div className="rounded-lg border border-rose-200 bg-rose-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-rose-700">Metric failure</h3>
          <p className="mt-3 text-sm leading-6 text-rose-950">
            Recall@k, MRR, and nDCG answer different questions, so a retrieval evaluation should report more than one
            number.
          </p>
        </div>
      </section>

      <AssessmentPanel lessonId="rag-retrieval-evaluation" title="RAG Retrieval Evaluation check" />
    </div>
  );
}
