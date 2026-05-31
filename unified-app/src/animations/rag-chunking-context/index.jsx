import React, { useMemo, useState } from 'react';
import { PackageOpen, RotateCcw, Scissors, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const SENTENCES = [
  { id: 's1', text: 'Annual subscriptions renew on the original purchase date.', tokens: 9 },
  { id: 's2', text: 'Customers can cancel from the billing page at any time.', tokens: 10 },
  { id: 's3', text: 'A prorated refund is available within 30 days after renewal.', tokens: 11, relevant: true },
  { id: 's4', text: 'After 30 days, access continues until the annual term ends.', tokens: 11, relevant: true },
  { id: 's5', text: 'Refund exceptions must be escalated to the finance queue.', tokens: 9 },
  { id: 's6', text: 'Invoices are sent to workspace owners and billing admins.', tokens: 9 },
];

function buildChunks(chunkSize, overlap) {
  const step = Math.max(1, chunkSize - overlap);
  const chunks = [];
  for (let start = 0; start < SENTENCES.length; start += step) {
    const sentences = SENTENCES.slice(start, start + chunkSize);
    if (!sentences.length) continue;
    chunks.push({
      id: `chunk-${start}`,
      index: chunks.length + 1,
      sentences,
      tokenCount: sentences.reduce((sum, sentence) => sum + sentence.tokens, 0),
      relevantCount: sentences.filter((sentence) => sentence.relevant).length,
    });
    if (start + chunkSize >= SENTENCES.length) break;
  }
  return chunks;
}

function scoreChunk(chunk) {
  const terms = ['annual', 'refund', 'cancel', 'renewal', '30'];
  const text = chunk.sentences.map((sentence) => sentence.text.toLowerCase()).join(' ');
  return terms.reduce((score, term) => score + (text.includes(term) ? 1 : 0), 0) + chunk.relevantCount * 2;
}

function packContext(chunks, topK, budget) {
  const ranked = [...chunks].map((chunk) => ({ ...chunk, score: scoreChunk(chunk) })).sort((a, b) => b.score - a.score);
  let used = 0;
  return ranked.slice(0, topK).map((chunk) => {
    const fits = used + chunk.tokenCount <= budget;
    if (fits) used += chunk.tokenCount;
    return { ...chunk, packed: fits };
  });
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

export default function RagChunkingContextAnimation() {
  const [chunkSize, setChunkSize] = useState(2);
  const [overlap, setOverlap] = useState(1);
  const [topK, setTopK] = useState(4);
  const [budget, setBudget] = useState(44);

  const chunks = useMemo(() => buildChunks(chunkSize, Math.min(overlap, chunkSize - 1)), [chunkSize, overlap]);
  const packed = useMemo(() => packContext(chunks, topK, budget), [chunks, topK, budget]);
  const packedChunks = packed.filter((chunk) => chunk.packed);
  const usedTokens = packedChunks.reduce((sum, chunk) => sum + chunk.tokenCount, 0);
  const recoveredRelevant = new Set(packedChunks.flatMap((chunk) => chunk.sentences.filter((sentence) => sentence.relevant).map((sentence) => sentence.id))).size;

  const reset = () => {
    setChunkSize(2);
    setOverlap(1);
    setTopK(4);
    setBudget(44);
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">RAG document preparation</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">RAG Chunking And Context Packing</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Chunking decides what retrieval can find. Context packing decides which retrieved chunks actually fit beside
              the user question, instructions, and answer space.
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
          Pipeline controls
        </div>
        <div className="grid gap-4 md:grid-cols-4">
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Chunk size: {chunkSize} sentences
            <input min="1" max="4" step="1" type="range" value={chunkSize} onChange={(event) => setChunkSize(Number(event.target.value))} />
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Overlap: {Math.min(overlap, chunkSize - 1)}
            <input min="0" max="3" step="1" type="range" value={overlap} onChange={(event) => setOverlap(Number(event.target.value))} />
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Top-k retrieved: {topK}
            <input min="1" max="6" step="1" type="range" value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Context budget: {budget} tokens
            <input min="20" max="70" step="4" type="range" value={budget} onChange={(event) => setBudget(Number(event.target.value))} />
          </label>
        </div>
      </section>

      <div className="grid gap-3 md:grid-cols-4">
        <Stat label="Chunks" value={chunks.length} detail="indexed units" />
        <Stat label="Packed" value={packedChunks.length} detail="chunks that fit" />
        <Stat label="Context used" value={`${usedTokens}/${budget}`} detail="token budget consumed" />
        <Stat label="Recall" value={`${recoveredRelevant}/2`} detail="relevant facts packed" />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_0.9fr]">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <Scissors size={16} />
            Indexed chunks
          </h3>
          <div className="mt-4 grid gap-3">
            {chunks.map((chunk) => (
              <div key={chunk.id} className="rounded-lg border border-cyan-200 bg-cyan-50 p-3">
                <p className="text-xs font-black uppercase tracking-wide text-cyan-700">Chunk {chunk.index} - {chunk.tokenCount} tokens</p>
                <p className="mt-2 text-sm leading-6 text-cyan-950">{chunk.sentences.map((sentence) => sentence.text).join(' ')}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <PackageOpen size={16} />
            Packed context
          </h3>
          <div className="mt-4 space-y-3">
            {packed.map((chunk) => (
              <div key={chunk.id} className={`rounded-lg border p-3 ${chunk.packed ? 'border-emerald-300 bg-emerald-50' : 'border-slate-200 bg-slate-50 text-slate-500'}`}>
                <p className="text-xs font-black uppercase tracking-wide">Score {chunk.score} - {chunk.tokenCount} tokens - {chunk.packed ? 'packed' : 'dropped'}</p>
                <p className="mt-2 text-sm leading-6">{chunk.sentences.map((sentence) => sentence.text).join(' ')}</p>
              </div>
            ))}
          </div>
        </section>
      </div>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-5 text-cyan-950">
          <h3 className="text-sm font-black uppercase tracking-wide text-cyan-700">Problem solved</h3>
          <p className="mt-3 text-sm leading-6">Good chunking preserves answerable facts in retrievable units without wasting context on unrelated text.</p>
        </div>
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-5 text-amber-950">
          <h3 className="text-sm font-black uppercase tracking-wide text-amber-700">Core math</h3>
          <p className="mt-3 text-sm leading-6">Chunk count depends on size and overlap; context packing is a ranked selection problem under a token budget.</p>
        </div>
        <div className="rounded-lg border border-rose-200 bg-rose-50 p-5 text-rose-950">
          <h3 className="text-sm font-black uppercase tracking-wide text-rose-700">Mistake to avoid</h3>
          <p className="mt-3 text-sm leading-6">More overlap and larger top-k can improve recall, but they can also crowd out better evidence or answer space.</p>
        </div>
      </section>

      <AssessmentPanel lessonId="rag-chunking-context" title="RAG Chunking And Context Packing check" />
    </div>
  );
}
