import React, { useMemo, useState } from 'react';
import {
  AlertTriangle,
  FileText,
  GitBranch,
  RotateCcw,
  Search,
  ShieldCheck,
} from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const CLAIMS = [
  {
    id: 'c-refund',
    text: 'Annual billing is eligible for prorated refunds only within 30 days of renewal.',
  },
  {
    id: 'c-access',
    text: 'After the refund window, annual customers keep access until the term ends.',
  },
  {
    id: 'c-escalation',
    text: 'Unusual refund disputes should route through finance escalation.',
  },
];

const RAW_CANDIDATES = [
  {
    id: 'doc-billing-policy',
    source: 'Billing Policy',
    title: 'Annual billing and refund terms',
    baseScore: 0.86,
    lexicalBoost: 0.03,
    crossEncoderBoost: 0.16,
    metadataAwareBoost: 0.06,
    supports: ['c-refund', 'c-access'],
    stale: false,
    conflicting: false,
    preview: 'Annual plans can request a prorated refund within 30 days after renewal. After 30 days access remains active until term expiry.',
  },
  {
    id: 'doc-support-playbook',
    source: 'Support SOP',
    title: 'Legacy refund checklist',
    baseScore: 0.9,
    lexicalBoost: 0.08,
    crossEncoderBoost: -0.05,
    metadataAwareBoost: -0.06,
    supports: ['c-refund'],
    stale: true,
    conflicting: true,
    preview: 'All annual refunds can be approved directly by support agents after one confirmation call.',
  },
  {
    id: 'doc-admin-guideline',
    source: 'Admin Handbook',
    title: 'Account and invoice policy',
    baseScore: 0.74,
    lexicalBoost: 0.02,
    crossEncoderBoost: 0.04,
    metadataAwareBoost: 0.04,
    supports: ['c-escalation'],
    stale: false,
    conflicting: false,
    preview: 'Disputed refunds require finance escalation and a documented reason code before approval.',
  },
  {
    id: 'doc-ads-doc',
    source: 'Marketing Notes',
    title: 'Product launch highlights',
    baseScore: 0.88,
    lexicalBoost: -0.05,
    crossEncoderBoost: -0.12,
    metadataAwareBoost: -0.1,
    supports: [],
    stale: false,
    conflicting: false,
    preview: 'Launch timeline and rollout notes unrelated to billing behavior or refund eligibility.',
  },
];

function rankCandidates(candidates, rerankerMode, rerankerWeight) {
  const boostByMode = {
    none: () => 0,
    lexical: (candidate) => candidate.lexicalBoost,
    'cross-encoder': (candidate) => candidate.crossEncoderBoost,
    'metadata-aware': (candidate) => candidate.metadataAwareBoost,
  };

  const getBoost = boostByMode[rerankerMode] || boostByMode.none;
  return [...candidates]
    .map((candidate) => ({
      ...candidate,
      rerankScore: candidate.baseScore + getBoost(candidate) * rerankerWeight,
    }))
    .sort((a, b) => b.rerankScore - a.rerankScore || b.baseScore - a.baseScore);
}

function isUsable(candidate, strictness) {
  if (strictness >= 0.75) return !candidate.stale && !candidate.conflicting;
  if (strictness >= 0.45) return !candidate.stale;
  return true;
}

function groundedClaims(claims, selected, strictness) {
  const usable = selected.filter((candidate) => isUsable(candidate, strictness));
  const usableSources = new Set(usable.map((candidate) => candidate.id));

  return claims.map((claim) => {
    const provider = usable.find((candidate) => candidate.supports.includes(claim.id));
    return {
      ...claim,
      grounded: Boolean(provider),
      source: provider ? provider.title : null,
      sourceId: provider ? provider.source : null,
      sourceUsable: provider && usableSources.has(provider.id),
      issue: provider && provider.stale ? 'stale evidence' : provider && provider.conflicting ? 'conflicting evidence' : null,
    };
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

export default function RagRerankingGroundingAnimation() {
  const [rerankerMode, setRerankerMode] = useState('cross-encoder');
  const [rerankerWeight, setRerankerWeight] = useState(0.78);
  const [topK, setTopK] = useState(3);
  const [citationStrictness, setCitationStrictness] = useState(0.62);

  const firstPass = useMemo(() => [...RAW_CANDIDATES].sort((a, b) => b.baseScore - a.baseScore), []);
  const reranked = useMemo(
    () => rankCandidates(RAW_CANDIDATES, rerankerMode, rerankerWeight),
    [rerankerMode, rerankerWeight],
  );
  const selected = useMemo(() => reranked.slice(0, topK), [reranked, topK]);
  const claimStatus = useMemo(() => groundedClaims(CLAIMS, selected, citationStrictness), [citationStrictness, selected]);
  const groundedCount = claimStatus.filter((status) => status.grounded && status.sourceUsable).length;
  const groundedRatio = groundedCount / CLAIMS.length;
  const firstPassHasAllClaims = CLAIMS.every((claim) => (
    firstPass.slice(0, topK).some((candidate) => candidate.supports.includes(claim.id))
  ));
  const rerankRecall = selected.reduce((acc, candidate) => acc + candidate.supports.length, 0);

  const reset = () => {
    setRerankerMode('cross-encoder');
    setRerankerWeight(0.78);
    setTopK(3);
    setCitationStrictness(0.62);
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">RAG retrieval pipeline quality</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">RAG Reranking & Grounding</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Reranking changes which retrieved chunks move to the top, and grounding checks whether claims in the draft answer
              are supported by selected, non-conflicting evidence before marking them as cited.
            </p>
          </div>
          <button
            type="button"
            onClick={reset}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-black text-slate-800"
          >
            <RotateCcw size={16} />
            Reset
          </button>
        </div>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <Search size={16} />
          Pipeline controls
        </div>
        <div className="grid gap-4 lg:grid-cols-4">
          <label className="grid gap-2 text-sm font-black text-slate-700">
            Reranker:
            <select
              className="rounded-lg border border-slate-200 bg-white px-3 py-2"
              value={rerankerMode}
              onChange={(event) => setRerankerMode(event.target.value)}
            >
              <option value="none">None</option>
              <option value="lexical">Lexical</option>
              <option value="cross-encoder">Cross-Encoder</option>
              <option value="metadata-aware">Metadata-aware</option>
            </select>
          </label>
          <label className="grid gap-2 text-sm font-black text-slate-700">
            Reranker weight: {rerankerWeight.toFixed(2)}
            <input
              min="0"
              max="1"
              step="0.1"
              type="range"
              value={rerankerWeight}
              onChange={(event) => setRerankerWeight(Number(event.target.value))}
            />
          </label>
          <label className="grid gap-2 text-sm font-black text-slate-700">
            Top-k context: {topK}
            <input
              min="2"
              max="5"
              step="1"
              type="range"
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
            />
          </label>
          <label className="grid gap-2 text-sm font-black text-slate-700">
            Citation strictness: {citationStrictness.toFixed(2)}
            <input
              min="0"
              max="1"
              step="0.1"
              type="range"
              value={citationStrictness}
              onChange={(event) => setCitationStrictness(Number(event.target.value))}
            />
          </label>
        </div>
      </section>

      <div className="grid gap-3 md:grid-cols-4">
        <Stat label="Claims grounded" value={`${groundedCount}/${CLAIMS.length}`} detail="supported without strictness violations" />
        <Stat label="Grounded ratio" value={`${Math.round(groundedRatio * 100)}%`} detail="claim-level support quality" />
        <Stat label="Top-k support pool" value={rerankRecall} detail="raw support hits in reranked top-k set" />
        <Stat label="First-pass coverage" value={firstPassHasAllClaims ? 'Yes' : 'No'} detail="are all claims present before reranking?" />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <FileText size={16} />
            First-pass candidate order
          </h3>
          <div className="mt-4 space-y-3">
            {firstPass.slice(0, topK).map((candidate, index) => (
              <article key={candidate.id} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <strong className="text-sm text-slate-950">
                    rank {index + 1}: {candidate.title}
                  </strong>
                  <span className="rounded bg-white px-2 py-1 text-xs font-black text-slate-700">
                    {candidate.baseScore.toFixed(2)} score
                  </span>
                </div>
                <p className="mt-2 text-sm text-slate-700">{candidate.preview}</p>
                <span className="mt-3 block text-xs text-slate-500">{candidate.source}</span>
              </article>
            ))}
          </div>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <ShieldCheck size={16} />
            Reranked top-k
          </h3>
          <div className="mt-4 space-y-3">
            {selected.map((candidate, index) => (
              <article
                key={candidate.id}
                className={`rounded-lg border p-4 ${candidate.conflicting ? 'border-rose-200 bg-rose-50' : candidate.stale ? 'border-amber-200 bg-amber-50' : 'border-emerald-200 bg-emerald-50'}`}
              >
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <strong className="text-sm text-slate-950">
                    rank {index + 1}: {candidate.title}
                  </strong>
                  <span className="rounded bg-white px-2 py-1 text-xs font-black text-slate-700">
                    {candidate.rerankScore?.toFixed(2) || candidate.baseScore.toFixed(2)}
                  </span>
                </div>
                <p className="mt-2 text-sm text-slate-700">{candidate.preview}</p>
                <div className="mt-3 flex items-center gap-2 text-xs text-slate-600">
                  {candidate.stale && <span className="rounded bg-amber-100 px-2 py-1 font-black">stale</span>}
                  {candidate.conflicting && <span className="rounded bg-rose-100 px-2 py-1 font-black">conflict</span>}
                  {!candidate.stale && !candidate.conflicting && (
                    <span className="rounded bg-emerald-100 px-2 py-1 font-black">usable</span>
                  )}
                  {!isUsable(candidate, citationStrictness) && (
                    <span className="rounded bg-slate-100 px-2 py-1 font-black">filtered at this strictness</span>
                  )}
                </div>
                <span className="mt-3 block text-xs text-slate-500">{candidate.source}</span>
              </article>
            ))}
          </div>
        </section>
      </div>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Claim grounding check</h3>
        <div className="mt-3 space-y-3">
          {claimStatus.map((claim) => {
            const statusClass = claim.grounded
              ? claim.sourceUsable
                ? 'border-emerald-200 bg-emerald-50'
                : 'border-rose-200 bg-rose-50'
              : 'border-slate-200 bg-slate-50';

            return (
              <article key={claim.id} className={`rounded-lg border p-4 ${statusClass}`}>
                <div className="flex items-start justify-between gap-3">
                  <p className="text-sm text-slate-900">{claim.text}</p>
                  <GitBranch size={16} />
                </div>
                <p className="mt-2 text-sm text-slate-700">
                  {claim.grounded ? (
                    claim.sourceUsable ? (
                      `Grounded by ${claim.source} from ${claim.sourceId}.`
                    ) : (
                      `Evidence exists but fails strictness rules: ${claim.issue}.`
                    )
                  ) : (
                    'No supporting chunk in top-k context.'
                  )}
                </p>
              </article>
            );
          })}
        </div>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Failure modes</h3>
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <div className="rounded-lg border border-slate-200 bg-cyan-50 p-4">
            <h4 className="text-sm font-black uppercase tracking-wide text-cyan-700">Missing evidence</h4>
            <p className="mt-2 text-sm text-cyan-950">
              If a claim is not present in the top-k pool, no amount of reranker strength can ground it.
            </p>
          </div>
          <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
            <h4 className="text-sm font-black uppercase tracking-wide text-amber-700">Stale evidence</h4>
            <p className="mt-2 text-sm text-amber-950">
              Older playbooks can be lexically relevant but wrong; strict grounding should filter them out.
            </p>
          </div>
          <div className="rounded-lg border border-rose-200 bg-rose-50 p-4">
            <h4 className="text-sm font-black uppercase tracking-wide text-rose-700">Conflicting chunks</h4>
            <p className="mt-2 text-sm text-rose-950">
              A reranker can reorder conflicting evidence but cannot resolve contradictions from the source policy itself.
            </p>
          </div>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-lg border border-indigo-200 bg-indigo-50 p-5 text-indigo-950">
          <div className="mb-2 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-indigo-700">
            <AlertTriangle size={16} />
            Core idea
          </div>
          <p className="text-sm leading-6">
            Reranking increases the chance of grounding the right facts early, but the final grounding gate is where stale,
            conflicting, or low-confidence chunks are rejected.
          </p>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-5 text-slate-950">
          <p className="text-sm font-black uppercase tracking-wide">Try experiment</p>
          <ul className="mt-3 space-y-2 text-sm leading-6 text-slate-700">
            <li>Set strictness high and observe conflicting chunk rejection.</li>
            <li>Switch reranker from cross-encoder to lexical and compare claim grounding stability.</li>
            <li>Increase top-k and verify whether missing claims become grounded.</li>
          </ul>
        </div>
      </section>

      <AssessmentPanel lessonId="rag-reranking-grounding" title="RAG Reranking &amp; Grounding check" />
    </div>
  );
}
