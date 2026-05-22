import React, { useMemo, useState } from 'react';
import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  RefreshCw,
  Search,
  ShieldCheck,
} from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const CLAIMS = [
  {
    id: 'c-refund-eligibility',
    text: 'Annual customers can get a prorated refund within 30 days of renewal.',
    truth: true,
    supports: ['doc-policy', 'doc-legacy'],
    conflicts: ['doc-legacy-no-time-limit'],
  },
  {
    id: 'c-stale-policy',
    text: 'Refund eligibility changed in the July 2026 policy update.',
    truth: true,
    supports: ['doc-policy-updated'],
    conflicts: ['doc-legacy', 'doc-legacy-no-time-limit'],
  },
  {
    id: 'c-escalation-flow',
    text: 'Unusual refund disputes require finance escalation and a documented reason code.',
    truth: true,
    supports: ['doc-governance'],
    conflicts: ['doc-marketing'],
  },
  {
    id: 'c-irrelevant',
    text: 'The model should avoid giving a refund on feature-availability questions.',
    truth: false,
    supports: ['doc-policy', 'doc-governance'],
    conflicts: [],
  },
];

const CANDIDATES = [
  {
    id: 'doc-policy',
    title: 'Billing Policy v1.4',
    source: 'Billing Policy',
    sourceType: 'fresh',
    baseScore: 0.92,
    lexicalBoost: 0.12,
    crossEncoderBoost: 0.1,
    metadataBoost: 0.08,
    supports: ['c-refund-eligibility', 'c-stale-policy'],
    stale: false,
    conflicting: [],
    preview:
      'Annual subscriptions support prorated refunds only within 30 days of renewal. Disputes after that require finance escalation.',
  },
  {
    id: 'doc-legacy',
    title: 'Legacy Refund Note (2022)',
    source: 'Legacy Handbook',
    sourceType: 'stale',
    baseScore: 0.95,
    lexicalBoost: 0.05,
    crossEncoderBoost: 0.02,
    metadataBoost: -0.05,
    supports: ['c-refund-eligibility'],
    stale: true,
    conflicting: ['c-refund-eligibility'],
    preview: 'Annual customers can get prorated refunds for any amount at any time in the contract.',
  },
  {
    id: 'doc-governance',
    title: 'Governance Escalation Rules',
    source: 'Operations',
    sourceType: 'fresh',
    baseScore: 0.85,
    lexicalBoost: -0.01,
    crossEncoderBoost: 0.08,
    metadataBoost: 0.1,
    supports: ['c-escalation-flow'],
    stale: false,
    conflicting: [],
    preview: 'Escalate abnormal refund disputes to finance with a risk reason code.',
  },
  {
    id: 'doc-policy-updated',
    title: 'Billing Policy Addendum',
    source: 'Billing Policy',
    sourceType: 'fresh',
    baseScore: 0.78,
    lexicalBoost: 0.09,
    crossEncoderBoost: -0.06,
    metadataBoost: 0.05,
    supports: ['c-stale-policy'],
    stale: false,
    conflicting: [],
    preview: 'The July update introduced explicit eligibility date boundaries and time-stamp governance.',
  },
  {
    id: 'doc-marketing',
    title: 'Campaign Notes',
    source: 'Marketing',
    sourceType: 'irrelevant',
    baseScore: 0.91,
    lexicalBoost: 0.22,
    crossEncoderBoost: -0.14,
    metadataBoost: -0.09,
    supports: ['c-irrelevant'],
    stale: false,
    conflicting: [],
    preview: 'Feature adoption is up across enterprise plans this quarter.',
  },
  {
    id: 'doc-qa-bad-match',
    title: 'Support Q&A Archive',
    source: 'Support',
    sourceType: 'conflicting',
    baseScore: 0.74,
    lexicalBoost: -0.04,
    crossEncoderBoost: -0.02,
    metadataBoost: 0.01,
    supports: [],
    stale: false,
    conflicting: ['c-escalation-flow'],
    preview: 'Billing escalations are usually approved automatically for enterprise customers.',
  },
];

const DIAG_OPTIONS = [
  { id: 'grounded', label: 'Grounded and usable' },
  { id: 'missing', label: 'Missing evidence in top-k' },
  { id: 'stale', label: 'Stale evidence kept' },
  { id: 'irrelevant', label: 'Irrelevant top-k chunks' },
  { id: 'conflicting', label: 'Conflicting evidence dominates' },
];

const DIAGNOSIS_BY_FAILURE = {
  grounded: 'grounded and usable',
  missing: 'missing evidence in top-k',
  stale: 'stale evidence kept',
  irrelevant: 'irrelevant top-k chunks',
  conflicting: 'conflicting evidence dominates',
};

function rankCandidates(candidates, rerankerMode, rerankerWeight) {
  const modeWeight = {
    none: () => 0,
    lexical: (candidate) => candidate.lexicalBoost,
    'cross-encoder': (candidate) => candidate.crossEncoderBoost,
    metadata: (candidate) => candidate.metadataBoost,
  };

  const addBoost = modeWeight[rerankerMode] || modeWeight.none;

  return [...candidates].map((candidate) => ({
    ...candidate,
    rerankScore: candidate.baseScore + addBoost(candidate) * rerankerWeight,
  })).sort((left, right) => (
    right.rerankScore - left.rerankScore || right.baseScore - left.baseScore
  ));
}

function useable(candidate, strictness) {
  if (strictness >= 0.7) return !candidate.stale && candidate.conflicting.length === 0 && candidate.sourceType !== 'irrelevant';
  if (strictness >= 0.45) return !candidate.stale && candidate.sourceType !== 'irrelevant';
  return candidate.sourceType !== 'irrelevant';
}

function claimDiagnostic(claim, selectedCandidates, strictness) {
  const active = selectedCandidates.filter((candidate) => (
    candidate.supports.includes(claim.id) || candidate.conflicting.includes(claim.id)
  ));
  if (active.length === 0) {
    const relevantInPool = CANDIDATES.some((candidate) => (
      candidate.supports.includes(claim.id)
      || candidate.conflicting.includes(claim.id)
    ));
    return {
      state: relevantInPool ? 'missing' : 'irrelevant',
      detail: relevantInPool ? 'No relevant source reached top-k.' : 'No query-aligned source exists in top-k.',
      evidence: [],
    };
  }

  const usableSupport = active.find((candidate) => candidate.supports.includes(claim.id) && useable(candidate, strictness));
  const usableConflict = active.find((candidate) => candidate.conflicting.includes(claim.id) && useable(candidate, strictness));
  const hasStaleSupport = active.some((candidate) => (
    candidate.supports.includes(claim.id) && candidate.stale
  ));
  const hasConflict = active.some((candidate) => candidate.conflicting.includes(claim.id));

  if (usableSupport) {
    return {
      state: hasConflict ? 'conflicting' : 'grounded',
      detail: hasConflict
        ? `Supports the claim but also has conflicting signals in top-k.`
        : `Usable support candidate found above strictness threshold.`,
      evidence: active,
    };
  }

  if (hasStaleSupport) {
    return {
      state: 'stale',
      detail: `Only stale support candidates remain inside top-k at this strictness.`,
      evidence: active,
    };
  }

  if (hasConflict) {
    return {
      state: 'conflicting',
      detail: `No usable support is present; conflicting candidates remain visible in top-k.`,
      evidence: active,
    };
  }

  return {
    state: 'irrelevant',
    detail: 'Top-k is dominated by evidence that is not aligned to this claim.',
    evidence: active,
  };
}

function statusPill(state) {
  if (state === 'grounded') return 'bg-emerald-100 text-emerald-700';
  if (state === 'missing') return 'bg-rose-100 text-rose-700';
  if (state === 'stale') return 'bg-amber-100 text-amber-700';
  if (state === 'conflicting') return 'bg-purple-100 text-purple-700';
  return 'bg-slate-100 text-slate-700';
}

export default function RagFailureModesAnimation() {
  const [topK, setTopK] = useState(3);
  const [rerankerMode, setRerankerMode] = useState('cross-encoder');
  const [rerankerWeight, setRerankerWeight] = useState(0.7);
  const [strictness, setStrictness] = useState(0.6);
  const [diagnoses, setDiagnoses] = useState({});

  const reranked = useMemo(
    () => rankCandidates(CANDIDATES, rerankerMode, rerankerWeight),
    [rerankerMode, rerankerWeight],
  );
  const selected = useMemo(() => reranked.slice(0, topK), [reranked, topK]);
  const diagnostics = useMemo(
    () => CLAIMS.map((claim) => ({ ...claim, ...claimDiagnostic(claim, selected, strictness) })),
    [strictness, selected],
  );

  const groundedClaims = diagnostics.filter((entry) => entry.state === 'grounded').length;
  const staleClaims = diagnostics.filter((entry) => entry.state === 'stale').length;
  const missingClaims = diagnostics.filter((entry) => entry.state === 'missing').length;
  const conflictClaims = diagnostics.filter((entry) => entry.state === 'conflicting').length;
  const irrelevantClaims = diagnostics.filter((entry) => entry.state === 'irrelevant').length;

  const reset = () => {
    setTopK(3);
    setRerankerMode('cross-encoder');
    setRerankerWeight(0.7);
    setStrictness(0.6);
    setDiagnoses({});
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">RAG quality triage</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">RAG Failure Modes</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Before tuning model prompts, diagnose how the retrieval pipeline fails: missing evidence, stale chunks, irrelevant content,
              or conflicting claims. The goal is cleaner context before generation and a cleaner answer after.
            </p>
          </div>
          <button type="button" onClick={reset} className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-black text-slate-800">
            <RefreshCw size={16} />
            Reset
          </button>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-4">
        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <p className="text-xs font-black uppercase tracking-wide text-slate-500">Grounded claims</p>
          <strong className="text-2xl font-black text-emerald-700">{groundedClaims}/4</strong>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <p className="text-xs font-black uppercase tracking-wide text-slate-500">Failure mix</p>
          <strong className="text-2xl font-black text-slate-950">{staleClaims + conflictClaims + missingClaims + irrelevantClaims}</strong>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <p className="text-xs font-black uppercase tracking-wide text-slate-500">Top-k coverage</p>
          <strong className="text-2xl font-black text-slate-950">{topK}</strong>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <p className="text-xs font-black uppercase tracking-wide text-slate-500">Mode severity</p>
          <strong className="text-2xl font-black text-slate-950">
            {Math.round(((staleClaims + conflictClaims) / 4) * 100)}%
          </strong>
        </div>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <Search size={16} />
          Pipeline controls
        </div>
        <div className="grid gap-4 md:grid-cols-4">
          <label className="grid gap-2 text-sm font-black text-slate-700">
            Reranker
            <select
              className="rounded-lg border border-slate-200 bg-white px-3 py-2"
              value={rerankerMode}
              onChange={(event) => setRerankerMode(event.target.value)}
            >
              <option value="cross-encoder">Cross-encoder</option>
              <option value="lexical">Lexical</option>
              <option value="metadata">Metadata-aware</option>
            </select>
          </label>
          <label className="grid gap-2 text-sm font-black text-slate-700">
            Rerank weight: {rerankerWeight.toFixed(2)}
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={rerankerWeight}
              onChange={(event) => setRerankerWeight(Number(event.target.value))}
            />
          </label>
          <label className="grid gap-2 text-sm font-black text-slate-700">
            Top-k context: {topK}
            <input
              type="range"
              min="2"
              max="6"
              step="1"
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
            />
          </label>
          <label className="grid gap-2 text-sm font-black text-slate-700">
            Grounding strictness: {strictness.toFixed(2)}
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={strictness}
              onChange={(event) => setStrictness(Number(event.target.value))}
            />
          </label>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-[2fr_1fr]">
        <article className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <BookOpen size={16} />
            First-pass then rerank top-k
          </h3>
          <div className="mt-3 space-y-3">
            {selected.map((candidate, index) => (
              <div key={candidate.id} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="text-sm font-black text-slate-900">
                    rank {index + 1}: {candidate.title}
                  </p>
                  <span className="rounded bg-white px-2 py-1 text-xs font-black text-slate-700">
                    {candidate.rerankScore.toFixed(2)}
                  </span>
                </div>
                <p className="mt-2 text-sm text-slate-700">{candidate.preview}</p>
                <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                  {candidate.sourceType === 'stale' && <span className="rounded bg-amber-100 px-2 py-1 font-black">stale</span>}
                  {candidate.sourceType === 'irrelevant' && <span className="rounded bg-slate-200 px-2 py-1 font-black">irrelevant</span>}
                  {candidate.conflicting.length > 0 && <span className="rounded bg-rose-100 px-2 py-1 font-black">conflict</span>}
                  {candidate.sourceType === 'fresh' && !candidate.conflicting.length && (
                    <span className="rounded bg-emerald-100 px-2 py-1 font-black">fresh</span>
                  )}
                  {!useable(candidate, strictness) && <span className="rounded bg-slate-100 px-2 py-1 font-black">filtered by strictness</span>}
                </div>
              </div>
            ))}
          </div>
        </article>

        <article className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <ShieldCheck size={16} />
            First-pass coverage insight
          </h3>
          <div className="mt-3 space-y-3">
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <p className="text-xs font-black uppercase tracking-wide text-slate-500">Evidence found in first-pass top-k</p>
              <strong className="mt-1 block text-xl text-slate-900">
                {groundedClaims}/4 grounded
              </strong>
            </div>
            <div className="rounded-lg border border-amber-200 bg-amber-50 p-3">
              <p className="text-xs font-black uppercase tracking-wide text-amber-700">Takeaway</p>
              <p className="mt-2 text-sm text-amber-950">
                Missing-evidence failures can only be fixed by improving candidate generation, not by reranking alone.
              </p>
            </div>
            <div className="rounded-lg border border-rose-200 bg-rose-50 p-3">
              <p className="text-xs font-black uppercase tracking-wide text-rose-700">Takeaway</p>
              <p className="mt-2 text-sm text-rose-950">
                Strict grounding helps reject stale or conflicting chunks but cannot invent evidence.
              </p>
            </div>
          </div>
        </article>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <h3 className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <AlertTriangle size={16} />
          Diagnose each claim
        </h3>
        <div className="space-y-3">
          {diagnostics.map((claim) => {
            const diagnosis = diagnoses[claim.id];
            const isCorrect = diagnosis && diagnosis === claim.state;
            const isMismatch = diagnosis && diagnosis !== claim.state;

            return (
              <div key={claim.id} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <p className="max-w-3xl text-sm text-slate-900">{claim.text}</p>
                  <span className={`rounded px-2 py-1 text-xs font-black uppercase ${statusPill(claim.state)}`}>
                    {claim.state}
                  </span>
                </div>
                <p className="mt-2 text-sm text-slate-600">{claim.detail}</p>
                <label className="mt-3 flex flex-wrap items-center gap-2 text-sm font-black text-slate-700">
                  How did this claim fail?
                  <select
                    className="rounded border border-slate-200 bg-white px-2 py-1 text-sm"
                    value={diagnosis || ''}
                    onChange={(event) => setDiagnoses((current) => ({
                      ...current,
                      [claim.id]: event.target.value,
                    }))}
                  >
                    <option value="">Choose</option>
                    {DIAG_OPTIONS.map((option) => (
                      <option key={option.id} value={option.id}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
                {diagnosis && (
                  <p className={`mt-2 text-sm ${isCorrect ? 'text-emerald-700' : 'text-rose-700'}`}>
                    {isCorrect ? (
                      <span className="font-black">Correct diagnosis.</span>
                    ) : (
                      <span className="font-black">Expected: {DIAGNOSIS_BY_FAILURE[claim.state]}.</span>
                    )}
                    <span className="ml-2">Evidence seen in top-k: {claim.evidence.map((item) => item.title).join(', ') || 'none'}.</span>
                  </p>
                )}
              </div>
            );
          })}
        </div>
      </section>

      <section className="rounded-lg border border-slate-200 bg-indigo-50 p-5">
        <p className="text-sm text-indigo-900">
          Checklist:
        </p>
        <ul className="mt-2 space-y-2 text-sm text-indigo-950">
          <li className="flex items-start gap-2">
            <CheckCircle2 size={16} />
            Raise strictness when stale/conflicting chunks are visible.
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle2 size={16} />
            Increase retrieval width only when missing claims are common and corpus quality is strong.
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle2 size={16} />
            If most failures are irrelevant, fix indexing and preprocessing rather than just reranker strength.
          </li>
        </ul>
      </section>

      <AssessmentPanel lessonId="rag-failure-modes" title="RAG failure modes check" />
    </div>
  );
}
