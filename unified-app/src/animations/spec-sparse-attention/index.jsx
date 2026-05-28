import React, { useMemo, useState } from 'react';
import {
  ArrowRight,
  BarChart3,
  BookOpen,
  CheckCircle2,
  ClipboardList,
  Code2,
  Cpu,
  Database,
  FileText,
  Gauge,
  GitBranch,
  HelpCircle,
  Layers3,
  Link as LinkIcon,
  RotateCcw,
  SlidersHorizontal,
  Sparkles,
  Zap,
} from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const BLOCK_COUNT = 32;
const BLOCK_IDS = Array.from({ length: BLOCK_COUNT }, (_, id) => id);
const DRAFT_TOKENS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'];

const MODES = [
  { id: 'dense', label: 'Dense decoding', detail: 'One target pass per token.' },
  { id: 'speculative', label: 'Speculative decoding', detail: 'Draft several tokens, verify together.' },
  { id: 'specAttn', label: 'SpecAttn drafting', detail: 'Sparse draft selection is guided by verification.' },
  { id: 'specSA', label: 'SpecSA verification', detail: 'Sparse verifier queries are grouped for reuse.' },
  { id: 'combined', label: 'Combined', detail: 'Guided sparse drafting plus grouped sparse verification.' },
];

const VERIFICATION_MODES = [
  { id: 'dense', label: 'Dense' },
  { id: 'naiveSparse', label: 'Sparse naive' },
  { id: 'exactMerged', label: 'Exact merged schedule' },
  { id: 'sharedIndex', label: 'Approx shared-index' },
];

const TRAVERSALS = [
  { id: 'flat', label: 'Flat' },
  { id: 'bfs', label: 'BFS tree' },
  { id: 'dfs', label: 'DFS tree' },
];

const PRECISION_CLASSES = [
  { id: 'strict', label: 'Strict' },
  { id: 'reuseOnly', label: 'Reuse-only' },
  { id: 'approxOnly', label: 'Approx-only' },
  { id: 'approxReuse', label: 'Approx+Reuse' },
];

const STORYBOARD = [
  ['The KV cache wall', 'One generated token can require scanning a huge prefix.'],
  ['Speculative decoding amortizes verification', 'Verify several future positions in one target pass.'],
  ['Sparse attention reduces each query working set', 'Fewer KV blocks are read per attention computation.'],
  ['The mismatch', 'Speculative verification wants shared work; dynamic sparsity creates query-specific work.'],
  ['SpecAttn maps the next draft', 'Full-attention verification yields accept decisions and critical KV scores.'],
  ['Collect-2-Query', 'Boundary verifier rows capture useful diversity without dumping every logit row.'],
  ['SpecSA exact merged schedule', 'Load the union once, then mask per query.'],
  ['SpecSA approximate shared-index', 'Reuse one representative sparse layout for nearby verifier queries.'],
  ['Refresh/reuse layers', 'Recompute sparse routing only at refresh layers.'],
  ['Prompt-adaptive planner', 'Choose draft shape, traversal, grouping, precision class, and refresh period.'],
];

const EXERCISES = [
  ['01_accept_prefix.rs', 'Count draft tokens accepted before the first mismatch.'],
  ['02_sparse_ratio.rs', 'Convert sparse ratio into a selected KV count.'],
  ['03_critical_kv_scores.rs', 'Average verification logits and select top critical entries.'],
  ['04_collect_two_query.rs', 'Pick the first draft token and bonus token rows.'],
  ['05_block_overlap.rs', 'Compute intersection-over-union for selected KV blocks.'],
  ['06_exact_merged_schedule.rs', 'Build the sorted union and preserve per-query masks.'],
  ['07_shared_index.rs', 'Copy a representative sparse layout across the group.'],
  ['08_refresh_reuse.rs', 'Mark transformer layers as refresh or reuse.'],
  ['09_strategy_planner.rs', 'Pick the strategy with highest accepted-token throughput.'],
];

const SOURCE_LINKS = [
  {
    label: 'SpecSA paper',
    href: 'https://arxiv.org/html/2605.19893v1',
    note: 'Sparse speculative verification, overlap-aware query grouping, exact and approximate scheduling, and profile-guided orchestration.',
  },
  {
    label: 'SpecAttn paper',
    href: 'https://arxiv.org/html/2602.07223v1',
    note: 'Self-speculative sparse drafting guided by full-attention verification logits and Collect-2-Query.',
  },
];

const QUESTIONS = [
  ['What are the two bottlenecks this lesson combines?', 'Speculative decoding reduces target-model passes; sparse attention reduces KV entries read per pass.'],
  ['Why does sparse attention help more in long-context decoding?', 'KV-cache access grows with context length, so skipping irrelevant blocks saves more memory traffic.'],
  ['What does speculative decoding accept?', 'The longest prefix of draft tokens that passes target-model verification.'],
  ['In SpecAttn, what model drafts the tokens?', 'The original target model, but run with sparse attention during drafting.'],
  ['Why does full-attention verification preserve quality?', 'The full target model remains the authority for final accepted and fallback tokens.'],
  ['What extra information does verification provide?', 'Attention logits or weights that reveal which KV entries were important.'],
  ['Why is selecting from only the last accepted token brittle?', 'It can overfit to one position and hurt later draft-token acceptance.'],
  ['Why use first draft token and bonus token for Collect-2-Query?', 'They are far apart in the draft chain and capture more diversity with low overhead.'],
  ['Why does naive sparse verification duplicate work?', 'Nearby verifier queries may overlap, but independent sparse kernels reload shared blocks.'],
  ['What does exact merged scheduling preserve?', 'Each query original selected-block semantics.'],
  ['What does approximate shared-index scheduling trade?', 'It trades exact per-query layouts for simpler execution and greater reuse.'],
  ['What is a refresh layer?', 'A layer that recomputes selected sparse indices.'],
  ['What is a reuse layer?', 'A layer that reuses the selected indices from the latest refresh layer.'],
  ['Why does SpecSA need a planner?', 'Draft shape, traversal, grouping, precision class, and refresh schedule interact with prompt behavior and kernel cost.'],
];

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function pct(value) {
  return `${Math.round(value * 100)}%`;
}

function fmt(value, digits = 1) {
  return Number(value).toFixed(digits);
}

function selectedCountForRatio(sparseRatio) {
  return Math.max(4, Math.ceil(BLOCK_COUNT * sparseRatio));
}

function sortedUnique(values) {
  return [...new Set(values)].sort((a, b) => a - b);
}

function patternForQuery(queryIndex, sparseRatio, traversal) {
  const target = selectedCountForRatio(sparseRatio);
  const traversalOffset = traversal === 'bfs' ? 2 : traversal === 'dfs' ? 5 : 0;
  const selected = [0, 1, BLOCK_COUNT - 2, BLOCK_COUNT - 1];
  let step = 0;

  while (selected.length < target) {
    const candidate = 2 + ((queryIndex * 3 + step * 5 + traversalOffset) % (BLOCK_COUNT - 4));
    if (!selected.includes(candidate)) selected.push(candidate);
    step += 1;
  }

  return sortedUnique(selected);
}

function makeQueries({ draftLength, sparseRatio, traversal }) {
  const count = Math.max(1, draftLength);
  return Array.from({ length: count }, (_, index) => ({
    id: index + 1,
    position: 128000 + index,
    selectedBlocks: patternForQuery(index, sparseRatio, traversal),
    groupId: Math.floor(index / 4),
  }));
}

function makeCriticality(sparseRatio) {
  const first = BLOCK_IDS.map((id) => {
    const sink = id <= 1 ? 4.7 : 0;
    const local = id >= 28 ? 3.2 : 0;
    const fact = [6, 11, 17].includes(id) ? 3.8 : 0;
    const ripple = ((id * 7) % 13) / 13;
    return 0.55 + sink + local + fact + ripple;
  });

  const bonus = BLOCK_IDS.map((id) => {
    const sink = id <= 1 ? 4.1 : 0;
    const local = id >= 28 ? 2.9 : 0;
    const fact = [11, 17, 23].includes(id) ? 4.2 : 0;
    const ripple = ((id * 5 + 3) % 17) / 15;
    return 0.45 + sink + local + fact + ripple;
  });

  const average = first.map((score, index) => (score + bonus[index]) / 2);
  const topK = selectedCountForRatio(sparseRatio);
  const selected = average
    .map((score, index) => ({ index, score }))
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .slice(0, topK)
    .map((item) => item.index)
    .sort((a, b) => a - b);

  return { first, bonus, average, selected };
}

function effectiveVerificationMode(mode, requestedMode) {
  if (mode === 'dense') return 'dense';
  if (mode === 'speculative' || mode === 'specAttn') return 'dense';
  return requestedMode === 'dense' ? 'exactMerged' : requestedMode;
}

function buildSimulation(config) {
  const {
    mode,
    sparseRatio,
    draftLength,
    verificationMode: requestedVerificationMode,
    traversal,
    precisionClass,
    refreshEvery,
  } = config;
  const queryCount = mode === 'dense' ? 1 : Math.max(1, draftLength);
  const queries = makeQueries({ draftLength: queryCount, sparseRatio, traversal });
  const verificationMode = effectiveVerificationMode(mode, requestedVerificationMode);
  const representativeIndex = Math.min(1, queries.length - 1);
  const representativeBlocks = queries[representativeIndex].selectedBlocks;
  const unionBlocks = sortedUnique(queries.flatMap((query) => query.selectedBlocks));
  const denseRows = queries.map((query) => ({ ...query, selectedBlocks: BLOCK_IDS }));
  const sharedRows = queries.map((query) => ({ ...query, selectedBlocks: representativeBlocks }));

  const displayRows = verificationMode === 'dense'
    ? denseRows
    : verificationMode === 'sharedIndex'
      ? sharedRows
      : queries;

  const totalIndependentLoads = verificationMode === 'dense'
    ? queryCount * BLOCK_COUNT
    : queries.reduce((total, query) => total + query.selectedBlocks.length, 0);

  const kvBlocksLoaded = verificationMode === 'dense'
    ? queryCount * BLOCK_COUNT
    : verificationMode === 'naiveSparse'
      ? totalIndependentLoads
      : verificationMode === 'exactMerged'
        ? unionBlocks.length
        : representativeBlocks.length;

  const uniqueBlocksLoaded = verificationMode === 'dense'
    ? BLOCK_COUNT
    : verificationMode === 'sharedIndex'
      ? representativeBlocks.length
      : unionBlocks.length;

  const duplicateLoadCount = verificationMode === 'naiveSparse'
    ? Math.max(0, totalIndependentLoads - unionBlocks.length)
    : verificationMode === 'dense'
      ? Math.max(0, queryCount * BLOCK_COUNT - BLOCK_COUNT)
      : 0;

  const sparseAccuracy = clamp(0.42 + sparseRatio * 2.6, 0.38, 0.93);
  const modeBoost =
    mode === 'combined' ? 0.2
      : mode === 'specAttn' ? 0.17
        : mode === 'specSA' ? 0.06
          : mode === 'speculative' ? 0.03
            : 0;
  const approxPenalty = verificationMode === 'sharedIndex' && precisionClass !== 'strict' ? 0.04 : 0;
  const acceptanceRate = clamp(sparseAccuracy + modeBoost - approxPenalty, 0.18, 0.96);
  const acceptedLength = mode === 'dense'
    ? 1
    : Math.max(1, Math.min(draftLength, Math.round(draftLength * acceptanceRate)));

  const sparseDrafting = mode === 'specAttn' || mode === 'combined';
  const draftLatency = mode === 'dense'
    ? 0
    : draftLength * (sparseDrafting ? 0.55 + sparseRatio * 5.5 : 1.12);

  const refreshDiscount = refreshEvery > 1 && (mode === 'specSA' || mode === 'combined') ? 0.68 : 1;
  const precisionDiscount = precisionClass === 'approxReuse'
    ? 0.74
    : precisionClass === 'approxOnly'
      ? 0.82
      : precisionClass === 'reuseOnly'
        ? 0.9
        : 1;
  const selectionOverhead = verificationMode === 'dense'
    ? 0
    : verificationMode === 'naiveSparse'
      ? queryCount * 0.3
      : verificationMode === 'exactMerged'
        ? 1.5 + unionBlocks.length * 0.04
        : 0.55;
  const verifyLatency = (2.4 + kvBlocksLoaded * 0.18 + selectionOverhead) * refreshDiscount * precisionDiscount;
  const roundLatency = draftLatency + verifyLatency;
  const acceptedTokenThroughput = acceptedLength / (roundLatency / 1000);

  const criticality = makeCriticality(sparseRatio);

  return {
    acceptedLength,
    acceptedTokenThroughput,
    criticality,
    displayRows,
    draftLatency,
    duplicateLoadCount,
    kvBlocksLoaded,
    queryCount,
    queries,
    representativeBlocks,
    roundLatency,
    selectionOverhead,
    uniqueBlocksLoaded,
    unionBlocks,
    verificationMode,
    verifyLatency,
  };
}

function SectionHeader({ icon: Icon, eyebrow, title, children }) {
  return (
    <div className="mb-4">
      <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-cyan-700">
        <Icon size={16} />
        {eyebrow}
      </p>
      <h3 className="mt-2 text-xl font-black tracking-normal text-slate-950">{title}</h3>
      {children ? <p className="mt-2 text-sm leading-6 text-slate-700">{children}</p> : null}
    </div>
  );
}

function ControlButton({ active, children, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`w-full rounded-lg border px-3 py-2 text-left text-sm font-bold transition ${
        active
          ? 'border-cyan-800 bg-cyan-800 text-white'
          : 'border-slate-200 bg-white text-slate-700 hover:border-cyan-600'
      }`}
    >
      {children}
    </button>
  );
}

function Metric({ icon: Icon, label, value, detail }) {
  return (
    <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-4">
      <div className="mb-2 flex min-w-0 flex-wrap items-center gap-2 text-xs font-black uppercase tracking-wide text-slate-500">
        <Icon size={16} />
        {label}
      </div>
      <strong className="block text-2xl font-black text-slate-950">{value}</strong>
      <p className="mt-1 break-words text-sm leading-6 text-slate-600">{detail}</p>
    </div>
  );
}

function TokenTimeline({ mode, draftLength, acceptedLength }) {
  if (mode === 'dense') {
    return (
      <div className="grid gap-3">
        {['q_t', 'attention', 'token'].map((item, index) => (
          <div key={item} className="grid grid-cols-[48px_1fr] items-center gap-3">
            <span className="font-mono text-xs font-black text-slate-500">{index + 1}</span>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm font-bold text-slate-800">
              {index === 0 ? 'Single decode query reads the visible KV cache' : index === 1 ? 'Dense attention over the prefix' : 'One new token is committed'}
            </div>
          </div>
        ))}
      </div>
    );
  }

  const tokens = DRAFT_TOKENS.slice(0, draftLength);
  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
        <div className="mb-3 text-xs font-black uppercase tracking-wide text-slate-500">Drafter proposes</div>
        <div className="flex flex-wrap gap-2">
          {tokens.map((token) => (
            <span key={token} className="inline-flex h-10 min-w-10 items-center justify-center rounded-lg border border-slate-300 bg-white px-3 font-mono text-sm font-black text-slate-900">
              {token}
            </span>
          ))}
        </div>
      </div>

      <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
        <div className="mb-3 text-xs font-black uppercase tracking-wide text-slate-500">Target verifies</div>
        <div className="flex flex-wrap gap-2">
          {tokens.map((token, index) => {
            const accepted = index < acceptedLength;
            return (
              <span
                key={token}
                className={`inline-flex h-10 min-w-10 items-center justify-center rounded-lg border px-3 font-mono text-sm font-black ${
                  accepted
                    ? 'border-emerald-600 bg-emerald-50 text-emerald-700'
                    : 'border-rose-300 bg-rose-50 text-rose-700'
                }`}
              >
                {accepted ? token : `${token}x`}
              </span>
            );
          })}
          {acceptedLength < draftLength ? (
            <span className="inline-flex h-10 min-w-10 items-center justify-center rounded-lg border border-cyan-500 bg-cyan-50 px-3 font-mono text-sm font-black text-cyan-800">
              {tokens[acceptedLength]}'
            </span>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function BlockCell({ selected, duplicate, selectedByCriticality, dense }) {
  let background = '#f4f0e7';
  let border = '#dfd7ca';

  if (dense) {
    background = '#dbeafe';
    border = '#93c5fd';
  } else if (selected) {
    background = '#dff4ef';
    border = '#58a087';
  }

  if (selectedByCriticality) {
    background = selected ? '#d9ece5' : '#fff3d1';
    border = '#b8822f';
  }

  return (
    <span
      className="relative h-5 min-w-4 rounded-sm border"
      style={{ backgroundColor: background, borderColor: border }}
    >
      {duplicate ? <i className="absolute inset-x-1 bottom-0 h-1 rounded-full bg-orange-500" /> : null}
    </span>
  );
}

function KVGrid({ simulation, showCriticality = false }) {
  const { displayRows, verificationMode, queries, unionBlocks, criticality } = simulation;
  const hitCounts = useMemo(() => {
    const counts = new Map();
    queries.forEach((query) => {
      query.selectedBlocks.forEach((block) => counts.set(block, (counts.get(block) || 0) + 1));
    });
    return counts;
  }, [queries]);
  const selectedCritical = new Set(criticality.selected);
  const rows = verificationMode === 'exactMerged'
    ? [{ id: 'union', label: 'union', selectedBlocks: unionBlocks, union: true }, ...displayRows]
    : displayRows.map((row) => ({ ...row, label: `q${row.id}` }));

  return (
    <div className="overflow-x-auto">
      <div className="min-w-[720px]">
        <div className="mb-2 grid gap-1" style={{ gridTemplateColumns: `76px repeat(${BLOCK_COUNT}, minmax(12px, 1fr))` }}>
          <span />
          {BLOCK_IDS.map((id) => (
            <span key={id} className="text-center font-mono text-[10px] font-black text-slate-400">
              {id % 4 === 0 ? id : ''}
            </span>
          ))}
        </div>
        <div className="grid gap-1">
          {rows.map((row) => {
            const dense = verificationMode === 'dense';
            const selectedSet = new Set(row.selectedBlocks);
            return (
              <div key={row.id} className="grid items-center gap-1" style={{ gridTemplateColumns: `76px repeat(${BLOCK_COUNT}, minmax(12px, 1fr))` }}>
                <span className={`font-mono text-xs font-black ${row.union ? 'text-cyan-700' : 'text-slate-500'}`}>
                  {row.label}
                </span>
                {BLOCK_IDS.map((blockId) => {
                  const selected = selectedSet.has(blockId);
                  return (
                    <BlockCell
                      key={blockId}
                      dense={dense}
                      selected={selected}
                      duplicate={verificationMode === 'naiveSparse' && selected && hitCounts.get(blockId) > 1}
                      selectedByCriticality={showCriticality && selectedCritical.has(blockId)}
                    />
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function CriticalityHeatmap({ criticality }) {
  const maxScore = Math.max(...criticality.first, ...criticality.bonus, ...criticality.average);
  const rows = [
    ['first draft', criticality.first],
    ['bonus token', criticality.bonus],
    ['average', criticality.average],
  ];
  const selected = new Set(criticality.selected);

  return (
    <div className="overflow-x-auto">
      <div className="min-w-[720px]">
        <div className="grid gap-1">
          {rows.map(([label, values]) => (
            <div key={label} className="grid items-center gap-1" style={{ gridTemplateColumns: `96px repeat(${BLOCK_COUNT}, minmax(12px, 1fr))` }}>
              <span className="font-mono text-xs font-black text-slate-500">{label}</span>
              {values.map((value, index) => {
                const alpha = 0.12 + (value / maxScore) * 0.78;
                return (
                  <span
                    key={index}
                    className={`h-8 rounded-sm border ${selected.has(index) && label === 'average' ? 'border-amber-600' : 'border-slate-200'}`}
                    style={{ backgroundColor: `rgba(38, 66, 115, ${alpha})` }}
                  />
                );
              })}
            </div>
          ))}
        </div>
        <p className="mt-3 text-sm leading-6 text-slate-700">
          The average row becomes a criticality score. The highlighted cells are carried into the next sparse draft.
        </p>
      </div>
    </div>
  );
}

function QueryLayout({ simulation }) {
  const rows = simulation.queries.slice(0, Math.min(6, simulation.queries.length));
  return (
    <div className="grid gap-3">
      {rows.map((query) => (
        <div key={query.id} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div className="mb-2 flex items-center justify-between gap-3 text-xs font-black uppercase tracking-wide text-slate-500">
            <span>q{query.id}</span>
            <span>group {query.groupId + 1}</span>
          </div>
          <div className="flex flex-wrap gap-1">
            {query.selectedBlocks.map((block) => (
              <span key={block} className="rounded border border-slate-300 bg-white px-2 py-1 font-mono text-xs font-black text-slate-700">
                {block}
              </span>
            ))}
          </div>
        </div>
      ))}
      <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-3 text-sm leading-6 text-cyan-950">
        <strong className="block text-cyan-950">Current execution plan</strong>
        {simulation.verificationMode === 'exactMerged'
          ? `Union has ${simulation.unionBlocks.length} unique blocks. Each query keeps its own mask.`
          : simulation.verificationMode === 'sharedIndex'
            ? `Representative layout has ${simulation.representativeBlocks.length} blocks and is copied to the group.`
            : simulation.verificationMode === 'naiveSparse'
              ? `${simulation.duplicateLoadCount} duplicate loads remain because each row runs independently.`
              : 'Dense verification keeps a regular batch, but every query reads every block.'}
      </div>
    </div>
  );
}

function LayerStack({ refreshEvery }) {
  const layers = Array.from({ length: 8 }, (_, index) => ({
    id: index + 1,
    mode: index % refreshEvery === 0 ? 'Refresh' : 'Reuse',
  }));

  return (
    <div className="grid gap-2">
      {layers.map((layer) => (
        <div
          key={layer.id}
          className={`grid grid-cols-[72px_1fr] items-center gap-3 rounded-lg border p-3 ${
            layer.mode === 'Refresh'
              ? 'border-cyan-300 bg-cyan-50'
              : 'border-slate-200 bg-slate-50'
          }`}
        >
          <span className="font-mono text-xs font-black text-slate-500">L{layer.id}</span>
          <span className="text-sm font-black text-slate-800">{layer.mode}</span>
        </div>
      ))}
    </div>
  );
}

function ThroughputChart({ config }) {
  const chartConfigs = [
    ['Dense', { ...config, mode: 'dense', verificationMode: 'dense' }],
    ['Spec dense', { ...config, mode: 'speculative', verificationMode: 'dense' }],
    ['Naive sparse', { ...config, mode: 'specSA', verificationMode: 'naiveSparse', precisionClass: 'strict', refreshEvery: 1 }],
    ['SpecAttn', { ...config, mode: 'specAttn', verificationMode: 'dense' }],
    ['SpecSA exact', { ...config, mode: 'specSA', verificationMode: 'exactMerged', precisionClass: 'reuseOnly', refreshEvery: Math.max(2, config.refreshEvery) }],
    ['Approx+reuse', { ...config, mode: 'combined', verificationMode: 'sharedIndex', precisionClass: 'approxReuse', refreshEvery: Math.max(2, config.refreshEvery) }],
  ];
  const rows = chartConfigs.map(([label, scenario]) => ({
    label,
    simulation: buildSimulation(scenario),
  }));
  const maxValue = Math.max(...rows.map((row) => row.simulation.acceptedTokenThroughput));

  return (
    <div className="grid gap-3">
      {rows.map((row) => {
        const width = `${Math.max(6, (row.simulation.acceptedTokenThroughput / maxValue) * 100)}%`;
        return (
          <div key={row.label} className="grid grid-cols-[112px_1fr_76px] items-center gap-3">
            <span className="text-xs font-black uppercase tracking-wide text-slate-500">{row.label}</span>
            <div className="h-4 rounded-full bg-slate-100">
              <div className="h-4 rounded-full bg-cyan-700" style={{ width }} />
            </div>
            <span className="font-mono text-xs font-black text-slate-700">
              {fmt(row.simulation.acceptedTokenThroughput, 0)}/s
            </span>
          </div>
        );
      })}
      <p className="text-sm leading-6 text-slate-600">
        Values are synthetic and illustrative. Use the relative movement to reason about accepted tokens per round and KV traffic.
      </p>
    </div>
  );
}

function SourceCard({ source }) {
  return (
    <a
      href={source.href}
      target="_blank"
      rel="noreferrer"
      className="rounded-lg border border-slate-200 bg-slate-50 p-4 transition hover:border-cyan-500"
    >
      <span className="font-black text-slate-950">{source.label}</span>
      <span className="mt-1 block text-sm leading-6 text-slate-600">{source.note}</span>
    </a>
  );
}

export default function SpecSparseAttention() {
  const [mode, setMode] = useState('combined');
  const [sparseRatio, setSparseRatio] = useState(0.1);
  const [draftLength, setDraftLength] = useState(6);
  const [verificationMode, setVerificationMode] = useState('exactMerged');
  const [refreshEvery, setRefreshEvery] = useState(3);
  const [traversal, setTraversal] = useState('bfs');
  const [precisionClass, setPrecisionClass] = useState('reuseOnly');

  const config = {
    mode,
    sparseRatio,
    draftLength,
    verificationMode,
    refreshEvery,
    traversal,
    precisionClass,
  };

  const simulation = useMemo(() => buildSimulation(config), [
    mode,
    sparseRatio,
    draftLength,
    verificationMode,
    refreshEvery,
    traversal,
    precisionClass,
  ]);

  const reset = () => {
    setMode('combined');
    setSparseRatio(0.1);
    setDraftLength(6);
    setVerificationMode('exactMerged');
    setRefreshEvery(3);
    setTraversal('bfs');
    setPrecisionClass('reuseOnly');
  };

  return (
    <div className="min-w-0 space-y-6">
      <section className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-cyan-700">
              <BookOpen size={16} />
              Two-paper lesson - Transformers and attention
            </p>
            <h2 className="mt-2 max-w-4xl text-2xl font-black tracking-normal text-slate-950 md:text-3xl">
              SpecSA / SpecAttn: Draft Many Tokens, Read Fewer KV Blocks
            </h2>
            <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-700">
              Speculative decoding tries to verify multiple future tokens at once. Sparse attention tries to read only
              the useful parts of the KV cache. The hard part is preserving cross-query reuse when each verifier query
              wants its own sparse KV layout.
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

      <section className="grid min-w-0 gap-4 md:grid-cols-2 xl:grid-cols-4">
        {[
          ['Full attention', 'Read the entire library every time you write one word.'],
          ['Sparse attention', 'Read only the shelves that look relevant.'],
          ['SpecAttn', 'Watch which shelves the professor used during verification.'],
          ['SpecSA', 'Group similar requests and fetch shared shelves once.'],
        ].map(([title, body]) => (
          <article key={title} className="min-w-0 rounded-lg border border-slate-200 bg-white p-4">
            <h3 className="text-lg font-black text-slate-950">{title}</h3>
            <p className="mt-2 text-sm leading-6 text-slate-700">{body}</p>
          </article>
        ))}
      </section>

      <section className="grid min-w-0 gap-4 2xl:grid-cols-[340px_minmax(0,1fr)]">
        <aside className="min-w-0 rounded-lg border border-slate-200 bg-white p-4">
          <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <SlidersHorizontal size={16} />
            Controls
          </div>

          <div className="space-y-5">
            <div>
              <div className="mb-2 text-sm font-black text-slate-700">Mode</div>
              <div className="grid gap-2">
                {MODES.map((item) => (
                  <ControlButton key={item.id} active={mode === item.id} onClick={() => setMode(item.id)}>
                    <span className="block">{item.label}</span>
                    <span className="mt-1 block text-xs font-semibold opacity-80">{item.detail}</span>
                  </ControlButton>
                ))}
              </div>
            </div>

            <label className="grid gap-2 text-sm font-bold text-slate-700">
              Sparse ratio: {pct(sparseRatio)}
              <input
                type="range"
                min="1"
                max="20"
                step="1"
                value={Math.round(sparseRatio * 100)}
                onChange={(event) => setSparseRatio(Number(event.target.value) / 100)}
                className="accent-cyan-800"
              />
              <span className="text-xs font-semibold text-slate-500">
                Higher ratios improve sparse draft accuracy but load more KV blocks.
              </span>
            </label>

            <label className="grid gap-2 text-sm font-bold text-slate-700">
              Draft length: {draftLength}
              <input
                type="range"
                min="1"
                max="12"
                step="1"
                value={draftLength}
                onChange={(event) => setDraftLength(Number(event.target.value))}
                className="accent-cyan-800"
              />
            </label>

            <div>
              <div className="mb-2 text-sm font-black text-slate-700">Verification mode</div>
              <div className="grid gap-2">
                {VERIFICATION_MODES.map((item) => (
                  <ControlButton
                    key={item.id}
                    active={verificationMode === item.id}
                    onClick={() => setVerificationMode(item.id)}
                  >
                    {item.label}
                  </ControlButton>
                ))}
              </div>
            </div>

            <div>
              <div className="mb-2 text-sm font-black text-slate-700">Traversal</div>
              <div className="grid grid-cols-3 gap-2">
                {TRAVERSALS.map((item) => (
                  <ControlButton key={item.id} active={traversal === item.id} onClick={() => setTraversal(item.id)}>
                    {item.label}
                  </ControlButton>
                ))}
              </div>
            </div>

            <div>
              <div className="mb-2 text-sm font-black text-slate-700">Precision class</div>
              <div className="grid gap-2">
                {PRECISION_CLASSES.map((item) => (
                  <ControlButton
                    key={item.id}
                    active={precisionClass === item.id}
                    onClick={() => setPrecisionClass(item.id)}
                  >
                    {item.label}
                  </ControlButton>
                ))}
              </div>
            </div>

            <label className="grid gap-2 text-sm font-bold text-slate-700">
              Refresh every: {refreshEvery} layer{refreshEvery === 1 ? '' : 's'}
              <input
                type="range"
                min="1"
                max="4"
                step="1"
                value={refreshEvery}
                onChange={(event) => setRefreshEvery(Number(event.target.value))}
                className="accent-cyan-800"
              />
            </label>
          </div>
        </aside>

        <main className="min-w-0 space-y-4">
          <section className="grid min-w-0 gap-4 md:grid-cols-2 xl:grid-cols-4">
            <Metric
              icon={CheckCircle2}
              label="Accepted length"
              value={simulation.acceptedLength}
              detail="Accepted draft tokens in the current synthetic round."
            />
            <Metric
              icon={Zap}
              label="Draft latency"
              value={`${fmt(simulation.draftLatency)} ms`}
              detail="Toy cost for producing the draft tokens before verification."
            />
            <Metric
              icon={Gauge}
              label="Verification latency"
              value={`${fmt(simulation.verifyLatency)} ms`}
              detail="Toy verifier pass cost after sparse scheduling and layer reuse."
            />
            <Metric
              icon={Database}
              label="KV blocks loaded"
              value={simulation.kvBlocksLoaded}
              detail="Total block reads charged to this verifier schedule."
            />
            <Metric
              icon={Database}
              label="Unique KV blocks loaded"
              value={simulation.uniqueBlocksLoaded}
              detail="Distinct KV blocks touched after grouping or shared-index reuse."
            />
            <Metric
              icon={Cpu}
              label="Duplicate loads"
              value={simulation.duplicateLoadCount}
              detail="Repeated KV-block traffic that the grouped schedule tries to remove."
            />
            <Metric
              icon={Cpu}
              label="Selection overhead"
              value={`${fmt(simulation.selectionOverhead)} ms`}
              detail="Routing, merging, masking, or shared-index setup cost."
            />
            <Metric
              icon={Gauge}
              label="Accepted-token throughput"
              value={`${fmt(simulation.acceptedTokenThroughput, 0)}/s`}
              detail={`${fmt(simulation.roundLatency)} ms synthetic round latency.`}
            />
          </section>

          <section className="grid min-w-0 gap-4 xl:grid-cols-2">
            <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
              <SectionHeader icon={Zap} eyebrow="Panel 1" title="Token timeline">
                Draft tokens cheaply, verify them together, then keep the longest accepted prefix.
              </SectionHeader>
              <TokenTimeline mode={mode} draftLength={draftLength} acceptedLength={simulation.acceptedLength} />
            </div>

            <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
              <SectionHeader icon={Database} eyebrow="Panel 2" title="KV block selection">
                Dense rows are regular. Sparse rows are smaller, but query-specific layouts can hide shared work.
              </SectionHeader>
              <KVGrid simulation={simulation} showCriticality={mode === 'specAttn' || mode === 'combined'} />
            </div>
          </section>

          <section className="grid min-w-0 gap-4 xl:grid-cols-[0.85fr_1.15fr]">
            <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
              <SectionHeader icon={GitBranch} eyebrow="Panel 3" title="Verifier query layout">
                Nearby verifier rows often select overlapping KV blocks.
              </SectionHeader>
              <QueryLayout simulation={simulation} />
            </div>

            <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
              <SectionHeader icon={BarChart3} eyebrow="Panel 4" title="Throughput and acceptance">
                Speed comes from accepted tokens per verification round divided by end-to-end round latency.
              </SectionHeader>
              <ThroughputChart config={config} />
            </div>
          </section>
        </main>
      </section>

      <section className="grid min-w-0 gap-4 xl:grid-cols-[1.15fr_0.85fr]">
        <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
          <SectionHeader icon={Sparkles} eyebrow="SpecAttn" title="Verification-guided sparse drafting">
            Full-attention verification already computes attention evidence. SpecAttn turns that evidence into the
            next sparse drafting map.
          </SectionHeader>
          <CriticalityHeatmap criticality={simulation.criticality} />
        </div>

        <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
          <SectionHeader icon={FileText} eyebrow="Collect-2-Query" title="First draft plus bonus token">
            Collecting every verifier logit row costs memory and bandwidth. Boundary rows capture a wider draft-chain span.
          </SectionHeader>
          <div className="grid gap-3">
            {[
              ['1', 'first draft token', 'near the committed prefix'],
              ['2', 'bonus token', 'farther down the speculative chain'],
              ['avg', 'criticality score', 'mean over collected rows and heads'],
              ['top-k', 'selected KV entries', `${simulation.criticality.selected.length} blocks for the next draft`],
            ].map(([index, title, detail]) => (
              <div key={index} className="grid grid-cols-[56px_1fr] gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
                <span className="font-mono text-xs font-black text-cyan-700">{index}</span>
                <span>
                  <strong className="block text-sm text-slate-950">{title}</strong>
                  <em className="mt-1 block text-sm not-italic leading-6 text-slate-600">{detail}</em>
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="grid min-w-0 gap-4 xl:grid-cols-[1fr_0.9fr]">
        <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
          <SectionHeader icon={Layers3} eyebrow="SpecSA" title="Sparse verification as a grouped workload">
            Exact mode loads the union once and masks each query. Approx mode uses one shared index set for simpler execution.
          </SectionHeader>
          <div className="grid min-w-0 gap-4 lg:grid-cols-2">
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <h4 className="text-sm font-black uppercase tracking-wide text-slate-700">Exact merged schedule</h4>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                Union: [{simulation.unionBlocks.join(', ')}]
              </p>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                Load each unique block once. Preserve each row original selected-block mask.
              </p>
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <h4 className="text-sm font-black uppercase tracking-wide text-slate-700">Approx shared-index</h4>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                Representative: [{simulation.representativeBlocks.join(', ')}]
              </p>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                Reuse one sparse layout across the group when the reuse win is worth the approximation.
              </p>
            </div>
          </div>
        </div>

        <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
          <SectionHeader icon={Cpu} eyebrow="Layer schedule" title="Refresh and reuse">
            If sparse layouts are stable across nearby layers, refresh routing less often and enter a fused reuse path.
          </SectionHeader>
          <LayerStack refreshEvery={refreshEvery} />
        </div>
      </section>

      <section className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
        <SectionHeader icon={ClipboardList} eyebrow="Storyboard" title="Ten teaching panels">
          The page is structured around the conflict first, then the two paper mechanisms.
        </SectionHeader>
        <div className="grid gap-3 md:grid-cols-2">
          {STORYBOARD.map(([title, body], index) => (
            <div key={title} className="grid gap-2 rounded-lg border border-slate-200 bg-slate-50 p-3 sm:grid-cols-[42px_1fr]">
              <span className="font-mono text-xs font-black text-cyan-700">{index + 1}</span>
              <span>
                <strong className="block text-sm text-slate-950">{title}</strong>
                <em className="mt-1 block text-sm not-italic leading-6 text-slate-600">{body}</em>
              </span>
            </div>
          ))}
        </div>
      </section>

      <section className="grid min-w-0 gap-4 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
          <SectionHeader icon={Code2} eyebrow="Rustlings-style lab" title="mini-spec-sparse exercises">
            The standalone Rust crate keeps the core mechanisms small enough to implement by hand.
          </SectionHeader>
          <div className="grid gap-3">
            {EXERCISES.map(([file, body], index) => (
              <div key={file} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <strong className="font-mono text-sm text-slate-950">
                  {String(index + 1).padStart(2, '0')} {file}
                </strong>
                <p className="mt-1 text-sm leading-6 text-slate-600">{body}</p>
              </div>
            ))}
          </div>
          <pre className="mt-4 overflow-x-auto rounded-lg border border-slate-200 bg-slate-950 p-3 text-xs leading-5 text-slate-100">cd mini-spec-sparse
cargo test --bins</pre>
        </div>

        <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
          <SectionHeader icon={HelpCircle} eyebrow="Questions" title="Fast recall check" />
          <div className="grid gap-3">
            {QUESTIONS.map(([question, answer], index) => (
              <details key={question} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <summary className="cursor-pointer text-sm font-black text-slate-950">
                  {index + 1}. {question}
                </summary>
                <p className="mt-2 text-sm leading-6 text-slate-700">{answer}</p>
              </details>
            ))}
          </div>
        </div>
      </section>

      <section className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
        <SectionHeader icon={LinkIcon} eyebrow="Paper trail" title="Sources and scope">
          The simulator uses fixed synthetic arrays. It teaches mechanisms, not benchmark guarantees.
        </SectionHeader>
        <div className="grid gap-3 md:grid-cols-2">
          {SOURCE_LINKS.map((source) => (
            <SourceCard key={source.href} source={source} />
          ))}
        </div>
      </section>

      <AssessmentPanel lessonId="spec-sparse-attention" title="SpecSA / SpecAttn check" />
    </div>
  );
}
