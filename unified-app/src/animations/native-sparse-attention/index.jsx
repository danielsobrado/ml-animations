import React, { useMemo, useState } from 'react';
import {
  ArrowRight,
  BarChart3,
  BookOpen,
  Boxes,
  Code2,
  Cpu,
  Database,
  FileText,
  Flashlight,
  Gauge,
  HelpCircle,
  Layers3,
  Link as LinkIcon,
  Map,
  RotateCcw,
  SlidersHorizontal,
  SquareStack,
  Zap,
} from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const DISPLAY_BLOCKS = 48;

const MODES = [
  { id: 'full', label: 'Full Attention', detail: 'Every query reads every prior token.' },
  { id: 'sliding', label: 'Local Window', detail: 'Only the recent window is visible.' },
  { id: 'compression', label: 'Compression Only', detail: 'Read a coarse global map.' },
  { id: 'selection', label: 'Selection Only', detail: 'Zoom into important fine blocks.' },
  { id: 'nsa', label: 'NSA: All Branches', detail: 'Map, selected blocks, and local window.' },
];

const HARDWARE_VIEWS = [
  { id: 'tokens', label: 'token count' },
  { id: 'blocks', label: 'block loads' },
  { id: 'traffic', label: 'memory traffic' },
  { id: 'intensity', label: 'arithmetic intensity' },
];

const STORYBOARD = [
  ['The attention square', 'A long context creates an enormous causal attention matrix.'],
  ['Sparse is not automatically fast', 'Scattered single-token reads can waste GPU bandwidth.'],
  ['Compress the past', 'Blocks become summary tokens that form a cheap global map.'],
  ['Score the map', 'Compressed attention scores reveal which regions may matter.'],
  ['Select important blocks', 'Top blocks expand back into fine-grained token attention.'],
  ['Keep the local window', 'Recent context remains directly visible for fluency and syntax.'],
  ['Gate the branches', 'Global, selected, and local outputs are mixed by learned weights.'],
  ['Group by GQA heads', 'Heads sharing KV cache should reuse the same selected blocks.'],
  ['Hardware-aligned kernel', 'Contiguous blocks move from HBM to SRAM for block attention.'],
  ['Native training', 'The model learns under sparse attention instead of receiving it post-hoc.'],
];

const QUESTIONS = {
  'Warm-up': [
    ['Why is full attention expensive for long context?', 'Every query can compare against every previous key, creating quadratic prefill work and large KV-cache reads during decoding.'],
    ['Why is sparse attention not automatically fast?', 'Scattered token reads can produce inefficient memory access even when fewer attention scores are computed.'],
    ['What are NSA three branches?', 'Compression, selection, and sliding window.'],
    ['What does the compression branch provide?', 'Coarse global awareness of the whole context.'],
    ['What does the selection branch provide?', 'Fine-grained access to important blocks.'],
    ['What does the sliding window branch provide?', 'Recent local context.'],
  ],
  Concept: [
    ['Why does NSA use blocks instead of arbitrary tokens?', 'Blocks enable contiguous memory access and GPU-friendly attention kernels.'],
    ['Why use compressed scores to select blocks?', 'The compressed branch already performs a cheap global scan, so its scores can guide where detailed attention should zoom in.'],
    ['Why isolate the local window branch?', 'Local patterns are strong, so a separate branch keeps local fluency from short-circuiting long-range branch learning.'],
    ['What does natively trainable mean?', 'The sparse mechanism is present during training rather than imposed only at inference time.'],
    ['Why is GQA relevant?', 'Shared KV groups benefit when all query heads in the group reuse the same selected KV blocks.'],
  ],
  Challenge: [
    ['Why might inference-only sparse attention hurt quality?', 'A dense model learned to rely on full attention paths that post-hoc sparsity can remove.'],
    ['Why combine global summaries and selected fine tokens?', 'Summaries give broad awareness; selected fine tokens restore detailed evidence where needed.'],
    ['During decoding, which resource often matters most?', 'Memory bandwidth, because one new token may still load many cached K/V vectors.'],
    ['What is the selected-block tradeoff?', 'More selected blocks can improve recall, but they increase KV loads and latency.'],
  ],
};

const EXERCISES = [
  ['01_blockify.rs', 'Return half-open contiguous block ranges over a sequence.'],
  ['02_sliding_window.rs', 'Compute the visible recent token indices for a query.'],
  ['03_compress_blocks.rs', 'Mean-pool token vectors as a toy block compressor.'],
  ['04_block_scores.rs', 'Score compressed blocks with query dot products.'],
  ['05_topk_blocks.rs', 'Select the highest-scoring block ids.'],
  ['06_selected_tokens.rs', 'Expand selected blocks into fine token indices.'],
  ['07_gated_merge.rs', 'Weighted-sum the branch outputs.'],
  ['08_gqa_shared_selection.rs', 'Average block scores across heads in one GQA group.'],
  ['09_memory_access.rs', 'Estimate tokens read by full attention versus NSA.'],
  ['10_nsa_budget.rs', 'Pick the sparse plan with the best quality per loaded token.'],
];

const SOURCE_LINKS = [
  {
    label: 'Native Sparse Attention',
    href: 'https://arxiv.org/html/2502.11089v2',
    note: 'Hardware-aligned and natively trainable sparse attention with compression, selection, sliding window, learned gates, and blockwise kernels.',
  },
  {
    label: 'Flash Sparse Attention follow-up',
    href: 'https://arxiv.org/abs/2508.18224',
    note: 'An alternative NSA kernel discussion, useful as a caveat for smaller GQA group layouts and implementation-specific speedups.',
  },
];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function formatCount(value) {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return String(Math.round(value));
}

function pct(value) {
  return `${Math.round(value * 100)}%`;
}

function makeBlocks(sequenceLength, selectionBlockSize, compressionBlockSize, selectedBlockCount, slidingWindow) {
  const blockCount = Math.ceil(sequenceLength / selectionBlockSize);
  const queryPosition = sequenceLength - 1;
  const localStart = Math.max(0, queryPosition - slidingWindow + 1);
  const needle = Math.floor(blockCount * 0.38);
  const topicA = Math.floor(blockCount * 0.17);
  const topicB = Math.floor(blockCount * 0.73);

  const blocks = Array.from({ length: blockCount }, (_, id) => {
    const start = id * selectionBlockSize;
    const end = Math.min(sequenceLength, start + selectionBlockSize);
    const center = (start + end) / 2;
    const recency = clamp((center - localStart) / Math.max(1, sequenceLength - localStart), 0, 1);
    const needleScore = Math.exp(-Math.abs(id - needle) / 2.4) * 3.2;
    const topicScore = Math.exp(-Math.abs(id - topicA) / 3.6) * 1.9 + Math.exp(-Math.abs(id - topicB) / 4.2) * 2.2;
    const ripple = ((id * 17 + 11) % 29) / 29;
    return {
      id,
      start,
      end,
      compressedId: Math.floor(start / compressionBlockSize),
      compressedScore: 0.55 + recency * 1.7 + needleScore + topicScore + ripple,
      local: end > localStart,
    };
  });

  const selectedIds = new Set(
    [...blocks]
      .sort((a, b) => b.compressedScore - a.compressedScore || a.id - b.id)
      .slice(0, selectedBlockCount)
      .map((block) => block.id),
  );

  return blocks.map((block) => ({ ...block, selected: selectedIds.has(block.id) }));
}

function sampleBlocks(blocks) {
  if (blocks.length <= DISPLAY_BLOCKS) return blocks;
  return Array.from({ length: DISPLAY_BLOCKS }, (_, index) => {
    const start = Math.floor((index * blocks.length) / DISPLAY_BLOCKS);
    const end = Math.floor(((index + 1) * blocks.length) / DISPLAY_BLOCKS);
    const slice = blocks.slice(start, Math.max(start + 1, end));
    return {
      id: index,
      start: slice[0].start,
      end: slice[slice.length - 1].end,
      compressedScore: Math.max(...slice.map((block) => block.compressedScore)),
      compressed: slice.some((block) => block.id % 4 === 0),
      selected: slice.some((block) => block.selected),
      local: slice.some((block) => block.local),
    };
  });
}

function buildMetrics({
  mode,
  sequenceLength,
  compressionBlockSize,
  selectionBlockSize,
  selectedBlockCount,
  slidingWindow,
  gqaGroupSize,
  gateCompression,
  gateSelection,
  gateSliding,
}) {
  const compressionTokens = Math.ceil(sequenceLength / compressionBlockSize);
  const selectedFineTokens = selectedBlockCount * selectionBlockSize;
  const localTokens = Math.min(sequenceLength, slidingWindow);
  const nsaTokens = compressionTokens + selectedFineTokens + localTokens;
  const fullTokens = sequenceLength;
  const nsaBlockLoads = compressionTokens + selectedBlockCount + Math.ceil(localTokens / selectionBlockSize);
  const naiveHeadLoads = nsaBlockLoads * gqaGroupSize;
  const sharedLoads = nsaBlockLoads;
  const trafficRatio = mode === 'full' ? 1 : clamp(nsaTokens / fullTokens, 0, 1);
  const totalGate = gateCompression + gateSelection + gateSliding || 1;

  return {
    arithmeticIntensity: clamp(1 / Math.max(0.04, trafficRatio), 1, 32),
    compressionTokens,
    duplicateAvoided: Math.max(0, naiveHeadLoads - sharedLoads),
    fullTokens,
    localTokens,
    nsaBlockLoads,
    nsaTokens,
    selectedFineTokens,
    sharedLoads,
    trafficRatio,
    gates: {
      compression: gateCompression / totalGate,
      selection: gateSelection / totalGate,
      sliding: gateSliding / totalGate,
    },
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

function Slider({ label, value, min, max, step, onChange, formatter = (x) => x }) {
  return (
    <label className="grid gap-2 text-sm font-bold text-slate-700">
      <span className="flex items-center justify-between gap-3">
        <span>{label}</span>
        <span className="font-mono text-xs text-slate-900">{formatter(value)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="accent-cyan-800"
      />
    </label>
  );
}

function Metric({ icon: Icon, label, value, detail }) {
  return (
    <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-4">
      <div className="mb-2 flex min-w-0 flex-wrap items-center gap-2 text-xs font-black uppercase tracking-wide text-slate-500">
        <Icon size={16} />
        {label}
      </div>
      <strong className="block break-words text-2xl font-black text-slate-950">{value}</strong>
      <p className="mt-1 text-sm leading-6 text-slate-600">{detail}</p>
    </div>
  );
}

function MaskCell({ kind }) {
  const styles = {
    full: 'border-slate-500 bg-slate-800',
    compressed: 'border-blue-400 bg-blue-200',
    selected: 'border-amber-500 bg-amber-300',
    local: 'border-emerald-500 bg-emerald-300',
    both: 'border-teal-600 bg-gradient-to-br from-amber-300 to-emerald-300',
    skipped: 'border-slate-200 bg-white',
  };
  return <span className={`h-5 min-w-3 rounded-sm border ${styles[kind] || styles.skipped}`} />;
}

function AttentionMatrix({ blocks, mode }) {
  const rows = [
    ['compression', 'Compression branch'],
    ['selection', 'Selection branch'],
    ['sliding', 'Sliding branch'],
    ['combined', 'Combined NSA'],
  ];

  function kindFor(block, row) {
    if (mode === 'full') return 'full';
    if (row === 'compression') return mode === 'compression' || mode === 'nsa' ? 'compressed' : 'skipped';
    if (row === 'selection') return (mode === 'selection' || mode === 'nsa') && block.selected ? 'selected' : 'skipped';
    if (row === 'sliding') return (mode === 'sliding' || mode === 'nsa') && block.local ? 'local' : 'skipped';
    if (mode !== 'nsa') return 'skipped';
    if (block.selected && block.local) return 'both';
    if (block.selected) return 'selected';
    if (block.local) return 'local';
    return block.compressed ? 'compressed' : 'skipped';
  }

  return (
    <div className="overflow-x-auto">
      <div className="min-w-[760px]">
        <div className="mb-2 grid gap-1" style={{ gridTemplateColumns: `132px repeat(${blocks.length}, minmax(10px, 1fr))` }}>
          <span />
          {blocks.map((block, index) => (
            <span key={block.id} className="text-center font-mono text-[10px] font-black text-slate-400">
              {index % 6 === 0 ? formatCount(block.start) : ''}
            </span>
          ))}
        </div>
        <div className="grid gap-1">
          {rows.map(([rowId, label]) => (
            <div key={rowId} className="grid items-center gap-1" style={{ gridTemplateColumns: `132px repeat(${blocks.length}, minmax(10px, 1fr))` }}>
              <span className="truncate text-xs font-black text-slate-600">{label}</span>
              {blocks.map((block) => <MaskCell key={`${rowId}-${block.id}`} kind={kindFor(block, rowId)} />)}
            </div>
          ))}
        </div>
        <div className="mt-4 flex flex-wrap gap-3 text-xs font-bold text-slate-600">
          <span className="inline-flex items-center gap-2"><MaskCell kind="compressed" /> blue = compressed summaries</span>
          <span className="inline-flex items-center gap-2"><MaskCell kind="selected" /> gold = selected blocks</span>
          <span className="inline-flex items-center gap-2"><MaskCell kind="local" /> green = local window</span>
          <span className="inline-flex items-center gap-2"><MaskCell kind="skipped" /> white = skipped</span>
        </div>
      </div>
    </div>
  );
}

function BranchDiagram({ metrics }) {
  const gates = [
    ['Global map', metrics.gates.compression, 'blue'],
    ['Selected details', metrics.gates.selection, 'amber'],
    ['Local window', metrics.gates.sliding, 'emerald'],
  ];
  const colors = {
    blue: 'border-blue-300 bg-blue-50 text-blue-950',
    amber: 'border-amber-300 bg-amber-50 text-amber-950',
    emerald: 'border-emerald-300 bg-emerald-50 text-emerald-950',
  };

  return (
    <div className="grid gap-4">
      {[
        ['K,V cache', 'block compressor', 'compressed attention', 'global map'],
        ['K,V cache', 'block selector', 'selected fine attention', 'flashlight'],
        ['recent tokens', 'sliding window', 'local attention', 'recent notes'],
      ].map(([input, op, attn, output]) => (
        <div key={op} className="grid items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 p-3 sm:grid-cols-[1fr_24px_1fr_24px_1fr_24px_1fr]">
          <span className="rounded-md border border-slate-300 bg-white px-2 py-2 text-center font-mono text-xs font-black text-slate-800">{input}</span>
          <ArrowRight size={16} className="hidden text-slate-400 sm:block" />
          <span className="rounded-md border border-cyan-300 bg-cyan-50 px-2 py-2 text-center font-mono text-xs font-black text-cyan-900">{op}</span>
          <ArrowRight size={16} className="hidden text-slate-400 sm:block" />
          <span className="rounded-md border border-slate-300 bg-white px-2 py-2 text-center font-mono text-xs font-black text-slate-800">{attn}</span>
          <ArrowRight size={16} className="hidden text-slate-400 sm:block" />
          <span className="rounded-md border border-emerald-300 bg-emerald-50 px-2 py-2 text-center font-mono text-xs font-black text-emerald-900">{output}</span>
        </div>
      ))}

      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <div className="mb-3 text-xs font-black uppercase tracking-wide text-slate-500">Learned gate simulation</div>
        <div className="grid gap-3">
          {gates.map(([label, value, tone]) => (
            <div key={label} className="grid grid-cols-[120px_1fr_44px] items-center gap-3">
              <span className={`rounded-md border px-2 py-1 text-xs font-black ${colors[tone]}`}>{label}</span>
              <div className="h-3 rounded-full bg-slate-100">
                <div className="h-3 rounded-full bg-cyan-800" style={{ width: `${Math.max(4, value * 100)}%` }} />
              </div>
              <span className="font-mono text-xs font-black text-slate-600">{pct(value)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function BlockSelectionTimeline({ blocks, selectedBlockCount }) {
  const top = [...blocks].sort((a, b) => b.compressedScore - a.compressedScore || a.id - b.id).slice(0, 14);
  const maxScore = Math.max(...top.map((block) => block.compressedScore));
  return (
    <div className="grid gap-3">
      {top.map((block, index) => (
        <div key={block.id} className="grid grid-cols-[72px_1fr_68px] items-center gap-3">
          <span className={`font-mono text-xs font-black ${index < selectedBlockCount ? 'text-amber-700' : 'text-slate-500'}`}>
            B{block.id}
          </span>
          <div className="h-4 rounded-full bg-slate-100">
            <div className="h-4 rounded-full bg-amber-500" style={{ width: `${Math.max(6, (block.compressedScore / maxScore) * 100)}%` }} />
          </div>
          <span className="font-mono text-xs font-black text-slate-600">{block.compressedScore.toFixed(1)}</span>
        </div>
      ))}
      <p className="text-sm leading-6 text-slate-600">
        The top compressed scores become the fine-grained blocks for the selection branch.
      </p>
    </div>
  );
}

function HardwarePanel({ metrics, view }) {
  const rows = [
    ['Naive sparse', metrics.sharedLoads + metrics.duplicateAvoided, 'Random or head-specific block choices reload overlapping KV blocks.'],
    ['NSA blockwise', metrics.sharedLoads, 'Shared GQA-group block indices fetch contiguous KV chunks once.'],
    ['Full attention', Math.ceil(metrics.fullTokens / 64), 'Regular, dense, but it scales with the whole prefix.'],
  ];
  const max = Math.max(...rows.map((row) => row[1]));
  const headline = {
    tokens: `${formatCount(metrics.nsaTokens)} vs ${formatCount(metrics.fullTokens)} tokens`,
    blocks: `${formatCount(metrics.sharedLoads)} shared block loads`,
    traffic: `${pct(metrics.trafficRatio)} relative traffic`,
    intensity: `${metrics.arithmeticIntensity.toFixed(1)}x toy intensity`,
  }[view];

  return (
    <div className="grid gap-4">
      <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-4">
        <div className="text-xs font-black uppercase tracking-wide text-cyan-800">Hardware view</div>
        <strong className="mt-1 block text-2xl font-black text-cyan-950">{headline}</strong>
      </div>
      {rows.map(([label, value, detail]) => (
        <div key={label} className="grid grid-cols-[112px_1fr_72px] items-center gap-3">
          <span className="text-xs font-black uppercase tracking-wide text-slate-500">{label}</span>
          <div className="h-4 rounded-full bg-slate-100">
            <div className="h-4 rounded-full bg-cyan-800" style={{ width: `${Math.max(5, (value / max) * 100)}%` }} />
          </div>
          <span className="font-mono text-xs font-black text-slate-700">{formatCount(value)}</span>
          <p className="col-span-3 text-sm leading-6 text-slate-600">{detail}</p>
        </div>
      ))}
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
      <span className="flex items-center gap-2 font-black text-slate-950">
        <LinkIcon size={16} />
        {source.label}
      </span>
      <span className="mt-2 block text-sm leading-6 text-slate-600">{source.note}</span>
    </a>
  );
}

export default function NativeSparseAttention() {
  const [mode, setMode] = useState('nsa');
  const [sequenceLength, setSequenceLength] = useState(65536);
  const [compressionBlockSize, setCompressionBlockSize] = useState(64);
  const [selectionBlockSize, setSelectionBlockSize] = useState(64);
  const [selectedBlockCount, setSelectedBlockCount] = useState(32);
  const [slidingWindow, setSlidingWindow] = useState(512);
  const [gqaGroupSize, setGqaGroupSize] = useState(8);
  const [hardwareView, setHardwareView] = useState('traffic');
  const [gateMode, setGateMode] = useState('learned');

  const gateCompression = gateMode === 'equal' ? 1 : 0.32;
  const gateSelection = gateMode === 'equal' ? 1 : 0.43;
  const gateSliding = gateMode === 'equal' ? 1 : 0.25;

  const blocks = useMemo(
    () => makeBlocks(sequenceLength, selectionBlockSize, compressionBlockSize, selectedBlockCount, slidingWindow),
    [sequenceLength, selectionBlockSize, compressionBlockSize, selectedBlockCount, slidingWindow],
  );
  const displayBlocks = useMemo(() => sampleBlocks(blocks), [blocks]);
  const metrics = useMemo(
    () => buildMetrics({
      mode,
      sequenceLength,
      compressionBlockSize,
      selectionBlockSize,
      selectedBlockCount,
      slidingWindow,
      gqaGroupSize,
      gateCompression,
      gateSelection,
      gateSliding,
    }),
    [mode, sequenceLength, compressionBlockSize, selectionBlockSize, selectedBlockCount, slidingWindow, gqaGroupSize, gateCompression, gateSelection, gateSliding],
  );

  const reset = () => {
    setMode('nsa');
    setSequenceLength(65536);
    setCompressionBlockSize(64);
    setSelectionBlockSize(64);
    setSelectedBlockCount(32);
    setSlidingWindow(512);
    setGqaGroupSize(8);
    setHardwareView('traffic');
    setGateMode('learned');
  };

  return (
    <div className="min-w-0 space-y-6">
      <section className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-cyan-700">
              <BookOpen size={16} />
              Paper lesson - long-context attention
            </p>
            <h2 className="mt-2 max-w-5xl text-2xl font-black tracking-normal text-slate-950 md:text-3xl">
              Native Sparse Attention: Read the Map, Then Zoom In
            </h2>
            <p className="mt-3 max-w-5xl text-sm leading-6 text-slate-700">
              MLA asks whether each old token can store less. NSA asks whether each new query can read less:
              compressed global summaries, selected fine-grained blocks, and a recent sliding window.
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

      <section className="grid min-w-0 gap-4 md:grid-cols-3">
        {[
          [Map, 'Compression: map', 'Cheap global awareness over the whole past.'],
          [Flashlight, 'Selection: flashlight', 'Zoom into important original token blocks.'],
          [SquareStack, 'Sliding window: notes', 'Preserve recent local syntax and fluency.'],
        ].map(([Icon, title, body]) => (
          <article key={title} className="min-w-0 rounded-lg border border-slate-200 bg-white p-4">
            <div className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-cyan-700">
              <Icon size={18} />
              {title}
            </div>
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

            <Slider label="Sequence length" value={sequenceLength} min={4096} max={65536} step={4096} onChange={setSequenceLength} formatter={formatCount} />
            <Slider label="Compression block" value={compressionBlockSize} min={16} max={128} step={16} onChange={setCompressionBlockSize} />
            <Slider label="Selection block" value={selectionBlockSize} min={16} max={128} step={16} onChange={setSelectionBlockSize} />
            <Slider label="Selected blocks" value={selectedBlockCount} min={1} max={64} step={1} onChange={setSelectedBlockCount} />
            <Slider label="Sliding window" value={slidingWindow} min={128} max={2048} step={128} onChange={setSlidingWindow} />
            <Slider label="GQA group size" value={gqaGroupSize} min={1} max={16} step={1} onChange={setGqaGroupSize} />

            <div>
              <div className="mb-2 text-sm font-black text-slate-700">Gating</div>
              <div className="grid grid-cols-2 gap-2">
                <ControlButton active={gateMode === 'equal'} onClick={() => setGateMode('equal')}>equal</ControlButton>
                <ControlButton active={gateMode === 'learned'} onClick={() => setGateMode('learned')}>learned</ControlButton>
              </div>
            </div>

            <div>
              <div className="mb-2 text-sm font-black text-slate-700">Hardware view</div>
              <div className="grid grid-cols-2 gap-2">
                {HARDWARE_VIEWS.map((item) => (
                  <ControlButton key={item.id} active={hardwareView === item.id} onClick={() => setHardwareView(item.id)}>
                    {item.label}
                  </ControlButton>
                ))}
              </div>
            </div>
          </div>
        </aside>

        <main className="min-w-0 space-y-4">
          <section className="grid min-w-0 gap-4 md:grid-cols-2 xl:grid-cols-4">
            <Metric icon={Database} label="Full attention reads" value={formatCount(metrics.fullTokens)} detail="Per-query prefix tokens at the current sequence length." />
            <Metric icon={Boxes} label="NSA reads" value={formatCount(metrics.nsaTokens)} detail={`${formatCount(metrics.compressionTokens)} summaries + ${formatCount(metrics.selectedFineTokens)} selected + ${formatCount(metrics.localTokens)} local.`} />
            <Metric icon={Gauge} label="Relative traffic" value={pct(metrics.trafficRatio)} detail="Toy memory-access ratio versus dense full attention." />
            <Metric icon={Cpu} label="Shared block loads" value={formatCount(metrics.sharedLoads)} detail={`${formatCount(metrics.duplicateAvoided)} duplicate GQA-group loads avoided in the toy model.`} />
          </section>

          <section className="grid min-w-0 gap-4 xl:grid-cols-[1.05fr_0.95fr]">
            <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
              <SectionHeader icon={Layers3} eyebrow="Panel 1" title="Split attention matrix">
                NSA does not randomly drop context. It gives the model three ways to read the past: summarize, select, and stay local.
              </SectionHeader>
              <AttentionMatrix blocks={displayBlocks} mode={mode} />
            </div>

            <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
              <SectionHeader icon={Map} eyebrow="Panel 2" title="Three-branch NSA diagram">
                The branch outputs are mixed by learned gates so one sparse pattern does not have to solve every case.
              </SectionHeader>
              <BranchDiagram metrics={metrics} />
            </div>
          </section>

          <section className="grid min-w-0 gap-4 xl:grid-cols-2">
            <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
              <SectionHeader icon={Flashlight} eyebrow="Panel 3" title="Compressed scores choose fine blocks">
                First scan the map. Then zoom into the blocks that look important.
              </SectionHeader>
              <BlockSelectionTimeline blocks={blocks} selectedBlockCount={Math.min(selectedBlockCount, 14)} />
            </div>

            <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
              <SectionHeader icon={Cpu} eyebrow="Panel 4" title="Memory and compute view">
                Sparse is fast only when the remaining reads are useful, trainable, and hardware-aligned.
              </SectionHeader>
              <HardwarePanel metrics={metrics} view={hardwareView} />
            </div>
          </section>
        </main>
      </section>

      <section className="grid min-w-0 gap-4 xl:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
          <SectionHeader icon={Zap} eyebrow="Reported results" title="Paper numbers, carefully scoped">
            The reported speedups are not universal guarantees; they depend on architecture, kernel, hardware, context length, and GQA layout.
          </SectionHeader>
          <div className="grid gap-3 sm:grid-cols-3">
            {[
              ['9.0x', 'forward speedup at 64k in the paper Triton/A100 comparison'],
              ['6.0x', 'backward speedup at 64k in the same reported setup'],
              ['11.6x', 'expected decoding speedup from reduced memory-access volume'],
            ].map(([value, label]) => (
              <div key={value} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                <strong className="block text-2xl font-black text-slate-950">{value}</strong>
                <p className="mt-2 text-sm leading-6 text-slate-600">{label}</p>
              </div>
            ))}
          </div>
          <p className="mt-4 text-sm leading-6 text-slate-600">
            A later Flash Sparse Attention paper argues that vanilla NSA kernel behavior can depend strongly on GQA group size and proposes an alternative implementation for smaller groups.
          </p>
        </div>

        <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
          <SectionHeader icon={FileText} eyebrow="Native training" title="Sparse from the beginning">
            A model trained to read sparsely can learn different habits than a dense model suddenly forced to read less.
          </SectionHeader>
          <div className="grid gap-3">
            {[
              ['Post-hoc sparsity', 'dense training -> sparse inference -> mismatch risk'],
              ['Native sparsity', 'sparse training -> sparse inference -> learned sparse habits'],
              ['Hardware alignment', 'blockwise selection -> contiguous KV loads -> reuse across GQA heads'],
            ].map(([title, body]) => (
              <div key={title} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                <strong className="block text-sm text-slate-950">{title}</strong>
                <code className="mt-2 block rounded-md bg-slate-950 p-2 text-xs text-slate-100">{body}</code>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
        <SectionHeader icon={BarChart3} eyebrow="Storyboard" title="Ten teaching panels">
          The lesson moves from the dense attention bottleneck to the three branches, then to native training and GPU alignment.
        </SectionHeader>
        <div className="grid gap-3 md:grid-cols-2">
          {STORYBOARD.map(([title, body], index) => (
            <div key={title} className="grid gap-2 rounded-lg border border-slate-200 bg-slate-50 p-3 sm:grid-cols-[42px_1fr]">
              <span className="font-mono text-xs font-black text-cyan-700">{String(index + 1).padStart(2, '0')}</span>
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
          <SectionHeader icon={Code2} eyebrow="Rustlings-style lab" title="mini-native-sparse-attention exercises">
            The standalone Rust crate keeps blocks, windows, scores, gates, and memory estimates small enough to implement by hand.
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
          <pre className="mt-4 overflow-x-auto rounded-lg border border-slate-200 bg-slate-950 p-3 text-xs leading-5 text-slate-100">cd mini-native-sparse-attention{'\n'}cargo test --bins</pre>
        </div>

        <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
          <SectionHeader icon={HelpCircle} eyebrow="Questions" title="Fast recall check" />
          <div className="grid gap-4">
            {Object.entries(QUESTIONS).map(([group, questions]) => (
              <div key={group}>
                <h4 className="mb-2 text-sm font-black uppercase tracking-wide text-slate-600">{group}</h4>
                <div className="grid gap-2">
                  {questions.map(([question, answer], index) => (
                    <details key={question} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                      <summary className="cursor-pointer text-sm font-black text-slate-950">
                        {index + 1}. {question}
                      </summary>
                      <p className="mt-2 text-sm leading-6 text-slate-700">{answer}</p>
                    </details>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
        <SectionHeader icon={LinkIcon} eyebrow="Paper trail" title="Sources and scope">
          The simulator uses deterministic toy scores. It teaches the mechanism, not a benchmark guarantee.
        </SectionHeader>
        <div className="grid gap-3 md:grid-cols-2">
          {SOURCE_LINKS.map((source) => (
            <SourceCard key={source.href} source={source} />
          ))}
        </div>
      </section>

      <section className="rounded-lg border border-slate-200 bg-slate-950 p-5 text-white">
        <div className="grid gap-3 text-sm font-bold md:grid-cols-3">
          <span>Full attention reads everything.</span>
          <span>NSA reads summaries, selected details, and recent context.</span>
          <span>Blockwise sparse attention is a model-and-hardware design.</span>
        </div>
      </section>

      <AssessmentPanel lessonId="native-sparse-attention" title="Native Sparse Attention check" />
    </div>
  );
}
