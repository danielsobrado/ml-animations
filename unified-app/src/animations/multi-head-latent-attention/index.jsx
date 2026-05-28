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
  Gauge,
  GitBranch,
  HelpCircle,
  Layers3,
  Link as LinkIcon,
  RotateCcw,
  SlidersHorizontal,
  Sparkles,
  Workflow,
  Zap,
} from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';
import { ATTENTION_MODES, PAPER_ANCHORS } from './data';

const MODE_ORDER = ['mha', 'mqa', 'gqa', 'mla', 'transmla'];

const METRICS = [
  { id: 'cache', label: 'KV cache size' },
  { id: 'diversity', label: 'head diversity' },
  { id: 'compute', label: 'projection compute' },
  { id: 'traffic', label: 'memory traffic' },
];

const STORYBOARD = [
  ['Full KV cache', 'MHA gives every head its own key and value cache.'],
  ['Grouped Query Attention', 'GQA saves memory by making query heads share fewer KV heads.'],
  ['Latent Attention', 'MLA stores one compressed latent state for the token.'],
  ['Reconstruct head views', 'Learned up-projections recover expressive per-head behavior.'],
  ['Cache comparison', 'MLA reduces the cache by changing what is stored.'],
  ['Absorb the up-projection', 'Move the key expansion to the query side and attend to the latent cache.'],
  ['RoPE interrupts absorption', 'A position-dependent rotation blocks simple matrix regrouping.'],
  ['Decoupled RoPE', 'Keep compressed content separate from a small positional key cache.'],
  ['TransMLA conversion', 'Rewrite GQA repetition as parameter-side replication, then factorize it.'],
  ['Final comparison', 'MHA is expressive but memory-heavy; MLA is compact and expressive.'],
];

const QUESTIONS = {
  'Warm-up': [
    ['Why does the KV cache grow during decoding?', 'Every new token stores key and value information for future tokens to attend to.'],
    ['What does GQA do to reduce cache size?', 'It lets many query heads share fewer key/value heads.'],
    ['What does MLA cache instead of full K/V heads?', 'A compressed latent KV vector, plus a small decoupled RoPE key in the DeepSeek-style design.'],
    ['Why is MLA different from ordinary quantization?', 'Quantization stores fewer bits per number; MLA changes the architecture so fewer cached numbers exist.'],
  ],
  Concept: [
    ['Why can GQA reduce expressiveness?', 'Heads inside a group share the same K/V source, so their key/value views are less independent.'],
    ['How does MLA restore expressiveness?', 'The latent state is up-projected through learned matrices into richer head-specific behavior.'],
    ['What is the MLA tradeoff?', 'Lower cache memory and bandwidth, with extra projection work and implementation complexity.'],
    ['Why does RoPE cause trouble?', 'RoPE is position-sensitive, so it can sit between linear maps that would otherwise be regrouped.'],
    ['What does TransMLA prove?', 'At the same KV-cache overhead, MLA can represent GQA, but GQA cannot represent all MLA configurations.'],
  ],
  Challenge: [
    ['Why does memory bandwidth dominate long-context decoding?', 'Each new token repeatedly reads the growing cache, so moving cache data can dominate arithmetic.'],
    ['Why does MLA become more attractive as context grows?', 'The per-token cache saving is multiplied by context length and number of layers.'],
    ['Why does equal cache size not imply equal expressiveness?', 'Two designs can store the same number of values but use different transformations to create attention behavior.'],
    ['Why is TransMLA practical?', 'It suggests a path to convert existing GQA checkpoints into MLA-style models instead of training from scratch.'],
  ],
};

const EXERCISES = [
  ['01_kv_cache_size.rs', 'Compute MHA, GQA, and MLA per-token cache elements.'],
  ['02_mha_gqa_cache.rs', 'Convert full and compressed cache widths into a compression ratio.'],
  ['03_latent_cache.rs', 'Down-project a hidden vector into a cached latent state.'],
  ['04_down_up_projection.rs', 'Up-project the latent into a reconstructed head view.'],
  ['05_absorb_projection.rs', 'Verify q dot (W_up c) = (W_up^T q) dot c.'],
  ['06_rope_non_commute.rs', 'Show that rotation and anisotropic scaling do not commute.'],
  ['07_decoupled_rope_cache.rs', 'Count latent content plus positional RoPE cache elements.'],
  ['08_gqa_repetition.rs', 'Repeat shared KV heads across query groups.'],
  ['09_low_rank_factorization.rs', 'Count distinct repeated rows as a toy rank proxy.'],
  ['10_strategy_tradeoff.rs', 'Pick the compact expressive strategy under a simple score.'],
];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function formatCount(value) {
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(2)}B`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return String(Math.round(value));
}

function pct(value) {
  return `${Math.round(value * 100)}%`;
}

function Slider({ label, value, min, max, step = 1, onChange, help }) {
  return (
    <label className="grid gap-2 text-sm font-bold text-slate-700">
      <span className="flex items-center justify-between gap-3">
        <span>{label}</span>
        <span className="font-mono text-xs text-slate-900">{value}</span>
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
      {help ? <span className="text-xs font-semibold text-slate-500">{help}</span> : null}
    </label>
  );
}

function SectionHeader({ icon: Icon, eyebrow, title, children }) {
  return (
    <div className="mb-4">
      <p className="flex items-center gap-2 text-xs font-black uppercase text-cyan-700">
        <Icon size={16} />
        {eyebrow}
      </p>
      <h3 className="mt-2 text-xl font-black text-slate-950">{title}</h3>
      {children ? <p className="mt-2 text-sm leading-6 text-slate-700">{children}</p> : null}
    </div>
  );
}

function ControlButton({ active, children, icon: Icon, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`inline-flex min-h-10 items-center gap-2 rounded-lg border px-3 py-2 text-sm font-bold transition ${
        active
          ? 'border-cyan-800 bg-cyan-800 text-white'
          : 'border-slate-200 bg-white text-slate-700 hover:border-cyan-600'
      }`}
    >
      {Icon ? <Icon size={16} /> : null}
      <span>{children}</span>
    </button>
  );
}

function ToggleButton({ active, onClick, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`rounded-lg border px-3 py-2 text-sm font-bold transition ${
        active
          ? 'border-emerald-700 bg-emerald-700 text-white'
          : 'border-slate-200 bg-white text-slate-700 hover:border-emerald-500'
      }`}
    >
      {children}
    </button>
  );
}

function MetricCard({ icon: Icon, label, value, detail }) {
  return (
    <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-4">
      <div className="mb-2 flex items-center gap-2 text-xs font-black uppercase text-slate-500">
        <Icon size={16} />
        {label}
      </div>
      <strong className="block break-words text-2xl font-black text-slate-950">{value}</strong>
      <p className="mt-1 text-sm leading-6 text-slate-600">{detail}</p>
    </div>
  );
}

function BoxLabel({ children, tone = 'slate' }) {
  const tones = {
    slate: 'border-slate-300 bg-slate-50 text-slate-800',
    cyan: 'border-cyan-300 bg-cyan-50 text-cyan-900',
    emerald: 'border-emerald-300 bg-emerald-50 text-emerald-900',
    amber: 'border-amber-300 bg-amber-50 text-amber-900',
    purple: 'border-purple-300 bg-purple-50 text-purple-900',
    rose: 'border-rose-300 bg-rose-50 text-rose-900',
  };
  return (
    <span className={`inline-flex min-h-8 items-center justify-center rounded-md border px-2 py-1 text-center font-mono text-xs font-black ${tones[tone]}`}>
      {children}
    </span>
  );
}

function Arrow() {
  return <ArrowRight size={18} className="shrink-0 text-slate-400" />;
}

function ArchitectureDiagram({ mode, queryHeads, kvHeads }) {
  const visibleHeads = Math.min(queryHeads, 8);
  const visibleKv = Math.min(Math.max(1, kvHeads), 4);

  if (mode === 'transmla') {
    return (
      <div className="grid gap-4">
        {[
          ['1', 'GQA layer', 'hidden -> grouped W_K/W_V -> shared KV heads -> repeat'],
          ['2', 'Parameter-side repeat', 'move head repetition into the K/V projection matrices'],
          ['3', 'Low-rank factorization', 'repeated W_KV ~= W_down x W_up'],
          ['4', 'MLA-style cache', 'hidden -> cKV cache -> up-projected head views'],
        ].map(([step, title, body]) => (
          <div key={step} className="grid grid-cols-[44px_1fr] gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-md bg-cyan-800 font-mono text-xs font-black text-white">{step}</span>
            <span>
              <strong className="block text-sm text-slate-950">{title}</strong>
              <span className="mt-1 block font-mono text-xs leading-5 text-slate-600">{body}</span>
            </span>
          </div>
        ))}
      </div>
    );
  }

  if (mode === 'mla') {
    return (
      <div className="grid gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <BoxLabel tone="slate">h_t</BoxLabel>
          <Arrow />
          <BoxLabel tone="purple">W_DKV down</BoxLabel>
          <Arrow />
          <BoxLabel tone="purple">cKV_t cached</BoxLabel>
          <Arrow />
          <BoxLabel tone="cyan">W_UK / W_UV</BoxLabel>
          <Arrow />
          <BoxLabel tone="emerald">per-head views</BoxLabel>
        </div>
        <div className="grid gap-2 sm:grid-cols-4">
          {Array.from({ length: 4 }, (_, index) => (
            <div key={index} className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-center">
              <span className="font-mono text-xs font-black text-emerald-900">head {index + 1}</span>
              <span className="mt-1 block text-xs text-emerald-800">different view from shared latent</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (mode === 'gqa') {
    return (
      <div className="grid gap-4">
        <div className="flex flex-wrap gap-2">
          {Array.from({ length: visibleHeads }, (_, index) => (
            <BoxLabel key={index} tone="cyan">Q{index + 1}</BoxLabel>
          ))}
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          {Array.from({ length: visibleKv }, (_, index) => (
            <div key={index} className="rounded-lg border border-amber-200 bg-amber-50 p-3">
              <div className="mb-2 flex gap-2">
                <BoxLabel tone="emerald">K{index + 1}</BoxLabel>
                <BoxLabel tone="amber">V{index + 1}</BoxLabel>
              </div>
              <p className="text-xs leading-5 text-amber-900">shared by query group {index + 1}</p>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (mode === 'mqa') {
    return (
      <div className="grid gap-4">
        <div className="flex flex-wrap gap-2">
          {Array.from({ length: visibleHeads }, (_, index) => (
            <BoxLabel key={index} tone="cyan">Q{index + 1}</BoxLabel>
          ))}
        </div>
        <div className="flex items-center gap-3 rounded-lg border border-amber-200 bg-amber-50 p-4">
          <BoxLabel tone="emerald">Kshared</BoxLabel>
          <BoxLabel tone="amber">Vshared</BoxLabel>
          <p className="text-sm leading-6 text-amber-900">all query heads read one shared key/value source</p>
        </div>
      </div>
    );
  }

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap gap-2">
        {Array.from({ length: visibleHeads }, (_, index) => (
          <BoxLabel key={index} tone="cyan">Q{index + 1}</BoxLabel>
        ))}
      </div>
      <div className="grid gap-2 sm:grid-cols-4">
        {Array.from({ length: Math.min(visibleHeads, 4) }, (_, index) => (
          <div key={index} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <div className="flex gap-2">
              <BoxLabel tone="emerald">K{index + 1}</BoxLabel>
              <BoxLabel tone="amber">V{index + 1}</BoxLabel>
            </div>
            <span className="mt-2 block text-xs text-slate-600">own K/V cache</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function CacheBars({ sizes, activeMode }) {
  const rows = [
    ['MHA', 'mha', sizes.mha, 'Full K/V for every head'],
    ['MQA', 'mqa', sizes.mqa, 'One shared K/V head'],
    ['GQA', 'gqa', sizes.gqa, 'Fewer grouped K/V heads'],
    ['MLA', 'mla', sizes.mla, 'Latent cKV plus RoPE key'],
  ];
  const max = Math.max(...rows.map((row) => row[2]));

  return (
    <div className="grid gap-3">
      {rows.map(([label, id, value, detail]) => {
        const width = `${Math.max(6, (value / max) * 100)}%`;
        const active = activeMode === id || (activeMode === 'transmla' && id === 'mla');
        return (
          <div key={id} className={`rounded-lg border p-3 ${active ? 'border-cyan-700 bg-cyan-50' : 'border-slate-200 bg-slate-50'}`}>
            <div className="mb-2 flex items-center justify-between gap-3 text-sm">
              <strong className="text-slate-950">{label}</strong>
              <span className="font-mono text-xs text-slate-600">{formatCount(value)} elements/token</span>
            </div>
            <div className="h-4 rounded-full bg-white">
              <div className="h-4 rounded-full bg-cyan-800 transition-all" style={{ width }} />
            </div>
            <p className="mt-2 text-xs leading-5 text-slate-600">{detail}</p>
          </div>
        );
      })}
    </div>
  );
}

function LatentGeometry({ queryHeads, headDim, latentDim, ropeDim }) {
  const fullWidth = 2 * queryHeads * headDim;
  const latentWidth = latentDim + ropeDim;
  const latentPct = clamp(latentWidth / fullWidth, 0.03, 1);
  return (
    <div className="grid gap-5">
      <div>
        <div className="mb-2 flex justify-between text-xs font-bold text-slate-600">
          <span>full K/V width</span>
          <span className="font-mono">{formatCount(fullWidth)}</span>
        </div>
        <div className="h-8 rounded-lg border border-slate-200 bg-slate-100">
          <div className="h-full rounded-lg bg-slate-700" style={{ width: '100%' }} />
        </div>
      </div>
      <div>
        <div className="mb-2 flex justify-between text-xs font-bold text-slate-600">
          <span>latent content + RoPE key</span>
          <span className="font-mono">{formatCount(latentWidth)}</span>
        </div>
        <div className="h-8 rounded-lg border border-purple-200 bg-purple-50">
          <div className="h-full rounded-lg bg-purple-700" style={{ width: `${latentPct * 100}%` }} />
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-3 rounded-lg border border-slate-200 bg-slate-50 p-4">
        <BoxLabel tone="slate">h_t</BoxLabel>
        <Arrow />
        <BoxLabel tone="purple">cKV</BoxLabel>
        <Arrow />
        <div className="flex flex-wrap gap-2">
          {Array.from({ length: 4 }, (_, index) => (
            <BoxLabel key={index} tone="emerald">view {index + 1}</BoxLabel>
          ))}
        </div>
      </div>
    </div>
  );
}

function RopeAbsorption({ absorptionEnabled, decoupledRope }) {
  return (
    <div className="grid gap-4">
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
        <div className="mb-2 text-xs font-black uppercase text-slate-500">Absorption algebra</div>
        <div className="grid gap-2 font-mono text-sm text-slate-900">
          <span>q dot (W_up c) = (W_up^T q) dot c</span>
          <span className={absorptionEnabled ? 'text-emerald-700' : 'text-rose-700'}>
            {absorptionEnabled ? 'up-projection grouped with query/output path' : 'naive path materializes expanded K/V views'}
          </span>
        </div>
      </div>
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
        <div className="mb-2 text-xs font-black uppercase text-slate-500">RoPE path</div>
        {decoupledRope ? (
          <div className="grid gap-3">
            <div className="flex flex-wrap items-center gap-2">
              <BoxLabel tone="purple">content latent cKV</BoxLabel>
              <BoxLabel tone="cyan">small RoPE key</BoxLabel>
            </div>
            <p className="text-sm leading-6 text-slate-700">
              MLA compresses content memory while a separate positional key keeps RoPE compatible with efficient inference.
            </p>
          </div>
        ) : (
          <div className="grid gap-3">
            <div className="flex flex-wrap items-center gap-2">
              <BoxLabel tone="rose">content + position tangled</BoxLabel>
              <BoxLabel tone="amber">absorption blocked</BoxLabel>
            </div>
            <p className="text-sm leading-6 text-slate-700">
              A position-dependent rotation in the compressed key path prevents the clean linear regrouping.
            </p>
          </div>
        )}
      </div>
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
        {source.title}
      </span>
      <span className="mt-2 block text-sm leading-6 text-slate-600">{source.takeaway}</span>
      <span className="mt-2 block font-mono text-xs text-slate-500">{source.citation}</span>
    </a>
  );
}

export default function MultiHeadLatentAttention() {
  const [mode, setMode] = useState('mla');
  const [queryHeads, setQueryHeads] = useState(32);
  const [kvHeads, setKvHeads] = useState(8);
  const [latentDim, setLatentDim] = useState(512);
  const [headDim, setHeadDim] = useState(128);
  const [ropeDim, setRopeDim] = useState(64);
  const [contextLength, setContextLength] = useState(32768);
  const [layers, setLayers] = useState(32);
  const [absorptionEnabled, setAbsorptionEnabled] = useState(true);
  const [decoupledRope, setDecoupledRope] = useState(true);
  const [metric, setMetric] = useState('cache');

  const metrics = useMemo(() => {
    const boundedKvHeads = clamp(kvHeads, 1, queryHeads);
    const mha = 2 * queryHeads * headDim;
    const mqa = 2 * headDim;
    const gqa = 2 * boundedKvHeads * headDim;
    const mla = latentDim + (decoupledRope ? ropeDim : queryHeads * ropeDim);
    const active = mode === 'mha'
      ? mha
      : mode === 'mqa'
        ? mqa
        : mode === 'gqa'
          ? gqa
          : mla;
    const distinctViews = mode === 'mha'
      ? queryHeads
      : mode === 'mqa'
        ? 1
        : mode === 'gqa'
          ? boundedKvHeads
          : queryHeads;
    const diversityScore = mode === 'mha'
      ? 1
      : mode === 'mqa'
        ? 1 / queryHeads
        : mode === 'gqa'
          ? boundedKvHeads / queryHeads
          : clamp(0.58 + latentDim / (queryHeads * headDim), 0.58, 0.98);
    const projectionWork = mode === 'mla' || mode === 'transmla'
      ? queryHeads * headDim * latentDim * (absorptionEnabled ? 1 : 2)
      : queryHeads * headDim * Math.max(1, boundedKvHeads);
    const totalElements = active * contextLength * layers;
    const bandwidthPressure = active / mha;

    return {
      active,
      bandwidthPressure,
      distinctViews,
      diversityScore,
      projectionWork,
      relative: active / mha,
      sizes: { mha, mqa, gqa, mla },
      totalElements,
    };
  }, [absorptionEnabled, contextLength, decoupledRope, headDim, kvHeads, latentDim, layers, mode, queryHeads, ropeDim]);

  const reset = () => {
    setMode('mla');
    setQueryHeads(32);
    setKvHeads(8);
    setLatentDim(512);
    setHeadDim(128);
    setRopeDim(64);
    setContextLength(32768);
    setLayers(32);
    setAbsorptionEnabled(true);
    setDecoupledRope(true);
    setMetric('cache');
  };

  const activeMode = ATTENTION_MODES[mode] || ATTENTION_MODES.mla;

  return (
    <div className="min-w-0 space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="flex items-center gap-2 text-xs font-black uppercase text-cyan-700">
              <BookOpen size={16} />
              Paper lesson - KV cache architecture
            </p>
            <h2 className="mt-2 max-w-5xl text-2xl font-black text-slate-950 md:text-3xl">
              MLA: Compress the Cache, Reconstruct the Heads
            </h2>
            <p className="mt-3 max-w-5xl text-sm leading-6 text-slate-700">
              Multi-head attention stores full keys and values for every head. GQA saves memory by sharing K/V heads.
              MLA goes further: cache a compact latent KV state, then use learned projections to recover expressive
              per-head behavior when attention needs it.
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

      <section className="grid gap-4 md:grid-cols-3">
        {[
          ['MHA', 'Store one full notebook per attention head.'],
          ['GQA', 'Let several heads share the same notebook.'],
          ['MLA', 'Store compressed notes, then reconstruct each head view.'],
        ].map(([title, body]) => (
          <article key={title} className="rounded-lg border border-slate-200 bg-white p-4">
            <h3 className="text-lg font-black text-slate-950">{title}</h3>
            <p className="mt-2 text-sm leading-6 text-slate-700">{body}</p>
          </article>
        ))}
      </section>

      <section className="grid min-w-0 gap-4 2xl:grid-cols-[360px_minmax(0,1fr)]">
        <aside className="min-w-0 rounded-lg border border-slate-200 bg-white p-4">
          <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase text-slate-600">
            <SlidersHorizontal size={16} />
            Controls
          </div>

          <div className="space-y-5">
            <div>
              <div className="mb-2 text-sm font-black text-slate-700">Attention type</div>
              <div className="grid gap-2">
                {MODE_ORDER.map((id) => (
                  <ControlButton key={id} active={mode === id} onClick={() => setMode(id)}>
                    {ATTENTION_MODES[id]?.label || id.toUpperCase()}
                  </ControlButton>
                ))}
              </div>
            </div>

            <Slider label="Query heads" value={queryHeads} min={4} max={32} step={4} onChange={setQueryHeads} />
            <Slider label="KV heads / groups" value={kvHeads} min={1} max={16} step={1} onChange={setKvHeads} />
            <Slider label="Latent dimension" value={latentDim} min={64} max={512} step={64} onChange={setLatentDim} />
            <Slider label="Head dimension" value={headDim} min={64} max={128} step={64} onChange={setHeadDim} />
            <Slider label="RoPE dimension" value={ropeDim} min={16} max={128} step={16} onChange={setRopeDim} />
            <Slider label="Context length" value={contextLength} min={4096} max={131072} step={4096} onChange={setContextLength} />
            <Slider label="Layers" value={layers} min={8} max={80} step={4} onChange={setLayers} />

            <div className="grid gap-2">
              <div className="text-sm font-black text-slate-700">RoPE</div>
              <div className="grid grid-cols-2 gap-2">
                <ToggleButton active={!decoupledRope} onClick={() => setDecoupledRope(false)}>coupled</ToggleButton>
                <ToggleButton active={decoupledRope} onClick={() => setDecoupledRope(true)}>decoupled</ToggleButton>
              </div>
            </div>

            <div className="grid gap-2">
              <div className="text-sm font-black text-slate-700">Absorption</div>
              <div className="grid grid-cols-2 gap-2">
                <ToggleButton active={!absorptionEnabled} onClick={() => setAbsorptionEnabled(false)}>off</ToggleButton>
                <ToggleButton active={absorptionEnabled} onClick={() => setAbsorptionEnabled(true)}>on</ToggleButton>
              </div>
            </div>

            <div>
              <div className="mb-2 text-sm font-black text-slate-700">Metric focus</div>
              <div className="grid gap-2">
                {METRICS.map((item) => (
                  <ControlButton key={item.id} active={metric === item.id} onClick={() => setMetric(item.id)}>
                    {item.label}
                  </ControlButton>
                ))}
              </div>
            </div>
          </div>
        </aside>

        <main className="min-w-0 space-y-4">
          <section className="grid min-w-0 gap-4 md:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              icon={Database}
              label="KV cache per token"
              value={formatCount(metrics.active)}
              detail={`${activeMode.cacheType}. Relative to MHA: ${pct(metrics.relative)}.`}
            />
            <MetricCard
              icon={Sparkles}
              label="Distinct head views"
              value={metrics.distinctViews}
              detail={`Toy diversity score ${metrics.diversityScore.toFixed(2)} for the selected layout.`}
            />
            <MetricCard
              icon={Cpu}
              label="Projection compute"
              value={formatCount(metrics.projectionWork)}
              detail={mode === 'mla' || mode === 'transmla' ? 'MLA shifts work from cache reads to projections.' : 'Baseline head projection proxy.'}
            />
            <MetricCard
              icon={Gauge}
              label="Total cache elements"
              value={formatCount(metrics.totalElements)}
              detail={`${formatCount(contextLength)} tokens across ${layers} layers.`}
            />
          </section>

          <section className="grid gap-4 xl:grid-cols-2">
            <div className="rounded-lg border border-slate-200 bg-white p-5">
              <SectionHeader icon={Workflow} eyebrow="Panel 1" title="Attention architecture">
                {activeMode.description} {activeMode.memoryStrategy}
              </SectionHeader>
              <ArchitectureDiagram mode={mode} queryHeads={queryHeads} kvHeads={kvHeads} />
            </div>

            <div className="rounded-lg border border-slate-200 bg-white p-5">
              <SectionHeader icon={BarChart3} eyebrow="Panel 2" title="KV cache memory comparison">
                Cache width per old token. The `2` in MHA and GQA is for key plus value.
              </SectionHeader>
              <CacheBars sizes={metrics.sizes} activeMode={mode} />
            </div>
          </section>

          <section className="grid gap-4 xl:grid-cols-2">
            <div className="rounded-lg border border-slate-200 bg-white p-5">
              <SectionHeader icon={Boxes} eyebrow="Panel 3" title="Latent bottleneck geometry">
                Cache the compressed memory, not every expanded head.
              </SectionHeader>
              <LatentGeometry queryHeads={queryHeads} headDim={headDim} latentDim={latentDim} ropeDim={ropeDim} />
            </div>

            <div className="rounded-lg border border-slate-200 bg-white p-5">
              <SectionHeader icon={Zap} eyebrow="Panel 4" title="RoPE and absorption algebra">
                If linear maps are adjacent, regroup matrix multiplications. RoPE makes the positional path special.
              </SectionHeader>
              <RopeAbsorption absorptionEnabled={absorptionEnabled} decoupledRope={decoupledRope} />
            </div>
          </section>
        </main>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <SectionHeader icon={GitBranch} eyebrow="TransMLA" title="Why the conversion paper matters">
          TransMLA asks whether existing GQA checkpoints can be moved toward MLA without training an MLA model from scratch.
        </SectionHeader>
        <div className="grid gap-4 lg:grid-cols-[1fr_0.9fr]">
          <div className="grid gap-3">
            {[
              ['GQA repeats KV heads', 'K1 is copied across query heads 1-4; K2 is copied across query heads 5-8.'],
              ['Move repetition into parameters', 'The same repeated structure can be represented by expanded projection matrices.'],
              ['Factorize the repeated matrix', 'A low-rank W_down x W_up form creates an MLA-style cached latent.'],
              ['Same cache, more expressive family', 'MLA can represent GQA at the same cache overhead, but not every MLA is a GQA.'],
            ].map(([title, body], index) => (
              <div key={title} className="grid gap-2 rounded-lg border border-slate-200 bg-slate-50 p-3 sm:grid-cols-[44px_1fr]">
                <span className="flex h-9 w-9 items-center justify-center rounded-md bg-cyan-800 font-mono text-xs font-black text-white">
                  {index + 1}
                </span>
                <span>
                  <strong className="block text-sm text-slate-950">{title}</strong>
                  <span className="mt-1 block text-sm leading-6 text-slate-600">{body}</span>
                </span>
              </div>
            ))}
          </div>
          <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-5">
            <h4 className="text-lg font-black text-cyan-950">Teaching line</h4>
            <p className="mt-3 text-sm leading-6 text-cyan-900">
              GQA is like copying the same key to many heads. MLA stores the recipe that can produce richer
              head-specific keys.
            </p>
            <pre className="mt-4 overflow-x-auto rounded-lg bg-slate-950 p-3 text-xs leading-5 text-slate-100">{`GQA repeated heads
  -> parameter-side replication
  -> low-rank factorization
  -> MLA latent cache + up-projection`}</pre>
          </div>
        </div>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <SectionHeader icon={Layers3} eyebrow="Storyboard" title="Ten teaching panels">
          The lesson moves from cache width, to sharing, to factorization, to RoPE, then to TransMLA.
        </SectionHeader>
        <div className="grid gap-3 md:grid-cols-2">
          {STORYBOARD.map(([title, body], index) => (
            <div key={title} className="grid gap-2 rounded-lg border border-slate-200 bg-slate-50 p-3 sm:grid-cols-[42px_1fr]">
              <span className="font-mono text-xs font-black text-cyan-700">{String(index + 1).padStart(2, '0')}</span>
              <span>
                <strong className="block text-sm text-slate-950">{title}</strong>
                <span className="mt-1 block text-sm leading-6 text-slate-600">{body}</span>
              </span>
            </div>
          ))}
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <SectionHeader icon={Code2} eyebrow="Rustlings-style lab" title="mini-mla exercises">
            The standalone Rust crate keeps the cache, projection, RoPE, and TransMLA ideas small enough to implement by hand.
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
          <pre className="mt-4 overflow-x-auto rounded-lg border border-slate-200 bg-slate-950 p-3 text-xs leading-5 text-slate-100">cd mini-mla
cargo test --bins</pre>
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <SectionHeader icon={HelpCircle} eyebrow="Questions" title="Fast recall check" />
          <div className="grid gap-4">
            {Object.entries(QUESTIONS).map(([group, questions]) => (
              <div key={group}>
                <h4 className="mb-2 text-sm font-black uppercase text-slate-600">{group}</h4>
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

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <SectionHeader icon={FileText} eyebrow="Paper trail" title="Sources and scope">
          The simulator uses synthetic arithmetic for teaching. The paper links anchor the reported MLA and TransMLA claims.
        </SectionHeader>
        <div className="grid gap-3 md:grid-cols-2">
          {PAPER_ANCHORS.map((source) => (
            <SourceCard key={source.id} source={source} />
          ))}
        </div>
      </section>

      <section className="rounded-lg border border-slate-200 bg-slate-950 p-5 text-white">
        <div className="grid gap-3 text-sm font-bold md:grid-cols-3">
          <span>MHA stores full heads.</span>
          <span>GQA shares heads.</span>
          <span>MLA stores a latent and reconstructs expressive views.</span>
        </div>
      </section>

      <AssessmentPanel lessonId="multi-head-latent-attention" title="MLA / TransMLA check" />
    </div>
  );
}
