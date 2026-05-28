import React, { useMemo, useState } from 'react';
import {
  ArrowRight,
  BarChart3,
  BookOpen,
  CheckCircle2,
  ClipboardList,
  Code2,
  Cpu,
  FileText,
  Gauge,
  GitBranch,
  HelpCircle,
  Layers3,
  Link as LinkIcon,
  RotateCcw,
  SlidersHorizontal,
  Sparkles,
  XCircle,
  Zap,
} from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const MODES = [
  { id: 'vanilla', label: 'Vanilla', drift: 0, acceptBoost: 0 },
  { id: 'speculative', label: 'Speculative', drift: 0.08, acceptBoost: -0.04 },
  { id: 'eagle3', label: 'EAGLE-3', drift: 0.16, acceptBoost: 0.08 },
  { id: 'eagle31', label: 'EAGLE 3.1', drift: 0.035, acceptBoost: 0.24 },
];

const CONTEXTS = [
  { id: 'short', label: 'Short', drift: 0, detail: 'Prompt anchor remains close.' },
  { id: 'long', label: 'Long context', drift: 0.14, detail: 'More room for sink-token drift.' },
  { id: 'template', label: 'Changed chat template', drift: 0.1, detail: 'Format shift stresses the drafter.' },
  { id: 'ood', label: 'OOD system prompt', drift: 0.17, detail: 'Unfamiliar instructions shift hidden-state patterns.' },
];

const NORMALIZATIONS = [
  { id: 'off', label: 'Off', stability: 0, detail: 'Raw residual feedback keeps growing.' },
  { id: 'fc', label: 'FC norm only', stability: 0.12, detail: 'Target feature streams are balanced before fusion.' },
  { id: 'post', label: 'Post-norm only', stability: 0.16, detail: 'Recursive feedback stays closer to training scale.' },
  { id: 'full', label: 'FC norm + post-norm', stability: 0.31, detail: 'Fusion and feedback are stabilized.' },
];

const METRICS = [
  { id: 'acceptance', label: 'Acceptance length' },
  { id: 'attention', label: 'Attention mass' },
  { id: 'norm', label: 'Hidden-state RMS' },
];

const TOKENS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];

const OBJECTIVES = [
  'Explain why normal autoregressive decoding is sequential.',
  'Trace speculative decoding as draft, verify, accept, and fallback.',
  'Describe how EAGLE-style drafting uses target-model hidden states.',
  'Define acceptance length and connect it to speedup.',
  'Recognize attention drift in a speculation-depth heatmap.',
  'Explain why hidden-state scale matters in a recursive drafter.',
  'Name EAGLE 3.1\'s two normalization fixes and what each stabilizes.',
];

const LADDER = [
  {
    level: '1',
    title: 'The autocomplete assistant',
    body: 'A fast assistant guesses several next tokens, then the careful target model checks them all at once.',
    code: 'Assistant: the cat sat on\nTarget:    the cat sat near\nAccepted:  the cat sat\nFallback:  near',
  },
  {
    level: '2',
    title: 'EAGLE is not random drafting',
    body: 'EAGLE-3 gives the drafter fused low-, middle-, and high-layer target-model features.',
    code: 'low: syntax / local shape\nmid: phrase meaning\nhigh: next-token intent\n        -> fusion -> drafter',
  },
  {
    level: '3',
    title: 'Attention drift',
    body: 'As draft depth grows, attention can move from sink and prompt tokens toward the drafter\'s own recent tokens.',
    code: 'step 1: mostly prompt\nstep 2: prompt + draft-1\nstep 3: more draft history\nstep 4: mostly its own notes',
  },
  {
    level: '4',
    title: 'Normalization keeps the loop familiar',
    body: 'FC norm balances target streams before fusion. Post-norm feedback prevents hidden-state scale from accumulating.',
    code: 'target h -> RMSNorm -> FC fusion\ndraft h  -> PostNorm -> next step',
  },
];

const STORYBOARD = [
  ['The decoding treadmill', 'Vanilla LLM decoding emits one token per large-model pass.'],
  ['Draft then verify', 'Speculative decoding proposes multiple tokens and verifies them in parallel.'],
  ['EAGLE-3 feature fusion', 'The drafter consumes low-, middle-, and high-layer target hidden states.'],
  ['Attention drift heatmap', 'Deeper draft rows shift mass from prompt anchors to recent draft tokens.'],
  ['Hidden-state norm grows', 'Unnormalized residual feedback changes scale across speculative depth.'],
  ['EAGLE-3 vs EAGLE 3.1', 'FC normalization and post-norm feedback stabilize the recursive drafter.'],
  ['Acceptance length payoff', 'More stable drafts keep longer accepted prefixes per target verification round.'],
];

const EXERCISES = [
  ['01_accept_prefix.rs', 'Accept matching draft tokens until the first mismatch.'],
  ['02_speculative_round.rs', 'Append accepted tokens and the target replacement after rejection.'],
  ['03_attention_drift.rs', 'Compare attention mass on sink/context positions and recent draft tokens.'],
  ['04_rms_norm.rs', 'Normalize a vector by root-mean-square scale.'],
  ['05_feature_fusion.rs', 'RMS-normalize each hidden stream before concatenating features.'],
  ['06_post_norm_feedback.rs', 'Apply post-norm after every recursive draft step.'],
  ['07_acceptance_length_sim.rs', 'Compute expected accepted length from conditional acceptance probabilities.'],
];

const MENTAL_MODELS = [
  ['Target model', 'Careful professor'],
  ['Drafter', 'Fast teaching assistant'],
  ['Verification', 'Professor checks several guesses at once'],
  ['Acceptance length', 'How many assistant guesses survive'],
  ['Attention sink', 'Anchor point the model keeps looking back to'],
  ['Attention drift', 'Assistant starts reading its own notes'],
  ['Hidden-state growth', 'Notes get louder each time they are copied'],
  ['FC norm', 'Equalize the microphones before mixing'],
  ['Post-norm feedback', 'Reset the volume before the next recursive call'],
];

const FINAL_QUIZ = [
  ['What makes speculative decoding fast?', 'It verifies multiple drafted tokens in one target-model pass.'],
  ['What determines most of the speedup?', 'Average accepted prefix length.'],
  ['What is attention drift?', 'The drafter\'s attention moves from prompt/sink/context tokens toward its own recent draft tokens.'],
  ['Why can hidden-state scale cause problems?', 'Each recursive draft step may receive a representation with a different magnitude distribution.'],
  ['What are EAGLE 3.1\'s two architectural fixes?', 'FC normalization before fusion and post-norm hidden-state feedback.'],
  ['Does EAGLE 3.1 remove target-model verification?', 'No. The target model still verifies draft tokens.'],
];

const SOURCE_LINKS = [
  {
    label: 'MarkTechPost overview',
    href: 'https://www.marktechpost.com/2026/05/27/meet-eagle-3-1-the-speculative-decoding-algorithm-that-fixes-attention-drift-in-llm-inference/',
    note: 'Speculative decoding framing: draft cheaply, verify with the target model, fall back on rejection.',
  },
  {
    label: 'vLLM EAGLE 3.1 post',
    href: 'https://vllm.ai/blog/2026-05-26-eagle-3-1',
    note: 'FC normalization, post-norm feedback, robustness, and benchmark scenario.',
  },
  {
    label: 'EAGLE-3 paper',
    href: 'https://arxiv.org/html/2503.01840v1',
    note: 'Direct token prediction and multi-layer feature fusion.',
  },
  {
    label: 'Attention Drift paper',
    href: 'https://arxiv.org/html/2605.09992v1',
    note: 'Attention shifts from prompt and sink tokens toward drafted tokens.',
  },
];

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function pct(value) {
  return `${Math.round(value * 100)}%`;
}

function ControlButton({ active, children, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-lg border px-3 py-2 text-left text-sm font-bold transition ${
        active
          ? 'border-cyan-800 bg-cyan-800 text-white'
          : 'border-slate-200 bg-white text-slate-700 hover:border-cyan-500'
      }`}
    >
      {children}
    </button>
  );
}

function Metric({ label, value, detail, icon: Icon }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="mb-2 flex items-center gap-2 text-xs font-black uppercase tracking-wide text-slate-500">
        <Icon size={16} />
        {label}
      </div>
      <strong className="block text-2xl font-black text-slate-950">{value}</strong>
      <p className="mt-1 text-sm leading-6 text-slate-600">{detail}</p>
    </div>
  );
}

function TokenPill({ token, accepted, rejected }) {
  return (
    <span
      className={`inline-flex h-10 min-w-10 items-center justify-center rounded-lg border px-3 font-mono text-sm font-black ${
        accepted
          ? 'border-emerald-600 bg-emerald-600 text-white'
          : rejected
            ? 'border-rose-300 bg-rose-50 text-rose-700'
            : 'border-slate-200 bg-slate-50 text-slate-700'
      }`}
    >
      {token}
    </span>
  );
}

function VanillaTimeline({ generatedTokens }) {
  const rows = Array.from({ length: generatedTokens }, (_, index) => {
    const past = index === 0
      ? 'Prompt'
      : `Prompt + ${Array.from({ length: index }, (__, tokenIndex) => `token ${tokenIndex + 1}`).join(' + ')}`;
    return [past, 'Target LLM', `token ${index + 1}`];
  });

  return (
    <div className="grid gap-3">
      {rows.map((row) => (
        <div key={row.join('-')} className="grid gap-2 rounded-lg border border-slate-200 bg-slate-50 p-3 md:grid-cols-[1fr_auto_1fr_auto_1fr] md:items-center">
          <span className="font-mono text-sm font-bold text-slate-700">{row[0]}</span>
          <ArrowRight className="hidden h-4 w-4 text-slate-400 md:block" />
          <span className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-center text-sm font-black text-slate-900">{row[1]}</span>
          <ArrowRight className="hidden h-4 w-4 text-slate-400 md:block" />
          <TokenPill token={row[2]} accepted />
        </div>
      ))}
      <p className="text-sm leading-6 text-slate-700">
        KV cache avoids recomputing the past, but the future is still generated one token at a time.
      </p>
    </div>
  );
}

function DraftVerifyPanel({ mode, depth, acceptedCount }) {
  if (mode === 'vanilla') return <VanillaTimeline generatedTokens={Math.min(depth, 4)} />;

  const visibleTokens = TOKENS.slice(0, depth);
  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
        <div className="mb-3 text-xs font-black uppercase tracking-wide text-slate-500">Fast drafter proposes</div>
        <div className="flex flex-wrap gap-2">
          {visibleTokens.map((token) => <TokenPill key={token} token={token} />)}
        </div>
      </div>
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
        <div className="mb-3 text-xs font-black uppercase tracking-wide text-slate-500">Target model verifies in one pass</div>
        <div className="flex flex-wrap gap-2">
          {visibleTokens.map((token, index) => (
            <TokenPill key={token} token={token} accepted={index < acceptedCount} rejected={index >= acceptedCount} />
          ))}
          {acceptedCount < depth ? <TokenPill token={`${visibleTokens[acceptedCount]}'`} accepted /> : null}
        </div>
      </div>
      <p className="text-sm leading-6 text-slate-700">
        Accepted prefix length is {acceptedCount}. A rejected token discards the remaining draft path and the target
        supplies the replacement, so draft mistakes do not become unchecked output.
      </p>
    </div>
  );
}

function FeatureFusionPanel() {
  const streams = [
    ['low h', 'syntax / local token shape'],
    ['mid h', 'phrase structure'],
    ['high h', 'next-token intent'],
  ];

  return (
    <div className="grid gap-3 lg:grid-cols-[1fr_auto_1fr] lg:items-center">
      <div className="grid gap-2">
        {streams.map(([label, detail]) => (
          <div key={label} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <strong className="font-mono text-sm text-slate-950">{label}</strong>
            <span className="ml-3 text-sm text-slate-600">{detail}</span>
          </div>
        ))}
      </div>
      <ArrowRight className="hidden h-5 w-5 text-slate-400 lg:block" />
      <div className="rounded-lg border border-cyan-700 bg-cyan-50 p-4 text-center">
        <strong className="block text-slate-950">{'concat -> FC fusion -> EAGLE drafter'}</strong>
        <span className="mt-2 block text-sm leading-6 text-slate-700">
          The drafter sees richer target-model evidence than a separate small model would.
        </span>
      </div>
    </div>
  );
}

function AttentionHeatmap({ depth, drift }) {
  const columns = ['sink', 'prompt', 'user ctx', 'accepted', 'draft 1', 'draft 2', 'draft 3', 'draft 4'];
  const rows = Array.from({ length: depth }, (_, index) => index + 1);

  return (
    <div className="overflow-x-auto">
      <div className="min-w-[640px]">
        <div className="mb-2 grid grid-cols-[76px_repeat(8,minmax(54px,1fr))] gap-1 text-center text-[11px] font-bold text-slate-500">
          <span />
          {columns.map((column) => <span key={column}>{column}</span>)}
        </div>
        <div className="grid gap-1">
          {rows.map((row) => {
            const rowDrift = clamp(drift + (row - 1) * 0.075);
            return (
              <div key={row} className="grid grid-cols-[76px_repeat(8,minmax(54px,1fr))] gap-1">
                <span className="flex items-center text-xs font-black text-slate-500">depth {row}</span>
                {columns.map((column, index) => {
                  const isDraft = index >= 4;
                  const isAnchor = index <= 2;
                  const base = isDraft ? rowDrift : isAnchor ? 1 - rowDrift : 0.58 - rowDrift * 0.2;
                  const taper = isDraft ? Math.max(0.2, 1 - (index - 4) * 0.14) : Math.max(0.42, 1 - index * 0.16);
                  const intensity = clamp(base * taper);
                  return (
                    <span
                      key={`${row}-${column}`}
                      className="h-9 rounded-md border border-slate-200"
                      style={{
                        backgroundColor: isDraft
                          ? `rgba(225, 29, 72, ${0.12 + intensity * 0.76})`
                          : `rgba(8, 145, 178, ${0.12 + intensity * 0.76})`,
                      }}
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

function MassChart({ promptMass, draftMass }) {
  const rows = [
    ['sink + prompt mass', promptMass, 'bg-cyan-700'],
    ['recent draft mass', draftMass, 'bg-rose-600'],
  ];
  return (
    <div className="mt-4 grid gap-3">
      {rows.map(([label, value, color]) => (
        <div key={label} className="grid grid-cols-[150px_1fr_52px] items-center gap-3">
          <span className="text-xs font-black uppercase tracking-wide text-slate-500">{label}</span>
          <div className="h-3 rounded-full bg-slate-100">
            <div className={`h-3 rounded-full ${color}`} style={{ width: pct(value) }} />
          </div>
          <span className="font-mono text-xs font-black text-slate-700">{pct(value)}</span>
        </div>
      ))}
    </div>
  );
}

function NormChart({ depth, normalization, drift }) {
  const postNorm = normalization === 'post' || normalization === 'full';
  const fcNorm = normalization === 'fc' || normalization === 'full';
  const growth = postNorm ? 0.03 : fcNorm ? 0.22 : 0.45;
  const values = Array.from({ length: depth + 1 }, (_, index) => 0.78 + index * growth + (postNorm ? 0 : drift * 0.28));
  const max = Math.max(...values);

  return (
    <div className="space-y-3">
      {values.map((value, index) => (
        <div key={index} className="grid grid-cols-[72px_1fr_56px] items-center gap-3">
          <span className="font-mono text-xs font-bold text-slate-500">d{index}</span>
          <div className="h-4 rounded-full bg-slate-100">
            <div className="h-4 rounded-full bg-cyan-700" style={{ width: `${Math.max(9, (value / max) * 100)}%` }} />
          </div>
          <span className="font-mono text-xs font-bold text-slate-700">{value.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

function ArchitecturePanel() {
  const box = 'rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm font-bold text-slate-800';
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <h3 className="text-lg font-black text-slate-950">EAGLE-3: attention drift</h3>
        <div className="mt-4 grid gap-2">
          <div className={box}>low h / mid h / high h</div>
          <ArrowRight className="mx-auto h-4 w-4 text-slate-400" />
          <div className={box}>FC fusion</div>
          <ArrowRight className="mx-auto h-4 w-4 text-slate-400" />
          <div className={box}>{'drafter -> raw hidden -> next step'}</div>
        </div>
        <p className="mt-4 text-sm leading-6 text-slate-700">
          Larger hidden-state streams can dominate fusion, and raw residual feedback grows across the speculative chain.
        </p>
      </div>
      <div className="rounded-lg border border-cyan-700 bg-cyan-50 p-4">
        <h3 className="text-lg font-black text-slate-950">EAGLE 3.1: normalized recursive drafting</h3>
        <div className="mt-4 grid gap-2">
          <div className={box}>{'low h -> RMSNorm / mid h -> RMSNorm / high h -> RMSNorm'}</div>
          <ArrowRight className="mx-auto h-4 w-4 text-slate-500" />
          <div className={box}>FC fusion</div>
          <ArrowRight className="mx-auto h-4 w-4 text-slate-500" />
          <div className={box}>{'drafter -> PostNorm -> next step'}</div>
        </div>
        <p className="mt-4 text-sm leading-6 text-slate-700">
          The drafter behaves more like the same small autoregressive model being called repeatedly.
        </p>
      </div>
    </div>
  );
}

function AcceptanceChart({ depth, context }) {
  const contextPenalty = context === 'short' ? 0 : context === 'long' ? 1 : 2;
  const rows = [
    { label: 'EAGLE-3', values: [4, 3, 2, 1].map((v) => Math.max(1, Math.min(depth, v - contextPenalty))), color: 'bg-rose-500' },
    { label: 'EAGLE 3.1', values: [4, 4, 3, 3].map((v) => Math.max(1, Math.min(depth, v))), color: 'bg-emerald-600' },
  ];
  return (
    <div className="space-y-4">
      {rows.map((row) => (
        <div key={row.label}>
          <div className="mb-2 flex justify-between text-sm font-black text-slate-700">
            <span>{row.label}</span>
            <span>{row.values.join(' -> ')}</span>
          </div>
          <div className="grid grid-cols-4 gap-2">
            {row.values.map((value, index) => (
              <div key={index} className="h-24 rounded-lg border border-slate-200 bg-slate-50 p-2">
                <div className="flex h-full items-end">
                  <div className={`w-full rounded ${row.color}`} style={{ height: `${(value / Math.max(1, depth)) * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function SectionHeader({ icon: Icon, eyebrow, title, children }) {
  return (
    <div className="mb-4">
      <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-cyan-700">
        <Icon size={16} />
        {eyebrow}
      </p>
      <h3 className="mt-2 text-xl font-black text-slate-950">{title}</h3>
      {children ? <p className="mt-2 text-sm leading-6 text-slate-700">{children}</p> : null}
    </div>
  );
}

export default function Eagle31SpeculativeDecoding() {
  const [mode, setMode] = useState('eagle31');
  const [depth, setDepth] = useState(4);
  const [context, setContext] = useState('long');
  const [normalization, setNormalization] = useState('full');
  const [metric, setMetric] = useState('acceptance');

  const state = useMemo(() => {
    const modeConfig = MODES.find((item) => item.id === mode);
    const contextConfig = CONTEXTS.find((item) => item.id === context);
    const normConfig = NORMALIZATIONS.find((item) => item.id === normalization);
    const isVanilla = mode === 'vanilla';
    const activeDepth = isVanilla ? Math.min(depth, 4) : depth;
    const drift = isVanilla
      ? 0
      : clamp(0.1 + modeConfig.drift * activeDepth + contextConfig.drift - normConfig.stability);
    const promptMass = clamp(0.82 - drift * 0.52, 0.18, 0.9);
    const draftMass = clamp(0.12 + drift * 0.68, 0.04, 0.84);
    const acceptedCount = isVanilla
      ? 1
      : Math.max(1, Math.min(activeDepth, Math.round(activeDepth * clamp(0.82 - drift * 0.48 + modeConfig.acceptBoost, 0.16, 1))));
    const speedup = isVanilla ? 1 : 1 + acceptedCount / Math.max(2, activeDepth + 1);

    return {
      activeDepth,
      acceptedCount,
      contextConfig,
      drift,
      draftMass,
      modeConfig,
      normConfig,
      promptMass,
      speedup,
    };
  }, [context, depth, mode, normalization]);

  const selectMode = (nextMode) => {
    setMode(nextMode);
    if (nextMode === 'eagle31') setNormalization('full');
    if (nextMode === 'eagle3') setNormalization('off');
    if (nextMode === 'vanilla' || nextMode === 'speculative') setNormalization('off');
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-cyan-700">
              <BookOpen size={16} />
              Full lesson pack - Papers / Transformer inference
            </p>
            <h2 className="mt-2 text-2xl font-black text-slate-950 md:text-3xl">
              EAGLE 3.1: When the Drafter Starts Trusting Itself
            </h2>
            <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-700">
              Large language models usually generate one token at a time. Speculative decoding asks a fast drafter to
              guess several future tokens, then asks the full target model to verify them in parallel. EAGLE 3.1 fixes
              a deeper failure mode: the drafter can stop grounding itself in the original context and start trusting
              its own speculative hidden states too much.
            </p>
          </div>
          <button
            type="button"
            onClick={() => {
              setMode('eagle31');
              setDepth(4);
              setContext('long');
              setNormalization('full');
              setMetric('acceptance');
            }}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-bold text-slate-800"
          >
            <RotateCcw size={16} />
            Reset
          </button>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {LADDER.map((item) => (
          <article key={item.level} className="rounded-lg border border-slate-200 bg-white p-4">
            <span className="font-mono text-xs font-black text-cyan-700">Level {item.level}</span>
            <h3 className="mt-2 text-lg font-black text-slate-950">{item.title}</h3>
            <p className="mt-2 text-sm leading-6 text-slate-700">{item.body}</p>
            <pre className="mt-3 overflow-x-auto rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs leading-5 text-slate-700">{item.code}</pre>
          </article>
        ))}
      </section>

      <section className="grid gap-4 2xl:grid-cols-[340px_minmax(0,1fr)]">
        <aside className="rounded-lg border border-slate-200 bg-white p-4">
          <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <SlidersHorizontal size={16} />
            Controls
          </div>

          <div className="space-y-5">
            <div>
              <div className="mb-2 text-sm font-black text-slate-700">Mode</div>
              <div className="grid gap-2">
                {MODES.map((item) => (
                  <ControlButton key={item.id} active={mode === item.id} onClick={() => selectMode(item.id)}>
                    {item.label}
                  </ControlButton>
                ))}
              </div>
            </div>

            <label className="grid gap-2 text-sm font-bold text-slate-700">
              Speculation depth: {state.activeDepth}
              <input
                type="range"
                min="1"
                max="8"
                value={depth}
                onChange={(event) => setDepth(Number(event.target.value))}
                className="accent-cyan-800"
              />
            </label>

            <div>
              <div className="mb-2 text-sm font-black text-slate-700">Prompt type</div>
              <div className="grid gap-2">
                {CONTEXTS.map((item) => (
                  <ControlButton key={item.id} active={context === item.id} onClick={() => setContext(item.id)}>
                    <span className="block">{item.label}</span>
                    <span className="mt-1 block text-xs font-semibold opacity-80">{item.detail}</span>
                  </ControlButton>
                ))}
              </div>
            </div>

            <div>
              <div className="mb-2 text-sm font-black text-slate-700">Normalization</div>
              <div className="grid gap-2">
                {NORMALIZATIONS.map((item) => (
                  <ControlButton key={item.id} active={normalization === item.id} onClick={() => setNormalization(item.id)}>
                    <span className="block">{item.label}</span>
                    <span className="mt-1 block text-xs font-semibold opacity-80">{item.detail}</span>
                  </ControlButton>
                ))}
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

        <main className="space-y-4">
          <div className="grid gap-4 md:grid-cols-4">
            <Metric icon={Zap} label="Draft depth" value={state.activeDepth} detail="How many future tokens are proposed before target verification." />
            <Metric icon={CheckCircle2} label="Accepted tokens" value={state.acceptedCount} detail="Accepted prefix before fallback or the next round." />
            <Metric icon={Cpu} label="Prompt attention" value={pct(state.promptMass)} detail="Estimated mass on sink and prompt context." />
            <Metric icon={Gauge} label="Draft attention" value={pct(state.draftMass)} detail="Estimated mass on speculative self-history." />
          </div>

          <section className="rounded-lg border border-slate-200 bg-white p-5">
            <SectionHeader icon={Zap} eyebrow="Animation 1-2" title="Token timeline: draft, verify, accept">
              Speculation helps only when the draft prefix survives verification. Current estimated speedup is {state.speedup.toFixed(2)}x in this toy model.
            </SectionHeader>
            <DraftVerifyPanel mode={mode} depth={state.activeDepth} acceptedCount={state.acceptedCount} />
          </section>

          <section className="rounded-lg border border-slate-200 bg-white p-5">
            <SectionHeader icon={Layers3} eyebrow="Animation 3" title="EAGLE-3 feature fusion">
              EAGLE-3 moved from constrained feature prediction toward direct token prediction with multi-layer target features.
            </SectionHeader>
            <FeatureFusionPanel />
          </section>

          <section className="grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
            <div className={`rounded-lg border bg-white p-5 ${metric === 'attention' ? 'border-cyan-700' : 'border-slate-200'}`}>
              <SectionHeader icon={Layers3} eyebrow="Animation 4" title="Attention drift heatmap">
                Bright blue cells are prompt anchors. Bright red cells are recent draft-token positions.
              </SectionHeader>
              <AttentionHeatmap depth={state.activeDepth} drift={state.drift} />
              <MassChart promptMass={state.promptMass} draftMass={state.draftMass} />
            </div>

            <div className={`rounded-lg border bg-white p-5 ${metric === 'norm' ? 'border-cyan-700' : 'border-slate-200'}`}>
              <SectionHeader icon={BarChart3} eyebrow="Animation 5" title="Hidden-state norm grows">
                If scale keeps changing, the next draft step receives a different kind of object than the previous one.
              </SectionHeader>
              <NormChart depth={state.activeDepth} normalization={normalization} drift={state.drift} />
            </div>
          </section>

          <section className="rounded-lg border border-slate-200 bg-white p-5">
            <SectionHeader icon={Cpu} eyebrow="Animation 6" title="EAGLE-3 vs EAGLE 3.1 architecture">
              EAGLE 3.1 adds normalization before FC fusion and after drafter output feedback.
            </SectionHeader>
            <ArchitecturePanel />
          </section>

          <section className="grid gap-4 lg:grid-cols-[0.95fr_1.05fr]">
            <div className={`rounded-lg border bg-white p-5 ${metric === 'acceptance' ? 'border-cyan-700' : 'border-slate-200'}`}>
              <SectionHeader icon={CheckCircle2} eyebrow="Animation 7" title="Acceptance length payoff">
                Longer accepted prefixes explain where the inference speedup comes from.
              </SectionHeader>
              <AcceptanceChart depth={state.activeDepth} context={context} />
            </div>

            <div className="rounded-lg border border-slate-200 bg-white p-5">
              <SectionHeader icon={XCircle} eyebrow="Benchmark caution" title="Specific vLLM-reported setup">
                These numbers are one early reported system result, not a universal guarantee.
              </SectionHeader>
              <div className="grid gap-3">
                {[
                  ['Concurrency 1', '2.03x per-user output throughput'],
                  ['Concurrency 4', '1.71x'],
                  ['Concurrency 16', '1.66x'],
                ].map(([label, value]) => (
                  <div key={label} className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 p-3">
                    <span className="font-bold text-slate-700">{label}</span>
                    <span className="font-mono text-lg font-black text-cyan-800">{value}</span>
                  </div>
                ))}
              </div>
              <p className="mt-4 text-sm leading-6 text-slate-700">
                Reported for Kimi-K2.6-NVFP4 with vLLM, GB200, and SPEED-Bench coding.
              </p>
            </div>
          </section>
        </main>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <SectionHeader icon={ClipboardList} eyebrow="Learning objectives" title="What this pack should leave behind">
          The lesson pairs visual intuition with small implementation tasks.
        </SectionHeader>
        <div className="grid gap-3 md:grid-cols-2">
          {OBJECTIVES.map((objective, index) => (
            <div key={objective} className="flex gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
              <span className="font-mono text-xs font-black text-cyan-700">{String(index + 1).padStart(2, '0')}</span>
              <p className="text-sm leading-6 text-slate-700">{objective}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <SectionHeader icon={FileText} eyebrow="Storyboard" title="Seven animation panels" />
          <div className="grid gap-3">
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
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <SectionHeader icon={Code2} eyebrow="Rustlings-style lab" title="mini-eagle exercises">
            Each file intentionally contains TODOs and tests that fail until the learner repairs the algorithm.
          </SectionHeader>
          <div className="grid gap-3">
            {EXERCISES.map(([file, body], index) => (
              <div key={file} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <strong className="font-mono text-sm text-slate-950">{String(index + 1).padStart(2, '0')} {file}</strong>
                <p className="mt-1 text-sm leading-6 text-slate-600">{body}</p>
              </div>
            ))}
          </div>
          <pre className="mt-4 overflow-x-auto rounded-lg border border-slate-200 bg-slate-950 p-3 text-xs leading-5 text-slate-100">cd mini-eagle
cargo test --bins</pre>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <SectionHeader icon={Sparkles} eyebrow="Mental models" title="Keep the vocabulary grounded" />
          <div className="grid gap-2">
            {MENTAL_MODELS.map(([concept, model]) => (
              <div key={concept} className="grid grid-cols-[150px_1fr] gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm">
                <strong className="text-slate-950">{concept}</strong>
                <span className="text-slate-700">{model}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <SectionHeader icon={HelpCircle} eyebrow="Final quiz" title="Fast recall check" />
          <div className="grid gap-3">
            {FINAL_QUIZ.map(([question, answer], index) => (
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

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <SectionHeader icon={LinkIcon} eyebrow="Paper trail" title="Sources and scope" />
        <div className="grid gap-3 md:grid-cols-2">
          {SOURCE_LINKS.map((source) => (
            <a
              key={source.href}
              href={source.href}
              target="_blank"
              rel="noreferrer"
              className="rounded-lg border border-slate-200 bg-slate-50 p-4 transition hover:border-cyan-500"
            >
              <span className="font-black text-slate-950">{source.label}</span>
              <span className="mt-1 block text-sm leading-6 text-slate-600">{source.note}</span>
            </a>
          ))}
        </div>
      </section>

      <AssessmentPanel lessonId="eagle-3-1-speculative-decoding" title="EAGLE 3.1 check" />
    </div>
  );
}
