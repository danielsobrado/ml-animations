import React, { useMemo, useState } from 'react';
import {
  ArrowRight,
  BarChart3,
  Binary,
  BookOpen,
  CheckCircle2,
  Code2,
  Cpu,
  Database,
  Gauge,
  HelpCircle,
  Layers3,
  Link as LinkIcon,
  RotateCcw,
  SlidersHorizontal,
  Sparkles,
  Target,
  Zap,
} from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const CONTEXT_LENGTHS = [4096, 16384, 65536, 131072];

const MODES = [
  { id: 'fp16', label: 'FP16 KV cache', detail: 'Reference cache with no quantization error.' },
  { id: 'naive', label: 'Naive scalar quantization', detail: 'Round each coordinate to a low-bit code.' },
  { id: 'mse', label: 'MSE-only TurboQuant', detail: 'Rotate and quantize for reconstruction error.' },
  { id: 'turbo', label: 'TurboQuant + QJL residual', detail: 'Add a 1-bit residual sketch for dot products.' },
];

const HEAD_MODES = [
  { id: 'mha', label: 'MHA', heads: 32 },
  { id: 'gqa', label: 'GQA', heads: 8 },
  { id: 'mqa', label: 'MQA', heads: 1 },
];

const STORYBOARD = [
  ['The growing notebook', 'Long-context decoding stores K and V vectors for every previous token at every layer.'],
  ['Why keys matter', 'Compressing keys must preserve query-key attention scores.'],
  ['Why values matter', 'Compressing values must preserve the information carried into the context vector.'],
  ['Naive rounding', 'Low-bit rounding saves memory but can change attention rankings.'],
  ['Rotate before quantizing', 'Random rotation spreads energy so coordinate quantization behaves more predictably.'],
  ['Quantize the main vector', 'Most bits store a near-optimal approximation of the rotated vector.'],
  ['The residual is small but important', 'Small norm error can still be biased for dot products.'],
  ['QJL residual sketch', 'One-bit signs carry a correction for the residual contribution.'],
  ['Attention score restored', 'The corrected estimator aims to preserve scores, not only vector appearance.'],
  ['Memory payoff', 'Lower KV precision means longer contexts, larger batches, or cheaper inference.'],
];

const EXERCISES = [
  ['01_kv_cache_size.rs', 'Compute KV cache bits and compression ratios.'],
  ['02_uniform_quantization.rs', 'Map scalars to low-bit codes and back.'],
  ['03_dot_product_error.rs', 'Measure query-key score error directly.'],
  ['04_random_rotation.rs', 'Show that a 2D rotation preserves L2 norm.'],
  ['05_mse_vs_inner_product.rs', 'Compare squared error with signed dot-product error.'],
  ['06_residual_correction.rs', 'Build a toy sign-sketch residual correction.'],
  ['07_outlier_channels.rs', 'Select high-magnitude channels and compute effective bits.'],
  ['08_attention_topk_agreement.rs', 'Measure top-k attention ranking overlap.'],
  ['09_compression_tradeoff.rs', 'Pick the smallest safe bit-width under an error budget.'],
];

const QUESTIONS = [
  ['What is stored in the KV cache?', 'Key and value vectors for previous tokens at each transformer layer.'],
  ['Why does KV cache memory grow during decoding?', 'Each generated token appends another key and value vector to the cache.'],
  ['Why is KV cache critical for long-context inference?', 'Cache size grows with context length, so memory and bandwidth pressure become large.'],
  ['What is the attention score between a query and key?', 'Their dot product, usually scaled before softmax.'],
  ['What does quantization do?', 'It maps high-precision values to a smaller set of low-bit codes.'],
  ['Why can quantization hurt attention?', 'It can change query-key dot products and therefore change which tokens receive attention.'],
  ['Why is minimizing MSE not always enough?', 'A vector can reconstruct well while still producing biased inner products.'],
  ['What does random rotation help with?', 'It spreads vector energy more evenly across coordinates.'],
  ['What are TurboQuant main stages?', 'MSE-focused vector quantization followed by a 1-bit QJL residual correction.'],
  ['What does the QJL residual stage try to fix?', 'Bias in inner-product estimation.'],
  ['Why is inner-product unbiasedness important for keys?', 'Attention logits are query-key inner products.'],
  ['Why might outlier channels get more bits?', 'Some dimensions contribute disproportionately to error, so they deserve higher precision.'],
  ['What does online mean here?', 'Vectors can be quantized as they are produced during inference without training a model-specific codebook.'],
  ['How is KV-cache quantization different from weight quantization?', 'Weight quantization compresses model parameters; KV-cache quantization compresses request-specific runtime memory.'],
  ['Why can a 3-bit KV cache still work?', 'High-dimensional geometry, rotation, and residual correction can preserve important dot products.'],
  ['Which metric matters more for key compression?', 'Dot-product error is directly tied to attention logits, though MSE still matters.'],
  ['Why are values different from keys?', 'Keys decide attention weights; values are mixed by those weights into the next context vector.'],
  ['Why can TurboQuant help batching?', 'Smaller KV caches let more requests or longer contexts fit in the same memory budget.'],
];

const SOURCE_LINKS = [
  {
    title: 'TurboQuant paper',
    href: 'https://openreview.net/forum?id=tO3ASKZlok',
    detail: 'Online vector quantization for MSE and inner-product distortion, with rotation and QJL residual correction.',
  },
  {
    title: 'Google Research blog',
    href: 'https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/',
    detail: 'KV-cache bottlenecks, PolarQuant intuition, QJL explanation, and reported long-context benchmark results.',
  },
  {
    title: 'arXiv HTML',
    href: 'https://arxiv.org/html/2504.19874v1',
    detail: 'Detailed algorithm discussion, unbiased inner-product estimator, and mixed low-bit KV-cache experiments.',
  },
];

const ORIGINAL_VECTOR = [0.92, -0.12, 0.34, 1.41, -0.77, 0.05, 0.58, -1.08];
const QUERY = [0.72, -0.31, 0.48, 0.22];
const KEYS = [
  [0.82, -0.44, 0.11, 0.28],
  [-0.24, 0.51, 0.42, -0.18],
  [0.64, -0.18, 0.72, 0.36],
  [0.31, -0.61, 0.17, 0.74],
];

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function fmt(value, digits = 1) {
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function fmtInt(value) {
  return Number(value).toLocaleString();
}

function bytesFromBits(bits) {
  const bytes = bits / 8;
  if (bytes >= 1024 ** 3) return `${fmt(bytes / 1024 ** 3)} GiB`;
  if (bytes >= 1024 ** 2) return `${fmt(bytes / 1024 ** 2)} MiB`;
  if (bytes >= 1024) return `${fmt(bytes / 1024)} KiB`;
  return `${fmt(bytes)} B`;
}

function dot(a, b) {
  return a.reduce((sum, x, index) => sum + x * b[index], 0);
}

function rotateToy(vector, enabled) {
  if (!enabled) return vector;
  const theta = Math.PI / 5;
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  const rotated = [];

  for (let i = 0; i < vector.length; i += 2) {
    const x = vector[i];
    const y = vector[i + 1] ?? 0;
    rotated.push(x * cos - y * sin);
    rotated.push(x * sin + y * cos);
  }

  return rotated.map((value, index) => value * (index % 3 === 0 ? 0.86 : index % 3 === 1 ? 1.08 : 0.96));
}

function quantizeValue(value, bits) {
  const levels = 2 ** bits;
  const min = -1.5;
  const max = 1.5;
  const clamped = clamp(value, min, max);
  const code = Math.round(((clamped - min) / (max - min)) * (levels - 1));
  const decoded = min + (code / (levels - 1)) * (max - min);
  return { code, decoded };
}

function rankScores(scores) {
  return scores
    .map((score, index) => ({ score, index }))
    .sort((a, b) => b.score - a.score)
    .map((item) => item.index);
}

function rankCorrelation(a, b) {
  const n = a.length;
  const posA = new Map(a.map((item, index) => [item, index]));
  const posB = new Map(b.map((item, index) => [item, index]));
  const sumSq = a.reduce((sum, item) => {
    const diff = posA.get(item) - posB.get(item);
    return sum + diff * diff;
  }, 0);
  return clamp(1 - (6 * sumSq) / (n * (n * n - 1)), -1, 1);
}

function buildSimulation({ mode, contextLength, bits, headMode, rotationEnabled, qjlEnabled, outlierFraction }) {
  const layers = 32;
  const headDim = 128;
  const kvHeads = HEAD_MODES.find((item) => item.id === headMode)?.heads ?? 8;
  const outlierBits = Math.min(8, bits + 1);
  const regularFraction = 1 - outlierFraction;
  const weightedBits = mode === 'fp16' ? 16 : regularFraction * bits + outlierFraction * outlierBits;
  const residualBits = mode === 'turbo' && qjlEnabled ? 0.5 : 0;
  const effectiveBits = weightedBits + residualBits;

  const fp16Bits = layers * contextLength * kvHeads * headDim * 2 * 16;
  const compressedBits = layers * contextLength * kvHeads * headDim * 2 * effectiveBits;
  const compressionRatio = fp16Bits / compressedBits;
  const baseError = 2 ** -bits;
  const rotationFactor = rotationEnabled ? 0.72 : 1.15;
  const qjlFactor = mode === 'turbo' && qjlEnabled ? 0.2 : 1;

  const mseError = mode === 'fp16'
    ? 0
    : baseError * rotationFactor * (mode === 'naive' ? 1.35 : mode === 'mse' ? 0.76 : 0.82);
  const dotBias = mode === 'fp16'
    ? 0
    : baseError * rotationFactor * (
      mode === 'naive' ? -1.4
        : mode === 'mse' ? -0.8
          : -0.8 * qjlFactor
    );
  const top1Agreement = mode === 'fp16'
    ? 1
    : clamp(1 - Math.abs(dotBias) * 2.4 + (qjlEnabled && mode === 'turbo' ? 0.16 : 0), 0.5, 1);
  const attentionLogitSpeed = mode === 'fp16'
    ? 1
    : clamp(16 / effectiveBits * (mode === 'turbo' ? 1.18 : 1) * (rotationEnabled ? 0.96 : 1), 1, 8.4);

  const rotated = rotateToy(ORIGINAL_VECTOR, rotationEnabled);
  const quantized = rotated.map((value) => quantizeValue(value, bits));
  const reconstructed = quantized.map((item) => item.decoded);
  const residual = rotated.map((value, index) => value - reconstructed[index]);
  const signs = residual.map((value) => (value >= 0 ? '+' : '-'));

  const trueScores = KEYS.map((key) => dot(QUERY, key));
  const quantizedScores = trueScores.map((score, index) => {
    if (mode === 'fp16') return score;
    const drift = dotBias * (index % 2 === 0 ? 1 : -0.55) + mseError * (index === 0 ? 0.75 : -0.35);
    return score + drift;
  });
  const correctedScores = trueScores.map((score, index) => {
    if (mode !== 'turbo' || !qjlEnabled) return quantizedScores[index];
    return score + dotBias * 0.25 * (index % 2 === 0 ? 1 : -1);
  });
  const fullRank = rankScores(trueScores);
  const quantRank = rankScores(mode === 'turbo' && qjlEnabled ? correctedScores : quantizedScores);
  const top1Preserved = fullRank[0] === quantRank[0];
  const rankCorr = rankCorrelation(fullRank, quantRank);
  const outlierChannels = Math.round(headDim * outlierFraction);

  return {
    attentionLogitSpeed,
    compressedBits,
    compressionRatio,
    correctedScores,
    dotBias,
    effectiveBits,
    fp16Bits,
    fullRank,
    headDim,
    kvHeads,
    layers,
    mseError,
    outlierBits,
    outlierChannels,
    quantized,
    quantizedScores,
    quantRank,
    rankCorr,
    reconstructed,
    residual,
    signs,
    top1Agreement,
    top1Preserved,
    trueScores,
    weightedBits,
  };
}

function SectionHeader({ icon: Icon, eyebrow, title, children }) {
  return (
    <div className="mb-4 min-w-0">
      <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-cyan-700">
        <Icon size={16} />
        {eyebrow}
      </p>
      <h3 className="mt-2 text-xl font-black tracking-normal text-slate-950">{title}</h3>
      {children ? <p className="mt-2 break-words text-sm leading-6 text-slate-700">{children}</p> : null}
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
      <strong className="block break-words text-2xl font-black text-slate-950">{value}</strong>
      <p className="mt-1 break-words text-sm leading-6 text-slate-600">{detail}</p>
    </div>
  );
}

function MemoryWall({ simulation, contextLength, mode }) {
  const rows = [
    ['FP16', simulation.fp16Bits, '#1d4ed8'],
    ['INT8', simulation.fp16Bits / 2, '#64748b'],
    ['4-bit', simulation.fp16Bits / 4, '#0f766e'],
    ['3-bit', simulation.fp16Bits * (3 / 16), '#ca8a04'],
    ['current', simulation.compressedBits, mode === 'turbo' ? '#0891b2' : '#334155'],
  ];

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
        <div className="text-xs font-black uppercase tracking-wide text-slate-500">
          Context length
        </div>
        <div className="mt-2 flex flex-wrap items-end gap-2">
          <span className="text-3xl font-black text-slate-950">{fmtInt(contextLength)}</span>
          <span className="pb-1 text-sm font-bold text-slate-600">tokens cached per layer</span>
        </div>
      </div>

      <div className="space-y-2">
        {rows.map(([label, bitsValue, color]) => {
          const width = `${clamp((bitsValue / simulation.fp16Bits) * 100, 3, 100)}%`;
          return (
            <div key={label} className="grid grid-cols-[72px_minmax(0,1fr)_86px] items-center gap-3 text-xs font-black uppercase tracking-wide text-slate-600">
              <span>{label}</span>
              <div className="h-5 rounded-sm bg-slate-100">
                <div className="h-5 rounded-sm" style={{ width, backgroundColor: color }} />
              </div>
              <span className="text-right normal-case tracking-normal">{bytesFromBits(bitsValue)}</span>
            </div>
          );
        })}
      </div>

      <p className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm leading-6 text-amber-900">
        Toy simulator values. The paper and blog report benchmark results; this panel only shows how memory scales.
      </p>
    </div>
  );
}

function ValueStrip({ label, values, formatter = (value) => fmt(value, 2), highlightSigns = false }) {
  return (
    <div className="min-w-0">
      <div className="mb-2 text-xs font-black uppercase tracking-wide text-slate-500">{label}</div>
      <div className="flex min-w-0 flex-wrap gap-2">
        {values.map((value, index) => {
          const text = typeof value === 'string' ? value : formatter(value);
          const positive = text === '+' || value > 0;
          return (
            <span
              key={`${label}-${index}`}
              className={`inline-flex min-h-9 min-w-12 items-center justify-center rounded-lg border px-2 font-mono text-xs font-black ${
                highlightSigns
                  ? positive
                    ? 'border-emerald-300 bg-emerald-50 text-emerald-800'
                    : 'border-rose-300 bg-rose-50 text-rose-800'
                  : 'border-slate-200 bg-white text-slate-800'
              }`}
            >
              {text}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function GeometryPanel({ simulation, rotationEnabled, bits }) {
  return (
    <div className="grid min-w-0 gap-4">
      <div className="grid min-w-0 gap-3 rounded-lg border border-slate-200 bg-slate-50 p-4">
        <div className="flex flex-wrap items-center gap-2 text-xs font-black uppercase tracking-wide text-slate-500">
          <span>Original vector</span>
          <ArrowRight size={14} />
          <span>{rotationEnabled ? 'rotated coordinates' : 'same coordinates'}</span>
          <ArrowRight size={14} />
          <span>{bits}-bit codes</span>
        </div>
        <ValueStrip label="original" values={ORIGINAL_VECTOR} />
        <ValueStrip label={rotationEnabled ? 'after rotation' : 'rotation off'} values={rotateToy(ORIGINAL_VECTOR, rotationEnabled)} />
        <ValueStrip label="quantized reconstruction" values={simulation.reconstructed} />
        <ValueStrip label="residual signs" values={simulation.signs} highlightSigns />
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        {[
          ['rotation', rotationEnabled ? 'energy spread' : 'disabled'],
          ['main quantizer', `${bits}-bit scalar codes`],
          ['QJL residual', 'sign sketch of leftover error'],
        ].map(([title, detail]) => (
          <div key={title} className="rounded-lg border border-slate-200 bg-white p-3">
            <div className="text-xs font-black uppercase tracking-wide text-cyan-700">{title}</div>
            <p className="mt-2 text-sm font-semibold text-slate-700">{detail}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function AttentionScores({ simulation, mode, qjlEnabled }) {
  const visibleScores = mode === 'turbo' && qjlEnabled ? simulation.correctedScores : simulation.quantizedScores;
  const maxScore = Math.max(...simulation.trueScores, ...visibleScores);
  const minScore = Math.min(...simulation.trueScores, ...visibleScores);
  const range = Math.max(0.1, maxScore - minScore);

  return (
    <div className="grid min-w-0 gap-4">
      <div className="grid gap-3">
        {simulation.trueScores.map((score, index) => {
          const approx = visibleScores[index];
          const trueWidth = `${clamp(((score - minScore) / range) * 86 + 8, 8, 94)}%`;
          const approxWidth = `${clamp(((approx - minScore) / range) * 86 + 8, 8, 94)}%`;
          return (
            <div key={`key-${index}`} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="mb-2 flex items-center justify-between text-xs font-black uppercase tracking-wide text-slate-500">
                <span>K{index + 1}</span>
                <span>true {fmt(score, 2)} / shown {fmt(approx, 2)}</span>
              </div>
              <div className="grid gap-1">
                <div className="h-3 rounded-sm bg-slate-200">
                  <div className="h-3 rounded-sm bg-blue-600" style={{ width: trueWidth }} />
                </div>
                <div className="h-3 rounded-sm bg-slate-200">
                  <div className={`h-3 rounded-sm ${mode === 'turbo' && qjlEnabled ? 'bg-cyan-600' : 'bg-amber-600'}`} style={{ width: approxWidth }} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-lg border border-slate-200 bg-white p-3">
          <div className="text-xs font-black uppercase tracking-wide text-slate-500">Full precision ranking</div>
          <p className="mt-2 font-mono text-sm font-black text-slate-900">
            {simulation.fullRank.map((index) => `K${index + 1}`).join(' > ')}
          </p>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-3">
          <div className="text-xs font-black uppercase tracking-wide text-slate-500">Current ranking</div>
          <p className="mt-2 font-mono text-sm font-black text-slate-900">
            {simulation.quantRank.map((index) => `K${index + 1}`).join(' > ')}
          </p>
        </div>
      </div>
    </div>
  );
}

function TradeoffChart({ bits, simulation }) {
  const points = [
    ['FP16', 16, 0.01],
    ['8-bit', 8, 0.03],
    ['4-bit', 4, 0.08],
    ['3-bit TQ', 3.5, 0.05],
    ['current', simulation.effectiveBits, Math.abs(simulation.dotBias)],
  ];

  return (
    <div className="grid min-w-0 gap-4">
      <div className="space-y-3">
        {points.map(([label, pointBits, error]) => {
          const memoryWidth = `${clamp((pointBits / 16) * 100, 6, 100)}%`;
          const errorWidth = `${clamp(error * 400, 4, 96)}%`;
          return (
            <div key={label} className="grid grid-cols-[80px_minmax(0,1fr)] gap-3 text-xs font-black uppercase tracking-wide text-slate-600">
              <span>{label}</span>
              <div className="grid gap-1">
                <div className="h-3 rounded-sm bg-slate-100">
                  <div className="h-3 rounded-sm bg-cyan-700" style={{ width: memoryWidth }} />
                </div>
                <div className="h-3 rounded-sm bg-slate-100">
                  <div className="h-3 rounded-sm bg-orange-500" style={{ width: errorWidth }} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm leading-6 text-slate-700">
        <span className="font-black text-slate-950">Reading the chart:</span> blue is memory used versus FP16; orange is toy dot-product error. At {bits} bits, the useful question is whether score error stays low enough for attention ranking.
      </div>
    </div>
  );
}

function OutlierPanel({ simulation, outlierFraction, bits }) {
  const total = 64;
  const highlighted = Math.round(total * outlierFraction);

  return (
    <div className="grid min-w-0 gap-4">
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="text-xs font-black uppercase tracking-wide text-slate-500">
            Head dimension strip
          </div>
          <div className="text-xs font-black uppercase tracking-wide text-cyan-700">
            {simulation.outlierChannels} / {simulation.headDim} outlier channels
          </div>
        </div>
        <div className="grid gap-1" style={{ gridTemplateColumns: 'repeat(16, minmax(0, 1fr))' }}>
          {Array.from({ length: total }, (_, index) => {
            const isOutlier = index < highlighted;
            return (
              <span
                key={index}
                className={`h-4 rounded-sm border ${isOutlier ? 'border-amber-500 bg-amber-200' : 'border-slate-200 bg-white'}`}
                title={isOutlier ? 'outlier channel' : 'regular channel'}
              />
            );
          })}
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-lg border border-slate-200 bg-white p-3">
          <div className="text-xs font-black uppercase tracking-wide text-slate-500">regular</div>
          <div className="mt-1 text-xl font-black text-slate-950">{bits}-bit</div>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-3">
          <div className="text-xs font-black uppercase tracking-wide text-slate-500">outlier</div>
          <div className="mt-1 text-xl font-black text-slate-950">{simulation.outlierBits}-bit</div>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-3">
          <div className="text-xs font-black uppercase tracking-wide text-slate-500">effective</div>
          <div className="mt-1 text-xl font-black text-slate-950">{fmt(simulation.weightedBits, 2)} bits</div>
        </div>
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
      className="block rounded-lg border border-slate-200 bg-slate-50 p-4 transition hover:border-cyan-700 hover:bg-cyan-50"
    >
      <div className="flex items-center gap-2 text-sm font-black text-slate-950">
        <LinkIcon size={16} />
        {source.title}
      </div>
      <p className="mt-2 text-sm leading-6 text-slate-700">{source.detail}</p>
    </a>
  );
}

export default function TurboQuant() {
  const [mode, setMode] = useState('turbo');
  const [contextLength, setContextLength] = useState(65536);
  const [bits, setBits] = useState(3);
  const [headMode, setHeadMode] = useState('gqa');
  const [rotationEnabled, setRotationEnabled] = useState(true);
  const [qjlEnabled, setQjlEnabled] = useState(true);
  const [outlierFraction, setOutlierFraction] = useState(0.1);

  const simulation = useMemo(() => buildSimulation({
    mode,
    contextLength,
    bits,
    headMode,
    rotationEnabled,
    qjlEnabled,
    outlierFraction,
  }), [mode, contextLength, bits, headMode, rotationEnabled, qjlEnabled, outlierFraction]);

  const reset = () => {
    setMode('turbo');
    setContextLength(65536);
    setBits(3);
    setHeadMode('gqa');
    setRotationEnabled(true);
    setQjlEnabled(true);
    setOutlierFraction(0.1);
  };

  return (
    <div className="min-w-0 space-y-6">
      <section className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-cyan-700">
              <BookOpen size={16} />
              Modern inference track - Transformers and attention
            </p>
            <h2 className="mt-2 max-w-4xl text-2xl font-black tracking-normal text-slate-950 md:text-3xl">
              TurboQuant: Shrinking the KV Cache Without Forgetting
            </h2>
            <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-700">
              Long-context inference stores key and value vectors for every previous token. TurboQuant compresses those vectors to low bit-widths while preserving the query-key dot products attention relies on.
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
          ['Full precision', 'Every remembered vector is stored with many bits per coordinate.'],
          ['Naive quantization', 'Rounding saves memory but may shift attention scores.'],
          ['TurboQuant', 'Rotate, quantize the main vector, then correct residual bias.'],
          ['Key metric', 'For keys, preserving q times k matters more than pretty reconstruction.'],
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
              Context length: {fmtInt(contextLength)}
              <select
                value={contextLength}
                onChange={(event) => setContextLength(Number(event.target.value))}
                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-bold"
              >
                {CONTEXT_LENGTHS.map((value) => (
                  <option key={value} value={value}>{fmtInt(value)} tokens</option>
                ))}
              </select>
            </label>

            <label className="grid gap-2 text-sm font-bold text-slate-700">
              Bit-width: {bits}-bit
              <input
                type="range"
                min="2"
                max="8"
                step="1"
                value={bits}
                onChange={(event) => setBits(Number(event.target.value))}
                disabled={mode === 'fp16'}
                className="accent-cyan-800"
              />
            </label>

            <div>
              <div className="mb-2 text-sm font-black text-slate-700">KV heads</div>
              <div className="grid grid-cols-3 gap-2">
                {HEAD_MODES.map((item) => (
                  <ControlButton key={item.id} active={headMode === item.id} onClick={() => setHeadMode(item.id)}>
                    {item.label}
                  </ControlButton>
                ))}
              </div>
            </div>

            <div className="grid gap-2">
              <ControlButton active={rotationEnabled} onClick={() => setRotationEnabled((value) => !value)}>
                Rotation: {rotationEnabled ? 'On' : 'Off'}
              </ControlButton>
              <ControlButton active={qjlEnabled} onClick={() => setQjlEnabled((value) => !value)}>
                Residual correction: {qjlEnabled ? '1-bit QJL' : 'Off'}
              </ControlButton>
            </div>

            <label className="grid gap-2 text-sm font-bold text-slate-700">
              Outlier channels: {Math.round(outlierFraction * 100)}%
              <input
                type="range"
                min="0"
                max="25"
                step="5"
                value={Math.round(outlierFraction * 100)}
                onChange={(event) => setOutlierFraction(Number(event.target.value) / 100)}
                className="accent-cyan-800"
              />
            </label>
          </div>
        </aside>

        <main className="min-w-0 space-y-4">
          <section className="grid min-w-0 gap-4 md:grid-cols-2 xl:grid-cols-4">
            <Metric
              icon={Database}
              label="KV memory"
              value={bytesFromBits(simulation.compressedBits)}
              detail={`${simulation.layers} layers, ${simulation.kvHeads} KV heads, ${simulation.headDim} head dimension.`}
            />
            <Metric
              icon={Binary}
              label="Compression ratio"
              value={`${fmt(simulation.compressionRatio, 1)}x`}
              detail={`${fmt(simulation.effectiveBits, 2)} effective bits per KV value in this toy setup.`}
            />
            <Metric
              icon={Target}
              label="Dot-product bias"
              value={fmt(simulation.dotBias, 3)}
              detail="Signed synthetic q times k shift. QJL reduces systematic bias."
            />
            <Metric
              icon={Gauge}
              label="Attention top-1 agreement"
              value={simulation.top1Preserved ? 'preserved' : `${fmt(simulation.top1Agreement * 100, 0)}%`}
              detail={`Rank correlation ${fmt(simulation.rankCorr, 2)} under current settings.`}
            />
            <Metric
              icon={Sparkles}
              label="MSE error"
              value={fmt(simulation.mseError, 3)}
              detail="Reconstruction error is useful, but it is not the same as attention-score error."
            />
            <Metric
              icon={Cpu}
              label="Estimated logit speed"
              value={`${fmt(simulation.attentionLogitSpeed, 1)}x`}
              detail="Synthetic relative cost from fewer key bits; not a hardware benchmark."
            />
            <Metric
              icon={Layers3}
              label="KV heads"
              value={simulation.kvHeads}
              detail="GQA and MQA reduce the number of cached KV heads before quantization."
            />
            <Metric
              icon={CheckCircle2}
              label="Outlier channels"
              value={simulation.outlierChannels}
              detail="High-magnitude channels receive one extra bit in the mixed-precision strip."
            />
          </section>

          <section className="grid min-w-0 gap-4 xl:grid-cols-2">
            <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
              <SectionHeader icon={Database} eyebrow="Panel 1" title="KV cache memory wall">
                Weight quantization shrinks the model. KV-cache quantization shrinks the conversation memory.
              </SectionHeader>
              <MemoryWall simulation={simulation} contextLength={contextLength} mode={mode} />
            </div>

            <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
              <SectionHeader icon={Binary} eyebrow="Panel 2" title="Quantization geometry">
                TurboQuant is not just rounding. It rotates vectors, stores the main shape, then sketches the residual.
              </SectionHeader>
              <GeometryPanel simulation={simulation} rotationEnabled={rotationEnabled} bits={bits} />
            </div>
          </section>

          <section className="grid min-w-0 gap-4 xl:grid-cols-[1.05fr_0.95fr]">
            <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
              <SectionHeader icon={Target} eyebrow="Panel 3" title="Attention score preservation">
                The learner should see ranking preservation, not only smaller memory numbers.
              </SectionHeader>
              <AttentionScores simulation={simulation} mode={mode} qjlEnabled={qjlEnabled} />
            </div>

            <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
              <SectionHeader icon={BarChart3} eyebrow="Panel 4" title="Compression and error tradeoff">
                Lower bits help only while the attention score error remains acceptable.
              </SectionHeader>
              <TradeoffChart bits={bits} simulation={simulation} />
            </div>
          </section>
        </main>
      </section>

      <section className="grid min-w-0 gap-4 xl:grid-cols-[1fr_0.9fr]">
        <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
          <SectionHeader icon={Zap} eyebrow="Pipeline" title="Rotate, quantize, correct">
            TurboQuant first captures the vector with an MSE quantizer, then uses QJL on the residual so inner products are not systematically shifted.
          </SectionHeader>
          <div className="grid gap-3 md:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr] md:items-center">
            {[
              ['K or V vector', 'Runtime cache entry'],
              ['Random rotation', 'Spread coordinate energy'],
              ['MSE quantizer', 'Store low-bit main vector'],
              ['QJL residual', 'One-bit sign correction'],
            ].map(([title, body], index) => (
              <React.Fragment key={title}>
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                  <div className="text-sm font-black text-slate-950">{title}</div>
                  <p className="mt-2 text-sm leading-6 text-slate-700">{body}</p>
                </div>
                {index < 3 ? <ArrowRight className="hidden text-cyan-700 md:block" size={22} /> : null}
              </React.Fragment>
            ))}
          </div>
        </div>

        <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
          <SectionHeader icon={Layers3} eyebrow="Mixed precision" title="Outlier channels">
            Spend extra bits where quantization error hurts most.
          </SectionHeader>
          <OutlierPanel simulation={simulation} outlierFraction={outlierFraction} bits={bits} />
        </div>
      </section>

      <section className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
        <SectionHeader icon={BookOpen} eyebrow="Storyboard" title="Ten teaching panels" />
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
          {STORYBOARD.map(([title, caption], index) => (
            <article key={title} className="min-w-0 rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="text-xs font-black uppercase tracking-wide text-cyan-700">Panel {index + 1}</div>
              <h4 className="mt-2 text-sm font-black text-slate-950">{title}</h4>
              <p className="mt-2 text-sm leading-6 text-slate-700">{caption}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="grid min-w-0 gap-4 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
          <SectionHeader icon={Code2} eyebrow="Rustlings" title="mini-turboquant exercises">
            Nine small TODO files reinforce cache sizing, quantization, dot-product error, rotation, residual correction, outliers, and tradeoff planning.
          </SectionHeader>
          <div className="grid gap-2">
            {EXERCISES.map(([file, detail]) => (
              <div key={file} className="grid grid-cols-[180px_minmax(0,1fr)] gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm">
                <code className="font-black text-cyan-800">{file}</code>
                <span className="text-slate-700">{detail}</span>
              </div>
            ))}
          </div>
          <pre className="mt-4 overflow-x-auto rounded-lg border border-slate-200 bg-slate-950 p-3 text-xs leading-5 text-slate-100">cd mini-turboquant
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
          The simulator uses deterministic toy arrays. Reported compression and speedup claims are linked as source material, not treated as universal app benchmarks.
        </SectionHeader>
        <div className="grid gap-3 md:grid-cols-3">
          {SOURCE_LINKS.map((source) => (
            <SourceCard key={source.href} source={source} />
          ))}
        </div>
      </section>

      <AssessmentPanel lessonId="turboquant" title="TurboQuant check" />
    </div>
  );
}
