import React, { useMemo, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Brain,
  Cpu,
  Database,
  GitBranch,
  Image,
  Layers,
  MessageSquare,
  Network,
  RotateCcw,
  SlidersHorizontal,
  Video,
  Volume2,
} from 'lucide-react';
import {
  ARCHITECTURE_FAMILIES,
  ATTENTION_MODES,
  COMPARISON_ROWS,
  CONTEXT_LENGTHS,
  FAMILY_BY_ID,
  HIDDEN_OPTIONS,
  LAYER_OPTIONS,
  PAPER_SIGNAL_CARDS,
  PRECISION_BYTES,
  QUESTION_AXES,
} from './data';

const TABS = [
  { id: 'map', label: 'Architecture Map' },
  { id: 'dense', label: 'Dense Baseline' },
  { id: 'attention', label: 'Attention Memory' },
  { id: 'moe', label: 'Sparse MoE' },
  { id: 'long-context', label: 'Long Context' },
  { id: 'ssm', label: 'SSM / Recurrent' },
  { id: 'diffusion', label: 'Diffusion LM' },
  { id: 'omni', label: 'Omni Multimodal' },
  { id: 'compare', label: 'Compare' },
];

const MODALITY_STYLES = {
  text: 'border-blue-200 bg-blue-50 text-blue-950',
  image: 'border-violet-200 bg-violet-50 text-violet-950',
  audio: 'border-orange-200 bg-orange-50 text-orange-950',
  video: 'border-emerald-200 bg-emerald-50 text-emerald-950',
};

const FAMILY_ACCENTS = {
  'dense-transformer': 'border-blue-300 bg-blue-50 text-blue-950',
  'sparse-moe': 'border-amber-300 bg-amber-50 text-amber-950',
  'attention-compressed': 'border-cyan-300 bg-cyan-50 text-cyan-950',
  'long-context': 'border-emerald-300 bg-emerald-50 text-emerald-950',
  'state-space-hybrid': 'border-slate-300 bg-slate-50 text-slate-950',
  'diffusion-language': 'border-fuchsia-300 bg-fuchsia-50 text-fuchsia-950',
  'omni-multimodal': 'border-rose-300 bg-rose-50 text-rose-950',
};

function formatCompact(value) {
  if (value >= 1000000) return `${(value / 1000000).toFixed(value % 1000000 === 0 ? 0 : 1)}M`;
  if (value >= 1000) return `${Math.round(value / 1000)}K`;
  return String(value);
}

function formatGb(value) {
  if (value < 1) return `${(value * 1024).toFixed(0)} MB`;
  return `${value.toFixed(value >= 10 ? 0 : 1)} GB`;
}

function ButtonGroup({ label, options, value, onChange, formatter = String }) {
  return (
    <div>
      <div className="mb-2 text-xs font-black uppercase tracking-wide text-slate-500">{label}</div>
      <div className="flex flex-wrap gap-2">
        {options.map((option) => (
          <button
            key={option}
            type="button"
            onClick={() => onChange(option)}
            className={`rounded-lg border px-3 py-2 text-sm font-bold transition ${
              value === option
                ? 'border-slate-950 bg-slate-950 text-white'
                : 'border-slate-200 bg-white text-slate-700 hover:border-slate-400'
            }`}
          >
            {formatter(option)}
          </button>
        ))}
      </div>
    </div>
  );
}

function MetricChip({ label, value, tone = 'slate', helper }) {
  const tones = {
    slate: 'border-slate-200 bg-white text-slate-950',
    green: 'border-emerald-200 bg-emerald-50 text-emerald-950',
    amber: 'border-amber-200 bg-amber-50 text-amber-950',
    red: 'border-rose-200 bg-rose-50 text-rose-950',
    blue: 'border-blue-200 bg-blue-50 text-blue-950',
  };

  return (
    <div className={`rounded-lg border p-3 ${tones[tone] || tones.slate}`}>
      <div className="text-xs font-black uppercase tracking-wide opacity-70">{label}</div>
      <div className="mt-1 text-lg font-black">{value}</div>
      {helper && <p className="mt-1 text-xs leading-5 opacity-80">{helper}</p>}
    </div>
  );
}

function TokenChip({ children, kind = 'text', active = true }) {
  return (
    <span
      className={`inline-flex min-h-[34px] items-center rounded-lg border px-3 py-1 text-sm font-black shadow-sm ${
        active ? MODALITY_STYLES[kind] || MODALITY_STYLES.text : 'border-slate-200 bg-slate-100 text-slate-500'
      }`}
    >
      {children}
    </span>
  );
}

function SectionCard({ title, children, icon: Icon = Layers }) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
        <Icon size={16} />
        {title}
      </h3>
      {children}
    </section>
  );
}

function ArchitectureSelector({ selectedFamily, onSelect }) {
  return (
    <aside className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
        <SlidersHorizontal size={16} />
        Switchboard
      </div>
      <div className="space-y-2">
        {ARCHITECTURE_FAMILIES.map((family) => (
          <button
            key={family.id}
            type="button"
            onClick={() => onSelect(family.id)}
            className={`w-full rounded-lg border p-3 text-left transition ${
              selectedFamily === family.id
                ? FAMILY_ACCENTS[family.id]
                : 'border-slate-200 bg-slate-50 text-slate-700 hover:border-slate-400'
            }`}
          >
            <span className="block text-sm font-black">{family.name}</span>
            <span className="mt-1 block text-xs leading-5 opacity-80">{family.tradeoff}</span>
          </button>
        ))}
      </div>
    </aside>
  );
}

function PipelineBlock({ label, active, detail }) {
  return (
    <div
      className={`rounded-lg border p-3 text-center transition ${
        active ? 'border-amber-300 bg-amber-50 text-amber-950 shadow-sm' : 'border-slate-200 bg-slate-50 text-slate-500'
      }`}
    >
      <div className="text-sm font-black">{label}</div>
      <div className="mt-1 text-xs leading-5">{detail}</div>
    </div>
  );
}

function ArchitectureMap({ familyId }) {
  const active = {
    modality: familyId === 'omni-multimodal',
    attention: ['dense-transformer', 'sparse-moe', 'attention-compressed', 'long-context', 'omni-multimodal'].includes(familyId),
    long: familyId === 'long-context',
    ssm: familyId === 'state-space-hybrid',
    diffusion: familyId === 'diffusion-language',
    moe: familyId === 'sparse-moe',
    dense: familyId === 'dense-transformer',
    output: true,
  };

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="grid gap-3 lg:grid-cols-[1fr_auto_1.1fr_auto_1fr_auto_1fr] lg:items-center">
        <PipelineBlock label="Input tokens" active detail="text, position, or modality tokens" />
        <ArrowRight className="hidden text-slate-400 lg:block" size={20} />
        <div className="space-y-2">
          <PipelineBlock label="Embedding / encoder" active={active.modality} detail={active.modality ? 'modality adapters active' : 'text embedding path'} />
          <PipelineBlock label="Sequence mixer" active={active.attention || active.ssm || active.diffusion} detail="attention, state, or denoising" />
        </div>
        <ArrowRight className="hidden text-slate-400 lg:block" size={20} />
        <div className="space-y-2">
          <PipelineBlock label="MHA / GQA / MLA" active={active.attention} detail="KV layout changes" />
          <PipelineBlock label="Long-context strategy" active={active.long} detail="full, local, RAG, compressed" />
          <PipelineBlock label="SSM / recurrent state" active={active.ssm} detail="state update path" />
          <PipelineBlock label="Diffusion refinement" active={active.diffusion} detail="masked denoising path" />
        </div>
        <ArrowRight className="hidden text-slate-400 lg:block" size={20} />
        <div className="space-y-2">
          <PipelineBlock label="Dense FFN" active={active.dense || active.attention || active.long} detail="all tokens use it" />
          <PipelineBlock label="MoE experts" active={active.moe} detail="selected experts light up" />
          <PipelineBlock label="Output head" active={active.output} detail="next token, denoised tokens, speech, tools" />
        </div>
      </div>
    </div>
  );
}

function TokenJourney({ familyId, controls }) {
  const family = FAMILY_BY_ID[familyId];
  const isOmni = familyId === 'omni-multimodal';
  const tokens = isOmni
    ? [
        ['text', 'Analyze'],
        ['text', 'chart'],
        ['image', 'patches'],
        ['audio', 'speech'],
        ['video', 'frames'],
      ]
    : [
        ['text', 'token t'],
        ['text', 'query'],
        ['text', 'context'],
      ];

  return (
    <SectionCard title="Architecture switchboard token journey" icon={Network}>
      <div className="mt-4 grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
          <div className="flex flex-wrap items-center gap-2">
            {tokens.map(([kind, label], index) => (
              <React.Fragment key={`${kind}-${label}`}>
                <TokenChip kind={kind}>{label}</TokenChip>
                {index < tokens.length - 1 && <ArrowRight size={16} className="text-slate-400" />}
              </React.Fragment>
            ))}
          </div>
          <div className="mt-5 grid gap-3 md:grid-cols-4">
            {familyId === 'dense-transformer' && ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4'].map((layer) => (
              <div key={layer} className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-center text-sm font-black text-amber-950">
                {layer}<span className="block text-xs font-semibold">all blocks active</span>
              </div>
            ))}
            {familyId === 'sparse-moe' && Array.from({ length: controls.experts }, (_, index) => {
              const active = index === 1 || index === Math.min(controls.experts - 1, 6) || (controls.topK > 2 && index === 3);
              return (
                <div key={index} className={`rounded-lg border p-3 text-center text-sm font-black ${active ? 'border-amber-300 bg-amber-50 text-amber-950' : 'border-slate-200 bg-slate-100 text-slate-500'}`}>
                  Expert {index + 1}
                </div>
              );
            })}
            {familyId === 'attention-compressed' && <AttentionMemorySketch controls={controls} />}
            {familyId === 'long-context' && <LongContextSketch controls={controls} />}
            {familyId === 'state-space-hybrid' && <StateSketch controls={controls} />}
            {familyId === 'diffusion-language' && <DiffusionSketch controls={controls} />}
            {familyId === 'omni-multimodal' && <OmniSketch controls={controls} />}
          </div>
        </div>

        <div className={`rounded-lg border p-4 ${FAMILY_ACCENTS[familyId]}`}>
          <div className="text-sm font-black uppercase tracking-wide opacity-70">Read the paper as</div>
          <h4 className="mt-2 text-xl font-black">{family.name}</h4>
          <p className="mt-3 text-sm leading-6">{family.readInPaper}</p>
          <div className="mt-4 grid grid-cols-2 gap-2">
            <MetricChip label="Active compute" value={family.activeCompute} />
            <MetricChip label="KV memory" value={family.kvMemory} />
            <MetricChip label="Context access" value={family.contextScaling} />
            <MetricChip label="Generation" value={family.generationOrder} />
          </div>
        </div>
      </div>
    </SectionCard>
  );
}

function AttentionMemorySketch({ controls }) {
  const mode = ATTENTION_MODES[controls.attentionMode];
  const kvCount = controls.attentionMode === 'MLA' ? 1 : mode.kvHeads(controls.queryHeads);
  const label = controls.attentionMode === 'MLA' ? 'latent KV capsule' : `${kvCount} K/V blocks`;
  return (
    <div className="col-span-full grid gap-3">
      <div className="flex flex-wrap gap-2">
        {Array.from({ length: Math.min(controls.queryHeads, 16) }, (_, index) => (
          <span key={index} className="rounded-md border border-blue-200 bg-blue-50 px-2 py-1 text-xs font-black text-blue-950">Q{index + 1}</span>
        ))}
      </div>
      <div className="flex flex-wrap gap-2">
        {Array.from({ length: Math.min(kvCount, 12) }, (_, index) => (
          <span key={index} className={`rounded-lg border px-3 py-2 text-xs font-black ${controls.attentionMode === 'MLA' ? 'border-cyan-300 bg-cyan-50 text-cyan-950' : 'border-amber-300 bg-amber-50 text-amber-950'}`}>
            {controls.attentionMode === 'MLA' ? label : `K/V ${index + 1}`}
          </span>
        ))}
      </div>
    </div>
  );
}

function LongContextSketch({ controls }) {
  const segments = ['Intro', 'Facts', 'Distractors', 'Needle', 'More distractors', 'Question'];
  return (
    <div className="col-span-full space-y-3">
      <div className="grid grid-cols-6 gap-1">
        {segments.map((segment) => {
          const isNeedle = segment === 'Needle';
          const missed = controls.contextStrategy === 'sliding-window' && controls.needlePosition !== 'end';
          return (
            <div key={segment} className={`rounded-lg border p-2 text-center text-xs font-black ${isNeedle && !missed ? 'border-emerald-300 bg-emerald-50 text-emerald-950' : isNeedle ? 'border-rose-300 bg-rose-50 text-rose-950' : 'border-slate-200 bg-white text-slate-600'}`}>
              {segment}
            </div>
          );
        })}
      </div>
      <p className="text-sm text-slate-700">
        Strategy: {controls.contextStrategy}. Confidence changes because long context is retrieval plus memory pressure, not just a larger number.
      </p>
    </div>
  );
}

function StateSketch({ controls }) {
  return (
    <div className="col-span-full flex flex-wrap items-center gap-2">
      {Array.from({ length: 5 }, (_, index) => (
        <React.Fragment key={index}>
          <div className="rounded-lg border border-slate-300 bg-white px-4 py-3 text-sm font-black text-slate-900">
            state {index}
            <span className="block text-xs font-semibold text-slate-500">{index % 2 ? 'write/read' : 'keep/forget'}</span>
          </div>
          {index < 4 && <ArrowRight size={16} className="text-slate-400" />}
        </React.Fragment>
      ))}
      <div className="w-full text-sm text-slate-700">Hybrid mode: {controls.hybridMode}</div>
    </div>
  );
}

function DiffusionSketch({ controls }) {
  const steps = [
    ['[MASK]', '[MASK]', '[MASK]', '[MASK]'],
    ['The', '[MASK]', 'sat', '[MASK]'],
    ['The', 'model', 'sat', '[MASK]'],
    ['The', 'model', 'sat', 'there'],
  ];
  return (
    <div className="col-span-full space-y-2">
      {steps.slice(0, Math.min(steps.length, Math.max(2, controls.denoiseSteps / 4))).map((row, rowIndex) => (
        <div key={rowIndex} className="flex flex-wrap items-center gap-2">
          <span className="w-16 text-xs font-black uppercase text-slate-500">step {rowIndex}</span>
          {row.map((token, index) => (
            <TokenChip key={`${rowIndex}-${index}`} active={token !== '[MASK]'}>{token}</TokenChip>
          ))}
        </div>
      ))}
    </div>
  );
}

function OmniSketch({ controls }) {
  return (
    <div className="col-span-full grid gap-3 md:grid-cols-[1fr_auto_1fr_auto_1fr] md:items-center">
      <div className="flex flex-wrap gap-2">
        <TokenChip kind="text">text</TokenChip>
        <TokenChip kind="image">image</TokenChip>
        <TokenChip kind="audio">audio</TokenChip>
        <TokenChip kind="video">video</TokenChip>
      </div>
      <ArrowRight className="hidden text-slate-400 md:block" size={18} />
      <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-center text-sm font-black text-amber-950">
        shared thinker
        <span className="block text-xs font-semibold">{controls.fusion} fusion</span>
      </div>
      <ArrowRight className="hidden text-slate-400 md:block" size={18} />
      <div className="rounded-lg border border-slate-300 bg-white p-3 text-center text-sm font-black text-slate-950">
        {controls.outputMode} output
      </div>
    </div>
  );
}

function SharedControls({ controls, setControls, tab }) {
  const update = (patch) => setControls((current) => ({ ...current, ...patch }));

  return (
    <SectionCard title="Controls" icon={SlidersHorizontal}>
      <div className="mt-4 grid gap-5 md:grid-cols-2 xl:grid-cols-4">
        <ButtonGroup label="Context" options={CONTEXT_LENGTHS} value={controls.contextLength} onChange={(contextLength) => update({ contextLength })} formatter={formatCompact} />
        <ButtonGroup label="Layers" options={LAYER_OPTIONS} value={controls.layers} onChange={(layers) => update({ layers })} />
        <ButtonGroup label="Hidden size" options={HIDDEN_OPTIONS} value={controls.hiddenSize} onChange={(hiddenSize) => update({ hiddenSize })} formatter={formatCompact} />
        <ButtonGroup label="Precision" options={['fp16', 'fp8', 'int8']} value={controls.precision} onChange={(precision) => update({ precision })} />
      </div>

      {tab === 'attention' && (
        <div className="mt-5 grid gap-5 md:grid-cols-3">
          <ButtonGroup label="Attention mode" options={Object.keys(ATTENTION_MODES)} value={controls.attentionMode} onChange={(attentionMode) => update({ attentionMode })} />
          <ButtonGroup label="Query heads" options={[8, 16, 32]} value={controls.queryHeads} onChange={(queryHeads) => update({ queryHeads })} />
          <ButtonGroup label="Latent dim" options={[64, 128, 256, 512]} value={controls.latentDim} onChange={(latentDim) => update({ latentDim })} />
        </div>
      )}

      {tab === 'moe' && (
        <div className="mt-5 grid gap-5 md:grid-cols-4">
          <ButtonGroup label="Experts" options={[4, 8, 16, 64]} value={controls.experts} onChange={(experts) => update({ experts })} />
          <ButtonGroup label="Top-k" options={[1, 2, 4]} value={controls.topK} onChange={(topK) => update({ topK })} />
          <ButtonGroup label="Load balance" options={['none', 'aux-loss', 'aux-loss-free']} value={controls.loadBalance} onChange={(loadBalance) => update({ loadBalance })} />
          <ButtonGroup label="Token batch" options={['balanced', 'skewed', 'domain-specific']} value={controls.tokenBatch} onChange={(tokenBatch) => update({ tokenBatch })} />
        </div>
      )}

      {tab === 'long-context' && (
        <div className="mt-5 grid gap-5 md:grid-cols-3">
          <ButtonGroup label="Strategy" options={['full', 'sliding-window', 'global-local', 'rag', 'compressed-memory', 'ssm']} value={controls.contextStrategy} onChange={(contextStrategy) => update({ contextStrategy })} />
          <ButtonGroup label="Needle" options={['start', 'middle', 'end']} value={controls.needlePosition} onChange={(needlePosition) => update({ needlePosition })} />
          <ButtonGroup label="Distractors" options={['low', 'medium', 'high']} value={controls.distractorDensity} onChange={(distractorDensity) => update({ distractorDensity })} />
        </div>
      )}

      {tab === 'ssm' && (
        <div className="mt-5 grid gap-5 md:grid-cols-3">
          <ButtonGroup label="Hybrid mode" options={['ssm-only', 'attention-only', 'alternating']} value={controls.hybridMode} onChange={(hybridMode) => update({ hybridMode })} />
          <ButtonGroup label="State size" options={[512, 1024, 4096]} value={controls.stateSize} onChange={(stateSize) => update({ stateSize })} />
          <ButtonGroup label="Forget rate" options={['low', 'medium', 'high']} value={controls.forgetRate} onChange={(forgetRate) => update({ forgetRate })} />
        </div>
      )}

      {tab === 'diffusion' && (
        <div className="mt-5 grid gap-5 md:grid-cols-3">
          <ButtonGroup label="Mask ratio" options={[0.15, 0.5, 0.9]} value={controls.maskRatio} onChange={(maskRatio) => update({ maskRatio })} />
          <ButtonGroup label="Denoise steps" options={[4, 8, 16, 32]} value={controls.denoiseSteps} onChange={(denoiseSteps) => update({ denoiseSteps })} />
          <ButtonGroup label="Generation order" options={['fixed', 'confidence-based', 'random']} value={controls.generationOrder} onChange={(generationOrder) => update({ generationOrder })} />
        </div>
      )}

      {tab === 'omni' && (
        <div className="mt-5 grid gap-5 md:grid-cols-3">
          <ButtonGroup label="Fusion" options={['early', 'late', 'cross-attention']} value={controls.fusion} onChange={(fusion) => update({ fusion })} />
          <ButtonGroup label="Output" options={['text', 'speech', 'structured-json']} value={controls.outputMode} onChange={(outputMode) => update({ outputMode })} />
          <ButtonGroup label="Speech decoder" options={['codec-ar', 'diffusion', 'causal-convnet']} value={controls.speechDecoder} onChange={(speechDecoder) => update({ speechDecoder })} />
        </div>
      )}
    </SectionCard>
  );
}

function DensePanel({ controls }) {
  const activePath = controls.layers * controls.hiddenSize * 4;
  return (
    <div className="grid gap-5 xl:grid-cols-[1fr_0.8fr]">
      <SectionCard title="Dense decoder block" icon={Cpu}>
        <div className="mt-4 grid gap-3">
          {['Token embedding', 'RMSNorm / LayerNorm', 'Self-attention', 'Residual add', 'RMSNorm / LayerNorm', 'Dense FFN / MLP', 'Residual add'].map((stage, index) => (
            <div key={stage} className={`rounded-lg border p-3 text-sm font-black ${index === 5 ? 'border-amber-300 bg-amber-50 text-amber-950' : 'border-slate-200 bg-slate-50 text-slate-800'}`}>
              {stage}
            </div>
          ))}
        </div>
      </SectionCard>
      <SectionCard title="Baseline readout" icon={Activity}>
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          <MetricChip label="Active params per token" value="all layer path" helper={`${formatCompact(activePath)} toy units`} />
          <MetricChip label="KV cache" value="grows with length" helper={`${formatCompact(controls.contextLength)} visible prefix tokens`} />
          <MetricChip label="Context access" value="attention prefix" helper="causal decoder reads earlier tokens" />
          <MetricChip label="Bottleneck" value="compute + KV memory" tone="amber" />
        </div>
        <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-4 font-mono text-sm text-slate-800">
          x_l+1 = x_l + Attn(x_l) + FFN(x_l)
        </div>
        <p className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm leading-6 text-amber-950">
          Dense does not mean small. Dense means every token uses the same full parameter path.
        </p>
      </SectionCard>
    </div>
  );
}

function AttentionPanel({ controls }) {
  const bytes = PRECISION_BYTES[controls.precision];
  const mode = ATTENTION_MODES[controls.attentionMode];
  const classicKvHeads = controls.attentionMode === 'MLA' ? controls.queryHeads : mode.kvHeads(controls.queryHeads);
  const headDim = Math.max(64, Math.round(controls.hiddenSize / controls.queryHeads));
  const mhaGb = (controls.layers * controls.contextLength * controls.queryHeads * headDim * 2 * bytes) / 1024 ** 3;
  const modeGb = controls.attentionMode === 'MLA'
    ? (controls.layers * controls.contextLength * controls.latentDim * bytes) / 1024 ** 3
    : (controls.layers * controls.contextLength * classicKvHeads * headDim * 2 * bytes) / 1024 ** 3;
  const memoryRatio = Math.min(1, modeGb / mhaGb);

  return (
    <div className="grid gap-5">
      <div className="grid gap-4 md:grid-cols-4">
        {Object.entries(ATTENTION_MODES).map(([id, item]) => (
          <div key={id} className={`rounded-lg border p-4 ${controls.attentionMode === id ? 'border-cyan-300 bg-cyan-50 text-cyan-950' : 'border-slate-200 bg-white text-slate-700'}`}>
            <div className="text-lg font-black">{item.label}</div>
            <p className="mt-2 text-sm leading-6">{item.description}</p>
          </div>
        ))}
      </div>
      <div className="grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
        <SectionCard title="KV memory layout" icon={Database}>
          <div className="mt-4">
            <AttentionMemorySketch controls={controls} />
          </div>
          <div className="mt-5 h-4 overflow-hidden rounded-full bg-slate-100">
            <div className="h-full bg-cyan-600" style={{ width: `${Math.max(4, memoryRatio * 100)}%` }} />
          </div>
          <p className="mt-2 text-sm text-slate-600">{controls.attentionMode} uses about {(memoryRatio * 100).toFixed(1)}% of the full MHA toy KV cache.</p>
        </SectionCard>
        <SectionCard title="Math notes" icon={Brain}>
          <div className="mt-4 space-y-3">
            <MetricChip label="MHA toy cache" value={formatGb(mhaGb)} />
            <MetricChip label={`${controls.attentionMode} toy cache`} value={formatGb(modeGb)} tone="blue" />
            <MetricChip label="Head independence" value={`${mode.independence}%`} helper={mode.complexity} />
          </div>
          <pre className="mt-4 overflow-auto rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs text-slate-800">{`KV cache bytes approx
layers * tokens * KV_heads * head_dim * 2 * bytes

MLA latent cache approx
layers * tokens * latent_dim * bytes`}</pre>
        </SectionCard>
      </div>
    </div>
  );
}

function MoEPanel({ controls }) {
  const loadPattern = Array.from({ length: controls.experts }, (_, index) => {
    const base = controls.tokenBatch === 'balanced' ? 60 : index === 1 ? 96 : index === 2 ? 82 : 28 + ((index * 13) % 35);
    if (controls.loadBalance !== 'none') return Math.round((base + 60) / 2);
    return base;
  });
  const maxLoad = Math.max(...loadPattern);

  return (
    <div className="grid gap-5 xl:grid-cols-[1.1fr_0.9fr]">
      <SectionCard title="Router and experts" icon={GitBranch}>
        <div className="mt-4 grid gap-3 md:grid-cols-[1fr_160px_1.5fr] md:items-center">
          <div className="space-y-2">
            {['math token', 'code token', 'image token', 'legal token'].map((token, index) => (
              <TokenChip key={token} kind={index === 2 ? 'image' : 'text'}>{token}</TokenChip>
            ))}
          </div>
          <div className="rounded-lg border border-slate-300 bg-slate-50 p-4 text-center text-sm font-black text-slate-950">
            Router
            <span className="block text-xs text-slate-500">top-{controls.topK}</span>
          </div>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {Array.from({ length: controls.experts }, (_, index) => {
              const active = index < controls.topK || index === Math.min(controls.experts - 1, 7);
              return (
                <div key={index} className={`rounded-lg border p-3 text-center text-sm font-black ${active ? 'border-amber-300 bg-amber-50 text-amber-950' : 'border-slate-200 bg-slate-100 text-slate-500'}`}>
                  Expert {index + 1}
                </div>
              );
            })}
          </div>
        </div>
        <p className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm leading-6 text-amber-950">
          MoE does not mean the model is small. It means only part of a large model is active for each token.
        </p>
      </SectionCard>

      <SectionCard title="Load and capacity" icon={Activity}>
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          <MetricChip label="Total capacity" value={`${controls.experts} experts`} />
          <MetricChip label="Active experts" value={`${controls.topK}${controls.sharedExpert ? ' + shared' : ''}`} />
          <MetricChip label="Load balancing" value={controls.loadBalance} tone={controls.loadBalance === 'none' ? 'amber' : 'green'} />
          <MetricChip label="Communication cost" value={controls.experts > 16 ? 'high' : 'medium'} />
        </div>
        <div className="mt-5 space-y-2">
          {loadPattern.slice(0, Math.min(controls.experts, 16)).map((load, index) => (
            <div key={index} className="grid grid-cols-[72px_1fr_36px] items-center gap-2 text-xs font-bold text-slate-700">
              <span>Expert {index + 1}</span>
              <div className="h-3 overflow-hidden rounded-full bg-slate-100">
                <div className={`h-full ${load > 85 ? 'bg-rose-500' : 'bg-amber-500'}`} style={{ width: `${(load / maxLoad) * 100}%` }} />
              </div>
              <span>{load}</span>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

function LongContextPanel({ controls }) {
  const fullMatrixCells = controls.contextLength * controls.contextLength;
  const confidence = (() => {
    if (controls.contextStrategy === 'full') return controls.distractorDensity === 'high' ? 72 : 88;
    if (controls.contextStrategy === 'sliding-window') return controls.needlePosition === 'end' ? 76 : 28;
    if (controls.contextStrategy === 'rag') return controls.distractorDensity === 'high' ? 55 : 82;
    if (controls.contextStrategy === 'compressed-memory') return controls.needlePosition === 'middle' ? 48 : 68;
    if (controls.contextStrategy === 'ssm') return 58;
    return 70;
  })();

  return (
    <div className="grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
      <SectionCard title="Needle in a long context" icon={Database}>
        <div className="mt-4">
          <LongContextSketch controls={controls} />
        </div>
        <div className="mt-5 grid gap-3 md:grid-cols-3">
          <MetricChip label="Attention matrix" value={`${formatCompact(fullMatrixCells)} cells`} helper="toy score matrix size" />
          <MetricChip label="Retrieval confidence" value={`${confidence}%`} tone={confidence < 50 ? 'red' : confidence < 75 ? 'amber' : 'green'} />
          <MetricChip label="Failure mode" value={confidence < 50 ? 'missed evidence' : 'distractor risk'} />
        </div>
      </SectionCard>
      <SectionCard title="What long context stresses" icon={AlertTriangle}>
        <div className="mt-4 grid gap-2">
          {['KV cache memory', 'attention compute', 'position encoding', 'retrieval over distractors', 'lost-in-the-middle behavior', 'latency'].map((item) => (
            <div key={item} className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm font-bold text-slate-800">{item}</div>
          ))}
        </div>
        <p className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm leading-6 text-amber-950">
          A longer context window does not guarantee better use of long context.
        </p>
      </SectionCard>
    </div>
  );
}

function SsmPanel({ controls }) {
  const memoryGrowth = controls.hybridMode === 'attention-only' ? 'grows with tokens' : 'state sized';
  const lookup = controls.hybridMode === 'alternating' ? 'stronger' : controls.hybridMode === 'attention-only' ? 'direct' : 'weaker';
  return (
    <div className="grid gap-5 xl:grid-cols-[1.1fr_0.9fr]">
      <SectionCard title="Attention vs state update" icon={Network}>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <div className="rounded-lg border border-blue-200 bg-blue-50 p-4 text-blue-950">
            <div className="text-sm font-black uppercase">Attention</div>
            <p className="mt-2 text-sm leading-6">token t attends to tokens 1...t</p>
            <div className="mt-4 grid grid-cols-5 gap-1">
              {Array.from({ length: 25 }, (_, index) => (
                <div key={index} className={`h-6 rounded ${index % 6 <= Math.floor(index / 5) ? 'bg-blue-600' : 'bg-blue-100'}`} />
              ))}
            </div>
          </div>
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-slate-950">
            <div className="text-sm font-black uppercase">SSM / recurrent</div>
            <p className="mt-2 text-sm leading-6">state_t = update(state_t-1, token_t)</p>
            <div className="mt-4">
              <StateSketch controls={controls} />
            </div>
          </div>
        </div>
      </SectionCard>
      <SectionCard title="Tradeoff readout" icon={Activity}>
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          <MetricChip label="Memory growth" value={memoryGrowth} />
          <MetricChip label="State size" value={formatCompact(controls.stateSize)} />
          <MetricChip label="Content lookup" value={lookup} />
          <MetricChip label="Long-range retention" value={controls.forgetRate === 'high' ? 'fragile' : 'better'} />
        </div>
        <p className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm leading-6 text-slate-700">
          Recurrent/state-space does not mean old RNN. Modern selective SSMs are built for efficient long-sequence and hardware-aware computation.
        </p>
      </SectionCard>
    </div>
  );
}

function DiffusionPanel({ controls }) {
  const cost = controls.denoiseSteps;
  const parallelism = Math.round((1 - controls.maskRatio / 2) * 100);
  return (
    <div className="grid gap-5 xl:grid-cols-[1fr_1fr]">
      <SectionCard title="Autoregressive vs diffusion" icon={MessageSquare}>
        <div className="mt-4 grid gap-4">
          <div className="rounded-lg border border-blue-200 bg-blue-50 p-4">
            <div className="text-sm font-black uppercase text-blue-950">Autoregressive</div>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              {['The', 'cat', 'sat', 'on', 'the', 'mat'].map((token, index) => (
                <React.Fragment key={token + index}>
                  <TokenChip>{token}</TokenChip>
                  {index < 5 && <ArrowRight size={14} className="text-blue-400" />}
                </React.Fragment>
              ))}
            </div>
          </div>
          <div className="rounded-lg border border-fuchsia-200 bg-fuchsia-50 p-4">
            <div className="text-sm font-black uppercase text-fuchsia-950">Diffusion LM</div>
            <div className="mt-3">
              <DiffusionSketch controls={controls} />
            </div>
          </div>
        </div>
      </SectionCard>
      <SectionCard title="Refinement metrics" icon={Activity}>
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          <MetricChip label="Parallelism proxy" value={`${parallelism}%`} />
          <MetricChip label="Forward passes" value={`${cost} steps`} />
          <MetricChip label="Error propagation" value={controls.generationOrder === 'confidence-based' ? 'can remask' : 'order dependent'} />
          <MetricChip label="Latency" value={cost > 16 ? 'high' : 'moderate'} tone={cost > 16 ? 'amber' : 'green'} />
        </div>
        <p className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm leading-6 text-amber-950">
          Diffusion language models usually operate over discrete token masking and refinement, not pixel noise copied from image diffusion.
        </p>
      </SectionCard>
    </div>
  );
}

function OmniPanel({ controls }) {
  const tokenBudget = {
    text: 800,
    image: 4096,
    audio: 12000,
    video: 48000,
  };
  const total = Object.values(tokenBudget).reduce((sum, value) => sum + value, 0);
  return (
    <div className="grid gap-5 xl:grid-cols-[1.1fr_0.9fr]">
      <SectionCard title="Modality stream" icon={Image}>
        <div className="mt-4">
          <OmniSketch controls={controls} />
        </div>
        <div className="mt-5 grid gap-3 md:grid-cols-4">
          <MetricChip label="Text" value={`${tokenBudget.text} tokens`} />
          <MetricChip label="Image" value={`${tokenBudget.image} patches`} />
          <MetricChip label="Audio" value={`${formatCompact(tokenBudget.audio)} frames`} />
          <MetricChip label="Video" value={`${formatCompact(tokenBudget.video)} frames`} tone="amber" />
        </div>
      </SectionCard>
      <SectionCard title="Thinker and talker" icon={Volume2}>
        <div className="mt-4 grid gap-3">
          <div className="rounded-lg border border-blue-200 bg-blue-50 p-3 text-sm font-bold text-blue-950"><MessageSquare size={16} className="mr-2 inline" /> text tokens enter directly</div>
          <div className="rounded-lg border border-violet-200 bg-violet-50 p-3 text-sm font-bold text-violet-950"><Image size={16} className="mr-2 inline" /> image patches use projectors</div>
          <div className="rounded-lg border border-orange-200 bg-orange-50 p-3 text-sm font-bold text-orange-950"><Volume2 size={16} className="mr-2 inline" /> speech can use codec tokens</div>
          <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm font-bold text-emerald-950"><Video size={16} className="mr-2 inline" /> video stresses token budget fastest</div>
        </div>
        <MetricChip label="Total toy budget" value={`${formatCompact(total)} modality tokens`} helper="Different modalities are not handled the same way." />
      </SectionCard>
    </div>
  );
}

function ComparePanel() {
  return (
    <div className="grid gap-5">
      <SectionCard title="Architecture comparison table" icon={Layers}>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full table-fixed border-collapse text-left text-sm">
            <colgroup>
              <col className="w-[23%]" />
              <col className="w-[29%]" />
              <col className="w-[24%]" />
              <col className="w-[24%]" />
            </colgroup>
            <thead>
              <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
                <th className="break-words py-3 pr-4">Architecture family</th>
                <th className="break-words py-3 pr-4">What changes?</th>
                <th className="break-words py-3 pr-4">Main advantage</th>
                <th className="break-words py-3">Main cost / risk</th>
              </tr>
            </thead>
            <tbody>
              {COMPARISON_ROWS.map(([family, changes, advantage, risk]) => (
                <tr key={family} className="border-b border-slate-100">
                  <td className="break-words py-3 pr-4 font-black text-slate-950">{family}</td>
                  <td className="break-words py-3 pr-4 text-slate-700">{changes}</td>
                  <td className="break-words py-3 pr-4 text-slate-700">{advantage}</td>
                  <td className="break-words py-3 text-slate-700">{risk}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Paper annotation overlay" icon={Brain}>
        <div className="mt-4 grid gap-3 lg:grid-cols-5">
          {PAPER_SIGNAL_CARDS.map((card) => (
            <div key={card.title} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <div className="text-base font-black text-slate-950">{card.title}</div>
              <p className="mt-2 text-sm leading-6 text-slate-700"><strong>Signals:</strong> {card.signals}</p>
              <p className="mt-2 text-sm leading-6 text-slate-700"><strong>Read as:</strong> {card.readAs}</p>
              <p className="mt-3 text-xs font-bold uppercase leading-5 text-slate-500">{card.question}</p>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

export default function FrontierLlmArchitectureOverview() {
  const [tab, setTab] = useState('map');
  const [selectedFamily, setSelectedFamily] = useState('sparse-moe');
  const [controls, setControls] = useState({
    contextLength: 128000,
    layers: 32,
    hiddenSize: 4096,
    precision: 'fp16',
    attentionMode: 'GQA',
    queryHeads: 16,
    latentDim: 128,
    experts: 8,
    topK: 2,
    sharedExpert: true,
    loadBalance: 'aux-loss-free',
    tokenBatch: 'skewed',
    contextStrategy: 'rag',
    needlePosition: 'middle',
    distractorDensity: 'high',
    hybridMode: 'alternating',
    stateSize: 1024,
    forgetRate: 'medium',
    maskRatio: 0.5,
    denoiseSteps: 8,
    generationOrder: 'confidence-based',
    fusion: 'early',
    outputMode: 'speech',
    speechDecoder: 'codec-ar',
  });

  const activeFamily = FAMILY_BY_ID[selectedFamily];
  const tabFamily = useMemo(() => {
    if (tab === 'dense') return 'dense-transformer';
    if (tab === 'attention') return 'attention-compressed';
    if (tab === 'moe') return 'sparse-moe';
    if (tab === 'long-context') return 'long-context';
    if (tab === 'ssm') return 'state-space-hybrid';
    if (tab === 'diffusion') return 'diffusion-language';
    if (tab === 'omni') return 'omni-multimodal';
    return selectedFamily;
  }, [selectedFamily, tab]);

  const reset = () => {
    setTab('map');
    setSelectedFamily('sparse-moe');
  };

  return (
    <div className="ua-lesson-stage min-h-full">
      <div className="mx-auto max-w-7xl space-y-6 p-4 md:p-6">
        <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <div className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-500">
                <Cpu size={17} />
                Frontier LLM map lesson
              </div>
              <h1 className="mt-2 text-2xl font-black text-slate-950 md:text-3xl">
                Read modern architecture diagrams by following one token
              </h1>
              <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-700">
                The original Transformer replaced recurrence and convolution with attention. Modern frontier systems
                change that base pattern to reduce active compute, reduce KV-cache memory, scale context, route tokens
                to experts, add modalities, or change the generation process itself.
              </p>
            </div>
            <button
              type="button"
              onClick={reset}
              className="inline-flex w-fit items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-bold text-slate-800"
            >
              <RotateCcw size={16} />
              Reset
            </button>
          </div>
        </section>

        <nav className="flex gap-2 overflow-x-auto rounded-lg border border-slate-200 bg-white p-2 shadow-sm" aria-label="Lesson tabs">
          {TABS.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setTab(item.id)}
              className={`whitespace-nowrap rounded-lg px-3 py-2 text-sm font-black transition ${
                tab === item.id ? 'bg-slate-950 text-white' : 'text-slate-600 hover:bg-slate-100'
              }`}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <div className="grid gap-6 xl:grid-cols-[300px_1fr]">
          <ArchitectureSelector selectedFamily={selectedFamily} onSelect={setSelectedFamily} />
          <div className="min-w-0 space-y-6">
            <div className="grid gap-3 md:grid-cols-3">
              <MetricChip label="Selected family" value={activeFamily.shortName} tone="blue" />
              <MetricChip label="Best for" value={activeFamily.bestFor} />
              <MetricChip label="New failure mode" value={activeFamily.failureMode} tone="amber" />
            </div>

            <ArchitectureMap familyId={tabFamily} />

            {tab === 'map' && (
              <>
                <TokenJourney familyId={selectedFamily} controls={controls} />
                <SectionCard title="Same questions for every architecture" icon={Brain}>
                  <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                    {QUESTION_AXES.map((axis) => (
                      <div key={axis} className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm font-bold leading-6 text-slate-800">
                        {axis}
                      </div>
                    ))}
                  </div>
                </SectionCard>
              </>
            )}

            {tab !== 'map' && tab !== 'compare' && (
              <SharedControls controls={controls} setControls={setControls} tab={tab} />
            )}

            {tab === 'dense' && <DensePanel controls={controls} />}
            {tab === 'attention' && <AttentionPanel controls={controls} />}
            {tab === 'moe' && <MoEPanel controls={controls} />}
            {tab === 'long-context' && <LongContextPanel controls={controls} />}
            {tab === 'ssm' && <SsmPanel controls={controls} />}
            {tab === 'diffusion' && <DiffusionPanel controls={controls} />}
            {tab === 'omni' && <OmniPanel controls={controls} />}
            {tab === 'compare' && <ComparePanel />}

            <SectionCard title="How to read this in a paper" icon={AlertTriangle}>
              <div className="mt-4 grid gap-3 md:grid-cols-2">
                {[
                  'When a paper says active / total parameters, look for MoE routing.',
                  'When a paper says MLA, look for compressed KV cache.',
                  'When a paper says thinking budget, look for inference-time compute control.',
                  'When a paper says omni, look for modality encoders, shared thinker, and output decoders.',
                  'When a paper says long context, ask how memory, position, and distractors are handled.',
                  'When a paper says diffusion language model, ask how masked tokens are refined over steps.',
                ].map((item) => (
                  <div key={item} className="rounded-lg border border-slate-200 bg-white p-3 text-sm font-bold leading-6 text-slate-800">
                    {item}
                  </div>
                ))}
              </div>
            </SectionCard>

          </div>
        </div>
      </div>
    </div>
  );
}
