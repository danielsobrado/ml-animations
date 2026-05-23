import React, { useMemo, useState } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  Blocks,
  Brain,
  CheckCircle2,
  Clock,
  Edit3,
  Gauge,
  GitBranch,
  Layers,
  Lock,
  Repeat2,
  ScanLine,
  Shuffle,
  Sparkles,
  Timer,
  Unlock,
  Zap,
} from 'lucide-react';
import {
  DIFFUSION_LM_FAILURES,
  DIFFUSION_TABS,
  GENERATION_MODES,
  PAPER_CARDS,
  SAMPLE_TOKENS,
} from './data';

function clamp(value) {
  return Math.max(0, Math.min(100, Math.round(value)));
}

function MetricBar({ label, value, tone = 'bg-[var(--ds-accent)]' }) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between gap-3 text-xs">
        <span className="font-semibold text-[var(--ds-ink)]">{label}</span>
        <span className="font-mono text-[var(--ds-faint)]">{clamp(value)}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded border border-[var(--ds-rule)] bg-[var(--ds-paper-2)]">
        <div className={`h-full ${tone}`} style={{ width: `${clamp(value)}%` }} />
      </div>
    </div>
  );
}

function MetricTile({ label, value, icon: Icon }) {
  return (
    <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-3">
      <div className="mb-2 flex items-center gap-2 text-[var(--ds-faint)]">
        <Icon className="h-4 w-4" />
        <span className="text-xs font-bold uppercase tracking-wide">{label}</span>
      </div>
      <div className="font-mono text-lg font-bold text-[var(--ds-ink)]">{value}</div>
    </div>
  );
}

function ControlButton({ active, children, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded border px-3 py-2 text-left text-xs font-bold transition ${
        active
          ? 'border-[var(--ds-accent)] bg-[var(--ds-accent)] text-[var(--ds-paper)]'
          : 'border-[var(--ds-rule)] bg-[var(--ds-paper)] text-[var(--ds-ink)] hover:border-[var(--ds-accent)]'
      }`}
    >
      {children}
    </button>
  );
}

function PanelFrame({ title, icon: Icon, children, aside }) {
  return (
    <section className="grid gap-4 lg:grid-cols-[1.08fr_0.92fr]">
      <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
        <div className="mb-4 flex items-center gap-2">
          <Icon className="h-5 w-5 text-[var(--ds-accent)]" />
          <h2 className="text-lg font-bold text-[var(--ds-ink)]">{title}</h2>
        </div>
        {children}
      </div>
      <div className="space-y-4">{aside}</div>
    </section>
  );
}

function Token({ text, state = 'mask', confidence = 0, compact = false }) {
  const styles = {
    locked: 'border-[var(--ds-accent)] bg-[var(--ds-accent)] text-[var(--ds-paper)]',
    candidate: 'border-[var(--ds-accent-2)] bg-[var(--ds-paper)] text-[var(--ds-ink)]',
    mask: 'border-[var(--ds-rule)] bg-[var(--ds-paper-2)] text-[var(--ds-faint)]',
    revised: 'border-[var(--ds-warn)] bg-[var(--ds-paper)] text-[var(--ds-ink)]',
  };
  return (
    <div className={`rounded border px-2 py-2 text-center font-mono text-xs ${styles[state]} ${compact ? 'min-w-0' : 'min-w-[76px]'}`}>
      <div className="truncate">{state === 'mask' ? '[MASK]' : text}</div>
      {confidence > 0 && <div className="mt-1 text-[10px] opacity-75">{confidence.toFixed(2)}</div>}
    </div>
  );
}

function DiffusionSequence({ step, threshold, allowRemasking }) {
  return (
    <div className="grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-6">
      {SAMPLE_TOKENS.map((token, index) => {
        const confidence = Math.min(0.96, 0.38 + step * 0.09 + ((index % 4) * 0.05));
        const revealed = index < step * 2 || confidence >= threshold;
        const revised = allowRemasking && step > 2 && index === 3;
        return (
          <Token
            key={`${token}-${index}`}
            text={revised ? 'produce' : token}
            state={revealed ? (revised ? 'revised' : 'locked') : 'mask'}
            confidence={revealed ? confidence : 0}
          />
        );
      })}
    </div>
  );
}

function Timeline({ mode, sequenceLength, diffusionSteps }) {
  const arTicks = Math.min(32, sequenceLength);
  const diffTicks = diffusionSteps;
  const ticks = mode === 'autoregressive' ? arTicks : diffTicks;
  return (
    <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <span className="text-sm font-bold text-[var(--ds-ink)]">{GENERATION_MODES[mode].label} timeline</span>
        <span className="font-mono text-xs text-[var(--ds-faint)]">{ticks} sequential passes</span>
      </div>
      <div className="grid grid-cols-8 gap-1 md:grid-cols-16">
        {Array.from({ length: Math.min(32, ticks) }).map((_, index) => (
          <span
            key={index}
            className={`h-4 rounded-sm ${mode === 'autoregressive' ? 'bg-[var(--ds-accent-2)]' : 'bg-[var(--ds-accent)]'}`}
            title={`step ${index + 1}`}
          />
        ))}
      </div>
    </div>
  );
}

function MaskCorruption({ timestep, corruptionType }) {
  const clean = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'warm'];
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-5 gap-2 md:grid-cols-10">
        {clean.map((token, index) => {
          const masked = index / clean.length < timestep || (corruptionType === 'random-token-replace' && index % 4 === 1);
          return (
            <Token
              key={index}
              text={corruptionType === 'random-token-replace' && masked ? 'table' : token}
              state={masked ? (corruptionType === 'random-token-replace' ? 'revised' : 'mask') : 'locked'}
              compact
            />
          );
        })}
      </div>
      <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper-2)] p-3 text-sm text-[var(--ds-muted)]">
        q(x_t | x_0) {corruptionType === 'masking' ? 'replaces selected tokens with [MASK]' : 'corrupts selected token states'} at timestep {timestep.toFixed(2)}.
      </div>
    </div>
  );
}

function BlockStrip({ blockSize, blockSteps }) {
  const blocks = [0, 1, 2, 3];
  return (
    <div className="space-y-3">
      {blocks.map((block) => (
        <div key={block} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3">
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="font-bold text-[var(--ds-ink)]">Block {block + 1}</span>
            <span className="font-mono text-[var(--ds-faint)]">{blockSize} tokens / {blockSteps} denoise steps</span>
          </div>
          <div className="grid grid-cols-8 gap-1">
            {Array.from({ length: 8 }).map((_, index) => (
              <span
                key={index}
                className={`h-4 rounded-sm ${block === 0 || index < blockSteps / 2 ? 'bg-[var(--ds-accent)]' : 'bg-[var(--ds-rule)]'}`}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function Workbench({
  mode,
  setMode,
  sequenceLength,
  setSequenceLength,
  diffusionSteps,
  setDiffusionSteps,
  threshold,
  setThreshold,
  allowRemasking,
  setAllowRemasking,
}) {
  const forwardPasses = mode === 'autoregressive' ? sequenceLength : diffusionSteps;
  const updatesPerPass = mode === 'autoregressive' ? 1 : Math.round(sequenceLength / diffusionSteps);
  const modeInfo = GENERATION_MODES[mode];
  return (
    <section className="grid gap-4 lg:grid-cols-[1.06fr_0.94fr]">
      <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
        <div className="mb-4 flex items-center gap-2">
          <Shuffle className="h-5 w-5 text-[var(--ds-accent)]" />
          <h2 className="text-lg font-bold text-[var(--ds-ink)]">Masked Token Diffusion Workbench</h2>
        </div>
        <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
          <p className="text-xs font-bold uppercase tracking-wide text-[var(--ds-faint)]">Prompt</p>
          <p className="mt-2 text-base font-semibold text-[var(--ds-ink)]">
            Explain why diffusion language models differ from autoregressive models.
          </p>
        </div>
        <div className="mt-4 grid gap-2 md:grid-cols-3">
          {Object.entries(GENERATION_MODES).map(([key, value]) => (
            <ControlButton key={key} active={mode === key} onClick={() => setMode(key)}>
              {value.shortLabel}
            </ControlButton>
          ))}
        </div>
        <div className="mt-4">
          <DiffusionSequence step={Math.max(1, Math.round(diffusionSteps / 4))} threshold={threshold} allowRemasking={allowRemasking} />
        </div>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <label className="space-y-2 text-xs font-bold text-[var(--ds-ink)]">
            Sequence length: {sequenceLength}
            <input type="range" min="16" max="256" step="16" value={sequenceLength} onChange={(event) => setSequenceLength(Number(event.target.value))} className="w-full accent-[var(--ds-accent)]" />
          </label>
          <label className="space-y-2 text-xs font-bold text-[var(--ds-ink)]">
            Diffusion steps: {diffusionSteps}
            <input type="range" min="2" max="32" step="2" value={diffusionSteps} onChange={(event) => setDiffusionSteps(Number(event.target.value))} className="w-full accent-[var(--ds-accent)]" />
          </label>
          <label className="space-y-2 text-xs font-bold text-[var(--ds-ink)]">
            Confidence threshold: {threshold.toFixed(2)}
            <input type="range" min="0.3" max="0.9" step="0.1" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} className="w-full accent-[var(--ds-accent)]" />
          </label>
          <button
            type="button"
            onClick={() => setAllowRemasking((value) => !value)}
            className={`rounded border px-3 py-2 text-xs font-bold ${allowRemasking ? 'border-[var(--ds-accent)] bg-[var(--ds-accent)] text-[var(--ds-paper)]' : 'border-[var(--ds-rule)] bg-[var(--ds-paper)] text-[var(--ds-ink)]'}`}
          >
            {allowRemasking ? 'Remasking allowed' : 'Remasking off'}
          </button>
        </div>
      </div>
      <div className="space-y-4">
        <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
          <h3 className="mb-2 text-sm font-bold text-[var(--ds-ink)]">{modeInfo.label}</h3>
          <p className="text-sm text-[var(--ds-muted)]">{modeInfo.description}</p>
          <div className="mt-4 space-y-3">
            <MetricBar label="Parallelism" value={modeInfo.parallelism} />
            <MetricBar label="Fluency proxy" value={modeInfo.fluency} tone="bg-[var(--ds-accent-2)]" />
            <MetricBar label="Editability" value={modeInfo.editability} tone="bg-[var(--ds-success)]" />
          </div>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          <MetricTile label="Forward passes" value={forwardPasses} icon={Repeat2} />
          <MetricTile label="Updates/pass" value={updatesPerPass} icon={Zap} />
          <MetricTile label="Locked tokens" value={`${clamp(100 - threshold * 70)}%`} icon={Lock} />
          <MetricTile label="Revision risk" value={`${allowRemasking ? 'medium' : 'low'}`} icon={Unlock} />
        </div>
      </div>
    </section>
  );
}

function TabPanel({ tab, state }) {
  const {
    mode,
    sequenceLength,
    diffusionSteps,
    threshold,
    allowRemasking,
    maskSchedule,
    setMaskSchedule,
    corruptionType,
    setCorruptionType,
    timestep,
    setTimestep,
    blockSize,
    setBlockSize,
    blockSteps,
    setBlockSteps,
    conversionStage,
    setConversionStage,
    alignmentStage,
    setAlignmentStage,
  } = state;
  const modeInfo = GENERATION_MODES[mode];

  if (tab === 'ar-vs-diffusion') {
    return (
      <PanelFrame
        title="AR vs Diffusion Language Generation"
        icon={ArrowRight}
        aside={(
          <>
            <Timeline mode="autoregressive" sequenceLength={sequenceLength} diffusionSteps={diffusionSteps} />
            <Timeline mode={mode === 'autoregressive' ? 'fullDiffusion' : mode} sequenceLength={sequenceLength} diffusionSteps={diffusionSteps} />
          </>
        )}
      >
        <div className="grid gap-4 md:grid-cols-2">
          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
            <h3 className="mb-3 text-sm font-bold text-[var(--ds-ink)]">Autoregressive</h3>
            {['The', 'model', 'learns', 'fast'].map((token, index) => (
              <div key={token} className="mb-2 flex items-center gap-2">
                <span className="font-mono text-xs text-[var(--ds-faint)]">step {index + 1}</span>
                <Token text={['The', 'model', 'learns', 'fast'].slice(0, index + 1).join(' ')} state="candidate" />
              </div>
            ))}
          </div>
          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
            <h3 className="mb-3 text-sm font-bold text-[var(--ds-ink)]">Masked diffusion</h3>
            <DiffusionSequence step={3} threshold={threshold} allowRemasking={allowRemasking} />
          </div>
        </div>
        <p className="mt-4 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper-2)] p-3 text-sm text-[var(--ds-muted)]">
          Misconception: diffusion language generation is not randomly filling blanks. It is learned denoising over token distributions.
        </p>
      </PanelFrame>
    );
  }

  if (tab === 'discrete-diffusion') {
    return (
      <PanelFrame
        title="Discrete Diffusion Over Tokens"
        icon={ScanLine}
        aside={(
          <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
            <h3 className="mb-3 text-sm font-bold text-[var(--ds-ink)]">Corruption controls</h3>
            <div className="grid gap-2">
              {['masking', 'random-token-replace', 'absorbing-mask'].map((type) => (
                <ControlButton key={type} active={corruptionType === type} onClick={() => setCorruptionType(type)}>{type}</ControlButton>
              ))}
            </div>
            <label className="mt-4 block space-y-2 text-xs font-bold text-[var(--ds-ink)]">
              Timestep: {timestep.toFixed(2)}
              <input type="range" min="0" max="1" step="0.05" value={timestep} onChange={(event) => setTimestep(Number(event.target.value))} className="w-full accent-[var(--ds-accent)]" />
            </label>
          </div>
        )}
      >
        <MaskCorruption timestep={timestep} corruptionType={corruptionType} />
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <MetricTile label="Masked positions" value={`${Math.round(timestep * 100)}%`} icon={Shuffle} />
          <MetricTile label="Visible context" value={`${Math.round((1 - timestep) * 100)}%`} icon={CheckCircle2} />
          <MetricTile label="Ambiguity" value={timestep > 0.7 ? 'high' : 'medium'} icon={AlertTriangle} />
        </div>
      </PanelFrame>
    );
  }

  if (tab === 'reverse-denoising') {
    return (
      <PanelFrame
        title="Reverse Denoising and Remasking"
        icon={Repeat2}
        aside={(
          <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
            <h3 className="mb-3 text-sm font-bold text-[var(--ds-ink)]">Confidence loop</h3>
            <MetricBar label="Average confidence" value={threshold * 92} />
            <MetricBar label="Remasked tokens" value={allowRemasking ? 34 : 5} tone="bg-[var(--ds-warn)]" />
            <MetricBar label="Remaining uncertainty" value={Math.max(8, 92 - diffusionSteps * 4)} tone="bg-[var(--ds-accent-2)]" />
          </div>
        )}
      >
        {[0, 1, 2, 3].map((step) => (
          <div key={step} className="mb-3 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3">
            <div className="mb-2 font-mono text-xs font-bold text-[var(--ds-faint)]">denoising step {step}</div>
            <DiffusionSequence step={step} threshold={threshold} allowRemasking={allowRemasking} />
          </div>
        ))}
      </PanelFrame>
    );
  }

  if (tab === 'parallel-decoding') {
    const arSteps = sequenceLength;
    const diffusionPasses = diffusionSteps;
    return (
      <PanelFrame
        title="Parallel Decoding"
        icon={Zap}
        aside={(
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-1">
            <MetricTile label="AR sequential steps" value={arSteps} icon={Timer} />
            <MetricTile label="Diffusion passes" value={diffusionPasses} icon={Repeat2} />
            <MetricTile label="Parallelism score" value={`${clamp((sequenceLength / diffusionSteps) * 6)}%`} icon={Gauge} />
          </div>
        )}
      >
        <Timeline mode="autoregressive" sequenceLength={sequenceLength} diffusionSteps={diffusionSteps} />
        <div className="mt-4">
          <Timeline mode="fullDiffusion" sequenceLength={sequenceLength} diffusionSteps={diffusionSteps} />
        </div>
        <p className="mt-4 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper-2)] p-3 text-sm text-[var(--ds-muted)]">
          Parallel decoding is a potential advantage, not a guaranteed speedup. Quality targets, step count, hardware, and confidence policy still decide latency.
        </p>
      </PanelFrame>
    );
  }

  if (tab === 'block-diffusion') {
    return (
      <PanelFrame
        title="Block Diffusion"
        icon={Blocks}
        aside={(
          <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
            <h3 className="mb-3 text-sm font-bold text-[var(--ds-ink)]">Block controls</h3>
            <label className="block space-y-2 text-xs font-bold text-[var(--ds-ink)]">
              Block size: {blockSize}
              <input type="range" min="8" max="128" step="8" value={blockSize} onChange={(event) => setBlockSize(Number(event.target.value))} className="w-full accent-[var(--ds-accent)]" />
            </label>
            <label className="mt-4 block space-y-2 text-xs font-bold text-[var(--ds-ink)]">
              Steps per block: {blockSteps}
              <input type="range" min="2" max="16" step="2" value={blockSteps} onChange={(event) => setBlockSteps(Number(event.target.value))} className="w-full accent-[var(--ds-accent)]" />
            </label>
            <MetricBar label="Flexible length support" value={78} />
            <MetricBar label="Boundary coherence" value={Math.max(35, 95 - blockSize / 2)} tone="bg-[var(--ds-accent-2)]" />
          </div>
        )}
      >
        <BlockStrip blockSize={blockSize} blockSteps={blockSteps} />
      </PanelFrame>
    );
  }

  if (tab === 'conversion') {
    const stages = ['ar-base', 'warm-up-block', 'full-sequence-stable', 'compact-block-decay', 'sft', 'dpo'];
    return (
      <PanelFrame
        title="Conversion from AR Models to Diffusion LMs"
        icon={GitBranch}
        aside={(
          <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
            <h3 className="mb-3 text-sm font-bold text-[var(--ds-ink)]">Stage</h3>
            <div className="grid gap-2">
              {stages.map((stage) => (
                <ControlButton key={stage} active={conversionStage === stage} onClick={() => setConversionStage(stage)}>{stage}</ControlButton>
              ))}
            </div>
          </div>
        )}
      >
        <div className="space-y-3">
          {stages.map((stage, index) => (
            <div key={stage} className={`flex items-center gap-3 rounded border p-3 ${conversionStage === stage ? 'border-[var(--ds-accent)] bg-[var(--ds-paper)]' : 'border-[var(--ds-rule)] bg-[var(--ds-paper-2)]'}`}>
              <div className="flex h-8 w-8 items-center justify-center rounded border border-[var(--ds-rule)] font-mono text-xs">{index + 1}</div>
              <div>
                <div className="text-sm font-bold text-[var(--ds-ink)]">{stage}</div>
                <div className="text-xs text-[var(--ds-muted)]">Adapt masks, objective, schedule, and alignment for denoising generation.</div>
              </div>
            </div>
          ))}
        </div>
      </PanelFrame>
    );
  }

  if (tab === 'alignment') {
    return (
      <PanelFrame
        title="SFT and DPO Alignment"
        icon={Brain}
        aside={(
          <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
            <h3 className="mb-3 text-sm font-bold text-[var(--ds-ink)]">Alignment stage</h3>
            <div className="grid gap-2">
              {['pretrain', 'sft', 'dpo'].map((stage) => (
                <ControlButton key={stage} active={alignmentStage === stage} onClick={() => setAlignmentStage(stage)}>{stage}</ControlButton>
              ))}
            </div>
            <div className="mt-4 space-y-3">
              <MetricBar label="Instruction adherence" value={alignmentStage === 'pretrain' ? 38 : alignmentStage === 'sft' ? 78 : 86} />
              <MetricBar label="Preference alignment" value={alignmentStage === 'dpo' ? 84 : 46} tone="bg-[var(--ds-success)]" />
            </div>
          </div>
        )}
      >
        <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
          <p className="text-xs font-bold uppercase tracking-wide text-[var(--ds-faint)]">Prompt visible</p>
          <p className="mt-2 text-sm font-semibold text-[var(--ds-ink)]">Explain block diffusion in one paragraph.</p>
          <div className="mt-4 grid grid-cols-2 gap-2 md:grid-cols-4">
            {['Block', '[MASK]', 'denoises', '[MASK]', 'inside', 'chunks', 'while', '[MASK]'].map((token, index) => (
              <Token key={index} text={token} state={token === '[MASK]' ? 'mask' : 'candidate'} compact />
            ))}
          </div>
        </div>
        <p className="mt-4 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper-2)] p-3 text-sm text-[var(--ds-muted)]">
          Diffusion LMs still need instruction and preference training. Denoising alone does not make a model aligned.
        </p>
      </PanelFrame>
    );
  }

  if (tab === 'strengths') {
    const tasks = ['continuation', 'infilling', 'editing', 'speech tokens', 'multimodal'];
    return (
      <PanelFrame
        title="Strengths vs Autoregressive LMs"
        icon={Sparkles}
        aside={(
          <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
            <h3 className="mb-3 text-sm font-bold text-[var(--ds-ink)]">{modeInfo.label}</h3>
            <MetricBar label="Global refinement" value={modeInfo.editability} />
            <MetricBar label="Native infilling" value={mode === 'autoregressive' ? 36 : 88} tone="bg-[var(--ds-success)]" />
            <MetricBar label="Serving maturity" value={mode === 'autoregressive' ? 92 : 46} tone="bg-[var(--ds-accent-2)]" />
          </div>
        )}
      >
        <div className="grid gap-3 md:grid-cols-2">
          {tasks.map((task) => (
            <div key={task} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3">
              <div className="mb-2 text-sm font-bold text-[var(--ds-ink)]">{task}</div>
              <p className="text-xs text-[var(--ds-muted)]">Compare fluency, editability, control, and latency before choosing AR, full diffusion, or block diffusion.</p>
            </div>
          ))}
        </div>
      </PanelFrame>
    );
  }

  if (tab === 'failures') {
    return (
      <PanelFrame
        title="Weaknesses and Failure Modes"
        icon={AlertTriangle}
        aside={(
          <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
            <h3 className="mb-3 text-sm font-bold text-[var(--ds-ink)]">Schedule pressure</h3>
            <MetricBar label="Coherence" value={maskSchedule === 'bad-schedule' ? 34 : 78} />
            <MetricBar label="Revision instability" value={maskSchedule === 'confidence-based' ? 28 : 58} tone="bg-[var(--ds-warn)]" />
            <MetricBar label="Latency" value={diffusionSteps * 3} tone="bg-[var(--ds-accent-2)]" />
          </div>
        )}
      >
        <div className="mb-4 grid gap-2 md:grid-cols-4">
          {['linear', 'cosine', 'confidence-based', 'bad-schedule'].map((schedule) => (
            <ControlButton key={schedule} active={maskSchedule === schedule} onClick={() => setMaskSchedule(schedule)}>{schedule}</ControlButton>
          ))}
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          {DIFFUSION_LM_FAILURES.map((failure) => (
            <div key={failure.id} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3">
              <div className="text-sm font-bold text-[var(--ds-ink)]">{failure.label}</div>
              <p className="mt-1 text-xs text-[var(--ds-muted)]">{failure.symptom}</p>
              <p className="mt-2 text-xs font-semibold text-[var(--ds-ink)]">{failure.mitigation}</p>
            </div>
          ))}
        </div>
      </PanelFrame>
    );
  }

  if (tab === 'editing') {
    return (
      <PanelFrame
        title="Diffusion LMs for Editing and Infilling"
        icon={Edit3}
        aside={(
          <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
            <h3 className="mb-3 text-sm font-bold text-[var(--ds-ink)]">Edit quality</h3>
            <MetricBar label="Left context use" value={82} />
            <MetricBar label="Right context use" value={86} tone="bg-[var(--ds-success)]" />
            <MetricBar label="Minimal-change score" value={72} tone="bg-[var(--ds-accent-2)]" />
          </div>
        )}
      >
        <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
          <p className="text-sm leading-7 text-[var(--ds-ink)]">
            The model <span className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper-2)] px-2 py-1 font-mono text-xs">[MASK]</span> tokens in parallel, uses context on both sides, and can <span className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper-2)] px-2 py-1 font-mono text-xs">[MASK]</span> uncertain spans without restarting.
          </p>
          <div className="mt-4 grid grid-cols-2 gap-2 md:grid-cols-4">
            {['fill span', 'replace phrase', 'insert sentence', 'repair code'].map((edit) => (
              <div key={edit} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper-2)] p-3 text-xs font-bold text-[var(--ds-ink)]">{edit}</div>
            ))}
          </div>
        </div>
      </PanelFrame>
    );
  }

  return (
    <PanelFrame
      title="Paper Decoder"
      icon={Layers}
      aside={(
        <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
          <h3 className="mb-2 text-sm font-bold text-[var(--ds-ink)]">Reading frame</h3>
          <p className="text-sm text-[var(--ds-muted)]">Ask what changes in objective, generation order, length handling, alignment, and serving tradeoffs.</p>
        </div>
      )}
    >
      <div className="grid gap-3 md:grid-cols-2">
        {PAPER_CARDS.map((paper) => (
          <div key={paper.id} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
            <h3 className="text-sm font-bold text-[var(--ds-ink)]">{paper.label}</h3>
            <ul className="mt-3 space-y-1 text-xs text-[var(--ds-muted)]">
              {paper.signals.map((signal) => <li key={signal}>- {signal}</li>)}
            </ul>
            <p className="mt-3 text-xs font-semibold text-[var(--ds-ink)]">{paper.interpretation}</p>
          </div>
        ))}
      </div>
    </PanelFrame>
  );
}

export default function DiffusionLanguageModels() {
  const [activeTab, setActiveTab] = useState(DIFFUSION_TABS[0].id);
  const [mode, setMode] = useState('fullDiffusion');
  const [sequenceLength, setSequenceLength] = useState(64);
  const [diffusionSteps, setDiffusionSteps] = useState(8);
  const [threshold, setThreshold] = useState(0.7);
  const [allowRemasking, setAllowRemasking] = useState(true);
  const [maskSchedule, setMaskSchedule] = useState('confidence-based');
  const [corruptionType, setCorruptionType] = useState('masking');
  const [timestep, setTimestep] = useState(0.55);
  const [blockSize, setBlockSize] = useState(16);
  const [blockSteps, setBlockSteps] = useState(6);
  const [conversionStage, setConversionStage] = useState('warm-up-block');
  const [alignmentStage, setAlignmentStage] = useState('sft');

  const state = useMemo(() => ({
    mode,
    sequenceLength,
    diffusionSteps,
    threshold,
    allowRemasking,
    maskSchedule,
    setMaskSchedule,
    corruptionType,
    setCorruptionType,
    timestep,
    setTimestep,
    blockSize,
    setBlockSize,
    blockSteps,
    setBlockSteps,
    conversionStage,
    setConversionStage,
    alignmentStage,
    setAlignmentStage,
  }), [alignmentStage, allowRemasking, blockSize, blockSteps, conversionStage, corruptionType, diffusionSteps, maskSchedule, mode, sequenceLength, threshold, timestep]);

  return (
    <main className="min-h-screen bg-[var(--ds-bg)] text-[var(--ds-ink)]">
      <div className="mx-auto max-w-7xl px-4 py-8">
        <header className="mb-6">
          <div className="mb-3 inline-flex items-center gap-2 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] px-3 py-1 text-xs font-bold uppercase tracking-wide text-[var(--ds-faint)]">
            <Sparkles className="h-4 w-4 text-[var(--ds-accent)]" />
            Frontier LLMs
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-[var(--ds-ink)] md:text-5xl">Diffusion Language Models</h1>
          <p className="mt-3 max-w-3xl text-base leading-7 text-[var(--ds-muted)]">
            How masked-token denoising challenges left-to-right language generation through parallel refinement, remasking, block schedules, editing, and new alignment tradeoffs.
          </p>
        </header>

        <Workbench
          mode={mode}
          setMode={setMode}
          sequenceLength={sequenceLength}
          setSequenceLength={setSequenceLength}
          diffusionSteps={diffusionSteps}
          setDiffusionSteps={setDiffusionSteps}
          threshold={threshold}
          setThreshold={setThreshold}
          allowRemasking={allowRemasking}
          setAllowRemasking={setAllowRemasking}
        />

        <nav className="my-6 flex gap-2 overflow-x-auto pb-2" aria-label="Diffusion language model lesson tabs">
          {DIFFUSION_TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`shrink-0 rounded border px-3 py-2 text-xs font-bold transition ${
                activeTab === tab.id
                  ? 'border-[var(--ds-accent)] bg-[var(--ds-accent)] text-[var(--ds-paper)]'
                  : 'border-[var(--ds-rule)] bg-[var(--ds-paper)] text-[var(--ds-ink)] hover:border-[var(--ds-accent)]'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        <TabPanel tab={activeTab} state={state} />

        <section className="mt-6 grid gap-4 md:grid-cols-3">
          <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
            <h2 className="mb-2 flex items-center gap-2 text-sm font-bold text-[var(--ds-ink)]"><Clock className="h-4 w-4 text-[var(--ds-accent)]" /> Core formula</h2>
            <p className="font-mono text-xs leading-6 text-[var(--ds-muted)]">p(x) = product_t p(x_t | x_&lt;t)</p>
            <p className="font-mono text-xs leading-6 text-[var(--ds-muted)]">q(x_t | x_0) = mask tokens by t</p>
            <p className="font-mono text-xs leading-6 text-[var(--ds-muted)]">p_theta(x_i | x_t,t) for masked i</p>
          </div>
          <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
            <h2 className="mb-2 flex items-center gap-2 text-sm font-bold text-[var(--ds-ink)]"><Gauge className="h-4 w-4 text-[var(--ds-accent)]" /> Derived metrics</h2>
            <p className="text-xs leading-6 text-[var(--ds-muted)]">Parallelism score = average tokens updated per denoising step.</p>
            <p className="text-xs leading-6 text-[var(--ds-muted)]">Completion quality = fluency + coherence + task match - unfilled masks.</p>
          </div>
          <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
            <h2 className="mb-2 flex items-center gap-2 text-sm font-bold text-[var(--ds-ink)]"><AlertTriangle className="h-4 w-4 text-[var(--ds-accent)]" /> Misconception</h2>
            <p className="text-xs leading-6 text-[var(--ds-muted)]">Diffusion LMs are not diffusion pasted onto text. They are discrete token denoising systems with different generation order, revision behavior, and serving tradeoffs.</p>
          </div>
        </section>
      </div>
    </main>
  );
}

