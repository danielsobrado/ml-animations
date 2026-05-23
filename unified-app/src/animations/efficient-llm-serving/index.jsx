import React, { useMemo, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Boxes,
  BrainCircuit,
  CheckCircle2,
  Clock,
  Cpu,
  Database,
  Gauge,
  GitBranch,
  Layers,
  MemoryStick,
  Network,
  PackageCheck,
  Radio,
  Repeat2,
  Route,
  Server,
  Share2,
  SplitSquareHorizontal,
  Timer,
  Zap,
} from 'lucide-react';
import {
  PAPER_CARDS,
  REQUESTS,
  SERVING_FAILURES,
  SERVING_TABS,
  SERVING_TECHNIQUES,
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

function RequestCard({ request, mode }) {
  const longPrompt = request.prompt > 8000;
  const longOutput = request.output > 500;
  return (
    <div className={`rounded border p-3 ${longPrompt ? 'border-[var(--ds-warn)] bg-[var(--ds-paper)]' : 'border-[var(--ds-rule)] bg-[var(--ds-paper-2)]'}`}>
      <div className="mb-2 flex items-center justify-between">
        <span className="font-mono text-xs font-bold text-[var(--ds-ink)]">{request.id}</span>
        <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)]">{request.prefix ? 'shared prefix' : mode}</span>
      </div>
      <div className="space-y-1">
        <MetricBar label="Prompt" value={Math.min(100, request.prompt / 320)} />
        <MetricBar label="Output" value={Math.min(100, request.output / 12)} tone={longOutput ? 'bg-[var(--ds-warn)]' : 'bg-[var(--ds-accent-2)]'} />
      </div>
    </div>
  );
}

function RequestQueue({ mode }) {
  return (
    <div className="grid gap-2 md:grid-cols-3">
      {REQUESTS.map((request) => (
        <RequestCard key={request.id} request={request} mode={mode} />
      ))}
    </div>
  );
}

function KVBlocks({ allocationMode, blockSize, kvPrecision }) {
  const blocks = Array.from({ length: 36 });
  return (
    <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
      <div className="mb-3 flex items-center justify-between gap-2">
        <span className="text-sm font-bold text-[var(--ds-ink)]">{allocationMode === 'paged' ? 'Paged KV blocks' : 'Contiguous reservations'}</span>
        <span className="font-mono text-xs text-[var(--ds-faint)]">{blockSize} token blocks / {kvPrecision}</span>
      </div>
      <div className="grid grid-cols-9 gap-1">
        {blocks.map((_, index) => {
          const active = allocationMode === 'paged' ? index % 5 !== 0 : index < 25 && index % 7 !== 0;
          const wasted = allocationMode === 'contiguous' && index % 7 === 0;
          return (
            <span
              key={index}
              className={`h-6 rounded-sm border ${
                active
                  ? 'border-[var(--ds-accent)] bg-[var(--ds-accent)]'
                  : wasted
                    ? 'border-[var(--ds-warn)] bg-[var(--ds-warn)]/30'
                    : 'border-[var(--ds-rule)] bg-[var(--ds-paper-2)]'
              }`}
            />
          );
        })}
      </div>
    </div>
  );
}

function BatchTimeline({ batchingMode }) {
  const rows = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6'];
  return (
    <div className="space-y-2">
      {rows.map((row, rowIndex) => (
        <div key={row} className="grid grid-cols-[36px_1fr] items-center gap-2">
          <span className="font-mono text-xs text-[var(--ds-faint)]">{row}</span>
          <div className="grid grid-cols-12 gap-1">
            {Array.from({ length: 12 }).map((_, index) => {
              const active = batchingMode === 'continuous'
                ? index >= rowIndex && index < rowIndex + 6
                : index < 8 && rowIndex < 4;
              return <span key={index} className={`h-4 rounded-sm ${active ? 'bg-[var(--ds-accent)]' : 'bg-[var(--ds-rule)]'}`} />;
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

function SpeculationTree({ draftLength, acceptance }) {
  return (
    <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
      <div className="mb-3 text-sm font-bold text-[var(--ds-ink)]">Draft / verify path</div>
      <div className="flex flex-wrap gap-2">
        {Array.from({ length: draftLength }).map((_, index) => {
          const accepted = index < Math.round((acceptance / 100) * draftLength);
          return (
            <div key={index} className={`rounded border px-3 py-2 font-mono text-xs ${accepted ? 'border-[var(--ds-success)] bg-[var(--ds-success)] text-[var(--ds-paper)]' : 'border-[var(--ds-warn)] bg-[var(--ds-paper-2)] text-[var(--ds-ink)]'}`}>
              t+{index + 1}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ServingWorkbench({
  trafficPattern,
  setTrafficPattern,
  modelSize,
  setModelSize,
  servingMode,
  setServingMode,
  maxBatchTokens,
  setMaxBatchTokens,
  sloMs,
  setSloMs,
}) {
  const fullStack = servingMode === 'full-stack';
  const utilization = clamp(36 + (servingMode !== 'naive' ? 24 : 0) + (fullStack ? 24 : 0) + maxBatchTokens / 4096);
  const kvMemory = modelSize === '70B' ? 86 : modelSize === 'MoE' ? 72 : 48;
  const p99 = Math.max(420, 4200 - utilization * 22 + (trafficPattern === 'long-prompt-heavy' ? 1100 : 0));
  const goodput = clamp(utilization - Math.max(0, (p99 - sloMs) / 120));

  return (
    <section className="grid gap-4 lg:grid-cols-[1.08fr_0.92fr]">
      <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
        <div className="mb-4 flex items-center gap-2">
          <Server className="h-5 w-5 text-[var(--ds-accent)]" />
          <h2 className="text-lg font-bold text-[var(--ds-ink)]">20-Request Serving Engine</h2>
        </div>
        <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
          <p className="text-xs font-bold uppercase tracking-wide text-[var(--ds-faint)]">Incoming traffic</p>
          <p className="mt-2 text-sm leading-6 text-[var(--ds-muted)]">
            Mixed prompts, shared prefixes, long generations, and long-context requests enter a scheduler that must balance prefill, decode, KV memory, speculation, and latency SLOs.
          </p>
        </div>
        <div className="mt-4 grid gap-2 md:grid-cols-5">
          {['naive', 'continuous-batching', 'paged-kv', 'speculative', 'full-stack'].map((mode) => (
            <ControlButton key={mode} active={servingMode === mode} onClick={() => setServingMode(mode)}>{mode}</ControlButton>
          ))}
        </div>
        <div className="mt-4 grid gap-2 md:grid-cols-5">
          {['steady', 'bursty', 'chat-heavy', 'long-prompt-heavy', 'mixed'].map((pattern) => (
            <ControlButton key={pattern} active={trafficPattern === pattern} onClick={() => setTrafficPattern(pattern)}>{pattern}</ControlButton>
          ))}
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          <label className="space-y-2 text-xs font-bold text-[var(--ds-ink)]">
            Max batch tokens: {maxBatchTokens}
            <input type="range" min="512" max="65536" step="512" value={maxBatchTokens} onChange={(event) => setMaxBatchTokens(Number(event.target.value))} className="w-full accent-[var(--ds-accent)]" />
          </label>
          <label className="space-y-2 text-xs font-bold text-[var(--ds-ink)]">
            SLO: {sloMs} ms
            <input type="range" min="500" max="10000" step="500" value={sloMs} onChange={(event) => setSloMs(Number(event.target.value))} className="w-full accent-[var(--ds-accent)]" />
          </label>
        </div>
        <div className="mt-4 grid gap-2 md:grid-cols-3">
          {['7B', '13B', '70B', 'MoE'].map((size) => (
            <ControlButton key={size} active={modelSize === size} onClick={() => setModelSize(size)}>{size}</ControlButton>
          ))}
        </div>
      </div>
      <div className="space-y-4">
        <div className="grid gap-3 md:grid-cols-2">
          <MetricTile label="GPU utilization" value={`${utilization}%`} icon={Gauge} />
          <MetricTile label="KV memory" value={`${kvMemory}%`} icon={MemoryStick} />
          <MetricTile label="P99 latency" value={`${Math.round(p99)} ms`} icon={Clock} />
          <MetricTile label="Goodput" value={`${goodput}%`} icon={PackageCheck} />
        </div>
        <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
          <h3 className="mb-3 text-sm font-bold text-[var(--ds-ink)]">Technique stack</h3>
          <div className="grid gap-2">
            {Object.values(SERVING_TECHNIQUES).slice(0, 5).map((technique) => (
              <div key={technique.label} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3">
                <div className="text-xs font-bold text-[var(--ds-ink)]">{technique.label}</div>
                <div className="text-xs text-[var(--ds-muted)]">{technique.bottleneck}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function TabPanel({ tab, state }) {
  const {
    promptLength,
    setPromptLength,
    outputLength,
    setOutputLength,
    batchingMode,
    setBatchingMode,
    allocationMode,
    setAllocationMode,
    blockSize,
    setBlockSize,
    prefixLength,
    setPrefixLength,
    cacheHitRate,
    setCacheHitRate,
    chunkSize,
    setChunkSize,
    draftLength,
    setDraftLength,
    draftQuality,
    setDraftQuality,
    medusaHeads,
    setMedusaHeads,
    quantPrecision,
    setQuantPrecision,
    kvPrecision,
    setKvPrecision,
    parallelism,
    setParallelism,
    policy,
    setPolicy,
    sloMs,
  } = state;

  if (tab === 'serving-map') {
    const stages = ['Requests', 'Scheduler', 'Tokenizer', 'Prefill', 'KV allocation', 'Decode loop', 'Speculation', 'Streaming', 'SLO monitor'];
    return (
      <PanelFrame
        title="LLM Serving Map"
        icon={Route}
        aside={(
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-1">
            <MetricTile label="Requests queued" value="20" icon={Activity} />
            <MetricTile label="Running batches" value="4" icon={Layers} />
            <MetricTile label="SLO risk" value="medium" icon={AlertTriangle} />
          </div>
        )}
      >
        <div className="grid gap-3 md:grid-cols-3">
          {stages.map((stage, index) => (
            <div key={stage} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3">
              <div className="mb-2 font-mono text-xs text-[var(--ds-faint)]">stage {index + 1}</div>
              <div className="text-sm font-bold text-[var(--ds-ink)]">{stage}</div>
            </div>
          ))}
        </div>
        <div className="mt-4">
          <RequestQueue mode="serving" />
        </div>
      </PanelFrame>
    );
  }

  if (tab === 'prefill-decode') {
    const prefill = Math.round(promptLength / 48);
    const decode = Math.round(outputLength * 4.8);
    return (
      <PanelFrame
        title="Prefill vs Decode"
        icon={Timer}
        aside={(
          <div className="space-y-3">
            <MetricTile label="Prefill time" value={`${prefill} ms`} icon={Cpu} />
            <MetricTile label="Decode time" value={`${decode} ms`} icon={Repeat2} />
            <MetricTile label="TTFT" value={`${prefill + 70} ms`} icon={Clock} />
            <MetricTile label="TPOT" value="4.8 ms" icon={Gauge} />
          </div>
        )}
      >
        <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
          <MetricBar label="Prefill compute pressure" value={Math.min(100, promptLength / 320)} />
          <div className="mt-4">
            <MetricBar label="Decode serial pressure" value={Math.min(100, outputLength / 10)} tone="bg-[var(--ds-accent-2)]" />
          </div>
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          <label className="space-y-2 text-xs font-bold text-[var(--ds-ink)]">
            Prompt length: {promptLength}
            <input type="range" min="16" max="32000" step="16" value={promptLength} onChange={(event) => setPromptLength(Number(event.target.value))} className="w-full accent-[var(--ds-accent)]" />
          </label>
          <label className="space-y-2 text-xs font-bold text-[var(--ds-ink)]">
            Output length: {outputLength}
            <input type="range" min="16" max="1024" step="8" value={outputLength} onChange={(event) => setOutputLength(Number(event.target.value))} className="w-full accent-[var(--ds-accent)]" />
          </label>
        </div>
      </PanelFrame>
    );
  }

  if (tab === 'continuous-batching') {
    return (
      <PanelFrame
        title="Continuous Batching"
        icon={Activity}
        aside={(
          <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
            <h3 className="mb-3 text-sm font-bold text-[var(--ds-ink)]">Batching mode</h3>
            <div className="grid gap-2">
              {['static', 'continuous'].map((mode) => (
                <ControlButton key={mode} active={batchingMode === mode} onClick={() => setBatchingMode(mode)}>{mode}</ControlButton>
              ))}
            </div>
            <div className="mt-4 space-y-3">
              <MetricBar label="GPU occupancy" value={batchingMode === 'continuous' ? 86 : 52} />
              <MetricBar label="P95 stability" value={batchingMode === 'continuous' ? 72 : 58} tone="bg-[var(--ds-accent-2)]" />
            </div>
          </div>
        )}
      >
        <BatchTimeline batchingMode={batchingMode} />
      </PanelFrame>
    );
  }

  if (tab === 'paged-attention') {
    return (
      <PanelFrame
        title="PagedAttention and Paged KV Cache"
        icon={Database}
        aside={(
          <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
            <h3 className="mb-3 text-sm font-bold text-[var(--ds-ink)]">Allocator</h3>
            <div className="grid gap-2">
              {['contiguous', 'paged'].map((mode) => (
                <ControlButton key={mode} active={allocationMode === mode} onClick={() => setAllocationMode(mode)}>{mode}</ControlButton>
              ))}
            </div>
            <label className="mt-4 block space-y-2 text-xs font-bold text-[var(--ds-ink)]">
              KV block size: {blockSize}
              <input type="range" min="8" max="64" step="8" value={blockSize} onChange={(event) => setBlockSize(Number(event.target.value))} className="w-full accent-[var(--ds-accent)]" />
            </label>
            <div className="mt-4 space-y-3">
              <MetricBar label="KV utilization" value={allocationMode === 'paged' ? 89 : 58} />
              <MetricBar label="Fragmentation" value={allocationMode === 'paged' ? 12 : 42} tone="bg-[var(--ds-warn)]" />
            </div>
          </div>
        )}
      >
        <KVBlocks allocationMode={allocationMode} blockSize={blockSize} kvPrecision={kvPrecision} />
        <p className="mt-4 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper-2)] p-3 text-sm text-[var(--ds-muted)]">
          PagedAttention reduces KV waste; it does not make KV cache free.
        </p>
      </PanelFrame>
    );
  }

  if (tab === 'prefix-caching') {
    const saved = Math.round(prefixLength * 16 * cacheHitRate);
    return (
      <PanelFrame
        title="Prefix Caching"
        icon={Share2}
        aside={(
          <div className="space-y-3">
            <MetricTile label="Prefill saved" value={`${saved} tokens`} icon={CheckCircle2} />
            <MetricTile label="Hit rate" value={`${Math.round(cacheHitRate * 100)}%`} icon={Gauge} />
            <MetricTile label="Eviction risk" value={cacheHitRate > 0.7 ? 'medium' : 'low'} icon={AlertTriangle} />
          </div>
        )}
      >
        <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
          <div className="mb-4 grid gap-2 md:grid-cols-3">
            {['system prompt', 'policy', 'user suffix'].map((part, index) => (
              <div key={part} className={`rounded border p-3 text-xs font-bold ${index < 2 ? 'border-[var(--ds-accent)] bg-[var(--ds-accent)] text-[var(--ds-paper)]' : 'border-[var(--ds-rule)] bg-[var(--ds-paper-2)] text-[var(--ds-ink)]'}`}>{part}</div>
            ))}
          </div>
          <label className="block space-y-2 text-xs font-bold text-[var(--ds-ink)]">
            Shared prefix length: {prefixLength}
            <input type="range" min="0" max="8192" step="128" value={prefixLength} onChange={(event) => setPrefixLength(Number(event.target.value))} className="w-full accent-[var(--ds-accent)]" />
          </label>
          <label className="mt-4 block space-y-2 text-xs font-bold text-[var(--ds-ink)]">
            Cache hit rate: {cacheHitRate.toFixed(2)}
            <input type="range" min="0" max="0.9" step="0.05" value={cacheHitRate} onChange={(event) => setCacheHitRate(Number(event.target.value))} className="w-full accent-[var(--ds-accent)]" />
          </label>
        </div>
      </PanelFrame>
    );
  }

  if (tab === 'chunked-prefill') {
    return (
      <PanelFrame
        title="Chunked Prefill"
        icon={SplitSquareHorizontal}
        aside={(
          <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
            <label className="block space-y-2 text-xs font-bold text-[var(--ds-ink)]">
              Chunk size: {chunkSize}
              <input type="range" min="256" max="4096" step="256" value={chunkSize} onChange={(event) => setChunkSize(Number(event.target.value))} className="w-full accent-[var(--ds-accent)]" />
            </label>
            <div className="mt-4 space-y-3">
              <MetricBar label="Decode starvation avoided" value={Math.max(20, 100 - chunkSize / 55)} />
              <MetricBar label="Scheduling overhead" value={Math.max(8, 80 - chunkSize / 64)} tone="bg-[var(--ds-warn)]" />
            </div>
          </div>
        )}
      >
        <div className="space-y-3">
          {['prefill chunk 1 + decodes', 'prefill chunk 2 + decodes', 'prefill chunk 3 + decodes', 'first token ready'].map((step, index) => (
            <div key={step} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3">
              <div className="text-sm font-bold text-[var(--ds-ink)]">{step}</div>
              <div className="mt-2 grid grid-cols-12 gap-1">
                {Array.from({ length: 12 }).map((_, tokenIndex) => (
                  <span key={tokenIndex} className={`h-4 rounded-sm ${tokenIndex < 8 - index ? 'bg-[var(--ds-accent)]' : 'bg-[var(--ds-accent-2)]'}`} />
                ))}
              </div>
            </div>
          ))}
        </div>
      </PanelFrame>
    );
  }

  if (tab === 'speculative-decoding') {
    const acceptance = draftQuality === 'good' ? 78 : draftQuality === 'medium' ? 52 : 24;
    return (
      <PanelFrame
        title="Speculative Decoding"
        icon={Zap}
        aside={(
          <div className="space-y-3">
            <MetricTile label="Acceptance rate" value={`${acceptance}%`} icon={CheckCircle2} />
            <MetricTile label="Tokens / verify" value={(1 + (acceptance / 100) * draftLength).toFixed(1)} icon={Gauge} />
            <MetricTile label="Draft waste" value={`${100 - acceptance}%`} icon={AlertTriangle} />
          </div>
        )}
      >
        <SpeculationTree draftLength={draftLength} acceptance={acceptance} />
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          <label className="space-y-2 text-xs font-bold text-[var(--ds-ink)]">
            Draft length: {draftLength}
            <input type="range" min="1" max="16" step="1" value={draftLength} onChange={(event) => setDraftLength(Number(event.target.value))} className="w-full accent-[var(--ds-accent)]" />
          </label>
          <div className="grid gap-2">
            {['poor', 'medium', 'good'].map((quality) => (
              <ControlButton key={quality} active={draftQuality === quality} onClick={() => setDraftQuality(quality)}>{quality} draft</ControlButton>
            ))}
          </div>
        </div>
      </PanelFrame>
    );
  }

  if (tab === 'medusa') {
    return (
      <PanelFrame
        title="Medusa / Multi-Token Heads"
        icon={BrainCircuit}
        aside={(
          <div className="space-y-3">
            <MetricTile label="Heads" value={medusaHeads} icon={Layers} />
            <MetricTile label="Candidate branches" value={medusaHeads * 2} icon={GitBranch} />
            <MetricTile label="Speedup proxy" value={`${(1 + medusaHeads * 0.34).toFixed(1)}x`} icon={Zap} />
          </div>
        )}
      >
        <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
          <div className="mb-4 rounded border border-[var(--ds-accent)] bg-[var(--ds-paper-2)] p-3 text-sm font-bold text-[var(--ds-ink)]">Target backbone hidden state</div>
          <div className="grid gap-2 md:grid-cols-4">
            {Array.from({ length: medusaHeads }).map((_, index) => (
              <div key={index} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-3 text-xs font-bold text-[var(--ds-ink)]">head +{index + 1}</div>
            ))}
          </div>
        </div>
        <label className="mt-4 block space-y-2 text-xs font-bold text-[var(--ds-ink)]">
          Medusa heads: {medusaHeads}
          <input type="range" min="1" max="8" step="1" value={medusaHeads} onChange={(event) => setMedusaHeads(Number(event.target.value))} className="w-full accent-[var(--ds-accent)]" />
        </label>
      </PanelFrame>
    );
  }

  if (tab === 'eagle') {
    return (
      <PanelFrame
        title="EAGLE-Style Draft Models"
        icon={Network}
        aside={(
          <div className="space-y-3">
            <MetricTile label="Draft accuracy" value="high" icon={CheckCircle2} />
            <MetricTile label="Feature fusion" value="multi-layer" icon={Layers} />
            <MetricTile label="Batch throughput" value="+38%" icon={BarChart3} />
          </div>
        )}
      >
        <div className="grid gap-3 md:grid-cols-5">
          {['L1', 'L2', 'L3', 'L4', 'L5'].map((layer, index) => (
            <div key={layer} className={`rounded border p-4 text-center font-mono text-xs font-bold ${index > 1 ? 'border-[var(--ds-accent)] bg-[var(--ds-paper)] text-[var(--ds-ink)]' : 'border-[var(--ds-rule)] bg-[var(--ds-paper-2)] text-[var(--ds-faint)]'}`}>{layer}</div>
          ))}
        </div>
        <div className="mt-4 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4 text-sm text-[var(--ds-muted)]">
          Multi-layer features feed a stronger token draft path, then the target model verifies candidate continuations.
        </div>
      </PanelFrame>
    );
  }

  if (tab === 'quantization') {
    const bits = { fp16: 16, fp8: 8, int8: 8, int4: 4 }[quantPrecision];
    return (
      <PanelFrame
        title="Quantization"
        icon={Cpu}
        aside={(
          <div className="space-y-3">
            <MetricTile label="Weight memory" value={`${Math.round((bits / 16) * 100)}%`} icon={MemoryStick} />
            <MetricTile label="Batch headroom" value={`${Math.round(100 - (bits / 16) * 50)}%`} icon={Gauge} />
            <MetricTile label="Quality risk" value={bits <= 4 ? 'medium' : 'low'} icon={AlertTriangle} />
          </div>
        )}
      >
        <div className="grid gap-2 md:grid-cols-4">
          {['fp16', 'fp8', 'int8', 'int4'].map((precision) => (
            <ControlButton key={precision} active={quantPrecision === precision} onClick={() => setQuantPrecision(precision)}>{precision}</ControlButton>
          ))}
        </div>
        <div className="mt-4">
          <MetricBar label="Bandwidth saved" value={100 - (bits / 16) * 100} />
        </div>
      </PanelFrame>
    );
  }

  if (tab === 'kv-quantization') {
    const bits = { fp16: 16, int8: 8, int4: 4, int2: 2 }[kvPrecision];
    return (
      <PanelFrame
        title="KV Cache Quantization"
        icon={MemoryStick}
        aside={(
          <div className="space-y-3">
            <MetricTile label="KV memory saved" value={`${100 - Math.round((bits / 16) * 100)}%`} icon={Database} />
            <MetricTile label="Batch size possible" value={`${Math.round(16 / bits)}x`} icon={Boxes} />
            <MetricTile label="Attention error" value={bits <= 2 ? 'high' : bits <= 4 ? 'medium' : 'low'} icon={AlertTriangle} />
          </div>
        )}
      >
        <div className="grid gap-2 md:grid-cols-4">
          {['fp16', 'int8', 'int4', 'int2'].map((precision) => (
            <ControlButton key={precision} active={kvPrecision === precision} onClick={() => setKvPrecision(precision)}>{precision}</ControlButton>
          ))}
        </div>
        <div className="mt-4">
          <KVBlocks allocationMode="paged" blockSize={blockSize} kvPrecision={kvPrecision} />
        </div>
      </PanelFrame>
    );
  }

  if (tab === 'parallelism') {
    const overhead = { 'single-gpu': 4, tensor: 32, pipeline: 38, expert: 45, hybrid: 58 }[parallelism];
    return (
      <PanelFrame
        title="Tensor, Pipeline, and Expert Parallelism"
        icon={Share2}
        aside={(
          <div className="space-y-3">
            <MetricTile label="Communication" value={`${overhead}%`} icon={Radio} />
            <MetricTile label="Model fit" value={parallelism === 'single-gpu' ? 'limited' : 'yes'} icon={PackageCheck} />
            <MetricTile label="Pipeline bubbles" value={parallelism === 'pipeline' || parallelism === 'hybrid' ? 'medium' : 'low'} icon={Activity} />
          </div>
        )}
      >
        <div className="grid gap-2 md:grid-cols-5">
          {['single-gpu', 'tensor', 'pipeline', 'expert', 'hybrid'].map((mode) => (
            <ControlButton key={mode} active={parallelism === mode} onClick={() => setParallelism(mode)}>{mode}</ControlButton>
          ))}
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-4">
          {Array.from({ length: parallelism === 'single-gpu' ? 1 : 4 }).map((_, index) => (
            <div key={index} className="rounded border border-[var(--ds-accent)] bg-[var(--ds-paper)] p-4 text-center font-mono text-xs font-bold text-[var(--ds-ink)]">GPU {index}</div>
          ))}
        </div>
      </PanelFrame>
    );
  }

  if (tab === 'throughput-latency') {
    const rawThroughput = policy === 'max-throughput' ? 92 : policy === 'low-latency' ? 48 : 74;
    const tail = policy === 'max-throughput' ? 78 : policy === 'goodput-slo' ? 28 : 46;
    return (
      <PanelFrame
        title="Throughput vs Latency"
        icon={BarChart3}
        aside={(
          <div className="space-y-3">
            <MetricTile label="Tokens/sec" value={`${rawThroughput}K`} icon={Zap} />
            <MetricTile label="P95 latency" value={`${Math.round(400 + tail * 32)} ms`} icon={Clock} />
            <MetricTile label="Goodput" value={`${policy === 'goodput-slo' ? 86 : 62}%`} icon={PackageCheck} />
          </div>
        )}
      >
        <div className="grid gap-2 md:grid-cols-4">
          {['low-latency', 'max-throughput', 'balanced', 'goodput-slo'].map((nextPolicy) => (
            <ControlButton key={nextPolicy} active={policy === nextPolicy} onClick={() => setPolicy(nextPolicy)}>{nextPolicy}</ControlButton>
          ))}
        </div>
        <div className="mt-4 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
          <MetricBar label="Raw throughput" value={rawThroughput} />
          <div className="mt-3">
            <MetricBar label="Tail latency risk" value={tail} tone="bg-[var(--ds-warn)]" />
          </div>
          <div className="mt-3">
            <MetricBar label={`SLO satisfaction (${sloMs} ms)`} value={policy === 'goodput-slo' ? 90 : 62} tone="bg-[var(--ds-success)]" />
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
          <p className="text-sm text-[var(--ds-muted)]">Map each serving paper to the bottleneck it attacks: KV memory, prefill/decode scheduling, serial decoding, quantization, or SLO-aware goodput.</p>
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

export default function EfficientLLMServing() {
  const [activeTab, setActiveTab] = useState(SERVING_TABS[0].id);
  const [trafficPattern, setTrafficPattern] = useState('mixed');
  const [modelSize, setModelSize] = useState('70B');
  const [servingMode, setServingMode] = useState('full-stack');
  const [maxBatchTokens, setMaxBatchTokens] = useState(16384);
  const [sloMs, setSloMs] = useState(2000);
  const [promptLength, setPromptLength] = useState(4096);
  const [outputLength, setOutputLength] = useState(256);
  const [batchingMode, setBatchingMode] = useState('continuous');
  const [allocationMode, setAllocationMode] = useState('paged');
  const [blockSize, setBlockSize] = useState(16);
  const [prefixLength, setPrefixLength] = useState(1024);
  const [cacheHitRate, setCacheHitRate] = useState(0.5);
  const [chunkSize, setChunkSize] = useState(1024);
  const [draftLength, setDraftLength] = useState(4);
  const [draftQuality, setDraftQuality] = useState('medium');
  const [medusaHeads, setMedusaHeads] = useState(4);
  const [quantPrecision, setQuantPrecision] = useState('int8');
  const [kvPrecision, setKvPrecision] = useState('int4');
  const [parallelism, setParallelism] = useState('tensor');
  const [policy, setPolicy] = useState('balanced');

  const state = useMemo(() => ({
    promptLength,
    setPromptLength,
    outputLength,
    setOutputLength,
    batchingMode,
    setBatchingMode,
    allocationMode,
    setAllocationMode,
    blockSize,
    setBlockSize,
    prefixLength,
    setPrefixLength,
    cacheHitRate,
    setCacheHitRate,
    chunkSize,
    setChunkSize,
    draftLength,
    setDraftLength,
    draftQuality,
    setDraftQuality,
    medusaHeads,
    setMedusaHeads,
    quantPrecision,
    setQuantPrecision,
    kvPrecision,
    setKvPrecision,
    parallelism,
    setParallelism,
    policy,
    setPolicy,
    sloMs,
  }), [allocationMode, batchingMode, blockSize, cacheHitRate, chunkSize, draftLength, draftQuality, kvPrecision, medusaHeads, outputLength, parallelism, policy, prefixLength, promptLength, quantPrecision, sloMs]);

  return (
    <main className="min-h-screen bg-[var(--ds-bg)] text-[var(--ds-ink)]">
      <div className="mx-auto max-w-7xl px-4 py-8">
        <header className="mb-6">
          <div className="mb-3 inline-flex items-center gap-2 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] px-3 py-1 text-xs font-bold uppercase tracking-wide text-[var(--ds-faint)]">
            <Server className="h-4 w-4 text-[var(--ds-accent)]" />
            Frontier LLMs
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-[var(--ds-ink)] md:text-5xl">Efficient LLM Serving</h1>
          <p className="mt-3 max-w-3xl text-base leading-7 text-[var(--ds-muted)]">
            How production systems batch requests, manage KV memory, accelerate decoding, quantize cache, distribute models, and trade throughput against latency.
          </p>
        </header>

        <ServingWorkbench
          trafficPattern={trafficPattern}
          setTrafficPattern={setTrafficPattern}
          modelSize={modelSize}
          setModelSize={setModelSize}
          servingMode={servingMode}
          setServingMode={setServingMode}
          maxBatchTokens={maxBatchTokens}
          setMaxBatchTokens={setMaxBatchTokens}
          sloMs={sloMs}
          setSloMs={setSloMs}
        />

        <nav className="my-6 flex gap-2 overflow-x-auto pb-2" aria-label="Efficient LLM serving lesson tabs">
          {SERVING_TABS.map((tab) => (
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
            <h2 className="mb-2 flex items-center gap-2 text-sm font-bold text-[var(--ds-ink)]"><MemoryStick className="h-4 w-4 text-[var(--ds-accent)]" /> KV formula</h2>
            <p className="font-mono text-xs leading-6 text-[var(--ds-muted)]">KV bytes = layers x tokens x kv_heads x head_dim x 2 x bytes</p>
          </div>
          <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
            <h2 className="mb-2 flex items-center gap-2 text-sm font-bold text-[var(--ds-ink)]"><Gauge className="h-4 w-4 text-[var(--ds-accent)]" /> Derived metrics</h2>
            <p className="text-xs leading-6 text-[var(--ds-muted)]">Goodput = requests completed within SLO per second. Fragmentation = 1 - used KV / allocated KV.</p>
          </div>
          <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
            <h2 className="mb-2 flex items-center gap-2 text-sm font-bold text-[var(--ds-ink)]"><AlertTriangle className="h-4 w-4 text-[var(--ds-accent)]" /> Misconception</h2>
            <p className="text-xs leading-6 text-[var(--ds-muted)]">LLM serving is not just batching more. The scheduler must balance throughput, TTFT, TPOT, KV memory, and tail latency.</p>
          </div>
        </section>
      </div>
    </main>
  );
}
