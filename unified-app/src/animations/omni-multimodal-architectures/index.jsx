import React, { useMemo, useState } from 'react';
import {
  AlertTriangle,
  AudioLines,
  Brain,
  CheckCircle2,
  Clock,
  Eye,
  Film,
  Gauge,
  Grid3X3,
  Image as ImageIcon,
  Layers,
  LocateFixed,
  MessageSquare,
  Mic,
  Network,
  Radio,
  ScanSearch,
  Speech,
  Zap,
} from 'lucide-react';
import {
  FAILURE_MODES,
  FUSION_MODES,
  MODALITY_PIPELINES,
  OMNI_TABS,
  PAPER_CARDS,
} from './data';

const modalityKeys = Object.keys(MODALITY_PIPELINES);

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

function ModalityStream({ name, active, tokens, icon: Icon }) {
  return (
    <div
      className={`rounded border p-3 transition ${
        active
          ? 'border-[var(--ds-accent)] bg-[var(--ds-paper)] shadow-sm'
          : 'border-[var(--ds-rule)] bg-[var(--ds-paper-2)] opacity-55'
      }`}
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Icon className="h-4 w-4 text-[var(--ds-accent)]" />
          <span className="text-sm font-bold text-[var(--ds-ink)]">{name}</span>
        </div>
        <span className="font-mono text-xs text-[var(--ds-faint)]">{tokens}</span>
      </div>
      <div className="grid grid-cols-8 gap-1">
        {Array.from({ length: 16 }).map((_, index) => (
          <span
            key={index}
            className={`h-2 rounded-sm ${
              active && index < Math.max(3, Math.min(16, Math.ceil(tokens / 120)))
                ? 'bg-[var(--ds-accent)]'
                : 'bg-[var(--ds-rule)]'
            }`}
          />
        ))}
      </div>
    </div>
  );
}

function PatchGrid({ density = 4, highlight = 8 }) {
  const cells = density * density;
  return (
    <div className="grid aspect-square w-full max-w-[220px] gap-1" style={{ gridTemplateColumns: `repeat(${density}, minmax(0, 1fr))` }}>
      {Array.from({ length: cells }).map((_, index) => (
        <div
          key={index}
          className={`rounded border ${
            index === highlight
              ? 'border-[var(--ds-accent)] bg-[var(--ds-accent)]'
              : index % 5 === 0
                ? 'border-[var(--ds-accent-2)] bg-[var(--ds-accent-2)]/30'
                : 'border-[var(--ds-rule)] bg-[var(--ds-paper-2)]'
          }`}
        />
      ))}
    </div>
  );
}

function LatencyWaterfall({ chunkMs, thinkerDepth, talkerMode, networkMs, strictSafety }) {
  const thinkerMs = { short: 90, medium: 180, long: 360 }[thinkerDepth];
  const talkerMs = talkerMode === 'streaming-codec' ? 234 : 760;
  const safetyMs = strictSafety ? 120 : 40;
  const stages = [
    ['Input chunk', chunkMs],
    ['Encoder', 42],
    ['Thinker', thinkerMs],
    ['Talker start', talkerMs],
    ['Network', networkMs],
    ['Buffer / safety', 80 + safetyMs],
  ];
  const total = stages.reduce((sum, [, ms]) => sum + ms, 0);
  return (
    <div className="space-y-3">
      {stages.map(([label, ms]) => (
        <div key={label}>
          <div className="mb-1 flex justify-between text-xs">
            <span className="font-semibold text-[var(--ds-ink)]">{label}</span>
            <span className="font-mono text-[var(--ds-faint)]">{ms} ms</span>
          </div>
          <div className="h-3 overflow-hidden rounded border border-[var(--ds-rule)] bg-[var(--ds-paper-2)]">
            <div className="h-full bg-[var(--ds-accent)]" style={{ width: `${Math.max(8, (ms / total) * 100)}%` }} />
          </div>
        </div>
      ))}
      <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3 font-mono text-sm font-bold text-[var(--ds-ink)]">
        First-audio latency: {total} ms
      </div>
    </div>
  );
}

function OmniWorkbench({ modalities, setModalities, fusionMode, setFusionMode, outputMode, setOutputMode, latencyTarget, setLatencyTarget }) {
  const activeKeys = modalityKeys.filter((key) => modalities[key]);
  const tokens = activeKeys.reduce((sum, key) => sum + MODALITY_PIPELINES[key].rate * (key === 'video' ? 80 : key === 'audio' ? 55 : 40), 0);
  const fusion = FUSION_MODES[fusionMode];
  const speech = outputMode !== 'text';
  const latencyPenalty = latencyTarget === 'real-time' ? 25 : latencyTarget === 'interactive' ? 10 : 0;
  const grounding = clamp(44 + activeKeys.length * 9 + (fusionMode === 'early' ? 12 : 0) - (modalities.video ? 8 : 0));

  return (
    <section className="grid gap-4 lg:grid-cols-[1.05fr_0.95fr]">
      <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
        <div className="mb-4 flex items-center gap-2">
          <Network className="h-5 w-5 text-[var(--ds-accent)]" />
          <h2 className="text-lg font-bold text-[var(--ds-ink)]">Omni Token Stream Workbench</h2>
        </div>
        <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
          <p className="text-xs font-bold uppercase tracking-wide text-[var(--ds-faint)]">Multimodal prompt</p>
          <p className="mt-2 text-base font-semibold leading-6 text-[var(--ds-ink)]">
            Explain what happens in this chart screenshot and short clip, while listening to the spoken follow-up, then answer out loud.
          </p>
        </div>

        <div className="mt-4 grid gap-2 md:grid-cols-4">
          {[
            ['text', MessageSquare],
            ['image', ImageIcon],
            ['video', Film],
            ['audio', AudioLines],
          ].map(([key, Icon]) => (
            <button
              key={key}
              type="button"
              onClick={() => setModalities((current) => ({ ...current, [key]: !current[key] }))}
              className={`rounded border p-3 text-left transition ${
                modalities[key]
                  ? 'border-[var(--ds-accent)] bg-[var(--ds-accent)] text-[var(--ds-paper)]'
                  : 'border-[var(--ds-rule)] bg-[var(--ds-paper)] text-[var(--ds-ink)]'
              }`}
            >
              <Icon className="mb-2 h-4 w-4" />
              <span className="text-sm font-bold capitalize">{key}</span>
            </button>
          ))}
        </div>

        <div className="mt-4 grid gap-2 md:grid-cols-4">
          {Object.entries(FUSION_MODES).map(([key, mode]) => (
            <button
              key={key}
              type="button"
              onClick={() => setFusionMode(key)}
              className={`rounded border p-3 text-left text-xs font-semibold transition ${
                fusionMode === key
                  ? 'border-[var(--ds-accent)] bg-[var(--ds-paper)] text-[var(--ds-accent)]'
                  : 'border-[var(--ds-rule)] bg-[var(--ds-panel)] text-[var(--ds-ink)]'
              }`}
            >
              {mode.label}
            </button>
          ))}
        </div>

        <div className="mt-4 grid gap-2 md:grid-cols-3">
          {['text', 'speech', 'text+speech'].map((mode) => (
            <button
              key={mode}
              type="button"
              onClick={() => setOutputMode(mode)}
              className={`rounded border px-3 py-2 text-xs font-bold ${
                outputMode === mode ? 'border-[var(--ds-accent)] bg-[var(--ds-accent)] text-[var(--ds-paper)]' : 'border-[var(--ds-rule)] bg-[var(--ds-paper)]'
              }`}
            >
              {mode}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <ModalityStream name="Text tokens" active={modalities.text} tokens={modalities.text ? 160 : 0} icon={MessageSquare} />
          <ModalityStream name="Image patch tokens" active={modalities.image} tokens={modalities.image ? 360 : 0} icon={ImageIcon} />
          <ModalityStream name="Video frame tokens" active={modalities.video} tokens={modalities.video ? 2240 : 0} icon={Film} />
          <ModalityStream name="Audio tokens" active={modalities.audio} tokens={modalities.audio ? 880 : 0} icon={Mic} />
        </div>
        <div className="border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
          <h3 className="mb-2 font-bold text-[var(--ds-ink)]">{fusion.label}</h3>
          <p className="text-sm leading-6 text-[var(--ds-faint)]">{fusion.description}</p>
          <div className="mt-4 grid gap-3">
            <MetricBar label="Cross-modal interaction" value={fusion.interaction} />
            <MetricBar label="Grounding quality" value={grounding} tone="bg-[var(--ds-accent-2)]" />
            <MetricBar label="First-audio pressure" value={clamp(fusion.latency + latencyPenalty + (speech ? 16 : 0))} tone="bg-amber-500" />
          </div>
        </div>
        <div className="grid gap-3 sm:grid-cols-3">
          <MetricTile label="Total tokens" value={tokens.toLocaleString()} icon={Layers} />
          <MetricTile label="Output" value={outputMode} icon={Speech} />
          <MetricTile label="Target" value={latencyTarget} icon={Clock} />
        </div>
        <div className="grid gap-2 sm:grid-cols-3">
          {['offline', 'interactive', 'real-time'].map((target) => (
            <button
              key={target}
              type="button"
              onClick={() => setLatencyTarget(target)}
              className={`rounded border px-3 py-2 text-xs font-bold ${
                latencyTarget === target ? 'border-[var(--ds-accent)] bg-[var(--ds-accent)] text-[var(--ds-paper)]' : 'border-[var(--ds-rule)] bg-[var(--ds-paper)]'
              }`}
            >
              {target}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}

function ControlButton({ active, children, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded border px-3 py-2 text-xs font-bold transition ${
        active ? 'border-[var(--ds-accent)] bg-[var(--ds-accent)] text-[var(--ds-paper)]' : 'border-[var(--ds-rule)] bg-[var(--ds-paper)] text-[var(--ds-ink)]'
      }`}
    >
      {children}
    </button>
  );
}

function PanelFrame({ icon: Icon, title, children, misconception }) {
  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_0.55fr]">
      <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
        <div className="mb-4 flex items-center gap-2">
          <Icon className="h-5 w-5 text-[var(--ds-accent)]" />
          <h2 className="text-lg font-bold text-[var(--ds-ink)]">{title}</h2>
        </div>
        {children}
      </div>
      <div className="border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
        <div className="mb-2 flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-amber-500" />
          <h3 className="font-bold text-[var(--ds-ink)]">Misconception</h3>
        </div>
        <p className="text-sm leading-6 text-[var(--ds-faint)]">{misconception}</p>
      </div>
    </div>
  );
}

function TabPanel({ activeTab, modalities, fusionMode }) {
  const [patchSize, setPatchSize] = useState(16);
  const [groundingMode, setGroundingMode] = useState('bounding-box');
  const [frameRate, setFrameRate] = useState(4);
  const [chunkMs, setChunkMs] = useState(40);
  const [talker, setTalker] = useState('streaming');
  const [audioMode, setAudioMode] = useState('codec-autoregressive');
  const [reasoningTask, setReasoningTask] = useState('chart-qa');
  const [networkMs, setNetworkMs] = useState(50);

  if (activeTab === 'omni-map') {
    return (
      <PanelFrame icon={Network} title="Omni Architecture Map" misconception="Omni does not mean every modality is processed identically. Text, image, video, and audio usually need different encoders, positions, token rates, and decoders.">
        <div className="grid gap-4 md:grid-cols-[0.8fr_1fr]">
          <div className="space-y-3">
            {Object.entries(MODALITY_PIPELINES).map(([key, pipeline]) => (
              <div key={key} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3">
                <div className="text-sm font-bold text-[var(--ds-ink)]">{pipeline.label}</div>
                <div className="mt-1 text-xs text-[var(--ds-faint)]">{pipeline.raw} to {pipeline.encoder} to {pipeline.tokens}</div>
              </div>
            ))}
          </div>
          <div className="flex min-h-[280px] flex-col justify-center rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
            <div className="mx-auto w-full max-w-md rounded border border-[var(--ds-accent)] bg-[var(--ds-panel)] p-4 text-center">
              <Brain className="mx-auto mb-2 h-8 w-8 text-[var(--ds-accent)]" />
              <div className="font-bold text-[var(--ds-ink)]">Shared model / Thinker</div>
              <p className="mt-2 text-xs leading-5 text-[var(--ds-faint)]">Fusion mode: {FUSION_MODES[fusionMode].label}</p>
            </div>
            <div className="mt-4 grid gap-2 sm:grid-cols-2">
              <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper-2)] p-3 text-center text-sm font-bold text-[var(--ds-ink)]">Text output</div>
              <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper-2)] p-3 text-center text-sm font-bold text-[var(--ds-ink)]">Speech Talker</div>
            </div>
          </div>
        </div>
      </PanelFrame>
    );
  }

  if (activeTab === 'vision-projector') {
    const tokenCount = Math.round((224 / patchSize) ** 2);
    return (
      <PanelFrame icon={Eye} title="Vision Encoder + Projector" misconception="The projector is not decorative. It is the bridge that makes visual features usable by the language model.">
        <div className="grid gap-4 md:grid-cols-[0.55fr_1fr]">
          <PatchGrid density={Math.max(4, Math.min(8, Math.round(64 / patchSize)))} highlight={7} />
          <div className="space-y-4">
            <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
              <p className="text-sm leading-6 text-[var(--ds-faint)]">
                Image pixels pass through a vision encoder, become patch features, then a projector maps them into the same hidden size used by text tokens.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              {[8, 14, 16, 32].map((size) => (
                <ControlButton key={size} active={patchSize === size} onClick={() => setPatchSize(size)}>
                  patch {size}
                </ControlButton>
              ))}
            </div>
            <MetricBar label="Image tokens" value={clamp(tokenCount / 4)} />
            <MetricBar label="Spatial detail retained" value={clamp(100 - patchSize * 2)} tone="bg-[var(--ds-accent-2)]" />
            <MetricBar label="Alignment risk" value={clamp(patchSize * 2 + (tokenCount > 400 ? 18 : 0))} tone="bg-amber-500" />
          </div>
        </div>
      </PanelFrame>
    );
  }

  if (activeTab === 'fusion') {
    return (
      <PanelFrame icon={Layers} title="Early Fusion vs Late Fusion" misconception="Early fusion is powerful because modalities interact deeply, but it can be expensive because every modality competes for the same context budget.">
        <div className="grid gap-3 md:grid-cols-4">
          {Object.values(FUSION_MODES).map((mode) => (
            <div key={mode.label} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3">
              <h3 className="text-sm font-bold text-[var(--ds-ink)]">{mode.label}</h3>
              <p className="mt-2 text-xs leading-5 text-[var(--ds-faint)]">{mode.description}</p>
              <div className="mt-3 space-y-2">
                <MetricBar label="Interaction" value={mode.interaction} />
                <MetricBar label="Latency" value={mode.latency} tone="bg-amber-500" />
              </div>
            </div>
          ))}
        </div>
      </PanelFrame>
    );
  }

  if (activeTab === 'grounding') {
    const localization = groundingMode === 'bounding-box' ? 86 : groundingMode === 'attention-heatmap' ? 66 : groundingMode === 'segmentation-mask' ? 92 : 28;
    return (
      <PanelFrame icon={LocateFixed} title="Image Tokens and Grounding" misconception="A correct caption is not the same as grounding. Grounding means the model can connect the claim to the right visual region.">
        <div className="grid gap-4 md:grid-cols-[0.55fr_1fr]">
          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
            <PatchGrid density={5} highlight={13} />
            <div className="mt-3 rounded border border-[var(--ds-accent)] p-2 text-xs font-bold text-[var(--ds-accent)]">Highlighted region: red car blocking crosswalk</div>
          </div>
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2">
              {['none', 'attention-heatmap', 'bounding-box', 'segmentation-mask'].map((mode) => (
                <ControlButton key={mode} active={groundingMode === mode} onClick={() => setGroundingMode(mode)}>
                  {mode}
                </ControlButton>
              ))}
            </div>
            <MetricBar label="Localization accuracy" value={localization} />
            <MetricBar label="Grounded claims" value={localization - 10} tone="bg-[var(--ds-accent-2)]" />
            <MetricBar label="Hallucinated object risk" value={100 - localization} tone="bg-amber-500" />
          </div>
        </div>
      </PanelFrame>
    );
  }

  if (activeTab === 'video') {
    const tokens = frameRate * 10 * 196;
    return (
      <PanelFrame icon={Film} title="Video Frames and Temporal Tokens" misconception="Video understanding is not just image understanding repeated. Temporal order, sampling rate, and event alignment matter.">
        <div className="space-y-4">
          <div className="grid grid-cols-5 gap-2">
            {Array.from({ length: 5 }).map((_, index) => (
              <div key={index} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3 text-center">
                <div className="mx-auto mb-2 h-12 rounded bg-[var(--ds-paper-2)]" />
                <span className="font-mono text-xs text-[var(--ds-faint)]">t{index + 1}</span>
              </div>
            ))}
          </div>
          <div className="flex flex-wrap gap-2">
            {[1, 2, 4, 8].map((rate) => (
              <ControlButton key={rate} active={frameRate === rate} onClick={() => setFrameRate(rate)}>
                {rate} FPS
              </ControlButton>
            ))}
          </div>
          <MetricBar label="Temporal coverage" value={clamp(frameRate * 14)} />
          <MetricBar label="Event recall" value={clamp(38 + frameRate * 8)} tone="bg-[var(--ds-accent-2)]" />
          <MetricBar label="Context pressure" value={clamp(tokens / 180)} tone="bg-amber-500" />
        </div>
      </PanelFrame>
    );
  }

  if (activeTab === 'audio') {
    const tokensPerSecond = Math.round(1000 / chunkMs) * 4;
    return (
      <PanelFrame icon={AudioLines} title="Audio Features and Speech Codec Tokens" misconception="Speech output is not just text-to-speech glued on. In omni systems, the speech generator may be conditioned directly on the model's hidden reasoning state.">
        <div className="grid gap-4 md:grid-cols-2">
          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
            <div className="mb-3 flex h-20 items-end gap-1">
              {Array.from({ length: 32 }).map((_, index) => (
                <span key={index} className="w-full rounded-t bg-[var(--ds-accent)]" style={{ height: `${20 + ((index * 17) % 60)}%` }} />
              ))}
            </div>
            <p className="text-xs leading-5 text-[var(--ds-faint)]">Waveform to audio encoder to acoustic tokens. Talker to codec tokens to waveform.</p>
          </div>
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2">
              {[20, 40, 80, 160].map((ms) => (
                <ControlButton key={ms} active={chunkMs === ms} onClick={() => setChunkMs(ms)}>
                  {ms} ms chunks
                </ControlButton>
              ))}
            </div>
            <MetricBar label="Audio tokens per second" value={tokensPerSecond} />
            <MetricBar label="First-packet latency" value={clamp(100 - chunkMs / 2)} tone="bg-[var(--ds-accent-2)]" />
            <MetricBar label="Noise sensitivity" value={chunkMs < 40 ? 58 : 34} tone="bg-amber-500" />
          </div>
        </div>
      </PanelFrame>
    );
  }

  if (activeTab === 'thinker-talker') {
    return (
      <PanelFrame icon={Brain} title="Thinker-Talker Architecture" misconception="The Talker is not simply reading final text aloud. It can be a separate generation pathway conditioned on the Thinker's internal representations.">
        <div className="grid gap-4 md:grid-cols-[1fr_0.8fr]">
          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
            <div className="grid gap-3">
              <div className="rounded border border-[var(--ds-accent)] p-3 text-center font-bold text-[var(--ds-ink)]">Text / Image / Audio / Video</div>
              <div className="rounded border border-[var(--ds-accent)] bg-[var(--ds-panel)] p-4 text-center font-bold text-[var(--ds-ink)]">Thinker</div>
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded border border-[var(--ds-rule)] p-3 text-center text-sm font-bold text-[var(--ds-ink)]">Text head</div>
                <div className="rounded border border-[var(--ds-rule)] p-3 text-center text-sm font-bold text-[var(--ds-ink)]">Talker to speech tokens</div>
              </div>
            </div>
          </div>
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2">
              {['off', 'streaming', 'offline'].map((mode) => (
                <ControlButton key={mode} active={talker === mode} onClick={() => setTalker(mode)}>
                  Talker {mode}
                </ControlButton>
              ))}
            </div>
            <MetricBar label="Reasoning quality" value={talker === 'off' ? 72 : 84} />
            <MetricBar label="Speech first-packet latency" value={talker === 'streaming' ? 28 : talker === 'offline' ? 78 : 0} tone="bg-amber-500" />
            <MetricBar label="Synchronization quality" value={talker === 'streaming' ? 82 : 64} tone="bg-[var(--ds-accent-2)]" />
          </div>
        </div>
      </PanelFrame>
    );
  }

  if (activeTab === 'speech') {
    return (
      <PanelFrame icon={Speech} title="Speech-to-Speech Models" misconception="Speech-to-speech is not just faster TTS. It changes what information survives from the user's voice into the response.">
        <div className="grid gap-4 md:grid-cols-2">
          {[
            ['ASR to LLM to TTS pipeline', 'Audio is transcribed to text, answered as text, then synthesized as speech.', 62, 52],
            ['Direct omni speech pathway', 'Audio tokens condition a shared model and Talker generates speech tokens.', 86, 78],
          ].map(([label, desc, prosody, latency]) => (
            <div key={label} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
              <h3 className="font-bold text-[var(--ds-ink)]">{label}</h3>
              <p className="mt-2 text-sm leading-6 text-[var(--ds-faint)]">{desc}</p>
              <div className="mt-4 space-y-2">
                <MetricBar label="Emotion / prosody retention" value={prosody} />
                <MetricBar label="Turn-taking smoothness" value={latency} tone="bg-[var(--ds-accent-2)]" />
              </div>
            </div>
          ))}
        </div>
      </PanelFrame>
    );
  }

  if (activeTab === 'audio-generation') {
    const codecMode = audioMode !== 'block-diffusion';
    return (
      <PanelFrame icon={Radio} title="Diffusion Audio vs Codec Autoregression" misconception="Lower first-packet latency does not automatically mean better speech. It is a tradeoff among streaming, quality, stability, and controllability.">
        <div className="space-y-4">
          <div className="flex flex-wrap gap-2">
            {['block-diffusion', 'codec-autoregressive', 'causal-convnet-codec'].map((mode) => (
              <ControlButton key={mode} active={audioMode === mode} onClick={() => setAudioMode(mode)}>
                {mode}
              </ControlButton>
            ))}
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
              <h3 className="font-bold text-[var(--ds-ink)]">Generation path</h3>
              <p className="mt-2 text-sm leading-6 text-[var(--ds-faint)]">
                {codecMode ? 'Codec frames can be emitted one by one, so playback can start earlier.' : 'Diffusion waits through several denoising steps before audio starts.'}
              </p>
            </div>
            <div className="space-y-2">
              <MetricBar label="First-packet latency" value={codecMode ? 24 : 78} tone="bg-amber-500" />
              <MetricBar label="Streaming stability" value={codecMode ? 82 : 54} />
              <MetricBar label="Quality proxy" value={audioMode === 'block-diffusion' ? 88 : 74} tone="bg-[var(--ds-accent-2)]" />
            </div>
          </div>
        </div>
      </PanelFrame>
    );
  }

  if (activeTab === 'reasoning') {
    const depth = reasoningTask === 'audio-video-event' ? 86 : reasoningTask === 'chart-qa' ? 74 : 68;
    return (
      <PanelFrame icon={ScanSearch} title="Multimodal Reasoning" misconception="Multimodal reasoning is not just recognizing objects. It means combining evidence across modalities into a correct inference.">
        <div className="space-y-4">
          <div className="flex flex-wrap gap-2">
            {['chart-qa', 'diagram-explanation', 'audio-video-event', 'multi-image-comparison'].map((task) => (
              <ControlButton key={task} active={reasoningTask === task} onClick={() => setReasoningTask(task)}>
                {task}
              </ControlButton>
            ))}
          </div>
          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
            <p className="text-sm leading-6 text-[var(--ds-faint)]">
              The model links a spoken phrase to a chart point, a video event, and a text instruction before producing a grounded answer.
            </p>
          </div>
          <MetricBar label="Cross-modal link accuracy" value={depth} />
          <MetricBar label="Evidence use" value={depth - 8} tone="bg-[var(--ds-accent-2)]" />
          <MetricBar label="Distractor confusion" value={100 - depth} tone="bg-amber-500" />
        </div>
      </PanelFrame>
    );
  }

  if (activeTab === 'latency') {
    return (
      <PanelFrame icon={Clock} title="Real-Time Latency and System Tradeoffs" misconception="Real-time multimodal AI is a systems problem, not only a model architecture problem.">
        <div className="grid gap-4 md:grid-cols-[0.7fr_1fr]">
          <div className="space-y-3">
            <div className="flex flex-wrap gap-2">
              {[20, 40, 80, 160].map((ms) => (
                <ControlButton key={ms} active={chunkMs === ms} onClick={() => setChunkMs(ms)}>
                  {ms} ms chunk
                </ControlButton>
              ))}
            </div>
            <div className="flex flex-wrap gap-2">
              {[10, 50, 150, 500].map((ms) => (
                <ControlButton key={ms} active={networkMs === ms} onClick={() => setNetworkMs(ms)}>
                  {ms} ms network
                </ControlButton>
              ))}
            </div>
          </div>
          <LatencyWaterfall chunkMs={chunkMs} thinkerDepth="medium" talkerMode="streaming-codec" networkMs={networkMs} strictSafety />
        </div>
      </PanelFrame>
    );
  }

  return (
    <PanelFrame icon={CheckCircle2} title="Paper/Product Decoder" misconception="Product benchmark claims are anchors, not guarantees for every multimodal workflow. Match each claim to the architecture and evaluation it actually supports.">
      <div className="grid gap-3 md:grid-cols-2">
        {PAPER_CARDS.map((card) => (
          <div key={card.id} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
            <h3 className="font-bold text-[var(--ds-ink)]">{card.label}</h3>
            <ul className="mt-3 space-y-1 text-xs text-[var(--ds-faint)]">
              {card.signals.map((signal) => (
                <li key={signal}>- {signal}</li>
              ))}
            </ul>
            <p className="mt-3 text-sm leading-6 text-[var(--ds-ink)]">{card.interpretation}</p>
          </div>
        ))}
      </div>
    </PanelFrame>
  );
}

export default function OmniMultimodalArchitectures() {
  const [modalities, setModalities] = useState({ text: true, image: true, video: false, audio: true });
  const [fusionMode, setFusionMode] = useState('thinkerTalker');
  const [outputMode, setOutputMode] = useState('text+speech');
  const [latencyTarget, setLatencyTarget] = useState('interactive');
  const [activeTab, setActiveTab] = useState('omni-map');

  const derived = useMemo(() => {
    const activeCount = Object.values(modalities).filter(Boolean).length;
    const tokenPressure = activeCount * 18 + (modalities.video ? 28 : 0) + (modalities.audio ? 16 : 0);
    return {
      modalityCoverage: activeCount * 25,
      tokenPressure,
      latency: latencyTarget === 'real-time' ? 72 : latencyTarget === 'interactive' ? 48 : 28,
    };
  }, [modalities, latencyTarget]);

  return (
    <main className="min-h-screen bg-[var(--ds-paper)] text-[var(--ds-ink)]">
      <div className="mx-auto max-w-7xl px-4 py-6">
        <header className="mb-6 border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
          <div className="mb-3 inline-flex items-center gap-2 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] px-3 py-1 text-xs font-bold uppercase tracking-wide text-[var(--ds-faint)]">
            <Zap className="h-4 w-4 text-[var(--ds-accent)]" />
            Frontier LLMs
          </div>
          <h1 className="text-3xl font-black tracking-tight text-[var(--ds-ink)] md:text-5xl">Multimodal and Omni Models</h1>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-[var(--ds-faint)] md:text-base">
            How frontier systems turn text, images, video, and audio into shared reasoning and real-time speech.
          </p>
        </header>

        <OmniWorkbench
          modalities={modalities}
          setModalities={setModalities}
          fusionMode={fusionMode}
          setFusionMode={setFusionMode}
          outputMode={outputMode}
          setOutputMode={setOutputMode}
          latencyTarget={latencyTarget}
          setLatencyTarget={setLatencyTarget}
        />

        <section className="my-5 grid gap-3 md:grid-cols-3">
          <MetricTile label="Modality coverage" value={`${clamp(derived.modalityCoverage)}%`} icon={Grid3X3} />
          <MetricTile label="Token pressure" value={`${clamp(derived.tokenPressure)}%`} icon={Gauge} />
          <MetricTile label="Latency pressure" value={`${clamp(derived.latency)}%`} icon={Clock} />
        </section>

        <nav className="mb-4 flex gap-2 overflow-x-auto pb-2" aria-label="Multimodal and omni model panels">
          {OMNI_TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`shrink-0 rounded border px-3 py-2 text-xs font-bold transition ${
                activeTab === tab.id
                  ? 'border-[var(--ds-accent)] bg-[var(--ds-accent)] text-[var(--ds-paper)]'
                  : 'border-[var(--ds-rule)] bg-[var(--ds-panel)] text-[var(--ds-ink)]'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        <TabPanel activeTab={activeTab} modalities={modalities} fusionMode={fusionMode} />

        <section className="mt-5 grid gap-3 md:grid-cols-4">
          {FAILURE_MODES.map((failure) => (
            <div key={failure.id} className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
              <h3 className="font-bold text-[var(--ds-ink)]">{failure.label}</h3>
              <p className="mt-2 text-xs leading-5 text-[var(--ds-faint)]">{failure.symptom}</p>
              <p className="mt-3 text-xs font-semibold leading-5 text-[var(--ds-ink)]">{failure.mitigation}</p>
            </div>
          ))}
        </section>
      </div>
    </main>
  );
}
