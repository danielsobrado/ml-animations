import React, { useMemo, useState } from 'react';
import {
  AlertTriangle,
  BarChart3,
  Brain,
  CheckCircle2,
  Database,
  FileSearch,
  Gauge,
  GitBranch,
  Layers,
  Maximize2,
  Network,
  RotateCcw,
  Search,
  ShieldCheck,
  Zap,
} from 'lucide-react';
import {
  CONTEXT_STRATEGIES,
  LONG_CONTEXT_FAILURES,
  LONG_CONTEXT_TABS,
  PAPER_CARDS,
} from './data';

const CONTEXT_LENGTHS = [8000, 32000, 128000, 1000000, 10000000];
const STRATEGY_KEYS = Object.keys(CONTEXT_STRATEGIES);

function clamp(value) {
  return Math.max(0, Math.min(100, Math.round(value)));
}

function formatTokens(value) {
  if (value >= 1000000) return `${value / 1000000}M`;
  if (value >= 1000) return `${value / 1000}K`;
  return String(value);
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

function TokenPill({ children, active }) {
  return (
    <span
      className={`inline-flex rounded border px-2 py-1 font-mono text-[10px] ${
        active
          ? 'border-[var(--ds-accent)] bg-[var(--ds-accent)] text-[var(--ds-paper)]'
          : 'border-[var(--ds-rule)] bg-[var(--ds-paper)] text-[var(--ds-faint)]'
      }`}
    >
      {children}
    </span>
  );
}

function StrategyWorkbench({ strategyKey, setStrategyKey, corpusTokens, setCorpusTokens }) {
  const strategy = CONTEXT_STRATEGIES[strategyKey];
  const tokenScale = Math.log10(corpusTokens) / 7;
  const evidenceCoverage = clamp(strategy.coverage - (strategyKey === 'rag' ? 10 : 0) + (strategyKey === 'hybrid' ? 6 : 0));
  const distractorLoad = clamp(strategy.distractors * tokenScale + (strategyKey === 'fullContext' ? 18 : 0));
  const latency = clamp(strategy.cost * tokenScale + (corpusTokens >= 1000000 ? 20 : 0));
  const grounding = clamp(evidenceCoverage - distractorLoad * 0.18 + (strategyKey === 'hybrid' ? 8 : 0));

  return (
    <section className="grid gap-4 lg:grid-cols-[1fr_0.9fr]">
      <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
        <div className="mb-4 flex items-center gap-2">
          <Maximize2 className="h-5 w-5 text-[var(--ds-accent)]" />
          <h2 className="text-lg font-bold text-[var(--ds-ink)]">Context Strategy Workbench</h2>
        </div>
        <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
          <p className="text-xs font-bold uppercase tracking-wide text-[var(--ds-faint)]">Task</p>
          <p className="mt-2 text-base font-semibold leading-6 text-[var(--ds-ink)]">
            Across 400 documents, find the clause that modifies the renewal fee, compare it with the invoice, and explain whether the invoice is compliant.
          </p>
        </div>

        <div className="mt-4 grid gap-2 md:grid-cols-4">
          {STRATEGY_KEYS.map((key) => {
            const item = CONTEXT_STRATEGIES[key];
            const isActive = key === strategyKey;
            return (
              <button
                key={key}
                data-math-control
                onClick={() => setStrategyKey(key)}
                className={`border p-3 text-left transition ${
                  isActive
                    ? 'border-[var(--ds-accent)] bg-[var(--ds-paper)] text-[var(--ds-ink)]'
                    : 'border-[var(--ds-rule)] bg-transparent text-[var(--ds-muted)] hover:bg-[var(--ds-paper)]'
                }`}
              >
                <span className="block text-sm font-bold">{item.label}</span>
                <span className="mt-2 block text-[11px] leading-4 text-[var(--ds-faint)]">{item.advantage}</span>
              </button>
            );
          })}
        </div>

        <div className="mt-4">
          <p className="mb-2 text-xs font-bold uppercase tracking-wide text-[var(--ds-faint)]">Corpus tokens</p>
          <div className="flex flex-wrap gap-2">
            {CONTEXT_LENGTHS.map((value) => (
              <button key={value} data-math-control onClick={() => setCorpusTokens(value)}>
                <TokenPill active={value === corpusTokens}>{formatTokens(value)}</TokenPill>
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
        <h3 className="text-lg font-bold text-[var(--ds-ink)]">{strategy.label}</h3>
        <p className="mt-2 text-sm leading-6 text-[var(--ds-muted)]">{strategy.description}</p>
        <div className="mt-4 grid gap-3">
          <MetricBar label="Evidence coverage" value={evidenceCoverage} tone="bg-emerald-600" />
          <MetricBar label="Distractor load" value={distractorLoad} tone="bg-rose-600" />
          <MetricBar label="Latency / cost proxy" value={latency} tone="bg-amber-600" />
          <MetricBar label="Grounding confidence" value={grounding} tone="bg-[var(--ds-accent)]" />
        </div>
        <div className="mt-4 border-l-2 border-amber-500 bg-amber-50 p-3 text-sm text-amber-950">
          {strategy.risk}
        </div>
      </div>
    </section>
  );
}

function ContextStrip({ needlePosition }) {
  const positions = ['start', 'early', 'middle', 'late', 'end'];
  return (
    <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
      <p className="mb-3 text-xs font-bold uppercase tracking-wide text-[var(--ds-faint)]">Evidence position</p>
      <div className="grid grid-cols-5 gap-1">
        {positions.map((position) => (
          <div
            key={position}
            className={`min-h-20 border p-2 text-center text-xs ${
              position === needlePosition
                ? 'border-[var(--ds-accent)] bg-[var(--ds-paper)] text-[var(--ds-ink)]'
                : 'border-[var(--ds-rule)] bg-[var(--ds-paper-2)] text-[var(--ds-faint)]'
            }`}
          >
            <div className="font-mono uppercase">{position}</div>
            {position === needlePosition ? <div className="mt-4 font-bold text-[var(--ds-accent)]">evidence</div> : null}
          </div>
        ))}
      </div>
    </div>
  );
}

function ClaimedEffectivePanel({ corpusTokens }) {
  const claimed = corpusTokens >= 1000000 ? 100 : 74;
  const native = corpusTokens >= 1000000 ? 42 : 68;
  const reasoning = corpusTokens >= 1000000 ? 48 : 72;
  return (
    <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
      <p className="mb-3 text-xs font-bold uppercase tracking-wide text-[var(--ds-faint)]">Advertised window is not task reliability</p>
      <div className="grid gap-3">
        <MetricBar label="Claimed context" value={claimed} />
        <MetricBar label="Native training length proxy" value={native} tone="bg-emerald-600" />
        <MetricBar label="Effective reasoning length" value={reasoning} tone="bg-amber-600" />
      </div>
    </div>
  );
}

function PositionPanel({ corpusTokens }) {
  const stress = clamp(Math.log10(corpusTokens / 4096) * 30);
  return (
    <div className="grid gap-4 md:grid-cols-3">
      {[
        ['Standard RoPE', 78],
        ['Scaled RoPE', 52],
        ['iRoPE-inspired mix', 38],
      ].map(([label, base]) => (
        <div key={label} className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
          <RotateCcw className="mb-3 h-5 w-5 text-[var(--ds-accent)]" />
          <h3 className="font-bold text-[var(--ds-ink)]">{label}</h3>
          <p className="mt-2 text-xs leading-5 text-[var(--ds-faint)]">
            {label === 'iRoPE-inspired mix'
              ? 'Interleave RoPE layers with layers that reduce direct position dependence.'
              : 'Stretch or reuse position rotations beyond familiar training lengths.'}
          </p>
          <div className="mt-3">
            <MetricBar label="Extrapolation stress" value={(base + stress) / 2} tone="bg-rose-600" />
          </div>
        </div>
      ))}
    </div>
  );
}

function KVCostPanel({ corpusTokens }) {
  const base = Math.log10(corpusTokens) * 12;
  return (
    <div className="grid gap-3">
      {[
        ['MHA full cache', base + 34],
        ['GQA', base + 12],
        ['MLA', base - 6],
        ['RAG packed', Math.max(16, base - 28)],
      ].map(([label, value]) => (
        <div key={label} className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
          <MetricBar label={label} value={value} tone={label === 'MHA full cache' ? 'bg-rose-600' : 'bg-emerald-600'} />
        </div>
      ))}
    </div>
  );
}

function NeedlePanel() {
  const rows = [
    ['Single needle', 94, 'Exact phrase lookup'],
    ['Multi-needle', 72, 'Several similar facts'],
    ['Semantic needle', 54, 'Low lexical overlap'],
    ['Needle chain', 47, 'Facts must be linked'],
  ];
  return (
    <div className="grid gap-2">
      {rows.map(([label, value, note]) => (
        <div key={label} className="grid gap-3 border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4 md:grid-cols-[0.7fr_1fr]">
          <div>
            <h3 className="font-bold text-[var(--ds-ink)]">{label}</h3>
            <p className="text-xs text-[var(--ds-faint)]">{note}</p>
          </div>
          <MetricBar label="Success proxy" value={value} tone="bg-[var(--ds-accent)]" />
        </div>
      ))}
    </div>
  );
}

function MultiHopPanel({ strategyKey }) {
  const strategyBoost = strategyKey === 'hybrid' ? 18 : strategyKey === 'rag' ? -4 : strategyKey === 'fullContext' ? 7 : -12;
  const hops = [
    ['Hop 1: find clause A', 82 + strategyBoost],
    ['Hop 2: link amendment B', 64 + strategyBoost],
    ['Hop 3: compare invoice C', 52 + strategyBoost],
  ];
  return (
    <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
      <div className="mb-3 flex items-center gap-2">
        <Network className="h-5 w-5 text-[var(--ds-accent)]" />
        <span className="text-xs font-bold uppercase tracking-wide text-[var(--ds-faint)]">Graphwalk-style state tracking</span>
      </div>
      <div className="grid gap-3">
        {hops.map(([label, value]) => (
          <MetricBar key={label} label={label} value={value} tone="bg-emerald-600" />
        ))}
      </div>
    </div>
  );
}

function CompressionPanel() {
  return (
    <div className="grid gap-3 md:grid-cols-4">
      {[
        ['Flat summary', 65, 22],
        ['Hierarchical summary', 76, 38],
        ['Entity memory', 82, 29],
        ['Hybrid memory', 88, 46],
      ].map(([label, retention, cost]) => (
        <div key={label} className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
          <Database className="mb-3 h-5 w-5 text-[var(--ds-accent)]" />
          <h3 className="font-bold text-[var(--ds-ink)]">{label}</h3>
          <div className="mt-3 space-y-3">
            <MetricBar label="Fact retention" value={retention} tone="bg-emerald-600" />
            <MetricBar label="Token cost" value={cost} tone="bg-amber-600" />
          </div>
        </div>
      ))}
    </div>
  );
}

function FailureDashboard() {
  return (
    <div className="grid gap-2 md:grid-cols-5">
      {LONG_CONTEXT_FAILURES.map((failure) => (
        <div key={failure.id} className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-3">
          <AlertTriangle className="mb-2 h-4 w-4 text-amber-600" />
          <h3 className="text-sm font-bold text-[var(--ds-ink)]">{failure.label}</h3>
          <p className="mt-2 text-[11px] leading-4 text-[var(--ds-faint)]">{failure.mitigation}</p>
        </div>
      ))}
    </div>
  );
}

function TabPanel({ tab, corpusTokens, strategyKey }) {
  const [needlePosition, setNeedlePosition] = useState('middle');
  const Icon = {
    'strategy-map': Search,
    'claimed-effective': Gauge,
    pretraining: Layers,
    position: RotateCcw,
    'kv-cost': Database,
    'lost-middle': AlertTriangle,
    'needle-limits': FileSearch,
    'multi-hop': Network,
    compression: Brain,
    'rag-hybrid': GitBranch,
    evaluation: BarChart3,
  }[tab.id] || Maximize2;

  const lostMiddleRisk = needlePosition === 'middle' ? 82 : needlePosition === 'early' || needlePosition === 'late' ? 48 : 22;

  return (
    <section className="grid gap-4 lg:grid-cols-[0.8fr_1.2fr]">
      <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
        <div className="mb-3 flex items-center gap-2">
          <Icon className="h-5 w-5 text-[var(--ds-accent)]" />
          <h2 className="text-lg font-bold text-[var(--ds-ink)]">{tab.title}</h2>
        </div>
        <p className="text-sm leading-6 text-[var(--ds-muted)]">{tab.purpose}</p>
        <div className="mt-4 border-l-2 border-amber-500 bg-amber-50 p-3 text-sm text-amber-950">
          {tab.misconception}
        </div>
        <div className="mt-4 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3">
          <p className="text-xs font-bold uppercase tracking-wide text-[var(--ds-faint)]">Practice lab</p>
          <p className="mt-2 text-sm text-[var(--ds-ink)]">{tab.lab}</p>
        </div>
      </div>

      <div className="grid gap-4">
        {tab.id === 'claimed-effective' ? <ClaimedEffectivePanel corpusTokens={corpusTokens} /> : null}
        {tab.id === 'pretraining' ? <ClaimedEffectivePanel corpusTokens={Math.min(1000000, corpusTokens)} /> : null}
        {tab.id === 'position' ? <PositionPanel corpusTokens={corpusTokens} /> : null}
        {tab.id === 'kv-cost' ? <KVCostPanel corpusTokens={corpusTokens} /> : null}
        {tab.id === 'lost-middle' ? (
          <>
            <ContextStrip needlePosition={needlePosition} />
            <div className="flex flex-wrap gap-2">
              {['start', 'early', 'middle', 'late', 'end'].map((position) => (
                <button key={position} data-math-control onClick={() => setNeedlePosition(position)}>
                  <TokenPill active={needlePosition === position}>{position}</TokenPill>
                </button>
              ))}
            </div>
            <MetricBar label="Lost-in-middle risk" value={lostMiddleRisk} tone="bg-rose-600" />
          </>
        ) : null}
        {tab.id === 'needle-limits' ? <NeedlePanel /> : null}
        {tab.id === 'multi-hop' ? <MultiHopPanel strategyKey={strategyKey} /> : null}
        {tab.id === 'compression' ? <CompressionPanel /> : null}
        {tab.id === 'rag-hybrid' || tab.id === 'strategy-map' ? <FailureDashboard /> : null}
        {tab.id === 'evaluation' ? (
          <div className="grid gap-3 md:grid-cols-2">
            {PAPER_CARDS.map((card) => (
              <article key={card.title} className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
                <h3 className="font-bold text-[var(--ds-ink)]">{card.title}</h3>
                <p className="mt-2 text-xs leading-5 text-[var(--ds-faint)]">{card.signal}</p>
                <p className="mt-3 text-sm leading-5 text-[var(--ds-muted)]">{card.interpretation}</p>
              </article>
            ))}
          </div>
        ) : null}
      </div>
    </section>
  );
}

export default function LongContextFrontierModels() {
  const [activeTab, setActiveTab] = useState('strategy-map');
  const [strategyKey, setStrategyKey] = useState('hybrid');
  const [corpusTokens, setCorpusTokens] = useState(1000000);
  const tab = useMemo(() => LONG_CONTEXT_TABS.find((item) => item.id === activeTab) || LONG_CONTEXT_TABS[0], [activeTab]);

  return (
    <div className="ua-lesson-stage space-y-6">
      <header className="border-b border-[var(--ds-rule)] pb-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.22em] text-[var(--ds-faint)]">Frontier LLMs</p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight text-[var(--ds-ink)]">Long Context: 1M to 10M Tokens</h1>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-[var(--ds-muted)]">
              Learn long context as a strategy space over direct evidence access, retrieval recall, distractor control, memory compression, position generalization, multi-hop reasoning, and serving cost.
            </p>
          </div>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-3">
              <p className="font-mono text-lg font-bold text-[var(--ds-ink)]">1M-10M</p>
              <p className="text-[10px] uppercase tracking-wide text-[var(--ds-faint)]">tokens</p>
            </div>
            <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-3">
              <p className="font-mono text-lg font-bold text-[var(--ds-ink)]">11</p>
              <p className="text-[10px] uppercase tracking-wide text-[var(--ds-faint)]">panels</p>
            </div>
            <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-3">
              <p className="font-mono text-lg font-bold text-[var(--ds-ink)]">Advanced</p>
              <p className="text-[10px] uppercase tracking-wide text-[var(--ds-faint)]">level</p>
            </div>
          </div>
        </div>
      </header>

      <StrategyWorkbench
        strategyKey={strategyKey}
        setStrategyKey={setStrategyKey}
        corpusTokens={corpusTokens}
        setCorpusTokens={setCorpusTokens}
      />

      <nav className="flex gap-2 overflow-x-auto border-y border-[var(--ds-rule)] py-3">
        {LONG_CONTEXT_TABS.map((item) => (
          <button
            key={item.id}
            data-math-control
            onClick={() => setActiveTab(item.id)}
            className={`shrink-0 border px-3 py-2 text-xs font-semibold transition ${
              activeTab === item.id
                ? 'border-[var(--ds-accent)] bg-[var(--ds-accent)] text-[var(--ds-paper)]'
                : 'border-[var(--ds-rule)] bg-[var(--ds-panel)] text-[var(--ds-muted)] hover:bg-[var(--ds-paper)]'
            }`}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <TabPanel tab={tab} corpusTokens={corpusTokens} strategyKey={strategyKey} />

      <section className="grid gap-4 lg:grid-cols-4">
        {[
          ['Evidence recall', CONTEXT_STRATEGIES[strategyKey].coverage, ShieldCheck],
          ['Context efficiency', strategyKey === 'fullContext' ? 18 : strategyKey === 'hybrid' ? 54 : 70, Zap],
          ['Distractor ratio', CONTEXT_STRATEGIES[strategyKey].distractors, AlertTriangle],
          ['Answer confidence', strategyKey === 'hybrid' ? 84 : strategyKey === 'fullContext' ? 69 : strategyKey === 'rag' ? 64 : 52, CheckCircle2],
        ].map(([label, value, Icon]) => (
          <div key={label} className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
            <Icon className="mb-3 h-5 w-5 text-[var(--ds-accent)]" />
            <MetricBar label={label} value={value} tone={label === 'Distractor ratio' ? 'bg-rose-600' : 'bg-emerald-600'} />
          </div>
        ))}
      </section>
    </div>
  );
}
