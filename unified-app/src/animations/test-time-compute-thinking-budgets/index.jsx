import React, { useState, useMemo } from 'react';
import {
  Activity,
  AlertTriangle,
  Brain,
  Clock,
  Cpu,
  Database,
  GitBranch,
  List,
  RefreshCw,
  Repeat,
  Settings,
  Sliders,
  Sparkles,
  Target,
  TrendingUp,
  Wrench,
  Zap,
  ChevronRight,
  BarChart2,
  ArrowRight,
  Filter,
} from 'lucide-react';
import {
  TTC_SCALING_REGIMES,
  TTC_STRATEGY_FAMILY,
  BON_STRATEGY_DATA,
  BEAM_SEARCH_STEPS,
  THINKING_BUDGET_EXAMPLES,
  BUDGET_FORCING_DATA,
  TTC_FAILURE_MODES,
  REACT_AGENT_STEPS,
  SCALING_LAWS_DATA,
  THINKING_BUDGET_STRATEGIES,
  METRICS_COMPARISON,
} from './data';

const TABS = [
  { id: 'scaling-regimes', label: 'TTC vs Training', icon: TrendingUp },
  { id: 'strategy-map', label: 'Strategy Map', icon: GitBranch },
  { id: 'best-of-n', label: 'Best-of-N Lab', icon: List },
  { id: 'beam-search', label: 'Tree Search', icon: Filter },
  { id: 'thinking-budget', label: 'Thinking Budgets', icon: Clock },
  { id: 'budget-forcing', label: 'Budget Forcing', icon: Sliders },
  { id: 'tool-augmented', label: 'Tool-Augmented', icon: Wrench },
  { id: 'failure-modes', label: 'Failure Modes', icon: AlertTriangle },
  { id: 'metrics', label: 'Cost vs Accuracy', icon: BarChart2 },
  { id: 'budget-strategies', label: 'Budget Policies', icon: Settings },
];

// Small helpers

function Pill({ children, color = 'stone' }) {
  const map = {
    stone: 'bg-stone-100 text-stone-700 border-stone-200',
    amber: 'bg-amber-50 text-amber-800 border-amber-200',
    blue: 'bg-blue-50 text-blue-800 border-blue-200',
    green: 'bg-green-50 text-green-800 border-green-200',
    red: 'bg-red-50 text-red-800 border-red-200',
  };
  return (
    <span className={`inline-block px-2 py-0.5 rounded border text-[10px] font-bold uppercase tracking-wide ${map[color] || map.stone}`}>
      {children}
    </span>
  );
}

function SectionHeader({ icon: Icon, title }) {
  return (
    <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-4 mb-1">
      <Icon className="w-5 h-5 text-[var(--ds-accent)]" />
      <h2 className="text-md font-bold uppercase tracking-wider text-[var(--ds-ink)]">{title}</h2>
    </div>
  );
}

function InfoCard({ className = '', children }) {
  return (
    <div className={`border border-[var(--ds-rule)] bg-[var(--ds-panel)] rounded p-5 space-y-4 ${className}`}>
      {children}
    </div>
  );
}

function SliderField({ label, min, max, step, value, onChange, formatFn, hint }) {
  return (
    <div className="space-y-1">
      <label className="block text-[10px] font-bold uppercase text-[var(--ds-faint)] tracking-wider">
        {label}: <span className="text-[var(--ds-ink)]">{formatFn ? formatFn(value) : value}</span>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="ds-range"
      />
      {hint && <span className="text-[10px] text-[var(--ds-faint)] block leading-relaxed">{hint}</span>}
    </div>
  );
}

// Tab: Scaling Regimes

function TabScalingRegimes() {
  const [selected, setSelected] = useState('inference-time');
  const regime = TTC_SCALING_REGIMES.find((r) => r.id === selected);

  return (
    <div className="space-y-6">
      <InfoCard>
        <SectionHeader icon={TrendingUp} title="Test-Time Compute vs Training-Time Compute" />
        <p className="text-xs text-[var(--ds-faint)] leading-relaxed">
          Modern reasoning systems scale along two orthogonal axes. Training-time scaling buys a permanently smarter model; 
          inference-time (test-time) scaling rents additional compute per query. Together they form a Pareto frontier 
          so you can trade dollars at training time for dollars at inference time.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {TTC_SCALING_REGIMES.map((r) => {
            const isSelected = r.id === selected;
            return (
              <button
                key={r.id}
                data-math-control
                onClick={() => setSelected(r.id)}
                className={`p-5 border rounded text-left transition-all ${
                  isSelected
                    ? 'border-[var(--ds-accent)] bg-[var(--ds-warm)]'
                    : 'border-[var(--ds-rule)] bg-[var(--ds-paper)] hover:bg-[var(--ds-paper-2)]'
                }`}
              >
                <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block mb-1">Scaling Axis</span>
                <h3 className="text-sm font-bold text-[var(--ds-ink)] mb-1">{r.label}</h3>
                <p className="text-xs text-[var(--ds-faint)] leading-relaxed">{r.lever}</p>
              </button>
            );
          })}
        </div>

        {regime && (
          <div className="bg-[var(--ds-paper)] border border-[var(--ds-rule)] rounded p-5 grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <h3 className="text-sm font-bold text-[var(--ds-ink)]">{regime.label}</h3>
              <p className="text-xs text-[var(--ds-faint)] leading-relaxed">{regime.description}</p>
              <div className="space-y-1.5 text-xs">
                <div className="flex gap-2">
                  <span className="font-bold text-[var(--ds-faint)] w-16 shrink-0">Cost:</span>
                  <span className="text-[var(--ds-ink)]">{regime.cost}</span>
                </div>
                <div className="flex gap-2">
                  <span className="font-bold text-[var(--ds-faint)] w-16 shrink-0">Ceiling:</span>
                  <span className="text-[var(--ds-ink)]">{regime.ceiling}</span>
                </div>
              </div>
            </div>
            <div className="space-y-2">
              <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block">Examples</span>
              {regime.examples.map((ex, i) => (
                <div key={i} className="flex items-center gap-2 text-xs">
                  <ChevronRight className="w-3.5 h-3.5 text-[var(--ds-accent)] shrink-0" />
                  <span className="text-[var(--ds-ink)]">{ex}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </InfoCard>

      {/* Visual crossover concept */}
      <InfoCard>
        <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block mb-3">Conceptual Crossover</span>
        <div className="bg-[var(--ds-paper)] border border-[var(--ds-rule)] rounded p-4 text-xs text-[var(--ds-faint)] leading-relaxed">
          <p>{SCALING_LAWS_DATA.crossoverNote}</p>
          <div className="mt-4 grid grid-cols-2 md:grid-cols-5 gap-2">
            {SCALING_LAWS_DATA.testTimePoints.map((pt) => (
              <div key={pt.tokens} className="bg-[var(--ds-panel)] border border-[var(--ds-rule)] rounded p-2 text-center">
                <span className="block text-[10px] font-mono font-bold text-[var(--ds-ink)]">{pt.tokens} tok</span>
                <span className="block text-[10px] text-[var(--ds-faint)]">to {pt.accuracy}% acc</span>
              </div>
            ))}
          </div>
          <p className="text-[10px] text-[var(--ds-faint)] mt-2">
            Accuracy on a hard MATH benchmark as thinking-token budget grows (illustrative; actual numbers model-dependent).
          </p>
        </div>
      </InfoCard>
    </div>
  );
}

// Tab: Strategy Map

function TabStrategyMap() {
  const [selected, setSelected] = useState('best-of-n');
  const strat = TTC_STRATEGY_FAMILY.find((s) => s.id === selected);

  return (
    <div className="space-y-6">
      <InfoCard>
        <SectionHeader icon={GitBranch} title="Test-Time Compute Strategy Family" />
        <p className="text-xs text-[var(--ds-faint)] leading-relaxed">
          Every test-time scaling approach trades tokens (latency + cost) for accuracy.
          Click each strategy to see its mechanics and tradeoffs.
        </p>

        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {TTC_STRATEGY_FAMILY.map((s) => {
            const isSelected = s.id === selected;
            const icons = {
              'sequential-refinement': Repeat,
              'best-of-n': List,
              'beam-search': GitBranch,
              'adaptive-budget': Sliders,
              'tool-use': Wrench,
            };
            const Icon = icons[s.id] || Sparkles;
            return (
              <button
                key={s.id}
                data-math-control
                onClick={() => setSelected(s.id)}
                className={`p-3 border rounded text-left text-xs transition-all flex flex-col gap-1 ${
                  isSelected
                    ? 'border-[var(--ds-accent)] bg-[var(--ds-warm)]'
                    : 'border-[var(--ds-rule)] bg-[var(--ds-paper)] hover:bg-[var(--ds-paper-2)]'
                }`}
              >
                <Icon className="w-4 h-4 text-[var(--ds-accent)]" />
                <span className="font-bold text-[var(--ds-ink)]">{s.label}</span>
              </button>
            );
          })}
        </div>

        {strat && (
          <div className="bg-[var(--ds-paper)] border border-[var(--ds-rule)] rounded p-5 grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <h3 className="text-sm font-bold text-[var(--ds-ink)]">{strat.label}</h3>
              <p className="text-xs text-[var(--ds-faint)] leading-relaxed">{strat.description}</p>
              <div className="text-xs space-y-1">
                <div className="flex gap-2">
                  <span className="font-bold text-[var(--ds-faint)] w-28 shrink-0">Cost per query:</span>
                  <code className="font-mono text-[var(--ds-ink)]">{strat.cost_per_answer}</code>
                </div>
              </div>
              <div className="text-xs italic text-[var(--ds-faint)] bg-[var(--ds-panel)] p-2 rounded border border-[var(--ds-rule)]">
                {strat.example}
              </div>
            </div>
            <div className="space-y-3">
              <div>
                <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block mb-1">Key Tradeoff</span>
                <p className="text-xs text-[var(--ds-ink)] leading-relaxed">{strat.tradeoff}</p>
              </div>
            </div>
          </div>
        )}
      </InfoCard>
    </div>
  );
}

// Tab: Best-of-N

function TabBestOfN() {
  const [n, setN] = useState(8);
  const [verifierQuality, setVerifierQuality] = useState(0.8);

  const currentRow = useMemo(() => {
    const closest = BON_STRATEGY_DATA.reduce((prev, curr) =>
      Math.abs(curr.n - n) < Math.abs(prev.n - n) ? curr : prev
    );
    // Effective accuracy = oracle * verifier quality + expected * (1 - quality)
    const effective = Math.round(
      closest.oracleBound * verifierQuality + closest.expectedAcc * (1 - verifierQuality)
    );
    return { ...closest, effectiveAcc: Math.min(effective, closest.oracleBound) };
  }, [n, verifierQuality]);

  const maxAccBar = 100;

  return (
    <div className="space-y-6">
      <InfoCard>
        <SectionHeader icon={List} title="Best-of-N Sampling Lab" />
        <p className="text-xs text-[var(--ds-faint)] leading-relaxed">
          Generate N independent answers and pick the best. With a perfect verifier you hit the oracle bound;
          with a weak verifier you fall back toward the expected (no-selection) accuracy.
          Adjust N and verifier quality to explore the trade-space.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-[var(--ds-paper)] border border-[var(--ds-rule)] rounded p-5 space-y-5">
            <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--ds-ink)]">Controls</h3>
            <SliderField
              label="Sample count N"
              min={1}
              max={128}
              step={1}
              value={n}
              onChange={setN}
              hint="Higher N increases cost linearly but accuracy sub-logarithmically."
            />
            <SliderField
              label="Verifier Quality"
              min={0.2}
              max={1.0}
              step={0.05}
              value={verifierQuality}
              onChange={setVerifierQuality}
              formatFn={(v) => `${Math.round(v * 100)}%`}
              hint="1.0 = perfect oracle; 0.2 = random selection."
            />
          </div>

          <div className="md:col-span-2 space-y-3">
            <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block">Live Accuracy Forecast</span>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {[
                { label: 'Expected (no selection)', value: currentRow.expectedAcc, color: 'bg-stone-200' },
                { label: 'Oracle Bound', value: currentRow.oracleBound, color: 'bg-blue-200' },
                {
                  label: `Effective (verifier ${Math.round(verifierQuality * 100)}%)`,
                  value: currentRow.effectiveAcc,
                  color: 'bg-amber-200',
                },
              ].map((item) => (
                <div key={item.label} className="bg-[var(--ds-paper)] border border-[var(--ds-rule)] rounded p-4 text-center space-y-2">
                  <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block">{item.label}</span>
                  <span className="text-3xl font-bold font-mono text-[var(--ds-ink)]">{item.value}%</span>
                  <div className="w-full h-2 bg-[var(--ds-rule)] rounded overflow-hidden">
                    <div
                      className={`h-2 rounded transition-all duration-300 ${item.color}`}
                      style={{ width: `${(item.value / maxAccBar) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            <div className="bg-[var(--ds-panel)] border border-[var(--ds-rule)] rounded p-4 space-y-2">
              <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block">N vs Accuracy curve</span>
              <div className="flex items-end gap-1 h-24">
                {BON_STRATEGY_DATA.map((row) => {
                  const isActive = Math.abs(row.n - n) < n * 0.25 || row.n === n;
                  return (
                    <div
                      key={row.n}
                      className="flex flex-col items-center gap-1 flex-1 min-w-0"
                    >
                      <div
                        className={`w-full rounded-sm transition-all ${isActive ? 'bg-[var(--ds-accent)]' : 'bg-stone-300'}`}
                        style={{ height: `${(row.expectedAcc / 100) * 90}px` }}
                      />
                    </div>
                  );
                })}
              </div>
              <div className="flex gap-1">
                {BON_STRATEGY_DATA.map((row) => (
                  <span key={row.n} className="text-[9px] font-mono text-[var(--ds-faint)] flex-1 text-center truncate">
                    {row.n}
                  </span>
                ))}
              </div>
              <p className="text-[10px] text-[var(--ds-faint)]">
                Accuracy gains from sampling N=1 to 32 are large; beyond 32 returns diminish steeply.
              </p>
            </div>
          </div>
        </div>

        {n > 32 && (
          <div className="flex items-center gap-3 p-3 rounded border border-amber-300 bg-amber-50 text-amber-900 text-xs">
            <AlertTriangle className="w-4 h-4 shrink-0 text-amber-600" />
            <span>
              <strong>Cost warning:</strong> N={n} means {n}x the token cost of single-sample generation. Gains above N=32 are often marginal without a better verifier.
            </span>
          </div>
        )}
      </InfoCard>
    </div>
  );
}

// Tab: Tree / Beam Search

function TabBeamSearch() {
  const [currentDepth, setCurrentDepth] = useState(1);
  const visibleSteps = BEAM_SEARCH_STEPS.slice(0, currentDepth + 1);

  const depthColors = ['bg-[var(--ds-panel)]', 'bg-blue-50', 'bg-amber-50', 'bg-green-50'];
  const depthBorders = ['border-[var(--ds-rule)]', 'border-blue-200', 'border-amber-200', 'border-green-300'];

  return (
    <div className="space-y-6">
      <InfoCard>
        <SectionHeader icon={GitBranch} title="Tree Search / Beam Search Walkthrough" />
        <p className="text-xs text-[var(--ds-faint)] leading-relaxed">
          Beam search expands a tree of partial reasoning paths. A process reward model (PRM) scores each node.
          Low-scoring branches are pruned; high-scoring ones are expanded further. Step through the tree below.
        </p>

        <div className="flex items-center gap-3">
          <button
            data-math-control
            onClick={() => setCurrentDepth(Math.max(0, currentDepth - 1))}
            disabled={currentDepth === 0}
            className="ds-btn bg-[var(--ds-paper)] border border-[var(--ds-rule)] text-xs font-bold py-1.5 px-3 rounded transition-all disabled:opacity-40"
          >
            Collapse
          </button>
          <span className="text-xs font-mono text-[var(--ds-faint)]">Depth: {currentDepth} / {BEAM_SEARCH_STEPS.length - 1}</span>
          <button
            data-math-control
            onClick={() => setCurrentDepth(Math.min(BEAM_SEARCH_STEPS.length - 1, currentDepth + 1))}
            disabled={currentDepth >= BEAM_SEARCH_STEPS.length - 1}
            className="ds-btn bg-[var(--ds-accent)] text-[var(--ds-paper)] text-xs font-bold py-1.5 px-3 rounded transition-all disabled:opacity-40"
          >
            Expand
          </button>
        </div>

        <div className="space-y-3">
          {visibleSteps.map((step, depthIdx) => (
            <div key={depthIdx} className="space-y-2">
              <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block">Depth {depthIdx}</span>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                {step.nodes.map((node) => (
                  <div
                    key={node.id}
                    className={`p-3 border rounded text-xs transition-all ${
                      node.pruned
                        ? 'opacity-40 border-red-200 bg-red-50 line-through'
                        : `${depthColors[depthIdx] || ''} ${depthBorders[depthIdx] || 'border-[var(--ds-rule)]'}`
                    }`}
                  >
                    <p className="text-[var(--ds-ink)] leading-relaxed">{node.text}</p>
                    {node.score !== null && (
                      <div className="mt-2 flex items-center gap-2">
                        <span className="text-[10px] text-[var(--ds-faint)]">PRM Score:</span>
                        <span
                          className={`font-mono font-bold text-[10px] ${
                            node.pruned ? 'text-red-700' : node.score >= 0.9 ? 'text-green-700' : 'text-amber-700'
                          }`}
                        >
                          {node.pruned ? 'PRUNED' : node.score.toFixed(2)}
                        </span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="text-xs text-[var(--ds-faint)] bg-[var(--ds-paper)] border border-[var(--ds-rule)] rounded p-3 leading-relaxed">
          <strong>Key insight:</strong> Beam search is more compute-efficient than Best-of-N because it allocates 
          search effort to promising paths only. But it requires a PRM that can score partial reasoning chains;
          not just final answers.
        </div>
      </InfoCard>
    </div>
  );
}

// Tab: Thinking Budgets

function TabThinkingBudget() {
  const [selectedId, setSelectedId] = useState('moderate');
  const example = THINKING_BUDGET_EXAMPLES.find((e) => e.id === selectedId);

  const difficultyColors = {
    trivial: 'text-green-700',
    easy: 'text-green-600',
    medium: 'text-amber-600',
    hard: 'text-red-600',
  };

  return (
    <div className="space-y-6">
      <InfoCard>
        <SectionHeader icon={Clock} title="Thinking Budget Examples" />
        <p className="text-xs text-[var(--ds-faint)] leading-relaxed">
          The optimal thinking budget is problem-dependent. Allocating too few tokens truncates reasoning; 
          too many wastes compute on trivial queries. Explore how different query types require different budgets.
        </p>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {THINKING_BUDGET_EXAMPLES.map((ex) => (
            <button
              key={ex.id}
              data-math-control
              onClick={() => setSelectedId(ex.id)}
              className={`p-3 border rounded text-left text-xs transition-all ${
                selectedId === ex.id
                  ? 'border-[var(--ds-accent)] bg-[var(--ds-warm)]'
                  : 'border-[var(--ds-rule)] bg-[var(--ds-paper)] hover:bg-[var(--ds-paper-2)]'
              }`}
            >
              <span className="font-bold block text-[var(--ds-ink)]">{ex.label}</span>
              <span className={`text-[10px] font-bold block mt-0.5 ${difficultyColors[ex.difficulty] || ''}`}>
                {ex.difficulty}
              </span>
              <span className="text-[10px] text-[var(--ds-faint)] block mt-1">Optimal: ~{ex.optimalTokens} tokens</span>
            </button>
          ))}
        </div>

        {example && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="md:col-span-2 space-y-3">
              <div>
                <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block mb-1">Query</span>
                <p className="text-xs font-mono bg-stone-900 text-stone-200 p-3 rounded leading-relaxed">{example.query}</p>
              </div>
              <div>
                <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block mb-1">Generated Reasoning Trace</span>
                <pre className="text-[11px] font-mono bg-stone-900 text-stone-200 p-4 rounded overflow-x-auto max-h-48 leading-relaxed whitespace-pre-wrap">
                  {example.trace}
                </pre>
              </div>
            </div>

            <div className="space-y-4 bg-[var(--ds-paper)] border border-[var(--ds-rule)] rounded p-4">
              <div>
                <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block mb-2">Budget Analysis</span>
                <div className="space-y-1.5 text-xs font-mono">
                  <div className="flex justify-between border-b border-[var(--ds-rule)] pb-1">
                    <span className="text-[var(--ds-faint)]">Difficulty:</span>
                    <span className={`font-bold ${difficultyColors[example.difficulty] || ''}`}>{example.difficulty}</span>
                  </div>
                  <div className="flex justify-between border-b border-[var(--ds-rule)] pb-1">
                    <span className="text-[var(--ds-faint)]">Optimal tokens:</span>
                    <span className="font-bold">{example.optimalTokens}</span>
                  </div>
                  <div className="flex justify-between border-b border-[var(--ds-rule)] pb-1">
                    <span className="text-[var(--ds-faint)]">Wasted tokens:</span>
                    <span className={`font-bold ${example.wastedTokens > 0 ? 'text-red-700' : 'text-green-700'}`}>
                      {example.wastedTokens}
                    </span>
                  </div>
                </div>
              </div>
              <div>
                <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block mb-1">Insight</span>
                <p className="text-xs text-[var(--ds-ink)] leading-relaxed">{example.note}</p>
              </div>
            </div>
          </div>
        )}
      </InfoCard>
    </div>
  );
}

// Tab: Budget Forcing

function TabBudgetForcing() {
  const [budget, setBudget] = useState(512);

  const closest = useMemo(() => {
    return BUDGET_FORCING_DATA.reduce((prev, curr) =>
      Math.abs(curr.budget - budget) < Math.abs(prev.budget - budget) ? curr : prev
    );
  }, [budget]);

  const isOverthinking = budget >= 4096;
  const isTruncated = budget <= 128;

  return (
    <div className="space-y-6">
      <InfoCard>
        <SectionHeader icon={Sliders} title="Budget Forcing Simulator" />
        <p className="text-xs text-[var(--ds-faint)] leading-relaxed">
          Budget forcing caps the number of thinking tokens available per query. 
          Explore how latency and accuracy respond as the thinking budget changes.
        </p>

        {isTruncated && (
          <div className="flex items-center gap-3 p-3 rounded border border-red-300 bg-red-50 text-red-900 text-xs">
            <AlertTriangle className="w-4 h-4 shrink-0 text-red-600" />
            <span>
              <strong>Truncation risk:</strong> At {budget} tokens, complex reasoning chains will be cut off before completion. 
              Hard problems will show degraded accuracy.
            </span>
          </div>
        )}

        {isOverthinking && (
          <div className="flex items-center gap-3 p-3 rounded border border-amber-300 bg-amber-50 text-amber-900 text-xs">
            <AlertTriangle className="w-4 h-4 shrink-0 text-amber-600" />
            <span>
              <strong>Diminishing returns:</strong> Beyond 4k tokens, accuracy gains are minimal while latency continues to grow linearly.
            </span>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-[var(--ds-paper)] border border-[var(--ds-rule)] rounded p-5 space-y-5">
            <h3 className="text-xs font-bold uppercase tracking-wider">Thinking Token Budget</h3>
            <SliderField
              label="Max thinking tokens"
              min={64}
              max={8192}
              step={64}
              value={budget}
              onChange={setBudget}
              hint="Slide to set the hard cap on reasoning tokens per query."
            />
            <div className="border-t border-[var(--ds-rule)] pt-4 text-xs space-y-1.5 font-mono">
              <div className="flex justify-between">
                <span className="text-[var(--ds-faint)]">Budget:</span>
                <span className="font-bold">{budget} tokens</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[var(--ds-faint)]">Accuracy:</span>
                <span className="font-bold text-[var(--ds-accent)]">{closest.accuracy}%</span>
              </div>
              <div className="flex justify-between border-t border-[var(--ds-rule)] pt-1">
                <span className="text-[var(--ds-faint)]">Latency:</span>
                <span className="font-bold">{(closest.latencyMs / 1000).toFixed(1)}s</span>
              </div>
            </div>
          </div>

          <div className="md:col-span-2 space-y-3">
            <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block">Budget vs Accuracy & Latency</span>
            <div className="space-y-2">
              {BUDGET_FORCING_DATA.map((row) => {
                const isCurrent = row.budget === closest.budget;
                return (
                  <div
                    key={row.budget}
                    className={`grid grid-cols-4 gap-3 items-center text-xs p-2 rounded border transition-all ${
                      isCurrent ? 'border-[var(--ds-accent)] bg-[var(--ds-warm)]' : 'border-[var(--ds-rule)] bg-[var(--ds-paper)]'
                    }`}
                  >
                    <span className="font-mono font-bold text-[var(--ds-ink)]">{row.budget} tok</span>
                    <div className="flex items-center gap-2 col-span-2">
                      <div className="flex-1 h-2 bg-[var(--ds-rule)] rounded overflow-hidden">
                        <div
                          className="h-2 bg-[var(--ds-accent)] rounded transition-all"
                          style={{ width: `${row.accuracy}%` }}
                        />
                      </div>
                      <span className="font-mono font-bold w-10 text-right">{row.accuracy}%</span>
                    </div>
                    <span className="text-[var(--ds-faint)] text-right">{(row.latencyMs / 1000).toFixed(1)}s</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </InfoCard>
    </div>
  );
}

// Tab: Tool-Augmented

function TabToolAugmented() {
  const [visibleSteps, setVisibleSteps] = useState(1);
  const allSteps = REACT_AGENT_STEPS;

  const typeConfig = {
    thought: { bg: 'bg-blue-50', border: 'border-blue-200', icon: Brain, label: 'THOUGHT' },
    action: { bg: 'bg-amber-50', border: 'border-amber-200', icon: Wrench, label: 'ACTION' },
    observation: { bg: 'bg-green-50', border: 'border-green-200', icon: Activity, label: 'OBSERVATION' },
    answer: { bg: 'bg-[var(--ds-warm)]', border: 'border-[var(--ds-accent)]', icon: Sparkles, label: 'FINAL ANSWER' },
  };

  return (
    <div className="space-y-6">
      <InfoCard>
        <SectionHeader icon={Wrench} title="Tool-Augmented Compute (ReAct Pattern)" />
        <p className="text-xs text-[var(--ds-faint)] leading-relaxed">
          Instead of reasoning through all sub-tasks in tokens, the model calls external tools (search, calculator, code interpreter).
          Tool results are injected as observations. This extends effective compute beyond the model's parameter-bound knowledge.
        </p>
        <p className="text-xs font-mono bg-stone-900 text-stone-200 p-2 rounded mb-4">
          ReAct Loop: Thought to Action(tool) to Observation to Thought to Final Answer
        </p>

        <div className="flex items-center gap-3 mb-2">
          <button
            data-math-control
            onClick={() => setVisibleSteps(Math.max(1, visibleSteps - 1))}
            disabled={visibleSteps <= 1}
            className="ds-btn bg-[var(--ds-paper)] border border-[var(--ds-rule)] text-xs font-bold py-1.5 px-3 rounded disabled:opacity-40"
          >
            Back
          </button>
          <span className="text-xs font-mono text-[var(--ds-faint)]">Step {visibleSteps} / {allSteps.length}</span>
          <button
            data-math-control
            onClick={() => setVisibleSteps(Math.min(allSteps.length, visibleSteps + 1))}
            disabled={visibleSteps >= allSteps.length}
            className="ds-btn bg-[var(--ds-accent)] text-[var(--ds-paper)] text-xs font-bold py-1.5 px-3 rounded disabled:opacity-40"
          >
            Next Step
          </button>
        </div>

        <div className="space-y-2">
          {allSteps.slice(0, visibleSteps).map((step) => {
            const cfg = typeConfig[step.type] || typeConfig.thought;
            const Icon = cfg.icon;
            return (
              <div key={step.id} className={`p-3 border rounded text-xs ${cfg.bg} ${cfg.border}`}>
                <div className="flex items-center gap-2 mb-1">
                  <Icon className="w-3.5 h-3.5" />
                  <span className="text-[10px] font-bold uppercase tracking-wider">{cfg.label}</span>
                  {step.toolName && (
                    <span className="text-[9px] font-mono px-1.5 py-0.5 bg-white/60 border border-current rounded">
                      {step.toolName}
                    </span>
                  )}
                </div>
                <p className="font-mono leading-relaxed">{step.content}</p>
              </div>
            );
          })}
        </div>

        {visibleSteps >= allSteps.length && (
          <div className="text-xs text-[var(--ds-faint)] bg-[var(--ds-paper)] border border-[var(--ds-rule)] rounded p-3 leading-relaxed">
            <strong>Key insight:</strong> Tool calls consumed 3 real-world API round-trips but saved ~1 500 reasoning tokens.
            The model did not need to memorize population numbers; it delegated fact retrieval to a search tool.
          </div>
        )}
      </InfoCard>
    </div>
  );
}

// Tab: Failure Modes

function TabFailureModes() {
  const [activeId, setActiveId] = useState('overthinking');
  const [fixApplied, setFixApplied] = useState({});
  const failure = TTC_FAILURE_MODES.find((f) => f.id === activeId);

  const severityColors = {
    high: { badge: 'bg-red-100 text-red-800', dot: 'bg-red-500' },
    medium: { badge: 'bg-amber-100 text-amber-800', dot: 'bg-amber-500' },
  };

  return (
    <div className="space-y-6">
      <InfoCard>
        <SectionHeader icon={AlertTriangle} title="Test-Time Compute Failure Modes" />
        <p className="text-xs text-[var(--ds-faint)] leading-relaxed">
          Each TTC strategy introduces new failure modes. Click each to explore its symptom, trigger, and recommended fix.
        </p>

        <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
          {TTC_FAILURE_MODES.map((f) => {
            const sv = severityColors[f.severity] || severityColors.medium;
            return (
              <button
                key={f.id}
                data-math-control
                onClick={() => setActiveId(f.id)}
                className={`p-3 border rounded text-left text-xs transition-all ${
                  activeId === f.id
                    ? 'border-[var(--ds-accent)] bg-[var(--ds-warm)]'
                    : 'border-[var(--ds-rule)] bg-[var(--ds-paper)] hover:bg-[var(--ds-paper-2)]'
                }`}
              >
                <span className="flex items-center gap-1.5 mb-1">
                  <span className={`w-2 h-2 rounded-full ${sv.dot}`} />
                  <span className={`text-[9px] font-bold uppercase ${sv.badge.split(' ')[1]}`}>{f.severity}</span>
                </span>
                <span className="font-bold text-[var(--ds-ink)] block">{f.name}</span>
              </button>
            );
          })}
        </div>

        {failure && (
          <div className="bg-[var(--ds-paper)] border border-[var(--ds-rule)] rounded p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-[var(--ds-ink)]">{failure.name}</h3>
              <Pill color={failure.severity === 'high' ? 'red' : 'amber'}>{failure.severity} severity</Pill>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              <div className="space-y-1">
                <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block">Symptom</span>
                <p className="text-[var(--ds-ink)] leading-relaxed">{failure.description}</p>
              </div>
              <div className="space-y-1">
                <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block">Signal</span>
                <p className="text-[var(--ds-ink)] leading-relaxed italic">{failure.signal}</p>
              </div>
              <div className="space-y-2">
                <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block">Fix</span>
                <p className={`text-[var(--ds-ink)] leading-relaxed ${fixApplied[failure.id] ? 'line-through opacity-50' : ''}`}>
                  {failure.fix}
                </p>
                <button
                  data-math-control
                  onClick={() => setFixApplied((prev) => ({ ...prev, [failure.id]: !prev[failure.id] }))}
                  className="ds-btn text-[10px] font-bold uppercase py-1 px-2 rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] hover:bg-[var(--ds-paper-2)]"
                >
                  {fixApplied[failure.id] ? 'Unfixed' : 'Mark Fixed'}
                </button>
              </div>
            </div>
          </div>
        )}
      </InfoCard>
    </div>
  );
}

// Tab: Cost vs Accuracy

function TabMetrics() {
  const [sortBy, setSortBy] = useState('accuracy');
  const sorted = useMemo(() => {
    return [...METRICS_COMPARISON].sort((a, b) =>
      sortBy === 'accuracy' ? b.accuracy - a.accuracy : a.avgTokens - b.avgTokens
    );
  }, [sortBy]);

  const maxAcc = Math.max(...METRICS_COMPARISON.map((r) => r.accuracy));

  return (
    <div className="space-y-6">
      <InfoCard>
        <SectionHeader icon={BarChart2} title="Strategy Cost vs Accuracy Comparison" />
        <p className="text-xs text-[var(--ds-faint)] leading-relaxed">
          Different test-time compute strategies occupy different points on the cost-accuracy Pareto frontier.
          The best strategy depends on your latency SLA and per-query budget.
        </p>

        <div className="flex items-center gap-3">
          <span className="text-xs font-bold text-[var(--ds-faint)]">Sort by:</span>
          {['accuracy', 'tokens'].map((s) => (
            <button
              key={s}
              data-math-control
              onClick={() => setSortBy(s)}
              className={`text-xs font-bold py-1 px-3 rounded border transition-all ${
                sortBy === s
                  ? 'border-[var(--ds-accent)] bg-[var(--ds-warm)] text-[var(--ds-ink)]'
                  : 'border-[var(--ds-rule)] bg-[var(--ds-paper)] text-[var(--ds-faint)]'
              }`}
            >
              {s === 'accuracy' ? 'Accuracy desc' : 'Token Cost asc'}
            </button>
          ))}
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="border-b border-[var(--ds-rule)]">
                {['Model', 'Strategy', 'Accuracy', 'Avg Tokens', 'Latency', 'Cost/Query'].map((h) => (
                  <th key={h} className="text-left py-2 px-3 text-[10px] font-bold uppercase text-[var(--ds-faint)] tracking-wider">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((row, i) => (
                <tr
                  key={i}
                  className={`border-b border-[var(--ds-rule)] transition-colors ${
                    i === 0 ? 'bg-[var(--ds-warm)]' : 'hover:bg-[var(--ds-paper-2)]'
                  }`}
                >
                  <td className="py-2.5 px-3 font-bold text-[var(--ds-ink)]">{row.model}</td>
                  <td className="py-2.5 px-3 text-[var(--ds-faint)]">{row.ttcStrategy}</td>
                  <td className="py-2.5 px-3">
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 bg-[var(--ds-rule)] rounded overflow-hidden">
                        <div
                          className="h-1.5 bg-[var(--ds-accent)] rounded"
                          style={{ width: `${(row.accuracy / maxAcc) * 100}%` }}
                        />
                      </div>
                      <span className="font-mono font-bold">{row.accuracy}%</span>
                    </div>
                  </td>
                  <td className="py-2.5 px-3 font-mono">{row.avgTokens}</td>
                  <td className="py-2.5 px-3 font-mono">{(row.latencyMs / 1000).toFixed(1)}s</td>
                  <td className="py-2.5 px-3 font-mono">{row.costPerQuery}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="text-[10px] text-[var(--ds-faint)] leading-relaxed">
          Numbers are illustrative benchmarks (MATH-500 level difficulty, frontier reasoning model). 
          Actual figures vary by model, hardware, and prompt distribution.
        </p>
      </InfoCard>
    </div>
  );
}

// Tab: Budget Policies

function TabBudgetStrategies() {
  const [selected, setSelected] = useState('adaptive');

  const strat = THINKING_BUDGET_STRATEGIES.find((s) => s.id === selected);

  return (
    <div className="space-y-6">
      <InfoCard>
        <SectionHeader icon={Settings} title="Thinking Budget Allocation Policies" />
        <p className="text-xs text-[var(--ds-faint)] leading-relaxed">
          Production reasoning systems need a policy for allocating thinking tokens.
          Each policy suits a different deployment context.
        </p>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {THINKING_BUDGET_STRATEGIES.map((s) => (
            <button
              key={s.id}
              data-math-control
              onClick={() => setSelected(s.id)}
              className={`p-3 border rounded text-left text-xs transition-all ${
                selected === s.id
                  ? 'border-[var(--ds-accent)] bg-[var(--ds-warm)]'
                  : 'border-[var(--ds-rule)] bg-[var(--ds-paper)] hover:bg-[var(--ds-paper-2)]'
              }`}
            >
              <span className="font-bold block text-[var(--ds-ink)]">{s.label}</span>
              <span className="text-[10px] text-[var(--ds-faint)] mt-1 block">{s.description.slice(0, 55)}...</span>
            </button>
          ))}
        </div>

        {strat && (
          <div className="bg-[var(--ds-paper)] border border-[var(--ds-rule)] rounded p-5 grid grid-cols-1 md:grid-cols-3 gap-5">
            <div className="md:col-span-2 space-y-3">
              <h3 className="text-sm font-bold text-[var(--ds-ink)]">{strat.label}</h3>
              <p className="text-xs text-[var(--ds-faint)] leading-relaxed">{strat.description}</p>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-[10px] font-bold uppercase text-green-700 block mb-1">Pros</span>
                  {strat.pros.map((p, i) => (
                    <div key={i} className="flex items-start gap-1.5 text-xs text-[var(--ds-ink)] mb-1">
                      <span className="text-green-600 mt-0.5">+</span>
                      {p}
                    </div>
                  ))}
                </div>
                <div>
                  <span className="text-[10px] font-bold uppercase text-red-700 block mb-1">Cons</span>
                  {strat.cons.map((c, i) => (
                    <div key={i} className="flex items-start gap-1.5 text-xs text-[var(--ds-ink)] mb-1">
                      <span className="text-red-600 mt-0.5">-</span>
                      {c}
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="bg-[var(--ds-panel)] border border-[var(--ds-rule)] rounded p-4">
              <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block mb-2">Best For</span>
              <p className="text-xs text-[var(--ds-ink)] leading-relaxed">{strat.bestFor}</p>
            </div>
          </div>
        )}
      </InfoCard>
    </div>
  );
}

// Root Component

export default function TestTimeComputeThinkingBudgets() {
  const [activeTab, setActiveTab] = useState('scaling-regimes');

  const tabPanels = {
    'scaling-regimes': <TabScalingRegimes />,
    'strategy-map': <TabStrategyMap />,
    'best-of-n': <TabBestOfN />,
    'beam-search': <TabBeamSearch />,
    'thinking-budget': <TabThinkingBudget />,
    'budget-forcing': <TabBudgetForcing />,
    'tool-augmented': <TabToolAugmented />,
    'failure-modes': <TabFailureModes />,
    'metrics': <TabMetrics />,
    'budget-strategies': <TabBudgetStrategies />,
  };

  return (
    <div className="flex flex-col min-h-screen bg-[var(--ds-paper)] text-[var(--ds-ink)] font-sans antialiased">
      {/* Header */}
      <div className="border-b border-[var(--ds-rule)] bg-[var(--ds-panel)] p-6">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2 py-0.5 text-[10px] font-bold tracking-wider uppercase bg-[var(--ds-warm)] border border-[var(--ds-rule)] text-[var(--ds-ink)] rounded">
                Frontier LLMs
              </span>
              <span className="px-2 py-0.5 text-[10px] font-bold tracking-wider uppercase bg-amber-500 text-white rounded">
                Advanced Module
              </span>
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-[var(--ds-ink)] font-display">
              Test-Time Compute &amp; Thinking Budgets
            </h1>
            <p className="text-xs text-[var(--ds-faint)] mt-1 max-w-2xl">
              Modern reasoning systems scale at inference time by spending more compute on harder problems.
              Explore Best-of-N sampling, tree search, budget forcing, tool use, and adaptive thinking policies.
            </p>
          </div>
          <div className="flex items-center gap-4 bg-[var(--ds-paper)] p-3 border border-[var(--ds-rule)] rounded">
            <div className="text-center border-r border-[var(--ds-rule)] pr-4">
              <span className="block text-[10px] font-bold text-[var(--ds-faint)] uppercase">Core Insight</span>
              <span className="text-lg font-bold text-[var(--ds-accent)]">Tokens = Compute</span>
            </div>
            <div className="text-center pl-2">
              <span className="block text-[10px] font-bold text-[var(--ds-faint)] uppercase">Key Trade-off</span>
              <span className="text-lg font-bold text-[var(--ds-ink)]">Latency vs Accuracy</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-[var(--ds-rule)] bg-[var(--ds-panel)] sticky top-0 z-30 overflow-x-auto">
        <div className="max-w-7xl mx-auto flex">
          {TABS.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                data-math-control
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-5 py-4 text-xs font-bold whitespace-nowrap transition-all border-b-2 ${
                  isActive
                    ? 'border-[var(--ds-accent)] text-[var(--ds-accent)] bg-[var(--ds-paper)]'
                    : 'border-transparent text-[var(--ds-faint)] hover:text-[var(--ds-ink)] hover:bg-[var(--ds-paper-2)]'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 max-w-7xl w-full mx-auto p-6">
        {tabPanels[activeTab]}
      </div>
    </div>
  );
}
