import React, { useMemo, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Code2,
  Cpu,
  Database,
  FileSearch,
  GitBranch,
  GitPullRequest,
  Lock,
  RotateCcw,
  Search,
  ShieldCheck,
  Terminal,
  Users,
  XCircle,
} from 'lucide-react';
import {
  CODING_AGENT_STAGES,
  FAILURE_SCENARIOS,
  MODULES,
  PRODUCT_CARDS,
} from './data';

const REPO_FILES = [
  { path: 'src/parser/token_stream.py', hit: 95, kind: 'source' },
  { path: 'tests/parser/test_nested.py', hit: 91, kind: 'test' },
  { path: 'src/parser/whitespace.py', hit: 77, kind: 'source' },
  { path: 'docs/parser_grammar.md', hit: 58, kind: 'doc' },
  { path: 'src/cli/format.py', hit: 24, kind: 'noise' },
];

const COMMANDS = [
  { command: 'rg "nested parentheses" src tests', status: 'allowed', icon: Search },
  { command: 'pytest tests/parser/test_nested.py', status: 'allowed', icon: Terminal },
  { command: 'npm install parser-plugin', status: 'approval', icon: Lock },
  { command: 'git push origin main', status: 'blocked', icon: XCircle },
];

const TEST_ROWS = [
  { name: 'FAIL_TO_PASS parser crash', before: 'fail', after: 'pass', weight: 34 },
  { name: 'PASS_TO_PASS token stream', before: 'pass', after: 'pass', weight: 22 },
  { name: 'affected parser module', before: 'pass', after: 'pass', weight: 18 },
  { name: 'lint/type smoke', before: 'pass', after: 'pass', weight: 12 },
  { name: 'slow integration suite', before: 'skip', after: 'skip', weight: 6 },
];

const PLAN_STEPS = [
  'Reproduce the nested-parentheses parser crash with a failing test.',
  'Use the stack trace to inspect token consumption around closing tokens.',
  'Patch whitespace handling without rewriting the parser.',
  'Run targeted parser tests, then affected regression tests.',
  'Review the diff for unrelated files before PR submission.',
];

function pct(value) {
  return `${Math.max(0, Math.min(100, Math.round(value)))}%`;
}

function MetricBar({ label, value, tone = 'bg-[var(--ds-accent)]' }) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between gap-3 text-xs">
        <span className="font-semibold text-[var(--ds-ink)]">{label}</span>
        <span className="font-mono text-[var(--ds-faint)]">{pct(value)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded bg-[var(--ds-paper-2)] border border-[var(--ds-rule)]">
        <div className={`h-full ${tone}`} style={{ width: pct(value) }} />
      </div>
    </div>
  );
}

function StatusPill({ status }) {
  const classes = {
    pass: 'border-emerald-300 bg-emerald-50 text-emerald-900',
    fail: 'border-rose-300 bg-rose-50 text-rose-900',
    skip: 'border-slate-300 bg-slate-50 text-slate-700',
    allowed: 'border-emerald-300 bg-emerald-50 text-emerald-900',
    approval: 'border-amber-300 bg-amber-50 text-amber-900',
    blocked: 'border-rose-300 bg-rose-50 text-rose-900',
  };
  return (
    <span className={`inline-flex rounded border px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide ${classes[status]}`}>
      {status}
    </span>
  );
}

function StageMap({ activeStage }) {
  return (
    <div className="grid gap-2 md:grid-cols-7">
      {CODING_AGENT_STAGES.map((stage, index) => {
        const isActive = stage.id === activeStage;
        return (
          <div
            key={stage.id}
            className={`border p-3 transition ${
              isActive
                ? 'border-[var(--ds-accent)] bg-[var(--ds-paper)] shadow-sm'
                : 'border-[var(--ds-rule)] bg-[var(--ds-panel)]'
            }`}
          >
            <div className="mb-2 flex items-center justify-between gap-2">
              <span className="font-mono text-[10px] text-[var(--ds-faint)]">{String(index + 1).padStart(2, '0')}</span>
              {isActive ? <CheckCircle2 className="h-4 w-4 text-[var(--ds-accent)]" /> : null}
            </div>
            <h3 className="text-sm font-bold text-[var(--ds-ink)]">{stage.label}</h3>
            <p className="mt-2 text-xs text-[var(--ds-faint)]">{stage.tool}</p>
          </div>
        );
      })}
    </div>
  );
}

function IssuePatchLoop({ activeStage, setActiveStage }) {
  return (
    <div className="grid gap-4 lg:grid-cols-[1.05fr_0.95fr]">
      <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
        <div className="mb-4 flex items-center gap-2">
          <GitBranch className="h-4 w-4 text-[var(--ds-accent)]" />
          <span className="text-xs font-bold uppercase tracking-wider text-[var(--ds-faint)]">Issue to patch loop</span>
        </div>
        <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-[var(--ds-faint)]">Issue</p>
          <p className="mt-2 text-lg font-bold text-[var(--ds-ink)]">
            Parser crashes on nested parentheses when whitespace appears before closing token.
          </p>
        </div>

        <div className="mt-4 grid gap-2">
          {CODING_AGENT_STAGES.map((stage) => (
            <button
              key={stage.id}
              data-math-control
              onClick={() => setActiveStage(stage.id)}
              className={`flex items-center justify-between border px-3 py-2 text-left text-sm transition ${
                activeStage === stage.id
                  ? 'border-[var(--ds-accent)] bg-[var(--ds-paper)] text-[var(--ds-ink)]'
                  : 'border-[var(--ds-rule)] bg-transparent text-[var(--ds-faint)] hover:bg-[var(--ds-paper)]'
              }`}
            >
              <span>{stage.label}</span>
              <span className="font-mono text-[10px]">{stage.output}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
        <div className="mb-3 flex items-center gap-2">
          <Code2 className="h-4 w-4 text-[var(--ds-accent)]" />
          <span className="text-xs font-bold uppercase tracking-wider text-[var(--ds-faint)]">Live artifact</span>
        </div>
        <div className="font-mono text-xs">
          <div className="border border-[var(--ds-rule)] bg-slate-950 p-3 text-slate-100">
            <p className="text-emerald-300">+ def consume_closing(self):</p>
            <p className="text-slate-300">+     self.skip_whitespace()</p>
            <p className="text-emerald-300">+     return self.expect(TOKEN_CLOSE)</p>
            <p className="mt-3 text-rose-300">- return self.expect(TOKEN_CLOSE)</p>
          </div>
        </div>
        <div className="mt-4 grid gap-3">
          <MetricBar label="Issue understanding" value={activeStage === 'understand' ? 86 : 72} />
          <MetricBar label="Relevant file recall" value={activeStage === 'navigate' ? 91 : 68} tone="bg-emerald-600" />
          <MetricBar label="Patch confidence" value={activeStage === 'review' || activeStage === 'submit' ? 84 : 57} tone="bg-amber-600" />
          <MetricBar label="Scope drift risk" value={activeStage === 'edit' ? 29 : 14} tone="bg-rose-600" />
        </div>
      </div>
    </div>
  );
}

function ModulePanel({ module }) {
  const iconMap = {
    'system-map': Cpu,
    'swe-bench': CheckCircle2,
    'repo-navigation': FileSearch,
    'plan-edit-test': Code2,
    sandbox: ShieldCheck,
    'diff-review': GitPullRequest,
    memory: Database,
    checkpoints: RotateCcw,
    'multi-agent': Users,
    'regression-filtering': Terminal,
    approval: Lock,
    'failure-modes': AlertTriangle,
  };
  const Icon = iconMap[module.id] || Cpu;

  return (
    <section className="grid gap-4 lg:grid-cols-[0.85fr_1.15fr]">
      <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
        <div className="mb-3 flex items-center gap-2">
          <Icon className="h-5 w-5 text-[var(--ds-accent)]" />
          <h2 className="text-lg font-bold text-[var(--ds-ink)]">{module.title}</h2>
        </div>
        <p className="text-sm leading-6 text-[var(--ds-muted)]">{module.purpose}</p>
        <div className="mt-4 border-l-2 border-amber-500 bg-amber-50 p-3 text-sm text-amber-950">
          {module.misconception}
        </div>
        <div className="mt-4 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3">
          <p className="text-xs font-bold uppercase tracking-wide text-[var(--ds-faint)]">Practice lab</p>
          <p className="mt-2 text-sm text-[var(--ds-ink)]">{module.lab}</p>
        </div>
      </div>

      <div className="grid gap-4">
        {module.id === 'repo-navigation' ? <RepoNavigation /> : null}
        {module.id === 'sandbox' || module.id === 'approval' ? <SandboxPanel /> : null}
        {module.id === 'diff-review' ? <DiffReview /> : null}
        {module.id === 'regression-filtering' || module.id === 'swe-bench' ? <TestMatrix /> : null}
        {module.id === 'checkpoints' ? <CheckpointPanel /> : null}
        {module.id === 'multi-agent' ? <MultiAgentPanel /> : null}
        {module.id === 'failure-modes' ? <FailurePanel /> : null}
        {module.id === 'memory' ? <MemoryPanel /> : null}
        {module.id === 'plan-edit-test' || module.id === 'system-map' ? <PlanPanel /> : null}
      </div>
    </section>
  );
}

function RepoNavigation() {
  return (
    <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
      <p className="mb-3 text-xs font-bold uppercase tracking-wide text-[var(--ds-faint)]">Context selection</p>
      <div className="space-y-2">
        {REPO_FILES.map((file) => (
          <div key={file.path} className="grid grid-cols-[1fr_auto] items-center gap-3 border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3">
            <div>
              <p className="font-mono text-xs text-[var(--ds-ink)]">{file.path}</p>
              <p className="text-[10px] uppercase tracking-wide text-[var(--ds-faint)]">{file.kind}</p>
            </div>
            <span className="font-mono text-xs font-bold text-[var(--ds-accent)]">{file.hit}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function PlanPanel() {
  return (
    <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
      <p className="mb-3 text-xs font-bold uppercase tracking-wide text-[var(--ds-faint)]">Minimal patch plan</p>
      <ol className="space-y-2">
        {PLAN_STEPS.map((step, index) => (
          <li key={step} className="grid grid-cols-[auto_1fr] gap-3 border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3 text-sm">
            <span className="font-mono text-xs text-[var(--ds-faint)]">{index + 1}</span>
            <span className="text-[var(--ds-ink)]">{step}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function SandboxPanel() {
  return (
    <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
      <p className="mb-3 text-xs font-bold uppercase tracking-wide text-[var(--ds-faint)]">Command policy</p>
      <div className="space-y-2">
        {COMMANDS.map((item) => {
          const Icon = item.icon;
          return (
            <div key={item.command} className="grid grid-cols-[auto_1fr_auto] items-center gap-3 border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3">
              <Icon className="h-4 w-4 text-[var(--ds-accent)]" />
              <code className="text-xs text-[var(--ds-ink)]">{item.command}</code>
              <StatusPill status={item.status} />
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TestMatrix() {
  return (
    <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
      <p className="mb-3 text-xs font-bold uppercase tracking-wide text-[var(--ds-faint)]">Test evidence matrix</p>
      <div className="space-y-2">
        {TEST_ROWS.map((row) => (
          <div key={row.name} className="grid grid-cols-[1fr_auto_auto_auto] items-center gap-3 border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3">
            <span className="text-sm font-semibold text-[var(--ds-ink)]">{row.name}</span>
            <StatusPill status={row.before} />
            <StatusPill status={row.after} />
            <span className="font-mono text-xs text-[var(--ds-faint)]">{row.weight}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function DiffReview() {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <div className="border border-emerald-300 bg-emerald-50 p-4">
        <h3 className="font-bold text-emerald-950">Minimal fix</h3>
        <p className="mt-2 text-sm text-emerald-900">2 files changed, one behavior patch, one reproduction test.</p>
        <MetricBar label="Reviewability" value={88} tone="bg-emerald-600" />
      </div>
      <div className="border border-rose-300 bg-rose-50 p-4">
        <h3 className="font-bold text-rose-950">Overbroad patch</h3>
        <p className="mt-2 text-sm text-rose-900">17 files changed, parser rewrite, package updates, unrelated formatting.</p>
        <MetricBar label="Scope drift" value={79} tone="bg-rose-600" />
      </div>
    </div>
  );
}

function MemoryPanel() {
  return (
    <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
      <pre className="overflow-auto rounded border border-[var(--ds-rule)] bg-slate-950 p-4 text-xs leading-6 text-slate-100">
{`# AGENTS.md
- Use pytest tests/parser for parser changes.
- Do not edit generated snapshots unless the test owner approves.
- Prefer small diffs and explain every changed file.
- Ask before package installs, migrations, secrets, pushes, or deploys.`}
      </pre>
    </div>
  );
}

function CheckpointPanel() {
  const checkpoints = ['clean checkout', 'failing reproduction', 'first patch', 'target tests pass', 'regression found', 'rollback and narrow fix'];
  return (
    <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
      <div className="grid gap-2">
        {checkpoints.map((checkpoint, index) => (
          <div key={checkpoint} className="grid grid-cols-[auto_1fr] items-center gap-3">
            <span className={`h-4 w-4 rounded-full ${index < 4 ? 'bg-emerald-600' : index === 4 ? 'bg-rose-600' : 'bg-[var(--ds-accent)]'}`} />
            <span className="border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3 text-sm text-[var(--ds-ink)]">{checkpoint}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function MultiAgentPanel() {
  const roles = ['Planner', 'Navigator', 'Implementer', 'Tester', 'Reviewer', 'Security reviewer'];
  return (
    <div className="grid gap-2 md:grid-cols-3">
      {roles.map((role, index) => (
        <div key={role} className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
          <Users className="mb-3 h-5 w-5 text-[var(--ds-accent)]" />
          <h3 className="font-bold text-[var(--ds-ink)]">{role}</h3>
          <p className="mt-2 text-xs text-[var(--ds-faint)]">
            {index === 4 ? 'flags overbroad edits and missing tests' : 'contributes a focused signal to the patch loop'}
          </p>
        </div>
      ))}
    </div>
  );
}

function FailurePanel() {
  const [activeFailure, setActiveFailure] = useState('overbroad-refactor');
  const failure = FAILURE_SCENARIOS.find((item) => item.id === activeFailure) || FAILURE_SCENARIOS[0];
  return (
    <div className="grid gap-4 md:grid-cols-[0.8fr_1.2fr]">
      <div className="space-y-2">
        {FAILURE_SCENARIOS.map((item) => (
          <button
            key={item.id}
            data-math-control
            onClick={() => setActiveFailure(item.id)}
            className={`w-full border p-3 text-left text-sm ${
              activeFailure === item.id ? 'border-[var(--ds-accent)] bg-[var(--ds-paper)]' : 'border-[var(--ds-rule)] bg-[var(--ds-panel)]'
            }`}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
        <h3 className="text-lg font-bold text-[var(--ds-ink)]">{failure.label}</h3>
        <p className="mt-2 text-sm text-[var(--ds-muted)]">{failure.symptom}</p>
        <p className="mt-3 text-sm font-semibold text-[var(--ds-ink)]">{failure.mitigation}</p>
        <div className="mt-4">
          <MetricBar label="Failure risk" value={failure.risk} tone="bg-rose-600" />
        </div>
      </div>
    </div>
  );
}

export default function AgenticCodingSystems() {
  const [activeStage, setActiveStage] = useState('understand');
  const [activeModule, setActiveModule] = useState('system-map');
  const module = useMemo(() => MODULES.find((item) => item.id === activeModule) || MODULES[0], [activeModule]);

  return (
    <div className="ua-lesson-stage space-y-6">
      <header className="border-b border-[var(--ds-rule)] pb-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.22em] text-[var(--ds-faint)]">Frontier LLMs</p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight text-[var(--ds-ink)]">Agentic Coding Systems</h1>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-[var(--ds-muted)]">
              Learn coding agents as permissioned software-engineering loops over context, planning, edits, execution, evidence, review, rollback, and human approval.
            </p>
          </div>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-3">
              <p className="font-mono text-lg font-bold text-[var(--ds-ink)]">12</p>
              <p className="text-[10px] uppercase tracking-wide text-[var(--ds-faint)]">modules</p>
            </div>
            <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-3">
              <p className="font-mono text-lg font-bold text-[var(--ds-ink)]">4-6h</p>
              <p className="text-[10px] uppercase tracking-wide text-[var(--ds-faint)]">path time</p>
            </div>
            <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-3">
              <p className="font-mono text-lg font-bold text-[var(--ds-ink)]">Advanced</p>
              <p className="text-[10px] uppercase tracking-wide text-[var(--ds-faint)]">level</p>
            </div>
          </div>
        </div>
      </header>

      <IssuePatchLoop activeStage={activeStage} setActiveStage={setActiveStage} />
      <StageMap activeStage={activeStage} />

      <nav className="flex gap-2 overflow-x-auto border-y border-[var(--ds-rule)] py-3">
        {MODULES.map((item) => (
          <button
            key={item.id}
            data-math-control
            onClick={() => setActiveModule(item.id)}
            className={`shrink-0 border px-3 py-2 text-xs font-semibold transition ${
              activeModule === item.id
                ? 'border-[var(--ds-accent)] bg-[var(--ds-accent)] text-[var(--ds-paper)]'
                : 'border-[var(--ds-rule)] bg-[var(--ds-panel)] text-[var(--ds-muted)] hover:bg-[var(--ds-paper)]'
            }`}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <ModulePanel module={module} />

      <section className="grid gap-4 lg:grid-cols-4">
        {PRODUCT_CARDS.map((card) => (
          <article key={card.title} className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-4">
            <h3 className="font-bold text-[var(--ds-ink)]">{card.title}</h3>
            <p className="mt-2 text-xs leading-5 text-[var(--ds-faint)]">{card.signal}</p>
            <p className="mt-3 text-sm leading-5 text-[var(--ds-muted)]">{card.interpretation}</p>
          </article>
        ))}
      </section>
    </div>
  );
}
