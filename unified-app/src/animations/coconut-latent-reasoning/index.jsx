import React, { useMemo, useState } from 'react';
import {
  ArrowRight,
  BarChart3,
  BookOpen,
  Brain,
  Code2,
  Cpu,
  ExternalLink,
  FileText,
  GitBranch,
  Layers3,
  Link as LinkIcon,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Target,
  Zap,
} from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const REASONING_MODES = [
  { id: 'no-cot', label: 'No-CoT', detail: 'Answer directly after the question.' },
  { id: 'cot', label: 'CoT', detail: 'Write intermediate reasoning as visible text tokens.' },
  { id: 'pause', label: 'Pause Tokens', detail: 'Spend extra steps on learned filler tokens.' },
  { id: 'coconut', label: 'Coconut', detail: 'Feed hidden states back as continuous thoughts.' },
];

const LATENT_STEP_OPTIONS = [0, 1, 2, 4, 8];

const TASK_TYPES = [
  { id: 'arithmetic', label: 'Arithmetic', detail: 'Short equation with little backtracking.' },
  { id: 'multi-hop', label: 'Multi-hop logic', detail: 'Several facts must be chained in order.' },
  { id: 'planning', label: 'Planning with dead ends', detail: 'Wrong early branch choices can trap the trace.' },
];

const SHOW_OPTIONS = [
  { id: 'vector-norm', label: 'vector norm' },
  { id: 'nearest-token', label: 'nearest token probe' },
  { id: 'path-distribution', label: 'path distribution' },
];

const FAITHFULNESS_TESTS = [
  { id: 'none', label: 'none', detail: 'Read the latent path without intervening.' },
  { id: 'remove-latent', label: 'remove latent', detail: 'Ablate the continuous thought region.' },
  { id: 'perturb-latent', label: 'perturb latent', detail: 'Add noise to a hidden state.' },
  { id: 'shortcut-dataset', label: 'shortcut dataset', detail: 'Check whether a shortcut still works out of distribution.' },
];

const BRANCHES = [
  { id: 'lempus', label: 'lempus', detail: 'bridge to rorpus', tone: 'emerald' },
  { id: 'sterpus', label: 'sterpus', detail: 'dead end after one hop', tone: 'rose' },
  { id: 'grimpus', label: 'grimpus', detail: 'distractor relation', tone: 'amber' },
];

const BASE_DISTRIBUTIONS = {
  'no-cot': [
    { lempus: 0.36, sterpus: 0.37, grimpus: 0.27 },
  ],
  pause: [
    { lempus: 0.38, sterpus: 0.34, grimpus: 0.28 },
    { lempus: 0.43, sterpus: 0.31, grimpus: 0.26 },
    { lempus: 0.46, sterpus: 0.29, grimpus: 0.25 },
    { lempus: 0.48, sterpus: 0.27, grimpus: 0.25 },
  ],
  cot: [
    { lempus: 0.55, sterpus: 0.3, grimpus: 0.15 },
    { lempus: 0.7, sterpus: 0.2, grimpus: 0.1 },
    { lempus: 0.87, sterpus: 0.08, grimpus: 0.05 },
  ],
  coconut: [
    { lempus: 0.34, sterpus: 0.32, grimpus: 0.34 },
    { lempus: 0.52, sterpus: 0.16, grimpus: 0.32 },
    { lempus: 0.68, sterpus: 0.07, grimpus: 0.25 },
    { lempus: 0.92, sterpus: 0.01, grimpus: 0.07 },
  ],
};

const TASK_PRESETS = {
  arithmetic: {
    prompt: 'Solve: If 2x + 3 = 11, what is x?',
    graphTitle: 'short algebra path',
    answer: 'x = 4',
    baseAccuracy: 0.58,
    planningWeight: 0.45,
  },
  'multi-hop': {
    prompt: 'If Alex is a lempus, every lempus is a rorpus, and every rorpus is gladen, is Alex gladen?',
    graphTitle: 'two-hop proof path',
    answer: 'yes, Alex is gladen',
    baseAccuracy: 0.5,
    planningWeight: 0.72,
  },
  planning: {
    prompt: 'Find a path from Alex to the target while avoiding relations that lead to dead ends.',
    graphTitle: 'branching proof graph',
    answer: 'Alex -> lempus -> rorpus -> target',
    baseAccuracy: 0.42,
    planningWeight: 0.9,
  },
};

const CURRICULUM_STAGES = [
  ['text step 1', 'text step 2', 'text step 3', 'answer'],
  ['latent', 'text step 2', 'text step 3', 'answer'],
  ['latent', 'latent', 'text step 3', 'answer'],
  ['latent', 'latent', 'latent', 'answer'],
];

const QUESTIONS = [
  [
    'What does Coconut use as an intermediate reasoning state?',
    'The previous position last hidden state, fed back as the next input embedding during latent mode.',
  ],
  [
    'What do <bot> and <eot> mark?',
    'They mark the beginning and end of the latent reasoning region.',
  ],
  [
    'What does latent mode skip?',
    'It skips decoding the hidden state into a word token before feeding it into the next position.',
  ],
  [
    'Why is a continuous thought different from a pause token?',
    'A pause token has a learned embedding shared across contexts; a continuous thought is a context-dependent hidden state.',
  ],
  [
    'Why does Coconut need a curriculum?',
    'Learning useful latent thoughts directly from question-answer pairs is hard, so training gradually replaces early text reasoning steps with latent states.',
  ],
  [
    'What is delayed commitment?',
    'The model keeps several possible branches softly active before committing to one explicit answer path.',
  ],
  [
    'What is the caveat?',
    'Latent thoughts are opaque and may not be faithful reasoning states, so perturbations, probes, ablations, and shortcut tests matter.',
  ],
];

const EXERCISES = [
  ['01_mode_switch.rs', 'Switch between language and latent modes with <bot> and <eot>.'],
  ['02_latent_feedback.rs', 'Choose token embeddings in language mode and hidden states in latent mode.'],
  ['03_masked_loss.rs', 'Apply loss to future text targets, not question or latent positions.'],
  ['04_curriculum_schedule.rs', 'Replace a growing prefix of text reasoning steps with latent thoughts.'],
  ['05_pause_vs_continuous.rs', 'Contrast fixed pause embeddings with context-dependent hidden states.'],
  ['06_branch_entropy.rs', 'Measure how spread out the active branch distribution is.'],
  ['07_delayed_commitment.rs', 'Find the first step where one branch probability crosses a threshold.'],
  ['08_latent_perturbation.rs', 'Perturb hidden states and measure how far they move.'],
  ['09_token_budget.rs', 'Separate visible text tokens from total compute steps.'],
  ['10_probe_nearest_tokens.rs', 'Probe a hidden state with nearest-token dot products.'],
];

const SOURCE_LINKS = [
  {
    label: 'Coconut paper',
    href: 'https://ar5iv.org/html/2412.06769v3',
    note: 'Continuous thoughts, language/latent modes, <bot>/<eot>, curriculum replacement, delayed-commitment interpretation, and reported benchmark results.',
  },
  {
    label: 'Do Latent Tokens Think?',
    href: 'https://arxiv.org/abs/2512.21711',
    note: 'Critical analysis of faithfulness, causal interventions, shortcut behavior, and robustness concerns for latent-token reasoning.',
  },
  {
    label: 'SoftCoT++',
    href: 'https://arxiv.org/abs/2505.11484',
    note: 'Optional sequel direction: perturb and diversify soft thoughts for test-time latent reasoning exploration.',
  },
];

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function fmt(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function entropy(distribution) {
  return Object.values(distribution).reduce((acc, probability) => {
    if (probability <= 0) return acc;
    return acc - probability * Math.log(probability);
  }, 0);
}

function normalize(distribution) {
  const positiveEntries = Object.entries(distribution).map(([key, value]) => [key, Math.max(0, value)]);
  const total = positiveEntries.reduce((acc, [, value]) => acc + value, 0);
  if (!total) return distribution;
  return Object.fromEntries(positiveEntries.map(([key, value]) => [key, value / total]));
}

function applyFaithfulness(distribution, faithfulnessTest, perturbation) {
  if (faithfulnessTest === 'none') return distribution;

  if (faithfulnessTest === 'remove-latent') {
    return normalize({
      lempus: distribution.lempus * 0.72,
      sterpus: distribution.sterpus + 0.15,
      grimpus: distribution.grimpus + 0.1,
    });
  }

  if (faithfulnessTest === 'perturb-latent') {
    const noise = perturbation / 100;
    return normalize({
      lempus: distribution.lempus * (1 - 0.35 * noise),
      sterpus: distribution.sterpus + 0.18 * noise,
      grimpus: distribution.grimpus + 0.08 * noise,
    });
  }

  return normalize({
    lempus: distribution.lempus * 0.78,
    sterpus: distribution.sterpus * 1.2,
    grimpus: distribution.grimpus * 1.25,
  });
}

function commitmentStep(distributions, threshold = 0.85) {
  const index = distributions.findIndex((distribution) => Math.max(...Object.values(distribution)) >= threshold);
  return index === -1 ? null : index + 1;
}

function getTimeline(mode, latentSteps) {
  if (mode === 'no-cot') {
    return [
      { label: 'Question', type: 'text' },
      { label: 'Answer', type: 'answer' },
    ];
  }

  if (mode === 'cot') {
    return [
      { label: 'Question', type: 'text' },
      { label: 'First', type: 'token' },
      { label: 'compute', type: 'token' },
      { label: 'compare', type: 'token' },
      { label: 'therefore', type: 'token' },
      { label: 'Answer', type: 'answer' },
    ];
  }

  if (mode === 'pause') {
    const count = Math.max(1, Math.min(latentSteps || 1, 6));
    return [
      { label: 'Question', type: 'text' },
      ...Array.from({ length: count }, (_, index) => ({ label: `<pause ${index + 1}>`, type: 'pause' })),
      { label: 'Answer', type: 'answer' },
    ];
  }

  const count = Math.max(1, Math.min(latentSteps || 1, 6));
  return [
    { label: 'Question', type: 'text' },
    { label: '<bot>', type: 'marker' },
    ...Array.from({ length: count }, (_, index) => ({ label: `h${index + 1}`, type: 'latent' })),
    { label: '<eot>', type: 'marker' },
    { label: 'Answer', type: 'answer' },
  ];
}

function getDistributions(mode, latentSteps, taskType, faithfulnessTest, perturbation) {
  const base = BASE_DISTRIBUTIONS[mode] || BASE_DISTRIBUTIONS.coconut;
  const count = mode === 'coconut'
    ? Math.max(1, Math.min(latentSteps || 1, base.length))
    : mode === 'pause'
      ? Math.max(1, Math.min(latentSteps || 1, base.length))
      : base.length;
  const weight = TASK_PRESETS[taskType].planningWeight;

  return base.slice(0, count).map((distribution, index) => {
    const taskAdjusted = normalize({
      lempus: distribution.lempus + (weight - 0.6) * 0.08 * (index + 1),
      sterpus: distribution.sterpus - (weight - 0.6) * 0.05 * (index + 1),
      grimpus: distribution.grimpus - (weight - 0.6) * 0.03 * (index + 1),
    });
    return mode === 'coconut' ? applyFaithfulness(taskAdjusted, faithfulnessTest, perturbation) : taskAdjusted;
  });
}

function ControlButton({ active, children, onClick }) {
  return (
    <button
      type="button"
      data-math-control
      onClick={onClick}
      className={`min-h-[2.5rem] rounded border px-3 py-2 text-xs font-black uppercase tracking-wide transition ${
        active
          ? 'border-[var(--ds-accent)] bg-[var(--ds-accent)] text-[var(--ds-paper)]'
          : 'border-[var(--ds-rule)] bg-[var(--ds-paper)] text-[var(--ds-ink)] hover:border-[var(--ds-accent)]'
      }`}
    >
      {children}
    </button>
  );
}

function SectionHeader({ icon: Icon, eyebrow, title, children }) {
  return (
    <div className="mb-4">
      <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-[var(--ds-accent)]">
        <Icon size={16} />
        {eyebrow}
      </p>
      <h2 className="mt-1 text-xl font-black tracking-tight md:text-2xl">{title}</h2>
      {children ? <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--ds-muted)]">{children}</p> : null}
    </div>
  );
}

function SliderControl({ label, value, min, max, step, onChange, display }) {
  return (
    <label className="grid gap-2 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3">
      <span className="flex items-center justify-between gap-3 text-xs font-bold uppercase tracking-wide text-[var(--ds-muted)]">
        {label}
        <span className="font-mono text-[var(--ds-ink)]">{display || value}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="w-full accent-[var(--ds-accent)]"
      />
    </label>
  );
}

function MetricTile({ label, value, detail }) {
  return (
    <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
      <span className="text-[10px] font-black uppercase tracking-wide text-[var(--ds-muted)]">{label}</span>
      <strong className="mt-1 block break-words font-mono text-xl text-[var(--ds-ink)]">{value}</strong>
      <p className="mt-1 text-xs leading-5 text-[var(--ds-muted)]">{detail}</p>
    </div>
  );
}

function TimelineToken({ token }) {
  const tone = {
    text: 'border-[var(--ds-rule)] bg-[var(--ds-paper)] text-[var(--ds-ink)]',
    token: 'border-sky-700/30 bg-sky-50 text-sky-950',
    pause: 'border-amber-700/30 bg-amber-50 text-amber-950',
    latent: 'border-emerald-700/30 bg-emerald-50 text-emerald-950',
    marker: 'border-[var(--ds-accent)] bg-[var(--ds-warm)] text-[var(--ds-ink)]',
    answer: 'border-[var(--ds-accent)] bg-[var(--ds-accent)] text-[var(--ds-paper)]',
  }[token.type];

  return (
    <span className={`min-w-[5.5rem] rounded border px-3 py-2 text-center font-mono text-xs font-black ${tone}`}>
      {token.label}
    </span>
  );
}

function BranchBar({ branch, probability }) {
  const color = {
    emerald: 'bg-emerald-600',
    rose: 'bg-rose-500',
    amber: 'bg-amber-500',
  }[branch.tone];

  return (
    <div className="grid grid-cols-[5rem_1fr_3.5rem] items-center gap-3 text-xs">
      <span className="font-mono font-black">{branch.label}</span>
      <div className="h-3 overflow-hidden rounded border border-[var(--ds-rule)] bg-[var(--ds-paper-2)]">
        <div className={`h-full ${color}`} style={{ width: `${clamp(probability, 0.02, 1) * 100}%` }} />
      </div>
      <span className="text-right font-mono">{Math.round(probability * 100)}%</span>
    </div>
  );
}

function CurriculumStage({ index, activeStage }) {
  const steps = CURRICULUM_STAGES[index];
  return (
    <article className={`rounded border p-4 ${index === activeStage ? 'border-[var(--ds-accent)] bg-[var(--ds-warm)]' : 'border-[var(--ds-rule)] bg-[var(--ds-paper)]'}`}>
      <h3 className="mb-3 text-sm font-black uppercase tracking-wide">Stage {index}</h3>
      <div className="flex flex-wrap items-center gap-2">
        {steps.map((step, stepIndex) => (
          <React.Fragment key={`${index}-${stepIndex}-${step}`}>
            <span className={`rounded border px-3 py-2 text-xs font-bold ${
              step === 'latent'
                ? 'border-emerald-700/30 bg-emerald-50 text-emerald-950'
                : step === 'answer'
                  ? 'border-[var(--ds-accent)] bg-[var(--ds-accent)] text-[var(--ds-paper)]'
                  : 'border-[var(--ds-rule)] bg-[var(--ds-panel)] text-[var(--ds-ink)]'
            }`}>
              {step}
            </span>
            {stepIndex < steps.length - 1 ? <ArrowRight size={14} className="text-[var(--ds-muted)]" /> : null}
          </React.Fragment>
        ))}
      </div>
    </article>
  );
}

export default function CoconutLatentReasoning() {
  const [mode, setMode] = useState('coconut');
  const [latentSteps, setLatentSteps] = useState(3);
  const [trainingStage, setTrainingStage] = useState(2);
  const [taskType, setTaskType] = useState('planning');
  const [showOption, setShowOption] = useState('path-distribution');
  const [faithfulnessTest, setFaithfulnessTest] = useState('none');
  const [perturbation, setPerturbation] = useState(35);

  const activeMode = REASONING_MODES.find((item) => item.id === mode) || REASONING_MODES[0];
  const task = TASK_PRESETS[taskType];

  const timeline = useMemo(() => getTimeline(mode, latentSteps), [mode, latentSteps]);
  const distributions = useMemo(
    () => getDistributions(mode, latentSteps, taskType, faithfulnessTest, perturbation),
    [mode, latentSteps, taskType, faithfulnessTest, perturbation],
  );

  const stats = useMemo(() => {
    const lastDistribution = distributions[distributions.length - 1] || distributions[0];
    const firstEntropy = entropy(distributions[0]);
    const finalEntropy = entropy(lastDistribution);
    const visibleTextTokens = mode === 'cot'
      ? 38
      : mode === 'pause'
        ? 8 + Math.max(1, latentSteps)
        : mode === 'coconut'
          ? 11
          : 7;
    const latentCompute = mode === 'coconut' ? Math.max(1, latentSteps) : 0;
    const pauseCompute = mode === 'pause' ? Math.max(1, latentSteps) : 0;
    const rawAccuracy = task.baseAccuracy
      + (mode === 'coconut' ? 0.16 : mode === 'cot' ? 0.12 : mode === 'pause' ? 0.04 : 0)
      + (mode === 'coconut' ? Math.min(latentSteps, 6) * 0.025 : 0)
      + (task.planningWeight - 0.6) * (mode === 'coconut' ? 0.18 : 0.05)
      - (faithfulnessTest === 'remove-latent' && mode === 'coconut' ? 0.16 : 0)
      - (faithfulnessTest === 'perturb-latent' && mode === 'coconut' ? perturbation * 0.0012 : 0)
      - (faithfulnessTest === 'shortcut-dataset' && mode === 'coconut' ? 0.1 : 0);

    return {
      finalDistribution: lastDistribution,
      firstEntropy,
      finalEntropy,
      entropyDrop: firstEntropy - finalEntropy,
      commitment: commitmentStep(distributions),
      visibleTextTokens,
      computeSteps: visibleTextTokens + latentCompute + pauseCompute,
      latentCompute,
      accuracy: clamp(rawAccuracy, 0.1, 0.96),
      probeToken: lastDistribution.lempus > lastDistribution.grimpus ? 'lempus' : 'grimpus',
      probeConfidence: Math.max(...Object.values(lastDistribution)),
      perturbationSensitivity: mode === 'coconut' && faithfulnessTest !== 'none'
        ? clamp((faithfulnessTest === 'perturb-latent' ? perturbation / 100 : 0.55), 0, 1)
        : 0.12,
    };
  }, [distributions, faithfulnessTest, latentSteps, mode, perturbation, task.baseAccuracy, task.planningWeight]);

  const languagePathActive = mode === 'cot' || mode === 'no-cot' || mode === 'pause';
  const latentPathActive = mode === 'coconut';

  return (
    <div className="min-h-screen bg-[var(--ds-paper)] text-[var(--ds-ink)]">
      <header className="border-b border-[var(--ds-rule)] bg-[var(--ds-panel)]">
        <div className="mx-auto flex max-w-7xl flex-col gap-6 px-5 py-8 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-3xl">
            <div className="mb-3 flex flex-wrap gap-2">
              <span className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] px-2 py-1 text-[10px] font-black uppercase tracking-wide text-[var(--ds-muted)]">
                Papers
              </span>
              <span className="rounded border border-[var(--ds-rule)] bg-[var(--ds-warm)] px-2 py-1 text-[10px] font-black uppercase tracking-wide text-[var(--ds-ink)]">
                Transformers & Attention
              </span>
              <span className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] px-2 py-1 text-[10px] font-black uppercase tracking-wide text-[var(--ds-muted)]">
                Chain of Continuous Thought
              </span>
            </div>
            <h1 className="text-3xl font-black tracking-tight md:text-5xl">
              Coconut: Reasoning Between Tokens
            </h1>
            <p className="mt-4 max-w-2xl text-sm leading-7 text-[var(--ds-muted)] md:text-base">
              How an LLM can think in hidden-state vectors instead of writing every intermediate step as text.
            </p>
          </div>
          <div className="grid min-w-[280px] grid-cols-2 gap-3 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
            <MetricTile label="mode" value={activeMode.label} detail={activeMode.detail} />
            <MetricTile label="latent steps" value={mode === 'coconut' ? latentSteps : 0} detail="hidden-state feedback passes" />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-5 py-6">
        <section className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
          <SectionHeader icon={SlidersHorizontal} eyebrow="interactive controls" title="Text Tokens Or Continuous Thoughts">
            Compare direct answers, visible chain-of-thought, pause tokens, and Coconut latent feedback on the same toy reasoning task.
          </SectionHeader>

          <div className="grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
            <div className="grid gap-4">
              <div>
                <p className="mb-2 text-xs font-black uppercase tracking-wide text-[var(--ds-muted)]">Reasoning mode</p>
                <div className="grid gap-2 md:grid-cols-4">
                  {REASONING_MODES.map((item) => (
                    <ControlButton key={item.id} active={mode === item.id} onClick={() => setMode(item.id)}>
                      {item.label}
                    </ControlButton>
                  ))}
                </div>
              </div>

              <div className="grid gap-3 md:grid-cols-3">
                <div>
                  <p className="mb-2 text-xs font-black uppercase tracking-wide text-[var(--ds-muted)]">Latent thoughts</p>
                  <div className="grid grid-cols-5 gap-2">
                    {LATENT_STEP_OPTIONS.map((count) => (
                      <ControlButton key={count} active={latentSteps === count} onClick={() => setLatentSteps(count)}>
                        {count}
                      </ControlButton>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="mb-2 text-xs font-black uppercase tracking-wide text-[var(--ds-muted)]">Task type</p>
                  <div className="grid gap-2">
                    {TASK_TYPES.map((item) => (
                      <ControlButton key={item.id} active={taskType === item.id} onClick={() => setTaskType(item.id)}>
                        {item.label}
                      </ControlButton>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="mb-2 text-xs font-black uppercase tracking-wide text-[var(--ds-muted)]">Show hidden state</p>
                  <div className="grid gap-2">
                    {SHOW_OPTIONS.map((item) => (
                      <ControlButton key={item.id} active={showOption === item.id} onClick={() => setShowOption(item.id)}>
                        {item.label}
                      </ControlButton>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <div className="grid gap-3">
              <SliderControl
                label="training stage"
                value={trainingStage}
                min={0}
                max={3}
                step={1}
                onChange={setTrainingStage}
                display={`stage ${trainingStage}`}
              />
              <SliderControl
                label="latent perturbation"
                value={perturbation}
                min={0}
                max={100}
                step={5}
                onChange={setPerturbation}
                display={`${perturbation}%`}
              />
              <div>
                <p className="mb-2 text-xs font-black uppercase tracking-wide text-[var(--ds-muted)]">Faithfulness test</p>
                <div className="grid grid-cols-2 gap-2">
                  {FAITHFULNESS_TESTS.map((item) => (
                    <ControlButton key={item.id} active={faithfulnessTest === item.id} onClick={() => setFaithfulnessTest(item.id)}>
                      {item.label}
                    </ControlButton>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="mt-6 grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <SectionHeader icon={FileText} eyebrow="main animation" title="CoT Writes Thoughts. Coconut Feeds Hidden States.">
              The same question can move through visible text tokens or through a latent vector region before the final answer.
            </SectionHeader>
            <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
              <p className="text-[10px] font-black uppercase tracking-wide text-[var(--ds-muted)]">Prompt</p>
              <p className="mt-1 font-mono text-sm text-[var(--ds-ink)]">{task.prompt}</p>
            </div>
            <div className="mt-4 flex flex-wrap items-center gap-2">
              {timeline.map((token, index) => (
                <React.Fragment key={`${token.label}-${index}`}>
                  <TimelineToken token={token} />
                  {index < timeline.length - 1 ? <ArrowRight size={16} className="text-[var(--ds-muted)]" /> : null}
                </React.Fragment>
              ))}
            </div>
            <div className="mt-4 grid gap-3 md:grid-cols-2">
              <div className={`rounded border p-4 ${languagePathActive ? 'border-sky-700/30 bg-sky-50' : 'border-[var(--ds-rule)] bg-[var(--ds-paper)]'}`}>
                <h3 className="text-sm font-black">Language mode</h3>
                <p className="mt-2 font-mono text-xs">{'hidden -> LM head -> token -> embedding -> next input'}</p>
                <p className="mt-2 text-xs leading-5 text-[var(--ds-muted)]">Normal autoregressive decoding turns each reasoning state into a word-like token first.</p>
              </div>
              <div className={`rounded border p-4 ${latentPathActive ? 'border-emerald-700/30 bg-emerald-50' : 'border-[var(--ds-rule)] bg-[var(--ds-paper)]'}`}>
                <h3 className="text-sm font-black">Latent mode</h3>
                <p className="mt-2 font-mono text-xs">{'hidden h_t -> next input x_t+1'}</p>
                <p className="mt-2 text-xs leading-5 text-[var(--ds-muted)]">Coconut bypasses the LM head so the hidden state becomes the next input embedding directly.</p>
              </div>
            </div>
          </div>

          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <SectionHeader icon={Cpu} eyebrow="feedback loop" title="Where The LM Head Is Skipped" />
            <div className="grid gap-3">
              <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
                <div className="grid grid-cols-[1fr_auto_1fr_auto_1fr] items-center gap-2 text-center text-xs font-black">
                  <span className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] px-3 py-3">input embedding</span>
                  <ArrowRight size={16} />
                  <span className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] px-3 py-3">Transformer</span>
                  <ArrowRight size={16} />
                  <span className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] px-3 py-3">last hidden state</span>
                </div>
              </div>
              <div className="grid gap-3 md:grid-cols-2">
                <div className={`rounded border p-4 ${languagePathActive ? 'border-sky-700/30 bg-sky-50' : 'border-[var(--ds-rule)] bg-[var(--ds-paper)]'}`}>
                  <p className="text-[10px] font-black uppercase tracking-wide text-[var(--ds-muted)]">Language mode equation</p>
                  <p className="mt-2 font-mono text-sm">x_t+1 = Embed(sample(LMHead(h_t)))</p>
                </div>
                <div className={`rounded border p-4 ${latentPathActive ? 'border-emerald-700/30 bg-emerald-50' : 'border-[var(--ds-rule)] bg-[var(--ds-paper)]'}`}>
                  <p className="text-[10px] font-black uppercase tracking-wide text-[var(--ds-muted)]">Latent mode equation</p>
                  <p className="mt-2 font-mono text-sm">x_t+1 = h_t</p>
                </div>
              </div>
              <p className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3 text-xs leading-5 text-[var(--ds-muted)]">
                A continuous thought still costs a forward pass. It can reduce visible text tokens, but it does not make reasoning free.
              </p>
            </div>
          </div>
        </section>

        <section className="mt-6 grid gap-6 xl:grid-cols-[0.92fr_1.08fr]">
          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <SectionHeader icon={Layers3} eyebrow="training curriculum" title="Replace Text Reasoning With Latent Steps">
              Coconut starts from language chain-of-thought data, then gradually replaces early text steps with hidden-state thoughts.
            </SectionHeader>
            <div className="grid gap-3">
              {[0, 1, 2, 3].map((stage) => (
                <CurriculumStage key={stage} index={stage} activeStage={trainingStage} />
              ))}
            </div>
            <p className="mt-4 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3 text-xs leading-5 text-[var(--ds-muted)]">
              The loss is applied to remaining text targets. Question tokens and latent thoughts are masked, so latent states are trained to help predict future reasoning and the final answer rather than reproduce missing words directly.
            </p>
          </div>

          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <SectionHeader icon={GitBranch} eyebrow="delayed commitment" title="Keep Branches Alive, Then Prune">
              The planning view treats hidden states as soft branch distributions that sharpen as uncertainty drops.
            </SectionHeader>
            <div className="mb-4 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
              <p className="text-[10px] font-black uppercase tracking-wide text-[var(--ds-muted)]">{task.graphTitle}</p>
              <div className="mt-3 grid gap-2 font-mono text-xs">
                <span>Alex</span>
                <span className="pl-4">{'|- lempus -> rorpus -> target'}</span>
                <span className="pl-4">{'|- sterpus -> dead end'}</span>
                <span className="pl-4">{'`- grimpus -> distractor'}</span>
              </div>
            </div>
            <div className="grid gap-3">
              {distributions.map((distribution, step) => (
                <article key={step} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
                  <div className="mb-3 flex items-center justify-between gap-3">
                    <h3 className="text-sm font-black">{mode === 'coconut' ? `latent thought h${step + 1}` : `step ${step + 1}`}</h3>
                    <span className="font-mono text-xs text-[var(--ds-muted)]">entropy {fmt(entropy(distribution), 2)}</span>
                  </div>
                  <div className="grid gap-2">
                    {BRANCHES.map((branch) => (
                      <BranchBar key={branch.id} branch={branch} probability={distribution[branch.id]} />
                    ))}
                  </div>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="mt-6 grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <SectionHeader icon={BarChart3} eyebrow="dynamic metrics" title="Visible Tokens, Compute, Entropy, Sensitivity" />
            <div className="grid gap-3 md:grid-cols-3">
              <MetricTile label="visible text tokens" value={stats.visibleTextTokens} detail="language tokens shown to the user" />
              <MetricTile label="compute steps" value={stats.computeSteps} detail="text plus latent or pause passes" />
              <MetricTile label="answer accuracy" value={`${Math.round(stats.accuracy * 100)}%`} detail="deterministic toy estimate" />
              <MetricTile label="branch entropy" value={fmt(stats.finalEntropy, 2)} detail={`drop ${fmt(stats.entropyDrop, 2)} from first step`} />
              <MetricTile label="commitment step" value={stats.commitment || 'none'} detail="first max branch >= 85%" />
              <MetricTile label="perturb sensitivity" value={`${Math.round(stats.perturbationSensitivity * 100)}%`} detail="faithfulness test readout" />
            </div>
            <div className="mt-4 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
              <h3 className="text-sm font-black">Hidden-state view: {SHOW_OPTIONS.find((item) => item.id === showOption)?.label}</h3>
              {showOption === 'vector-norm' ? (
                <p className="mt-2 text-sm leading-6 text-[var(--ds-muted)]">
                  The toy latent norm rises from {fmt(1.1 + stats.firstEntropy, 2)} to {fmt(1.35 + stats.entropyDrop, 2)} as the branch distribution becomes more focused.
                </p>
              ) : showOption === 'nearest-token' ? (
                <p className="mt-2 text-sm leading-6 text-[var(--ds-muted)]">
                  A nearest-token probe reads the final latent state as closest to <span className="font-mono font-bold">{stats.probeToken}</span> with {Math.round(stats.probeConfidence * 100)}% confidence. A probe is useful evidence, not proof of faithful reasoning.
                </p>
              ) : (
                <p className="mt-2 text-sm leading-6 text-[var(--ds-muted)]">
                  The path distribution ends at {Math.round(stats.finalDistribution.lempus * 100)}% lempus, {Math.round(stats.finalDistribution.sterpus * 100)}% sterpus, and {Math.round(stats.finalDistribution.grimpus * 100)}% grimpus.
                </p>
              )}
            </div>
          </div>

          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <SectionHeader icon={ShieldCheck} eyebrow="faithfulness caveat" title="Powerful-Looking, But Opaque">
              Latent thoughts can help performance while still being hard to interpret or vulnerable to shortcuts.
            </SectionHeader>
            <div className="grid gap-3">
              {FAITHFULNESS_TESTS.map((test) => (
                <article key={test.id} className={`rounded border p-4 ${faithfulnessTest === test.id ? 'border-[var(--ds-accent)] bg-[var(--ds-warm)]' : 'border-[var(--ds-rule)] bg-[var(--ds-paper)]'}`}>
                  <h3 className="text-sm font-black">{test.label}</h3>
                  <p className="mt-2 text-xs leading-5 text-[var(--ds-muted)]">{test.detail}</p>
                </article>
              ))}
            </div>
            <p className="mt-4 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3 text-xs leading-5 text-[var(--ds-muted)]">
              If changing or removing a latent state does not affect branch distributions or answers, that state may not be carrying causal reasoning. Robust lessons should pair probes with interventions and shortcut-resistant datasets.
            </p>
          </div>
        </section>

        <section className="mt-6 grid gap-6 xl:grid-cols-3">
          <article className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <Sparkles className="mb-3 h-5 w-5 text-[var(--ds-accent)]" />
            <h2 className="text-lg font-black">CoT</h2>
            <p className="mt-2 text-sm leading-6 text-[var(--ds-muted)]">Readable but verbose: reasoning is communication, one token at a time.</p>
          </article>
          <article className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <Zap className="mb-3 h-5 w-5 text-[var(--ds-accent)]" />
            <h2 className="text-lg font-black">Pause Tokens</h2>
            <p className="mt-2 text-sm leading-6 text-[var(--ds-muted)]">Extra compute with fixed learned embeddings, but not a contextual hidden-state thought.</p>
          </article>
          <article className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <Brain className="mb-3 h-5 w-5 text-[var(--ds-accent)]" />
            <h2 className="text-lg font-black">Coconut</h2>
            <p className="mt-2 text-sm leading-6 text-[var(--ds-muted)]">Compact and flexible, but harder to inspect because intermediate states are vectors.</p>
          </article>
        </section>

        <section className="mt-6 grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <SectionHeader icon={Target} eyebrow="results framing" title="Where The Intuition Is Strongest">
              The paper reports gains on planning-heavy logical reasoning tasks and emphasizes fewer generated tokens, but this is not a universal guarantee.
            </SectionHeader>
            <div className="grid gap-3">
              {[
                ['Math word problems', 'CoT remains a strong baseline when explicit symbolic work matters.'],
                ['Logical planning', 'Latent branch distributions are intuitive when bad early choices lead to dead ends.'],
                ['Backtracking tasks', 'Coconut can delay hard commitment before emitting visible text.'],
              ].map(([title, detail]) => (
                <article key={title} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
                  <h3 className="text-sm font-black">{title}</h3>
                  <p className="mt-2 text-xs leading-5 text-[var(--ds-muted)]">{detail}</p>
                </article>
              ))}
            </div>
          </div>

          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <SectionHeader icon={BookOpen} eyebrow="questions" title="Quick Checks" />
            <div className="grid gap-3">
              {QUESTIONS.map(([question, answer], index) => (
                <details key={question} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
                  <summary className="cursor-pointer text-sm font-black">
                    Q{index + 1}. {question}
                  </summary>
                  <p className="mt-3 text-sm leading-6 text-[var(--ds-muted)]">{answer}</p>
                </details>
              ))}
            </div>
          </div>
        </section>

        <section className="mt-6 grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <SectionHeader icon={Code2} eyebrow="Rustlings-style lab" title="mini-coconut Exercises">
              The companion Rust pack turns latent-mode mechanics into small functions for mode switching, masked loss, entropy, perturbation, budgets, and probes.
            </SectionHeader>
            <div className="grid gap-2 md:grid-cols-2">
              {EXERCISES.map(([file, description]) => (
                <div key={file} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3">
                  <p className="font-mono text-xs font-black text-[var(--ds-ink)]">{file}</p>
                  <p className="mt-1 text-xs leading-5 text-[var(--ds-muted)]">{description}</p>
                </div>
              ))}
            </div>
            <pre className="mt-4 overflow-x-auto rounded border border-[var(--ds-rule)] bg-stone-950 p-3 text-xs leading-5 text-stone-100">cd mini-coconut
cargo test --bins</pre>
          </div>

          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <SectionHeader icon={LinkIcon} eyebrow="source trail" title="Primary Sources" />
            <div className="grid gap-3">
              {SOURCE_LINKS.map((source) => (
                <a
                  key={source.href}
                  href={source.href}
                  target="_blank"
                  rel="noreferrer"
                  className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4 transition hover:border-[var(--ds-accent)]"
                >
                  <span className="flex items-center gap-2 text-sm font-black text-[var(--ds-ink)]">
                    <ExternalLink size={16} />
                    {source.label}
                  </span>
                  <span className="mt-2 block text-xs leading-5 text-[var(--ds-muted)]">{source.note}</span>
                </a>
              ))}
            </div>
            <div className="mt-4 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
              <h3 className="text-sm font-black">Core lesson</h3>
              <p className="mt-2 text-sm leading-6 text-[var(--ds-muted)]">
                Reasoning and explanation are not the same operation. Text is how the model communicates a solution; latent space may be another place where it searches for one.
              </p>
            </div>
          </div>
        </section>

        <div className="mt-6">
          <AssessmentPanel lessonId="coconut-latent-reasoning" title="Coconut latent reasoning check" />
        </div>
      </main>
    </div>
  );
}
