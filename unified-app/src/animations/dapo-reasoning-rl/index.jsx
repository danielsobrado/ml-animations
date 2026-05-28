import React, { useMemo, useState } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  BarChart3,
  BookOpen,
  Brain,
  CheckCircle2,
  Code2,
  ExternalLink,
  Filter,
  Gauge,
  GitBranch,
  Layers3,
  Link as LinkIcon,
  Maximize2,
  ShieldCheck,
  SlidersHorizontal,
  Target,
  TrendingDown,
  TrendingUp,
  XCircle,
  Zap,
} from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const MODES = [
  { id: 'grpo', label: 'GRPO baseline', detail: 'Sample groups, score answers, compare within each prompt.' },
  { id: 'dynamic-sampling', label: '+ Dynamic Sampling', detail: 'Keep groups with mixed success and useful gradients.' },
  { id: 'clip-higher', label: '+ Clip-Higher', detail: 'Raise the upper clip bound for positive updates.' },
  { id: 'token-level', label: '+ Token-Level Loss', detail: 'Aggregate the policy-gradient loss over tokens.' },
  { id: 'overlong-shaping', label: '+ Overlong Shaping', detail: 'Replace a hard length cliff with a smoother penalty.' },
  { id: 'full-dapo', label: 'Full DAPO', detail: 'Combine sampling, clipping, token credit, and shaped rewards.' },
];

const GROUP_SIZES = [2, 4, 8, 16];
const MAX_LENGTHS = [512, 1024, 2048, 4096];

const GROUP_SPECS = [
  { id: 'P1', target: 0.12, prompt: 'Linear equation warm-up' },
  { id: 'P2', target: 0.24, prompt: 'Short arithmetic proof' },
  { id: 'P3', target: 0.38, prompt: 'Number theory factor search' },
  { id: 'P4', target: 0.5, prompt: 'Frontier algebra problem' },
  { id: 'P5', target: 0.58, prompt: 'Combinatorics with cases' },
  { id: 'P6', target: 0.7, prompt: 'Geometry proof with trap' },
  { id: 'P7', target: 0.82, prompt: 'Long olympiad-style problem' },
  { id: 'P8', target: 0.92, prompt: 'Nearly impossible verifier task' },
];

const THRESHOLDS = [
  0.08, 0.22, 0.41, 0.63, 0.79, 0.91, 0.35, 0.56,
  0.13, 0.48, 0.71, 0.87, 0.29, 0.68, 0.95, 0.53,
];

const TOKEN_LABELS = ['setup', 'plan', 'derive', 'check', 'repair', 'final'];

const QUESTIONS = [
  ['What problem does DAPO build on?', 'GRPO-style reinforcement learning for reasoning LLMs.'],
  ['What does DAPO stand for?', 'Decoupled Clip and Dynamic sAmpling Policy Optimization.'],
  ['What are the four named techniques?', 'Clip-Higher, Dynamic Sampling, Token-Level Policy Gradient Loss, and Overlong Reward Shaping.'],
  ['Why can an all-correct group be unhelpful?', 'All samples have similar reward, so group-relative advantage has little or no contrast.'],
  ['Why can an all-wrong group be unhelpful?', 'There is no winning sample to reinforce relative to the others.'],
  ['What does Dynamic Sampling try to keep?', 'Prompt groups with mixed success and therefore useful gradients.'],
  ['What does Clip-Higher change?', 'It increases the upper clipping bound so positive-advantage updates have more room.'],
  ['Why is token-level loss useful for long-CoT RL?', 'A long reasoning trace contains many token decisions, so token-level updates provide denser credit assignment.'],
  ['What does Overlong Reward Shaping fix?', 'It reduces noisy hard penalties near or beyond response length limits.'],
  ['Why is DAPO more of a recipe than a single equation?', 'The result depends on batch construction, reward shaping, clipping, and loss granularity together.'],
];

const EXERCISES = [
  ['01_group_accuracy.rs', 'Detect all-correct, all-wrong, and mixed reward groups.'],
  ['02_dynamic_sampling.rs', 'Filter rollout groups to keep only useful contrast.'],
  ['03_group_advantage.rs', 'Normalize rewards into group-relative advantages.'],
  ['04_clip_higher.rs', 'Create asymmetric clipping ranges and clamp ratios.'],
  ['05_token_level_loss.rs', 'Average PPO-style objective terms over tokens.'],
  ['06_overlong_reward.rs', 'Turn a hard length cliff into a soft penalty.'],
  ['07_entropy_collapse.rs', 'Measure whether reasoning choices still have entropy.'],
  ['08_effective_batch.rs', 'Count useful groups and samples in a batch.'],
  ['09_dapo_objective.rs', 'Combine shaping, advantages, ratios, and clipping.'],
  ['10_training_dashboard.rs', 'Classify training health from signal, entropy, and length metrics.'],
];

const SOURCE_LINKS = [
  {
    label: 'DAPO arXiv paper',
    href: 'https://arxiv.org/abs/2503.14476',
    note: 'Open-source DAPO system, four named techniques, and the reported Qwen2.5-32B AIME 2024 result.',
  },
  {
    label: 'DAPO project page',
    href: 'https://dapo-sia.github.io/',
    note: 'High-level explanation of Clip-Higher, Dynamic Sampling, Token-Level Policy Gradient Loss, and Overlong Reward Shaping.',
  },
  {
    label: 'verl DAPO recipe',
    href: 'https://verl.readthedocs.io/en/latest/algo/dapo.html',
    note: 'Implementation recipe for separated clip epsilons, group filtering, token-mean loss aggregation, and overlong buffers.',
  },
];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function mean(xs) {
  if (!xs.length) return 0;
  return xs.reduce((sum, x) => sum + x, 0) / xs.length;
}

function std(xs) {
  if (!xs.length) return 0;
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
}

function variance(xs) {
  const s = std(xs);
  return s * s;
}

function fmt(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function pct(value) {
  return `${Math.round(value * 100)}%`;
}

function signed(value, digits = 2) {
  const rendered = fmt(value, digits);
  return value > 0 ? `+${rendered}` : rendered;
}

function difficultyLabel(value) {
  if (value < 0.38) return 'easy';
  if (value > 0.68) return 'hard';
  return 'frontier';
}

function groupAccuracy(group) {
  if (!group.completions.length) return 0;
  return group.completions.filter((completion) => completion.baseReward > 0).length / group.completions.length;
}

function hasContrast(group) {
  const accuracy = groupAccuracy(group);
  return accuracy > 0 && accuracy < 1;
}

function overlongPenalty(length, maxLength, margin, mode) {
  if (mode === 'hard') return length > maxLength ? 1 : 0;

  const safeLength = maxLength - margin;
  if (length <= safeLength) return 0;
  if (length >= maxLength) return 1;

  const t = clamp((length - safeLength) / Math.max(1, margin), 0, 1);
  if (mode === 'soft') return t * t * (3 - 2 * t);
  return t;
}

function shapeReward(baseReward, length, maxLength, margin, mode) {
  if (mode === 'hard') return length > maxLength ? 0 : baseReward;
  return clamp(baseReward - overlongPenalty(length, maxLength, margin, mode), 0, 1);
}

function clippedRatio(ratio, lowerClip, upperClip) {
  return clamp(ratio, lowerClip, upperClip);
}

function tokenContribution(ratio, advantage, lowerClip, upperClip) {
  const unclipped = ratio * advantage;
  const clipped = clippedRatio(ratio, lowerClip, upperClip) * advantage;
  return Math.min(unclipped, clipped);
}

function makePromptGroups({ groupSize, difficulty, maxResponseLength }) {
  return GROUP_SPECS.map((spec, groupIndex) => {
    const successProbability = clamp(1.05 - difficulty - spec.target * 0.36 + (groupIndex % 3) * 0.035, 0.02, 0.98);

    const completions = Array.from({ length: groupSize }, (_, sampleIndex) => {
      const threshold = THRESHOLDS[(groupIndex * 3 + sampleIndex) % THRESHOLDS.length];
      const baseReward = threshold < successProbability ? 1 : 0;
      const lengthScale = 0.34 + difficulty * 0.48 + spec.target * 0.28 + sampleIndex * 0.025 + (threshold > 0.7 ? 0.18 : 0);
      const length = Math.round(clamp(maxResponseLength * lengthScale, 96, maxResponseLength * 1.38));
      const ratioBase = 0.78 + ((groupIndex + 1) * (sampleIndex + 3) % 8) * 0.075 + (baseReward ? 0.09 : -0.03);
      const tokenRatios = TOKEN_LABELS.map((_, tokenIndex) => (
        clamp(ratioBase + (tokenIndex - 2) * 0.055 + (threshold - 0.5) * 0.12, 0.55, 1.62)
      ));

      return {
        id: `${spec.id}-C${sampleIndex + 1}`,
        label: `C${sampleIndex + 1}`,
        baseReward,
        length,
        tokenRatios,
        trace: baseReward
          ? sampleIndex % 2 === 0
            ? 'finds the key step and checks the final answer'
            : 'takes a detour, self-corrects, then verifies'
          : sampleIndex % 2 === 0
            ? 'follows a plausible route but misses the verifier'
            : 'runs long and never resolves the contradiction',
      };
    });

    return {
      ...spec,
      successProbability,
      completions,
    };
  });
}

function annotateGroups(groups, { lowerClip, upperClip, maxResponseLength, margin, shapingMode, overlongActive }) {
  return groups.map((group) => {
    const shapedRewards = group.completions.map((completion) => (
      overlongActive
        ? shapeReward(completion.baseReward, completion.length, maxResponseLength, margin, shapingMode)
        : completion.baseReward
    ));
    const m = mean(shapedRewards);
    const s = std(shapedRewards);

    return {
      ...group,
      accuracy: groupAccuracy(group),
      contrast: hasContrast(group),
      completions: group.completions.map((completion, index) => {
        const advantage = s > 1e-6 ? (shapedRewards[index] - m) / s : 0;
        const tokenContributions = completion.tokenRatios.map((ratio) => (
          tokenContribution(ratio, advantage, lowerClip, upperClip)
        ));
        const clippedTokens = completion.tokenRatios.filter((ratio) => ratio !== clippedRatio(ratio, lowerClip, upperClip)).length;

        return {
          ...completion,
          reward: shapedRewards[index],
          advantage,
          tokenContributions,
          clippedTokens,
        };
      }),
    };
  });
}

function flattenCompletions(groups) {
  return groups.flatMap((group) => group.completions.map((completion) => ({ ...completion, groupId: group.id })));
}

function clipPathPoints(upperClip, lowerClip) {
  return Array.from({ length: 34 }, (_, index) => {
    const ratio = 0.5 + (index / 33) * 1.12;
    const clipped = clippedRatio(ratio, lowerClip, upperClip);
    const x = 24 + ((ratio - 0.5) / 1.12) * 252;
    const y = 160 - ((clipped - 0.5) / 1.12) * 132;
    return `${x},${y}`;
  }).join(' ');
}

function rewardCurvePoints(mode, maxLength, margin) {
  const samples = Array.from({ length: 44 }, (_, index) => 0.55 + (index / 43) * 0.62);
  return samples.map((scale) => {
    const length = maxLength * scale;
    const reward = shapeReward(1, length, maxLength, margin, mode);
    const x = 24 + ((scale - 0.55) / 0.62) * 252;
    const y = 154 - reward * 118;
    return `${x},${y}`;
  }).join(' ');
}

function healthLabel(stats) {
  if (stats.effectiveGroups < Math.ceil(stats.totalGroups / 2)) return 'low signal';
  if (stats.entropy < 0.5) return 'entropy collapse';
  if (stats.overlongRate > 0.3) return 'too overlong';
  return 'healthy';
}

function ControlButton({ active, children, onClick, icon: Icon }) {
  return (
    <button
      type="button"
      data-math-control
      onClick={onClick}
      className={`flex min-h-[3rem] items-center gap-2 rounded border px-3 py-2 text-left text-xs font-bold transition ${
        active
          ? 'border-[var(--ds-accent)] bg-[var(--ds-accent)] text-[var(--ds-paper)]'
          : 'border-[var(--ds-rule)] bg-[var(--ds-paper)] text-[var(--ds-ink)] hover:bg-[var(--ds-paper-2)]'
      }`}
    >
      {Icon ? <Icon size={15} className="shrink-0" /> : null}
      <span>{children}</span>
    </button>
  );
}

function SliderControl({ label, value, min, max, step, onChange, display }) {
  return (
    <label className="grid gap-2 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3">
      <span className="flex items-center justify-between gap-3 text-[10px] font-black uppercase tracking-wide text-[var(--ds-muted)]">
        {label}
        <span className="font-mono text-[var(--ds-ink)]">{display || value}</span>
      </span>
      <input
        data-math-control
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

function SectionHeader({ icon: Icon, eyebrow, title, children }) {
  return (
    <div className="mb-4">
      <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-[var(--ds-accent)]">
        <Icon size={16} />
        {eyebrow}
      </p>
      <h2 className="mt-1 text-xl font-black tracking-tight text-[var(--ds-ink)]">{title}</h2>
      {children ? <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--ds-muted)]">{children}</p> : null}
    </div>
  );
}

function MetricTile({ label, value, detail, tone = 'neutral' }) {
  const toneClass = {
    neutral: 'border-[var(--ds-rule)] bg-[var(--ds-paper)]',
    good: 'border-emerald-300 bg-emerald-50',
    warn: 'border-amber-300 bg-amber-50',
    bad: 'border-rose-300 bg-rose-50',
    blue: 'border-sky-300 bg-sky-50',
  }[tone] || 'border-[var(--ds-rule)] bg-[var(--ds-paper)]';

  return (
    <div className={`rounded border p-4 ${toneClass}`}>
      <span className="text-[10px] font-black uppercase tracking-wide text-[var(--ds-muted)]">{label}</span>
      <strong className="mt-1 block break-words font-mono text-xl text-[var(--ds-ink)]">{value}</strong>
      <p className="mt-1 text-xs leading-5 text-[var(--ds-muted)]">{detail}</p>
    </div>
  );
}

function GroupCard({ group, dynamicActive }) {
  const kept = group.contrast;
  const status = group.accuracy === 1 ? 'all correct' : group.accuracy === 0 ? 'all wrong' : 'mixed';
  const statusClass = kept
    ? 'border-emerald-300 bg-emerald-50'
    : 'border-stone-300 bg-stone-100 opacity-70';

  return (
    <article className={`rounded border p-3 transition ${dynamicActive ? statusClass : 'border-[var(--ds-rule)] bg-[var(--ds-paper)]'}`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="font-mono text-[10px] font-black uppercase tracking-wide text-[var(--ds-muted)]">{group.id}</p>
          <h3 className="text-sm font-black text-[var(--ds-ink)]">{group.prompt}</h3>
        </div>
        <span className={`rounded border px-2 py-1 text-[10px] font-black uppercase tracking-wide ${
          kept ? 'border-emerald-300 text-emerald-800' : 'border-stone-300 text-stone-600'
        }`}>
          {dynamicActive ? (kept ? 'keep' : 'drop') : status}
        </span>
      </div>
      <div className="mt-3 grid grid-cols-4 gap-2">
        {group.completions.slice(0, 8).map((completion) => {
          const Icon = completion.baseReward > 0 ? CheckCircle2 : XCircle;
          return (
            <div
              key={completion.id}
              className={`flex min-h-[2.35rem] items-center justify-center rounded border ${
                completion.baseReward > 0
                  ? 'border-emerald-300 bg-white text-emerald-700'
                  : 'border-rose-300 bg-white text-rose-700'
              }`}
              title={`${completion.label}: reward ${completion.baseReward}`}
            >
              <Icon size={16} />
            </div>
          );
        })}
      </div>
      <div className="mt-3 flex items-center justify-between text-xs text-[var(--ds-muted)]">
        <span>accuracy {pct(group.accuracy)}</span>
        <span>{status}</span>
      </div>
    </article>
  );
}

function ClipCurve({ lowerClip, upperClip, clipHigherActive }) {
  const standardUpper = 1.2;
  const xFor = (ratio) => 24 + ((ratio - 0.5) / 1.12) * 252;

  return (
    <svg viewBox="0 0 300 190" className="h-56 w-full rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)]">
      <line x1="24" y1="160" x2="276" y2="160" stroke="#a8a29e" strokeWidth="1" />
      <line x1="24" y1="28" x2="24" y2="160" stroke="#a8a29e" strokeWidth="1" />
      <polyline points={clipPathPoints(standardUpper, 0.8)} fill="none" stroke="#a8a29e" strokeWidth="3" strokeDasharray="5 5" />
      <polyline points={clipPathPoints(upperClip, lowerClip)} fill="none" stroke="#0f766e" strokeWidth="4" />
      <line x1={xFor(standardUpper)} y1="30" x2={xFor(standardUpper)} y2="162" stroke="#a8a29e" strokeWidth="1.5" strokeDasharray="4 4" />
      <line x1={xFor(upperClip)} y1="30" x2={xFor(upperClip)} y2="162" stroke={clipHigherActive ? '#0f766e' : '#78716c'} strokeWidth="1.5" />
      <text x="28" y="22" className="fill-stone-700 text-[10px] font-bold">clipped contribution</text>
      <text x="220" y="178" className="fill-stone-700 text-[10px] font-bold">ratio</text>
      <text x={xFor(standardUpper) - 24} y="46" className="fill-stone-500 text-[10px]">1.20</text>
      <text x={xFor(upperClip) + 4} y="62" className="fill-teal-800 text-[10px] font-bold">{fmt(upperClip, 2)}</text>
    </svg>
  );
}

function TokenTimeline({ completion, tokenLevelActive, lowerClip, upperClip }) {
  if (!completion) return null;

  return (
    <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <p className="text-[10px] font-black uppercase tracking-wide text-[var(--ds-muted)]">selected completion</p>
          <h3 className="text-sm font-black text-[var(--ds-ink)]">
            {completion.groupId} / {completion.label}
          </h3>
        </div>
        <span className="rounded border border-[var(--ds-rule)] px-2 py-1 font-mono text-[10px] font-black">
          adv {signed(completion.advantage, 2)}
        </span>
      </div>

      {tokenLevelActive ? (
        <div className="grid gap-2 md:grid-cols-3">
          {completion.tokenRatios.map((ratio, index) => {
            const clipped = clippedRatio(ratio, lowerClip, upperClip);
            const clippedNow = clipped !== ratio;
            return (
              <div
                key={`${completion.id}-${TOKEN_LABELS[index]}`}
                className={`rounded border p-3 text-xs ${
                  clippedNow ? 'border-amber-300 bg-amber-50' : 'border-sky-200 bg-sky-50'
                }`}
              >
                <p className="font-black uppercase tracking-wide text-[var(--ds-muted)]">{TOKEN_LABELS[index]}</p>
                <div className="mt-2 grid grid-cols-2 gap-1 font-mono text-[11px]">
                  <span>r {fmt(ratio, 2)}</span>
                  <span>clip {fmt(clipped, 2)}</span>
                  <span className="col-span-2">term {signed(completion.tokenContributions[index], 2)}</span>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="rounded border border-stone-300 bg-stone-100 p-4">
          <p className="text-xs font-bold text-[var(--ds-ink)]">Sample-level view</p>
          <p className="mt-2 text-sm leading-6 text-[var(--ds-muted)]">
            The whole response is treated as one coarse action. That is simple, but a long reasoning trace hides many different token decisions behind one advantage.
          </p>
        </div>
      )}
    </div>
  );
}

function RewardCurve({ shapingMode, maxResponseLength, margin }) {
  return (
    <svg viewBox="0 0 300 182" className="h-52 w-full rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)]">
      <line x1="24" y1="154" x2="276" y2="154" stroke="#a8a29e" strokeWidth="1" />
      <line x1="24" y1="32" x2="24" y2="154" stroke="#a8a29e" strokeWidth="1" />
      <line x1="206" y1="32" x2="206" y2="154" stroke="#a8a29e" strokeWidth="1.5" strokeDasharray="5 5" />
      <polyline points={rewardCurvePoints(shapingMode, maxResponseLength, margin)} fill="none" stroke="#be123c" strokeWidth="4" />
      <text x="28" y="24" className="fill-stone-700 text-[10px] font-bold">reward after length rule</text>
      <text x="206" y="170" className="fill-stone-600 text-[10px]">max</text>
      <text x="218" y="48" className="fill-rose-800 text-[10px] font-bold">{shapingMode}</text>
    </svg>
  );
}

function MechanismBadge({ active, icon: Icon, title, body }) {
  return (
    <article className={`rounded border p-4 ${
      active ? 'border-[var(--ds-accent)] bg-[var(--ds-warm)]' : 'border-[var(--ds-rule)] bg-[var(--ds-paper)]'
    }`}>
      <div className="mb-2 flex items-center gap-2">
        <Icon className={active ? 'text-[var(--ds-accent)]' : 'text-[var(--ds-muted)]'} size={18} />
        <h3 className="text-sm font-black text-[var(--ds-ink)]">{title}</h3>
      </div>
      <p className="text-xs leading-5 text-[var(--ds-muted)]">{body}</p>
    </article>
  );
}

export default function DAPOReasoningRL() {
  const [mode, setMode] = useState('full-dapo');
  const [groupSize, setGroupSize] = useState(4);
  const [promptDifficulty, setPromptDifficulty] = useState(0.52);
  const [upperClip, setUpperClip] = useState(1.28);
  const [lowerClip, setLowerClip] = useState(0.8);
  const [maxResponseLength, setMaxResponseLength] = useState(2048);
  const [softLengthMargin, setSoftLengthMargin] = useState(256);
  const [overlongShaping, setOverlongShaping] = useState('linear');
  const [lossGranularity, setLossGranularity] = useState('token');

  const dynamicActive = mode === 'dynamic-sampling' || mode === 'full-dapo';
  const clipHigherActive = mode === 'clip-higher' || mode === 'full-dapo';
  const tokenLevelActive = mode === 'token-level' || mode === 'full-dapo' || (mode !== 'grpo' && lossGranularity === 'token');
  const overlongActive = mode === 'overlong-shaping' || mode === 'full-dapo';
  const activeUpperClip = clipHigherActive ? upperClip : 1.2;
  const activeLowerClip = lowerClip;
  const activeShaping = overlongActive ? overlongShaping : 'hard';

  const groups = useMemo(() => {
    const rawGroups = makePromptGroups({
      groupSize,
      difficulty: promptDifficulty,
      maxResponseLength,
    });

    return annotateGroups(rawGroups, {
      lowerClip: activeLowerClip,
      upperClip: activeUpperClip,
      maxResponseLength,
      margin: softLengthMargin,
      shapingMode: activeShaping,
      overlongActive,
    });
  }, [groupSize, promptDifficulty, maxResponseLength, activeLowerClip, activeUpperClip, softLengthMargin, activeShaping, overlongActive]);

  const trainingGroups = useMemo(() => {
    const qualified = groups.filter(hasContrast);
    if (dynamicActive) return qualified.slice(0, 5);
    return groups.slice(0, 5);
  }, [groups, dynamicActive]);

  const selectedCompletion = useMemo(() => {
    const completions = flattenCompletions(trainingGroups);
    return completions.find((completion) => completion.baseReward > 0 && completion.length >= maxResponseLength - softLengthMargin)
      || completions.find((completion) => completion.baseReward > 0)
      || completions[0]
      || null;
  }, [trainingGroups, maxResponseLength, softLengthMargin]);

  const stats = useMemo(() => {
    const allCompletions = flattenCompletions(trainingGroups);
    const allTokenRatios = allCompletions.flatMap((completion) => completion.tokenRatios);
    const rewards = allCompletions.map((completion) => completion.reward);
    const advantages = allCompletions.map((completion) => completion.advantage);
    const clippedTokens = allTokenRatios.filter((ratio) => ratio !== clippedRatio(ratio, activeLowerClip, activeUpperClip)).length;
    const overlongCount = allCompletions.filter((completion) => completion.length > maxResponseLength).length;
    const effectiveGroups = groups.filter(hasContrast).length;
    const signalRatio = groups.length ? effectiveGroups / groups.length : 0;
    const entropy = clamp(
      1.22
        + (clipHigherActive ? (activeUpperClip - 1.2) * 1.45 : -0.28)
        + (dynamicActive ? 0.14 : -0.08)
        - (signalRatio < 0.5 ? 0.3 : 0)
        - (overlongCount / Math.max(1, allCompletions.length)) * 0.28,
      0.08,
      1.75,
    );
    const clipFraction = allTokenRatios.length ? clippedTokens / allTokenRatios.length : 0;
    const overlongRate = allCompletions.length ? overlongCount / allCompletions.length : 0;
    const nonzeroAdvantages = advantages.filter((advantage) => Math.abs(advantage) > 0.05).length;
    const tokenGradientCount = tokenLevelActive
      ? allCompletions
        .filter((completion) => Math.abs(completion.advantage) > 0.05)
        .reduce((sum, completion) => sum + completion.tokenRatios.length, 0)
      : nonzeroAdvantages;
    const efficiency = clamp(
      100
        * signalRatio
        * clamp(entropy / 1.35, 0.1, 1.18)
        * (1 - overlongRate * 0.45)
        * (tokenLevelActive ? 1 : 0.62)
        * (clipHigherActive ? 1.05 : 0.86),
      0,
      100,
    );

    return {
      totalGroups: groups.length,
      batchGroups: trainingGroups.length,
      effectiveGroups,
      meanReward: mean(rewards),
      rewardVariance: variance(rewards),
      nonzeroAdvantages,
      entropy,
      clipFraction,
      averageLength: mean(allCompletions.map((completion) => completion.length)),
      overlongRate,
      tokenGradientCount,
      efficiency,
    };
  }, [trainingGroups, groups, activeLowerClip, activeUpperClip, maxResponseLength, tokenLevelActive, clipHigherActive, dynamicActive]);

  const activeMode = MODES.find((item) => item.id === mode) || MODES[0];
  const health = healthLabel(stats);
  const healthTone = health === 'healthy' ? 'good' : health === 'low signal' ? 'warn' : 'bad';

  return (
    <div className="min-h-screen bg-[var(--ds-paper)] text-[var(--ds-ink)]">
      <header className="border-b border-[var(--ds-rule)] bg-[var(--ds-panel)]">
        <div className="mx-auto grid max-w-7xl gap-6 px-5 py-7 lg:grid-cols-[1fr_25rem] lg:items-end">
          <div>
            <div className="mb-3 flex flex-wrap gap-2">
              <span className="rounded border border-[var(--ds-rule)] bg-[var(--ds-warm)] px-2 py-1 text-[10px] font-black uppercase tracking-wide">
                Reinforcement Learning
              </span>
              <span className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] px-2 py-1 text-[10px] font-black uppercase tracking-wide text-[var(--ds-muted)]">
                sequel to GRPO
              </span>
            </div>
            <h1 className="max-w-4xl text-3xl font-black tracking-tight md:text-5xl">
              DAPO: Fixing the Hidden Failure Modes of GRPO
            </h1>
            <p className="mt-4 max-w-3xl text-sm leading-7 text-[var(--ds-muted)] md:text-base">
              GRPO gives the clean learning signal: sample a group, score answers, compare within the group, and update the policy. DAPO teaches the practical recipe needed for long chain-of-thought RL: better batches, safer clipping, token-level credit, and less noisy length rewards.
            </p>
          </div>
          <div className="grid gap-3 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
            <MetricTile label="active mode" value={activeMode.label} detail={activeMode.detail} tone="blue" />
            <div className="grid grid-cols-2 gap-3">
              <MetricTile label="health" value={health} detail="live batch diagnosis" tone={healthTone} />
              <MetricTile label="efficiency" value={pct(stats.efficiency / 100)} detail="useful signal score" />
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-5 py-6">
        <section className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
          <SectionHeader icon={SlidersHorizontal} eyebrow="interactive controls" title="Turn GRPO Into DAPO">
            Move the batch, clipping, loss, and length controls to see why large-scale reasoning RL is a system, not just one objective.
          </SectionHeader>

          <div className="grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
            <div className="grid gap-3">
              <div>
                <p className="mb-2 text-xs font-black uppercase tracking-wide text-[var(--ds-muted)]">Mode</p>
                <div className="grid gap-2 md:grid-cols-3">
                  {MODES.map((item) => (
                    <ControlButton key={item.id} active={mode === item.id} onClick={() => setMode(item.id)}>
                      {item.label}
                    </ControlButton>
                  ))}
                </div>
              </div>
              <div className="grid gap-3 md:grid-cols-3">
                <div>
                  <p className="mb-2 text-xs font-black uppercase tracking-wide text-[var(--ds-muted)]">Group size</p>
                  <div className="grid grid-cols-4 gap-2">
                    {GROUP_SIZES.map((size) => (
                      <ControlButton key={size} active={groupSize === size} onClick={() => setGroupSize(size)}>
                        {size}
                      </ControlButton>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="mb-2 text-xs font-black uppercase tracking-wide text-[var(--ds-muted)]">Max response length</p>
                  <div className="grid grid-cols-2 gap-2">
                    {MAX_LENGTHS.map((length) => (
                      <ControlButton key={length} active={maxResponseLength === length} onClick={() => setMaxResponseLength(length)}>
                        {length}
                      </ControlButton>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="mb-2 text-xs font-black uppercase tracking-wide text-[var(--ds-muted)]">Overlong shaping</p>
                  <div className="grid gap-2">
                    {['hard', 'linear', 'soft'].map((item) => (
                      <ControlButton key={item} active={overlongShaping === item} onClick={() => setOverlongShaping(item)}>
                        {item}
                      </ControlButton>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <div className="grid gap-3">
              <SliderControl
                label="prompt difficulty"
                value={promptDifficulty}
                min={0.18}
                max={0.88}
                step={0.01}
                onChange={setPromptDifficulty}
                display={difficultyLabel(promptDifficulty)}
              />
              <SliderControl label="upper clip" value={upperClip} min={1.1} max={1.4} step={0.01} onChange={setUpperClip} display={fmt(upperClip, 2)} />
              <SliderControl label="lower clip" value={lowerClip} min={0.7} max={0.9} step={0.01} onChange={setLowerClip} display={fmt(lowerClip, 2)} />
              <SliderControl label="soft length margin" value={softLengthMargin} min={64} max={768} step={64} onChange={setSoftLengthMargin} display={`${softLengthMargin} tokens`} />
              <div className="grid grid-cols-2 gap-2">
                <ControlButton active={lossGranularity === 'sample'} onClick={() => setLossGranularity('sample')} icon={Layers3}>
                  sample-level loss
                </ControlButton>
                <ControlButton active={lossGranularity === 'token'} onClick={() => setLossGranularity('token')} icon={Maximize2}>
                  token-level loss
                </ControlButton>
              </div>
            </div>
          </div>
        </section>

        <section className="mt-6 grid gap-6 xl:grid-cols-2">
          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <SectionHeader icon={Brain} eyebrow="rollout groups" title="GRPO Works When Siblings Disagree">
              All-correct and all-wrong groups consume rollout budget but provide weak relative contrast. Mixed groups carry the useful signal.
            </SectionHeader>
            <div className="grid gap-3 md:grid-cols-2">
              {groups.map((group) => (
                <GroupCard key={group.id} group={group} dynamicActive={dynamicActive} />
              ))}
            </div>
          </div>

          <div className="grid gap-6">
            <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
              <SectionHeader icon={Filter} eyebrow="dynamic sampling" title="Keep The Useful Batch">
                Dynamic Sampling filters prompt groups whose accuracies are all 1 or all 0, keeping the optimizer focused on effective gradients.
              </SectionHeader>
              <div className="grid gap-3 md:grid-cols-2">
                <MetricTile label="raw groups" value={stats.totalGroups} detail="rollout pool" />
                <MetricTile label="effective groups" value={stats.effectiveGroups} detail="groups with mixed success" tone={stats.effectiveGroups >= stats.totalGroups / 2 ? 'good' : 'warn'} />
                <MetricTile label="batch groups" value={stats.batchGroups} detail={dynamicActive ? 'kept after filter' : 'sent to optimizer'} />
                <MetricTile label="nonzero advantages" value={stats.nonzeroAdvantages} detail="samples with update direction" />
              </div>
              <div className="mt-4 grid gap-2">
                {groups.map((group) => (
                  <div key={`${group.id}-filter`} className="grid grid-cols-[3rem_1fr_4rem] items-center gap-3 text-xs">
                    <span className="font-mono font-black">{group.id}</span>
                    <div className="h-3 overflow-hidden rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)]">
                      <div
                        className={group.contrast ? 'h-full bg-emerald-600' : 'h-full bg-stone-400'}
                        style={{ width: `${Math.max(8, group.accuracy * 100)}%` }}
                      />
                    </div>
                    <span className="text-right font-mono">{group.contrast ? 'keep' : 'drop'}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
              <SectionHeader icon={Gauge} eyebrow="training health" title="Signal Dashboard" />
              <div className="grid gap-3 md:grid-cols-3">
                <MetricTile label="mean reward" value={fmt(stats.meanReward, 2)} detail="after shaping" />
                <MetricTile label="reward variance" value={fmt(stats.rewardVariance, 3)} detail="contrast inside batch" />
                <MetricTile label="entropy" value={fmt(stats.entropy, 2)} detail="diversity proxy" tone={stats.entropy < 0.5 ? 'bad' : 'good'} />
                <MetricTile label="clip fraction" value={pct(stats.clipFraction)} detail="tokens ratio-clamped" />
                <MetricTile label="avg length" value={Math.round(stats.averageLength)} detail="response tokens" />
                <MetricTile label="overlong rate" value={pct(stats.overlongRate)} detail="past max length" tone={stats.overlongRate > 0.3 ? 'bad' : 'neutral'} />
              </div>
            </div>
          </div>
        </section>

        <section className="mt-6 grid gap-6 xl:grid-cols-2">
          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <SectionHeader icon={ShieldCheck} eyebrow="clip-higher" title="Give Positive Updates More Room">
              PPO-style clipping prevents policy drift. DAPO decouples lower and upper clip bounds so successful traces can grow without making negative updates loose.
            </SectionHeader>
            <ClipCurve lowerClip={activeLowerClip} upperClip={activeUpperClip} clipHigherActive={clipHigherActive} />
            <div className="mt-4 grid gap-3 md:grid-cols-3">
              <MetricTile label="lower bound" value={fmt(activeLowerClip, 2)} detail="controls negative drift" />
              <MetricTile label="standard upper" value="1.20" detail="symmetric PPO-style cap" />
              <MetricTile label="active upper" value={fmt(activeUpperClip, 2)} detail={clipHigherActive ? 'Clip-Higher active' : 'baseline cap'} tone={clipHigherActive ? 'good' : 'neutral'} />
            </div>
          </div>

          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <SectionHeader icon={TrendingDown} eyebrow="overlong shaping" title="Replace A Noisy Length Cliff">
              A one-token boundary should not always flip a useful trace from full reward to zero. Shaping makes the length signal smoother near the max.
            </SectionHeader>
            <RewardCurve shapingMode={activeShaping} maxResponseLength={maxResponseLength} margin={softLengthMargin} />
            <div className="mt-4 grid gap-3 md:grid-cols-3">
              <MetricTile label="max length" value={maxResponseLength} detail="hard response cap" />
              <MetricTile label="soft margin" value={softLengthMargin} detail="penalty starts early" />
              <MetricTile label="mode" value={activeShaping} detail={overlongActive ? 'shaping active' : 'baseline hard cliff'} tone={overlongActive ? 'good' : 'neutral'} />
            </div>
          </div>
        </section>

        <section className="mt-6 grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <SectionHeader icon={Target} eyebrow="token-level policy gradient" title="A Long Response Is Many Actions">
              In long-CoT RL, token decisions accumulate over thousands of steps. Token-level loss makes the update denser than one response-sized block.
            </SectionHeader>
            <TokenTimeline
              completion={selectedCompletion}
              tokenLevelActive={tokenLevelActive}
              lowerClip={activeLowerClip}
              upperClip={activeUpperClip}
            />
            <div className="mt-4 grid gap-3 md:grid-cols-3">
              <MetricTile label="loss granularity" value={tokenLevelActive ? 'token' : 'sample'} detail="active view" />
              <MetricTile label="tokens contributing" value={stats.tokenGradientCount} detail="gradient terms counted" />
              <MetricTile label="selected length" value={selectedCompletion?.length || 0} detail="completion tokens" />
            </div>
          </div>

          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <SectionHeader icon={GitBranch} eyebrow="full pipeline" title="DAPO Is A Recipe">
              The paper lesson is not just a new name for GRPO. It is the combination of rollout filtering, shaped rewards, group-relative advantages, token-level loss, and asymmetric clipping.
            </SectionHeader>
            <div className="grid gap-3">
              {[
                ['Prompts', 'rollout groups'],
                ['Dynamic Sampling', 'keep mixed groups'],
                ['Reward', 'rule-based score plus length shaping'],
                ['Advantage', 'compare inside group'],
                ['Loss', 'token-level policy gradient'],
                ['Objective', 'Clip-Higher update'],
              ].map(([left, right], index, arr) => (
                <div key={left} className="flex items-center gap-3">
                  <div className="grid min-h-[3.25rem] flex-1 grid-cols-[9rem_1fr] items-center rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] px-3 py-2 text-sm">
                    <span className="font-black text-[var(--ds-ink)]">{left}</span>
                    <span className="text-xs leading-5 text-[var(--ds-muted)]">{right}</span>
                  </div>
                  {index < arr.length - 1 ? <ArrowRight className="h-4 w-4 shrink-0 text-[var(--ds-muted)]" /> : null}
                </div>
              ))}
            </div>
            <div className="mt-4 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
              <p className="text-sm font-black text-[var(--ds-ink)]">Reported result card</p>
              <p className="mt-2 text-xs leading-6 text-[var(--ds-muted)]">
                The DAPO paper reports 50 points on AIME 2024 with Qwen2.5-32B, compared with the reported 47 points for DeepSeek-R1-Zero-Qwen-32B while using 50 percent of the training steps. Treat this as an author-reported result, not a universal guarantee.
              </p>
            </div>
          </div>
        </section>

        <section className="mt-6 rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
          <SectionHeader icon={Zap} eyebrow="four fixes" title="What Each DAPO Fix Changes" />
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <MechanismBadge
              active={dynamicActive}
              icon={Filter}
              title="Dynamic Sampling"
              body="Drops prompt groups that are all correct or all wrong so the batch stays full of useful comparisons."
            />
            <MechanismBadge
              active={clipHigherActive}
              icon={TrendingUp}
              title="Clip-Higher"
              body="Keeps the lower ratio bound controlled while giving positive-advantage traces a higher upper cap."
            />
            <MechanismBadge
              active={tokenLevelActive}
              icon={Layers3}
              title="Token-Level Loss"
              body="Aggregates policy-gradient terms across tokens, which matters for long reasoning traces."
            />
            <MechanismBadge
              active={overlongActive}
              icon={AlertTriangle}
              title="Overlong Shaping"
              body="Turns the reward cliff near max response length into a smoother penalty signal."
            />
          </div>
        </section>

        <section className="mt-6 grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
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

          <div className="grid gap-6">
            <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
              <SectionHeader icon={Code2} eyebrow="Rustlings-style lab" title="mini-dapo Exercises">
                The local Rust pack turns DAPO mechanics into small functions for group filtering, clipped ratios, token loss, shaping, entropy, and training health.
              </SectionHeader>
              <div className="grid gap-2 md:grid-cols-2">
                {EXERCISES.map(([file, description]) => (
                  <div key={file} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3">
                    <p className="font-mono text-xs font-black text-[var(--ds-ink)]">{file}</p>
                    <p className="mt-1 text-xs leading-5 text-[var(--ds-muted)]">{description}</p>
                  </div>
                ))}
              </div>
              <pre className="mt-4 overflow-x-auto rounded border border-[var(--ds-rule)] bg-stone-950 p-3 text-xs leading-5 text-stone-100">cd mini-dapo
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
            </div>
          </div>
        </section>

        <div className="mt-6">
          <AssessmentPanel lessonId="dapo-reasoning-rl" title="DAPO reasoning RL check" />
        </div>
      </main>
    </div>
  );
}
