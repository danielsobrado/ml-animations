import React, { useMemo, useState } from 'react';
import {
  ArrowRight,
  BarChart3,
  BookOpen,
  Brain,
  CheckCircle2,
  Code2,
  ExternalLink,
  FileText,
  Gauge,
  GitBranch,
  HelpCircle,
  Layers3,
  Link as LinkIcon,
  RotateCcw,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  TrendingUp,
  XCircle,
} from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const TRAINING_MODES = [
  { id: 'sft', label: 'SFT', detail: 'Imitate one chosen answer.' },
  { id: 'rejection-sampling', label: 'Rejection Sampling', detail: 'Sample many, keep correct ones, then imitate.' },
  { id: 'ppo', label: 'PPO', detail: 'Use a critic/value model as the baseline.' },
  { id: 'grpo', label: 'GRPO', detail: 'Use sibling answer rewards as the baseline.' },
  { id: 'r1-zero', label: 'R1-Zero Pipeline', detail: 'Pure RL from a base model.' },
  { id: 'r1', label: 'R1 Pipeline', detail: 'Cold start, reasoning RL, SFT data, final RL.' },
];

const REWARD_TYPES = [
  { id: 'exact-answer', label: 'exact answer', detail: 'Binary final-answer checker.' },
  { id: 'format', label: 'format reward', detail: 'Small reward for required tags.' },
  { id: 'model-judge', label: 'model judge', detail: 'Soft correctness/helpfulness score.' },
  { id: 'mixed', label: 'mixed', detail: 'Correctness plus small shaping signals.' },
];

const DIFFICULTIES = [
  { id: 'easy', label: 'easy', prompt: 'Solve: If 2x + 3 = 11, what is x?' },
  { id: 'medium', label: 'medium', prompt: 'Find N if the proper factors of N sum to 12.' },
  { id: 'hard', label: 'hard', prompt: 'Solve the proof-style problem and verify the final answer.' },
];

const GROUP_SIZES = [2, 4, 8, 16];

const COMPLETION_BANK = [
  {
    id: 'a',
    label: 'A1',
    title: 'clean correct',
    answer: 'x = 4',
    trace: 'Subtract 3: 2x = 8. Divide by 2: x = 4. Check: 2*4+3=11.',
    correctnessReward: 1,
    formatReward: 0.1,
    judgeReward: 0.96,
    oldLogProb: -2.2,
    newLogProb: -1.8,
    length: 42,
  },
  {
    id: 'b',
    label: 'A2',
    title: 'wrong arithmetic',
    answer: 'x = 5',
    trace: 'Subtract 3 and divide, but mistakenly computes 8 / 2 as 5.',
    correctnessReward: 0,
    formatReward: 0.1,
    judgeReward: 0.2,
    oldLogProb: -2.0,
    newLogProb: -2.3,
    length: 34,
  },
  {
    id: 'c',
    label: 'A3',
    title: 'correct, short',
    answer: 'x = 4',
    trace: '2x + 3 = 11 -> 2x = 8 -> x = 4.',
    correctnessReward: 1,
    formatReward: 0,
    judgeReward: 0.78,
    oldLogProb: -1.9,
    newLogProb: -1.7,
    length: 18,
  },
  {
    id: 'd',
    label: 'A4',
    title: 'wrong guess',
    answer: 'x = 3',
    trace: 'Tries x = 3 and stops even though 2*3+3 = 9.',
    correctnessReward: 0,
    formatReward: 0,
    judgeReward: 0.08,
    oldLogProb: -2.4,
    newLogProb: -2.6,
    length: 20,
  },
  {
    id: 'e',
    label: 'A5',
    title: 'self-correcting',
    answer: 'x = 4',
    trace: 'First tries x = 5, checks 13, rewinds, solves 2x = 8, then returns x = 4.',
    correctnessReward: 1,
    formatReward: 0.1,
    judgeReward: 0.92,
    oldLogProb: -2.8,
    newLogProb: -2.2,
    length: 76,
  },
  {
    id: 'f',
    label: 'A6',
    title: 'format-only wrong',
    answer: 'x = 5',
    trace: '<think>Looks organized but keeps the wrong division.</think><answer>x = 5</answer>',
    correctnessReward: 0,
    formatReward: 0.1,
    judgeReward: 0.3,
    oldLogProb: -2.3,
    newLogProb: -2.4,
    length: 46,
  },
  {
    id: 'g',
    label: 'A7',
    title: 'overlong correct',
    answer: 'x = 4',
    trace: 'Solves correctly, then repeats the same check several times before finalizing the answer.',
    correctnessReward: 1,
    formatReward: 0.1,
    judgeReward: 0.74,
    oldLogProb: -2.5,
    newLogProb: -2.1,
    length: 140,
  },
  {
    id: 'h',
    label: 'A8',
    title: 'contradiction missed',
    answer: 'x = 3',
    trace: 'Finds 2x = 8, writes x = 3, and does not check the contradiction.',
    correctnessReward: 0,
    formatReward: 0.1,
    judgeReward: 0.18,
    oldLogProb: -2.1,
    newLogProb: -2.5,
    length: 38,
  },
  {
    id: 'i',
    label: 'A9',
    title: 'exact only',
    answer: '4',
    trace: '4',
    correctnessReward: 1,
    formatReward: 0,
    judgeReward: 0.55,
    oldLogProb: -1.8,
    newLogProb: -1.65,
    length: 1,
  },
  {
    id: 'j',
    label: 'A10',
    title: 'algebra slip',
    answer: 'x = 7',
    trace: 'Moves the 3 to the other side incorrectly and solves 2x = 14.',
    correctnessReward: 0,
    formatReward: 0,
    judgeReward: 0.12,
    oldLogProb: -2.6,
    newLogProb: -2.9,
    length: 28,
  },
  {
    id: 'k',
    label: 'A11',
    title: 'clean factor solution',
    answer: 'N = 121',
    trace: 'Let N = p^2. Proper factors are 1 and p, so 1+p=12 and p=11. Thus N=121.',
    correctnessReward: 1,
    formatReward: 0.1,
    judgeReward: 0.94,
    oldLogProb: -3.0,
    newLogProb: -2.45,
    length: 72,
  },
  {
    id: 'l',
    label: 'A12',
    title: 'wrong factor list',
    answer: 'N = 12',
    trace: 'Lists factors of 12 and confuses the target number with the desired sum.',
    correctnessReward: 0,
    formatReward: 0.1,
    judgeReward: 0.22,
    oldLogProb: -2.2,
    newLogProb: -2.55,
    length: 44,
  },
  {
    id: 'm',
    label: 'A13',
    title: 'verified factor search',
    answer: 'N = 121',
    trace: 'Tests several candidates, notices 21 is close but wrong, then verifies factors of 121 are 1 and 11.',
    correctnessReward: 1,
    formatReward: 0.1,
    judgeReward: 0.86,
    oldLogProb: -3.1,
    newLogProb: -2.5,
    length: 118,
  },
  {
    id: 'n',
    label: 'A14',
    title: 'all style',
    answer: 'N = 11',
    trace: '<think>Detailed tags and fluent prose, but answers the prime instead of its square.</think><answer>11</answer>',
    correctnessReward: 0,
    formatReward: 0.1,
    judgeReward: 0.28,
    oldLogProb: -2.7,
    newLogProb: -2.8,
    length: 64,
  },
  {
    id: 'o',
    label: 'A15',
    title: 'hard wrong',
    answer: 'unverified',
    trace: 'Tries a sophisticated route, reaches a contradiction, but still outputs the first answer.',
    correctnessReward: 0,
    formatReward: 0.1,
    judgeReward: 0.36,
    oldLogProb: -3.2,
    newLogProb: -3.0,
    length: 126,
  },
  {
    id: 'p',
    label: 'A16',
    title: 'hard self-check',
    answer: 'verified',
    trace: 'Breaks the problem into cases, checks the boundary case, rejects a false shortcut, then finalizes.',
    correctnessReward: 1,
    formatReward: 0.1,
    judgeReward: 0.9,
    oldLogProb: -3.4,
    newLogProb: -2.7,
    length: 154,
  },
];

const GROUP_ORDERS = {
  easy: ['a', 'c', 'i', 'g', 'e', 'b', 'd', 'h', 'f', 'j', 'k', 'm', 'l', 'n', 'o', 'p'],
  medium: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 'i', 'j', 'o', 'p'],
  hard: ['d', 'f', 'h', 'j', 'l', 'n', 'o', 'b', 'p', 'm', 'k', 'e', 'a', 'c', 'g', 'i'],
};

const PIPELINE_STAGES = [
  {
    title: 'R1-Zero',
    steps: ['Base model', 'Large-scale GRPO RL', 'Emergent reflection'],
    detail: 'Pure RL explores solution traces without a cold-start SFT stage.',
  },
  {
    title: 'DeepSeek-R1',
    steps: ['Cold-start traces', 'Reasoning RL', 'Rejection sampling + SFT', 'Final RL'],
    detail: 'Cold-start data shapes the behavior into a readable user-facing model.',
  },
  {
    title: 'Distillation',
    steps: ['Large R1 teacher', 'Generated reasoning data', 'Small student'],
    detail: 'Smaller models imitate high-reward reasoning traces from the stronger teacher.',
  },
];

const QUESTIONS = [
  ['What does GRPO sample for each prompt?', 'A group of candidate completions.'],
  ['What is the group baseline?', 'The average reward of the sampled answers for the same prompt.'],
  ['What happens to above-average answers?', 'The policy update makes their traces more likely.'],
  ['Why can all-correct or all-wrong groups be weak?', 'There is little contrast, so relative advantage gives little direction.'],
  ['Why keep a KL guardrail?', 'It discourages the updated policy from drifting too far from a reference model.'],
];

const EXERCISES = [
  ['01_group_mean.rs', 'Compute the mean reward inside a sampled group.'],
  ['02_group_advantage.rs', 'Normalize rewards into group-relative advantages.'],
  ['03_reward_correctness.rs', 'Build an exact-answer reward.'],
  ['04_format_reward.rs', 'Give a small reward for answer tags.'],
  ['05_policy_ratio.rs', 'Compute the new-policy to old-policy probability ratio.'],
  ['06_clip_objective.rs', 'Apply the clipped PPO-style surrogate term.'],
  ['07_kl_penalty.rs', 'Compute a toy discrete KL penalty.'],
  ['08_grpo_update_signal.rs', 'Map advantage signs to reinforce, suppress, or neutral.'],
  ['09_batch_filtering.rs', 'Detect groups with useful reward contrast.'],
  ['10_distillation_dataset.rs', 'Keep high-reward teacher outputs for a student dataset.'],
];

const SOURCE_LINKS = [
  {
    label: 'DeepSeek-R1 technical report',
    href: 'https://arxiv.org/html/2501.12948v1',
    note: 'R1-Zero pure RL, R1 cold-start pipeline, rule-based rewards, aha moment, and distillation.',
  },
  {
    label: 'DeepSeekMath GRPO paper',
    href: 'https://arxiv.org/html/2402.03300v3',
    note: 'GRPO as a PPO variant that avoids a learned critic by using group scores as the baseline.',
  },
];

function mean(xs) {
  if (!xs.length) return 0;
  return xs.reduce((sum, x) => sum + x, 0) / xs.length;
}

function std(xs) {
  if (!xs.length) return 0;
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function fmt(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function signed(value, digits = 2) {
  const rendered = fmt(value, digits);
  return value > 0 ? `+${rendered}` : rendered;
}

function rewardFor(completion, rewardType) {
  if (rewardType === 'exact-answer') return completion.correctnessReward;
  if (rewardType === 'format') return completion.formatReward;
  if (rewardType === 'model-judge') return completion.judgeReward;

  const lengthPenalty = Math.max(0, completion.length - 90) * 0.002;
  return clamp(
    completion.correctnessReward
      + completion.formatReward
      + completion.judgeReward * 0.2
      - lengthPenalty,
    0,
    1.3,
  );
}

function groupAdvantages(rewards) {
  const m = mean(rewards);
  const s = std(rewards);
  if (s === 0) return rewards.map(() => 0);
  return rewards.map((reward) => (reward - m) / s);
}

function getDirection(advantage) {
  if (advantage > 0.05) return 'reinforce';
  if (advantage < -0.05) return 'suppress';
  return 'neutral';
}

function ControlButton({ active, children, onClick }) {
  return (
    <button
      type="button"
      data-math-control
      onClick={onClick}
      className={`rounded border px-3 py-2 text-left text-xs font-bold transition ${
        active
          ? 'border-[var(--ds-accent)] bg-[var(--ds-accent)] text-[var(--ds-paper)]'
          : 'border-[var(--ds-rule)] bg-[var(--ds-paper)] text-[var(--ds-ink)] hover:bg-[var(--ds-paper-2)]'
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
      <h2 className="mt-1 text-xl font-black tracking-tight text-[var(--ds-ink)]">{title}</h2>
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

function CompletionCard({ completion }) {
  const isPositive = completion.advantage > 0.05;
  const isNegative = completion.advantage < -0.05;
  const Icon = isPositive ? CheckCircle2 : isNegative ? XCircle : HelpCircle;
  const tone = isPositive
    ? 'border-emerald-500 bg-emerald-50 text-emerald-900'
    : isNegative
      ? 'border-rose-400 bg-rose-50 text-rose-900'
      : 'border-[var(--ds-rule)] bg-[var(--ds-paper)] text-[var(--ds-ink)]';

  return (
    <article className={`rounded border p-4 ${tone}`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-[10px] font-black uppercase tracking-wide opacity-70">{completion.label}</p>
          <h3 className="text-sm font-black">{completion.title}</h3>
        </div>
        <Icon size={18} />
      </div>
      <p className="mt-3 min-h-[3rem] text-xs leading-5">{completion.trace}</p>
      <div className="mt-4 grid grid-cols-3 gap-2 text-center font-mono text-[11px]">
        <div className="rounded border border-current/20 bg-white/50 p-2">
          <span className="block opacity-70">reward</span>
          <strong>{fmt(completion.reward, 2)}</strong>
        </div>
        <div className="rounded border border-current/20 bg-white/50 p-2">
          <span className="block opacity-70">adv</span>
          <strong>{signed(completion.advantage, 2)}</strong>
        </div>
        <div className="rounded border border-current/20 bg-white/50 p-2">
          <span className="block opacity-70">update</span>
          <strong>{completion.direction}</strong>
        </div>
      </div>
    </article>
  );
}

function FlowBlock({ children, active }) {
  return (
    <span
      className={`rounded border px-3 py-2 text-center text-xs font-bold ${
        active
          ? 'border-[var(--ds-accent)] bg-[var(--ds-accent)] text-[var(--ds-paper)]'
          : 'border-[var(--ds-rule)] bg-[var(--ds-paper)] text-[var(--ds-ink)]'
      }`}
    >
      {children}
    </span>
  );
}

function FlowRow({ label, blocks, active }) {
  return (
    <div className={`rounded border p-4 ${active ? 'border-[var(--ds-accent)] bg-[var(--ds-warm)]' : 'border-[var(--ds-rule)] bg-[var(--ds-panel)]'}`}>
      <p className="mb-3 text-xs font-black uppercase tracking-wide text-[var(--ds-muted)]">{label}</p>
      <div className="flex flex-wrap items-center gap-2">
        {blocks.map((block, index) => (
          <React.Fragment key={`${label}-${block}`}>
            <FlowBlock active={active && index === blocks.length - 1}>{block}</FlowBlock>
            {index < blocks.length - 1 ? <ArrowRight className="h-4 w-4 text-[var(--ds-muted)]" /> : null}
          </React.Fragment>
        ))}
      </div>
    </div>
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

export default function GRPOReasoning() {
  const [mode, setMode] = useState('grpo');
  const [groupSize, setGroupSize] = useState(4);
  const [rewardType, setRewardType] = useState('mixed');
  const [klBeta, setKlBeta] = useState(0.08);
  const [clipEps, setClipEps] = useState(0.2);
  const [temperature, setTemperature] = useState(0.7);
  const [difficulty, setDifficulty] = useState('medium');

  const activePrompt = DIFFICULTIES.find((item) => item.id === difficulty)?.prompt || DIFFICULTIES[1].prompt;
  const activeMode = TRAINING_MODES.find((item) => item.id === mode) || TRAINING_MODES[3];

  const group = useMemo(() => {
    const byId = new Map(COMPLETION_BANK.map((completion) => [completion.id, completion]));
    const order = GROUP_ORDERS[difficulty] || GROUP_ORDERS.medium;
    const offset = clamp(Math.round((temperature - 0.2) * 2), 0, order.length - 1);
    const rotated = [...order.slice(offset), ...order.slice(0, offset)];
    const rawGroup = rotated.slice(0, groupSize).map((id) => byId.get(id)).filter(Boolean);
    const rewards = rawGroup.map((completion) => rewardFor(completion, rewardType));
    const advantages = groupAdvantages(rewards);

    return rawGroup.map((completion, index) => {
      const probabilityRatio = Math.exp(completion.newLogProb - completion.oldLogProb);
      const clippedRatio = clamp(probabilityRatio, 1 - clipEps, 1 + clipEps);
      const advantage = advantages[index];
      const surrogate = Math.min(probabilityRatio * advantage, clippedRatio * advantage);
      const direction = getDirection(advantage);

      return {
        ...completion,
        reward: rewards[index],
        advantage,
        probabilityRatio,
        clippedRatio,
        surrogate,
        direction,
      };
    });
  }, [difficulty, groupSize, rewardType, temperature, clipEps]);

  const stats = useMemo(() => {
    const rewards = group.map((completion) => completion.reward);
    const advantages = group.map((completion) => completion.advantage);
    const positive = group.filter((completion) => completion.advantage > 0.05).length;
    const negative = group.filter((completion) => completion.advantage < -0.05).length;
    const neutral = group.length - positive - negative;
    const allCorrect = group.every((completion) => completion.correctnessReward === 1);
    const allWrong = group.every((completion) => completion.correctnessReward === 0);
    const averageRatio = mean(group.map((completion) => completion.probabilityRatio));
    const drift = mean(group.map((completion) => Math.abs(completion.newLogProb - completion.oldLogProb)));
    const klPenalty = klBeta * drift * (0.75 + temperature * 0.35);
    const updateStrength = mean(advantages.map((advantage) => Math.abs(advantage))) * (1 - clamp(klBeta, 0, 0.6));
    const estimatedAccuracy = clamp(0.36 + positive * 0.06 + (difficulty === 'easy' ? 0.18 : difficulty === 'hard' ? -0.08 : 0), 0.05, 0.96);
    const reasoningLength = Math.round(36 + temperature * 50 + positive * 14 + (difficulty === 'hard' ? 46 : 0));

    return {
      meanReward: mean(rewards),
      stdReward: std(rewards),
      positive,
      negative,
      neutral,
      allCorrect,
      allWrong,
      averageRatio,
      klPenalty,
      updateStrength,
      estimatedAccuracy,
      reasoningLength,
    };
  }, [group, klBeta, temperature, difficulty]);

  const contrastMessage = stats.stdReward === 0
    ? 'No reward contrast: the group baseline cannot prefer one sibling answer over another.'
    : stats.allCorrect
      ? 'All answers are correct: the signal mostly separates style, length, and format.'
      : stats.allWrong
        ? 'All answers are wrong: relative rewards can accidentally reinforce the least bad wrong trace.'
        : 'Mixed quality group: GRPO has a clear reinforce/suppress signal.';

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
                Reinforcement Learning
              </span>
              <span className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] px-2 py-1 text-[10px] font-black uppercase tracking-wide text-[var(--ds-muted)]">
                DeepSeek-R1 / GRPO
              </span>
            </div>
            <h1 className="text-3xl font-black tracking-tight md:text-5xl">
              GRPO: Learning to Reason from Groups of Answers
            </h1>
            <p className="mt-4 max-w-2xl text-sm leading-7 text-[var(--ds-muted)] md:text-base">
              How reinforcement learning can make a language model spend more computation on hard problems, self-check its work, and prefer better solution traces.
            </p>
          </div>
          <div className="grid min-w-[260px] grid-cols-2 gap-3 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
            <MetricTile label="group size" value={groupSize} detail="siblings sampled per prompt" />
            <MetricTile label="mode" value={activeMode.label} detail={activeMode.detail} />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-5 py-6">
        <section className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
          <SectionHeader icon={SlidersHorizontal} eyebrow="interactive controls" title="Sample, Score, Compare, Update">
            Change the training mode and reward shape, then watch the group baseline decide which solution traces should become more likely.
          </SectionHeader>

          <div className="grid gap-4 lg:grid-cols-[1.2fr_0.8fr]">
            <div className="grid gap-3">
              <div>
                <p className="mb-2 text-xs font-black uppercase tracking-wide text-[var(--ds-muted)]">Mode</p>
                <div className="grid gap-2 md:grid-cols-3">
                  {TRAINING_MODES.map((item) => (
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
                  <p className="mb-2 text-xs font-black uppercase tracking-wide text-[var(--ds-muted)]">Reward type</p>
                  <div className="grid gap-2">
                    {REWARD_TYPES.map((item) => (
                      <ControlButton key={item.id} active={rewardType === item.id} onClick={() => setRewardType(item.id)}>
                        {item.label}
                      </ControlButton>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="mb-2 text-xs font-black uppercase tracking-wide text-[var(--ds-muted)]">Prompt difficulty</p>
                  <div className="grid gap-2">
                    {DIFFICULTIES.map((item) => (
                      <ControlButton key={item.id} active={difficulty === item.id} onClick={() => setDifficulty(item.id)}>
                        {item.label}
                      </ControlButton>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <div className="grid gap-3">
              <SliderControl label="KL strength" value={klBeta} min={0} max={0.3} step={0.01} onChange={setKlBeta} display={fmt(klBeta, 2)} />
              <SliderControl label="clip range" value={clipEps} min={0.05} max={0.4} step={0.01} onChange={setClipEps} display={`+/-${fmt(clipEps, 2)}`} />
              <SliderControl label="temperature" value={temperature} min={0.2} max={1.2} step={0.05} onChange={setTemperature} display={fmt(temperature, 2)} />
            </div>
          </div>
        </section>

        <section className="mt-6 grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <SectionHeader icon={FileText} eyebrow="central animation" title="One Prompt, Many Sampled Answers">
              GRPO turns a single prompt into a tiny competition among sibling completions from the current policy.
            </SectionHeader>
            <div className="mb-4 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
              <p className="text-[10px] font-black uppercase tracking-wide text-[var(--ds-muted)]">Prompt</p>
              <p className="mt-1 font-mono text-sm text-[var(--ds-ink)]">{activePrompt}</p>
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              {group.map((completion) => (
                <CompletionCard key={completion.id} completion={completion} />
              ))}
            </div>
          </div>

          <div className="grid gap-6">
            <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
              <SectionHeader icon={BarChart3} eyebrow="group-relative advantage" title="The Group Becomes the Baseline">
                The reward is useful when it makes better sampled solutions stand out from worse siblings.
              </SectionHeader>
              <div className="grid grid-cols-2 gap-3">
                <MetricTile label="mean reward" value={fmt(stats.meanReward, 3)} detail="group baseline" />
                <MetricTile label="reward std" value={fmt(stats.stdReward, 3)} detail="within-prompt contrast" />
                <MetricTile label="positive samples" value={stats.positive} detail="reinforce traces" />
                <MetricTile label="negative samples" value={stats.negative} detail="suppress traces" />
              </div>
              <div className="mt-4 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
                <p className="font-mono text-sm font-bold">A_i = (r_i - mean(r)) / std(r)</p>
                <p className="mt-2 text-xs leading-5 text-[var(--ds-muted)]">{contrastMessage}</p>
                <div className="mt-4 grid gap-2">
                  {group.map((completion) => (
                    <div key={`${completion.id}-bar`} className="grid grid-cols-[3rem_1fr_4rem] items-center gap-3 text-xs">
                      <span className="font-mono font-bold">{completion.label}</span>
                      <div className="h-3 overflow-hidden rounded border border-[var(--ds-rule)] bg-[var(--ds-paper-2)]">
                        <div
                          className={`h-full ${completion.advantage >= 0 ? 'bg-emerald-600' : 'bg-rose-500'}`}
                          style={{ width: `${clamp(Math.abs(completion.advantage) / 2, 0.05, 1) * 100}%` }}
                        />
                      </div>
                      <span className="text-right font-mono">{signed(completion.advantage, 1)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
              <SectionHeader icon={Gauge} eyebrow="policy guardrails" title="Clipping And KL Keep Updates Bounded" />
              <div className="grid gap-3 md:grid-cols-3">
                <MetricTile label="avg ratio" value={fmt(stats.averageRatio, 2)} detail="new prob / old prob" />
                <MetricTile label="KL penalty" value={fmt(stats.klPenalty, 3)} detail="reference drift cost" />
                <MetricTile label="update strength" value={fmt(stats.updateStrength, 2)} detail="advantage pressure" />
              </div>
              <p className="mt-4 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3 text-xs leading-5 text-[var(--ds-muted)]">
                Positive advantage raises probabilities only within the clipped ratio. Negative advantage suppresses traces, while KL discourages reward hacking drift away from the reference policy.
              </p>
            </div>
          </div>
        </section>

        <section className="mt-6 rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
          <SectionHeader icon={GitBranch} eyebrow="training styles" title="SFT Imitates. Rejection Sampling Filters. GRPO Updates Online.">
            The same prompt can be used in very different training loops.
          </SectionHeader>
          <div className="grid gap-4">
            <FlowRow
              label="SFT"
              active={mode === 'sft'}
              blocks={['prompt', 'gold answer', 'imitate tokens']}
            />
            <FlowRow
              label="Rejection sampling"
              active={mode === 'rejection-sampling'}
              blocks={['prompt', 'many answers', 'filter correct', 'imitate kept data']}
            />
            <FlowRow
              label="PPO"
              active={mode === 'ppo'}
              blocks={['prompt', 'sample', 'reward', 'critic baseline']}
            />
            <FlowRow
              label="GRPO"
              active={mode === 'grpo'}
              blocks={['prompt', 'group sample', 'group mean baseline', 'policy update']}
            />
          </div>
        </section>

        <section className="mt-6 grid gap-6 lg:grid-cols-3">
          {PIPELINE_STAGES.map((stage) => (
            <article key={stage.title} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
              <div className="mb-4 flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-[var(--ds-accent)]" />
                <h2 className="text-lg font-black">{stage.title}</h2>
              </div>
              <div className="grid gap-2">
                {stage.steps.map((step, index) => (
                  <div key={step} className="grid grid-cols-[2rem_1fr] items-center gap-2">
                    <span className="flex h-7 w-7 items-center justify-center rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] font-mono text-xs font-black">
                      {index + 1}
                    </span>
                    <span className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] px-3 py-2 text-sm font-bold">
                      {step}
                    </span>
                  </div>
                ))}
              </div>
              <p className="mt-4 text-sm leading-6 text-[var(--ds-muted)]">{stage.detail}</p>
            </article>
          ))}
        </section>

        <section className="mt-6 grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <SectionHeader icon={RotateCcw} eyebrow="aha moment" title="Reflection Emerges When It Wins">
              The paper reports that R1-Zero learned to allocate more thinking time by reevaluating its initial approach.
            </SectionHeader>
            <div className="space-y-4">
              {[
                ['early RL', 0.25, 0.32, 0.1, 'short answer, jumps to conclusion'],
                ['middle RL', 0.58, 0.62, 0.46, 'longer work, checks answer'],
                ['later RL', 0.9, 0.84, 0.82, 'notices contradiction, rewinds, tries another path'],
              ].map(([label, length, accuracy, selfCheck, detail]) => (
                <div key={label} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
                  <div className="mb-3 flex items-center justify-between gap-3">
                    <h3 className="text-sm font-black uppercase tracking-wide">{label}</h3>
                    <span className="text-xs text-[var(--ds-muted)]">{detail}</span>
                  </div>
                  {[
                    ['reasoning length', length],
                    ['accuracy', accuracy],
                    ['self-check phrases', selfCheck],
                  ].map(([name, value]) => (
                    <div key={name} className="mb-2 grid grid-cols-[8rem_1fr_3rem] items-center gap-3 text-xs">
                      <span className="text-[var(--ds-muted)]">{name}</span>
                      <div className="h-2 overflow-hidden rounded bg-[var(--ds-paper-2)]">
                        <div className="h-full bg-[var(--ds-accent)]" style={{ width: `${value * 100}%` }} />
                      </div>
                      <span className="text-right font-mono">{Math.round(value * 100)}%</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>

          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <SectionHeader icon={ShieldCheck} eyebrow="reward design" title="Cheap, Reliable, Hard To Game">
              Reasoning RL works best when the reward can cheaply separate correct outcomes from persuasive but wrong traces.
            </SectionHeader>
            <div className="grid gap-3 md:grid-cols-2">
              {REWARD_TYPES.map((reward) => (
                <article key={reward.id} className={`rounded border p-4 ${rewardType === reward.id ? 'border-[var(--ds-accent)] bg-[var(--ds-warm)]' : 'border-[var(--ds-rule)] bg-[var(--ds-paper)]'}`}>
                  <h3 className="text-sm font-black">{reward.label}</h3>
                  <p className="mt-2 text-xs leading-5 text-[var(--ds-muted)]">{reward.detail}</p>
                </article>
              ))}
            </div>
            <div className="mt-4 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-4">
              <h3 className="text-sm font-black">When GRPO struggles</h3>
              <div className="mt-3 grid gap-2 text-xs">
                <div className="flex justify-between border-b border-[var(--ds-rule)] pb-2"><span>all answers wrong</span><strong>no reliable winner</strong></div>
                <div className="flex justify-between border-b border-[var(--ds-rule)] pb-2"><span>all answers correct</span><strong>weak contrast</strong></div>
                <div className="flex justify-between"><span>mixed group</span><strong>strong signal</strong></div>
              </div>
            </div>
          </div>
        </section>

        <section className="mt-6 grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
          <div className="rounded border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5">
            <SectionHeader icon={Brain} eyebrow="live readout" title="Current Training Interpretation" />
            <div className="grid gap-3 md:grid-cols-2">
              <MetricTile label="estimated accuracy" value={`${Math.round(stats.estimatedAccuracy * 100)}%`} detail="toy trend from group contrast" />
              <MetricTile label="reasoning length" value={stats.reasoningLength} detail="estimated tokens per trace" />
              <MetricTile label="neutral samples" value={stats.neutral} detail="little direction from advantage" />
              <MetricTile label="reward type" value={REWARD_TYPES.find((item) => item.id === rewardType)?.label} detail={REWARD_TYPES.find((item) => item.id === rewardType)?.detail} />
            </div>
            <p className="mt-4 rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3 text-sm leading-6 text-[var(--ds-muted)]">
              The reward does not need to label every reasoning step. It only has to make better sampled solutions stand out often enough that useful behaviors such as checking, backtracking, and cleaner final answers get reinforced.
            </p>
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
            <SectionHeader icon={Code2} eyebrow="Rustlings-style lab" title="mini-grpo Exercises">
              Local exercises turn the math into small Rust functions: group mean, normalized advantages, clipping, KL, batch filtering, and distillation data.
            </SectionHeader>
            <div className="grid gap-2 md:grid-cols-2">
              {EXERCISES.map(([file, description]) => (
                <div key={file} className="rounded border border-[var(--ds-rule)] bg-[var(--ds-paper)] p-3">
                  <p className="font-mono text-xs font-black text-[var(--ds-ink)]">{file}</p>
                  <p className="mt-1 text-xs leading-5 text-[var(--ds-muted)]">{description}</p>
                </div>
              ))}
            </div>
            <pre className="mt-4 overflow-x-auto rounded border border-[var(--ds-rule)] bg-stone-950 p-3 text-xs leading-5 text-stone-100">cd mini-grpo
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
                GRPO is PPO-like, but the baseline comes from sibling answers instead of a learned critic. DeepSeek-R1-Zero shows the pure RL experiment; DeepSeek-R1 adds cold-start data and a staged pipeline to make the behavior readable and broadly useful.
              </p>
            </div>
          </div>
        </section>

        <div className="mt-6">
          <AssessmentPanel lessonId="grpo-reasoning" title="GRPO reasoning check" />
        </div>
      </main>
    </div>
  );
}
