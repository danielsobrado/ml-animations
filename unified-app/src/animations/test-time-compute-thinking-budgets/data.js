// Data for the "Test-Time Compute and Thinking Budgets" module

export const TTC_SCALING_REGIMES = [
  {
    id: 'training-time',
    label: 'Training-Time Scaling',
    icon: 'database',
    description:
      'Scale performance by increasing pretraining compute: more data, more parameters, more GPU-hours. Effects appear only after retraining from scratch.',
    lever: 'More data or parameters',
    cost: 'Months of pretraining, billions of dollars',
    ceiling: 'Diminishing returns on benchmark saturation; GPT-4 era plateau visible',
    examples: ['GPT-3 175B → GPT-4', 'LLaMA 7B → 70B → 405B'],
    color: 'blue',
  },
  {
    id: 'inference-time',
    label: 'Inference-Time (Test-Time) Scaling',
    icon: 'clock',
    description:
      'Scale performance per query by spending more compute at generation time. No retraining required; the model decides how much to think before answering.',
    lever: 'More tokens of reasoning per query',
    cost: 'Latency + serving cost per query',
    ceiling: 'Overthinking plateau; token budget limits; context window size',
    examples: ['o1 thinking mode', 'Gemini 2.0 Flash Thinking', 'DeepSeek-R1 extended trace'],
    color: 'amber',
  },
];

export const TTC_STRATEGY_FAMILY = [
  {
    id: 'sequential-refinement',
    label: 'Sequential Refinement',
    icon: 'repeat',
    description:
      'Generate an initial answer, then repeatedly critique and revise it. Each round is a new forward pass reading the prior draft.',
    cost_per_answer: 'O(R × L)',
    example: 'Self-critique: "Wait, let me check step 3 again."',
    tradeoff: 'Correct errors but can loop and hallucinate corrections',
  },
  {
    id: 'best-of-n',
    label: 'Best-of-N Sampling',
    icon: 'list',
    description:
      'Sample N independent completions from the policy in parallel, then pick the best one according to a verifier or reward model.',
    cost_per_answer: 'O(N × L)',
    example: 'Generate 32 code solutions, run tests, keep the one passing most test cases.',
    tradeoff: 'Simple but wasteful; oracle verifier required; scales sub-logarithmically',
  },
  {
    id: 'beam-search',
    label: 'Beam Search / Tree Search',
    icon: 'git-branch',
    description:
      'Expand a search tree of partial reasoning paths, pruning low-scoring branches and expanding promising ones according to a value function.',
    cost_per_answer: 'O(B × D × L)',
    example: 'MCTS over reasoning steps, scoring each partial proof with a PRM.',
    tradeoff: 'More sample-efficient than BoN but requires a strong per-step value model',
  },
  {
    id: 'adaptive-budget',
    label: 'Adaptive Thinking Budgets',
    icon: 'sliders',
    description:
      'Allocate thinking tokens dynamically based on predicted problem difficulty. Easy queries get short budgets; hard queries spend more.',
    cost_per_answer: 'O(difficulty × L)',
    example: 'Budget-forcing: if model is not done thinking at 512 tokens, allow up to 4 096.',
    tradeoff: 'Needs a difficulty classifier; budget overflow can truncate reasoning mid-chain',
  },
  {
    id: 'tool-use',
    label: 'Tool-Augmented Compute',
    icon: 'wrench',
    description:
      'Offload sub-tasks to deterministic tools (calculators, code interpreters, search engines) rather than reasoning through them in tokens.',
    cost_per_answer: 'O(tokens_until_tool_call + tool_latency)',
    example: 'ReAct: Thought → Action(python_exec) → Observation → Continue reasoning.',
    tradeoff: 'Dramatically improves precision; latency dominated by round-trip tool calls',
  },
];

export const BON_STRATEGY_DATA = [
  { n: 1, expectedAcc: 52, oracleBound: 52 },
  { n: 2, expectedAcc: 60, oracleBound: 68 },
  { n: 4, expectedAcc: 66, oracleBound: 76 },
  { n: 8, expectedAcc: 71, oracleBound: 84 },
  { n: 16, expectedAcc: 73, oracleBound: 90 },
  { n: 32, expectedAcc: 75, oracleBound: 93 },
  { n: 64, expectedAcc: 76, oracleBound: 96 },
  { n: 128, expectedAcc: 76.5, oracleBound: 98 },
];

export const BEAM_SEARCH_STEPS = [
  {
    depth: 0,
    nodes: [{ id: 'root', text: 'Is 1729 a sum of two cubes?', score: null }],
  },
  {
    depth: 1,
    nodes: [
      { id: 'a', text: 'Try 1³=1, 2³=8, 3³=27...', score: 0.6, pruned: false },
      { id: 'b', text: 'Check if it is Ramanujan number', score: 0.85, pruned: false },
      { id: 'c', text: '1729 = 2×864... wrong path', score: 0.2, pruned: true },
    ],
  },
  {
    depth: 2,
    nodes: [
      { id: 'a1', parent: 'a', text: '10³=1000, 9³=729, 1000+729=1729 ✓', score: 0.95, pruned: false },
      { id: 'b1', parent: 'b', text: '1729=1³+12³? 1+1728=1729 ✓', score: 0.98, pruned: false },
      { id: 'c1', parent: 'c', text: '(pruned branch)', score: 0, pruned: true },
    ],
  },
  {
    depth: 3,
    nodes: [
      { id: 'a2', parent: 'a1', text: 'Also 1³+12³=1729 ✓ — Both representations found!', score: 0.99, pruned: false },
      { id: 'b2', parent: 'b1', text: 'Final: Yes — 1³+12³ = 10³+9³ = 1729', score: 1.0, pruned: false },
    ],
  },
];

export const THINKING_BUDGET_EXAMPLES = [
  {
    id: 'trivial',
    label: 'Trivial Query',
    query: 'What is 15% of 200?',
    difficulty: 'easy',
    optimalTokens: 40,
    trace: '<think>\n15% of 200 = 0.15 × 200 = 30\n</think>\n\n30.',
    wastedTokens: 0,
    note: 'Short chain-of-thought is optimal here. Allocating 4 000 tokens would be pure overthinking waste.',
  },
  {
    id: 'moderate',
    label: 'Moderate Reasoning',
    query: 'Prove that √2 is irrational.',
    difficulty: 'medium',
    optimalTokens: 280,
    trace: '<think>\nAssume √2 = p/q in lowest terms (gcd(p,q)=1).\nThen 2 = p²/q², so p² = 2q².\nSince 2 | p², we know 2 | p. Write p = 2k.\nThen (2k)² = 2q² → 4k² = 2q² → q² = 2k².\nSo 2 | q², meaning 2 | q.\nBut then gcd(p,q) ≥ 2 — contradicts our assumption. ∎\n</think>\n\n√2 is irrational (shown by contradiction).',
    wastedTokens: 0,
    note: 'Medium budget fits. More tokens would not improve the proof; fewer would skip essential steps.',
  },
  {
    id: 'complex',
    label: 'Hard Competition Math',
    query: 'Find all positive integer solutions to x² + y² = z² + 1 with x, y, z ≤ 20.',
    difficulty: 'hard',
    optimalTokens: 1200,
    trace: '<think>\nSystematically enumerate...\nFor x=1: y²=z²+1-1=z², so y=z. All (1,z,z) for z≥1.\nFor x=2: 4+y²=z²+1 → z²-y²=3 → (z-y)(z+y)=3.\nSince z,y>0: z-y=1, z+y=3 → z=2, y=1. → (2,1,2).\nContinue for x=3..20... [extended search]\n</think>\n\n{complete solution set with all pairs}',
    wastedTokens: 0,
    note: 'Hard problems need large budgets. Truncating at 200 tokens would leave the search incomplete.',
  },
  {
    id: 'overthought',
    label: 'Overthinking (Wasteful)',
    query: 'What is 2 + 2?',
    difficulty: 'trivial',
    optimalTokens: 5,
    trace: "<think>\nLet me think about this carefully. 2 plus 2.\nIf I have 2 apples and someone gives me 2 more...\nThat's 4 apples. But let me verify.\nUsing number theory: 2 = S(S(0)), 2+2 = S(S(S(S(0)))) = 4.\nPeano axioms confirm: 4 is the successor of 3, which is the successor of 2...\nAlternatively in binary: 10 + 10 = 100 = 4 in decimal.\nChecking: 4 = 2×2 = 2+2 ✓\nI'm confident the answer is 4.\n</think>\n4.",
    wastedTokens: 450,
    note: 'Classic overthinking. No length penalty led to 450 wasted tokens. A budget cap or difficulty gate would prevent this.',
  },
];

export const BUDGET_FORCING_DATA = [
  { budget: 64, accuracy: 44, latencyMs: 200 },
  { budget: 128, accuracy: 52, latencyMs: 380 },
  { budget: 256, accuracy: 61, latencyMs: 720 },
  { budget: 512, accuracy: 68, latencyMs: 1400 },
  { budget: 1024, accuracy: 74, latencyMs: 2800 },
  { budget: 2048, accuracy: 78, latencyMs: 5500 },
  { budget: 4096, accuracy: 80, latencyMs: 11000 },
  { budget: 8192, accuracy: 80.5, latencyMs: 22000 },
];

export const TTC_FAILURE_MODES = [
  {
    id: 'overthinking',
    name: 'Overthinking Plateau',
    description: 'Model allocates huge budgets to trivial problems and loops without converging.',
    signal: 'Accuracy flat; latency explodes; token count >> complexity of query.',
    fix: 'Adaptive budget gate based on query difficulty classifier; length penalty in training.',
    severity: 'high',
  },
  {
    id: 'truncation',
    name: 'Hard Budget Truncation',
    description: 'Thinking trace cut off mid-reasoning by a fixed context limit.',
    signal: 'Final answer appears before reasoning chain is complete; accuracy drops on hard problems.',
    fix: 'Progressive budget extension; detect incomplete reasoning before finalizing answer.',
    severity: 'medium',
  },
  {
    id: 'verifier-gap',
    name: 'Verifier-Reasoning Gap',
    description: 'Model learns to produce reasoning traces that satisfy the verifier without actually solving the problem.',
    signal: 'High reward but low real-world accuracy; traces contain convincing-looking but incorrect steps.',
    fix: 'Use process reward models (PRMs) and diverse problem sets to prevent pattern matching.',
    severity: 'high',
  },
  {
    id: 'bon-oracle',
    name: 'Best-of-N Oracle Ceiling',
    description: 'Performance asymptotes because the verifier cannot identify the best sample without ground truth.',
    signal: 'Reward model ranking diverges from human preference; winner selection is random at the tail.',
    fix: 'Train stronger verifiers; use ensemble of reward models; switch to PRM-guided beam search.',
    severity: 'medium',
  },
  {
    id: 'latency-cost',
    name: 'Latency-Cost Explosion',
    description: 'Serving high-thinking-budget queries becomes economically infeasible at scale.',
    signal: 'P99 latency SLA violation; inference cost-per-query 100× baseline.',
    fix: 'Speculative decoding; caching of reasoning fragments; tier routing (fast/slow paths).',
    severity: 'high',
  },
];

export const REACT_AGENT_STEPS = [
  {
    id: 'thought-1',
    type: 'thought',
    content: 'I need to find the current population of Tokyo. I should look this up rather than guess.',
  },
  {
    id: 'action-1',
    type: 'action',
    content: 'search("Tokyo population 2024")',
    toolName: 'web_search',
  },
  {
    id: 'observation-1',
    type: 'observation',
    content: 'Result: Tokyo metropolitan area population is approximately 37.4 million as of 2024.',
  },
  {
    id: 'thought-2',
    type: 'thought',
    content: 'Good. Now I need to compare this to the world total to compute the percentage.',
  },
  {
    id: 'action-2',
    type: 'action',
    content: 'search("world population 2024")',
    toolName: 'web_search',
  },
  {
    id: 'observation-2',
    type: 'observation',
    content: 'Result: World population reached approximately 8.1 billion in 2024.',
  },
  {
    id: 'thought-3',
    type: 'thought',
    content: 'Tokyo / World = 37.4M / 8100M ≈ 0.462%. Let me compute precisely.',
  },
  {
    id: 'action-3',
    type: 'action',
    content: 'python_exec("37.4 / 8100 * 100")',
    toolName: 'code_interpreter',
  },
  {
    id: 'observation-3',
    type: 'observation',
    content: 'Result: 0.4617...',
  },
  {
    id: 'answer',
    type: 'answer',
    content: 'Tokyo\'s metropolitan area contains approximately 37.4 million people, about 0.46% of the world\'s 8.1 billion population.',
  },
];

export const SCALING_LAWS_DATA = {
  trainTimePoints: [
    { flops: 1e21, accuracy: 35 },
    { flops: 1e22, accuracy: 45 },
    { flops: 1e23, accuracy: 56 },
    { flops: 1e24, accuracy: 65 },
    { flops: 1e25, accuracy: 72 },
  ],
  testTimePoints: [
    { tokens: 64, accuracy: 52 },
    { tokens: 256, accuracy: 61 },
    { tokens: 1024, accuracy: 69 },
    { tokens: 4096, accuracy: 76 },
    { tokens: 16384, accuracy: 80 },
  ],
  crossoverNote:
    'Beyond a certain training-compute budget, additional thinking tokens become more cost-effective than training on more data.',
};

export const THINKING_BUDGET_STRATEGIES = [
  {
    id: 'fixed',
    label: 'Fixed Cap',
    description: 'Hard-coded max token limit applied to all queries.',
    pros: ['Predictable latency', 'Easy to implement'],
    cons: ['Wastes tokens on easy queries', 'Truncates hard ones'],
    bestFor: 'Latency-sensitive applications with homogeneous query types',
  },
  {
    id: 'adaptive',
    label: 'Adaptive Gate',
    description: 'Difficulty classifier routes queries to short/medium/long thinking budgets.',
    pros: ['Efficient on easy queries', 'Allocates budget where needed'],
    cons: ['Classifier can be wrong', 'Adds routing latency'],
    bestFor: 'Mixed-difficulty workloads (customer support, coding assistants)',
  },
  {
    id: 'model-controlled',
    label: 'Model-Controlled',
    description: 'The model learns when to stop thinking via RL length penalties.',
    pros: ['No external classifier', 'Emergent efficiency'],
    cons: ['Hard to guarantee latency SLAs', 'May still overthink'],
    bestFor: 'General-purpose reasoning agents trained with RLVR',
  },
  {
    id: 'speculative',
    label: 'Speculative / Parallel Drafting',
    description: 'Draft multiple short completions in parallel; promote best; skip long serial chain.',
    pros: ['Lower wall-clock latency', 'Good for constrained environments'],
    cons: ['High GPU memory for parallel drafts', 'Verifier required'],
    bestFor: 'Real-time interactive reasoning with strict SLAs',
  },
];

export const METRICS_COMPARISON = [
  {
    model: 'Base LLM (greedy)',
    ttcStrategy: 'None',
    accuracy: 52,
    avgTokens: 18,
    costPerQuery: '$0.0004',
    latencyMs: 180,
  },
  {
    model: 'Base LLM + BoN-16',
    ttcStrategy: 'Best-of-N',
    accuracy: 73,
    avgTokens: 288,
    costPerQuery: '$0.0058',
    latencyMs: 320,
  },
  {
    model: 'Reasoning Model (short)',
    ttcStrategy: 'Chain-of-Thought (512 tok)',
    accuracy: 74,
    avgTokens: 512,
    costPerQuery: '$0.010',
    latencyMs: 900,
  },
  {
    model: 'Reasoning Model (long)',
    ttcStrategy: 'Extended Thinking (4k tok)',
    accuracy: 80,
    avgTokens: 3800,
    costPerQuery: '$0.076',
    latencyMs: 7200,
  },
  {
    model: 'Reasoning Model + PRM Beam',
    ttcStrategy: 'Beam Search (beam=8)',
    accuracy: 83,
    avgTokens: 6400,
    costPerQuery: '$0.128',
    latencyMs: 12000,
  },
];
