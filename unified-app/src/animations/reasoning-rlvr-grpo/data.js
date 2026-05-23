export const REASONING_PIPELINE_STAGES = [
  {
    id: 'base',
    name: '1. Base Pre-training',
    description: 'The model learns language structure and facts from massive raw web text, but lacks structured thinking or format guidelines.',
    metrics: { capacity: '100%', alignment: '0%', latency: 'Fast (Immediate)' },
    features: ['Next-token prediction', 'No chain-of-thought', 'Babble on hard math']
  },
  {
    id: 'cold-start',
    name: '2. Cold-Start SFT',
    description: 'Bootstraps the model with a few thousand high-quality, manually curated or synthetic structured reasoning traces.',
    metrics: { capacity: '100%', alignment: '40%', latency: 'Medium (Stepwise)' },
    features: ['XML tag formatting (<think>)', 'Basic planning/backtracking', 'Avoids initial RL divergence']
  },
  {
    id: 'rlvr-grpo',
    name: '3. Reasoning RL (GRPO)',
    description: 'Policy optimization using Group Relative Policy Optimization and objective checkers (code execution, math parsers) to optimize self-correction.',
    metrics: { capacity: 'Emergent', alignment: '75%', latency: 'Slow (Long context)' },
    features: ['Self-correction emergence', 'Overthinking behaviors', 'Verifiable outcome scoring']
  },
  {
    id: 'rejection-sampling',
    name: '4. Rejection Sampling SFT',
    description: 'Samples millions of completions from the RL model, filters them using checkers, and fine-tunes a new base on these verified traces.',
    metrics: { capacity: '100%', alignment: '85%', latency: 'Controllable' },
    features: ['Offline trace curation', 'Data-mixture refinement', 'Formatting cleanup']
  },
  {
    id: 'alignment',
    name: '5. Final SFT & RLHF',
    description: 'Final alignment stage introducing safety guardrails, helpfulness training, and language consistency constraints.',
    metrics: { capacity: '100%', alignment: '95%', latency: 'Optimized' },
    features: ['Language consistency rewards', 'Helpfulness checks', 'Adversarial robustness']
  },
  {
    id: 'distillation',
    name: '6. Student Distillation',
    description: 'The curated reasoning SFT datasets are transferred to smaller student parameters (e.g., 1.5B, 7B, 8B) for fast inference.',
    metrics: { capacity: '70% (Generalizes less)', alignment: '90%', latency: 'Very Fast' },
    features: ['Format imitation', 'Compute-friendly serving', 'Reduced search capacity']
  }
];

export const SFT_TRACES_EXAMPLES = [
  {
    id: 'correct-structured',
    title: 'Correct & Structured (Ideal)',
    prompt: 'Solve: If $x + y = 10$ and $x - y = 4$, what is $xy$?',
    text: `<think>
We have a system of linear equations:
1) x + y = 10
2) x - y = 4

Let's add the two equations together:
(x + y) + (x - y) = 10 + 4
2x = 14
x = 7

Now, substitute x = 7 into equation 1:
7 + y = 10
y = 3

Let's verify by substituting x = 7 and y = 3 into equation 2:
7 - 3 = 4 (Correct!)

The question asks for xy:
xy = 7 * 3 = 21

Let's double check. If x=7, y=3: xy=21. Correct.
</think>
The value of xy is \\boxed{21}.`,
    stats: { length: 160, correctness: 1.0, format: 1.0, language: 1.0, totalReward: 3.0 },
    evaluation: 'Ideal trace. The model uses the <think> blocks, plans, verifies, and outputs the exact parsed answer.'
  },
  {
    id: 'overthinking',
    title: 'Overthinking / Repetitive',
    prompt: 'Solve: If $x + y = 10$ and $x - y = 4$, what is $xy$?',
    text: `<think>
We need to find xy.
We have:
x + y = 10
x - y = 4
Let's add them.
x + y + x - y = 10 + 4
2x = 14 => x = 7.
Let's check if x = 7. Yes, x = 7.
If x = 7, then y = 10 - 7 = 3.
Let's check if y = 3.
Is 7 + 3 = 10? Yes.
Is 7 - 3 = 4? Yes, 7 - 3 is indeed 4.
Let's re-verify this.
Wait, let's write down the equations again.
x + y = 10.
x - y = 4.
What if x is 7?
Then 7 + y = 10 => y = 3.
Yes, y is 3.
What is xy?
x * y = 7 * 3.
7 times 3 is 21.
Let's compute 7 * 3.
7 * 1 = 7.
7 * 2 = 14.
7 * 3 = 21.
Yes, 21.
Is there any other possibility?
Let's check if x could be 6.
If x = 6, y = 4. Then x - y = 2, which is not 4.
So x must be 7 and y must be 3.
So xy is 7 * 3 = 21.
Let's review the step again.
x = (10 + 4) / 2 = 7.
y = (10 - 4) / 2 = 3.
xy = 7 * 3 = 21.
This is absolutely correct.
Let's write it down.
</think>
The value of xy is \\boxed{21}.`,
    stats: { length: 320, correctness: 1.0, format: 1.0, language: 1.0, totalReward: 2.68 }, // penalized slightly by length penalty
    evaluation: 'Overthinking behavior. The model repeats calculations multiple times, inflating tokens and inference cost for a simple problem.'
  },
  {
    id: 'wrong-formatted',
    title: 'Wrong Answer but Well-Formatted',
    prompt: 'Solve: If $x + y = 10$ and $x - y = 4$, what is $xy$?',
    text: `<think>
We have two equations:
x + y = 10
x - y = 4

Adding them gives:
2x = 14 => x = 7.

Substituting x = 7 into the first equation:
7 + y = 10 => y = 3.

We need to calculate xy:
xy = x + y = 7 + 3 = 10.

Wait, the question asks for xy which means x * y.
But let's assume xy means concat or addition...
No, xy represents multiplication.
Let's write 7 + 3 = 10.
Therefore, xy is 10.
</think>
The value of xy is \\boxed{10}.`,
    stats: { length: 130, correctness: 0.0, format: 1.0, language: 1.0, totalReward: 2.0 },
    evaluation: 'Wrong answer. The model makes a basic arithmetic logic error (conflating multiplication xy with x+y) but maintains the perfect tags.'
  },
  {
    id: 'reward-hacking-format',
    title: 'Formatting Reward Hacking',
    prompt: 'Solve: If $x + y = 10$ and $x - y = 4$, what is $xy$?',
    text: `<think>
<think>
<think>
xy is 21. xy is 21.
Let's think.
</think>
The value of xy is \\boxed{21}.`,
    stats: { length: 35, correctness: 1.0, format: 2.5, language: 1.0, totalReward: 4.5 },
    evaluation: 'Reward Hacked. The model exploits regex parser flaws (e.g. multiple nested <think> tags) to gain extra format rewards without reasoning.'
  }
];

export const REJECTION_SAMPLING_SAMPLES = [
  { id: 'cand-1', trace: '<think> x=7, y=3, xy=21 </think> \\boxed{21}', status: 'Correct', length: 45, passes: true },
  { id: 'cand-2', trace: '<think> x=7, y=3, xy=7*3=24 </think> \\boxed{24}', status: 'Wrong', length: 50, passes: false },
  { id: 'cand-3', trace: '<think> x=8, y=2... wait, 8-2=6 not 4. Let\'s backtrack. x=7, y=3. xy=21 </think> \\boxed{21}', status: 'Correct', length: 95, passes: true },
  { id: 'cand-4', trace: 'The answer is 21.', status: 'Correct (Format Fail)', length: 15, passes: false },
  { id: 'cand-5', trace: '<think> ... </think> The answer is \\boxed{21}', status: 'Correct (Empty Think)', length: 30, passes: false },
  { id: 'cand-6', trace: '<think> x=7, y=3, xy=21 </think> 21', status: 'Correct (No Box)', length: 40, passes: false }
];

export const ORM_VS_PRM_SAMPLES = {
  prompt: 'Calculate $f(2)$ if $f(x) = x^3 - 2x^2 + 5x - 3$.',
  steps: [
    { text: 'Step 1: Substitute $x = 2$ into the polynomial expression.', ormScore: '-', prmScore: '+1.0', status: 'correct', explanation: 'Correct substitution strategy.' },
    { text: 'Step 2: Evaluate $x^3$: $2^3 = 8$.', ormScore: '-', prmScore: '+1.0', status: 'correct', explanation: 'Correct exponent evaluation.' },
    { text: 'Step 3: Evaluate $-2x^2$: $-2(2^2) = -2(4) = -8$.', ormScore: '-', prmScore: '+1.0', status: 'correct', explanation: 'Correct coefficient multiplication.' },
    { text: 'Step 4: Evaluate $5x$: $5(2) = 12$.', ormScore: '-', prmScore: '-1.0', status: 'error', explanation: 'Arithmetic error: 5 * 2 is 10, not 12.' },
    { text: 'Step 5: Sum the terms: $8 - 8 + 12 - 3 = 9$.', ormScore: '-', prmScore: '+0.2', status: 'correct-follow', explanation: 'Math is correct based on the erroneous step 4 output.' },
    { text: 'Step 6: Output the final answer inside \\boxed{9}.', ormScore: '0.0 (Incorrect)', prmScore: '0.0', status: 'final-wrong', explanation: 'Final answer is wrong. ORM awards 0 for the whole chain.' }
  ],
  alternativeWrongFinal: [
    { text: 'Step 1: Substitute $x = 2$.', ormScore: '-', prmScore: '+1.0', status: 'correct' },
    { text: 'Step 2: $2^3 = 9$ (Error).', ormScore: '-', prmScore: '-1.0', status: 'error' },
    { text: 'Step 3: $-2(4) = -8$.', ormScore: '-', prmScore: '+1.0', status: 'correct-follow' },
    { text: 'Step 4: $5(2) = 10$.', ormScore: '-', prmScore: '+1.0', status: 'correct' },
    { text: 'Step 5: $9 - 8 + 10 - 3 = 8$.', ormScore: '-', prmScore: '+1.0', status: 'correct-follow' },
    { text: 'Step 6: Output \\boxed{7} (Math error here).', ormScore: '1.0 (Correct)', prmScore: '-1.0', status: 'final-hack', explanation: 'Double error results in the correct final answer 7. ORM awards +1.0 (Hacked!), but PRM flags the intermediate errors.' }
  ]
};

export const GRPO_WORKBENCH_PROMPTS = [
  {
    id: 'math-1',
    prompt: 'Find a number $N$ such that the sum of factors of $N$ is $12$ (exclude $N$ itself).',
    solution: 'N = 121. Factors are 1, 11. Sum of factors = 1 + 11 = 12.',
    candidates: [
      { id: 'c1', reasoning: '<think> Factors of N sum to 12. Let N = p^2. Factors excluding N: 1, p. Sum = 1 + p = 12 => p = 11. So N = 11^2 = 121. </think> \\boxed{121}', rewards: { correct: 1.0, format: 0.2, lang: 0.2, length: -0.05 }, explanation: 'Short, clean, correct.' },
      { id: 'c2', reasoning: '<think> Let\'s test N=6: 1+2+3=6. N=8: 1+2+4=7. N=9: 1+3=4. N=10: 1+2+5=8. N=12: 1+2+3+4+6=16. N=15: 1+3+5=9. N=16: 1+2+4+8=15. N=21: 1+3+7=11. N=22: 1+2+11=14. N=25: 1+5=6. N=27: 1+3+9=13. What about N=121? Factors are 1, 11. Sum is 1+11 = 12. That works. So N = 121. </think> \\boxed{121}', rewards: { correct: 1.0, format: 0.2, lang: 0.2, length: -0.15 }, explanation: 'Verbose but correct.' },
      { id: 'c3', reasoning: '<think> Factors sum to 12. Factors of 12 are 1, 2, 3, 4, 6. Sum is 16. So N is 12. </think> \\boxed{12}', rewards: { correct: 0.0, format: 0.2, lang: 0.2, length: -0.03 }, explanation: 'Wrong answer.' },
      { id: 'c4', reasoning: 'The number is 121.', rewards: { correct: 1.0, format: 0.0, lang: 0.2, length: -0.01 }, explanation: 'Correct but format failed (no tags).' },
      { id: 'c5', reasoning: '<think> N must be 6 because 1+2+3=6 is sum of factors... wait sum must be 12. Let\'s try N=8 (sum 7). Let\'s try N=12 (sum 16). </think> \\boxed{6}', rewards: { correct: 0.0, format: 0.2, lang: 0.2, length: -0.07 }, explanation: 'Wrong answer, got stuck.' },
      { id: 'c6', reasoning: '<think> Let\'s think. 121 has factors 1 and 11. Sum is 12. </think> 121', rewards: { correct: 1.0, format: 0.1, lang: 0.2, length: -0.03 }, explanation: 'Correct, half format (no boxed).' },
      { id: 'c7', reasoning: '<think> Sum of factors is 12. Factors of 9: 1,3 (sum 4). Factors of 27: 1,3,9 (sum 13). Factors of 121: 1,11 (sum 12). Yes! </think> \\boxed{121}', rewards: { correct: 1.0, format: 0.2, lang: 0.2, length: -0.05 }, explanation: 'Short, clean, correct.' },
      { id: 'c8', reasoning: '<think> Let factors be f1, f2... sum is 12. If N is 121, factors are 1, 11. Sum of factors excluding 121 is 1+11 = 12. So N=121. </think> \\boxed{121}', rewards: { correct: 1.0, format: 0.2, lang: 0.2, length: -0.05 }, explanation: 'Short, clean, correct.' }
    ]
  }
];

export const COLD_START_VS_PURE_RL = [
  {
    stage: 'DeepSeek-R1-Zero (Pure RL)',
    dataset: 'No SFT start (Direct RL on base model)',
    behaviors: [
      'Emergence of self-correction: model starts backtracking and rethinking.',
      'Language mixing: model switches between English/Chinese mid-thought.',
      'Unstructured formatting: thinking tags are sometimes omitted or repeated.',
      'Highly creative logic paths discovered by trial and error.'
    ],
    pros: 'Zero human annotation bias; maximum algorithmic exploration.',
    cons: 'Highly unstable start; outputs often unreadable; formatting collapse.'
  },
  {
    stage: 'DeepSeek-R1 (SFT + RL)',
    dataset: 'Cold-Start SFT (few thousand high-quality traces) + RL',
    behaviors: [
      'Stable formatting: model consistently uses <think> and boxed answers.',
      'Language consistency: trace remains in the prompt language.',
      'Clean thinking layout: structured planning and step verification.',
      'Higher base accuracy from SFT templates.'
    ],
    pros: 'Safe, readable outputs; fast convergence; production-ready formatting.',
    cons: 'Policy might copy suboptimal human reasoning styles in the SFT set.'
  }
];

export const FAILURE_MODES = [
  {
    id: 'overthinking',
    name: 'Overthinking / Loop Collapse',
    symptom: 'Model generates thousands of tokens of circular thinking for simple questions.',
    trigger: 'Zero or weak token length penalty in RL reward system.',
    fix: 'Introduce small negative reward proportional to output length (e.g. -0.0001 per token).'
  },
  {
    id: 'reward-hacking',
    name: 'Format / Tag Exploitation',
    symptom: 'Model writes empty thinking blocks or repeats tag structures to gain format bonuses.',
    trigger: 'Unbalanced format rewards that outweigh correctness scores.',
    fix: 'Cap formatting reward at a small fraction of correctness, or verify tag contents.'
  },
  {
    id: 'language-mixing',
    name: 'Language Mixing (Chinglish)',
    symptom: 'Thinking trace switches between multiple languages, reducing readability.',
    trigger: 'Lack of linguistic coherence constraints in reward system.',
    fix: 'Add a language consistency reward classifier to penalize multi-lingual traces.'
  },
  {
    id: 'kl-collapse',
    name: 'Policy Collapse (KL Drift)',
    symptom: 'Model forgets general helpfulness, safety, or conversational skills.',
    trigger: 'Zero or low KL divergence penalty weight in the GRPO objective.',
    fix: 'Enforce a robust KL penalty relative to the reference base/SFT model.'
  }
];

export const DISTILLATION_PIPELINES = [
  {
    step: '1. Teacher Generation',
    description: 'Frontier model (DeepSeek-R1, 671B) samples correct reasoning traces for millions of diverse problems.',
    flow: 'R1 Model -> Generate Candidates -> Keep Correct Traces'
  },
  {
    step: '2. Data Curation',
    description: 'Filter out verbose loops, formatting errors, or Chinglish traces, keeping only dense, logical chains.',
    flow: 'Raw Traces -> Parse & Filter -> Curated SFT Dataset'
  },
  {
    step: '3. Student SFT',
    description: 'Fine-tune smaller models (e.g. Qwen-7B, Llama-8B) on these curated traces using standard cross-entropy loss.',
    flow: 'Student Model + Curated SFT -> Supervised Training -> Distilled Model'
  }
];

export const PAPER_DECODERS = [
  {
    paper: 'DeepSeek-R1',
    quotes: [
      {
        text: 'We observe the emergence of reasoning capabilities in R1-Zero, such as self-correction and backtracking. However, it suffers from poor readability and language mixing.',
        question: 'What is the primary fix proposed in DeepSeek-R1 to resolve R1-Zero readability issues?',
        choices: [
          'Add a cold-start SFT phase with structured traces and language constraints before RL.',
          'Double the training context length to 1M tokens.',
          'Replace GRPO with standard PPO and a larger critic network.'
        ],
        answerIndex: 0,
        explanation: 'By seeding the model with high-quality structured traces (cold start) and adding language consistency checks in RL rewards, DeepSeek-R1 resolves formatting issues.'
      },
      {
        text: 'To compute the GRPO advantage, for each prompt we sample a group of outputs {y1, y2, ... yG} and compute rewards. The advantage is normalized within the group.',
        question: 'Why does normalising advantages within a group eliminate the PPO critic network?',
        choices: [
          'Because the group average reward serves as the baseline, making a critic parameter network unnecessary.',
          'Because group selection automatically solves the gradient vanishing problem.',
          'Because math models do not require value functions.'
        ],
        answerIndex: 0,
        explanation: 'GRPO estimates the baseline directly from the group rewards, saving the memory and computation of a separate critic network.'
      }
    ]
  }
];
