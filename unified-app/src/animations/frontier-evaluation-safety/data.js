export const EVAL_LAYERS = {
  capability: {
    label: 'Capability',
    question: 'Can the model do the task?',
    examples: ['SWE-bench', 'GPQA', 'AIME', 'MRCR', 'Graphwalks'],
    risk: 'May not reflect workflow, tool, or safety behavior.',
  },
  product: {
    label: 'Product',
    question: 'Does the deployed workflow succeed reliably?',
    examples: ['support simulations', 'coding workflows', 'document review'],
    risk: 'Can miss adversarial or rare safety failures.',
  },
  agentSafety: {
    label: 'Agent Safety',
    question: 'Does the agent act safely with tools and autonomy?',
    examples: ['prompt injection', 'permission gates', 'tool audits'],
    risk: 'Guardrails can over-block or miss shifted attacks.',
  },
  frontierRisk: {
    label: 'Frontier Risk',
    question: 'Does the system cross dangerous capability thresholds?',
    examples: ['cyber', 'bio/chem', 'autonomy', 'scheming'],
    risk: 'Hard to validate with static tests alone.',
  },
};

export const FRONTIER_FAILURES = [
  {
    id: 'prompt-injection',
    label: 'Prompt injection',
    symptom: 'External content attempts to override system or developer instructions.',
    mitigation: 'Instruction hierarchy, source trust labels, prompt guards, and permission gates.',
  },
  {
    id: 'unsafe-tool-action',
    label: 'Unsafe tool action',
    symptom: 'Agent tries to send, delete, export, buy, deploy, or mutate state without authorization.',
    mitigation: 'Approval gates, sandboxing, action risk classification, audit logs, and rollback.',
  },
  {
    id: 'reward-hacking',
    label: 'Reward hacking',
    symptom: 'Agent optimizes the measured score while violating the intended goal.',
    mitigation: 'Multi-objective rewards, process checks, hidden tests, and human review.',
  },
  {
    id: 'stale-retrieval',
    label: 'Stale retrieval',
    symptom: 'Agent relies on outdated evidence or old policy text.',
    mitigation: 'Freshness checks, source dates, retrieval evaluation, and citation audits.',
  },
  {
    id: 'scheming-risk',
    label: 'Scheming risk',
    symptom: 'Agent may hide goal-conflicting behavior under evaluation pressure.',
    mitigation: 'Oversight randomization, action logs, stealth evals, and autonomy limits.',
  },
  {
    id: 'over-refusal',
    label: 'Over-refusal',
    symptom: 'System blocks safe requests because filtering is too broad.',
    mitigation: 'Deliberative policy reasoning and precision/recall safety evals.',
  },
];

export const SAFETY_TABS = [
  { id: 'map', label: 'Evaluation Map' },
  { id: 'capability-product', label: 'Capability vs Product' },
  { id: 'agent-benchmarks', label: 'Agent Benchmarks' },
  { id: 'long-context', label: 'Long-Context Evals' },
  { id: 'red-team', label: 'Red Teaming' },
  { id: 'prompt-injection', label: 'Prompt Injection + Tool Safety' },
  { id: 'autonomy', label: 'Autonomy Boundaries' },
  { id: 'cot-monitoring', label: 'CoT Monitoring Limits' },
  { id: 'reward-hacking', label: 'Reward Hacking' },
  { id: 'scheming', label: 'Scheming Risk' },
  { id: 'deliberative', label: 'Deliberative Alignment' },
  { id: 'preparedness', label: 'Preparedness Gates' },
  { id: 'papers', label: 'Paper/Product Decoder' },
];

export const BENCHMARKS = [
  {
    id: 'swe-bench',
    label: 'SWE-bench-style',
    format: 'Repo + issue -> patch -> tests',
    scoring: 'FAIL_TO_PASS fixes target behavior; PASS_TO_PASS protects regressions.',
    failure: 'Patch passes target test but breaks unrelated code.',
  },
  {
    id: 'osworld',
    label: 'OSWorld-style',
    format: 'User task -> GUI actions -> environment state',
    scoring: 'Execution-based validation in desktop and web environments.',
    failure: 'Agent misclicks or acts on the wrong visual element.',
  },
  {
    id: 'tau-bench',
    label: 'TAU-bench-style',
    format: 'User conversation -> policy -> API tools -> database state',
    scoring: 'Final state and pass^k reliability across repeated trials.',
    failure: 'Agent satisfies the user while violating domain policy.',
  },
];

export const PAPER_CARDS = [
  {
    title: 'OpenAI o3/o4-mini System Card',
    signals: ['reasoning models', 'tool use in chain-of-thought', 'deliberative alignment', 'Preparedness Framework'],
    interpretation: 'Frontier safety evals must cover reasoning, tools, policy reasoning, and risk thresholds.',
  },
  {
    title: 'Anthropic Claude 4',
    signals: ['extended thinking with tools', 'parallel tool execution', 'local-file memory', 'code execution and MCP connectors'],
    interpretation: 'Safety evaluation must treat model, tools, memory, and agent scaffold as one system.',
  },
  {
    title: 'Meta Llama 4 Safeguards',
    signals: ['pre/post-training mitigations', 'Llama Guard', 'Prompt Guard', 'CyberSecEval', 'GOAT red-teaming'],
    interpretation: 'Safety is layered across model training, classifiers, cybersecurity evals, and adversarial probing.',
  },
  {
    title: 'SWE-bench Verified',
    signals: ['real GitHub issues', 'FAIL_TO_PASS', 'PASS_TO_PASS', 'human-screened samples'],
    interpretation: 'Coding-agent evals need both bug-fix success and regression safety.',
  },
  {
    title: 'OSWorld',
    signals: ['real computer environment', 'multimodal GUI tasks', 'execution-based validation'],
    interpretation: 'Computer-use agents need environment-level evaluation, not only text-response scoring.',
  },
  {
    title: 'TAU-bench',
    signals: ['tool-agent-user interaction', 'domain policies', 'final database state', 'pass^k'],
    interpretation: 'Tool agents need consistency and policy following over repeated conversational trials.',
  },
  {
    title: 'Scheming Evals',
    signals: ['goal conflict', 'oversight evasion', 'situational awareness', 'stealth reasoning'],
    interpretation: 'Advanced evals test whether models can reason about oversight and hide unsafe strategies in sandboxes.',
  },
];

export const GUARDRAILS = [
  'input filter',
  'prompt guard',
  'source trust label',
  'permission gate',
  'tool policy',
  'citation checker',
  'action logger',
  'human approval',
];
