export const MOE_PRESETS = {
  toy: {
    id: 'toy',
    label: 'Toy MoE',
    totalExperts: 8,
    activeRoutedExperts: 2,
    sharedExperts: 0,
    totalParamsB: 8,
    activeParamsB: 2,
    contextLength: '4K',
    denseBackboneB: 1.0,
    expertSizeB: 0.5,
    note: 'Teaching-scale configuration',
  },
  deepseekV3: {
    id: 'deepseekV3',
    label: 'DeepSeek-V3 inspired',
    totalExperts: 256,
    activeRoutedExperts: 8,
    sharedExperts: 1,
    totalParamsB: 671,
    activeParamsB: 37,
    contextLength: '128K',
    denseBackboneB: 29.8,
    expertSizeB: 0.6,
    note: 'Uses DeepSeekMoE-style sparse experts and auxiliary-loss-free balancing',
  },
  qwen3Large: {
    id: 'qwen3Large',
    label: 'Qwen3-235B-A22B inspired',
    totalExperts: 128,
    activeRoutedExperts: 8,
    sharedExperts: 0,
    totalParamsB: 235,
    activeParamsB: 22,
    contextLength: '128K',
    denseBackboneB: 14.0,
    expertSizeB: 1.0,
    note: 'Large Qwen3 MoE preset',
  },
  llama4Maverick: {
    id: 'llama4Maverick',
    label: 'Llama 4 Maverick inspired',
    totalExperts: 128,
    activeRoutedExperts: 1,
    sharedExperts: 1,
    totalParamsB: 400,
    activeParamsB: 17,
    contextLength: '1M',
    denseBackboneB: 14.0,
    expertSizeB: 2.0,
    note: 'Shared expert plus one routed expert',
  },
};

export const TOKEN_DOMAINS = [
  { token: 'the', domain: 'common', colorClass: 'border-blue-200 bg-blue-50 text-blue-900' },
  { token: 'function', domain: 'code', colorClass: 'border-emerald-200 bg-emerald-50 text-emerald-900' },
  { token: 'integral', domain: 'math', colorClass: 'border-purple-200 bg-purple-50 text-purple-900' },
  { token: 'bonjour', domain: 'multilingual', colorClass: 'border-orange-200 bg-orange-50 text-orange-900' },
  { token: 'contract', domain: 'legal', colorClass: 'border-rose-200 bg-rose-50 text-rose-900' },
  { token: 'tensor', domain: 'ml', colorClass: 'border-pink-200 bg-pink-50 text-pink-900' },
];

export const FAILURE_MODES = [
  {
    id: 'expert-collapse',
    label: 'Expert collapse',
    symptom: 'Most tokens route to a small subset of experts.',
    fix: 'Increase balancing pressure or adjust router bias/capacity.',
  },
  {
    id: 'dead-experts',
    label: 'Dead experts',
    symptom: 'Some experts receive nearly no tokens.',
    fix: 'Improve exploration, balancing, initialization, or expert segmentation.',
  },
  {
    id: 'token-dropping',
    label: 'Token dropping',
    symptom: 'Overloaded experts exceed capacity.',
    fix: 'Increase capacity factor, improve load balance, or change routing.',
  },
  {
    id: 'communication-bottleneck',
    label: 'Communication bottleneck',
    symptom: 'Cross-GPU dispatch dominates runtime.',
    fix: 'Improve expert placement, batching, overlap, or parallelism strategy.',
  },
];

export const PAPER_SIGNAL_CARDS = [
  {
    title: 'DeepSeek-V3',
    signals: '671B total / 37B active, DeepSeekMoE, MLA, auxiliary-loss-free balancing, multi-token prediction',
    interpretation: 'Frontier sparse model with compressed attention and careful balancing.',
  },
  {
    title: 'DeepSeekMoE',
    signals: 'fine-grained expert segmentation, shared experts, routed experts, expert specialization',
    interpretation: 'MoE design focused on reducing expert redundancy and improving specialization.',
  },
  {
    title: 'Qwen3',
    signals: 'dense and MoE family, 30B-A3B and 235B-A22B, 128 total experts / 8 activated in MoE configs, thinking and non-thinking modes',
    interpretation: 'MoE is part of a family-level latency-quality tradeoff, not just a single model.',
  },
  {
    title: 'Llama 4 Maverick',
    signals: '17B active / 400B total, 128 routed experts, shared expert, distillation from Behemoth teacher',
    interpretation: 'Shared + routed experts plus teacher distillation for deployable frontier multimodal MoE.',
  },
];
