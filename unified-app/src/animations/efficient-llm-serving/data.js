export const SERVING_TECHNIQUES = {
  continuousBatching: {
    label: 'Continuous Batching',
    solves: 'Keeps GPU slots filled as requests finish at different times.',
    risk: 'Can increase tail latency if admission is too aggressive.',
    bottleneck: 'Small or irregular batches',
  },
  pagedAttention: {
    label: 'PagedAttention',
    solves: 'Reduces KV cache fragmentation and enables dynamic block allocation.',
    risk: 'Requires block tables and specialized attention kernels.',
    bottleneck: 'KV memory fragmentation',
  },
  prefixCaching: {
    label: 'Prefix Caching',
    solves: 'Reuses KV for shared prompt prefixes.',
    risk: 'Only works for exact reusable prefixes and needs eviction policy.',
    bottleneck: 'Repeated shared prompt prefill',
  },
  chunkedPrefill: {
    label: 'Chunked Prefill',
    solves: 'Prevents long prompts from blocking decode-heavy workloads.',
    risk: 'Too-small chunks can add scheduling overhead.',
    bottleneck: 'Long prompts blocking decodes',
  },
  speculativeDecoding: {
    label: 'Speculative Decoding',
    solves: 'Reduces serial decode steps with draft/verify.',
    risk: 'A bad draft wastes compute.',
    bottleneck: 'One-token-at-a-time decode',
  },
  medusa: {
    label: 'Medusa',
    solves: 'Predicts multiple future tokens with extra target-model heads.',
    risk: 'Needs head training and tree verification.',
    bottleneck: 'Decode seriality',
  },
  eagle: {
    label: 'EAGLE-style Drafting',
    solves: 'Improves speculative acceptance with stronger draft generation.',
    risk: 'Training and integration complexity.',
    bottleneck: 'Low speculation acceptance',
  },
  quantization: {
    label: 'Quantization',
    solves: 'Reduces model memory and bandwidth.',
    risk: 'Quality loss or unsupported kernels.',
    bottleneck: 'Model memory and bandwidth',
  },
  kvQuantization: {
    label: 'KV Cache Quantization',
    solves: 'Reduces per-request memory for long contexts and batching.',
    risk: 'Attention error and dequantization overhead.',
    bottleneck: 'KV cache too large',
  },
};

export const SERVING_FAILURES = [
  {
    id: 'kv-fragmentation',
    label: 'KV fragmentation',
    symptom: 'Free memory exists but cannot fit new requests efficiently.',
    mitigation: 'Use paged KV allocation.',
  },
  {
    id: 'decode-starvation',
    label: 'Decode starvation',
    symptom: 'Long prefills delay active users waiting for next tokens.',
    mitigation: 'Use chunked prefill and decode-priority scheduling.',
  },
  {
    id: 'low-batch-utilization',
    label: 'Low batch utilization',
    symptom: 'GPU sits underused because requests finish at different times.',
    mitigation: 'Use continuous batching.',
  },
  {
    id: 'speculation-waste',
    label: 'Speculation waste',
    symptom: 'Draft tokens are often rejected.',
    mitigation: 'Improve draft quality, shorten draft length, or disable speculation.',
  },
  {
    id: 'tail-latency-spike',
    label: 'Tail latency spike',
    symptom: 'P99 latency rises even while throughput looks good.',
    mitigation: 'Tune batch size, queue timeout, priority, and chunking.',
  },
];

export const SERVING_TABS = [
  { id: 'serving-map', label: 'Serving Map' },
  { id: 'prefill-decode', label: 'Prefill vs Decode' },
  { id: 'continuous-batching', label: 'Continuous Batching' },
  { id: 'paged-attention', label: 'PagedAttention' },
  { id: 'prefix-caching', label: 'Prefix Caching' },
  { id: 'chunked-prefill', label: 'Chunked Prefill' },
  { id: 'speculative-decoding', label: 'Speculative Decoding' },
  { id: 'medusa', label: 'Medusa Heads' },
  { id: 'eagle', label: 'EAGLE Drafting' },
  { id: 'quantization', label: 'Quantization' },
  { id: 'kv-quantization', label: 'KV Cache Quantization' },
  { id: 'parallelism', label: 'Parallelism' },
  { id: 'throughput-latency', label: 'Throughput vs Latency' },
  { id: 'papers', label: 'Paper Decoder' },
];

export const PAPER_CARDS = [
  {
    id: 'pagedattention',
    label: 'PagedAttention / vLLM',
    signals: ['KV cache is huge and dynamic', 'Fixed-size KV blocks', 'Logical-to-physical block tables', '2-4x throughput improvements in evaluated settings'],
    interpretation: 'Serving performance depends heavily on KV memory management.',
  },
  {
    id: 'sarathi',
    label: 'SARATHI / Chunked Prefill',
    signals: ['Prefill and decode separation', 'Chunked long prompts', 'Decode-maximal batching', 'Reduced pipeline bubbles'],
    interpretation: 'Serving schedulers must manage prefill and decode together.',
  },
  {
    id: 'speculative',
    label: 'Speculative Decoding',
    signals: ['Draft model proposes', 'Target verifies in parallel', 'Distribution can be preserved', '2-3x acceleration in original experiments'],
    interpretation: 'Decode seriality can be reduced by proposing several tokens and verifying them together.',
  },
  {
    id: 'medusa',
    label: 'Medusa',
    signals: ['No separate draft model', 'Extra future-token heads', 'Tree attention verification', 'Frozen or jointly fine-tuned modes'],
    interpretation: 'Multi-token heads are an architecture-side way to reduce decode steps.',
  },
  {
    id: 'eagle3',
    label: 'EAGLE-3',
    signals: ['Direct token prediction', 'Multi-layer feature fusion', 'Training-time test', 'Speedups up to 6.5x'],
    interpretation: 'Speculative serving improves when the draft path is accurate and integrated.',
  },
  {
    id: 'kivi',
    label: 'KIVI / KV Quantization',
    signals: ['KV cache bottleneck', '2-bit asymmetric KV quantization', 'Per-channel key quantization', 'Larger batch sizes'],
    interpretation: 'Compressing KV activations can unlock larger batches and longer contexts.',
  },
  {
    id: 'kvquant',
    label: 'KVQuant',
    signals: ['Sub-4-bit KV quantization', 'Long-context target', 'Pre-RoPE key quantization', 'Outlier-aware methods'],
    interpretation: 'KV cache compression is central to very long-context serving.',
  },
];

export const REQUESTS = [
  { id: 'R1', prompt: 200, output: 80, prefix: false },
  { id: 'R2', prompt: 12000, output: 50, prefix: false },
  { id: 'R3', prompt: 1000, output: 800, prefix: false },
  { id: 'R4', prompt: 900, output: 120, prefix: true },
  { id: 'R5', prompt: 920, output: 180, prefix: true },
  { id: 'R6', prompt: 880, output: 140, prefix: true },
  { id: 'R7', prompt: 32000, output: 90, prefix: false },
  { id: 'R8', prompt: 1500, output: 1100, prefix: false },
  { id: 'R9', prompt: 300, output: 40, prefix: false },
  { id: 'R10', prompt: 6000, output: 260, prefix: false },
  { id: 'R11', prompt: 480, output: 160, prefix: true },
  { id: 'R12', prompt: 510, output: 200, prefix: true },
];

