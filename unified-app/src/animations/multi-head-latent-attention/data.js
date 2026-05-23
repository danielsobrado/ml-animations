export const ATTENTION_MODES = {
  mha: {
    label: 'Multi-Head Attention (MHA)',
    cacheType: 'full-kv-per-head',
    description: 'Each query head has its own corresponding K/V heads.',
    memoryStrategy: 'Store all K and V heads separately for every layer and token.',
    tradeoff: 'Highest cache size, maximum head independence and expressiveness.',
    ratioFormula: '1.0 (100% of MHA)',
  },
  mqa: {
    label: 'Multi-Query Attention (MQA)',
    cacheType: 'single-shared-kv',
    description: 'All query heads share a single K/V head.',
    memoryStrategy: 'Store exactly one K/V head pair per layer, shared by all queries.',
    tradeoff: 'Smallest cache size, high risk of quality loss and representational capacity drop.',
    ratioFormula: '1 / H_q',
  },
  gqa: {
    label: 'Grouped-Query Attention (GQA)',
    cacheType: 'grouped-kv',
    description: 'Groups of query heads share K/V heads.',
    memoryStrategy: 'Store one K/V head pair per query head group.',
    tradeoff: 'Middle ground. Approaching MHA quality while keeping speed close to MQA.',
    ratioFormula: 'H_kv / H_q',
  },
  mla: {
    label: 'Multi-Head Latent Attention (MLA)',
    cacheType: 'compressed-latent-kv',
    description: 'K/V information is compressed and stored as a latent state vector.',
    memoryStrategy: 'Store a compressed latent state per token, up-projecting it during attention.',
    tradeoff: 'Very small cache size, preserves quality, but introduces query-time projection compute.',
    ratioFormula: 'd_c / (2 * H_q * d_h)',
  },
};

export const PAPER_ANCHORS = [
  {
    id: 'mqa',
    title: 'Fast Transformer Decoding (MQA)',
    takeaway: 'Multi-Query Attention (MQA) was proposed by Noam Shazeer to reduce memory bandwidth bottleneck during incremental autoregressive decoding.',
    citation: 'Shazeer, M. (2019). "Fast Transformer Decoding: One Write-Head is All You Need." arXiv:1911.02150.',
  },
  {
    id: 'gqa',
    title: 'Grouped-Query Attention (GQA)',
    takeaway: 'GQA generalizes MQA by using an intermediate number of K/V heads. It achieves quality close to MHA with speeds comparable to MQA.',
    citation: 'Ainslie, J. et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." arXiv:2305.13245.',
  },
  {
    id: 'deepseek-v2',
    title: 'DeepSeek-V2 (MLA)',
    takeaway: 'DeepSeek-V2 introduces Multi-head Latent Attention (MLA) to compress the KV cache into latent vectors, reducing KV cache by 93.3% and boosting throughput by 5.76×.',
    citation: 'DeepSeek-AI. (2024). "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." arXiv:2405.04434.',
  },
  {
    id: 'deepseek-v3',
    title: 'DeepSeek-V3 (SOTA MoE + MLA)',
    takeaway: 'DeepSeek-V3 adopts MLA and DeepSeekMoE to achieve extremely cost-effective training and high-throughput inference for a 671B parameter model with 37B active parameters.',
    citation: 'DeepSeek-AI. (2025). "DeepSeek-V3 Technical Report." arXiv:2412.19437.',
  },
  {
    id: 'transmla',
    title: 'TransMLA / post-training conversion',
    takeaway: 'MLA compresses KV cache size and shifts the workload from being memory-bandwidth limited to more compute-bound, showing how checkpoints can be converted.',
    citation: 'TransMLA authors. (2025). "TransMLA: Multi-head Latent Attention Translation." arXiv:2501.xxxxx.',
  },
];
