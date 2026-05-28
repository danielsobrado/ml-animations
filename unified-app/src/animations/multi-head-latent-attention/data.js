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
    tradeoff: 'Very small cache size with extra projection compute and implementation complexity.',
    ratioFormula: 'd_c / (2 * H_q * d_h)',
  },
  transmla: {
    label: 'TransMLA conversion',
    cacheType: 'gqa-to-latent-factorization',
    description: 'A GQA checkpoint is rewritten as an MLA-style low-rank latent cache.',
    memoryStrategy: 'Move GQA repetition to the parameter side, then factorize the repeated matrix.',
    tradeoff: 'Same cache budget as GQA can support a more expressive MLA parameter family.',
    ratioFormula: 'same overhead as converted GQA',
  },
};

export const PAPER_ANCHORS = [
  {
    id: 'mqa',
    title: 'Fast Transformer Decoding (MQA)',
    takeaway: 'Multi-Query Attention (MQA) reduces memory bandwidth bottlenecks during incremental autoregressive decoding by sharing one K/V source.',
    citation: 'Shazeer, M. (2019). "Fast Transformer Decoding: One Write-Head is All You Need." arXiv:1911.02150.',
    href: 'https://arxiv.org/abs/1911.02150',
  },
  {
    id: 'gqa',
    title: 'Grouped-Query Attention (GQA)',
    takeaway: 'GQA generalizes MQA with an intermediate number of K/V heads, preserving more quality while keeping cache traffic lower than MHA.',
    citation: 'Ainslie, J. et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." arXiv:2305.13245.',
    href: 'https://arxiv.org/abs/2305.13245',
  },
  {
    id: 'deepseek-v2',
    title: 'DeepSeek-V2 (MLA)',
    takeaway: 'DeepSeek-V2 introduces Multi-head Latent Attention with low-rank KV joint compression, reporting 93.3% lower KV cache and 5.76x higher maximum generation throughput versus DeepSeek 67B.',
    citation: 'DeepSeek-AI. (2024). "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." arXiv:2405.04434.',
    href: 'https://arxiv.org/abs/2405.04434',
  },
  {
    id: 'transmla',
    title: 'TransMLA / post-training conversion',
    takeaway: 'TransMLA shows that MLA can represent GQA at the same KV-cache overhead, while GQA cannot represent every MLA configuration, and proposes post-training conversion from GQA to MLA.',
    citation: 'Meng, F. et al. (2025). "TransMLA: Multi-Head Latent Attention Is All You Need." arXiv:2502.07864.',
    href: 'https://arxiv.org/abs/2502.07864',
  },
];
