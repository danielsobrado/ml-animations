function branch(id, label, type, children) {
  return { id, label, type, children };
}

function leaf(id, label, tip, lessonId) {
  return { id, label, tip, ...(lessonId ? { lessonId } : {}) };
}

export const MAPS = [
  {
    id: 'joint-attention',
    label: 'Joint Attention',
    center: {
      short: 'Joint attention lets tokens from different modalities—image patches and text tokens—attend to each other in one shared score matrix, mixing visual and linguistic context.',
      intuition: 'Each position asks a question; keys and values can come from either modality, so a caption token can read relevant pixels and vice versa.',
      formula: 'Attention(Q,K,V)=softmax(QK^T/\\sqrt{d_k})V',
      why: 'Joint attention powers vision-language models, multimodal LLMs, and cross-modal fusion where alignment between text and image regions matters.',
      trap: 'Joint attention can over-mix irrelevant cross-modal pairs if alignment or masking is weak.',
    },
    branches: [
      branch('prerequisites', 'Prerequisites', 'prerequisite', [
        leaf('self-attention-ja', 'Self-attention', { short: 'Each token scores all keys and mixes values.', intuition: 'Joint attention extends the same Q/K/V pattern across modalities.', trap: 'Self-attention alone does not fuse separate encoders without shared layers.', lessonId: 'self-attention' }),
        leaf('embeddings-ja', 'Embeddings', { short: 'Text and image patches start as vectors in a shared dimension.', intuition: 'Projections map each modality into one attention space.', trap: 'Mismatched embedding dims break Q/K/V alignment.', lessonId: 'embeddings' }),
        leaf('linear-projections-ja', 'Q / K / V projections', { short: 'Learned matrices create query, key, and value views.', intuition: 'Same token can ask, be searched, and carry information.', trap: 'Sharing weights across modalities can blur specialized features.' }),
        leaf('softmax-ja', 'Softmax', { short: 'Scores become nonnegative weights summing to 1 per query row.', intuition: 'Each query distributes attention budget across all visible keys.', trap: 'One huge score steals weight from every other key.', lessonId: 'softmax' }),
        leaf('attention-masks-ja', 'Attention masks', { short: 'Masks block illegal query-key pairs before softmax.', intuition: 'Padding, causal, or modality-specific visibility rules apply.', trap: 'Wrong mask lets text peek at future tokens or padded slots.', lessonId: 'attention-masks' }),
        leaf('positional-encoding-ja', 'Positional encoding', { short: 'Order information is added to patch and token embeddings.', intuition: 'Spatial layout and text order both affect which keys match.', trap: 'Permuting patches or tokens changes attention patterns.', lessonId: 'positional-encoding' }),
      ]),
      branch('mechanism', 'Core mechanism', 'mechanism', [
        leaf('unified-sequence-ja', 'Unified token sequence', { short: 'Concatenate image patches and text tokens into one sequence.', intuition: 'Attention runs over the combined length n_image + n_text.', trap: 'Different sequence lengths change memory and compute cost.' }),
        leaf('cross-modal-scores-ja', 'Cross-modal scores', { short: 'Text queries can score image keys and image queries can score text keys.', intuition: 'Off-diagonal blocks of the score matrix are cross-modal links.', trap: 'Treating modalities as isolated blocks defeats joint fusion.' }),
        leaf('within-modal-scores-ja', 'Within-modal scores', { short: 'Text-to-text and image-to-image attention still occur.', intuition: 'Local context within each modality is preserved.', trap: 'Over-strong cross-modal weights can ignore within-modal context.' }),
        leaf('weighted-value-mix-ja', 'Weighted value mix', { short: 'Output is a convex combination of value vectors from all attended keys.', intuition: 'High-weight keys pull their value information into the query output.', trap: 'Values from wrong modality can dominate if alignment is poor.' }),
        leaf('multi-head-ja', 'Multi-head attention', { short: 'Several heads run parallel attention with split dimensions.', intuition: 'One head may track objects; another tracks syntax.', trap: 'Too few heads collapse diverse alignment patterns.' }),
      ]),
      branch('intuitions', 'Intuitions', 'intuition', [
        leaf('caption-to-region-ja', 'Caption reads regions', { short: 'Word “dog” may attend strongly to dog-shaped patches.', intuition: 'Language grounds in visual evidence through attention weights.', trap: 'High weight does not prove causal importance or correctness.' }),
        leaf('region-to-word-ja', 'Regions read words', { short: 'Visual tokens can attend to descriptive text for disambiguation.', intuition: 'Text can resolve which object is “the small one”.', trap: 'Ambiguous captions can misdirect visual attention.' }),
        leaf('shared-workspace-ja', 'Shared workspace', { short: 'One attention layer is a meeting room for modalities.', intuition: 'Information exchange happens before downstream heads decide.', trap: 'Early fusion differs from late fusion via separate encoders only.' }),
        leaf('alignment-intuition-ja', 'Soft alignment', { short: 'Attention implements differentiable soft matching between tokens and patches.', intuition: 'Hard object boxes are not required for cross-modal links.', trap: 'Soft alignment can latch onto spurious correlations.' }),
        leaf('modality-gap-ja', 'Modality gap', { short: 'Raw image and text embeddings live in different statistical regimes.', intuition: 'Training aligns them so dot products become meaningful.', trap: 'Frozen unaligned encoders weaken joint attention.' }),
      ]),
      branch('formula-code', 'Formula / Code', 'formula', [
        leaf('score-formula-ja', 'Attention formula', { short: 'scores = QK^T / sqrt(d_k); weights = softmax(scores); out = weights V.', intuition: 'Same math as self-attention on a longer concatenated sequence.', formula: 'softmax(QK^T/\\sqrt{d_k})V', trap: 'Forgetting scaling lets softmax saturate on one key.' }),
        leaf('concat-layout-ja', 'Sequence layout', { short: '[image_tokens; text_tokens] or interleaved layouts vary by architecture.', intuition: 'Order affects which keys are nearby in the matrix.', code: 'x = concat(image_embeds, text_embeds, dim=1)', trap: 'Layout choice is not interchangeable across model families.' }),
        leaf('cross-block-ja', 'Cross-modal block', { short: 'Text row i attending image column j is score[i][j] in the cross block.', intuition: 'Inspect off-diagonal blocks to debug grounding.', trap: 'Averaging all weights hides sparse useful peaks.' }),
        leaf('mask-code-ja', 'Mask application', { short: 'Add large negative values to blocked score cells before softmax.', intuition: 'Padding and causal rules apply per query row.', code: 'scores = scores.masked_fill(mask == 0, -1e9)', trap: 'Mask dtype or shape bugs leak forbidden connections.' }),
        leaf('layer-stack-ja', 'Stacked layers', { short: 'Repeat joint attention blocks with MLP and residuals.', intuition: 'Deeper stacks refine cross-modal bindings.', trap: 'One shallow joint layer may not align fine details.' }),
      ]),
      branch('traps', 'Common traps', 'trap', [
        leaf('importance-trap-ja', 'Attention ≠ importance', { short: 'High cross-modal weight is not a certified causal attribution.', intuition: 'Weights are contextual mixtures, not ground truth.', trap: 'Using attention maps alone as explanation or safety proof.' }),
        leaf('spurious-alignment-trap-ja', 'Spurious alignment', { short: 'Models can align text to wrong regions that correlate in training.', intuition: 'Background cues can steal attention from objects.', trap: 'Assuming joint attention always finds the correct object.' }),
        leaf('modality-collapse-trap-ja', 'Modality collapse', { short: 'One modality can dominate value mixing if scales differ.', intuition: 'Normalization and balanced losses help both modalities speak.', trap: 'Text-heavy pretraining can ignore weak visual gradients.' }),
        leaf('length-trap-ja', 'Quadratic cost', { short: 'Full joint attention is O((n_img + n_txt)^2) in sequence length.', intuition: 'High-resolution images explode memory.', trap: 'Deploying full joint attention on long multimodal contexts without compression.' }),
        leaf('frozen-encoder-trap-ja', 'Frozen encoder trap', { short: 'Frozen vision backbones limit what joint layers can align.', intuition: 'End-to-end tuning often improves cross-modal scores.', trap: 'Expecting perfect grounding with mismatched frozen features.' }),
      ]),
      branch('used-later', 'Used later', 'application', [
        leaf('multimodal-llm-ja', 'Multimodal LLM', { short: 'Modern VLMs stack joint or cross-attention fusion layers.', intuition: 'Same principle scales to video, audio, and documents.', trap: 'Architecture names differ but fusion problem remains.', lessonId: 'multimodal-llm' }),
        leaf('rag-multimodal-ja', 'Multimodal RAG', { short: 'Retrieved images and text both enter the joint sequence.', intuition: 'Attention decides which evidence supports the answer.', trap: 'Irrelevant retrieved images can hijack attention.', lessonId: 'rag' }),
        leaf('transformer-ja', 'Transformer', { short: 'Joint attention is one block inside the transformer recipe.', intuition: 'Residuals, LayerNorm, and MLPs surround attention.', trap: 'Attention alone is not the full multimodal stack.', lessonId: 'transformer' }),
        leaf('flash-attention-ja', 'Flash Attention', { short: 'Tiled exact attention reduces memory for long joint sequences.', intuition: 'Implementation trick; math unchanged.', trap: 'Flash Attention does not fix bad cross-modal alignment.', lessonId: 'flash-attention' }),
        leaf('self-attention-used-ja', 'Self-attention', { short: 'Unimodal stacks reuse the same attention primitive.', intuition: 'Joint attention generalizes self-attention across modalities.', trap: 'Confusing cross-attention-only encoders with full joint fusion.', lessonId: 'self-attention' }),
      ]),
    ],
  },
];
