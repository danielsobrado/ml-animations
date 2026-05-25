export const NODE_TYPES = {
  prerequisite: {
    label: 'Prerequisite',
    color: '#234b8f',
    side: 'left',
  },
  mechanism: {
    label: 'Mechanism',
    color: '#1f1f1f',
    side: 'right',
  },
  intuition: {
    label: 'Intuition',
    color: '#4d6f39',
    side: 'left',
  },
  formula: {
    label: 'Formula / Code',
    color: '#8a5a2b',
    side: 'right',
  },
  trap: {
    label: 'Common trap',
    color: '#7a3f36',
    side: 'left',
  },
  application: {
    label: 'Used later',
    color: '#6b4d8a',
    side: 'right',
  },
};

function tip(fields) {
  return fields;
}

export const CONCEPT_MAPS = {
  'matrix-multiplication': {
    center: {
      id: 'matrix-multiplication',
      label: 'Matrix Multiplication',
      type: 'current',
      tooltip: tip({
        short: 'Matrix multiplication combines rows of A with columns of B. Each output cell is one dot product.',
        intuition: 'Rows of A meet columns of B; the shared index k walks across one row and down one column.',
        formula: 'Cᵢⱼ = Σₖ AᵢₖBₖⱼ',
        why: 'This is the basic operation behind linear layers, transformations, attention scores, least squares, and many ML computations.',
        trap: 'It is not elementwise multiplication of matching cells.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'scalar-multiplication',
            label: 'Scalar multiplication',
            tooltip: tip({
              short: 'Multiply two numbers.',
              intuition: 'Matrix multiplication repeats this tiny operation many times: one number from a row of A times one number from a column of B.',
              example: 'A[i][k] × B[k][j]',
              trap: 'One product is only one term, not the whole cell.',
            }),
          },
          {
            id: 'addition-accumulation',
            label: 'Addition / accumulation',
            tooltip: tip({
              short: 'A dot product adds all pair-products into one total.',
              intuition: 'The accumulator is the “sum” part of matrix multiplication.',
              example: '1×3 + 2×4 = 11',
              trap: 'Stopping after one multiplication gives one term, not the full cell.',
            }),
          },
          {
            id: 'vector',
            label: 'Vector',
            tooltip: tip({
              short: 'A vector is an ordered list of numbers.',
              intuition: 'A row of a matrix and a column of a matrix can both be treated as vectors.',
              example: 'Row [1, 2] and column [3, 4] meet in one dot product.',
              trap: 'Rows and columns are not interchangeable when building C[i][j].',
            }),
          },
          {
            id: 'dot-product',
            label: 'Dot product',
            tooltip: tip({
              short: 'Multiply matching entries, then add.',
              intuition: 'One matrix output cell is exactly one dot product.',
              example: '[1, 2] · [3, 4] = 1×3 + 2×4 = 11',
              trap: 'Do not forget the summation step.',
            }),
          },
          {
            id: 'matrix-shape',
            label: 'Matrix shape',
            tooltip: tip({
              short: 'Shape tells you rows × columns.',
              intuition: 'If A is m×n and B is n×p, then AB is m×p. The shared inner dimension n must match.',
              example: '(2×3)(3×4) → 2×4',
              trap: 'The inner dimensions must match.',
            }),
          },
          {
            id: 'row-column-indexing',
            label: 'Row / column indexing',
            tooltip: tip({
              short: 'To compute C[i][j], use row i of A and column j of B.',
              intuition: 'The shared index k walks across A’s row and down B’s column.',
              example: 'C[0][1] uses row 0 of A and column 1 of B.',
              trap: 'A common bug is using row j from B instead of column j.',
            }),
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'output-shape',
            label: 'Output shape',
            tooltip: tip({
              short: 'If A has m rows and B has p columns, the result has m rows and p columns.',
              intuition: 'The inner dimension disappears because it is summed over.',
              example: '2×3 times 3×4 gives 2×4.',
              trap: 'Do not expect the inner n to appear in the output shape.',
            }),
            highlightTarget: { panel: 'animation', type: 'matrix-shape' },
          },
          {
            id: 'one-output-cell',
            label: 'One output cell',
            tooltip: tip({
              short: 'C[i][j] is built from row i of A and column j of B.',
              intuition: 'Focus on one cell first; the whole matrix is just that rule repeated.',
              example: 'C[0][1] = row 0 of A · column 1 of B.',
              trap: 'Do not use row 2 of B; use column 2.',
            }),
            highlightTarget: { panel: 'animation', type: 'matrix-cell', row: 0, col: 1 },
          },
          {
            id: 'shared-index-k',
            label: 'Shared index k',
            tooltip: tip({
              short: 'k is the dimension that A and B share.',
              intuition: 'k walks across the selected row of A and down the selected column of B.',
              formula: 'Cᵢⱼ = Σₖ AᵢₖBₖⱼ',
              code: 'total += A[row][k] * B[k][col];',
              example: 'For C[0][1], k = 0 gives A[0][0] × B[0][1]; then k = 1 gives A[0][1] × B[1][1].',
              trap: 'Do not use B[col][k]. That reads a row of B, not a column.',
              why: 'Attention score matrices, matrix-vector multiplication, least squares.',
            }),
            highlightTarget: { panel: 'animation', type: 'shared-index-k' },
          },
          {
            id: 'sum-over-k',
            label: 'Sum over k',
            tooltip: tip({
              short: 'The formula Cᵢⱼ = Σₖ AᵢₖBₖⱼ means: multiply every matching pair along k, then add the products.',
              intuition: 'Σₖ is the loop that accumulates one cell.',
              formula: 'Cᵢⱼ = Σₖ AᵢₖBₖⱼ',
              example: 'C[0][1] = A[0][0]B[0][1] + A[0][1]B[1][1]',
              trap: 'Each cell is not one product; it is a sum of products.',
            }),
          },
          {
            id: 'repeat-for-every-cell',
            label: 'Repeat for every cell',
            tooltip: tip({
              short: 'Once you know how to compute one cell, use two outer loops: one for output rows i and one for output columns j.',
              intuition: 'The third loop over k computes one cell.',
              code: 'for each output row i:\n  for each output column j:\n    compute C[i][j]',
              trap: 'Mixing loop bounds for i, j, and k is a common shape bug.',
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'rows-meet-columns',
            label: 'Rows meet columns',
            tooltip: tip({
              short: 'Each row of A meets each column of B.',
              intuition: 'Their dot product becomes one cell in the output.',
              example: 'Row 1 of A meets column 3 of B.',
              trap: 'This is not matching cells element-by-element.',
            }),
          },
          {
            id: 'many-dot-products',
            label: 'Many dot products',
            tooltip: tip({
              short: 'Matrix multiplication is a grid of dot products.',
              intuition: 'If the output has 2×3 cells, you are doing 6 dot products.',
              example: 'A 2×3 output needs six row-column dot products.',
              trap: 'Thinking “one multiply” per cell misses the inner sum.',
            }),
          },
          {
            id: 'linear-combination-columns',
            label: 'Linear combination of columns',
            tooltip: tip({
              short: 'Multiplying A by a vector combines A’s columns using the vector’s entries as weights.',
              intuition: 'Matrix multiplication repeats that for many vectors at once.',
              example: 'Column 1 of A is weighted by the first entry of the vector.',
              trap: 'This view still uses the same row-column rule for full matrices.',
            }),
          },
          {
            id: 'composition-transformations',
            label: 'Composition of transformations',
            tooltip: tip({
              short: 'AB means apply B first, then A.',
              intuition: 'A matrix can represent a linear transformation; order matters.',
              example: 'Rotate then scale differs from scale then rotate.',
              trap: 'This is why AB usually differs from BA.',
            }),
          },
          {
            id: 'feature-mixing',
            label: 'Feature mixing',
            tooltip: tip({
              short: 'In ML, matrix multiplication mixes features.',
              intuition: 'A layer takes input features and combines them into new features using learned weights.',
              example: 'A linear layer computes XW.',
              trap: 'Weights decide the mixture, not just the input values.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'formula-cij',
            label: 'Cᵢⱼ = Σₖ AᵢₖBₖⱼ',
            tooltip: tip({
              short: 'Output cell at row i, column j equals the sum of products over the shared dimension k.',
              intuition: 'Read Cᵢⱼ as the cell; read Σₖ as the loop inside that cell.',
              formula: 'Cᵢⱼ = Σₖ AᵢₖBₖⱼ',
              example: 'C[0][1] = A[0][0]B[0][1] + A[0][1]B[1][1]',
              trap: 'i and j choose the output cell; k chooses terms inside it.',
            }),
            highlightTarget: { panel: 'code', type: 'formula' },
          },
          {
            id: 'cell-code',
            label: 'Cell code',
            tooltip: tip({
              short: 'total += A[row][k] * B[k][col]',
              intuition: 'This line is the heart of matrix multiplication.',
              code: 'total += A[row][k] * B[k][col];',
              example: 'Loop k from 0 to B.length - 1.',
              trap: 'A[row][k] and B[k][col] must share k.',
            }),
            highlightTarget: { panel: 'code', type: 'cell-loop' },
          },
          {
            id: 'three-loop-structure',
            label: 'Three-loop structure',
            tooltip: tip({
              short: 'Outer loops pick the output cell; inner loop sums over k.',
              intuition: 'for i, for j, then for k inside.',
              code: 'for (let i = 0; i < m; i++) {\n  for (let j = 0; j < p; j++) {\n    let total = 0;\n    for (let k = 0; k < n; k++) {\n      total += A[i][k] * B[k][j];\n    }\n    C[i][j] = total;\n  }\n}',
              trap: 'Swapping which loop is innermost changes what you compute.',
            }),
            highlightTarget: { panel: 'code', type: 'full-loops' },
          },
          {
            id: 'shape-check',
            label: 'Shape check',
            tooltip: tip({
              short: 'A[0].length === B.length',
              intuition: 'The columns of A must equal the rows of B.',
              code: 'if (A[0].length !== B.length) throw new Error("shape mismatch");',
              trap: 'Otherwise row-column dot products do not line up.',
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'elementwise-trap',
            label: 'Not elementwise',
            tooltip: tip({
              short: 'Matrix multiply is not A[i][j] × B[i][j].',
              intuition: 'Each output cell uses a whole row and a whole column.',
              example: 'A = [[1,2]], B = [[3],[4]] → [[11]], not elementwise [[3,8]].',
              trap: 'Elementwise multiplication has a different name and shape rule.',
            }),
            highlightTarget: { panel: 'animation', type: 'counterexample-elementwise' },
          },
          {
            id: 'shape-mismatch',
            label: 'Shape mismatch',
            tooltip: tip({
              short: 'A×B is valid only when columns(A) = rows(B).',
              intuition: 'If the inner dimensions do not match, the dot products are undefined.',
              example: '2×3 times 4×2 is invalid because 3 ≠ 4.',
              trap: 'Do not swap matrices just to make code run.',
            }),
          },
          {
            id: 'order-trap',
            label: 'Order matters',
            tooltip: tip({
              short: 'Usually AB ≠ BA.',
              intuition: 'Matrix multiplication represents composition, and changing order changes the transformation.',
              example: 'A 2×3 times 3×4 is valid, but 3×4 times 2×3 is not.',
              trap: 'Do not assume commutativity from scalar multiplication.',
            }),
          },
          {
            id: 'wrong-row-column',
            label: 'Wrong row / column',
            tooltip: tip({
              short: 'C[i][j] uses row i from A and column j from B.',
              intuition: 'Using row j from B is a frequent indexing bug.',
              example: 'Correct: B[k][j]. Wrong: B[j][k] for a column read.',
              trap: 'Math indices and JavaScript indices both need care.',
            }),
          },
          {
            id: 'off-by-one-indexing',
            label: 'Off-by-one indexing',
            tooltip: tip({
              short: 'Math notation often starts at 1; JavaScript arrays start at 0.',
              intuition: 'C₁₁ in math is C[0][0] in code.',
              example: 'Row i in math is index i - 1 in zero-based code.',
              trap: 'Mixing 1-based formulas with 0-based loops causes wrong cells.',
            }),
          },
          {
            id: 'forgetting-the-sum',
            label: 'Forgetting the sum',
            tooltip: tip({
              short: 'Each cell is a sum of products, not one product.',
              intuition: 'If you only multiply one pair, you computed one term, not the full cell.',
              example: 'A[0][0]×B[0][1] is only the k = 0 term of C[0][1].',
              trap: 'Initialize total = 0 and accumulate every k.',
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'matrix-vector-multiplication',
            label: 'Matrix-vector multiplication',
            tooltip: tip({
              short: 'Same row-column rule; B has only one column.',
              intuition: 'Each output entry is one row dotted with the vector.',
              example: 'y = A x combines rows of A with entries of x.',
              trap: 'Do not treat the vector as a row when the formula expects a column.',
              why: 'Feeds into linear layers and least squares.',
            }),
          },
          {
            id: 'identity-matrix',
            label: 'Identity matrix',
            tooltip: tip({
              short: 'I acts like the number 1 for matrices.',
              intuition: 'AI = A and IA = A when shapes line up.',
              example: 'A 2×2 identity has 1s on the diagonal and 0 elsewhere.',
              trap: 'Identity only composes cleanly when dimensions match.',
              why: 'Useful when undoing or chaining transformations.',
            }),
          },
          {
            id: 'projection',
            label: 'Projection',
            tooltip: tip({
              short: 'Projection finds the closest point in a subspace.',
              intuition: 'Matrix multiplication builds predicted targets Ax.',
              example: 'Residual b - Ax is orthogonal to the column space at the least-squares solution.',
              trap: 'Projection depends on the metric and subspace you choose.',
            }),
            lessonId: 'least-squares-projection',
          },
          {
            id: 'least-squares',
            label: 'Least squares',
            tooltip: tip({
              short: 'Predictions are Ax; residuals measure what is left over.',
              intuition: 'Matrix multiplication maps coefficients to predicted targets.',
              example: 'Residual = b - Ax.',
              trap: 'Least squares chooses the closest Ax, not always exact b.',
            }),
            lessonId: 'least-squares-projection',
          },
          {
            id: 'linear-layers',
            label: 'Linear layers',
            tooltip: tip({
              short: 'A neural-network linear layer computes XW + b.',
              intuition: 'The matrix multiplication mixes input features into output features.',
              example: 'Batch inputs times weight matrix.',
              trap: 'Shape errors are common in neural-network code.',
            }),
            lessonId: 'neural-network',
          },
          {
            id: 'attention-scores',
            label: 'Attention scores',
            tooltip: tip({
              short: 'Transformer attention uses QKᵀ.',
              intuition: 'Queries meet keys through matrix multiplication to compute similarity scores.',
              example: 'Each score is query dot key.',
              trap: 'Attention score matrices can become huge.',
            }),
            lessonId: 'self-attention',
          },
          {
            id: 'matrix-decompositions',
            label: 'Matrix decompositions',
            tooltip: tip({
              short: 'Decompositions split a matrix into useful factors.',
              intuition: 'Understanding multiplication makes QR, SVD, and eigen decompositions easier.',
              example: 'SVD rewrites one matrix multiply as three simpler ones.',
              trap: 'Different decompositions answer different questions.',
            }),
            lessonId: 'matrix-decompositions',
          },
        ],
      },
    ],
  },
  'self-attention': {
    center: {
      id: 'self-attention',
      label: 'Self-Attention',
      type: 'current',
      tooltip: tip({
        short: 'Each token builds a query, scores every key in the sequence, softmaxes the scores, and mixes value vectors into a new context-aware representation.',
        intuition: 'Every position asks the same sequence a different question; the answers are attention weights over values.',
        formula: 'Attention(Q, K, V) = softmax(QKᵀ / √d_k) V',
        why: 'Self-attention is the core mixing step inside transformers, BERT, GPT-style models, diffusion text encoders, and many multimodal stacks.',
        trap: 'It is not a fixed lookup table; weights are recomputed from the current sequence.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'matrix-multiplication',
            label: 'Matrix multiplication',
            tooltip: tip({
              short: 'Attention scores and outputs are built from matrix multiplies.',
              intuition: 'QKᵀ is rows of Q meeting columns of K; the weighted values step is another multiply.',
              example: 'Scores[i][j] = dot(query_i, key_j).',
              trap: 'Do not treat QKᵀ as elementwise multiplication.',
            }),
            lessonId: 'matrix-multiplication',
          },
          {
            id: 'dot-product',
            label: 'Dot product',
            tooltip: tip({
              short: 'A query-key score is one dot product.',
              intuition: 'Higher dot product means stronger alignment between what is asked and what is offered.',
              example: 'score = q · k = Σ_d q_d k_d',
              trap: 'Raw dot products can be huge before scaling; that is why √d_k appears.',
            }),
          },
          {
            id: 'softmax',
            label: 'Softmax',
            tooltip: tip({
              short: 'Turns scores into nonnegative weights that sum to 1.',
              intuition: 'Each query row becomes a probability distribution over keys.',
              example: 'weights = softmax(scores_row)',
              trap: 'Changing one score changes every weight in that row.',
            }),
            lessonId: 'softmax',
          },
          {
            id: 'embeddings',
            label: 'Token embeddings',
            tooltip: tip({
              short: 'Each token starts as a vector in ℝᵈ.',
              intuition: 'Q, K, and V are learned linear views of the same token vectors.',
              example: 'x_i → W_Q x_i, W_K x_i, W_V x_i',
              trap: 'Embeddings alone do not encode order; position information is added separately.',
            }),
            lessonId: 'embeddings',
          },
          {
            id: 'linear-projections',
            label: 'Q / K / V projections',
            tooltip: tip({
              short: 'Three weight matrices create query, key, and value views.',
              intuition: 'The same token plays three roles: question asker, searchable tag, and information carrier.',
              example: 'Q = XW_Q, K = XW_K, V = XW_V',
              trap: 'Q, K, and V are not interchangeable even though they come from the same tokens.',
            }),
          },
          {
            id: 'sequence-positions',
            label: 'Sequence positions',
            tooltip: tip({
              short: 'Self-attention runs over n tokens at once.',
              intuition: 'Position i produces one output vector that may read all n keys (unless masked).',
              example: 'A length-128 sentence yields a 128×128 score matrix before masking.',
              trap: 'Without positional encoding, permuting tokens permutes behavior.',
            }),
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'query-role',
            label: 'Query role',
            tooltip: tip({
              short: 'A query says what information this position wants.',
              intuition: 'Row i of Q asks every key how useful it is for updating token i.',
              example: 'query_i = W_Q x_i',
              trap: 'A query does not carry the final mixed content; values do.',
            }),
          },
          {
            id: 'key-role',
            label: 'Key role',
            tooltip: tip({
              short: 'A key advertises what a position can offer.',
              intuition: 'Column j of K is compared against every query to score position j.',
              example: 'key_j = W_K x_j',
              trap: 'Keys are scored, not mixed directly into the output.',
            }),
          },
          {
            id: 'value-role',
            label: 'Value role',
            tooltip: tip({
              short: 'Values are the information that gets mixed.',
              intuition: 'After weights are chosen, the output is a weighted sum of value vectors.',
              example: 'output_i = Σ_j α_ij v_j',
              trap: 'High score on a key that points the wrong way still pulls the wrong value.',
            }),
          },
          {
            id: 'score-matrix',
            label: 'Score matrix QKᵀ',
            tooltip: tip({
              short: 'Every query compares with every key.',
              intuition: 'Cell (i, j) measures how much position i wants information from position j.',
              example: 'scores[i][j] = dot(query_i, key_j)',
              trap: 'The full matrix is n×n; long contexts make this expensive.',
            }),
            highlightTarget: { panel: 'animation', type: 'score-matrix' },
          },
          {
            id: 'scaled-dot-product',
            label: 'Scale by √d_k',
            tooltip: tip({
              short: 'Divide scores by √d_k before softmax.',
              intuition: 'Dot products grow with dimension; scaling keeps softmax from saturating too early.',
              example: 'scaled = scores / Math.sqrt(d_k)',
              trap: 'Skipping the scale can make one key dominate every row.',
            }),
          },
          {
            id: 'weighted-value-mix',
            label: 'Weighted value mix',
            tooltip: tip({
              short: 'Each output row is a weighted sum of values.',
              intuition: 'Attention weights choose how much of each value vector to copy into the new representation.',
              example: 'context_i = Σ_j softmax(scores[i][j]) * value_j',
              trap: 'If weights are wrong, the mixture is wrong even when values are good.',
            }),
            highlightTarget: { panel: 'animation', type: 'context-output' },
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'every-token-asks',
            label: 'Every token asks',
            tooltip: tip({
              short: 'All positions run attention in parallel.',
              intuition: 'Each token issues its own query against the same sequence of keys.',
              example: 'Token “bank” may attend strongly to “river” or “money” depending on context.',
              trap: 'Parallel does not mean independent; softmax still couples keys within a row.',
            }),
          },
          {
            id: 'routing-mixture',
            label: 'Routing mixture',
            tooltip: tip({
              short: 'Attention routes information instead of compressing blindly.',
              intuition: 'The model decides which prior tokens to copy into the current representation.',
              example: 'A pronoun token may route weight to its antecedent noun.',
              trap: 'Routing is learned behavior, not guaranteed correct alignment.',
            }),
          },
          {
            id: 'context-vector',
            label: 'Context vector',
            tooltip: tip({
              short: 'The output is a new context-aware token representation.',
              intuition: 'After attention, each position has absorbed useful information from others.',
              example: 'Output at position 5 blends values from positions 2, 5, and 9.',
              trap: 'One attention layer may not finish disambiguation; deeper stacks help.',
            }),
          },
          {
            id: 'content-based-addressing',
            label: 'Content-based addressing',
            tooltip: tip({
              short: 'Matches depend on vector content, not fixed indices.',
              intuition: 'Similar queries and keys score higher even if positions change.',
              example: 'Repeated phrase pieces can attend to earlier matching spans.',
              trap: 'Content-based matching still needs position signals for word order tasks.',
            }),
          },
          {
            id: 'not-global-importance',
            label: 'Weights are local to the row',
            tooltip: tip({
              short: 'A high weight for token j means “useful for query i,” not “globally important.”',
              intuition: 'Different queries on the same sequence can highlight different keys.',
              example: 'The verb query and the subject query may peak on different columns.',
              trap: 'Do not read one attention map as a universal importance ranking.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'attention-formula',
            label: 'Attention(Q,K,V)',
            tooltip: tip({
              short: 'Attention(Q, K, V) = softmax(QKᵀ / √d_k) V.',
              intuition: 'Scores, normalize each row, then mix values.',
              formula: 'Attention(Q, K, V) = softmax(QKᵀ / √d_k) V',
              trap: 'Apply mask before softmax when some keys must be invisible.',
            }),
            highlightTarget: { panel: 'code', type: 'attention-formula' },
          },
          {
            id: 'scores-code',
            label: 'Score computation',
            tooltip: tip({
              short: 'scores = Q @ K.T / sqrt(d_k)',
              intuition: 'Matrix multiply all queries against all keys, then scale.',
              code: 'const scale = 1 / Math.sqrt(d_k);\nconst scores = matmul(Q, transpose(K)) * scale;',
              example: 'Shape: (n, d) × (d, n) → (n, n).',
              trap: 'Transpose K, not Q, when keys are stored row-wise per token.',
            }),
          },
          {
            id: 'softmax-row-code',
            label: 'Softmax per query row',
            tooltip: tip({
              short: 'Normalize each row of scores into attention weights.',
              intuition: 'Row i is the distribution of attention for query i.',
              code: 'const weights = softmax(scores[i], axis=keys);',
              trap: 'Softmax across the wrong axis breaks the probabilistic interpretation.',
            }),
          },
          {
            id: 'output-mix-code',
            label: 'Mix values',
            tooltip: tip({
              short: 'output = weights @ V',
              intuition: 'Each output row is a weighted average of value vectors.',
              code: 'const context = matmul(weights, V);',
              example: 'weights shape (n, n), V shape (n, d) → output (n, d).',
              trap: 'Value dimension d_v can differ from d_k in multi-head setups.',
            }),
          },
          {
            id: 'multi-head-sketch',
            label: 'Multi-head sketch',
            tooltip: tip({
              short: 'Run several attentions in parallel, then concatenate or project.',
              intuition: 'Different heads can specialize in different relation types.',
              example: 'h heads → h smaller Q/K/V triples → concat → W_O.',
              trap: 'Multi-head is still self-attention; it is not a different mixing rule.',
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'lookup-table-trap',
            label: 'Not a lookup table',
            tooltip: tip({
              short: 'Weights are recomputed from the current sequence.',
              intuition: 'There is no fixed “word A always attends to word B” table stored in the model.',
              example: 'The same token type can attend differently in two sentences.',
              trap: 'Memorizing one heatmap from one example misleads interpretation.',
            }),
          },
          {
            id: 'weight-equals-importance',
            label: 'Weight ≠ importance',
            tooltip: tip({
              short: 'A bright cell is query-specific, not a global salience score.',
              intuition: 'Interpretation and causality require more than one attention plot.',
              example: 'Head 3 may spike on punctuation while head 1 tracks entities.',
              trap: 'Do not use attention alone as an explanation method.',
            }),
          },
          {
            id: 'softmax-coupling',
            label: 'Softmax couples keys',
            tooltip: tip({
              short: 'Raising one score lowers relative mass on the other keys in the row.',
              intuition: 'Attention weights are competitive within each query row.',
              example: 'If one key becomes much larger, others shrink even if unchanged.',
              trap: 'Do not treat weights as independent per key.',
            }),
          },
          {
            id: 'order-blindness',
            label: 'Order blindness alone',
            tooltip: tip({
              short: 'Pure self-attention on embeddings is permutation-sensitive only through positions you add.',
              intuition: 'Swap token order without changing position signals and the set of pairwise scores changes structure.',
              example: '“dog bites man” needs positional encoding or RoPE to differ from “man bites dog”.',
              trap: 'Do not assume the model knows syntax without position information.',
            }),
          },
          {
            id: 'cross-attention-confusion',
            label: 'Cross-attention confusion',
            tooltip: tip({
              short: 'Self-attention reads from the same sequence; cross-attention reads another sequence.',
              intuition: 'Encoder-decoder models use both: self-attention inside each side, cross-attention between sides.',
              example: 'Decoder queries attend to encoder keys/values in translation.',
              trap: 'Calling every attention block “self-attention” blurs the data flow.',
            }),
          },
          {
            id: 'missing-scale',
            label: 'Missing √d_k scale',
            tooltip: tip({
              short: 'Large dimensions make dot products huge before softmax.',
              intuition: 'Scaling keeps gradients and softmax saturation healthier.',
              example: 'd_k = 64 → divide scores by 8.',
              trap: 'Training may partially compensate, but the scaled form is the standard definition.',
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'attention-masks',
            label: 'Attention masks',
            tooltip: tip({
              short: 'Masks remove illegal query-key pairs before softmax.',
              intuition: 'Causal decoding hides future keys; padding masks hide pad tokens.',
              example: 'Add -∞ to blocked cells so softmax assigns ~0 weight.',
              trap: 'Masks change visibility, not the definition of Q, K, and V.',
            }),
            lessonId: 'attention-masks',
          },
          {
            id: 'positional-encoding',
            label: 'Positional encoding',
            tooltip: tip({
              short: 'Inject order so identical tokens at different places differ.',
              intuition: 'Self-attention mixes content; positions tell the model where each token sat.',
              example: 'x_i + p_i before projections, or RoPE on Q and K.',
              trap: 'Position is complementary, not optional for word-order tasks.',
            }),
            lessonId: 'positional-encoding',
          },
          {
            id: 'transformer-stack',
            label: 'Transformer stack',
            tooltip: tip({
              short: 'Repeat attention, residuals, norms, and MLP blocks.',
              intuition: 'Attention mixes across tokens; MLP transforms each token; residuals preserve the stream.',
              example: 'One layer: self-attention → add & norm → MLP → add & norm.',
              trap: 'A transformer is not attention alone.',
            }),
            lessonId: 'transformer',
          },
          {
            id: 'kv-cache',
            label: 'KV cache',
            tooltip: tip({
              short: 'Store past keys and values during autoregressive decoding.',
              intuition: 'Only the new token needs fresh Q, K, V projections; old K/V are reused.',
              example: 'Step t attends over cached K/V from tokens 1…t.',
              trap: 'Caching saves projection work, not the need to attend.',
            }),
            lessonId: 'kv-cache',
          },
          {
            id: 'grouped-query-attention',
            label: 'Grouped-query attention',
            tooltip: tip({
              short: 'Share KV heads across multiple query heads to shrink cache.',
              intuition: 'Several queries read the same cached K/V with different projections.',
              example: '8 query heads may share 2 KV heads.',
              trap: 'Sharing reduces memory but can limit head specialization.',
            }),
            lessonId: 'grouped-query-attention',
          },
          {
            id: 'flash-attention',
            label: 'FlashAttention',
            tooltip: tip({
              short: 'Exact attention with less memory traffic.',
              intuition: 'Tile the QKᵀ and softmax math so the full n×n matrix never materializes in HBM.',
              example: 'Same formula, different execution schedule.',
              trap: 'It is not approximate attention.',
            }),
            lessonId: 'flash-attention',
          },
          {
            id: 'architecture-families',
            label: 'Encoder / decoder families',
            tooltip: tip({
              short: 'BERT-style encoders read both directions; GPT-style decoders use causal masks.',
              intuition: 'The attention block is similar; masks and objectives change what the model learns.',
              example: 'Encoder: bidirectional self-attention. Decoder: causal self-attention.',
              trap: 'Using the wrong mask for the task leaks answers or blocks needed context.',
            }),
            lessonId: 'transformer-architecture-families',
          },
        ],
      },
    ],
  },
'fundamental-subspaces': {
    center: {
      id: 'fundamental-subspaces',
      label: 'Fundamental Subspaces',
      type: 'current',
      tooltip: tip({
        short: 'Every matrix A splits its input and output worlds into four subspaces: row space and null space on the domain side, column space and left-null space on the codomain side.',
        intuition: 'Row space and column space describe what A can measure and reach; null spaces describe inputs erased to zero and outputs that can never be reached.',
        formula: 'dim Row(A)=dim Col(A)=r, dim Null(A)=n−r, dim Null(A^T)=m−r',
        why: 'These four spaces explain solvability of Ax=b, rank, projections, least squares, SVD, and why some directions are invisible or unreachable.',
        trap: 'Row space and column space usually live in different ambient spaces—even when they share the same dimension r.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'matrix-mult-fs',
            label: 'Matrix multiplication',
            tooltip: tip({
              short: 'Ax combines columns of A using entries of x.',
              intuition: 'Column space is exactly the set of all such combinations.',
              example: 'Col(A) = {Ax : x ∈ ℝⁿ}.',
              trap: 'Row operations and column operations answer different questions.',
            }),
            lessonId: 'matrix-multiplication',
          },
          {
            id: 'vector-fs',
            label: 'Vector',
            tooltip: tip({
              short: 'Inputs live in ℝⁿ; outputs live in ℝᵐ for an m×n matrix.',
              intuition: 'Subspaces are sets of vectors inside those spaces.',
              trap: 'Do not subtract vectors from different spaces.',
            }),
          },
          {
            id: 'linear-system-fs',
            label: 'Linear system Ax = b',
            tooltip: tip({
              short: 'Solvability asks whether b lies in the reachable output subspace.',
              intuition: 'Column space membership is the gate for exact solutions.',
              trap: 'Inconsistent b still allows a least-squares approximation.',
            }),
            lessonId: 'least-squares-projection',
          },
          {
            id: 'transpose-fs',
            label: 'Matrix transpose',
            tooltip: tip({
              short: 'A^T swaps rows and columns, linking domain and codomain views.',
              intuition: 'Row(A) in ℝⁿ pairs with Col(A^T); Col(A) pairs with Row(A^T).',
              trap: 'A^T b lives in ℝⁿ, not ℝᵐ.',
            }),
          },
          {
            id: 'span-independence',
            label: 'Span and independence',
            tooltip: tip({
              short: 'Rank r counts independent columns (and independent rows).',
              intuition: 'Bases for row and column spaces have exactly r vectors.',
              trap: 'Dependent columns shrink rank below min(m, n).',
            }),
          },
          {
            id: 'decomp-preview-fs',
            label: 'Decompositions preview',
            tooltip: tip({
              short: 'Factorizations expose rank, bases, and null directions explicitly.',
              intuition: 'RREF, QR, and SVD each reveal subspace structure differently.',
              trap: 'Numeric rank needs tolerance on tiny pivots or singular values.',
            }),
            lessonId: 'matrix-decompositions',
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'column-space',
            label: 'Column space Col(A)',
            tooltip: tip({
              short: 'All outputs Ax as x varies—every reachable b in exact solves.',
              intuition: 'Span of the columns of A inside ℝᵐ.',
              formula: 'Col(A) ⊆ ℝᵐ',
              trap: 'b ∉ Col(A) means no exact solution to Ax=b.',
            }),
          },
          {
            id: 'null-space',
            label: 'Null space Null(A)',
            tooltip: tip({
              short: 'All inputs x with Ax = 0.',
              intuition: 'Directions the matrix completely erases.',
              formula: 'Null(A) = {x : Ax = 0}',
              trap: 'Nonzero null space means solutions are not unique.',
            }),
          },
          {
            id: 'row-space',
            label: 'Row space Row(A)',
            tooltip: tip({
              short: 'Span of rows of A, a subspace of ℝⁿ.',
              intuition: 'Inputs that matter for output; orthogonal complement to Null(A) in standard inner product.',
              trap: 'Row space lives in input space, not output space.',
            }),
          },
          {
            id: 'left-null-space',
            label: 'Left-null space Null(A^T)',
            tooltip: tip({
              short: 'All y with A^T y = 0—constraints on outputs.',
              intuition: 'Orthogonal complement to Col(A) in ℝᵐ.',
              formula: 'Null(A^T) ⊥ Col(A)',
              trap: 'Left-null vectors are outputs, not inputs.',
            }),
          },
          {
            id: 'rank-nullity-theorem',
            label: 'Rank–nullity theorem',
            tooltip: tip({
              short: 'n = rank(A) + dim Null(A); m = rank(A) + dim Null(A^T).',
              intuition: 'Independent directions plus erased directions fill the ambient dimension.',
              formula: 'rank(A) + nullity(A) = n',
              trap: 'Rank counts both row and column independence simultaneously.',
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'domain-codomain-split',
            label: 'Domain vs codomain split',
            tooltip: tip({
              short: 'Two subspaces live in ℝⁿ (row, null); two live in ℝᵐ (column, left-null).',
              intuition: 'Think “input side” versus “output side” of the map.',
              trap: 'Do not plot all four in one axis system without labeling spaces.',
            }),
          },
          {
            id: 'reachable-outputs',
            label: 'Reachable outputs',
            tooltip: tip({
              short: 'Column space is the target subspace for exact hits on b.',
              intuition: 'Least squares projects b onto Col(A) when b is outside.',
              trap: 'Near membership with noise still fails exact solve tests.',
            }),
          },
          {
            id: 'invisible-inputs',
            label: 'Invisible inputs',
            tooltip: tip({
              short: 'Adding a null-space vector to x does not change Ax.',
              intuition: 'Null directions are blind spots of the measurement.',
              trap: 'Many x can yield the same Ax when nullity > 0.',
            }),
          },
          {
            id: 'consistency-constraint',
            label: 'Consistency constraint',
            tooltip: tip({
              short: 'b must be orthogonal to left-null space for Ax=b to exist.',
              intuition: 'Left-null directions detect impossible output components.',
              trap: 'Numerical tolerance matters when checking orthogonality.',
            }),
          },
          {
            id: 'four-spaces-orthogonality',
            label: 'Orthogonal pairs',
            tooltip: tip({
              short: 'Row(A) ⊥ Null(A) in ℝⁿ; Col(A) ⊥ Null(A^T) in ℝᵐ.',
              intuition: 'Each side splits into “seen” and “invisible” directions.',
              trap: 'Orthogonality uses the standard dot product unless weighted.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'dimension-formulas',
            label: 'Dimension formulas',
            tooltip: tip({
              short: 'dim Row = dim Col = r; nullities fill remaining n−r and m−r.',
              intuition: 'Rank is the shared thread across all four counts.',
              formula: 'r = rank(A)',
              trap: 'Full column rank (r=n) gives trivial Null(A).',
            }),
          },
          {
            id: 'solvability-test',
            label: 'Solvability test',
            tooltip: tip({
              short: 'Ax=b solvable iff b ∈ Col(A) iff b ⊥ Null(A^T).',
              intuition: 'Augment [A|b] and check for inconsistent rows in RREF.',
              trap: 'Floating-point elimination can hide tiny inconsistencies.',
            }),
          },
          {
            id: 'rref-bases',
            label: 'RREF bases',
            tooltip: tip({
              short: 'Pivot columns of A span Col(A); special RREF rows span Row(A).',
              intuition: 'Free variables parameterize Null(A).',
              trap: 'RREF bases depend on elimination choices unless standardized.',
            }),
          },
          {
            id: 'projection-link-fs',
            label: 'Projection link',
            tooltip: tip({
              short: 'Least-squares ŷ is the projection of b onto Col(A).',
              intuition: 'Residual lies in Null(A^T), the orthogonal complement of Col(A).',
              trap: 'Projection geometry lives in output space ℝᵐ.',
            }),
            lessonId: 'least-squares-projection',
          },
          {
            id: 'numpy-rank-null',
            label: 'Rank in code',
            tooltip: tip({
              short: 'Libraries expose rank, null space bases, and orthonormal splits.',
              intuition: 'SVD gives stable numerical rank and subspace bases.',
              code: 'r = np.linalg.matrix_rank(A)\nU, s, Vt = np.linalg.svd(A, full_matrices=True)',
              trap: 'Set tolerance explicitly for rank near singular cases.',
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'same-space-trap',
            label: 'Same ambient space trap',
            tooltip: tip({
              short: 'Row(A) and Col(A) are not generally subspaces of the same ℝⁿ unless m=n.',
              intuition: 'Dimensions m and n can differ for rectangular A.',
              trap: 'Statements like “b in Row(A)” confuse input and output roles.',
            }),
          },
          {
            id: 'null-not-error',
            label: 'Null is not “error”',
            tooltip: tip({
              short: 'Null space describes structural non-uniqueness, not bad data.',
              intuition: 'Many fitting problems intentionally have null directions.',
              trap: 'Do not treat nullity as a bug without context.',
            }),
          },
          {
            id: 'rank-vs-full-rank',
            label: 'Rank vs full rank',
            tooltip: tip({
              short: 'Full column rank means Null(A)={0}; full row rank means Null(A^T)={0}.',
              intuition: 'Rectangular full rank is one-sided.',
              trap: 'Square full rank means invertible—stronger than rectangular full rank.',
            }),
          },
          {
            id: 'left-null-ignored',
            label: 'Ignoring left-null',
            tooltip: tip({
              short: 'Forgetting Null(A^T) misses why some b are impossible.',
              intuition: 'Consistency is a subspace membership question.',
              trap: 'Least squares handles inconsistent b but changes the problem.',
            }),
          },
          {
            id: 'numeric-rank-trap',
            label: 'Numeric rank trap',
            tooltip: tip({
              short: 'Tiny pivots or σ can inflate rank on paper but not in data.',
              intuition: 'Set ε relative to largest pivot or σ₁.',
              trap: 'Exact RREF rank differs from stable numerical rank.',
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'least-squares-fs',
            label: 'Least squares projection',
            tooltip: tip({
              short: 'Fit uses Col(A); residual orthogonal to Col(A).',
              intuition: 'Fundamental spaces explain why normal equations encode orthogonality.',
              trap: 'Approximate b outside Col(A) still has geometric meaning.',
            }),
            lessonId: 'least-squares-projection',
          },
          {
            id: 'pseudoinverse-fs',
            label: 'Pseudoinverse',
            tooltip: tip({
              short: 'Rank-deficient solves use subspaces to pick minimum-norm least-squares x.',
              intuition: 'Null(A) directions are free; pseudoinverse picks the smallest norm.',
              trap: 'Many x solve projected equations when nullity > 0.',
            }),
            lessonId: 'pseudoinverse',
          },
          {
            id: 'svd-fs',
            label: 'SVD',
            tooltip: tip({
              short: 'Singular vectors span row/column spaces; zero σ reveal null directions.',
              intuition: 'SVD makes all four spaces visible simultaneously.',
              trap: 'Truncating tiny σ changes the effective subspaces.',
            }),
            lessonId: 'svd',
          },
          {
            id: 'projection-matrices-fs',
            label: 'Projection matrices',
            tooltip: tip({
              short: 'Orthogonal projectors onto Col(A) encode least-squares geometry.',
              intuition: 'P maps b to its Col(A) component; I−P maps to Null(A^T).',
              trap: 'Projectors depend on subspace and inner product.',
            }),
            lessonId: 'projection-matrices',
          },
          {
            id: 'eigenvalue-fs',
            label: 'Eigenvalues',
            tooltip: tip({
              short: 'Invariant directions refine subspace stories for square maps.',
              intuition: 'Eigenspaces are null spaces of A−λI for special λ.',
              trap: 'Eigenstructure requires square A; use SVD more generally.',
            }),
            lessonId: 'eigenvalue',
          },
        ],
      },
    ],
  },
  eigenvalue: {
    center: {
      id: 'eigenvalue',
      label: 'Eigenvalues',
      type: 'current',
      tooltip: tip({
        short: 'An eigenvector v satisfies Av = λv: the matrix stretches or flips v without rotating it off its line.',
        intuition: 'Eigenvalues are the stretch factors; eigenvectors are the special directions that survive the transform unchanged in direction.',
        formula: 'Av = λv, det(A − λI) = 0',
        why: 'Eigenstructure powers dynamics, PCA, graph spectra, stability analysis, and symmetric matrix geometry.',
        trap: 'Not every square matrix has enough real eigenvectors to diagonalize cleanly.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'matrix-mult-ev',
            label: 'Matrix multiplication',
            tooltip: tip({
              short: 'Av applies the linear map to candidate directions v.',
              intuition: 'Eigenanalysis asks which v only scale under this map.',
              trap: 'Zero vector is never a meaningful eigenvector.',
            }),
            lessonId: 'matrix-multiplication',
          },
          {
            id: 'vector-ev',
            label: 'Vector direction',
            tooltip: tip({
              short: 'Eigenvectors are defined up to nonzero scaling.',
              intuition: 'Only the line through v matters, not its length.',
              trap: 'Normalizing v does not change λ.',
            }),
          },
          {
            id: 'linear-map-ev',
            label: 'Linear transformation',
            tooltip: tip({
              short: 'Square A represents a map from ℝⁿ to itself.',
              intuition: 'Eigen directions are invariant lines of that map.',
              trap: 'Rectangular A needs SVD, not classical eigenvalues of A.',
            }),
          },
          {
            id: 'fundamental-subspaces-ev',
            label: 'Fundamental subspaces',
            tooltip: tip({
              short: 'Eigenspace for λ is Null(A − λI).',
              intuition: 'Zero eigenvalues connect to null space geometry.',
              trap: 'Multiple λ give different eigenspaces that may not span ℝⁿ.',
            }),
            lessonId: 'fundamental-subspaces',
          },
          {
            id: 'determinant-preview',
            label: 'Determinant preview',
            tooltip: tip({
              short: 'det(A − λI) = 0 selects eigenvalues.',
              intuition: 'Singular shift A−λI means a nontrivial null direction exists.',
              trap: 'Characteristic polynomial degree is n for n×n A.',
            }),
            lessonId: 'determinant-volume',
          },
          {
            id: 'change-of-basis-preview',
            label: 'Change of basis preview',
            tooltip: tip({
              short: 'Diagonalization rewrites A in an eigenvector coordinate system.',
              intuition: 'In eigen coordinates the map becomes simple scaling.',
              trap: 'Missing eigenvectors means no full diagonalization.',
            }),
            lessonId: 'change-of-basis',
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'eigenvalue-equation',
            label: 'Eigenvalue equation',
            tooltip: tip({
              short: 'Av = λv with v ≠ 0.',
              intuition: 'Apply A once: same direction, new length |λ| times (sign if λ<0).',
              formula: 'Av = λv',
              trap: 'λ=0 eigenvectors lie in Null(A).',
            }),
          },
          {
            id: 'characteristic-polynomial',
            label: 'Characteristic polynomial',
            tooltip: tip({
              short: 'Roots of det(A − λI) are eigenvalues (with multiplicity).',
              intuition: 'Each root λ yields at least one eigenvector if algebra allows.',
              formula: 'p(λ) = det(A − λI)',
              trap: 'Complex roots appear even for real A.',
            }),
          },
          {
            id: 'eigenspace',
            label: 'Eigenspace',
            tooltip: tip({
              short: 'All v with Av = λv form a subspace for fixed λ.',
              intuition: 'Geometric multiplicity counts independent eigen directions.',
              trap: 'Repeated λ can still have multiple independent eigenvectors.',
            }),
          },
          {
            id: 'diagonalization',
            label: 'Diagonalization',
            tooltip: tip({
              short: 'If A has n independent eigenvectors, A = VΛV⁻¹.',
              intuition: 'V changes basis; Λ scales along eigen axes.',
              formula: 'A = VΛV^{-1}',
              trap: 'Defective matrices lack a full eigenbasis.',
            }),
          },
          {
            id: 'symmetric-case',
            label: 'Symmetric case',
            tooltip: tip({
              short: 'Real symmetric A has real eigenvalues and orthogonal eigenvectors.',
              intuition: 'A = QΛQ^T with Q orthogonal—spectral theorem.',
              formula: 'A = QΛQ^T',
              trap: 'Symmetry is stronger than mere square real A.',
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'stretch-directions',
            label: 'Stretch directions',
            tooltip: tip({
              short: 'Each eigenvector axis scales by its eigenvalue.',
              intuition: 'Decompose motion into independent axis stretches.',
              trap: 'Negative λ flips direction while scaling magnitude.',
            }),
          },
          {
            id: 'ellipse-axes',
            label: 'Ellipse axes picture',
            tooltip: tip({
              short: 'For 2×2 maps, eigenvectors align with ellipse principal axes.',
              intuition: 'Eigenvalues set axis lengths after the transform.',
              trap: 'Shear without symmetric structure skews this picture.',
            }),
          },
          {
            id: 'invariance',
            label: 'Invariant subspaces',
            tooltip: tip({
              short: 'Eigenspaces are invariant: A sends them to themselves.',
              intuition: 'Power A^k acts as λ^k on each eigen direction.',
              trap: 'General invariant subspaces need not be single eigenspaces.',
            }),
          },
          {
            id: 'complex-eigenvalues',
            label: 'Complex eigenvalues',
            tooltip: tip({
              short: 'Rotation-like planar components appear as conjugate complex pairs.',
              intuition: 'Real trajectories can combine complex eigenmodes.',
              trap: 'Real-only eigenvectors may not span ℝⁿ.',
            }),
          },
          {
            id: 'dominant-mode',
            label: 'Dominant mode',
            tooltip: tip({
              short: 'Largest |λ| often dominates long-run behavior of A^k v.',
              intuition: 'Spectral radius drives growth or decay.',
              trap: 'Initial v orthogonal to dominant eigenspace hides it temporarily.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'characteristic-formula',
            label: 'det(A − λI) = 0',
            tooltip: tip({
              short: 'Solve the degree-n polynomial for eigenvalues.',
              intuition: 'Small n allows hand factorization; large n uses iterative methods.',
              formula: 'det(A - λI) = 0',
              trap: 'Floating roots of high-degree polynomials are ill-conditioned.',
            }),
          },
          {
            id: 'power-iteration',
            label: 'Power iteration',
            tooltip: tip({
              short: 'Repeated multiplication A^k v estimates dominant eigenvector.',
              intuition: 'Largest |λ| wins if v has a component along it.',
              code: 'v = A @ v; v = v / np.linalg.norm(v)',
              trap: 'Convergence rate depends on gap between |λ₁| and |λ₂|.',
            }),
          },
          {
            id: 'rayleigh-quotient',
            label: 'Rayleigh quotient',
            tooltip: tip({
              short: 'R(v) = v^T A v / v^T v approximates λ when v is near an eigenvector.',
              intuition: 'Stationary at exact eigenvectors for symmetric A.',
              formula: 'R(v) = (v^TAv)/(v^Tv)',
              trap: 'For nonsymmetric A, quotient need not equal an eigenvalue.',
            }),
          },
          {
            id: 'numpy-eig',
            label: 'numpy.linalg.eig',
            tooltip: tip({
              short: 'Returns eigenvalues and column eigenvectors for square A.',
              intuition: 'Columns of V satisfy A @ V[:,i] ≈ λ[i] * V[:,i].',
              code: 'w, V = np.linalg.eig(A)',
              trap: 'Verify ordering; libraries do not guarantee sorted λ.',
            }),
          },
          {
            id: 'pca-link-ev',
            label: 'PCA covariance link',
            tooltip: tip({
              short: 'PCA eigenvectors of covariance are variance axes.',
              intuition: 'Largest covariance eigenvalue = most variance direction.',
              trap: 'Covariance eigenvalues are not singular values of raw X without scaling.',
            }),
            lessonId: 'pca',
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'always-diagonalizable-trap',
            label: 'Always diagonalizable trap',
            tooltip: tip({
              short: 'Defective matrices lack a full set of eigenvectors.',
              intuition: 'Jordan blocks handle missing eigenvectors.',
              trap: 'A = VΛV⁻¹ requires invertible V.',
            }),
          },
          {
            id: 'eigenvector-uniqueness',
            label: 'Eigenvector uniqueness',
            tooltip: tip({
              short: 'Scaling and sign do not define a unique eigenvector.',
              intuition: 'Only the line direction is intrinsic.',
              trap: 'Numerical eigen solvers may flip signs between runs.',
            }),
          },
          {
            id: 'algebraic-geometric',
            label: 'Multiplicity confusion',
            tooltip: tip({
              short: 'Algebraic multiplicity (root degree) can exceed geometric (dimension of eigenspace).',
              intuition: 'Repeated eigenvalues may still lack enough vectors.',
              trap: 'Equal algebraic multiplicities do not guarantee diagonalizability.',
            }),
          },
          {
            id: 'eigen-vs-singular',
            label: 'Eigen vs singular values',
            tooltip: tip({
              short: 'Singular values of A are not the same as eigenvalues unless special structure.',
              intuition: 'Use SVD for general rectangular maps.',
              trap: 'σᵢ(A) = √λᵢ(A^T A) links but does not identify eigenvalues of A.',
            }),
            lessonId: 'svd',
          },
          {
            id: 'magnitude-trap',
            label: '|λ| vs importance',
            tooltip: tip({
              short: 'Small λ can still matter for sensitive initial directions.',
              intuition: 'All modes contribute unless projections vanish.',
              trap: 'Dropping small λ in simulation can destabilize long horizons.',
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'pca-ev',
            label: 'PCA',
            tooltip: tip({
              short: 'Principal components are eigenvectors of covariance.',
              intuition: 'Variance ranking follows descending eigenvalues.',
              trap: 'Center data before covariance eigenanalysis.',
            }),
            lessonId: 'pca',
          },
          {
            id: 'svd-ev',
            label: 'SVD',
            tooltip: tip({
              short: 'For symmetric A, singular values equal |eigenvalues| with aligned vectors.',
              intuition: 'SVD generalizes spectral ideas to rectangular A.',
              trap: 'Always verify whether you need eigen or singular structure.',
            }),
            lessonId: 'svd',
          },
          {
            id: 'matrix-decomp-ev',
            label: 'Matrix decompositions',
            tooltip: tip({
              short: 'Eigendecomposition is one factorization among LU, QR, SVD, Cholesky.',
              intuition: 'Pick eigen when powers, dynamics, or symmetric geometry matter.',
              trap: 'Stability often favors QR/SVD over explicit VΛV⁻¹.',
            }),
            lessonId: 'matrix-decompositions',
          },
          {
            id: 'condition-number-ev',
            label: 'Condition number',
            tooltip: tip({
              short: 'κ(A) for symmetric A relates extreme eigenvalue magnitudes.',
              intuition: 'Nearly dependent eigen directions amplify input noise.',
              trap: 'Nonsymmetric conditioning uses singular values, not raw eigenvalues.',
            }),
            lessonId: 'condition-number',
          },
          {
            id: 'low-rank-ev',
            label: 'Low-rank approximation',
            tooltip: tip({
              short: 'Truncating small eigen/singular modes compresses data.',
              intuition: 'Dominant eigen directions capture most energy.',
              trap: 'Truncation removes signal and noise together.',
            }),
            lessonId: 'low-rank-approximation',
          },
        ],
      },
    ],
  },
  'qr-decomposition': {
    center: {
      id: 'qr-decomposition',
      label: 'QR Decomposition',
      type: 'current',
      tooltip: tip({
        short: 'QR factorization writes A = QR with Q having orthonormal columns and R upper triangular—orthogonal directions plus triangular weights.',
        intuition: 'Q gives a stable orthonormal basis for Col(A); R records how original columns combine those basis vectors.',
        formula: 'A = QR, Q^TQ = I, R upper triangular',
        why: 'QR is the workhorse for least squares, eigenvalue algorithms, and numerically stable solves when normal equations fail.',
        trap: 'Q is not the original matrix with scaled columns—it is a new orthonormal basis for the same column space.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'matrix-mult-qr',
            label: 'Matrix multiplication',
            tooltip: tip({
              short: 'A = QR is a product of two structured matrices.',
              intuition: 'Column j of A is a linear combination of Q columns weighted by R entries.',
              trap: 'Inner dimensions must chain: Q is m×k, R is k×n with k = rank or m.',
            }),
            lessonId: 'matrix-multiplication',
          },
          {
            id: 'column-space-qr',
            label: 'Column space',
            tooltip: tip({
              short: 'Col(A) = Col(Q) when Q spans the same columns.',
              intuition: 'QR replaces messy columns with perpendicular axes.',
              trap: 'Economy QR drops redundant Q columns for rank-deficient A.',
            }),
            lessonId: 'fundamental-subspaces',
          },
          {
            id: 'least-squares-qr',
            label: 'Least squares preview',
            tooltip: tip({
              short: 'Stable least squares avoids forming A^T A explicitly.',
              intuition: 'Project b through Q^T then back-solve R.',
              trap: 'Normal equations square the condition number.',
            }),
            lessonId: 'least-squares-projection',
          },
          {
            id: 'orthogonality-qr',
            label: 'Orthogonal vectors',
            tooltip: tip({
              short: 'Columns of Q satisfy q_i^T q_j = 0 for i≠j and ||q_i||=1.',
              intuition: 'Orthogonality prevents numerical collapse in elimination.',
              trap: 'Nearly dependent columns make classical Gram-Schmidt fragile.',
            }),
          },
          {
            id: 'linear-independence-qr',
            label: 'Linear independence',
            tooltip: tip({
              short: 'Independent columns yield nonzero diagonal entries in R.',
              intuition: 'Dependent columns produce zero rows in R under economy factorization.',
              trap: 'Rank deficiency changes Q and R shapes.',
            }),
          },
          {
            id: 'triangular-qr',
            label: 'Upper triangular R',
            tooltip: tip({
              short: 'R entries below diagonal are zero; back substitution solves Rx = c.',
              intuition: 'Triangular structure makes solves cheap.',
              trap: 'Zero diagonal in R signals rank loss.',
            }),
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'gram-schmidt',
            label: 'Gram–Schmidt orthogonalization',
            tooltip: tip({
              short: 'Subtract projections of earlier columns to make each new column orthogonal.',
              intuition: 'Each step removes components already explained by previous q vectors.',
              trap: 'Classical Gram-Schmidt without reorthogonalization loses orthogonality in float.',
            }),
          },
          {
            id: 'build-q',
            label: 'Build Q',
            tooltip: tip({
              short: 'Normalize orthogonalized columns to unit length.',
              intuition: 'Q columns form an orthonormal basis for Col(A).',
              formula: 'Q^TQ = I',
              trap: 'Sign flips make Q non-unique.',
            }),
          },
          {
            id: 'build-r',
            label: 'Build R',
            tooltip: tip({
              short: 'R_ij = q_i^T a_j for i ≤ j; zeros below diagonal.',
              intuition: 'R stores projection coefficients from original columns onto Q.',
              trap: 'R is not A with scaled entries—it encodes the QR rewrite.',
            }),
          },
          {
            id: 'least-squares-via-qr',
            label: 'Least squares via QR',
            tooltip: tip({
              short: 'Minimize ‖b−Ax‖ by solving R x = Q^T b (when A has full column rank).',
              intuition: 'Q^T b is coordinates of b in the Q basis; R maps coefficients to A-columns.',
              formula: 'Rx = Q^Tb',
              trap: 'Rectangular or rank-deficient A needs pivoted or truncated QR.',
            }),
          },
          {
            id: 'householder-alternative',
            label: 'Householder reflections',
            tooltip: tip({
              short: 'Production QR often uses reflectors instead of Gram-Schmidt.',
              intuition: 'Reflections zero subdiagonal blocks stably.',
              trap: 'Algorithm choice matters more than memorizing one hand procedure.',
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'clean-axes',
            label: 'Clean perpendicular axes',
            tooltip: tip({
              short: 'Q columns are unit-length and mutually perpendicular.',
              intuition: 'Like replacing skewed rulers with a square grid.',
              trap: 'Original A columns may be far from orthogonal.',
            }),
          },
          {
            id: 'triangular-solve',
            label: 'Triangular solve picture',
            tooltip: tip({
              short: 'After Q^T, the hard part reduces to back substitution on R.',
              intuition: 'Last unknown solved first, moving upward.',
              trap: 'Ill-conditioned R still hurts accuracy.',
            }),
          },
          {
            id: 'projection-via-q',
            label: 'Projection via Q',
            tooltip: tip({
              short: 'Q Q^T projects onto Col(A) when Q spans that space.',
              intuition: 'Least-squares fitted vector uses this projector.',
              trap: 'Full Q versus economy Q changes projector dimensions.',
            }),
          },
          {
            id: 'economy-qr',
            label: 'Economy QR',
            tooltip: tip({
              short: 'Keep only r orthonormal columns when rank(A)=r.',
              intuition: 'Avoids storing redundant Q columns.',
              trap: 'Thin Q still spans Col(A) if chosen correctly.',
            }),
          },
          {
            id: 'stability-intuition',
            label: 'Stability intuition',
            tooltip: tip({
              short: 'QR avoids squaring A into A^T A.',
              intuition: 'Condition number of R is roughly that of A, not its square.',
              trap: 'Stability helps but does not fix rank-deficient noise.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'factorization-formula',
            label: 'A = QR',
            tooltip: tip({
              short: 'Full or economy shapes depending on m, n, and rank.',
              intuition: 'Multiply Q and R to reconstruct A exactly (up to float).',
              formula: 'A = QR',
              trap: 'Verify Q^TQ ≈ I numerically after computation.',
            }),
          },
          {
            id: 'qtb-step',
            label: 'Compute Q^T b',
            tooltip: tip({
              short: 'Project target b onto each Q direction.',
              intuition: 'These coordinates feed the triangular solve.',
              code: 'c = Q.T @ b',
              trap: 'Use Q from the same factorization as R.',
            }),
          },
          {
            id: 'backsolve-r',
            label: 'Back-solve R x = c',
            tooltip: tip({
              short: 'Solve upper triangular system from bottom row upward.',
              intuition: 'Each x_i depends only on x_{i+1}, …, x_n.',
              code: 'x = np.linalg.solve(R, c)',
              trap: 'Near-zero R_ii means sensitive solution.',
            }),
          },
          {
            id: 'numpy-qr',
            label: 'numpy.linalg.qr',
            tooltip: tip({
              short: 'mode="reduced" returns economy Q and R.',
              intuition: 'Householder backend is typical in NumPy/SciPy.',
              code: 'Q, R = np.linalg.qr(A, mode="reduced")',
              trap: 'Sign and column ordering may differ from textbook examples.',
            }),
          },
          {
            id: 'lstsq-qr-link',
            label: 'lstsq connection',
            tooltip: tip({
              short: 'np.linalg.lstsq often uses QR or SVD internally.',
              intuition: 'Prefer library solves over manual A^T A.',
              trap: 'Inspect residuals, not only coefficients.',
            }),
            lessonId: 'least-squares-projection',
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'classical-gs-trap',
            label: 'Classical Gram–Schmidt trap',
            tooltip: tip({
              short: 'Without reorthogonalization, Q columns drift from perpendicular.',
              intuition: 'Modified Gram-Schmidt or Householder fixes this.',
              trap: 'Hand GS on nearly dependent columns fails silently.',
            }),
          },
          {
            id: 'q-not-a-trap',
            label: 'Q is not A',
            tooltip: tip({
              short: 'Q columns are orthonormal combinations of A columns, not scaled originals.',
              intuition: 'R holds the combination weights.',
              trap: 'Do not read Q entries as feature importances.',
            }),
          },
          {
            id: 'sign-nonunique',
            label: 'Sign non-uniqueness',
            tooltip: tip({
              short: 'Flipping a Q column and R row sign leaves A unchanged.',
              intuition: 'Compare factorizations up to signs.',
              trap: 'Tests must not assume fixed signs.',
            }),
          },
          {
            id: 'normal-equations-trap-qr',
            label: 'Normal equations trap',
            tooltip: tip({
              short: 'A^T A x = A^T b can be much less stable than QR solve.',
              intuition: 'κ(A^T A) ≈ κ(A)².',
              trap: 'Correct formula on paper can fail in float.',
            }),
          },
          {
            id: 'rank-deficient-qr',
            label: 'Rank-deficient QR',
            tooltip: tip({
              short: 'Zero diagonal entries in R signal dependent columns.',
              intuition: 'Pivoted QR or SVD pseudoinverse handles deficiency.',
              trap: 'Plain back substitution breaks on zero pivots.',
            }),
            lessonId: 'pseudoinverse',
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'least-squares-used-qr',
            label: 'Least squares projection',
            tooltip: tip({
              short: 'QR is the standard stable route to least-squares solutions.',
              intuition: 'Projection geometry matches Q Q^T on Col(A).',
              trap: 'Still inspect residual norm for model mismatch.',
            }),
            lessonId: 'least-squares-projection',
          },
          {
            id: 'eigen-qr-algorithm',
            label: 'Eigenvalue algorithms',
            tooltip: tip({
              short: 'QR iteration builds eigenvalues for many matrices.',
              intuition: 'Repeated QR on shifts converges toward Schur form.',
              trap: 'Full eigen story needs more than one QR factorization.',
            }),
            lessonId: 'eigenvalue',
          },
          {
            id: 'svd-qr-relation',
            label: 'SVD relationship',
            tooltip: tip({
              short: 'Bidiagonalization plus QR ideas underpin SVD computation.',
              intuition: 'Both seek orthogonal structure before diagonal scaling.',
              trap: 'SVD solves broader problems than QR alone.',
            }),
            lessonId: 'svd',
          },
          {
            id: 'condition-qr',
            label: 'Condition number',
            tooltip: tip({
              short: 'R diagonal reflects sensitivity of triangular solve.',
              intuition: 'Tiny |R_ii| warns of ill-conditioned column directions.',
              trap: 'Scaling columns of A changes R without changing solution set properly.',
            }),
            lessonId: 'condition-number',
          },
          {
            id: 'linear-regression-qr',
            label: 'Linear regression',
            tooltip: tip({
              short: 'Design-matrix least squares is QR-friendly in practice.',
              intuition: 'Feature columns become A; QR avoids explicit X^T X.',
              trap: 'Feature scaling still affects conditioning.',
            }),
            lessonId: 'linear-regression',
          },
        ],
      },
    ],
  },
  'change-of-basis': {
    center: {
      id: 'change-of-basis',
      label: 'Change of Basis',
      type: 'current',
      tooltip: tip({
        short: 'A vector stays the same geometric object while its coordinate numbers change when you express it in a different basis.',
        intuition: 'The basis is the ruler; coordinates are measurements. Swap rulers and the numbers rewrite, not the arrow in space.',
        formula: 'v = P c,\\quad c = P^{-1} v',
        why: 'Change of basis explains diagonalization, PCA rotations, graphics frames, and why similar matrices describe the same map in new coordinates.',
        trap: 'Coordinates are not the vector—two different number tuples can describe the same v in different bases.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'vector-cob',
            label: 'Vector',
            tooltip: tip({
              short: 'A vector is a geometric object; coordinates are one representation of it.',
              intuition: 'Think “fixed arrow” versus “address written in a chosen language.”',
              trap: 'Never identify v with its coordinate column without naming the basis.',
            }),
          },
          {
            id: 'matrix-mult-cob',
            label: 'Matrix multiplication',
            tooltip: tip({
              short: 'v = P c combines basis columns of P using entries of c.',
              intuition: 'Each coordinate weights one basis direction.',
              trap: 'Column order of P defines which entry of c attaches to which basis vector.',
            }),
            lessonId: 'matrix-multiplication',
          },
          {
            id: 'basis-independence-cob',
            label: 'Basis and independence',
            tooltip: tip({
              short: 'A basis is an ordered set of independent vectors that span the space.',
              intuition: 'Independent columns make P invertible so coordinates are unique.',
              trap: 'Dependent columns mean many c map to the same v or none at all.',
            }),
          },
          {
            id: 'linear-map-cob',
            label: 'Linear transformation',
            tooltip: tip({
              short: 'A matrix A represents a fixed linear map; basis choice changes its coordinate formula.',
              intuition: 'The map acts on vectors; only its matrix entries depend on coordinates.',
              trap: 'Do not confuse changing coordinates with changing the underlying map.',
            }),
          },
          {
            id: 'inverse-preview-cob',
            label: 'Matrix inverse preview',
            tooltip: tip({
              short: 'Recover coordinates with c = P^{-1} v when P is square and invertible.',
              intuition: 'Inverse undoes the basis assembly v = P c.',
              trap: 'Non-invertible P means the claimed set is not a basis of the full space.',
            }),
            lessonId: 'matrix-decompositions',
          },
          {
            id: 'eigen-preview-cob',
            label: 'Eigenstructure preview',
            tooltip: tip({
              short: 'Diagonalization uses an eigenvector basis so A becomes simple scaling.',
              intuition: 'Eigenvectors supply a natural coordinate system for the map.',
              trap: 'Missing eigenvectors means no full diagonalizing basis.',
            }),
            lessonId: 'eigenvalue',
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'basis-matrix-p',
            label: 'Basis matrix P',
            tooltip: tip({
              short: 'Columns of P are the new basis vectors written in the old coordinates.',
              intuition: 'P assembles v from coordinate weights c.',
              formula: 'v = P c',
              trap: 'Column order is part of the basis definition.',
            }),
          },
          {
            id: 'coordinate-vector-c',
            label: 'Coordinate vector c',
            tooltip: tip({
              short: 'Entries of c tell how much of each basis column to combine.',
              intuition: 'c is the address of v in the P basis.',
              trap: 'The same geometric v gets different c in different bases.',
            }),
          },
          {
            id: 'forward-map-cob',
            label: 'Forward map v = P c',
            tooltip: tip({
              short: 'Multiply basis matrix by coordinates to recover the standard representation.',
              intuition: 'Linear combination of basis columns.',
              formula: 'v = \\sum_i c_i p_i',
              trap: 'P must use the same ambient space as v.',
            }),
          },
          {
            id: 'inverse-map-cob',
            label: 'Inverse map c = P^{-1} v',
            tooltip: tip({
              short: 'Solve for coordinates that reproduce v in the P basis.',
              intuition: 'Invert the assembly step when P is invertible.',
              formula: 'c = P^{-1} v',
              trap: 'Ill-conditioned P makes coordinate recovery numerically unstable.',
            }),
          },
          {
            id: 'similarity-transform',
            label: 'Similarity transform',
            tooltip: tip({
              short: 'The same linear map in new coordinates: A′ = P^{-1} A P.',
              intuition: 'Change coordinates on input and output, then express A in the new system.',
              formula: "A' = P^{-1} A P",
              trap: 'Similarity preserves eigenvalues but not individual matrix entries.',
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'same-arrow-new-address',
            label: 'Same arrow, new address',
            tooltip: tip({
              short: 'Geometry is invariant; only the numeric recipe for building v changes.',
              intuition: 'Like describing a location in street numbers versus GPS—same place, different labels.',
              trap: 'Comparing c vectors from different bases without conversion is meaningless.',
            }),
          },
          {
            id: 'rotated-graph-paper',
            label: 'Rotated graph paper',
            tooltip: tip({
              short: 'An orthonormal rotation of axes changes coordinates but preserves lengths and angles.',
              intuition: 'Orthonormal P makes coordinate geometry match standard dot products.',
              trap: 'Non-orthogonal bases skew length and angle formulas unless you use the metric P^T P.',
            }),
          },
          {
            id: 'diagonalization-picture',
            label: 'Diagonalization picture',
            tooltip: tip({
              short: 'In an eigenbasis, A acts by separate scalings on each axis.',
              intuition: 'P columns are eigenvectors; P^{-1} A P is diagonal when enough eigenvectors exist.',
              trap: 'Defective matrices lack a full eigenbasis for clean diagonal coordinates.',
            }),
          },
          {
            id: 'passive-vs-active',
            label: 'Passive vs active view',
            tooltip: tip({
              short: 'Passive: relabel coordinates. Active: move vectors/points in space.',
              intuition: 'Change-of-basis formulas usually describe passive relabeling of one fixed map.',
              trap: 'Graphics “transform the object” vs “transform the camera” swap P and P^{-1} roles.',
            }),
          },
          {
            id: 'orthonormal-simplification',
            label: 'Orthonormal simplification',
            tooltip: tip({
              short: 'When basis columns are orthonormal, P^{-1} = P^T.',
              intuition: 'Coordinate recovery becomes a cheap projection dot-product step.',
              trap: 'Applying P^T = P^{-1} without orthonormality gives wrong coordinates.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'assembly-formula',
            label: 'v = P c',
            tooltip: tip({
              short: 'Standard representation from basis columns and coordinates.',
              intuition: 'Matrix–vector multiply is the coordinate assembly recipe.',
              formula: 'v = P c',
              trap: 'Verify P and c live in compatible dimensions.',
            }),
          },
          {
            id: 'similarity-formula',
            label: "A′ = P^{-1} A P",
            tooltip: tip({
              short: 'Express the same linear map after a change of coordinates.',
              intuition: 'Convert input, apply A in old coords, convert output back—or view as new matrix A′.',
              formula: "A' = P^{-1} A P",
              trap: 'P must be the same basis change for domain and codomain when A is square.',
            }),
          },
          {
            id: 'diagonalization-formula',
            label: 'A = P D P^{-1}',
            tooltip: tip({
              short: 'Eigenvector columns in P; D holds eigenvalues on the diagonal.',
              intuition: 'In eigen coordinates the map is pure scaling.',
              formula: 'A = P D P^{-1}',
              trap: 'Requires enough independent eigenvectors to form invertible P.',
            }),
          },
          {
            id: 'numpy-solve-cob',
            label: 'Recover coordinates in code',
            tooltip: tip({
              short: 'Solve P c = v or use inv/solve depending on conditioning.',
              intuition: 'Prefer solve over explicit inverse for stability.',
              code: 'c = np.linalg.solve(P, v)\n# orthonormal basis:\n# c = P.T @ v',
              trap: 'np.linalg.inv(P) can be unstable when P is ill-conditioned.',
            }),
          },
          {
            id: 'orthonormal-code-cob',
            label: 'Orthonormal basis code',
            tooltip: tip({
              short: 'Use Q^T instead of Q^{-1} when columns are orthonormal.',
              intuition: 'QR or SVD supply orthonormal bases in practice.',
              code: 'c = Q.T @ v\nv = Q @ c',
              trap: 'Economy Q still requires consistent shapes with v.',
            }),
            lessonId: 'qr-decomposition',
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'coordinates-are-not-v',
            label: 'Coordinates are not v',
            tooltip: tip({
              short: 'A column of numbers is meaningless until you name the basis.',
              intuition: 'Always track which basis produced c.',
              trap: 'Adding coordinates from different bases is invalid.',
            }),
          },
          {
            id: 'non-invertible-p-trap',
            label: 'Non-invertible P',
            tooltip: tip({
              short: 'Dependent basis columns prevent unique coordinates in the full space.',
              intuition: 'Rank(P) must equal ambient dimension for a full basis.',
              trap: 'Wide or rank-deficient P still defines a subspace basis, not all of ℝⁿ.',
            }),
          },
          {
            id: 'column-order-trap',
            label: 'Column order matters',
            tooltip: tip({
              short: 'Swapping basis column order permutes entries of c.',
              intuition: 'P encodes an ordered basis, not an unordered set.',
              trap: 'Reusing someone else’s c without their column order fails.',
            }),
          },
          {
            id: 'similarity-vs-congruence',
            label: 'Similarity vs congruence',
            tooltip: tip({
              short: 'A′ = P^{-1} A P is similarity; quadratic forms use P^T A P.',
              intuition: 'Different transformation rules serve different geometric questions.',
              trap: 'Mixing formulas swaps whether you preserve eigenvalues or quadratic energy.',
            }),
          },
          {
            id: 'left-right-basis-mismatch',
            label: 'Domain/codomain mismatch',
            tooltip: tip({
              short: 'Rectangular maps may need different input and output basis changes.',
              intuition: 'Use compatible P_in and P_out so dimensions chain correctly.',
              trap: 'Applying one square P formula to rectangular A is dimensionally wrong.',
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'eigenvalue-used-cob',
            label: 'Eigenvalues',
            tooltip: tip({
              short: 'Diagonalization is change of basis into eigenvector coordinates.',
              intuition: 'Similar transforms expose invariant scalings on the diagonal.',
              trap: 'Not every matrix admits a complete eigenbasis over ℝ.',
            }),
            lessonId: 'eigenvalue',
          },
          {
            id: 'svd-used-cob',
            label: 'SVD',
            tooltip: tip({
              short: 'U and V are orthonormal basis changes on output and input sides.',
              intuition: 'Σ is the simple diagonal map between those orthonormal coordinates.',
              trap: 'Singular vectors are not eigenvectors of A in general.',
            }),
            lessonId: 'svd',
          },
          {
            id: 'pca-used-cob',
            label: 'PCA',
            tooltip: tip({
              short: 'PCA rotates coordinates to variance-aligned orthonormal axes.',
              intuition: 'Principal components are a chosen orthonormal basis for data.',
              trap: 'Centering changes the origin before rotating coordinates.',
            }),
            lessonId: 'pca',
          },
          {
            id: 'projection-used-cob',
            label: 'Projection matrices',
            tooltip: tip({
              short: 'Projectors simplify in bases aligned with the target subspace.',
              intuition: 'Diagonal {0,1} entries appear when the subspace spans coordinate axes.',
              trap: 'Changing basis changes P’s matrix even though the projector map is fixed.',
            }),
            lessonId: 'projection-matrices',
          },
          {
            id: 'decomp-used-cob',
            label: 'Matrix decompositions',
            tooltip: tip({
              short: 'LU, QR, SVD, and eigendecompositions each supply useful bases.',
              intuition: 'Factorizations are structured changes of coordinates plus scaling.',
              trap: 'Pick the factorization that matches the basis geometry you need.',
            }),
            lessonId: 'matrix-decompositions',
          },
        ],
      },
    ],
  },
  pseudoinverse: {
    center: {
      id: 'pseudoinverse',
      label: 'Pseudoinverse',
      type: 'current',
      tooltip: tip({
        short: 'The Moore–Penrose pseudoinverse A⁺ extends inversion to rectangular and rank-deficient matrices, returning the minimum-norm least-squares solution of Ax ≈ b.',
        intuition: 'When no true inverse exists, A⁺ picks the best compromise: closest output hit and smallest input norm among all such hits.',
        formula: 'A^+ = V \\Sigma^+ U^T,\\quad x = A^+ b',
        why: 'Pseudoinverse stabilizes over/under-determined fits, connects SVD to practical solves, and underpins regularized regression and low-rank recovery.',
        trap: 'A⁺ is not A^{-1}; reciprocating tiny singular values amplifies noise instead of helping.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'matrix-mult-pinv',
            label: 'Matrix multiplication',
            tooltip: tip({
              short: 'A maps x in ℝⁿ to Ax in ℝᵐ; pseudoinverse reverses the story approximately.',
              intuition: 'Rectangular shape already signals one-sided invertibility.',
              trap: 'Left and right inverses differ unless A is square and full rank.',
            }),
            lessonId: 'matrix-multiplication',
          },
          {
            id: 'subspaces-pinv',
            label: 'Fundamental subspaces',
            tooltip: tip({
              short: 'Column space reachability and null-space freedom govern sensible solves.',
              intuition: 'Least squares projects b onto Col(A); null directions do not affect Ax.',
              trap: 'Many x solve projected equations when nullity > 0.',
            }),
            lessonId: 'fundamental-subspaces',
          },
          {
            id: 'least-squares-pinv',
            label: 'Least squares',
            tooltip: tip({
              short: 'Minimize ‖b − Ax‖ when exact solves fail.',
              intuition: 'Geometrically project b onto Col(A).',
              trap: 'Least squares alone does not pick a unique x when columns are dependent.',
            }),
            lessonId: 'least-squares-projection',
          },
          {
            id: 'svd-preview-pinv',
            label: 'SVD preview',
            tooltip: tip({
              short: 'SVD exposes orthonormal directions and singular values for stable inversion.',
              intuition: 'Invert only nonzero σ to build Σ⁺.',
              trap: 'Tiny σ are numerically zero even if not exactly zero on paper.',
            }),
            lessonId: 'svd',
          },
          {
            id: 'rank-pinv',
            label: 'Rank and independence',
            tooltip: tip({
              short: 'Rank r counts independent columns/rows; deficiency breaks ordinary inverses.',
              intuition: 'Rank-deficiency creates null directions or unreachable components.',
              trap: 'Full row rank and full column rank mean different things for rectangular A.',
            }),
          },
          {
            id: 'transpose-pinv',
            label: 'Matrix transpose',
            tooltip: tip({
              short: 'A^T links normal equations and Moore–Penrose identities.',
              intuition: 'Transpose swaps domain/codomain roles in rectangular systems.',
              trap: 'A^T A can be singular even when A has useful column information.',
            }),
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'moore-penrose-conditions',
            label: 'Moore–Penrose conditions',
            tooltip: tip({
              short: 'A⁺ is the unique matrix satisfying four reverse identities with A and A^T.',
              intuition: 'Generalizes inverse behavior for range, null space, and transposes.',
              formula: 'A A^+ A = A,\\; A^+ A A^+ = A^+',
              trap: 'Memorizing all four identities matters less than the SVD construction.',
            }),
          },
          {
            id: 'svd-construction',
            label: 'SVD construction',
            tooltip: tip({
              short: 'From A = U Σ V^T, set A⁺ = V Σ⁺ U^T with reciprocal nonzero σ.',
              intuition: 'Undo scaling along singular directions; zero σ stay zero in Σ⁺.',
              formula: 'A^+ = V \\Sigma^+ U^T',
              trap: 'Σ⁺ truncates rank—do not invert noise-level singular values.',
            }),
          },
          {
            id: 'min-norm-least-squares',
            label: 'Minimum-norm least squares',
            tooltip: tip({
              short: 'x = A⁺ b minimizes ‖b − Ax‖ and, among all such x, minimizes ‖x‖.',
              intuition: 'Project onto Col(A), then pull back with smallest input norm.',
              formula: 'x = A^+ b',
              trap: 'Without the min-norm tie-break, x is not unique when Null(A) is nontrivial.',
            }),
          },
          {
            id: 'full-column-rank-case',
            label: 'Full column rank case',
            tooltip: tip({
              short: 'If columns are independent, A⁺ = (A^T A)^{-1} A^T for overdetermined A.',
              intuition: 'Normal-equation inverse works on the column space when A^T A is invertible.',
              formula: 'A^+ = (A^T A)^{-1} A^T',
              trap: 'Forming A^T A explicitly can square the condition number.',
            }),
          },
          {
            id: 'full-row-rank-case',
            label: 'Full row rank case',
            tooltip: tip({
              short: 'If rows are independent, A⁺ = A^T (A A^T)^{-1} for underdetermined A.',
              intuition: 'Minimum-norm solution among infinitely many exact fits.',
              formula: 'A^+ = A^T (A A^T)^{-1}',
              trap: 'Underdetermined exact solves still leave a null-space family without min-norm pin.',
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'best-compromise-inverse',
            label: 'Best compromise inverse',
            tooltip: tip({
              short: 'A⁺ acts like an inverse on reachable outputs and sends unreachable parts to zero.',
              intuition: 'It inverts the map on its column space and crushes null-space ambiguity.',
              trap: 'Unreachable b components are not magically fit—they are ignored in the min-residual sense.',
            }),
          },
          {
            id: 'over-vs-underdetermined',
            label: 'Over vs underdetermined',
            tooltip: tip({
              short: 'Tall A: closest fit. Wide A: exact fit with smallest ‖x‖ when consistent.',
              intuition: 'Shape tells you which error (output or coefficient) is being optimized.',
              trap: 'Wide inconsistent systems still need least-squares, not naive row solving.',
            }),
          },
          {
            id: 'null-directions-pinned',
            label: 'Null directions pinned',
            tooltip: tip({
              short: 'A⁺ returns no component of x in Null(A).',
              intuition: 'Free directions are set to zero to pick the shortest coefficient vector.',
              trap: 'Other least-squares solutions differ by null-space vectors.',
            }),
          },
          {
            id: 'truncate-sigma-intuition',
            label: 'Truncated Σ⁺ intuition',
            tooltip: tip({
              short: 'Dropping tiny σ before inversion is practical regularization.',
              intuition: 'Treat near-zero modes as noise rather than signal to amplify.',
              trap: 'Hard cutoffs introduce bias; soft ridge penalties smooth the same tradeoff.',
            }),
          },
          {
            id: 'project-then-solve',
            label: 'Project then solve',
            tooltip: tip({
              short: 'Conceptually project b onto Col(A), then invert A on that subspace.',
              intuition: 'Pseudoinverse factorizes “fit output” and “pick smallest x.”',
              trap: 'Skipping projection logic hides why residuals lie in Null(A^T).',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'pinv-svd-formula',
            label: 'A⁺ = V Σ⁺ U^T',
            tooltip: tip({
              short: 'Build Σ⁺ by reciprocating nonzero diagonal σ and leaving zeros at zero.',
              intuition: 'Orthonormal U,V preserve lengths; Σ⁺ controls gain per mode.',
              formula: 'A^+ = V \\Sigma^+ U^T',
              trap: 'Use economy SVD shapes consistent with A’s rank.',
            }),
          },
          {
            id: 'solve-x-formula',
            label: 'x = A⁺ b',
            tooltip: tip({
              short: 'One-shot minimum-norm least-squares solve.',
              intuition: 'Library pinv wraps SVD with a tolerance on tiny σ.',
              formula: 'x = A^+ b',
              trap: 'Inspect residual ‖b − Ax‖ even when x looks reasonable.',
            }),
          },
          {
            id: 'numpy-pinv',
            label: 'numpy.linalg.pinv',
            tooltip: tip({
              short: 'Computes Moore–Penrose inverse via SVD with default cutoff.',
              intuition: 'rcond parameter sets which σ are treated as zero.',
              code: 'A_plus = np.linalg.pinv(A, rcond=1e-10)\nx = A_plus @ b',
              trap: 'Default rcond may be too aggressive or too lax for your scale.',
            }),
          },
          {
            id: 'lstsq-link-pinv',
            label: 'lstsq connection',
            tooltip: tip({
              short: 'np.linalg.lstsq returns least-squares x; pinv adds explicit min-norm tie-break.',
              intuition: 'Both lean on QR/SVD internally for stability.',
              code: 'x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)',
              trap: 'lstsq rank cutoff changes which solution branch you get.',
            }),
            lessonId: 'least-squares-projection',
          },
          {
            id: 'ridge-link-pinv',
            label: 'Ridge / Tikhonov link',
            tooltip: tip({
              short: 'Adding λI before inversion smooths the same ill-posed directions Σ⁺ exposes.',
              intuition: 'Regularization replaces hard Σ⁺ truncation with softened reciprocals.',
              formula: 'x = (A^T A + \\lambda I)^{-1} A^T b',
              trap: 'λ trades bias for variance—no single universal choice.',
            }),
            lessonId: 'linear-regression',
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'not-a-true-inverse',
            label: 'Not a true inverse',
            tooltip: tip({
              short: 'Generally A A⁺ A = A but A⁺ A ≠ I unless A is invertible square.',
              intuition: 'Pseudoinverse reverses A only on the column space part.',
              trap: 'Treating A⁺ as A^{-1} in formulas without checking rank fails.',
            }),
          },
          {
            id: 'tiny-sigma-trap',
            label: 'Tiny σ amplification',
            tooltip: tip({
              short: 'Reciprocating noise-level singular values explodes coefficients.',
              intuition: 'Truncation or ridge dampens ill-posed modes.',
              trap: 'Perfect pinv on noisy data overfits spurious directions.',
            }),
          },
          {
            id: 'nonunique-without-minnorm',
            label: 'Non-uniqueness without min-norm',
            tooltip: tip({
              short: 'Any x + n with n ∈ Null(A) gives the same Ax when nullity > 0.',
              intuition: 'A⁺ picks the shortest representative.',
              trap: 'Comparing two least-squares solvers without min-norm may yield different x.',
            }),
          },
          {
            id: 'normal-equations-singular',
            label: 'Singular A^T A trap',
            tooltip: tip({
              short: 'Normal equations fail when columns are dependent even if a least-squares fit exists.',
              intuition: 'Use QR/SVD/pinv instead of raw (A^T A)^{-1} A^T.',
              trap: 'A^T A invertibility is stricter than useful least-squares solvability.',
            }),
          },
          {
            id: 'left-right-pinv-confusion',
            label: 'Left vs right inverse confusion',
            tooltip: tip({
              short: 'Left inverse (tall full column) and right inverse (wide full row) differ from A⁺.',
              intuition: 'Moore–Penrose unifies both rectangular stories in one matrix.',
              trap: 'Using A^{-1} formulas on the wrong shape silently mis-solves.',
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'linear-regression-pinv',
            label: 'Linear regression',
            tooltip: tip({
              short: 'Design-matrix fits use pseudoinverse thinking when features are collinear.',
              intuition: 'Minimum-norm coefficients stabilize underdetermined or rank-deficient designs.',
              trap: 'Feature scaling still affects which directions look small.',
            }),
            lessonId: 'linear-regression',
          },
          {
            id: 'low-rank-pinv',
            label: 'Low-rank approximation',
            tooltip: tip({
              short: 'Truncated SVD pseudoinverse keeps dominant modes only.',
              intuition: 'Same singular directions as compression—invert the kept part.',
              trap: 'Truncation removes signal and noise together.',
            }),
            lessonId: 'low-rank-approximation',
          },
          {
            id: 'condition-pinv',
            label: 'Condition number',
            tooltip: tip({
              short: 'κ(A) warns how pinv gains amplify input noise into x.',
              intuition: 'Large σ₁/σ_r ratio means fragile inversion directions.',
              trap: 'Good residual does not imply stable coefficients.',
            }),
            lessonId: 'condition-number',
          },
          {
            id: 'svd-used-pinv',
            label: 'SVD',
            tooltip: tip({
              short: 'Pseudoinverse is the operational face of SVD for solves.',
              intuition: 'Σ⁺ makes singular structure actionable in code.',
              trap: 'Always align truncation policy with SVD lesson tolerances.',
            }),
            lessonId: 'svd',
          },
          {
            id: 'qr-used-pinv',
            label: 'QR decomposition',
            tooltip: tip({
              short: 'Pivoted QR and SVD routes handle rank deficiency before back substitution breaks.',
              intuition: 'Orthonormal Q steps stabilize least squares when normal equations fail.',
              trap: 'Plain QR without pivoting can hide dependent columns.',
            }),
            lessonId: 'qr-decomposition',
          },
        ],
      },
    ],
  },
'model-debugging': {
    center: {
      id: 'model-debugging',
      label: 'Model Debugging',
      type: 'current',
      tooltip: tip({
        short: 'Model debugging narrows ML failures to data, training, evaluation, or serving before changing hyperparameters or architecture.',
        intuition: 'Treat the ML system like a pipeline: find where expected behavior first diverges, then run targeted experiments to confirm the cause.',
        formula: 'symptom → hypothesis → controlled check → localized fix → regression test',
        why: 'Systematic debugging saves weeks of random tuning and prevents fixes that mask deeper data or evaluation bugs.',
        trap: 'Random hyperparameter search is not debugging; it skips locating which subsystem is broken.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'train-val-test-split',
            label: 'Train / val / test split',
            tooltip: tip({
              short: 'Holdout sets isolate whether a bug is memorization, tuning noise, or true generalization.',
              intuition: 'Debugging starts by asking which split shows the failure first.',
              example: 'Train great, val bad → overfit or leakage; all splits bad → data or label bug.',
              trap: 'Tuning on the test set hides evaluation bugs.',
            }),
            lessonId: 'train-test-split',
          },
          {
            id: 'classification-metrics-prereq',
            label: 'Classification metrics',
            tooltip: tip({
              short: 'Precision, recall, F1, and confusion cells localize error types.',
              intuition: 'A “bad model” might be high false positives on one slice only.',
              example: 'Recall collapse often points to class imbalance or label noise on positives.',
              trap: 'Accuracy alone hides slice-specific failures.',
            }),
            lessonId: 'classification-metrics',
          },
          {
            id: 'loss-and-gradients',
            label: 'Loss and gradients',
            tooltip: tip({
              short: 'Training loss should decrease; flat or NaN curves signal optimization or data issues.',
              intuition: 'Loss behavior tells you whether the bug lives in optimization, not just final metrics.',
              example: 'Loss NaN → bad learning rate, bad scaling, or corrupted labels.',
              trap: 'Low training loss with bad val metrics still means something is wrong—often leakage or overfit.',
            }),
            lessonId: 'training-loop-dynamics',
          },
          {
            id: 'baselines',
            label: 'Simple baselines',
            tooltip: tip({
              short: 'Compare against majority class, linear model, or last production version.',
              intuition: 'If a dumb baseline beats your model, the bug is likely data or evaluation—not architecture.',
              example: 'Always predict “no fraud” beats a broken fraud model on imbalanced data.',
              trap: 'Skipping baselines makes every failure look like a “model problem.”',
            }),
          },
          {
            id: 'reproducibility',
            label: 'Reproducibility',
            tooltip: tip({
              short: 'Fix seeds, data order, and code version so failures can be replayed.',
              intuition: 'Non-reproducible bugs cannot be bisected or verified after a fix.',
              example: 'Same seed + same shard order → same bad batch every run.',
              trap: '“It worked once on my laptop” is not evidence of a fix.',
            }),
          },
          {
            id: 'logging-tracing',
            label: 'Logging and tracing',
            tooltip: tip({
              short: 'Log inputs, outputs, features, and versions at each pipeline stage.',
              intuition: 'You cannot debug what you cannot see in production or offline replay.',
              example: 'Log feature vector hash + model version + preprocessing code hash.',
              trap: 'Logging only final predictions hides upstream corruption.',
            }),
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'symptom-definition',
            label: 'Define the symptom',
            tooltip: tip({
              short: 'State exactly what is wrong: metric, slice, latency, or behavior.',
              intuition: 'Vague “model is bad” prevents targeted experiments.',
              example: '“Recall on mobile users dropped 12 pts since build 482.”',
              trap: 'Fixing a different metric than the reported symptom wastes time.',
            }),
          },
          {
            id: 'hypothesis-list',
            label: 'List hypotheses',
            tooltip: tip({
              short: 'Rank plausible causes: data, labels, features, train code, eval setup, serving.',
              intuition: 'Good debugging generates falsifiable guesses before touching weights.',
              example: 'Hypothesis: new parser strips currency symbols → feature drift.',
              trap: 'Jumping to “need bigger model” without a hypothesis is guessing.',
            }),
          },
          {
            id: 'controlled-check',
            label: 'Controlled check',
            tooltip: tip({
              short: 'Change one variable at a time against a known-good reference.',
              intuition: 'Ablations and diffs isolate the first broken step.',
              example: 'Run old preprocessing + new model vs new preprocessing + old model.',
              trap: 'Changing data, code, and hyperparameters together confounds the result.',
            }),
          },
          {
            id: 'slice-localization',
            label: 'Slice localization',
            tooltip: tip({
              short: 'Break metrics by time, region, device, class, or cohort.',
              intuition: 'Global averages hide concentrated failures.',
              example: 'Overall AUC flat but precision on segment A halved.',
              trap: 'Tiny slices with few examples produce noisy conclusions.',
            }),
          },
          {
            id: 'error-buckets',
            label: 'Error buckets',
            tooltip: tip({
              short: 'Group wrong predictions by pattern: false positives, false negatives, OOD inputs.',
              intuition: 'Buckets suggest different fixes: threshold vs labels vs features.',
              example: 'All FPs are blurry images → augmentation or camera pipeline bug.',
              trap: 'Inspecting random errors without bucketing misses structure.',
            }),
          },
          {
            id: 'verify-fix',
            label: 'Verify and regression-test',
            tooltip: tip({
              short: 'Confirm the fix on the failing slice and check you did not break others.',
              intuition: 'A local fix can regress global metrics or neighboring cohorts.',
              example: 'Re-run golden set + prior release benchmark before ship.',
              trap: 'Stopping after one improved number without regression checks.',
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'pipeline-not-weights',
            label: 'Pipeline, not just weights',
            tooltip: tip({
              short: 'Most production failures start outside the neural net: data, features, joins, serving.',
              intuition: 'The model is one stage in a longer system.',
              example: 'Train/serve skew from different imputation defaults.',
              trap: 'Retraining cannot fix a broken feature store join.',
            }),
          },
          {
            id: 'compare-to-control',
            label: 'Compare to a control',
            tooltip: tip({
              short: 'Always diff against yesterday’s good build or a minimal baseline.',
              intuition: 'Controls turn “something changed” into “this component changed.”',
              example: 'Same data, old model good → suspect new code path.',
              trap: 'Debugging without a control is storytelling.',
            }),
          },
          {
            id: 'smallest-repro',
            label: 'Smallest reproduction',
            tooltip: tip({
              short: 'Shrink the failing case to one batch, one feature, or one row.',
              intuition: 'Minimal repros make bisect and unit tests possible.',
              example: 'Single toxic label row causes loss spike every epoch.',
              trap: 'Only testing on full terabyte sets slows every iteration.',
            }),
          },
          {
            id: 'eval-bugs-first',
            label: 'Check evaluation setup',
            tooltip: tip({
              short: 'Leaky features, wrong labels, or shuffled ids can fake great offline scores.',
              intuition: 'Evaluation bugs look like genius models until deployment.',
              example: 'Target column accidentally included in features.',
              trap: 'Celebrating metrics before auditing the eval pipeline.',
            }),
            lessonId: 'data-leakage-deep-dive',
          },
          {
            id: 'serving-vs-training',
            label: 'Serving vs training',
            tooltip: tip({
              short: 'Offline metrics can look fine while online preprocessing diverges.',
              intuition: 'Training and serving are two implementations that must match.',
              example: 'Training uses pandas fillna(0); API uses null → different vector.',
              trap: 'Assuming “same model file” means identical behavior.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'confusion-slice',
            label: 'Confusion matrix slice',
            tooltip: tip({
              short: 'TP, FP, TN, FN counts on the failing cohort.',
              intuition: 'Cell counts tell you whether to tune threshold, labels, or features.',
              formula: 'precision = TP / (TP + FP); recall = TP / (TP + FN)',
              trap: 'Aggregating slices before computing rates hides imbalance.',
            }),
          },
          {
            id: 'metric-delta',
            label: 'Metric delta vs baseline',
            tooltip: tip({
              short: 'Δmetric = metric_new − metric_baseline on the same eval set.',
              intuition: 'Deltas on identical data isolate code or model changes.',
              example: 'ΔF1 = −0.08 only on night-time traffic.',
              trap: 'Comparing metrics computed on different eval sets.',
            }),
          },
          {
            id: 'bisect-script',
            label: 'Git bisect mindset',
            tooltip: tip({
              short: 'Binary search commits or builds to find first bad change.',
              intuition: 'Automate “pass/fail on golden set” for each step.',
              code: 'for commit in bisect_range:\n  run_golden_set(commit)\n  mark pass or fail',
              trap: 'Non-deterministic tests make bisect unreliable.',
            }),
          },
          {
            id: 'ablation-table',
            label: 'Ablation table',
            tooltip: tip({
              short: 'Remove one component at a time and measure impact.',
              intuition: 'Shows which part actually drives the symptom.',
              example: 'No augmentation: +2 F1 → augmentation was hurting this slice.',
              trap: 'Removing multiple components at once confounds attribution.',
            }),
          },
          {
            id: 'golden-set',
            label: 'Golden set check',
            tooltip: tip({
              short: 'Fixed labeled examples that must never regress.',
              intuition: 'Fast smoke test after every change.',
              example: '50 curated failure cases from last incident.',
              trap: 'Golden set too small or stale to catch new failure modes.',
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'random-hpo-trap',
            label: 'Random HPO first',
            tooltip: tip({
              short: 'Tuning learning rate before finding leakage wastes GPU and hides the bug.',
              intuition: 'Hyperparameters cannot fix wrong labels or duplicated targets.',
              trap: 'This is the most expensive non-debugging habit.',
            }),
          },
          {
            id: 'train-metric-only',
            label: 'Train metric only',
            tooltip: tip({
              short: 'Perfect training accuracy with broken validation means you have not debugged generalization.',
              intuition: 'Always ask what the holdout or production slice shows.',
              trap: 'Shipping because “loss went down.”',
            }),
          },
          {
            id: 'one-seed-trap',
            label: 'Single seed conclusion',
            tooltip: tip({
              short: 'One lucky or unlucky seed can fake success or failure.',
              intuition: 'Re-run critical checks with fixed and varied seeds.',
              trap: 'Declaring victory from one training run.',
            }),
          },
          {
            id: 'cherry-picked-examples',
            label: 'Cherry-picked examples',
            tooltip: tip({
              short: 'Hand-picked demos do not represent error distribution.',
              intuition: 'Sample systematically from error buckets.',
              trap: 'Fixing the one example shown in a slide deck.',
            }),
          },
          {
            id: 'ignore-data-version',
            label: 'Ignore data version',
            tooltip: tip({
              short: 'Silent upstream schema or label changes break models without code deploys.',
              intuition: 'Always log dataset snapshot id with each experiment.',
              trap: 'Blaming the model when the warehouse view changed.',
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'model-monitoring-lesson',
            label: 'Model monitoring',
            tooltip: tip({
              short: 'Production observability extends debugging into continuous drift and alert loops.',
              intuition: 'Incidents you debug offline become monitors online.',
              trap: 'Monitoring without prior root-cause discipline produces alert noise.',
            }),
            lessonId: 'model-monitoring',
          },
          {
            id: 'model-interpretability-lesson',
            label: 'Model interpretability',
            tooltip: tip({
              short: 'Attributions and counterfactuals help explain slice failures after localization.',
              intuition: 'Debugging finds where; interpretability often suggests why in the model.',
              trap: 'Explanations are hypotheses, not automatic fixes.',
            }),
            lessonId: 'model-interpretability',
          },
          {
            id: 'uncertainty-lesson',
            label: 'Uncertainty estimation',
            tooltip: tip({
              short: 'Uncertainty signals highlight cases worth manual review during debugging.',
              intuition: 'High-entropy or OOD flags prioritize error analysis queues.',
              trap: 'Uncalibrated confidence is a weak debug signal alone.',
            }),
            lessonId: 'uncertainty-estimation',
          },
          {
            id: 'leakage-lesson',
            label: 'Data leakage deep dive',
            tooltip: tip({
              short: 'Leakage is a top cause of “great offline, bad online” debugging stories.',
              intuition: 'Audit features, splits, and temporal boundaries systematically.',
              trap: 'Leakage fixes require data pipeline changes, not more layers.',
            }),
            lessonId: 'data-leakage-deep-dive',
          },
          {
            id: 'training-dynamics-lesson',
            label: 'Training loop dynamics',
            tooltip: tip({
              short: 'Optimization bugs—LR, batch norm, instability—live in the training loop.',
              intuition: 'Loss curves and gradient norms are primary debug instruments.',
              trap: 'Assuming every training failure needs architecture change.',
            }),
            lessonId: 'training-loop-dynamics',
          },
        ],
      },
    ],
  },
  'model-monitoring': {
    center: {
      id: 'model-monitoring',
      label: 'Model Monitoring',
      type: 'current',
      tooltip: tip({
        short: 'Model monitoring tracks whether live data, predictions, labels, performance, calibration, and latency stay within expected bounds after deployment.',
        intuition: 'Serving traffic is a continuous experiment: compare today’s stream to a trusted reference window and act when signals cross meaningful thresholds.',
        formula: 'alert when metric_t − baseline > threshold (with seasonality & sample checks)',
        why: 'Models decay silently—drift, broken pipelines, and calibration rot can happen while uptime stays at 100%.',
        trap: 'Checking only server health misses model health; successful HTTP responses do not mean correct decisions.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'model-debugging-prereq',
            label: 'Model debugging',
            tooltip: tip({
              short: 'Monitoring alerts should trigger the same disciplined investigation loop.',
              intuition: 'An alert is a symptom; debugging localizes cause.',
              trap: 'Auto-retraining on every alert without root-cause analysis.',
            }),
            lessonId: 'model-debugging',
          },
          {
            id: 'calibration-prereq',
            label: 'Calibration',
            tooltip: tip({
              short: 'Track whether predicted probabilities stay honest over time and segments.',
              intuition: 'Calibration drift is a first-class monitoring signal.',
              example: '0.8 bin observed rate falls from 0.78 to 0.52 over a month.',
              trap: 'Monitoring accuracy alone while probabilities drive decisions.',
            }),
            lessonId: 'calibration',
          },
          {
            id: 'classification-metrics-mon',
            label: 'Classification metrics',
            tooltip: tip({
              short: 'Precision, recall, and rates need labeled feedback or proxies in production.',
              intuition: 'Delayed labels mean performance monitors lag reality.',
              trap: 'Assuming instant ground truth in production.',
            }),
            lessonId: 'classification-metrics',
          },
          {
            id: 'reference-baseline',
            label: 'Reference baseline',
            tooltip: tip({
              short: 'A stable window or snapshot representing “healthy” behavior.',
              intuition: 'Drift is measured relative to something trustworthy.',
              example: 'Training distribution or first 30 days post-launch.',
              trap: 'Using a baseline that already contained drift.',
            }),
          },
          {
            id: 'structured-logging',
            label: 'Structured logging',
            tooltip: tip({
              short: 'Store features, scores, versions, and metadata for aggregation.',
              intuition: 'Monitors are only as good as logged fields.',
              trap: 'Logging predictions without input features blocks input-drift analysis.',
            }),
          },
          {
            id: 'deployment-versioning',
            label: 'Deployment versioning',
            tooltip: tip({
              short: 'Tag every prediction with model, data, and code version.',
              intuition: 'Incidents map to specific releases for rollback.',
              example: 'model_v3 + featurizer_commit_abc + calibrator_v2.',
              trap: 'Untagged traffic mixes multiple behaviors in one chart.',
            }),
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'input-drift',
            label: 'Input / covariate drift',
            tooltip: tip({
              short: 'Live feature distributions diverge from the reference.',
              intuition: 'The model may face inputs it was never trained to handle.',
              example: 'New merchant category spikes in fraud model features.',
              trap: 'Drift in one feature may matter more than global PSI suggests.',
            }),
          },
          {
            id: 'prediction-drift',
            label: 'Prediction drift',
            tooltip: tip({
              short: 'Score or label-rate distributions shift even if inputs look similar.',
              intuition: 'Can indicate upstream model change, threshold move, or population shift.',
              example: 'Average risk score rises 0.15 with flat input PSI.',
              trap: 'Prediction drift alone does not prove harm—check outcomes.',
            }),
          },
          {
            id: 'label-drift',
            label: 'Label / outcome drift',
            tooltip: tip({
              short: 'Observed positive rate or outcome mix changes over time.',
              intuition: 'World change vs model change must be disentangled.',
              example: 'Chargeback rate doubles after policy change.',
              trap: 'Confusing label drift with model calibration drift.',
            }),
          },
          {
            id: 'performance-monitoring',
            label: 'Performance monitoring',
            tooltip: tip({
              short: 'Track precision, recall, AUC, or business KPIs on labeled slices.',
              intuition: 'Ultimate ground truth when labels arrive.',
              example: 'Weekly refreshed labels on 5% audit sample.',
              trap: 'Tiny labeled samples create volatile performance charts.',
            }),
          },
          {
            id: 'calibration-monitoring',
            label: 'Calibration monitoring',
            tooltip: tip({
              short: 'Reliability bins and ECE tracked over rolling windows.',
              intuition: 'Probabilities may rank well while frequency meaning rots.',
              trap: 'Global calibration can hide subgroup miscalibration.',
            }),
          },
          {
            id: 'latency-cost',
            label: 'Latency and cost',
            tooltip: tip({
              short: 'p50/p95 latency, throughput, GPU spend, and queue depth.',
              intuition: 'Operational SLOs are part of model health.',
              example: 'p95 latency doubles after batch size change in serving.',
              trap: 'Ignoring cost drift until budget breach.',
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'expected-variation',
            label: 'Expected variation',
            tooltip: tip({
              short: 'Not every wiggle is an incident—seasonality and sample noise exist.',
              intuition: 'Good monitors separate signal from routine fluctuation.',
              example: 'Retail traffic spikes every Sunday without model failure.',
              trap: 'Alerting on raw daily noise causes fatigue.',
            }),
          },
          {
            id: 'action-thresholds',
            label: 'Action thresholds',
            tooltip: tip({
              short: 'Define what happens at yellow vs red: investigate, rollback, retrain, review.',
              intuition: 'Monitoring without playbooks is dashboard theater.',
              trap: 'Alerts that nobody owns become ignored noise.',
            }),
          },
          {
            id: 'segment-slices',
            label: 'Segment slices',
            tooltip: tip({
              short: 'Monitor globally and by region, product, cohort, or channel.',
              intuition: 'Aggregate health hides localized regressions.',
              example: 'Global AUC stable; mobile APAC recall collapsed.',
              trap: 'Too many micro-slices without minimum sample rules.',
            }),
          },
          {
            id: 'delayed-labels',
            label: 'Delayed labels',
            tooltip: tip({
              short: 'Performance truth arrives late; use proxies and input drift early.',
              intuition: 'Leading indicators buy time before business KPIs move.',
              trap: 'Waiting weeks to learn the model broke on day one.',
            }),
          },
          {
            id: 'human-review-loop',
            label: 'Human review loop',
            tooltip: tip({
              short: 'Sample high-uncertainty or high-impact cases for manual audit.',
              intuition: 'Labels from review feed back into monitoring.',
              trap: 'Review queues without prioritization overflow.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'psi-drift',
            label: 'Population stability index',
            tooltip: tip({
              short: 'PSI compares binned distributions between reference and current.',
              intuition: 'Large PSI suggests meaningful distributional shift.',
              formula: 'PSI = Σ (p_i − q_i) ln(p_i / q_i)',
              trap: 'PSI thresholds are heuristics; domain context matters.',
            }),
          },
          {
            id: 'rolling-window',
            label: 'Rolling window metrics',
            tooltip: tip({
              short: 'Compute stats over last N hours/days with minimum count guards.',
              intuition: 'Windows smooth noise while staying responsive.',
              code: 'if count(window) < min_samples: suppress_alert()',
              trap: 'Windows too short → noise; too long → slow detection.',
            }),
          },
          {
            id: 'ece-over-time',
            label: 'ECE over time',
            tooltip: tip({
              short: 'Track expected calibration error on sliding labeled batches.',
              intuition: 'Rising ECE warns before threshold-based KPIs move.',
              trap: 'ECE with sparse bins is unstable—require per-bin counts.',
            }),
            lessonId: 'calibration',
          },
          {
            id: 'alert-rule',
            label: 'Alert rule sketch',
            tooltip: tip({
              short: 'Combine magnitude, duration, and segment filters.',
              code: 'fire if psi > 0.2 for 3 consecutive days and n > 1000',
              intuition: 'Duration filters reduce one-day spikes.',
              trap: 'Single-threshold rules without duration cause flapping.',
            }),
          },
          {
            id: 'slo-latency',
            label: 'Latency SLO',
            tooltip: tip({
              short: 'p95 latency budget vs error budget for model serving.',
              intuition: 'Treat latency like reliability: burn rate alerts.',
              example: 'SLO: p95 < 120ms over 30d; page if 6h burn > 2×.',
              trap: 'Optimizing latency by skipping safety checks.',
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'uptime-only',
            label: 'Uptime-only monitoring',
            tooltip: tip({
              short: '200 OK responses while predictions degrade silently.',
              intuition: 'Model quality and service availability are different.',
              trap: 'Declaring healthy because the endpoint responds.',
            }),
          },
          {
            id: 'alert-fatigue',
            label: 'Alert fatigue',
            tooltip: tip({
              short: 'Too many low-severity pages train teams to ignore real incidents.',
              intuition: 'Tune thresholds and ownership before adding charts.',
              trap: 'Copying every metric from a blog post into PagerDuty.',
            }),
          },
          {
            id: 'no-baseline-window',
            label: 'Missing baseline',
            tooltip: tip({
              short: 'Drift scores need a reference; otherwise “change” is undefined.',
              intuition: 'Document when and why the baseline was chosen.',
              trap: 'Using a contaminated post-incident week as “normal.”',
            }),
          },
          {
            id: 'aggregate-only',
            label: 'Aggregate-only dashboards',
            tooltip: tip({
              short: 'Global charts miss fairness and segment regressions.',
              intuition: 'Slice monitors are mandatory for high-stakes systems.',
              trap: 'Green global AUC while one group is harmed.',
            }),
          },
          {
            id: 'auto-retrain-blind',
            label: 'Blind auto-retrain',
            tooltip: tip({
              short: 'Retraining on drifted or broken data encodes the bug.',
              intuition: 'Fix data pipelines before feeding drift into training.',
              trap: 'Scheduled retrain as substitute for investigation.',
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'fairness-monitoring',
            label: 'Fairness monitoring',
            tooltip: tip({
              short: 'Group metrics and calibration must be monitored, not just global KPIs.',
              intuition: 'Fairness regressions often appear in slices first.',
              trap: 'Fairness as one-time pre-launch check only.',
            }),
            lessonId: 'model-fairness',
          },
          {
            id: 'uncertainty-monitoring',
            label: 'Uncertainty signals',
            tooltip: tip({
              short: 'Track OOD rate, entropy, and deferral volume in production.',
              intuition: 'Spikes in uncertainty often precede performance drops.',
              trap: 'Uncalibrated softmax entropy mis-ranks risk.',
            }),
            lessonId: 'uncertainty-estimation',
          },
          {
            id: 'reliability-hub',
            label: 'Model reliability',
            tooltip: tip({
              short: 'Monitoring sits inside broader reliability: debug, interpret, govern.',
              intuition: 'Ops metrics plus responsible-ML metrics form trust.',
              trap: 'Silos between ML and platform on-call.',
            }),
            lessonId: 'model-reliability',
          },
          {
            id: 'roc-pr-monitoring',
            label: 'ROC / PR tracking',
            tooltip: tip({
              short: 'Threshold-free curves help when base rates shift.',
              intuition: 'Ranking drift appears before a fixed-threshold KPI moves.',
              trap: 'PR curves need careful comparison when prevalence changes.',
            }),
            lessonId: 'roc-pr-curves',
          },
          {
            id: 'ab-testing-guardrails',
            label: 'A/B guardrails',
            tooltip: tip({
              short: 'Ship model changes with experiment guardrail metrics.',
              intuition: 'Monitoring extends into controlled rollouts.',
              trap: 'Launching without rollback path or guardrail dashboards.',
            }),
            lessonId: 'ab-testing-foundations',
          },
        ],
      },
    ],
  },
  'model-fairness': {
    center: {
      id: 'model-fairness',
      label: 'Model Fairness',
      type: 'current',
      tooltip: tip({
        short: 'Model fairness compares error rates, selection rates, and calibration across groups to surface disparate impact and choose explicit tradeoffs.',
        intuition: 'Fairness is not one number—it is deciding which groups should receive comparable treatment for a specific decision context.',
        formula: 'compare metrics_g across groups g; choose criterion + mitigation',
        why: 'Deployed models allocate opportunities, risk, and cost unevenly unless you measure and govern group outcomes.',
        trap: 'Removing a protected column from features does not remove proxy bias or guarantee fair outcomes.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'classification-metrics-fair',
            label: 'Classification metrics',
            tooltip: tip({
              short: 'TPR, FPR, FNR, precision, and selection rate power group comparisons.',
              intuition: 'Different metrics encode different harms to different groups.',
              trap: 'Optimizing accuracy while one group pays most false negatives.',
            }),
            lessonId: 'classification-metrics',
          },
          {
            id: 'monitoring-prereq-fair',
            label: 'Model monitoring',
            tooltip: tip({
              short: 'Fairness metrics should be tracked continuously in production slices.',
              intuition: 'Pre-launch parity can erode under drift.',
              trap: 'One-off fairness audit before launch.',
            }),
            lessonId: 'model-monitoring',
          },
          {
            id: 'interpretability-prereq',
            label: 'Model interpretability',
            tooltip: tip({
              short: 'Attributions help inspect proxy features and slice failures.',
              intuition: 'Fairness work often starts from “why this group?”',
              trap: 'Treating SHAP as legal proof of non-discrimination.',
            }),
            lessonId: 'model-interpretability',
          },
          {
            id: 'base-rates',
            label: 'Base rates by group',
            tooltip: tip({
              short: 'Prevalence of the positive class often differs across groups.',
              intuition: 'Different base rates make some fairness criteria mathematically tensioned.',
              example: 'Hiring rate differs by cohort even with identical skill signal.',
              trap: 'Ignoring prevalence when interpreting parity gaps.',
            }),
          },
          {
            id: 'threshold-policy',
            label: 'Threshold policy',
            tooltip: tip({
              short: 'The decision cutoff converts scores into approvals, flags, or denials.',
              intuition: 'Same model + different thresholds → different fairness profile.',
              trap: 'Assuming one global threshold is neutral.',
            }),
          },
          {
            id: 'protected-vs-proxy',
            label: 'Protected vs proxy attributes',
            tooltip: tip({
              short: 'Sensitive attributes may be legally protected; proxies correlate with them.',
              intuition: 'Zip code, language, or device can encode group membership.',
              trap: '“We don’t use race” while correlated features remain.',
            }),
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'group-slices',
            label: 'Define groups and slices',
            tooltip: tip({
              short: 'Choose meaningful cohorts tied to stakeholder harm, not arbitrary cuts.',
              intuition: 'Fairness analysis is always relative to defined groups.',
              example: 'Region, language, age band, product tier—with sample size floors.',
              trap: 'Too-small groups produce unreliable rate estimates.',
            }),
          },
          {
            id: 'selection-rate',
            label: 'Selection / approval rate',
            tooltip: tip({
              short: 'Fraction of each group receiving the positive decision.',
              intuition: 'Demographic parity compares these rates across groups.',
              example: 'Approval rate 40% vs 22% between groups.',
              trap: 'Equal rates alone ignore qualification differences.',
            }),
          },
          {
            id: 'error-rate-parity',
            label: 'Error-rate parity',
            tooltip: tip({
              short: 'Compare FPR and FNR (or TPR) across groups.',
              intuition: 'Equal opportunity focuses on equal TPR; equalized odds pairs TPR and FPR.',
              example: 'Loan model flags qualified applicants in group A more often.',
              trap: 'Single-metric parity while other errors remain unequal.',
            }),
          },
          {
            id: 'calibration-by-group',
            label: 'Calibration by group',
            tooltip: tip({
              short: 'Among score bucket p, observed positive rate should match p within each group.',
              intuition: 'A model can rank fairly yet speak dishonest probabilities to one group.',
              trap: 'Global calibration hiding subgroup overconfidence.',
            }),
            lessonId: 'calibration',
          },
          {
            id: 'threshold-tradeoffs',
            label: 'Threshold tradeoffs',
            tooltip: tip({
              short: 'Moving cutoff shifts who bears false positives vs false negatives.',
              intuition: 'Fairness interventions often reallocate error costs.',
              example: 'Lower threshold helps recall for group B but raises FPR for group A.',
              trap: 'Picking threshold on majority group metrics only.',
            }),
          },
          {
            id: 'mitigation-options',
            label: 'Mitigation options',
            tooltip: tip({
              short: 'Pre-, in-, and post-processing: data balance, constrained training, group thresholds.',
              intuition: 'No free lunch—mitigation moves metrics, not magic.',
              example: 'Post-process thresholds per group to equalize TPR.',
              trap: 'Mitigation without stakeholder agreement on criteria.',
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'criteria-conflict',
            label: 'Criteria can conflict',
            tooltip: tip({
              short: 'Demographic parity, equalized odds, and calibration cannot all hold in general.',
              intuition: 'You must choose which error is least acceptable for the use case.',
              trap: 'Searching for one “fairness metric” to maximize.',
            }),
          },
          {
            id: 'harm-asymmetry',
            label: 'Harm asymmetry',
            tooltip: tip({
              short: 'False denial of opportunity and false flagging carry different ethical weights.',
              intuition: 'Context determines which metric gap matters most.',
              example: 'Medical screening vs ad targeting prioritize different errors.',
              trap: 'Treating all classification errors as equally costly.',
            }),
          },
          {
            id: 'proxy-persistence',
            label: 'Proxy persistence',
            tooltip: tip({
              short: 'Models exploit any signal correlated with the target, including proxies.',
              intuition: 'Fairness requires auditing outcomes, not just input columns.',
              trap: 'Checkbox compliance after dropping protected fields.',
            }),
          },
          {
            id: 'slice-not-global',
            label: 'Slice, not global only',
            tooltip: tip({
              short: 'Overall parity can hide worst-off minorities.',
              intuition: 'Intersectional slices reveal concentrated harm.',
              example: 'Global rates OK; elderly in region X face high FNR.',
              trap: 'Reporting one aggregate fairness number to regulators.',
            }),
          },
          {
            id: 'process-fairness',
            label: 'Process vs outcome',
            tooltip: tip({
              short: 'Stakeholders care about procedure, explanation, and recourse—not only rates.',
              intuition: 'Technical metrics supplement governance; they do not replace it.',
              trap: 'Assuming a parity chart equals organizational fairness.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'demographic-parity',
            label: 'Demographic parity gap',
            tooltip: tip({
              short: 'Compare P(ŷ=1 | group=a) vs P(ŷ=1 | group=b).',
              intuition: 'Measures equal selection rates, not equal error rates.',
              formula: 'gap = |P(ŷ=1|A) − P(ŷ=1|B)|',
              trap: 'Parity can conflict with merit-based base rates.',
            }),
          },
          {
            id: 'equal-opportunity',
            label: 'Equal opportunity',
            tooltip: tip({
              short: 'Equal true positive rate across groups among qualified positives.',
              intuition: 'Focuses on who gets correctly approved among those who should be.',
              formula: 'TPR_A ≈ TPR_B',
              trap: 'Silent on false positive disparities.',
            }),
          },
          {
            id: 'equalized-odds',
            label: 'Equalized odds',
            tooltip: tip({
              short: 'Match both TPR and FPR across groups.',
              intuition: 'Stronger than equal opportunity; harder to satisfy.',
              trap: 'Often incompatible with calibration when base rates differ.',
            }),
          },
          {
            id: 'group-ece',
            label: 'Group calibration (ECE)',
            tooltip: tip({
              short: 'Compute reliability / ECE separately per group.',
              intuition: 'Probability trust should not vary by demographics.',
              trap: 'Small groups → unstable calibration estimates.',
            }),
          },
          {
            id: 'postprocess-thresholds',
            label: 'Post-process thresholds',
            tooltip: tip({
              short: 'Learn group-specific cutoffs to satisfy a chosen constraint.',
              code: 'for each group g:\n  find threshold τ_g meeting TPR target',
              intuition: 'Same scores, different decision boundaries.',
              trap: 'Different thresholds need policy justification and monitoring.',
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'drop-column-trap',
            label: 'Drop-column “fix”',
            tooltip: tip({
              short: 'Removing protected attributes while proxies remain changes nothing fundamental.',
              intuition: 'Outcome audits matter more than input checklist.',
              trap: 'Legal sign-off based on feature list review only.',
            }),
          },
          {
            id: 'single-metric-fairness',
            label: 'Single-metric fairness',
            tooltip: tip({
              short: 'Optimizing one parity criterion while ignoring others and business costs.',
              intuition: 'Publish the full tradeoff table to stakeholders.',
              trap: 'Marketing “fair AI” from one chart.',
            }),
          },
          {
            id: 'tiny-group-stats',
            label: 'Tiny group statistics',
            tooltip: tip({
              short: 'Rates from handful of examples swing wildly.',
              intuition: 'Apply minimum-n rules and confidence intervals.',
              trap: 'Policy changes from 3-example slices.',
            }),
          },
          {
            id: 'ignore-base-rate',
            label: 'Ignore base-rate context',
            tooltip: tip({
              short: 'Parity gaps without prevalence context mislead.',
              intuition: 'Ask whether disparity reflects model, data, or world change.',
              trap: 'Demanding parity that contradicts verified ground truth differences.',
            }),
          },
          {
            id: 'fairness-once',
            label: 'One-time audit',
            tooltip: tip({
              short: 'Drift and retraining reintroduce disparity.',
              intuition: 'Fairness belongs in monitoring dashboards and release gates.',
              trap: 'Pre-launch report filed and forgotten.',
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'monitoring-fair-slices',
            label: 'Slice monitoring',
            tooltip: tip({
              short: 'Continuous group metrics and calibration in production.',
              intuition: 'Fairness regressions are operational incidents.',
              trap: 'Manual quarterly audits only.',
            }),
            lessonId: 'model-monitoring',
          },
          {
            id: 'uncertainty-defer',
            label: 'Deferral with uncertainty',
            tooltip: tip({
              short: 'Route ambiguous cases to human review instead of forced decisions.',
              intuition: 'Reduces harm when group error rates cannot yet be equalized.',
              trap: 'Deferring only disadvantaged groups—creates new bias.',
            }),
            lessonId: 'uncertainty-estimation',
          },
          {
            id: 'interpretability-audit',
            label: 'Explanation audits',
            tooltip: tip({
              short: 'Check whether reasons and attributions differ systematically by group.',
              intuition: 'Explanation disparity signals proxy reliance.',
              trap: 'Explanations shown to users without quality control.',
            }),
            lessonId: 'model-interpretability',
          },
          {
            id: 'ab-test-fairness',
            label: 'Experiment guardrails',
            tooltip: tip({
              short: 'Rollouts include group metric guardrails alongside global lift.',
              intuition: 'Experiments can improve average KPI while harming a slice.',
              trap: 'Ship winner on global metric with hidden slice harm.',
            }),
            lessonId: 'ab-testing-foundations',
          },
          {
            id: 'reliability-governance',
            label: 'Reliability governance',
            tooltip: tip({
              short: 'Fairness sits inside broader model reliability and risk review.',
              intuition: 'Connect metrics to escalation, documentation, and appeals.',
              trap: 'Fairness team isolated from model owners.',
            }),
            lessonId: 'model-reliability',
          },
        ],
      },
    ],
  },
  'uncertainty-estimation': {
    center: {
      id: 'uncertainty-estimation',
      label: 'Uncertainty Estimation',
      type: 'current',
      tooltip: tip({
        short: 'Uncertainty estimation quantifies when predictions are unreliable—via calibrated probabilities, intervals, ensemble disagreement, or out-of-distribution signals—so systems can defer, widen bounds, or escalate.',
        intuition: 'Confidence answers “how sure is the model?”; good uncertainty answers “should we act on this prediction?”',
        formula: 'decision = predict(x) if uncertainty(x) < τ else defer / widen / ask human',
        why: 'High-stakes ML needs abstention, human review, and honest risk communication—not raw softmax peaks.',
        trap: 'Maximum softmax probability is not a calibrated uncertainty measure.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'calibration-unc',
            label: 'Calibration',
            tooltip: tip({
              short: 'Probabilities must map to frequencies before they express confidence.',
              intuition: 'Uncertainty built on miscalibrated scores mis-ranks risk.',
              trap: '0.99 softmax on a wrong class is overconfidence, not certainty.',
            }),
            lessonId: 'calibration',
          },
          {
            id: 'variance-intuition',
            label: 'Variance and spread',
            tooltip: tip({
              short: 'Aleatoric noise is irreducible; epistemic uncertainty shrinks with more data.',
              intuition: 'Different uncertainty types need different tools.',
              example: 'Noisy labels vs unfamiliar input image.',
              trap: 'One number trying to capture both kinds.',
            }),
            lessonId: 'expected-value-variance',
          },
          {
            id: 'confidence-intervals',
            label: 'Confidence intervals',
            tooltip: tip({
              short: 'Intervals summarize sampling uncertainty around an estimate.',
              intuition: 'Prediction intervals extend the idea to future observations.',
              trap: 'Confusing confidence interval for individual prediction interval.',
            }),
            lessonId: 'sampling-confidence-intervals',
          },
          {
            id: 'softmax-scores',
            label: 'Softmax scores',
            tooltip: tip({
              short: 'Class probabilities from logits—useful after calibration, misleading alone.',
              intuition: 'Entropy of softmax can rank ambiguity if calibrated.',
              trap: 'Treating argmax probability as “model confidence.”',
            }),
            lessonId: 'softmax',
          },
          {
            id: 'ensemble-basics',
            label: 'Ensemble disagreement',
            tooltip: tip({
              short: 'Multiple models or dropout samples that disagree signal epistemic uncertainty.',
              intuition: 'Spread of predictions indicates lack of knowledge.',
              example: 'Five models vote 3–2 on class label.',
              trap: 'Identical clones disagreeing zero—need diverse members.',
            }),
          },
          {
            id: 'ood-preview',
            label: 'Out-of-distribution preview',
            tooltip: tip({
              short: 'Inputs far from training support should raise uncertainty regardless of softmax.',
              intuition: 'The model may be confidently wrong on novel data.',
              trap: 'High softmax on OOD adversarial or shifted inputs.',
            }),
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'epistemic-vs-aleatoric',
            label: 'Epistemic vs aleatoric',
            tooltip: tip({
              short: 'Epistemic: lack of knowledge; aleatoric: inherent randomness in labels.',
              intuition: 'More data helps epistemic; aleatoric may remain.',
              example: 'Blurry image (aleatoric) vs new product category (epistemic).',
              trap: 'Reporting one uncertainty bar without type.',
            }),
          },
          {
            id: 'probability-calibration-unc',
            label: 'Calibrated probabilities',
            tooltip: tip({
              short: 'Temperature scaling, Platt, or isotonic mapping align scores to frequencies.',
              intuition: 'Calibrated p enables threshold-based abstention.',
              trap: 'Calibrating on biased eval set misstates deployment uncertainty.',
            }),
          },
          {
            id: 'prediction-intervals',
            label: 'Prediction intervals',
            tooltip: tip({
              short: 'For regression: interval meant to cover future y with target coverage.',
              intuition: 'Wider interval communicates less precision.',
              example: '90% interval for demand forecast: [820, 1100].',
              trap: 'Intervals without coverage checks on fresh data.',
            }),
          },
          {
            id: 'ensemble-uncertainty',
            label: 'Deep ensembles / MC dropout',
            tooltip: tip({
              short: 'Sample multiple forward passes or models; use variance or vote entropy.',
              intuition: 'Disagreement flags epistemic uncertainty.',
              trap: 'Single-model dropout treated as full Bayesian posterior.',
            }),
          },
          {
            id: 'ood-detection',
            label: 'OOD detection',
            tooltip: tip({
              short: 'Score how far input lies from training manifold.',
              intuition: 'Combine with softmax: low OOD score + high softmax still risky.',
              example: 'Mahalanobis distance, energy score, or autoencoder error.',
              trap: 'OOD detector trained on wrong reference distribution.',
            }),
          },
          {
            id: 'abstention-policy',
            label: 'Abstention policy',
            tooltip: tip({
              short: 'When uncertainty exceeds τ, defer to human, fallback model, or safe default.',
              intuition: 'Policy connects estimates to actions.',
              example: 'Route top 5% entropy cases to review queue.',
              trap: 'Abstention that never reaches humans due to SLA pressure.',
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'confidence-vs-uncertainty',
            label: 'Confidence ≠ trust',
            tooltip: tip({
              short: 'A model can be confidently wrong; uncertainty tools surface that risk.',
              intuition: 'Trust requires calibration + OOD checks + domain validation.',
              trap: 'Showing users a single percentage from softmax.',
            }),
          },
          {
            id: 'when-to-defer',
            label: 'When to defer',
            tooltip: tip({
              short: 'Defer when error cost × uncertainty exceeds automation benefit.',
              intuition: 'Economics and ethics drive τ, not only math.',
              example: 'Low-cost spam filter vs high-cost medical triage.',
              trap: 'Same τ for all products and groups.',
            }),
          },
          {
            id: 'coverage-coverage',
            label: 'Coverage mindset',
            tooltip: tip({
              short: 'Intervals and abstention policies should meet promised coverage on live data.',
              intuition: 'Promised 90% intervals should cover ~90% over time.',
              trap: 'Validating intervals once offline forever.',
            }),
          },
          {
            id: 'shift-breaks-uncertainty',
            label: 'Shift breaks estimates',
            tooltip: tip({
              short: 'Distribution shift degrades both predictions and uncertainty scores.',
              intuition: 'Monitor uncertainty metrics alongside performance.',
              trap: 'Assuming calibration from last year holds today.',
            }),
          },
          {
            id: 'human-in-loop',
            label: 'Human-in-the-loop',
            tooltip: tip({
              short: 'Uncertainty prioritizes human attention to highest-risk cases.',
              intuition: 'Better than random audit sampling alone.',
              trap: 'Humans only see easy cases if queue sorting is wrong.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'temperature-scaling',
            label: 'Temperature scaling',
            tooltip: tip({
              short: 'Divide logits by T before softmax to calibrate confidence.',
              intuition: 'T > 1 softens overconfident peaks.',
              formula: 'p = softmax(z / T)',
              code: 'probs = softmax(logits / T)',
              trap: 'T fit on train set leaks optimism—use holdout.',
            }),
          },
          {
            id: 'predictive-entropy',
            label: 'Predictive entropy',
            tooltip: tip({
              short: 'H(p) = −Σ p_c log p_c measures class ambiguity.',
              intuition: 'High entropy → spread mass across classes.',
              trap: 'Entropy on uncalibrated p ranks poorly.',
            }),
          },
          {
            id: 'ensemble-variance',
            label: 'Ensemble variance',
            tooltip: tip({
              short: 'Variability of regression predictions or class probabilities across members.',
              intuition: 'Large variance → epistemic uncertainty signal.',
              code: 'unc = np.var([m(x) for m in models])',
              trap: 'Ensembles too similar → false low uncertainty.',
            }),
          },
          {
            id: 'conformal-sketch',
            label: 'Conformal prediction sketch',
            tooltip: tip({
              short: 'Use holdout nonconformity scores to build finite-sample coverage intervals.',
              intuition: 'Distribution-free coverage guarantees under exchangeability.',
              trap: 'Exchangeability fails under strong shift—coverage drops.',
            }),
          },
          {
            id: 'defer-rule',
            label: 'Defer rule',
            tooltip: tip({
              short: 'Act only if max(p) ≥ τ and ood_score ≤ δ.',
              code: 'if max(probs) < tau or ood(x) > delta: defer(x)',
              intuition: 'Combines class ambiguity and novelty.',
              trap: 'τ tuned only for average accuracy, not cost.',
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'softmax-max-trap',
            label: 'Softmax max trap',
            tooltip: tip({
              short: 'Argmax probability often looks high even when wrong.',
              intuition: 'Neural nets can be overconfident by default.',
              trap: 'Using max(p) as sole uncertainty in UI.',
            }),
          },
          {
            id: 'uncalibrated-abstention',
            label: 'Uncalibrated abstention',
            tooltip: tip({
              short: 'Abstaining on miscalibrated scores sends wrong cases to humans.',
              intuition: 'Calibrate before setting deferral thresholds.',
              trap: 'High abstention rate assumed to mean safe automation.',
            }),
          },
          {
            id: 'interval-without-coverage',
            label: 'Intervals without coverage audit',
            tooltip: tip({
              short: 'Pretty bands that cover 60% while claiming 90%.',
              intuition: 'Track empirical coverage on rolling labeled data.',
              trap: 'Assuming Gaussian residuals in heavy-tailed data.',
            }),
          },
          {
            id: 'ood-overtrust',
            label: 'Overtrust OOD scores',
            tooltip: tip({
              short: 'OOD detectors miss subtle shift and adversarial near-manifold points.',
              intuition: 'Use OOD as one signal in a stack.',
              trap: 'Green OOD light ends all scrutiny.',
            }),
          },
          {
            id: 'ignore-group-uncertainty',
            label: 'Ignore group uncertainty',
            tooltip: tip({
              short: 'Calibration and deferral rates may differ by cohort.',
              intuition: 'Fair uncertainty requires slice monitoring.',
              trap: 'Global τ hides underserved groups.',
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'monitoring-unc',
            label: 'Production monitoring',
            tooltip: tip({
              short: 'Track entropy, deferral rate, OOD fraction, and interval coverage live.',
              intuition: 'Uncertainty spikes often lead performance drops.',
              trap: 'Monitoring accuracy but not abstention quality.',
            }),
            lessonId: 'model-monitoring',
          },
          {
            id: 'fairness-deferral',
            label: 'Fair deferral',
            tooltip: tip({
              short: 'Ensure abstention and review queues do not burden groups unevenly.',
              intuition: 'Uncertainty policies have fairness side effects.',
              trap: 'Automating only “easy” groups.',
            }),
            lessonId: 'model-fairness',
          },
          {
            id: 'debugging-unc',
            label: 'Debugging with uncertainty',
            tooltip: tip({
              short: 'Prioritize error analysis on high-uncertainty wrong predictions.',
              intuition: 'Separates aleatoric noise from fixable epistemic gaps.',
              trap: 'Only reviewing low-uncertainty errors.',
            }),
            lessonId: 'model-debugging',
          },
          {
            id: 'reliability-unc',
            label: 'Model reliability',
            tooltip: tip({
              short: 'Trustworthy ML combines calibration, uncertainty, monitoring, and governance.',
              intuition: 'Uncertainty is one pillar of reliability, not an optional add-on.',
              trap: 'Shipping point predictions without an escalation path.',
            }),
            lessonId: 'model-reliability',
          },
          {
            id: 'rag-uncertainty',
            label: 'RAG failure modes',
            tooltip: tip({
              short: 'LLM pipelines use uncertainty and retrieval scores to reduce hallucination risk.',
              intuition: 'Low retrieval score + high answer confidence → danger pattern.',
              trap: 'Confident generation without evidence check.',
            }),
            lessonId: 'rag-failure-modes',
          },
        ],
      },
    ],
  },
  'model-reliability': {
    center: {
      id: 'model-reliability',
      label: 'Model Reliability',
      type: 'current',
      tooltip: tip({
        short: 'Model reliability is the practice of building ML systems that stay correct, calibrated, fair, explainable, and operable—from offline validation through production monitoring and incident response.',
        intuition: 'Reliability spans the whole lifecycle: debug before ship, monitor after ship, govern who is affected when things drift.',
        formula: 'trust = f(evaluation, calibration, monitoring, fairness, uncertainty, ops)',
        why: 'The model reliability track ties debugging, interpretability, monitoring, fairness, and uncertainty into one responsible-ML operating model.',
        trap: 'Strong offline accuracy alone is not reliability; silent drift, slice harm, and uncalibrated confidence break trust in production.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'classification-metrics-rel',
            label: 'Classification metrics',
            tooltip: tip({
              short: 'Core rates and confusion structure underpin every reliability check.',
              intuition: 'You cannot govern what you do not measure.',
              trap: 'Single global accuracy as north star.',
            }),
            lessonId: 'classification-metrics',
          },
          {
            id: 'calibration-rel',
            label: 'Calibration',
            tooltip: tip({
              short: 'Honest probabilities are the foundation for risk-aware decisions.',
              intuition: 'Reliability requires scores that mean what they say.',
              trap: 'Deploying ranking-only models when probabilities drive policy.',
            }),
            lessonId: 'calibration',
          },
          {
            id: 'leakage-rel',
            label: 'Data leakage awareness',
            tooltip: tip({
              short: 'Leakage creates false confidence that collapses in production.',
              intuition: 'Reliable eval is prerequisite to reliable deployment.',
              trap: 'Skipping leakage audit because offline metrics look great.',
            }),
            lessonId: 'data-leakage-deep-dive',
          },
          {
            id: 'train-serve-ops',
            label: 'Train / serve discipline',
            tooltip: tip({
              short: 'Versioned data, features, models, and reproducible pipelines.',
              intuition: 'Ops hygiene prevents untraceable regressions.',
              trap: 'Hand-edited features in production not in training.',
            }),
          },
          {
            id: 'stakeholder-context',
            label: 'Stakeholder context',
            tooltip: tip({
              short: 'Who is affected when the model errs? What recourse exists?',
              intuition: 'Reliability includes human impact, not only math.',
              trap: 'Purely technical metrics without domain review.',
            }),
          },
          {
            id: 'roc-pr-rel',
            label: 'ROC / PR curves',
            tooltip: tip({
              short: 'Threshold-free views support robust evaluation under shifting prevalence.',
              intuition: 'Reliability reviews should see ranking quality, not one cutoff.',
              trap: 'Choosing threshold once and never revisiting.',
            }),
            lessonId: 'roc-pr-curves',
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'validate-offline',
            label: 'Validate offline',
            tooltip: tip({
              short: 'Leakage checks, slices, baselines, calibration, and stress tests before launch.',
              intuition: 'Gate release on evidence, not enthusiasm.',
              example: 'Golden sets + subgroup metrics + calibration diagrams.',
              trap: 'Launching after one aggregate metric passes.',
            }),
          },
          {
            id: 'debug-incidents',
            label: 'Debug incidents',
            tooltip: tip({
              short: 'Systematic localization when metrics or behavior diverge.',
              intuition: 'Reliability teams need a shared debugging playbook.',
              trap: 'Permanent firefighting without postmortems.',
            }),
            lessonId: 'model-debugging',
          },
          {
            id: 'explain-decisions',
            label: 'Explain decisions',
            tooltip: tip({
              short: 'Interpretability supports audit, trust, and slice investigation.',
              intuition: 'Explanations are tools for review, not automatic justification.',
              trap: 'Showing attributions without stability checks.',
            }),
            lessonId: 'model-interpretability',
          },
          {
            id: 'monitor-live',
            label: 'Monitor live systems',
            tooltip: tip({
              short: 'Drift, performance, calibration, latency, cost, and alerts in production.',
              intuition: 'Reliability is continuous, not a launch-day checkbox.',
              trap: 'Dashboards without owners or runbooks.',
            }),
            lessonId: 'model-monitoring',
          },
          {
            id: 'govern-fairness',
            label: 'Govern fairness',
            tooltip: tip({
              short: 'Measure group outcomes, choose criteria, mitigate, and monitor.',
              intuition: 'Reliability includes equitable error allocation.',
              trap: 'Fairness as legal checkbox divorced from metrics.',
            }),
            lessonId: 'model-fairness',
          },
          {
            id: 'quantify-uncertainty',
            label: 'Quantify uncertainty',
            tooltip: tip({
              short: 'Calibrated confidence, intervals, OOD, and deferral when unsure.',
              intuition: 'Know when not to act autonomously.',
              trap: 'Automation without escalation path.',
            }),
            lessonId: 'uncertainty-estimation',
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'lifecycle-not-point',
            label: 'Lifecycle, not a point',
            tooltip: tip({
              short: 'Reliability work continues after the model file is saved.',
              intuition: 'Data, world, and code change under you.',
              trap: '“Ship and forget” culture.',
            }),
          },
          {
            id: 'slice-first-thinking',
            label: 'Slice-first thinking',
            tooltip: tip({
              short: 'Global averages hide failures that matter most to vulnerable cohorts.',
              intuition: 'Worst slice often defines real reliability.',
              trap: 'Reporting only mean KPI to leadership.',
            }),
          },
          {
            id: 'probabilities-matter',
            label: 'Probabilities matter',
            tooltip: tip({
              short: 'Many decisions use scores, not just labels.',
              intuition: 'Calibration and uncertainty are first-class reliability signals.',
              trap: 'Treating classifiers as rankers when policy uses thresholds.',
            }),
          },
          {
            id: 'human-oversight',
            label: 'Human oversight',
            tooltip: tip({
              short: 'Review queues, appeals, and kill switches are reliability features.',
              intuition: 'Automation boundary is a product choice.',
              trap: 'Removing humans to cut cost without risk analysis.',
            }),
          },
          {
            id: 'security-angle',
            label: 'Security and robustness',
            tooltip: tip({
              short: 'Adversarial inputs, poisoning, and prompt injection threaten reliability.',
              intuition: 'Safety extends beyond average-case metrics.',
              trap: 'Ignoring attack surface because offline accuracy is high.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'reliability-scorecard',
            label: 'Reliability scorecard',
            tooltip: tip({
              short: 'Track accuracy, calibration (ECE), slice gaps, deferral rate, latency SLO.',
              intuition: 'One page answers “is this system still trustworthy?”',
              trap: 'Scorecard with metrics nobody owns.',
            }),
          },
          {
            id: 'slice-metric-gap',
            label: 'Slice metric gap',
            tooltip: tip({
              short: 'gap = metric_worst_slice − metric_reference',
              intuition: 'Quantifies concentrated harm or regression.',
              trap: 'Slices too small for stable gaps.',
            }),
          },
          {
            id: 'release-gate-checklist',
            label: 'Release gate checklist',
            tooltip: tip({
              short: 'Binary gates: leakage audit, baseline beat, calibration, fairness review, rollback plan.',
              code: 'assert no_leakage and ece < 0.05 and rollback_ready',
              intuition: 'Automate what you can; document exceptions.',
              trap: 'Waiving gates under deadline without risk sign-off.',
            }),
          },
          {
            id: 'incident-severity',
            label: 'Incident severity rubric',
            tooltip: tip({
              short: 'Classify by user impact, breadth, reversibility, and duration.',
              intuition: 'Aligns monitoring alerts to response urgency.',
              trap: 'Every drift alert paged as critical.',
            }),
          },
          {
            id: 'coverage-defer-rate',
            label: 'Deferral and coverage KPIs',
            tooltip: tip({
              short: 'Track abstention volume, human turnaround, and interval empirical coverage.',
              intuition: 'Uncertainty policies need operational metrics.',
              trap: 'Deferral rate minimized to look “automated.”',
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'offline-equals-prod',
            label: 'Offline equals production',
            tooltip: tip({
              short: 'Eval set perfection while serve path diverges.',
              intuition: 'Reliability requires production-like validation.',
              trap: 'Different preprocessing in shadow vs live.',
            }),
          },
          {
            id: 'metric-silo',
            label: 'Metric silos',
            tooltip: tip({
              short: 'Platform watches latency; ML watches AUC; fairness audited once.',
              intuition: 'Reliability needs unified ownership and dashboards.',
              trap: 'No one sees calibration drift + latency spike together.',
            }),
          },
          {
            id: 'explain-away-risk',
            label: 'Explain away risk',
            tooltip: tip({
              short: 'Pretty attributions used to dismiss harm reports.',
              intuition: 'Explanation supports investigation; it does not close tickets alone.',
              trap: '“The model looked at legitimate features” as final answer.',
            }),
          },
          {
            id: 'fairness-reliability-split',
            label: 'Fairness vs reliability split',
            tooltip: tip({
              short: 'Treating fairness as separate from core SRE/ML ops.',
              intuition: 'Slice harm is a production incident.',
              trap: 'Different on-call for “fairness bugs.”',
            }),
          },
          {
            id: 'point-in-time-eval',
            label: 'Point-in-time eval only',
            tooltip: tip({
              short: 'No monitoring for drift, labels, or calibration rot.',
              intuition: 'Models age; data shifts; policies change.',
              trap: 'Annual model review replacing continuous monitors.',
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'debugging-lesson-rel',
            label: 'Model debugging',
            tooltip: tip({
              short: 'First response playbook when reliability signals fire.',
              intuition: 'Localize before retrain or rollback.',
              trap: 'Skipping root cause.',
            }),
            lessonId: 'model-debugging',
          },
          {
            id: 'monitoring-lesson-rel',
            label: 'Model monitoring',
            tooltip: tip({
              short: 'Continuous observability for the reliability scorecard.',
              intuition: 'Bridges validation and incident response.',
              trap: 'Alerts without runbooks.',
            }),
            lessonId: 'model-monitoring',
          },
          {
            id: 'fairness-lesson-rel',
            label: 'Model fairness',
            tooltip: tip({
              short: 'Group outcome governance within reliability program.',
              intuition: 'Equity metrics are release and monitor gates.',
              trap: 'Optional module mindset.',
            }),
            lessonId: 'model-fairness',
          },
          {
            id: 'uncertainty-lesson-rel',
            label: 'Uncertainty estimation',
            tooltip: tip({
              short: 'Escalation and coverage when autonomy should stop.',
              intuition: 'Pairs with calibration in trust stack.',
              trap: 'Point predictions only.',
            }),
            lessonId: 'uncertainty-estimation',
          },
          {
            id: 'interpretability-lesson-rel',
            label: 'Model interpretability',
            tooltip: tip({
              short: 'Audit and communication layer for reliable systems.',
              intuition: 'Supports debugging, fairness, and stakeholder review.',
              trap: 'Explanation as marketing only.',
            }),
            lessonId: 'model-interpretability',
          },
        ],
      },
    ],
  },
  'conditional-probability': {
    center: {
      id: 'conditional-probability',
      label: 'Conditional Probability',
      type: 'current',
      tooltip: tip({
        short: 'Conditional probability P(A|B) is the probability of A among outcomes where B already happened—it reweights the sample space by the evidence event.',
        intuition: 'Once B is known, impossible worlds drop out; every remaining probability must share the new denominator P(B).',
        formula: 'P(A\\mid B)=\\frac{P(A\\cap B)}{P(B)}',
        why: 'Conditioning is the language of Bayes rule, classification posteriors, filtering, and any ML update after new evidence.',
        trap: 'P(A|B) is not usually P(B|A); swapping condition and event reverses numerator and denominator roles.',
        example: 'If 10% of users churn and 40% of churners cancel billing, P(churn|cancel billing) ≠ P(cancel billing|churn).',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'probability-distributions-cp',
            label: 'Probability distributions',
            tooltip: tip({
              short: 'Probabilities assign mass to events on a sample space.',
              intuition: 'Conditioning stays inside the same probability model—only the active region shrinks.',
              example: 'Fair die: P(even)=1/2 before any condition.',
              trap: 'Probabilities must sum to 1 over the whole space, not over a condition alone.',
            }),
            lessonId: 'probability-distributions',
          },
          {
            id: 'sample-space-cp',
            label: 'Sample space',
            tooltip: tip({
              short: 'The set of all possible outcomes Ω.',
              intuition: 'Conditioning on B restricts attention to the subset B ⊆ Ω.',
              example: 'Medical test: Ω = {disease, no disease} × {+, −}.',
              trap: 'Forgetting which outcomes belong to B mis-scales every conditional.',
            }),
          },
          {
            id: 'events-intersection-cp',
            label: 'Events and intersection',
            tooltip: tip({
              short: 'A and B are subsets; A ∩ B means both occur.',
              intuition: 'The numerator counts outcomes where A and B overlap.',
              example: 'A = rain, B = cloudy → A ∩ B = rainy cloudy days.',
              trap: 'A ∩ B is not the same as A ∪ B.',
            }),
          },
          {
            id: 'marginal-probability-cp',
            label: 'Marginal probability P(B)',
            tooltip: tip({
              short: 'P(B) is the total probability of the conditioning event.',
              intuition: 'It becomes the new normalization constant for outcomes inside B.',
              example: 'P(B)=0.3 means 30% of mass lives in B.',
              trap: 'If P(B)=0, P(A|B) is undefined—division by zero.',
            }),
          },
          {
            id: 'joint-vs-conditional-cp',
            label: 'Joint vs conditional',
            tooltip: tip({
              short: 'P(A ∩ B) is a joint probability; P(A|B) is a reweighted conditional.',
              intuition: 'Joint counts overlap in the full space; conditional counts overlap relative to B.',
              example: 'P(A∩B)=0.06 and P(B)=0.2 → P(A|B)=0.3.',
              trap: 'Do not plug P(A) into the denominator.',
            }),
          },
          {
            id: 'independence-preview-cp',
            label: 'Independence preview',
            tooltip: tip({
              short: 'Independent events satisfy P(A|B)=P(A).',
              intuition: 'Learning B tells you nothing about A when they do not interact.',
              example: 'Fair coin flips: P(H2|H1)=P(H2).',
              trap: 'Assuming independence without checking is a common modeling bug.',
            }),
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'restrict-to-b-cp',
            label: 'Restrict to B',
            tooltip: tip({
              short: 'Conditioning throws away outcomes where B is false.',
              intuition: 'The new sample space is only the B slice.',
              example: 'Given “test positive,” only + outcomes matter for the next ratio.',
              trap: 'Outcomes outside B should get zero conditional mass.',
            }),
            highlightTarget: { panel: 'animation', type: 'restrict-to-b' },
          },
          {
            id: 'numerator-intersection-cp',
            label: 'Numerator P(A ∩ B)',
            tooltip: tip({
              short: 'Count outcomes where both A and B happen.',
              intuition: 'This is the overlap mass inside the restricted world.',
              formula: 'P(A\\cap B)',
              example: 'P(disease and +) in a confusion-table cell.',
              trap: 'Using P(A) alone ignores the requirement that B occurred.',
            }),
          },
          {
            id: 'denominator-pb-cp',
            label: 'Denominator P(B)',
            tooltip: tip({
              short: 'Renormalize so probabilities inside B sum to 1.',
              intuition: 'Every conditional is “share of B.”',
              formula: 'P(A\\mid B)=\\frac{P(A\\cap B)}{P(B)}',
              example: '0.06 / 0.20 = 0.30.',
              trap: 'A tiny P(B) makes conditional estimates very noisy.',
            }),
            highlightTarget: { panel: 'animation', type: 'denominator' },
          },
          {
            id: 'updated-probability-cp',
            label: 'Updated probability of A',
            tooltip: tip({
              short: 'P(A|B) can be higher or lower than P(A).',
              intuition: 'Evidence B shifts belief along the overlap structure.',
              example: 'Rare disease + positive test can still yield moderate P(disease|+).',
              trap: 'Higher P(A|B) does not prove causation from B to A.',
            }),
          },
          {
            id: 'partition-view-cp',
            label: 'Partition view',
            tooltip: tip({
              short: 'Conditioning on a partition splits the space into disjoint cases.',
              intuition: 'Law of total probability rebuilds marginals from conditionals.',
              formula: 'P(A)=\\sum_i P(A\\mid B_i)P(B_i)',
              example: 'Diagnose via P(A|B1)P(B1)+P(A|B2)P(B2).',
              trap: 'Partition pieces must be disjoint and cover relevant mass.',
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'information-update-cp',
            label: 'Information update',
            tooltip: tip({
              short: 'Conditioning means learning something new and revising odds.',
              intuition: 'Bayes rule is just structured conditioning with priors.',
              example: 'Posterior ∝ likelihood × prior uses the same ratio idea.',
              trap: 'Old P(A) is obsolete once B is observed—use P(A|B).',
            }),
          },
          {
            id: 'shrinking-world-cp',
            label: 'Shrinking world',
            tooltip: tip({
              short: 'Only B-world outcomes remain in play.',
              intuition: 'Think “zoom into the B row of a contingency table.”',
              example: 'Given paid user, free-user outcomes vanish from the denominator.',
              trap: 'Visualizing the full space while computing a conditional confuses scales.',
            }),
          },
          {
            id: 'not-symmetric-cp',
            label: 'Not symmetric',
            tooltip: tip({
              short: 'P(A|B) and P(B|A) answer different questions.',
              intuition: 'Evidence direction matters: symptom given disease ≠ disease given symptom.',
              example: 'P(+|disease) high but P(disease|+) moderate when disease is rare.',
              trap: 'Prosecutor’s fallacy swaps these deliberately or accidentally.',
            }),
          },
          {
            id: 'contingency-table-cp',
            label: 'Contingency table',
            tooltip: tip({
              short: 'Joint counts in cells make conditionals readable.',
              intuition: 'Divide a cell by its row or column total depending on what is given.',
              example: 'Row-normalize for P(column|row); column-normalize for P(row|column).',
              trap: 'Normalizing the wrong margin swaps the question being answered.',
            }),
          },
          {
            id: 'base-rate-cp',
            label: 'Base rate matters',
            tooltip: tip({
              short: 'Rare A means even good evidence may leave P(A|B) modest.',
              intuition: 'The denominator P(B) mixes rare hits with common false alarms.',
              example: 'Low disease prevalence limits posterior even with decent sensitivity.',
              trap: 'Ignoring prevalence is base-rate neglect.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'definition-formula-cp',
            label: 'Definition',
            tooltip: tip({
              short: 'P(A|B) = P(A ∩ B) / P(B) when P(B) > 0.',
              intuition: 'Overlap share of the conditioning event.',
              formula: 'P(A\\mid B)=\\frac{P(A\\cap B)}{P(B)}',
              example: 'Cell count / row total in empirical tables.',
              trap: 'Require P(B)>0 before dividing.',
            }),
            highlightTarget: { panel: 'code', type: 'formula' },
          },
          {
            id: 'counts-formula-cp',
            label: 'Count form',
            tooltip: tip({
              short: 'n(A∩B) / n(B) from finite samples estimates P(A|B).',
              intuition: 'Empirical conditioning is a ratio of counts inside B.',
              code: 'p_a_given_b = (a_and_b_count) / b_count',
              example: '50 conversions among 200 clicks → 0.25.',
              trap: 'Small B counts make ratio estimates unstable.',
            }),
          },
          {
            id: 'bayes-bridge-cp',
            label: 'Bayes bridge',
            tooltip: tip({
              short: 'P(A|B) P(B) = P(B|A) P(A)—same joint, two conditionals.',
              intuition: 'Swap which event is conditioned to translate questions.',
              formula: 'P(A\\mid B)=\\frac{P(B\\mid A)P(A)}{P(B)}',
              example: 'Sensitivity × prevalence feeds posterior disease risk.',
              trap: 'Need P(B) via total probability if not given directly.',
            }),
          },
          {
            id: 'numpy-conditional-cp',
            label: 'Filtering code',
            tooltip: tip({
              short: 'Filter rows where B is true, then measure A frequency.',
              intuition: 'Code mirrors the restrict-and-renormalize story.',
              code: 'mask = (B == True)\np_a_given_b = A[mask].mean()',
              example: 'pandas: df.loc[df["B"], "A"].mean()',
              trap: 'Filtering before computing P(B) on the wrong population biases ratios.',
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'inverse-trap-cp',
            label: 'Inverse confusion',
            tooltip: tip({
              short: 'P(A|B) ≠ P(B|A) in general.',
              intuition: 'They share P(A∩B) but divide by different marginals.',
              example: 'Most typos are human-written does not imply most human text is typos.',
              trap: 'Always ask which event is the evidence.',
            }),
          },
          {
            id: 'zero-denominator-cp',
            label: 'P(B)=0',
            tooltip: tip({
              short: 'Undefined conditional when evidence has zero probability.',
              intuition: 'You cannot renormalize an empty slice.',
              example: 'P(A|impossible event) is not 0—it is undefined.',
              trap: 'Regularization in code may hide this with pseudocounts.',
            }),
          },
          {
            id: 'assume-independence-cp',
            label: 'Assume independence',
            tooltip: tip({
              short: 'P(A|B)=P(A) only when A ⊥ B.',
              intuition: 'Correlation, confounding, or causal links break independence.',
              example: 'Click and purchase are not independent on the same session.',
              trap: 'Naive Bayes assumes feature independence—often approximate only.',
            }),
          },
          {
            id: 'ignore-base-rate-cp',
            label: 'Ignore base rate',
            tooltip: tip({
              short: 'Strong P(B|A) does not imply large P(A|B) when A is rare.',
              intuition: 'False positives dominate when prevalence is low.',
              example: '99% sensitivity with 1% prevalence still yields many false +.',
              trap: 'Screening stories must show prevalence explicitly.',
            }),
          },
          {
            id: 'conditioning-on-collider-cp',
            label: 'Conditioning on collider',
            tooltip: tip({
              short: 'Controlling for a common effect can invent spurious association.',
              intuition: 'Collider bias appears when you condition on a downstream variable.',
              example: 'Talent and luck both affect success; conditioning on success distorts talent–luck relation.',
              trap: 'Not every “control variable” is safe to condition on.',
            }),
          },
          {
            id: 'time-order-trap-cp',
            label: 'Time order',
            tooltip: tip({
              short: 'P(future|past) differs from P(past|future).',
              intuition: 'Prediction and explanation reverse conditioning direction.',
              example: 'P(rain|cloudy) ≠ P(cloudy|rain) as forecasting vs diagnosis.',
              trap: 'Using future features as “evidence” for past labels is leakage.',
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'bayes-rule-ml-used-cp',
            label: 'Bayes rule in ML',
            tooltip: tip({
              short: 'Posterior class probability is conditional probability with a model for P(x|y).',
              intuition: 'Classification thresholds act on P(y|x).',
              trap: 'Calibrated scores still need correct base rates at deployment.',
            }),
            lessonId: 'bayes-rule-ml',
          },
          {
            id: 'logistic-regression-used-cp',
            label: 'Logistic regression',
            tooltip: tip({
              short: 'Sigmoid outputs estimate P(y=1|x).',
              intuition: 'Features enter through log-odds, not raw marginals.',
              trap: 'Linear logits assume a specific link between features and conditional odds.',
            }),
            lessonId: 'logistic-regression',
          },
          {
            id: 'naive-bayes-used-cp',
            label: 'Naive Bayes',
            tooltip: tip({
              short: 'Class-conditional feature probabilities multiply into P(x|y).',
              intuition: 'Posterior combines likelihood pieces with priors via Bayes.',
              trap: 'Independence across features is a strong approximation.',
            }),
            lessonId: 'knn-naive-bayes-svm',
          },
          {
            id: 'hypothesis-testing-used-cp',
            label: 'Hypothesis testing',
            tooltip: tip({
              short: 'p-values are tail probabilities under a null conditional on that model.',
              intuition: '“Surprising if null true” is a conditional probability statement.',
              trap: 'p-value is not P(null true|data).',
            }),
            lessonId: 'hypothesis-testing-intuition',
          },
          {
            id: 'cross-entropy-used-cp',
            label: 'Cross-entropy',
            tooltip: tip({
              short: 'Training pushes model conditional P(y|x) toward observed labels.',
              intuition: 'Negative log conditional probability is the per-example loss.',
              trap: 'Overconfident wrong conditionals dominate the loss.',
            }),
            lessonId: 'cross-entropy',
          },
          {
            id: 'markov-used-cp',
            label: 'Markov chains',
            tooltip: tip({
              short: 'Next-state law is P(X_{t+1}|X_t)—pure conditional structure.',
              intuition: 'Memoryless updates chain conditionals across time.',
              trap: 'Hidden state breaks observed Markov property.',
            }),
            lessonId: 'markov-chains',
          },
          {
            id: 'causal-used-cp',
            label: 'Causal graphs',
            tooltip: tip({
              short: 'do-calculus asks which conditionals identify causal effects.',
              intuition: 'Observational P(y|x) may differ from interventional P(y|do(x)).',
              trap: 'Adjusting wrong variables biases causal estimates.',
            }),
            lessonId: 'causal-graphs-dags',
          },
        ],
      },
    ],
  },
  'conv-relu': {
    center: {
      id: 'conv-relu',
      label: 'Conv + ReLU',
      type: 'current',
      tooltip: tip({
        short: 'Conv + ReLU applies a sliding convolution filter to local patches, adds bias, then keeps only positive responses via ReLU—turning signed evidence into sparse feature detections.',
        intuition: 'Convolution proposes local pattern matches; ReLU gates which detections survive as active features.',
        formula: 'A=\\max(0, X*K+b)',
        why: 'This pair is the workhorse of early CNN vision stacks before pooling and deeper blocks.',
        trap: 'ReLU does not create features—it only zeroes negative pre-activations from convolution.',
        example: 'Edge filter with negative responses on dark side → ReLU zeros them, keeping bright-edge activations.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'conv2d-prereq-cr',
            label: 'Conv2D',
            tooltip: tip({
              short: 'Slide a kernel; each output is a weighted sum over a local patch.',
              intuition: 'Conv + ReLU stacks activation on top of that local linear response.',
              example: '3×3 kernel over RGB patch → one scalar pre-activation.',
              trap: 'Skipping conv understanding makes ReLU look like the pattern detector.',
            }),
            lessonId: 'conv2d',
          },
          {
            id: 'relu-prereq-cr',
            label: 'ReLU',
            tooltip: tip({
              short: 'ReLU(z)=max(0,z) passes positive values, zeros negatives.',
              intuition: 'Sparsity and nonlinearity enter after the linear conv response.',
              example: 'z=-0.4 → 0; z=1.2 → 1.2.',
              trap: 'Dead ReLU units stay at zero if never activated during training.',
            }),
            lessonId: 'relu',
          },
          {
            id: 'local-pattern-cr',
            label: 'Local pattern detector',
            tooltip: tip({
              short: 'Convolution responds when a patch aligns with learned weights.',
              intuition: 'Positive alignment yields positive pre-activation before ReLU.',
              example: 'Vertical edge kernel fires on vertical gradients.',
              trap: 'Opposite contrast can yield negative pre-activation, then ReLU silence.',
            }),
          },
          {
            id: 'bias-shift-cr',
            label: 'Bias shift',
            tooltip: tip({
              short: 'Per-filter bias b shifts the activation threshold.',
              intuition: 'Bias moves how much evidence is needed before ReLU passes signal.',
              example: 'Large negative bias suppresses weak matches entirely.',
              trap: 'Bias affects all spatial locations equally for that filter.',
            }),
          },
          {
            id: 'feature-map-cr',
            label: 'Feature map',
            tooltip: tip({
              short: 'A 2D grid of activations—one map per filter.',
              intuition: 'Conv+ReLU output is a stack of sparse detection maps.',
              example: '32 filters → 32 H×W feature maps.',
              trap: 'Zero cells are intentional sparsity, not missing data.',
            }),
          },
          {
            id: 'nn-stack-cr',
            label: 'Neural network layers',
            tooltip: tip({
              short: 'Conv+ReLU blocks compose into deeper vision networks.',
              intuition: 'Early blocks detect edges; later blocks combine lower features.',
              example: 'Conv-ReLU → pool → Conv-ReLU → …',
              trap: 'Depth without nonlinearity would collapse to one linear map.',
            }),
            lessonId: 'neural-network',
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'conv-step-cr',
            label: 'Convolution step',
            tooltip: tip({
              short: 'z = sum(patch ⊙ kernel) + b at each location.',
              intuition: 'Same kernel scans the whole input—parameter sharing.',
              formula: 'z_{i,j}=\\sum_{m,n} X_{i+m,j+n}K_{m,n}+b',
              example: 'Highlight one 3×3 window and its dot product with K.',
              trap: 'Stride and padding change which patches are visited.',
            }),
            highlightTarget: { panel: 'animation', type: 'convolution' },
          },
          {
            id: 'pre-activation-cr',
            label: 'Pre-activation z',
            tooltip: tip({
              short: 'Signed evidence before nonlinearity.',
              intuition: 'Negative z means anti-aligned or opposite-contrast match.',
              example: 'z=+2.1 strong match; z=-0.8 weak opposite response.',
              trap: 'Pre-activation sign depends on filter weights and input contrast.',
            }),
            highlightTarget: { panel: 'animation', type: 'pre-activation' },
          },
          {
            id: 'relu-gate-cr',
            label: 'ReLU gate',
            tooltip: tip({
              short: 'a = max(0, z) at every spatial location.',
              intuition: 'Only positive detections propagate; negatives become inactive.',
              formula: 'a=\\max(0,z)',
              example: 'Map goes sparse where z≤0 across large regions.',
              trap: 'All-negative feature maps become all zeros—not a bug.',
            }),
            highlightTarget: { panel: 'animation', type: 'relu' },
          },
          {
            id: 'spatial-sparsity-cr',
            label: 'Spatial sparsity',
            tooltip: tip({
              short: 'Inactive zeros highlight where the filter did not fire.',
              intuition: 'Sparsity saves compute in later layers and emphasizes salient regions.',
              example: 'Edge map nonzero only near boundaries.',
              trap: 'Too many dead filters reduce representational capacity.',
            }),
          },
          {
            id: 'multi-filter-cr',
            label: 'Many filters',
            tooltip: tip({
              short: 'Each filter learns a different local pattern.',
              intuition: 'Stack of maps = parallel detectors at every location.',
              example: 'One filter for horizontal edges, another for blobs.',
              trap: 'Filters are learned—not hand-designed after training.',
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'detect-then-gate-cr',
            label: 'Detect then gate',
            tooltip: tip({
              short: 'Convolution detects; ReLU decides what counts.',
              intuition: 'Two-stage story: linear evidence → nonlinear selection.',
              example: 'Weak negative evidence treated as “no detection.”',
              trap: 'ReLU cannot recover patterns convolution missed.',
            }),
          },
          {
            id: 'one-sided-evidence-cr',
            label: 'One-sided evidence',
            tooltip: tip({
              short: 'ReLU keeps only positive feature presence.',
              intuition: 'Opposite polarity needs another filter or signed activations elsewhere.',
              example: 'Dark-to-light vs light-to-dark edges may need separate filters.',
              trap: 'Assuming ReLU sees both polarities in one map is wrong.',
            }),
          },
          {
            id: 'translation-equiv-cr',
            label: 'Translation equivariance',
            tooltip: tip({
              short: 'Same filter response moves with the pattern in the input.',
              intuition: 'Detection location shifts when the object shifts.',
              example: 'Cat edge activates wherever the edge appears.',
              trap: 'Pooling later adds partial translation invariance.',
            }),
          },
          {
            id: 'contrast-sensitivity-cr',
            label: 'Contrast sensitivity',
            tooltip: tip({
              short: 'Input contrast and bias together set which cells activate.',
              intuition: 'Low-contrast images may yield fewer surviving ReLU units.',
              example: 'Increasing bias can activate more cells—sometimes noisily.',
              trap: 'Normalization layers often precede conv to stabilize this.',
            }),
          },
          {
            id: 'hierarchy-building-cr',
            label: 'Hierarchy building',
            tooltip: tip({
              short: 'Early Conv+ReLU maps feed deeper layers that compose motifs.',
              intuition: 'Simple parts combine into textures and object parts upstream.',
              example: 'Edges → corners → object fragments.',
              trap: 'One Conv+ReLU layer alone rarely solves complex vision.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'combined-formula-cr',
            label: 'Combined formula',
            tooltip: tip({
              short: 'A = max(0, X*K + b) elementwise after convolution.',
              intuition: 'Nonlinearity applies per spatial cell per channel.',
              formula: 'A=\\max(0, X*K+b)',
              example: 'PyTorch: F.relu(conv2d(x, w, b)).',
              trap: 'Broadcast bias correctly across H×W.',
            }),
            highlightTarget: { panel: 'code', type: 'formula' },
          },
          {
            id: 'pytorch-block-cr',
            label: 'PyTorch block',
            tooltip: tip({
              short: 'nn.Sequential(Conv2d, ReLU) is the idiomatic pair.',
              intuition: 'Conv produces z; ReLU module applies max(0,·).',
              code: 'nn.Sequential(\n  nn.Conv2d(in_c, out_c, 3, padding=1),\n  nn.ReLU(inplace=False)\n)',
              example: 'out_c filters → out_c sparse feature maps.',
              trap: 'inplace=True saves memory but complicates autograd hooks.',
            }),
            highlightTarget: { panel: 'code', type: 'block' },
          },
          {
            id: 'activation-stats-cr',
            label: 'Activation stats',
            tooltip: tip({
              short: 'Monitor fraction of zero activations after ReLU.',
              intuition: 'Extreme sparsity or all-dead maps signal init or LR issues.',
              code: 'dead_frac = (a == 0).float().mean()',
              example: 'Healthy early layers often show partial zeros, not 100%.',
              trap: 'BatchNorm before ReLU changes scale entering the gate.',
            }),
          },
          {
            id: 'filter-visual-cr',
            label: 'Filter inspection',
            tooltip: tip({
              short: 'Visualize conv weights to interpret what ReLU later gates.',
              intuition: 'Kernel image shows preferred local pattern polarity.',
              code: 'plt.imshow(model.conv1.weight[0,0].detach())',
              example: 'Gabor-like stripes in first-layer filters.',
              trap: 'Deep filters are less human-interpretable than early ones.',
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'relu-detects-trap-cr',
            label: 'ReLU detects patterns',
            tooltip: tip({
              short: 'Convolution detects; ReLU only thresholds.',
              intuition: 'Zero output may mean wrong sign, not absence of structure.',
              example: 'Flip input contrast → conv sign flips → ReLU may silence.',
              trap: 'Do not credit ReLU with learning edge templates.',
            }),
          },
          {
            id: 'dead-relu-trap-cr',
            label: 'Dead ReLU',
            tooltip: tip({
              short: 'Units stuck at zero for all inputs stop learning.',
              intuition: 'Gradient through ReLU is zero when z≤0.',
              example: 'Bad init + large LR can kill entire channels.',
              trap: 'Leaky ReLU and proper init mitigate dead units.',
            }),
            lessonId: 'leaky-relu',
          },
          {
            id: 'bias-only-trap-cr',
            label: 'Bias-only activation',
            tooltip: tip({
              short: 'Large bias can activate cells without real pattern match.',
              intuition: 'Constant offset passes ReLU everywhere if b>0 strongly.',
              example: 'All-ones feature map adds noise to downstream layers.',
              trap: 'Watch mean activation per filter during training.',
            }),
          },
          {
            id: 'signed-info-loss-cr',
            label: 'Signed information loss',
            tooltip: tip({
              short: 'ReLU discards negative pre-activations permanently at that layer.',
              intuition: 'Opposite-contrast evidence is thrown away, not inverted.',
              example: 'Single filter cannot encode both edge polarities post-ReLU.',
              trap: 'Need paired filters or later layers to recover polarity context.',
            }),
          },
          {
            id: 'forget-padding-cr',
            label: 'Padding confusion',
            tooltip: tip({
              short: 'Same conv+ReLU with different padding changes map size and border behavior.',
              intuition: 'Border cells use zero-padded or reflected patches.',
              example: 'valid vs same padding output shapes differ.',
              trap: 'Downstream layer shapes depend on this choice.',
            }),
          },
          {
            id: 'scale-trap-cr',
            label: 'Input scale',
            tooltip: tip({
              short: 'Unnormalized pixel scale shifts pre-activation distribution.',
              intuition: 'Tiny inputs may yield all-zero ReLU maps early in training.',
              example: '[0,1] vs [0,255] inputs need consistent scaling.',
              trap: 'Pair with normalization or careful weight init.',
            }),
            lessonId: 'feature-scaling-preprocessing',
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'max-pooling-used-cr',
            label: 'Max pooling',
            tooltip: tip({
              short: 'Pool shrinks Conv+ReLU maps while keeping strongest local activations.',
              intuition: 'Sparsity from ReLU makes pooling pick salient peaks.',
              trap: 'Pool size and stride control information loss.',
            }),
            lessonId: 'max-pooling',
          },
          {
            id: 'deep-cnn-used-cr',
            label: 'Deep CNN stacks',
            tooltip: tip({
              short: 'Repeated Conv-ReLU-(pool) blocks build hierarchical vision features.',
              intuition: 'Modern backbones stack dozens of such units.',
              example: 'ResNet blocks still use conv+activation patterns.',
              trap: 'Skip connections address vanishing signal, not replace conv.',
            }),
          },
          {
            id: 'batchnorm-used-cr',
            label: 'Batch normalization',
            tooltip: tip({
              short: 'BatchNorm stabilizes scale before or after activation in modern blocks.',
              intuition: 'Conv-BN-ReLU ordering is a common training recipe.',
              trap: 'Train/eval BN behavior must match at serve time.',
            }),
            lessonId: 'dropout-batchnorm',
          },
          {
            id: 'backprop-used-cr',
            label: 'Backpropagation',
            tooltip: tip({
              short: 'Gradients flow through ReLU only where pre-activation was positive.',
              intuition: 'Dead units receive zero gradient from ReLU gate.',
              trap: 'Gradient clipping does not fix dead channels alone.',
            }),
            lessonId: 'computation-graph-backprop',
          },
          {
            id: 'unet-used-cr',
            label: 'U-Net / diffusion encoders',
            tooltip: tip({
              short: 'Encoder towers use Conv+ReLU (or similar) to build multiscale features.',
              intuition: 'Same local detection story at multiple resolutions.',
              trap: 'Diffusion U-Nets often use GroupNorm + SiLU variants.',
            }),
            lessonId: 'unet-vs-dit',
          },
          {
            id: 'interpretability-used-cr',
            label: 'Activation maps',
            tooltip: tip({
              short: 'Visualizing post-ReLU maps shows where filters fired.',
              intuition: 'Class activation methods build on these sparse maps.',
              trap: 'Activation ≠ causal importance for the final class.',
            }),
            lessonId: 'model-interpretability',
          },
        ],
      },
    ],
  },
  conv2d: {
    center: {
      id: 'conv2d',
      label: 'Conv2D',
      type: 'current',
      tooltip: tip({
        short: 'Conv2D slides a small kernel across a 2D input, computing a weighted sum at each stop—reusing the same filter everywhere to detect local spatial patterns.',
        intuition: 'Each output cell asks how much the local patch matches the kernel; stride and padding control where the window lands and how output size changes.',
        formula: 'Y_{i,j}=\\sum_m\\sum_n X_{i+m,j+n}K_{m,n}',
        why: 'Conv2D is the foundation of CNN vision, many audio spectrogram models, and spatial inductive bias in deep learning.',
        trap: 'Convolution reuses one kernel—it is not a separate dense weight matrix for every patch location.',
        example: 'A 3×3 edge kernel on a 5×5 image with stride 1 and valid padding yields a 3×3 output map.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'matrix-mult-conv',
            label: 'Matrix multiplication',
            tooltip: tip({
              short: 'Each conv output is a dot product between flattened patch and kernel.',
              intuition: 'Convolution is structured sparse matrix multiply with sharing.',
              example: '9 patch pixels · 9 kernel weights → one scalar.',
              trap: 'Im2col expands conv to GEMM in optimized libraries.',
            }),
            lessonId: 'matrix-multiplication',
          },
          {
            id: 'dot-product-conv',
            label: 'Dot product',
            tooltip: tip({
              short: 'Align patch and kernel; multiply and sum.',
              intuition: 'High dot product = strong local match.',
              example: 'Identical patch and kernel → large positive response.',
              trap: 'Opposite patches yield negative responses before activation.',
            }),
          },
          {
            id: 'grid-indexing-conv',
            label: '2D grid indexing',
            tooltip: tip({
              short: 'Images are H×W (× channels) arrays with row/column coordinates.',
              intuition: 'Kernel anchor (i,j) picks which output cell is being computed.',
              example: 'Output[0,0] uses top-left patch when padding allows.',
              trap: 'Row/column order must match framework conventions.',
            }),
          },
          {
            id: 'kernel-concept-conv',
            label: 'Kernel / filter',
            tooltip: tip({
              short: 'Small learned weight window shared across all locations.',
              intuition: 'One kernel = one feature detector scanned everywhere.',
              example: '3×3 kernel has 9 learnable weights (+ bias).',
              trap: 'Kernel size trades receptive field vs parameter count.',
            }),
          },
          {
            id: 'channels-conv',
            label: 'Input channels',
            tooltip: tip({
              short: 'RGB has 3 channels; each output filter sums over all input channels.',
              intuition: 'Depth dimension stacks separate per-channel products into one response.',
              example: 'in_channels=3, out_channels=16 → 16 different 3×3×3 kernels.',
              trap: 'Channel depth must match between layers.',
            }),
          },
          {
            id: 'nn-preview-conv',
            label: 'Neural network preview',
            tooltip: tip({
              short: 'Conv layers replace dense layers when locality matters.',
              intuition: 'Parameter sharing makes spatial models efficient.',
              example: 'MNIST/CIFAR pipelines start with Conv2D stacks.',
              trap: 'Fully connected layers on raw pixels ignore spatial structure.',
            }),
            lessonId: 'neural-network',
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'sliding-window-conv',
            label: 'Sliding window',
            tooltip: tip({
              short: 'Kernel visits each output location according to stride.',
              intuition: 'Same weights evaluate every patch—translation equivariance.',
              example: 'Stride 2 skips every other position → smaller output.',
              trap: 'Stride>1 downsamples and may miss thin structures.',
            }),
            highlightTarget: { panel: 'animation', type: 'sliding-window' },
          },
          {
            id: 'patch-sum-conv',
            label: 'Patch weighted sum',
            tooltip: tip({
              short: 'Multiply aligned entries and accumulate.',
              intuition: 'This is the local linear filter response.',
              formula: 'Y_{i,j}=\\sum_{m,n} X_{i+m,j+n}K_{m,n}',
              example: 'Highlight 3×3 window and show sum of products.',
              trap: 'Off-by-one in patch extraction shifts detections.',
            }),
            highlightTarget: { panel: 'animation', type: 'patch-sum' },
          },
          {
            id: 'stride-conv',
            label: 'Stride',
            tooltip: tip({
              short: 'Step size between kernel placements.',
              intuition: 'Larger stride → fewer output cells → coarser map.',
              example: 'stride=2 halves spatial resolution (with same padding).',
              trap: 'Stride interacts with pooling—double downsampling risk.',
            }),
            highlightTarget: { panel: 'animation', type: 'stride' },
          },
          {
            id: 'padding-conv',
            label: 'Padding',
            tooltip: tip({
              short: 'Virtual border values so kernels can reach edges.',
              intuition: 'Same padding preserves H×W; valid padding shrinks output.',
              example: 'Zero pad 1 pixel on 5×5 with 3×3 kernel → 5×5 out (same).',
              trap: 'Reflect vs zero pad changes border behavior.',
            }),
            highlightTarget: { panel: 'animation', type: 'padding' },
          },
          {
            id: 'output-shape-conv',
            label: 'Output shape',
            tooltip: tip({
              short: 'H_out and W_out depend on input size, kernel, stride, padding.',
              intuition: 'Shape bugs break the next layer’s wiring.',
              formula: 'H_{out}=\\lfloor(H+2p-k)/s\\rfloor+1',
              example: '5×5, k=3, p=0, s=1 → 3×3.',
              trap: 'Framework formulas differ slightly on ceil/floor conventions.',
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'locality-conv',
            label: 'Locality',
            tooltip: tip({
              short: 'Each output depends only on a small neighborhood.',
              intuition: 'Nearby pixels matter most for edges, textures, parts.',
              example: '3×3 sees 9 pixels; receptive field grows with depth.',
              trap: 'Global context needs stacking layers or attention.',
            }),
          },
          {
            id: 'parameter-sharing-conv',
            label: 'Parameter sharing',
            tooltip: tip({
              short: 'One kernel scans the whole image—far fewer weights than dense.',
              intuition: 'Same edge detector works top-left and bottom-right.',
              example: '3×3 kernel = 9 weights reused H×W times.',
              trap: 'Sharing assumes patterns are translation-stable.',
            }),
          },
          {
            id: 'receptive-field-conv',
            label: 'Receptive field',
            tooltip: tip({
              short: 'Deeper layers indirectly see larger input regions.',
              intuition: 'Stacked 3×3 convs expand effective window multiplicatively.',
              example: 'Two 3×3 layers ≈ 5×5 receptive field.',
              trap: 'Dilation increases field without larger kernels.',
            }),
          },
          {
            id: 'depth-stack-conv',
            label: 'Depth stacks filters',
            tooltip: tip({
              short: 'out_channels parallel detectors at every spatial cell.',
              intuition: 'Each filter specializes on a different local motif.',
              example: '16 filters → 16-dimensional feature vector per pixel.',
              trap: 'Too few filters underfit; too many overfit small data.',
            }),
          },
          {
            id: 'equivariance-conv',
            label: 'Translation equivariance',
            tooltip: tip({
              short: 'Shift input → output shifts correspondingly (before pooling).',
              intuition: 'Detection location tracks pattern location.',
              example: 'Move digit in MNIST → activation map moves.',
              trap: 'Pooling adds approximate invariance, not equivariance.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'conv-formula-conv',
            label: '2D convolution',
            tooltip: tip({
              short: 'Sum of products over kernel support at each output coordinate.',
              intuition: 'Double sum over kernel indices m,n.',
              formula: 'Y_{i,j}=\\sum_m\\sum_n X_{i+m,j+n}K_{m,n}',
              example: 'Compute Y[0,0] from top-left patch manually.',
              trap: 'Cross-correlation vs convolution flip conventions differ by library.',
            }),
            highlightTarget: { panel: 'code', type: 'formula' },
          },
          {
            id: 'pytorch-conv2d-conv',
            label: 'nn.Conv2d',
            tooltip: tip({
              short: 'PyTorch Conv2d(in_c, out_c, kernel, stride, padding).',
              intuition: 'Weight shape [out_c, in_c, kH, kW].',
              code: 'conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)\ny = conv(x)  # x: [B,3,H,W]',
              example: 'padding=1 preserves H,W for 3×3 kernel.',
              trap: 'bias=False still needs BatchNorm or bias elsewhere.',
            }),
            highlightTarget: { panel: 'code', type: 'conv2d' },
          },
          {
            id: 'output-size-code-conv',
            label: 'Output size helper',
            tooltip: tip({
              short: 'Compute H_out,W_out before wiring the next layer.',
              intuition: 'Prevents runtime shape errors in deep stacks.',
              code: 'def out_size(h, k, p, s):\n  return (h + 2*p - k) // s + 1',
              example: 'out_size(32, 3, 1, 2) → 16.',
              trap: 'Dilated convolution uses effective kernel size.',
            }),
          },
          {
            id: 'im2col-note-conv',
            label: 'im2col / GEMM',
            tooltip: tip({
              short: 'Efficient implementations reshape patches into matrix multiply.',
              intuition: 'Same math, faster hardware utilization.',
              code: '# Conceptual: patches_matrix @ kernel_matrix',
              example: 'cuDNN uses optimized conv algorithms transparently.',
              trap: 'Memory spikes on im2col for huge feature maps.',
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'dense-per-patch-trap-conv',
            label: 'Not dense per patch',
            tooltip: tip({
              short: 'Same kernel everywhere—unlike unique weights per location.',
              intuition: 'Dense layer on flattened patches explodes parameter count.',
              example: '224×224 RGB → impractical fully connected first layer.',
              trap: '1×1 conv is dense across channels, still shared spatially.',
            }),
          },
          {
            id: 'shape-mismatch-conv',
            label: 'Shape mismatch',
            tooltip: tip({
              short: 'in_channels must match previous layer out_channels.',
              intuition: 'Conv weight depth dimension must align.',
              example: 'Conv 3→16 then Conv 16→32 is valid; 3→32 after 16 is not.',
              trap: 'Transposed conv upsampling has its own shape rules.',
            }),
          },
          {
            id: 'padding-mode-trap-conv',
            label: 'Padding mode',
            tooltip: tip({
              short: 'valid vs same vs explicit pad changes borders.',
              intuition: 'Edge pixels participate in fewer products without padding.',
              example: 'Corner output uses partial patch in valid conv.',
              trap: 'Asymmetric pad breaks center alignment assumptions.',
            }),
          },
          {
            id: 'correlation-vs-conv-trap',
            label: 'Correlation vs convolution',
            tooltip: tip({
              short: 'ML “conv” often means cross-correlation (no kernel flip).',
              intuition: 'Learned kernels adapt regardless of naming.',
              example: 'PyTorch Conv2d correlates by default.',
              trap: 'Signal-processing convolution flips the kernel.',
            }),
          },
          {
            id: 'dilation-stride-trap-conv',
            label: 'Stride vs dilation',
            tooltip: tip({
              short: 'Stride subsamples outputs; dilation spaces kernel taps.',
              intuition: 'Both enlarge receptive field differently.',
              example: 'dilation=2 on 3×3 skips every other input pixel.',
              trap: 'Combining stride and dilation needs careful shape planning.',
            }),
          },
          {
            id: 'no-activation-trap-conv',
            label: 'Linear-only stack',
            tooltip: tip({
              short: 'Stacking conv without nonlinearity collapses to one linear filter.',
              intuition: 'Depth needs activation between conv layers.',
              example: 'Conv+Conv equals single conv mathematically without σ.',
              trap: 'Always pair with ReLU or similar in hidden stacks.',
            }),
            lessonId: 'conv-relu',
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'conv-relu-used-conv',
            label: 'Conv + ReLU',
            tooltip: tip({
              short: 'Nonlinearity follows conv to build sparse feature maps.',
              intuition: 'Standard vision block after this lesson.',
              trap: 'Activation choice affects gradient flow.',
            }),
            lessonId: 'conv-relu',
          },
          {
            id: 'max-pool-used-conv',
            label: 'Max pooling',
            tooltip: tip({
              short: 'Downsample conv feature maps by local max.',
              intuition: 'Reduces spatial size and adds translation tolerance.',
              trap: 'Too much pooling loses spatial detail.',
            }),
            lessonId: 'max-pooling',
          },
          {
            id: 'resnet-used-conv',
            label: 'ResNet / CNN backbones',
            tooltip: tip({
              short: 'Modern vision models stack many conv blocks with skips.',
              intuition: 'Same conv mechanics at scale with residual paths.',
              trap: 'Depth needs normalization and good init.',
            }),
          },
          {
            id: 'unet-encoder-used-conv',
            label: 'U-Net encoder',
            tooltip: tip({
              short: 'Diffusion and segmentation encoders downsample via conv stacks.',
              intuition: 'Multiscale conv features feed decoders or denoisers.',
              trap: 'Channel width often doubles when spatial size halves.',
            }),
            lessonId: 'unet-vs-dit',
          },
          {
            id: 'clip-vision-used-conv',
            label: 'Vision encoders',
            tooltip: tip({
              short: 'CLIP and ViT hybrids still teach conv inductive bias history.',
              intuition: 'Local conv vs global attention is an architecture tradeoff.',
              trap: 'ViT patches replace conv on some frontiers—not everywhere.',
            }),
            lessonId: 'clip-encoder',
          },
          {
            id: 'backprop-used-conv',
            label: 'Backprop through conv',
            tooltip: tip({
              short: 'Gradients distribute to kernel weights and input patches.',
              intuition: 'Backward conv rotates/flips gradient patches symmetrically.',
              trap: 'Im2col backward must match forward layout.',
            }),
            lessonId: 'computation-graph-backprop',
          },
        ],
      },
    ],
  },
  'cross-validation': {
    center: {
      id: 'cross-validation',
      label: 'Cross-Validation & Data Leakage',
      type: 'current',
      tooltip: tip({
        short: 'Cross-validation rotates which fold acts as validation while the rest train, averages scores across rotations, and refits the whole pipeline inside each fold to estimate generalization without wasting data.',
        intuition: 'Each row (or group) gets one turn being “examined” while others teach the model—honest only if preprocessing and feature engineering stay inside the training folds.',
        formula: '\\operatorname{CV}_k=\\frac{1}{k}\\sum_{i=1}^{k} score_i',
        why: 'CV stabilizes model selection, hyperparameter tuning, and leakage audits compared with a single lucky or unlucky validation split.',
        trap: 'Preprocessing before the fold split lets validation statistics leak into training—CV does not fix that boundary mistake.',
        example: '5-fold CV with accuracy [0.82, 0.79, 0.81, 0.80, 0.78] → mean 0.80, std shows stability.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'train-val-test-cv',
            label: 'Train / val / test split',
            tooltip: tip({
              short: 'Train fits, validation tunes, test evaluates once at the end.',
              intuition: 'CV replaces a single validation draw with k rotated estimates.',
              example: 'Small data often skips large holdout in favor of CV.',
              trap: 'Test set must remain untouched during CV tuning.',
            }),
            lessonId: 'train-validation-test-split',
          },
          {
            id: 'data-leakage-cv',
            label: 'Data leakage',
            tooltip: tip({
              short: 'Evaluation information must not influence training folds.',
              intuition: 'Leakage makes CV scores optimistically biased.',
              example: 'Fit scaler on all data before CV inflates scores.',
              trap: 'Entity duplicates across folds mimic leakage.',
            }),
            lessonId: 'data-leakage-deep-dive',
          },
          {
            id: 'model-selection-cv',
            label: 'Model selection',
            tooltip: tip({
              short: 'Compare models or hyperparameters using validation performance.',
              intuition: 'CV mean score ranks candidates more reliably than one split.',
              example: 'Pick λ with lowest average CV error.',
              trap: 'Nested CV separates selection from performance estimation.',
            }),
          },
          {
            id: 'metric-choice-cv',
            label: 'Evaluation metric',
            tooltip: tip({
              short: 'Accuracy, AUC, RMSE—metric must match the task.',
              intuition: 'CV averages the same metric computed on each fold’s validation slice.',
              example: 'Imbalanced data → F1 or PR-AUC instead of accuracy.',
              trap: 'Optimizing the wrong metric optimizes the wrong behavior.',
            }),
            lessonId: 'classification-metrics',
          },
          {
            id: 'pipeline-refit-cv',
            label: 'Pipeline refit',
            tooltip: tip({
              short: 'Each fold retrains weights and refits preprocessors on training folds only.',
              intuition: 'CV score is meaningless if the same fitted scaler spans folds.',
              example: 'sklearn Pipeline + cross_val_score enforces this.',
              trap: 'Manual steps outside the pipeline leak easily.',
            }),
          },
          {
            id: 'group-structure-cv',
            label: 'Group / time structure',
            tooltip: tip({
              short: 'Splits must respect users, sessions, or chronology when needed.',
              intuition: 'Random row CV fails when rows are correlated.',
              example: 'GroupKFold keeps all rows of one user in one fold.',
              trap: 'IID assumption is often false in real ML data.',
            }),
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'k-folds-cv',
            label: 'k folds',
            tooltip: tip({
              short: 'Partition data into k disjoint subsets of roughly equal size.',
              intuition: 'Typical k is 5 or 10; leave-one-out is k=n.',
              example: '5-fold → 80% train, 20% val each rotation.',
              trap: 'Tiny folds make validation metrics very noisy.',
            }),
            highlightTarget: { panel: 'animation', type: 'k-folds' },
          },
          {
            id: 'rotate-validation-cv',
            label: 'Rotate validation',
            tooltip: tip({
              short: 'Each fold i serves once as validation; others train.',
              intuition: 'Every example contributes to validation exactly once.',
              example: 'Fold 3 val → train on folds 1,2,4,5.',
              trap: 'Shuffling before split spreads difficulty—unless time order matters.',
            }),
            highlightTarget: { panel: 'animation', type: 'rotate' },
          },
          {
            id: 'score-each-fold-cv',
            label: 'Score each fold',
            tooltip: tip({
              short: 'Compute validation metric after training on k−1 folds.',
              intuition: 'score_i measures generalization on fold i’s held-out slice.',
              example: 'Fold errors: 0.21, 0.19, 0.22, 0.20, 0.18.',
              trap: 'Training on too little data per fold raises variance.',
            }),
          },
          {
            id: 'average-cv-score-cv',
            label: 'Average CV score',
            tooltip: tip({
              short: 'CV_k = mean(score_i); std shows stability across folds.',
              intuition: 'Mean is the model selection statistic; std flags instability.',
              formula: '\\operatorname{CV}_k=\\frac{1}{k}\\sum_{i=1}^{k} score_i',
              example: 'Mean 0.80 ± 0.02 looks stable; ± 0.15 looks noisy.',
              trap: 'High variance means the model is split-sensitive.',
            }),
            highlightTarget: { panel: 'animation', type: 'average' },
          },
          {
            id: 'strict-pipeline-cv',
            label: 'Strict pipeline',
            tooltip: tip({
              short: 'All fit steps occur inside training folds for that rotation.',
              intuition: 'Validation fold must be truly unseen for every transformation.',
              example: 'Impute means from train fold only, apply to val fold.',
              trap: 'Feature selection on all data before CV is leakage.',
            }),
            highlightTarget: { panel: 'animation', type: 'pipeline' },
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'everyone-gets-exam-cv',
            label: 'Everyone takes the exam',
            tooltip: tip({
              short: 'Each example sits in validation exactly once across rotations.',
              intuition: 'Uses data more efficiently than one fixed 80/20 split.',
              example: '1000 rows → each appears in 200-row validation once in 5-fold.',
              trap: 'Still not free data—test set stays separate.',
            }),
          },
          {
            id: 'variance-reduction-cv',
            label: 'Variance reduction',
            tooltip: tip({
              short: 'Averaging k validation scores smooths one lucky split.',
              intuition: 'Selection based on mean CV is less brittle.',
              example: 'Single split 0.95 vs CV mean 0.81 reveals overfitting to split.',
              trap: 'CV variance remains when groups or time structure ignored.',
            }),
          },
          {
            id: 'selection-bias-cv',
            label: 'Selection bias',
            tooltip: tip({
              short: 'Choosing the best of many models using CV still consumes information.',
              intuition: 'Nested CV or a fresh test set validates the final pick.',
              example: 'Try 50 λ values → pick best CV → confirm on test once.',
              trap: 'Repeatedly peeking at test defeats its purpose.',
            }),
          },
          {
            id: 'grouped-intuition-cv',
            label: 'Grouped folds',
            tooltip: tip({
              short: 'Keep related rows together so validation simulates new entities.',
              intuition: 'User-level generalization needs user-level folds.',
              example: 'All sessions of user 42 stay in one fold.',
              trap: 'Random CV on repeated users overstates performance.',
            }),
          },
          {
            id: 'time-series-intuition-cv',
            label: 'Time-aware CV',
            tooltip: tip({
              short: 'Future must not appear in training when forecasting.',
              intuition: 'Rolling-origin or blocked time splits respect causality.',
              example: 'Train on Jan–Jun, validate Jul; slide window forward.',
              trap: 'Shuffled k-fold on time series leaks future into past.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'cv-mean-formula-cv',
            label: 'CV mean',
            tooltip: tip({
              short: 'Average validation score across k fold models.',
              intuition: 'Each score_i comes from a refit model.',
              formula: '\\operatorname{CV}_k=\\frac{1}{k}\\sum_{i=1}^{k} score_i',
              example: 'sklearn cross_val_score returns array of fold scores.',
              trap: 'Maximizing score vs minimizing loss—sign matters.',
            }),
            highlightTarget: { panel: 'code', type: 'formula' },
          },
          {
            id: 'sklearn-cv-cv',
            label: 'sklearn cross_val_score',
            tooltip: tip({
              short: 'Wrap estimator in Pipeline; CV handles fold rotation.',
              intuition: 'cv=5 default; use GroupKFold or TimeSeriesSplit when needed.',
              code: 'from sklearn.model_selection import cross_val_score\nscores = cross_val_score(pipeline, X, y, cv=5)',
              example: 'scores.mean(), scores.std()',
              trap: 'Pass groups= for GroupKFold via cross_validate.',
            }),
            highlightTarget: { panel: 'code', type: 'sklearn' },
          },
          {
            id: 'group-kfold-cv',
            label: 'GroupKFold',
            tooltip: tip({
              short: 'Same group never appears in both train and val within a fold.',
              intuition: 'groups vector labels entity id per row.',
              code: 'from sklearn.model_selection import GroupKFold\ngkf = GroupKFold(n_splits=5)\nfor tr, va in gkf.split(X, y, groups=user_id): ...',
              example: 'Medical records grouped by patient.',
              trap: 'Wrong group id still leaks related rows.',
            }),
          },
          {
            id: 'nested-cv-cv',
            label: 'Nested CV',
            tooltip: tip({
              short: 'Outer CV estimates performance; inner CV tunes hyperparameters.',
              intuition: 'Separates selection from honest error estimation.',
              code: 'from sklearn.model_selection import GridSearchCV, cross_val_score\ngrid = GridSearchCV(pipe, param_grid, cv=3)\nnested = cross_val_score(grid, X, y, cv=5)',
              example: 'Heavier compute but less optimistic bias.',
              trap: 'Still need held-out test for final deployment claim.',
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'preprocess-leak-trap-cv',
            label: 'Preprocess before split',
            tooltip: tip({
              short: 'Fitting scaler/PCA on all rows leaks validation statistics.',
              intuition: 'Move every fit step inside the CV training loop.',
              example: 'StandardScaler().fit(X) on full X before CV is wrong.',
              trap: 'Same bug hits train/test split if scaler fit on val.',
            }),
            lessonId: 'feature-scaling-preprocessing',
          },
          {
            id: 'duplicate-entity-trap-cv',
            label: 'Duplicate entities',
            tooltip: tip({
              short: 'Same user in train and val folds inflates scores.',
              intuition: 'Model memorizes entity-specific patterns.',
              example: 'Recommendations: user 7 in both folds.',
              trap: 'Use GroupKFold or custom group splits.',
            }),
          },
          {
            id: 'target-leak-trap-cv',
            label: 'Target leakage features',
            tooltip: tip({
              short: 'Features that encode the label leak regardless of CV.',
              intuition: 'CV cannot fix columns that should not exist at prediction time.',
              example: '“cancel_reason” when predicting churn before cancel.',
              trap: 'Audit feature timestamps, not just fold logic.',
            }),
            lessonId: 'data-leakage-deep-dive',
          },
          {
            id: 'test-peeking-trap-cv',
            label: 'Test set peeking',
            tooltip: tip({
              short: 'Using test scores to pick models after CV tuning.',
              intuition: 'Test should remain final; CV handles development choices.',
              example: 'Try ten models on test → best test score is biased.',
              trap: 'Hold a truly fresh deployment shadow set when possible.',
            }),
          },
          {
            id: 'wrong-k-trap-cv',
            label: 'Wrong k',
            tooltip: tip({
              short: 'Too few folds → noisy estimates; LOOCV → high variance and cost.',
              intuition: 'Balance bias-variance of the CV estimator itself.',
              example: 'k=2 unstable; LOOCV expensive on large n.',
              trap: 'StratifiedKFold needed for rare classes.',
            }),
          },
          {
            id: 'time-shuffle-trap-cv',
            label: 'Shuffle time series',
            tooltip: tip({
              short: 'Random k-fold on temporal data leaks future labels.',
              intuition: 'Validation must mimic forecasting deployment.',
              example: 'Stock features from Tuesday in train, Monday in val—invalid.',
              trap: 'Use TimeSeriesSplit or custom rolling windows.',
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'hyperparam-used-cv',
            label: 'Hyperparameter tuning',
            tooltip: tip({
              short: 'Grid/random/Bayesian search uses CV scores as objective.',
              intuition: 'Each candidate λ evaluated by mean CV metric.',
              trap: 'Wide search + same test peeking overfits selection.',
            }),
            lessonId: 'hyperparameter-tuning',
          },
          {
            id: 'bias-variance-used-cv',
            label: 'Bias-variance diagnosis',
            tooltip: tip({
              short: 'CV curves across complexity show over/underfit patterns.',
              intuition: 'Train vs CV gap signals variance; both high signals bias.',
              trap: 'Noisy CV can mimic bias-variance signatures.',
            }),
            lessonId: 'bias-variance-tradeoff',
          },
          {
            id: 'tree-ensembles-used-cv',
            label: 'Tree ensembles',
            tooltip: tip({
              short: 'Random forests and boosting benefit from honest CV when data is small.',
              intuition: 'Depth and n_estimators tuned via CV.',
              trap: 'Leakage in categorical encoding still breaks CV.',
            }),
            lessonId: 'tree-ensembles',
          },
          {
            id: 'monitoring-used-cv',
            label: 'Offline vs online',
            tooltip: tip({
              short: 'Good CV does not guarantee production performance under drift.',
              intuition: 'Monitoring catches serve-time skew CV cannot see.',
              trap: 'Re-run CV when population or features shift.',
            }),
            lessonId: 'model-monitoring',
          },
          {
            id: 'causal-used-cv',
            label: 'Causal evaluation',
            tooltip: tip({
              short: 'Randomized experiments avoid some CV pitfalls for treatment effects.',
              intuition: 'Observational CV still needs confounding controls.',
              trap: 'CV on biased observational data estimates association, not effect.',
            }),
            lessonId: 'treatment-effects',
          },
          {
            id: 'data-eng-used-cv',
            label: 'Data engineering',
            tooltip: tip({
              short: 'Point-in-time feature pipelines must be replayed inside each CV fold.',
              intuition: 'Feature store backtests mimic CV with temporal cuts.',
              trap: 'Global aggregate features leak future label information.',
            }),
            lessonId: 'data-engineering-for-ml-track',
          },
        ],
      },
    ],
  },
  'data-engineering-for-ml-track': {
    center: {
      id: 'data-engineering-for-ml-track',
      label: 'Data Engineering for ML',
      type: 'current',
      tooltip: tip({
        short: 'ML data engineering ensures features and labels are correct at prediction time—point-in-time joins, label windows, schema contracts, and train/serve parity prevent silent evaluation lies.',
        intuition: 'Most production ML failures are boundary bugs: timestamps, aggregates computed too late, or serving pipelines that diverge from training.',
        formula: 'feature\\_time\\le prediction\\_time<label\\_window\\_end',
        why: 'Valid models need valid data plumbing; monitoring and leakage lessons collapse without temporal and contract discipline.',
        trap: 'Feature pipelines are not neutral plumbing—target encodings and global aggregates can leak future labels unless computed inside the right window.',
        example: 'Predict churn on Monday using features known by Monday 00:00; label uses cancel events through Sunday only.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'leakage-prereq-de',
            label: 'Data leakage deep dive',
            tooltip: tip({
              short: 'Leakage is information crossing evaluation boundaries.',
              intuition: 'Data engineering operationalizes leakage prevention in pipelines.',
              example: 'Post-outcome columns, duplicate users, future timestamps.',
              trap: 'Leakage can live in SQL joins, not model code.',
            }),
            lessonId: 'data-leakage-deep-dive',
          },
          {
            id: 'scaling-prereq-de',
            label: 'Feature scaling',
            tooltip: tip({
              short: 'Scalers fit on training data and replay at serve time.',
              intuition: 'Train/serve skew often starts in preprocessing mismatch.',
              example: 'Training mean/std must match online feature computation.',
              trap: 'Serving null imputation different from training breaks parity.',
            }),
            lessonId: 'feature-scaling-preprocessing',
          },
          {
            id: 'monitoring-prereq-de',
            label: 'Model monitoring',
            tooltip: tip({
              short: 'Production drift detection catches schema and distribution breaks.',
              intuition: 'Data contracts feed monitoring alerts when pipelines regress.',
              example: 'Null-rate spike on a feature triggers page.',
              trap: 'Offline CV cannot detect serve-time-only bugs.',
            }),
            lessonId: 'model-monitoring',
          },
          {
            id: 'cv-prereq-de',
            label: 'Cross-validation',
            tooltip: tip({
              short: 'Backtests replay point-in-time logic per fold.',
              intuition: 'Each CV fold needs temporal cuts consistent with deployment.',
              example: 'Feature store offline replay for historical dates.',
              trap: 'Random CV on temporal data invalidates backtest.',
            }),
            lessonId: 'cross-validation',
          },
          {
            id: 'timestamp-prereq-de',
            label: 'Event timestamps',
            tooltip: tip({
              short: 'Every feature and label row carries when it was knowable or realized.',
              intuition: 'Time ordering gates legal joins.',
              example: 'click_time, feature_snapshot_time, label_end_time.',
              trap: 'Batch ETL completion time ≠ business event time.',
            }),
          },
          {
            id: 'entity-grain-de',
            label: 'Entity grain',
            tooltip: tip({
              short: 'Define prediction unit: user-day, session, listing, etc.',
              intuition: 'Grain fixes join keys and label windows.',
              example: 'User-level churn vs session-level bounce differ.',
              trap: 'Mixing grains across tables creates duplicate or orphan rows.',
            }),
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'point-in-time-de',
            label: 'Point-in-time correctness',
            tooltip: tip({
              short: 'Features may use only information available at prediction time.',
              intuition: 'As-of join: feature_time ≤ prediction_time.',
              formula: 'feature\\_time\\le prediction\\_time',
              example: 'Monday prediction uses aggregates up to Sunday 23:59.',
              trap: 'Including Tuesday activity in Monday features is leakage.',
            }),
            highlightTarget: { panel: 'animation', type: 'point-in-time' },
          },
          {
            id: 'label-window-de',
            label: 'Label window',
            tooltip: tip({
              short: 'Labels collect outcomes during a defined future interval after prediction.',
              intuition: 'prediction_time < label_window_end defines when outcome is observed.',
              formula: 'prediction\\_time<label\\_window\\_end',
              example: '7-day churn label counts cancels in next week.',
              trap: 'Training on immature labels mislabels still-active users.',
            }),
            highlightTarget: { panel: 'animation', type: 'label-window' },
          },
          {
            id: 'feature-store-de',
            label: 'Feature store',
            tooltip: tip({
              short: 'Central registry serves consistent offline and online features.',
              intuition: 'Same transformation code paths reduce train/serve skew.',
              example: 'Feast/Tecton materializes historical and live feature vectors.',
              trap: 'Online store lag vs offline backfill mismatch breaks parity.',
            }),
            highlightTarget: { panel: 'animation', type: 'feature-store' },
          },
          {
            id: 'data-contracts-de',
            label: 'Data contracts',
            tooltip: tip({
              short: 'Schema, freshness, null-rate, and range SLAs enforced on pipelines.',
              intuition: 'Contracts catch upstream breaks before bad rows train models.',
              example: 'Alert if age feature null_rate > 1% or max > 120.',
              trap: 'Silent schema drift (new column missing online) crashes or skews serve.',
            }),
            highlightTarget: { panel: 'animation', type: 'contracts' },
          },
          {
            id: 'train-serve-parity-de',
            label: 'Train/serve parity',
            tooltip: tip({
              short: 'Training SQL/logic must match production feature computation.',
              intuition: 'Skew is when offline and online features differ subtly.',
              example: 'Different imputation default online vs offline.',
              trap: 'Shadow mode comparison catches skew before full rollout.',
            }),
            highlightTarget: { panel: 'animation', type: 'parity' },
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'boundary-bugs-de',
            label: 'Boundary bugs dominate',
            tooltip: tip({
              short: 'Models fail more often from bad joins than from wrong algorithms.',
              intuition: 'Fix timestamps before tuning learning rate.',
              example: 'Amazing offline AUC, terrible live—check feature lag.',
              trap: 'Blaming the model hides pipeline root cause.',
            }),
          },
          {
            id: 'as-of-join-de',
            label: 'As-of join mental model',
            tooltip: tip({
              short: 'For each prediction row, grab latest feature snapshot not after that moment.',
              intuition: 'Time travel prevention in SQL.',
              example: 'merge_asof in pandas with direction=backward.',
              trap: 'Forward asof joins leak future feature values.',
            }),
          },
          {
            id: 'label-maturity-de',
            label: 'Label maturity',
            tooltip: tip({
              short: 'Recent rows lack finalized labels until window elapses.',
              intuition: 'Training sets often exclude last L days of immature labels.',
              example: 'Drop last 7 days when label is 7-day conversion.',
              trap: 'Scoring recent rows for training pollutes labels with NA/proxy error.',
            }),
          },
          {
            id: 'aggregate-leakage-de',
            label: 'Aggregate leakage',
            tooltip: tip({
              short: 'Global target encodings computed on full dataset leak validation folds.',
              intuition: 'Aggregates must be fit inside training boundaries only.',
              example: 'Category mean target encoded using val labels inflates score.',
              trap: 'Same leak hits CV if encoding uses all folds at once.',
            }),
          },
          {
            id: 'freshness-de',
            label: 'Freshness vs correctness',
            tooltip: tip({
              short: 'Stale features may be valid historically but wrong live.',
              intuition: 'Freshness SLAs complement correctness rules.',
              example: 'Real-time counter stale 6h → wrong ranking live.',
              trap: 'Fresh but wrong (buggy upstream) still fails silently without contracts.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'temporal-validity-de',
            label: 'Temporal validity',
            tooltip: tip({
              short: 'feature_time ≤ prediction_time < label_window_end.',
              intuition: 'Three timestamps anchor every supervised row.',
              formula: 'feature\\_time\\le prediction\\_time<label\\_window\\_end',
              example: 'Document each column’s event_time in feature spec.',
              trap: 'Implicit “now()” in SQL breaks backtests.',
            }),
            highlightTarget: { panel: 'code', type: 'formula' },
          },
          {
            id: 'merge-asof-de',
            label: 'merge_asof',
            tooltip: tip({
              short: 'Pandas as-of join picks last feature row at or before prediction time.',
              intuition: 'Sorted keys required; backward direction prevents future peek.',
              code: 'pd.merge_asof(preds.sort_values("t"), feats.sort_values("t"), on="t", direction="backward")',
              example: 'One row per prediction with latest valid features.',
              trap: 'Unsorted inputs give wrong matches silently in some engines.',
            }),
          },
          {
            id: 'contract-check-de',
            label: 'Contract check',
            tooltip: tip({
              short: 'Assert schema, ranges, and null rates on each batch.',
              intuition: 'Fail pipeline early instead of poisoning training.',
              code: 'assert df["age"].isna().mean() < 0.01\nassert df["age"].max() <= 120',
              example: 'Great Expectations / dbt tests / custom validators.',
              trap: 'Warnings-only contracts get ignored in production fires.',
            }),
          },
          {
            id: 'offline-online-de',
            label: 'Offline/online replay',
            tooltip: tip({
              short: 'Replay historical requests through serving path to diff features.',
              intuition: 'Shadow diffs quantify train/serve skew numerically.',
              code: 'skew = (offline_feat - online_feat).abs().mean()',
              example: 'Log top-10 features by mean absolute diff.',
              trap: 'Sampling only happy paths misses edge-case skew.',
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'target-encoding-trap-de',
            label: 'Target encoding leak',
            tooltip: tip({
              short: 'Encoding categories with global target mean uses label information improperly.',
              intuition: 'Compute encodings inside train folds or with Bayesian smoothing on train only.',
              example: 'city_mean_churn fit on full data before CV.',
              trap: 'Popular Kaggle trick fails in honest backtests.',
            }),
          },
          {
            id: 'future-aggregate-trap-de',
            label: 'Future aggregates',
            tooltip: tip({
              short: 'Rolling 30-day sum including days after prediction_time.',
              intuition: 'Window endpoints must truncate at as-of time.',
              example: 'User spend last 30d as of Sunday cannot include Monday spend.',
              trap: 'SQL BETWEEN without upper bound at prediction time leaks.',
            }),
          },
          {
            id: 'train-serve-skew-trap-de',
            label: 'Train/serve skew',
            tooltip: tip({
              short: 'Different code paths offline vs online for same feature name.',
              intuition: 'Model learns offline quirks that do not exist live.',
              example: 'Training uses SQL COALESCE(0); API returns null.',
              trap: 'Single shared feature definition file reduces drift.',
            }),
          },
          {
            id: 'immature-label-trap-de',
            label: 'Immature labels',
            tooltip: tip({
              short: 'Including recent rows before label window closes mislabels negatives.',
              intuition: 'Censor immature examples from training.',
              example: 'User has not had 7 days to churn yet → not a true negative.',
              trap: 'Proxy labels from early behavior differ from final outcome.',
            }),
          },
          {
            id: 'duplicate-entity-trap-de',
            label: 'Duplicate entity rows',
            tooltip: tip({
              short: 'Multiple prediction rows per user without dedup strategy.',
              intuition: 'Splits and joins must respect entity grain.',
              example: 'Daily user rows duplicated after bad join explode.',
              trap: 'Metrics and labels double-count the same user.',
            }),
          },
          {
            id: 'schema-drift-trap-de',
            label: 'Schema drift',
            tooltip: tip({
              short: 'New/missing columns break serving or impute wrongly.',
              intuition: 'Contracts should block deploy on schema mismatch.',
              example: 'Upstream removes column; model receives default zero forever.',
              trap: 'Silent defaults hide missing features until monitoring fires.',
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'recommender-used-de',
            label: 'Recommender systems',
            tooltip: tip({
              short: 'Ranking pipelines need point-in-time impressions and delayed rewards.',
              intuition: 'Label windows capture click/purchase after impression.',
              trap: 'Position bias and delayed feedback complicate labels.',
            }),
            lessonId: 'recommender-systems-ranking-track',
          },
          {
            id: 'forecast-used-de',
            label: 'Time series forecasting',
            tooltip: tip({
              short: 'Forecast features must respect causal ordering strictly.',
              intuition: 'Same temporal validity at finer grain for sequences.',
              trap: 'Global normalization using future stats leaks.',
            }),
            lessonId: 'time-series-forecasting-track',
          },
          {
            id: 'ab-test-used-de',
            label: 'Experiment analysis',
            tooltip: tip({
              short: 'Experiment metrics join treatment assignment with outcome windows.',
              intuition: 'Data engineering defines valid experiment rows.',
              trap: 'SRM or attrition bias from bad assignment logs.',
            }),
            lessonId: 'ab-testing-foundations',
          },
          {
            id: 'rag-used-de',
            label: 'RAG pipelines',
            tooltip: tip({
              short: 'Document freshness and index versioning are data contracts for retrieval.',
              intuition: 'Stale index vs fresh content is a serve-time skew story.',
              trap: 'Chunk timestamps matter for time-sensitive queries.',
            }),
            lessonId: 'rag-chunking-context',
          },
          {
            id: 'fairness-used-de',
            label: 'Fairness audits',
            tooltip: tip({
              short: 'Sensitive attributes and slice metrics need reliable feature lineage.',
              intuition: 'Bad joins can erase or mislabel protected groups.',
              trap: 'Proxy features reintroduce sensitive signal improperly.',
            }),
            lessonId: 'model-fairness',
          },
          {
            id: 'debugging-used-de',
            label: 'Model debugging',
            tooltip: tip({
              short: 'Slice failures often trace to cohort-specific pipeline bugs.',
              intuition: 'Compare feature distributions when one segment collapses.',
              trap: 'Fixing model weights without fixing data repeats failure.',
            }),
            lessonId: 'model-debugging',
          },
        ],
      },
    ],
  },
  'determinant-volume': {
    center: {
      id: 'determinant-volume',
      label: 'Determinant as Volume',
      type: 'current',
      tooltip: tip({
        short: 'The determinant measures how a square matrix scales signed area or volume—and whether the transformation collapses space to lower dimension (det = 0).',
        intuition: 'Apply A to the unit square or cube; |det A| is the resulting area or volume; sign tracks orientation flip.',
        formula: '\\det(A)=\\text{signed volume scale}',
        why: 'Determinants gate invertibility, change-of-basis Jacobians, multivariate Gaussians, and eigenvalue products.',
        trap: 'Determinant is defined for square matrices; non-square maps need different volume notions.',
        example: '2×2 matrix doubling width and halving height → |det|=1 (area preserved), possibly with sign flip if orientation reverses.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'matrix-mult-det',
            label: 'Matrix multiplication',
            tooltip: tip({
              short: 'A maps vectors; columns show where basis vectors land.',
              intuition: 'Determinant tracks how that column parallelogram scales.',
              example: 'A times unit square corners → parallelogram.',
              trap: 'Non-square A does not have a single det value.',
            }),
            lessonId: 'matrix-multiplication',
          },
          {
            id: 'vector-det',
            label: 'Vectors in ℝⁿ',
            tooltip: tip({
              short: 'Column vectors span parallelepipeds when combined.',
              intuition: 'Unit cube corners map to a skewed box under A.',
              example: 'Two 2D basis vectors span the unit square.',
              trap: 'Volume lives in the output space dimension n for n×n A.',
            }),
          },
          {
            id: 'linear-map-det',
            label: 'Linear transformation',
            tooltip: tip({
              short: 'A represents a linear map T(x)=Ax.',
              intuition: 'Determinant is a property of T for square maps.',
              example: 'Rotation preserves volume; scaling stretches it.',
              trap: 'Affine maps add translation—det still from linear part.',
            }),
          },
          {
            id: 'area-2d-det',
            label: '2×2 area formula',
            tooltip: tip({
              short: 'det [[a,b],[c,d]] = ad − bc equals signed parallelogram area.',
              intuition: 'First column and second column span the tile.',
              example: 'Shear preserves area → |det|=1.',
              trap: 'Order of columns matters for sign.',
            }),
          },
          {
            id: 'invertibility-preview-det',
            label: 'Invertibility preview',
            tooltip: tip({
              short: 'det(A)≠0 ⟺ A is invertible (square case).',
              intuition: 'Zero determinant = collapsed dimension = no inverse.',
              example: 'Projection to a line has det 0 in 2D.',
              trap: 'Near-zero det is ill-conditioned numerically.',
            }),
            lessonId: 'condition-number',
          },
          {
            id: 'product-rule-preview-det',
            label: 'Product of eigenvalues preview',
            tooltip: tip({
              short: 'det(A) equals product of eigenvalues (with multiplicity).',
              intuition: 'Each eigen direction scales by its eigenvalue.',
              example: 'Stretch 2× and 3× → det=6.',
              trap: 'Complex eigenvalues still multiply to real det for real A.',
            }),
            lessonId: 'eigenvalue',
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'unit-cube-det',
            label: 'Unit cube image',
            tooltip: tip({
              short: 'Map unit cube by A; |det A| is its volume.',
              intuition: 'Columns of A are where basis vectors go.',
              example: 'Animate square → parallelogram under 2×2 A.',
              trap: 'Degenerate columns → zero volume.',
            }),
            highlightTarget: { panel: 'animation', type: 'unit-cube' },
          },
          {
            id: 'signed-volume-det',
            label: 'Signed volume',
            tooltip: tip({
              short: 'Negative det means orientation reversal (mirror flip).',
              intuition: 'Right-hand rule orientation swaps under reflection.',
              example: 'Reflection det = −1 on line segment length in 1D analog.',
              trap: '|det| for magnitude; sign for orientation.',
            }),
            highlightTarget: { panel: 'animation', type: 'signed' },
          },
          {
            id: 'column-scaling-det',
            label: 'Column scaling',
            tooltip: tip({
              short: 'Scaling one column scales det by that factor.',
              intuition: 'Multiplying a side of the parallelepiped scales volume.',
              example: 'Double column 1 → det doubles.',
              trap: 'Row scaling same effect for square matrices.',
            }),
          },
          {
            id: 'zero-collapse-det',
            label: 'det = 0 collapse',
            tooltip: tip({
              short: 'Dependent columns flatten the parallelepiped to zero volume.',
              intuition: 'Map squashes space to lower dimension.',
              example: 'Two parallel columns in 2D → line image.',
              trap: 'Numeric det near zero signals unstable inverse.',
            }),
            highlightTarget: { panel: 'animation', type: 'collapse' },
          },
          {
            id: 'composition-det',
            label: 'Composition rule',
            tooltip: tip({
              short: 'det(AB) = det(A) det(B).',
              intuition: 'Volume scaling multiplies along composed maps.',
              formula: '\\det(AB)=\\det(A)\\det(B)',
              example: 'Rotate then scale: volumes multiply.',
              trap: 'Order of composition matches matrix multiply order.',
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'volume-not-length-det',
            label: 'Volume not length',
            tooltip: tip({
              short: 'Individual column lengths do not alone determine det.',
              intuition: 'Angle between columns matters—shear can preserve area.',
              example: 'Unit-length columns at 90° vs acute angle change area.',
              trap: '||col1||·||col2|| only equals area for orthogonal columns in 2D.',
            }),
          },
          {
            id: 'orientation-intuition-det',
            label: 'Orientation tracking',
            tooltip: tip({
              short: 'Sign of det detects reflection vs rotation (proper vs improper).',
              intuition: 'Proper rotations have det +1 in SO(n).',
              example: '2D rotation matrix det = 1.',
              trap: 'Improper orthogonal maps have det −1.',
            }),
          },
          {
            id: 'rank-connection-det',
            label: 'Rank connection',
            tooltip: tip({
              short: 'det≠0 ⟺ full rank for square A.',
              intuition: 'Full rank means n independent directions preserved.',
              example: 'Singular A loses at least one dimension.',
              trap: 'Rectangular rank uses different tests.',
            }),
            lessonId: 'fundamental-subspaces',
          },
          {
            id: 'jacobian-preview-det',
            label: 'Jacobian preview',
            tooltip: tip({
              short: 'In change of variables, det of Jacobian scales probability density.',
              intuition: 'Infinitesimal volume elements scale by |det J|.',
              example: 'Polar coordinates Jacobian includes r factor.',
              trap: 'Nonlinear maps use local Jacobian, not global det of one matrix.',
            }),
          },
          {
            id: 'measure-distortion-det',
            label: 'Measure distortion',
            tooltip: tip({
              short: 'det summarizes global volume distortion of linear A.',
              intuition: 'Small |det| compresses; large |det| expands.',
              example: 'Anisotropic scaling stretches volume product of axis scales.',
              trap: 'Condition number also matters for roundoff, not just det magnitude.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'two-by-two-det',
            label: '2×2 determinant',
            tooltip: tip({
              short: 'ad − bc for [[a,b],[c,d]].',
              intuition: 'Cross-multiply diagonal products with sign.',
              formula: '\\det\\begin{pmatrix}a&b\\\\c&d\\end{pmatrix}=ad-bc',
              example: '[[2,0],[0,3]] → det=6.',
              trap: 'Sign flips if columns swapped.',
            }),
            highlightTarget: { panel: 'code', type: 'formula-2x2' },
          },
          {
            id: 'numpy-det',
            label: 'numpy.linalg.det',
            tooltip: tip({
              short: 'Numeric determinant for square arrays.',
              intuition: 'Uses LU—watch conditioning for ill-posed matrices.',
              code: 'import numpy as np\nnp.linalg.det(A)',
              example: 'Compare |det| after animation scaling.',
              trap: 'Floating error on singular-ish matrices.',
            }),
            highlightTarget: { panel: 'code', type: 'numpy-det' },
          },
          {
            id: 'product-eigen-det',
            label: 'Product of eigenvalues',
            tooltip: tip({
              short: 'det(A) = ∏ λᵢ for square A.',
              intuition: 'Eigenvalues describe axis scaling in invariant directions.',
              formula: '\\det(A)=\\prod_i \\lambda_i',
              example: 'Diagonal matrix det is product of diagonal entries.',
              trap: 'Defective matrices still satisfy product formula over algebraic multiplicities.',
            }),
          },
          {
            id: 'log-abs-det',
            label: 'log|det| for stability',
            tooltip: tip({
              short: 'Summing log-singular values avoids overflow in products.',
              intuition: 'Common in Gaussian log-likelihoods and normalizing flows.',
              code: 'log_abs_det = np.linalg.slogdet(A)[1]',
              example: 'Used when det underflows/overflows in high dimension.',
              trap: 'slogdet returns sign and logabsdet separately.',
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'non-square-trap-det',
            label: 'Non-square matrix',
            tooltip: tip({
              short: 'Determinant is not defined for m×n when m≠n.',
              intuition: 'Volume scaling story needs square maps in n dimensions.',
              example: 'Tall skinny A maps to lower-dimensional image.',
              trap: 'Use rank or singular values for rectangular maps.',
            }),
          },
          {
            id: 'magnitude-only-trap-det',
            label: 'Ignore sign',
            tooltip: tip({
              short: '|det| alone misses orientation reversal.',
              intuition: 'Reflections have det −1 but preserve |volume| in magnitude sense.',
              example: 'Mirror flip in 2D: area same, sign negative.',
              trap: 'Physics and coordinate handedness care about sign.',
            }),
          },
          {
            id: 'column-length-trap-det',
            label: 'Column lengths only',
            tooltip: tip({
              short: 'Unit columns can still yield det≠1 if not orthogonal.',
              intuition: 'Shear with unit edges changes area.',
              example: 'Parallelogram with unit sides but 30° angle.',
              trap: 'Orthonormal columns give det magnitude 1 for square Q.',
            }),
          },
          {
            id: 'near-zero-trap-det',
            label: 'Near-zero numeric det',
            tooltip: tip({
              short: 'Tiny det suggests ill-conditioning, not necessarily exact singularity.',
              intuition: 'Floating point may report det≈1e-16 for bad matrices.',
              example: 'Hilbert matrices have tiny det but are hard inverses.',
              trap: 'Use condition number and SVD for stability decisions.',
            }),
            lessonId: 'condition-number',
          },
          {
            id: 'det-not-norm-trap-det',
            label: 'det ≠ matrix size',
            tooltip: tip({
              short: 'Large entries can yield huge det even for mild-looking maps.',
              intuition: 'det scales product of eigenvalues, sensitive in high n.',
              example: '10×10 identity scaled by 2 has det 2¹⁰.',
              trap: 'Log-determinants preferred in ML likelihoods.',
            }),
          },
          {
            id: 'global-nonlinear-trap-det',
            label: 'Nonlinear global det',
            tooltip: tip({
              short: 'One constant det does not describe nonlinear maps globally.',
              intuition: 'Use Jacobian determinant pointwise in calculus.',
              example: 'Polar map Jacobian depends on r.',
              trap: 'Linear det lesson is global; calculus det is local.',
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'eigenvalue-used-det',
            label: 'Eigenvalues',
            tooltip: tip({
              short: 'Characteristic polynomial det(A−λI)=0 defines eigenvalues.',
              intuition: 'det links spectrum to volume scaling product.',
              trap: 'Numeric eigen solvers avoid explicit characteristic poly in large n.',
            }),
            lessonId: 'eigenvalue',
          },
          {
            id: 'change-basis-used-det',
            label: 'Change of basis',
            tooltip: tip({
              short: 'Similarity transforms preserve determinant.',
              intuition: 'Volume scaling is intrinsic to the linear map.',
              example: 'det(P⁻¹AP)=det(A) for invertible P.',
              trap: 'Coordinates change; det of the map does not.',
            }),
            lessonId: 'change-of-basis',
          },
          {
            id: 'svd-used-det',
            label: 'SVD / volume',
            tooltip: tip({
              short: '|det| equals product of singular values for square A.',
              intuition: 'Singular values generalize axis scaling.',
              trap: 'Rectangular SVD uses different invariant quantities.',
            }),
            lessonId: 'svd',
          },
          {
            id: 'gaussian-used-det',
            label: 'Multivariate Gaussian',
            tooltip: tip({
              short: 'Density includes (2π)^{−n/2}|Σ|^{−1/2} exp(…).',
              intuition: 'Covariance determinant measures volume uncertainty.',
              example: 'Singular Σ means degenerate distribution on subspace.',
              trap: 'Use pseudo-det or reduce dimension when Σ singular.',
            }),
          },
          {
            id: 'flows-used-det',
            label: 'Normalizing flows',
            tooltip: tip({
              short: 'Change-of-variables uses log|det Jacobian| in log-likelihood.',
              intuition: 'Invertible neural layers track volume change.',
              trap: 'Enforcing invertibility is architectural constraint.',
            }),
          },
          {
            id: 'invertibility-used-det',
            label: 'Matrix inverse',
            tooltip: tip({
              short: 'Invertible ⟺ det≠0; adjugate formula uses det in theory.',
              intuition: 'Practical inverse uses LU/QR, not explicit 1/det.',
              example: 'Solve Ax=b when det nonzero.',
              trap: 'Never divide by near-zero det in code.',
            }),
            lessonId: 'matrix-decompositions',
          },
        ],
      },
    ],
  },
  'diffusion-basics': {
    center: {
      id: 'diffusion-basics',
      label: 'Diffusion Basics',
      type: 'current',
      tooltip: tip({
        short: 'Diffusion models learn to reverse a forward noising process: gradually corrupt data with noise, then train a network to predict and remove that noise step by step.',
        intuition: 'Generation starts from pure noise and walks backward through many denoising steps, each guided by the model’s noise prediction.',
        formula: 'x_t=\\sqrt{1-t}\\,x_0+\\sqrt{t}\\,\\epsilon,\\quad \\hat{x}_0\\leftarrow x_t-\\hat{\\epsilon}',
        why: 'Diffusion is the generative backbone for image models, audio, video, and discrete text diffusion variants.',
        trap: 'The model is not asked to invent the whole sample in one step—it learns many small denoising corrections.',
        example: 'At t=0.5, a clean cat image mixes with noise; the network predicts ε; subtracting ε estimate recovers a sharper x̂₀.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'noise-variance-db',
            label: 'Noise and variance',
            tooltip: tip({
              short: 'Gaussian noise ε with mean 0 adds stochastic corruption.',
              intuition: 'Forward diffusion increases variance until signal drowns.',
              example: 'ε ~ N(0, I) standard normal noise vector.',
              trap: 'Noise schedule must match training assumptions at sample time.',
            }),
            lessonId: 'expected-value-variance',
          },
          {
            id: 'neural-net-db',
            label: 'Neural network',
            tooltip: tip({
              short: 'A U-Net or similar predicts noise or score from x_t and t.',
              intuition: 'Same supervised learning loop with (x_t, t) inputs.',
              example: 'Denoiser θ(x_t, t) → ε̂.',
              trap: 'Capacity and conditioning on t are essential.',
            }),
            lessonId: 'neural-network',
          },
          {
            id: 'grad-descent-db',
            label: 'Gradient descent',
            tooltip: tip({
              short: 'Train by minimizing noise prediction error over timesteps.',
              intuition: 'Random t each batch teaches all noise levels.',
              example: 'MSE(ε, ε̂) averaged over samples and t.',
              trap: 'Poor t sampling coverage weakens some noise levels.',
            }),
            lessonId: 'gradient-descent',
          },
          {
            id: 'latent-vector-db',
            label: 'Data as vectors',
            tooltip: tip({
              short: 'Images flatten to tensors; noise adds elementwise.',
              intuition: 'High-dimensional x_0 lives in same space as ε.',
              example: 'RGB image tensor shape [C,H,W].',
              trap: 'Pixel scaling to [-1,1] affects noise signal ratio.',
            }),
          },
          {
            id: 'markov-forward-db',
            label: 'Markov forward chain',
            tooltip: tip({
              short: 'Each step adds noise conditioned on previous x_{t−1}.',
              intuition: 'Forward process is fixed—not learned.',
              example: 'x_t depends on x_{t−1} and fresh noise.',
              trap: 'Closed-form x_t from x_0 exists for linear schedules.',
            }),
            lessonId: 'markov-chains',
          },
          {
            id: 'vae-preview-db',
            label: 'Latent space preview',
            tooltip: tip({
              short: 'Some systems diffuse in VAE latent space, not pixels.',
              intuition: 'Lower dimension can mean faster training/sampling.',
              example: 'Stable Diffusion diffuses in latent z.',
              trap: 'Decoder quality still limits final image.',
            }),
            lessonId: 'diffusion-vae',
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'forward-noise-db',
            label: 'Forward noising',
            tooltip: tip({
              short: 'Mix clean x_0 with noise ε at level t.',
              intuition: 'Signal coefficient shrinks as noise coefficient grows.',
              formula: 'x_t=\\sqrt{1-t}\\,x_0+\\sqrt{t}\\,\\epsilon',
              example: 'Small t → mostly clean; large t → mostly noise.',
              trap: 'Schedule notation varies (ᾱ_t products vs sqrt form).',
            }),
            highlightTarget: { panel: 'animation', type: 'forward' },
          },
          {
            id: 'predict-noise-db',
            label: 'Predict noise ε̂',
            tooltip: tip({
              short: 'Network learns ε̂_θ(x_t, t) ≈ ε used in forward step.',
              intuition: 'Knowing noise reveals implied clean signal.',
              example: 'Training target is the ε that created x_t from x_0.',
              trap: 'Some parameterizations predict x_0 or score instead—convert consistently.',
            }),
            highlightTarget: { panel: 'animation', type: 'predict-noise' },
          },
          {
            id: 'estimate-x0-db',
            label: 'Estimate x̂_0',
            tooltip: tip({
              short: 'Rearrange forward formula to recover clean estimate from x_t and ε̂.',
              intuition: 'Subtract predicted noise component from x_t.',
              formula: '\\hat{x}_0\\leftarrow x_t-\\hat{\\epsilon}',
              example: 'Better ε̂ → sharper x̂_0 preview at same t.',
              trap: 'Overestimated noise removes too much signal.',
            }),
            highlightTarget: { panel: 'animation', type: 'estimate-x0' },
          },
          {
            id: 'reverse-step-db',
            label: 'Reverse denoise step',
            tooltip: tip({
              short: 'Walk from high t toward 0 using model-guided updates.',
              intuition: 'Many small steps compound into full generation.',
              example: 'T=1000 steps from noise to sample.',
              trap: 'Fewer steps need specialized samplers or distillation.',
            }),
            highlightTarget: { panel: 'animation', type: 'reverse' },
          },
          {
            id: 'timestep-conditioning-db',
            label: 'Timestep conditioning',
            tooltip: tip({
              short: 'Network input includes t so behavior adapts to noise level.',
              intuition: 'Early steps need coarse structure; late steps refine detail.',
              example: 'Sinusoidal t embedding injected into U-Net blocks.',
              trap: 'Missing t conditioning confuses noise levels.',
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'scratch-art-db',
            label: 'Sculpt from noise',
            tooltip: tip({
              short: 'Start with random static; gradually reveal structure.',
              intuition: 'Like carving signal out of noise layer by layer.',
              example: 'Early t: blob; late t: edges and texture.',
              trap: 'One-step denoise from pure noise cannot work well untrained.',
            }),
          },
          {
            id: 'many-steps-db',
            label: 'Many small steps',
            tooltip: tip({
              short: 'Quality comes from iterative refinement, not one shot.',
              intuition: 'Each step fixes local noise mistakes.',
              example: 'DDPM uses hundreds to thousands of steps.',
              trap: 'Sampling cost motivates faster samplers later.',
            }),
          },
          {
            id: 'noise-level-curriculum-db',
            label: 'All noise levels',
            tooltip: tip({
              short: 'Training samples random t each batch—curriculum over corruption.',
              intuition: 'Model must denoise at every severity.',
              example: 't uniform on {1,…,T} during training.',
              trap: 'Rare extreme t values need enough batch coverage.',
            }),
          },
          {
            id: 'error-amplification-db',
            label: 'Error amplification',
            tooltip: tip({
              short: 'Wrong ε̂ at early steps corrupts structure permanently.',
              intuition: 'Later steps cannot fully fix gross early mistakes.',
              example: 'Overestimated noise → washed-out x̂_0.',
              trap: 'Sampler choice affects error accumulation.',
            }),
          },
          {
            id: 'score-connection-db',
            label: 'Score connection',
            tooltip: tip({
              short: 'Noise prediction relates to score ∇_x log p(x).',
              intuition: 'Pointing toward higher-density regions of data.',
              example: 'Score-based models share diffusion sampling math.',
              trap: 'Different parameterizations need consistent conversion.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'forward-formula-db',
            label: 'Forward mix',
            tooltip: tip({
              short: 'x_t = √(1−t) x_0 + √t ε.',
              intuition: 'Variance-preserving style mixing (lesson notation).',
              formula: 'x_t=\\sqrt{1-t}\\,x_0+\\sqrt{t}\\,\\epsilon',
              example: 'Sample t~U(0,1), ε~N(0,I), build x_t on the fly.',
              trap: 'DDPM papers use ᾱ_t—map symbols carefully.',
            }),
            highlightTarget: { panel: 'code', type: 'forward-formula' },
          },
          {
            id: 'training-loss-db',
            label: 'Training loss',
            tooltip: tip({
              short: 'L = E[||ε − ε̂_θ(x_t,t)||²].',
              intuition: 'Simple MSE on noise is standard objective.',
              code: 'loss = F.mse_loss(model(x_t, t), eps)',
              example: 'Random t and ε each step.',
              trap: 'Weighted loss across t changes effective curriculum.',
            }),
            highlightTarget: { panel: 'code', type: 'loss' },
          },
          {
            id: 'x0-recovery-db',
            label: 'x̂_0 recovery',
            tooltip: tip({
              short: 'Rearrange forward mix to solve for x_0 given x_t, ε̂.',
              intuition: 'Lesson shorthand: x̂_0 ← x_t − ε̂ (scaled forms vary).',
              formula: '\\hat{x}_0\\leftarrow x_t-\\hat{\\epsilon}',
              example: 'Visualize x̂_0 preview improving as ε̂ improves.',
              trap: 'Clip x̂_0 to valid range in image space when needed.',
            }),
          },
          {
            id: 'sample-loop-db',
            label: 'Sampling loop',
            tooltip: tip({
              short: 'for t from T down to 1: update x using model prediction.',
              intuition: 'Exact update rule depends on sampler (DDPM, DDIM, …).',
              code: 'x = torch.randn_like(shape)\nfor t in reversed(range(T)):\n    eps_hat = model(x, t)\n    x = sampler_step(x, eps_hat, t)',
              example: 'Start x ~ N(0,I).',
              trap: 'Sampler must match training parameterization.',
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'one-step-trap-db',
            label: 'One-step generation',
            tooltip: tip({
              short: 'Expecting perfect samples from single denoise at t≈1.',
              intuition: 'Iterative process is the design, not a bug.',
              example: 'Untrained model at one step → mush.',
              trap: 'Distilled models approximate many steps in few—still trained for that.',
            }),
          },
          {
            id: 'overestimate-noise-trap-db',
            label: 'Overestimate noise',
            tooltip: tip({
              short: 'ε̂ too large removes real signal when recovering x̂_0.',
              intuition: 'Animation shows washed or dark previews.',
              example: 'Manipulate prediction error slider in lesson.',
              trap: 'Systematic bias at certain t breaks sampler stability.',
            }),
          },
          {
            id: 'schedule-mismatch-trap-db',
            label: 'Schedule mismatch',
            tooltip: tip({
              short: 'Training noise schedule ≠ sampling schedule breaks quality.',
              intuition: 'Forward definitions must pair with reverse updates.',
              example: 'Linear vs cosine β_t schedules.',
              trap: 'Porting checkpoints requires matching schedulers.',
            }),
          },
          {
            id: 'param-confusion-trap-db',
            label: 'Parameterization mix-up',
            tooltip: tip({
              short: 'Models may predict ε, x_0, or v—do not mix formulas.',
              intuition: 'Convert predictions before plugging into sampler.',
              example: 'v-prediction used in some Stable Diffusion variants.',
              trap: 'Wrong conversion yields blurry or noisy samples.',
            }),
          },
          {
            id: 'scale-trap-db',
            label: 'Data scaling',
            tooltip: tip({
              short: 'Pixels in [0,1] vs [−1,1] changes noise signal balance.',
              intuition: 'Training and sampling must share normalization.',
              example: 'Forgotten rescaling → poor denoising.',
              trap: 'VAE latents have their own scale conventions.',
            }),
          },
          {
            id: 'confuse-vae-trap-db',
            label: 'Confuse with VAE-only',
            tooltip: tip({
              short: 'Diffusion ≠ VAE; VAE may only provide latent space.',
              intuition: 'Diffusion handles generation trajectory; VAE compresses.',
              example: 'Stable Diffusion = VAE + latent diffusion + text encoder.',
              trap: 'VAE blur limits fine detail even with perfect diffusion.',
            }),
            lessonId: 'vae',
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'sampling-used-db',
            label: 'Diffusion sampling',
            tooltip: tip({
              short: 'DDPM, DDIM, and faster samplers turn denoiser into generator.',
              intuition: 'Sampler chooses stochasticity vs speed tradeoff.',
              trap: 'Few-step samplers sensitive to model errors.',
            }),
            lessonId: 'diffusion-sampling',
          },
          {
            id: 'cfg-used-db',
            label: 'Classifier-free guidance',
            tooltip: tip({
              short: 'Steers denoising with conditional vs unconditional noise difference.',
              intuition: 'Built on same ε̂ predictions at sample time.',
              trap: 'Extreme guidance causes artifacts.',
            }),
            lessonId: 'classifier-free-guidance',
          },
          {
            id: 'unet-dit-used-db',
            label: 'U-Net vs DiT',
            tooltip: tip({
              short: 'Backbone architecture choices for the denoiser network.',
              intuition: 'Conv U-Net vs transformer DiT at scale.',
              trap: 'Architecture ≠ sampler schedule.',
            }),
            lessonId: 'unet-vs-dit',
          },
          {
            id: 'sdlm-used-db',
            label: 'Diffusion language models',
            tooltip: tip({
              short: 'Discrete token masking replaces Gaussian noise in text domain.',
              intuition: 'Same iterative refinement idea on token sequences.',
              trap: 'Discrete state space needs different forward process.',
            }),
            lessonId: 'diffusion-language-models',
          },
          {
            id: 'sd3-used-db',
            label: 'SD3 / frontier stacks',
            tooltip: tip({
              short: 'Production image systems combine diffusion with encoders and schedulers.',
              intuition: 'Basics lesson isolates core denoise loop before SD3 complexity.',
              trap: 'Multi-component pipelines hide each piece’s role.',
            }),
            lessonId: 'sd3-overview',
          },
          {
            id: 'flow-used-db',
            label: 'Flow matching',
            tooltip: tip({
              short: 'Continuous-time generative paths related to diffusion/score ideas.',
              intuition: 'Alternative training objective with shared sampling intuition.',
              trap: 'Not identical math—compare objectives before equating.',
            }),
            lessonId: 'flow-matching',
          },
        ],
      },
    ],
  },
  'diffusion-language-models': {
    center: {
      id: 'diffusion-language-models',
      label: 'Diffusion Language Models',
      type: 'current',
      tooltip: tip({
        short: 'Diffusion language models generate text by iteratively denoising a masked or corrupted token sequence—refining many positions in parallel instead of committing left-to-right one token at a time.',
        intuition: 'Start from a draft full of masks or noise tokens; the model fills confident positions, revises uncertain ones, and repeats until a coherent sequence emerges.',
        formula: 'q(x_t|x_0)=\\operatorname{mask}(x_0,t) \\quad p_\\theta(x_i|x_t,t)',
        why: 'Diffusion LMs enable parallel decoding, infilling, editing, and alternative alignment workflows compared with autoregressive GPT-style generation.',
        trap: 'Diffusion LMs are not automatically faster or better than AR models—schedules, fluency, alignment, and serving stacks still dominate outcomes.',
        example: 'Fully masked answer slot → model predicts high-confidence tokens first → remasks low-confidence positions → repeats for T steps.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'diffusion-basics-dlm',
            label: 'Diffusion basics',
            tooltip: tip({
              short: 'Forward corruption + learned reverse denoising loop.',
              intuition: 'Text diffusion swaps Gaussian noise for mask/corrupt operators.',
              example: 'Image ε-prediction analog → token clean prediction.',
              trap: 'Continuous image formulas do not copy verbatim to discrete tokens.',
            }),
            lessonId: 'diffusion-basics',
          },
          {
            id: 'tokenization-dlm',
            label: 'Tokenization',
            tooltip: tip({
              short: 'Text becomes a sequence of discrete token IDs.',
              intuition: 'Diffusion state is a sequence in vocabulary space.',
              example: 'BPE merges word pieces into subword tokens.',
              trap: 'Mask token must exist in vocabulary or special handling.',
            }),
            lessonId: 'tokenization',
          },
          {
            id: 'ar-gen-dlm',
            label: 'Autoregressive generation',
            tooltip: tip({
              short: 'AR models factorize P(x) left-to-right one token at a time.',
              intuition: 'Diffusion LM contrasts with sequential commitment.',
              example: 'GPT decode: predict x_i given x_{<i}.',
              trap: 'AR KV-cache optimizations do not transfer directly.',
            }),
            lessonId: 'transformer-token-generation',
          },
          {
            id: 'transformer-dlm',
            label: 'Transformer denoiser',
            tooltip: tip({
              short: 'Bidirectional or masked transformer predicts clean tokens from corrupted sequence.',
              intuition: 'Same backbone family as BERT-style encoders or denoising decoders.',
              example: 'Inputs: x_t, t embedding; output: logits per position.',
              trap: 'Causal masking not required for full-sequence denoise steps.',
            }),
            lessonId: 'transformer',
          },
          {
            id: 'sampling-dlm',
            label: 'Sampling strategies',
            tooltip: tip({
              short: 'Temperature, top-k, and confidence thresholds shape token picks at each step.',
              intuition: 'Diffusion schedules decide which positions update when.',
              example: 'Lock high-confidence tokens; remask uncertain ones.',
              trap: 'Greedy locking can trap early wrong tokens.',
            }),
            lessonId: 'sampling-strategies',
          },
          {
            id: 'masking-objective-dlm',
            label: 'Masked LM objective',
            tooltip: tip({
              short: 'Predict hidden tokens from context—pretraining cousin of denoise step.',
              intuition: 'MLM teaches local infilling before full diffusion training.',
              example: 'BERT predicts [MASK] positions.',
              trap: 'MLM alone is not full generative diffusion without reverse schedule.',
            }),
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'forward-mask-dlm',
            label: 'Forward masking',
            tooltip: tip({
              short: 'Corrupt clean x_0 toward x_t by masking or replacing tokens by schedule.',
              intuition: 'Higher t → more masked/corrupted positions.',
              formula: 'q(x_t|x_0)=\\operatorname{mask}(x_0,t)',
              example: 't low: few masks; t high: mostly [MASK] tokens.',
              trap: 'Mask schedule defines training curriculum.',
            }),
            highlightTarget: { panel: 'animation', type: 'forward-mask' },
          },
          {
            id: 'reverse-predict-dlm',
            label: 'Reverse prediction',
            tooltip: tip({
              short: 'Model predicts clean token distribution at each position given x_t, t.',
              intuition: 'p_θ(x_i|x_t,t) or joint update over positions.',
              formula: 'p_\\theta(x_i|x_t,t)',
              example: 'Logits over vocabulary at masked sites.',
              trap: 'Independent position updates may need coupling for fluency.',
            }),
            highlightTarget: { panel: 'animation', type: 'reverse-predict' },
          },
          {
            id: 'confidence-lock-dlm',
            label: 'Confidence locking',
            tooltip: tip({
              short: 'Commit high-confidence predictions; leave uncertain positions masked.',
              intuition: 'Gradual unmasking reduces parallel error spread.',
              example: 'Threshold 0.9 → lock token if max prob ≥ 0.9.',
              trap: 'Too-aggressive locking freezes mistakes early.',
            }),
            highlightTarget: { panel: 'animation', type: 'confidence-lock' },
          },
          {
            id: 'remask-dlm',
            label: 'Remasking',
            tooltip: tip({
              short: 'Low-confidence committed tokens can return to mask state later.',
              intuition: 'Revision distinguishes diffusion from one-pass infilling.',
              example: 'Step 5 locks token; step 8 remasks after context shift.',
              trap: 'Without remask, errors persist to end.',
            }),
            highlightTarget: { panel: 'animation', type: 'remask' },
          },
          {
            id: 'block-diffusion-dlm',
            label: 'Block diffusion',
            tooltip: tip({
              short: 'Generate text in blocks—hybrid of AR chunks and intra-block diffusion.',
              intuition: 'Balances parallelism with long-range left-to-right structure.',
              example: 'Generate 64-token block in parallel, append, next block.',
              trap: 'Block boundaries affect coherence.',
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'draft-refine-dlm',
            label: 'Draft and refine',
            tooltip: tip({
              short: 'Whole response is a draft edited over multiple passes.',
              intuition: 'Unlike AR, later steps can revise earlier words.',
              example: 'Infilling middle of sentence after ends fixed.',
              trap: 'Revision depth depends on schedule and remask policy.',
            }),
          },
          {
            id: 'parallel-decode-dlm',
            label: 'Parallel decoding',
            tooltip: tip({
              short: 'Many positions update in same step—potential throughput win.',
              intuition: 'Wall-clock may beat serial AR if steps × width trade well.',
              example: '512 positions update per denoise step in theory.',
              trap: 'Transformer cost still scales with sequence length each step.',
            }),
          },
          {
            id: 'ar-vs-diffusion-dlm',
            label: 'AR vs diffusion tradeoff',
            tooltip: tip({
              short: 'AR: simple serving and strong left-to-right fluency; diffusion: editability and infilling.',
              intuition: 'Choose paradigm for task: continuation vs revise-in-place.',
              example: 'Editing paragraph vs streaming chat completion.',
              trap: 'Benchmarks on one task mis-rank the other.',
            }),
          },
          {
            id: 'infilling-strength-dlm',
            label: 'Infilling strength',
            tooltip: tip({
              short: 'Natural fit for holes in middle of text.',
              intuition: 'Bidirectional context at each denoise step helps fill gaps.',
              example: 'Prompt with [MASK] span to complete.',
              trap: 'Long-range coherence still needs many steps or blocks.',
            }),
          },
          {
            id: 'alignment-workflow-dlm',
            label: 'Alignment workflow',
            tooltip: tip({
              short: 'SFT/DPO on diffusion trajectories differs from AR token CE only.',
              intuition: 'Preference optimization must respect mask schedules.',
              example: 'DPO pairs on full denoised outputs vs partial paths.',
              trap: 'AR alignment recipes do not copy without adaptation.',
            }),
            lessonId: 'fine-tuning',
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'forward-q-dlm',
            label: 'Forward q(x_t|x_0)',
            tooltip: tip({
              short: 'Mask operator corrupts clean sequence toward noise state.',
              intuition: 'Discrete analog of adding noise in image diffusion.',
              formula: 'q(x_t|x_0)=\\operatorname{mask}(x_0,t)',
              example: 'Random mask probability increasing with t.',
              trap: 'Exact q family varies by model paper (absorbing, uniform replace).',
            }),
            highlightTarget: { panel: 'code', type: 'forward-q' },
          },
          {
            id: 'reverse-p-dlm',
            label: 'Reverse p_θ(x_i|x_t,t)',
            tooltip: tip({
              short: 'Predict clean token at position i from corrupted sequence.',
              intuition: 'Cross-entropy on masked positions during training.',
              formula: 'p_\\theta(x_i|x_t,t)',
              code: 'loss = F.cross_entropy(logits[mask], x0[mask])',
              example: 'Only compute loss on corrupted indices.',
              trap: 'Label smoothing affects confidence locking at sample time.',
            }),
            highlightTarget: { panel: 'code', type: 'reverse-p' },
          },
          {
            id: 'schedule-code-dlm',
            label: 'Denoise schedule',
            tooltip: tip({
              short: 'for t in T..1: predict, lock confident, remask weak.',
              intuition: 'Schedule hyperparameters control quality vs steps.',
              code: 'for t in reversed(range(T)):\n    logits = model(x_t, t)\n    x_t = update_and_remask(x_t, logits, threshold)',
              example: 'Tune threshold and T on dev set.',
              trap: 'Too few steps hurts fluency; too many slows serve.',
            }),
          },
          {
            id: 'compare-ar-dlm',
            label: 'AR baseline loop',
            tooltip: tip({
              short: 'for i in 1..L: append argmax p(x_i|x_{<i}).',
              intuition: 'Side-by-side evaluation on same task clarifies tradeoffs.',
              code: 'for _ in range(max_len):\n    logits = model(input_ids)\n    next_id = logits[0,-1].argmax()\n    input_ids = append(input_ids, next_id)',
              example: 'Compare latency, editability, and quality metrics.',
              trap: 'Fair compare needs matched model size and training data.',
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'faster-by-default-trap-dlm',
            label: 'Faster by default',
            tooltip: tip({
              short: 'Parallel steps still pay full-sequence transformer cost each time.',
              intuition: 'Speedup depends on T, hardware, and implementation.',
              example: 'T=64 steps × full attention may lose to AR with KV cache.',
              trap: 'Measure TTFT and tokens/sec on target workload.',
            }),
          },
          {
            id: 'image-diffusion-trap-dlm',
            label: 'Image diffusion paste',
            tooltip: tip({
              short: 'Discrete text needs mask schedules—not Gaussian ε on embeddings alone.',
              intuition: 'Token corruption and vocabulary constraints differ fundamentally.',
              example: 'Continuous embedding diffusion is a different branch.',
              trap: 'Read paper’s forward process before implementing sampler.',
            }),
          },
          {
            id: 'early-lock-trap-dlm',
            label: 'Early lock-in',
            tooltip: tip({
              short: 'Locking wrong tokens without remask propagates errors.',
              intuition: 'Confidence can be miscalibrated early in denoise.',
              example: 'High prob on wrong homophone frozen forever.',
              trap: 'Remask policy and temperature mitigate lock-in.',
            }),
          },
          {
            id: 'fluency-trap-dlm',
            label: 'Fluency vs editability',
            tooltip: tip({
              short: 'Revision flexibility can hurt left-to-right coherence vs strong AR.',
              intuition: 'Tasks needing streaming prose may still favor AR.',
              example: 'Long chat continuation benchmarks often AR-strong.',
              trap: 'Pick model family for product constraint, not hype.',
            }),
          },
          {
            id: 'serving-trap-dlm',
            label: 'Serving stack immaturity',
            tooltip: tip({
              short: 'KV-cache AR serving is mature; diffusion LM serving tooling is evolving.',
              intuition: 'Batch parallel denoise steps need new schedulers.',
              example: 'Speculative decoding tricks differ from AR Medusa.',
              trap: 'Production latency includes remask bookkeeping overhead.',
            }),
            lessonId: 'efficient-llm-serving',
          },
          {
            id: 'alignment-trap-dlm',
            label: 'Alignment transfer',
            tooltip: tip({
              short: 'RLHF/DPO recipes from AR may not align diffusion trajectories well.',
              intuition: 'Preference loss must cover mask paths and final outputs.',
              example: 'SFT on denoising objective ≠ chat alignment alone.',
              trap: 'Evaluate safety on full iterative decode, not single step.',
            }),
            lessonId: 'llm-training-objectives',
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'test-time-compute-dlm',
            label: 'Test-time compute',
            tooltip: tip({
              short: 'More denoise steps or remask rounds act like inference-time compute budget.',
              intuition: 'Thinking budgets apply to iterative refinement loops.',
              trap: 'Diminishing returns after sufficient T.',
            }),
            lessonId: 'test-time-compute-thinking-budgets',
          },
          {
            id: 'editing-apps-dlm',
            label: 'Text editing apps',
            tooltip: tip({
              short: 'Infilling and rewrite tasks leverage bidirectional denoise.',
              intuition: 'User selects span to regenerate while keeping context.',
              trap: 'Preserve constraints (length, tone) needs guided decoding.',
            }),
          },
          {
            id: 'dit-connection-dlm',
            label: 'DiT / multimodal',
            tooltip: tip({
              short: 'Shared diffusion mindset across image DiT and discrete text diffusion.',
              intuition: 'Frontier labs explore unified diffusion stacks.',
              trap: 'Modalities still differ in state space and encoders.',
            }),
            lessonId: 'dit',
          },
          {
            id: 'reasoning-models-dlm',
            label: 'Reasoning models',
            tooltip: tip({
              short: 'Iterative refinement parallels chain-of-thought revision patterns.',
              intuition: 'Draft answer → critique → revise maps to denoise/remask.',
              trap: 'Reasoning quality still needs verification and tools.',
            }),
            lessonId: 'tool-using-reasoning-models',
          },
          {
            id: 'frontier-arch-dlm',
            label: 'Frontier LLM architectures',
            tooltip: tip({
              short: 'Architecture catalogs compare AR, diffusion, and hybrid block models.',
              intuition: 'Placement in roadmap clarifies research vs production choices.',
              trap: 'Leaderboard on one benchmark does not crown a paradigm.',
            }),
            lessonId: 'frontier-llm-architecture-overview',
          },
          {
            id: 'diffusion-sampling-dlm',
            label: 'Diffusion sampling',
            tooltip: tip({
              short: 'Image sampling lessons inform step schedules and error accumulation.',
              intuition: 'Discrete text samplers are analogs of DDPM/DDIM ideas.',
              trap: 'Do not assume identical update equations.',
            }),
            lessonId: 'diffusion-sampling',
          },
        ],
      },
    ],
  },
  "actor-critic": {
    center: {
      id: "actor-critic",
      label: "Actor-Critic",
      type: 'current',
      tooltip: tip({
        short: "Actor-critic pairs a policy (actor) with a value estimator (critic) so policy updates use an advantage instead of raw returns alone.",
        intuition: "The actor tries actions; the critic answers “was this better than I expected?”—that gap is the learning signal.",
        formula: "A_t = G_t - V_\\phi(s_t),\\quad \\nabla J \\propto \\mathbb{E}[\\nabla\\log\\pi_\\theta(a_t|s_t)A_t]",
        why: "Actor-critic methods power modern RL from robotics to game playing by cutting policy-gradient variance while keeping direct policy optimization.",
        trap: "The critic does not choose actions; it estimates value so the actor learns from cleaner advantages.",
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: "ac-policy-gradients",
            label: "Policy gradients",
            tooltip: tip({
              short: "Optimize a stochastic policy by reinforcing sampled actions proportional to return.",
              intuition: "Actor-critic keeps this policy update but replaces raw return with advantage.",
              trap: "This is not tabular Q-learning; the policy parameters move directly.",
            }),
            lessonId: "policy-gradients",
          },
          {
            id: "ac-value-function",
            label: "Value function V(s)",
            tooltip: tip({
              short: "Expected return from state s under the current policy.",
              intuition: "The critic learns this baseline to judge whether an outcome beat expectations.",
              trap: "V(s) is not Q(s,a); it averages over actions the policy would take.",
            }),
            lessonId: "value-iteration",
          },
          {
            id: "ac-return",
            label: "Return G_t",
            tooltip: tip({
              short: "Discounted sum of future rewards from time t.",
              intuition: "Actor updates still depend on how well the trajectory paid off overall.",
              trap: "Monte Carlo returns can be high-variance without a critic baseline.",
            }),
            lessonId: "rl-foundations",
          },
          {
            id: "ac-exploration",
            label: "Exploration",
            tooltip: tip({
              short: "Try non-greedy actions so the policy discovers better behavior.",
              intuition: "Actor-critic still needs diverse samples to estimate both policy and value.",
              trap: "Pure exploitation freezes learning when the critic is wrong early.",
            }),
            lessonId: "rl-exploration",
          },
          {
            id: "ac-mdp",
            label: "MDP formalism",
            tooltip: tip({
              short: "States, actions, transitions, rewards, and discount γ frame sequential decisions.",
              intuition: "Both actor and critic updates assume Markovian state summaries.",
              trap: "Partial observability breaks naive V(s) unless state is augmented.",
            }),
            lessonId: "mdp-formalism",
          },
          {
            id: "ac-variance",
            label: "High-variance returns",
            tooltip: tip({
              short: "Single trajectories swing wildly around the true expected return.",
              intuition: "This motivates subtracting a learned baseline—the critic’s core job.",
              trap: "More episodes alone may not fix variance if reward scale is noisy.",
            }),
            lessonId: "expected-value-variance",
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: "ac-actor-role",
            label: "Actor role",
            tooltip: tip({
              short: "Parameterized policy π_θ(a|s) chooses action probabilities.",
              intuition: "Only the actor changes what gets executed in the environment.",
              trap: "Do not let the critic override action selection at deployment.",
            }),
          },
          {
            id: "ac-critic-role",
            label: "Critic role",
            tooltip: tip({
              short: "Value network V_φ(s) estimates expected return from each state.",
              intuition: "Critic targets move as the policy improves—bootstrapping tracks that drift.",
              trap: "A lagging critic sends wrong advantage signs to the actor.",
            }),
          },
          {
            id: "ac-advantage",
            label: "Advantage A_t",
            tooltip: tip({
              short: "Return minus baseline: was this step better than the critic expected?",
              intuition: "Positive advantage reinforces the sampled action; negative pushes it down.",
              formula: "A_t = G_t - V_\\phi(s_t)",
              trap: "Using raw return without baseline reintroduces high variance.",
            }),
          },
          {
            id: "ac-critic-loss",
            label: "Critic loss",
            tooltip: tip({
              short: "Fit V_φ(s_t) toward return or TD target from rewards and next values.",
              intuition: "Better value estimates make actor gradients more stable.",
              trap: "Overfitting the critic to old data mis-trains the actor after policy shifts.",
            }),
          },
          {
            id: "ac-joint-update",
            label: "Joint update loop",
            tooltip: tip({
              short: "Sample trajectory → update critic → compute advantage → update actor.",
              intuition: "Both networks co-evolve; neither is fixed while the other learns.",
              trap: "Updating the actor too fast relative to the critic destabilizes training.",
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: "ac-coach-analogy",
            label: "Coach and player",
            tooltip: tip({
              short: "Player (actor) acts; coach (critic) says whether performance beat expectation.",
              intuition: "You reinforce actions that outperform the coach’s forecast.",
              trap: "If the coach is always pessimistic, good actions get over-credited.",
            }),
          },
          {
            id: "ac-variance-cut",
            label: "Variance reduction",
            tooltip: tip({
              short: "Subtracting V(s) removes common “this state is just good/bad” noise.",
              intuition: "Advantage focuses credit on actions relative to state quality.",
              trap: "A biased critic still shifts all advantages the same direction.",
            }),
          },
          {
            id: "ac-bootstrap",
            label: "Bootstrapping",
            tooltip: tip({
              short: "Critic can learn from one-step TD targets, not only full returns.",
              intuition: "Faster feedback than waiting for episode end—common in A2C/A3C/PPO stacks.",
              trap: "Bootstrapping with bad V propagates error into advantages.",
            }),
          },
          {
            id: "ac-on-policy",
            label: "On-policy flavor",
            tooltip: tip({
              short: "Classic actor-critic often learns from data the current policy generated.",
              intuition: "Old trajectories misrepresent action probabilities after big policy moves.",
              trap: "Off-policy reuse needs importance sampling or separate replay tricks.",
            }),
          },
          {
            id: "ac-continuous-actions",
            label: "Continuous control fit",
            tooltip: tip({
              short: "Policy outputs distribution parameters (mean, std) for smooth action spaces.",
              intuition: "Same advantage idea applies whether actions are discrete or continuous.",
              trap: "Discrete softmax intuition does not transfer literally to Gaussian policies.",
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: "ac-advantage-formula",
            label: "Advantage formula",
            tooltip: tip({
              short: "A_t compares realized return to critic prediction at s_t.",
              intuition: "TD advantage variants use r + γV(s′) − V(s) for one-step credit.",
              formula: "A_t = G_t - V_\\phi(s_t)",
              trap: "Mixing Monte Carlo and TD advantages without naming which you use confuses debugging.",
            }),
          },
          {
            id: "ac-actor-grad",
            label: "Actor gradient",
            tooltip: tip({
              short: "Raise log-probability of taken actions weighted by advantage.",
              intuition: "Policy gradient theorem justifies this REINFORCE-style update with baseline.",
              formula: "\\nabla_\\theta J \\approx \\mathbb{E}[\\nabla_\\theta\\log\\pi_\\theta(a|s)A]",
              trap: "Zero advantage means zero policy gradient even if return was positive.",
            }),
          },
          {
            id: "ac-td-target",
            label: "TD critic target",
            tooltip: tip({
              short: "One-step target: r + γV(s′) for bootstrapped value learning.",
              intuition: "Blends immediate reward with critic’s opinion of what happens next.",
              formula: "y_t = r_t + \\gamma V_\\phi(s_{t+1})",
              trap: "Terminal states must zero out the bootstrap term.",
            }),
          },
          {
            id: "ac-gae-hint",
            label: "GAE intuition",
            tooltip: tip({
              short: "Generalized advantage estimation mixes multi-step returns with λ.",
              intuition: "λ→0 is one-step TD; λ→1 approaches Monte Carlo advantage.",
              trap: "λ is a bias–variance knob, not “always use 0.95.”",
            }),
          },
          {
            id: "ac-pseudocode",
            label: "Update pseudocode",
            tooltip: tip({
              short: "Collect rollout, compute advantages, backprop critic then actor.",
              intuition: "Implementation order and advantage normalization matter in practice.",
              code: "adv = returns - values\nloss_c = mse(values, targets)\nloss_a = -(log_probs * adv).mean()",
              trap: "Forgetting to detach value targets when backpropping actor causes critic leakage into policy.",
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: "ac-critic-controls",
            label: "Critic chooses actions",
            tooltip: tip({
              short: "Deployment policy is π_θ, not argmax of V or Q.",
              intuition: "Critic is training scaffolding, not the decision rule.",
              trap: "Evaluating argmax Q while training stochastic π confuses results.",
            }),
          },
          {
            id: "ac-moving-target",
            label: "Moving critic target",
            tooltip: tip({
              short: "As the actor improves, the same state’s true value changes.",
              intuition: "Non-stationary targets require conservative critic updates or target networks.",
              trap: "Chasing a critic trained on obsolete policy wastes samples.",
            }),
          },
          {
            id: "ac-wrong-baseline",
            label: "Wrong baseline",
            tooltip: tip({
              short: "State-independent or mis-scaled baselines do not center advantages.",
              intuition: "Baseline must track expected return conditioned on state.",
              trap: "Subtracting a global constant helps less than V(s) in long horizons.",
            }),
          },
          {
            id: "ac-entropy-collapse",
            label: "Entropy collapse",
            tooltip: tip({
              short: "Actor can become deterministic too fast and stop exploring.",
              intuition: "Entropy bonuses keep action diversity while the critic is immature.",
              trap: "Zero entropy with wrong critic locks in suboptimal behavior.",
            }),
          },
          {
            id: "ac-credit-noise",
            label: "Noisy credit",
            tooltip: tip({
              short: "Sparse or delayed rewards make single-step advantages misleading.",
              intuition: "Reward shaping or longer advantage horizons may be needed—but carefully.",
              trap: "Bad shaping teaches wrong skills even with a perfect critic.",
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: "ac-reward-shaping",
            label: "Reward shaping",
            tooltip: tip({
              short: "Dense hints can help actor-critic learn before sparse task reward arrives.",
              intuition: "Potential-based shaping preserves optimal policies when done correctly.",
              trap: "Shaping that changes the optimum breaks actor-critic convergence goals.",
            }),
            lessonId: "reward-shaping",
          },
          {
            id: "ac-ppo-family",
            label: "PPO / modern policy RL",
            tooltip: tip({
              short: "Production RL stacks combine actor-critic with clipped objectives and advantage norm.",
              intuition: "Same actor–critic split appears inside TRPO, PPO, SAC variants.",
              trap: "Hyperparameters from one algorithm do not transfer blindly.",
            }),
            lessonId: "policy-gradients",
          },
          {
            id: "ac-q-learning-contrast",
            label: "Q-learning contrast",
            tooltip: tip({
              short: "Q-learning stores action values; actor-critic stores policy plus V.",
              intuition: "Choose Q methods for discrete control tables; actor-critic for direct policies.",
              trap: "They solve overlapping problems with different data and stability profiles.",
            }),
            lessonId: "q-learning",
          },
          {
            id: "ac-exploration-used",
            label: "Exploration schedules",
            tooltip: tip({
              short: "Entropy and ε-noise interact with advantage quality early in training.",
              intuition: "Exploration feeds the critic diverse states to estimate V accurately.",
              trap: "Turning exploration off while critic is immature freezes bad policies.",
            }),
            lessonId: "rl-exploration",
          },
          {
            id: "ac-frontier-agents",
            label: "Tool-using agents",
            tooltip: tip({
              short: "RL fine-tuning of LLM agents often uses critic-like value baselines on token trajectories.",
              intuition: "High-variance sequence returns mirror classic actor-critic motivation.",
              trap: "Token-level credit assignment is harder than gridworld Monte Carlo.",
            }),
            lessonId: "tool-using-reasoning-models",
          },
        ],
      },
    ],
  },
  "agentic-coding-systems": {
    center: {
      id: "agentic-coding-systems",
      label: "Agentic Coding Systems",
      type: 'current',
      tooltip: tip({
        short: "Agentic coding loops connect issue understanding, repo search, planning, editing, testing, review, and approval into a controlled software-engineering workflow.",
        intuition: "The model is not a one-shot autocomplete—it is an operator that must prove a patch is narrow, tested, and safe before merge.",
        formula: "Issue \\rightarrow Search \\rightarrow Plan \\rightarrow Edit \\rightarrow Test \\rightarrow Review \\rightarrow PR",
        why: "Coding agents power issue triage, refactors, and test-driven fixes—but only when context, tests, and gates constrain scope.",
        trap: "Plausible code is not a fix; without FAIL_TO_PASS and PASS_TO_PASS evidence the patch may be wrong or unsafe.",
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: "acs-tool-agents",
            label: "Tool-using reasoning",
            tooltip: tip({
              short: "Agents call search, shell, browser, and editor tools inside a loop.",
              intuition: "Coding agents are tool policies with repo-specific permissions.",
              trap: "Unrestricted tools turn generation into uncontrolled execution.",
            }),
            lessonId: "tool-using-reasoning-models",
          },
          {
            id: "acs-token-gen",
            label: "Token generation",
            tooltip: tip({
              short: "Autoregressive decoding builds patches one token at a time.",
              intuition: "Long horizons increase drift unless checkpoints and plans anchor scope.",
              trap: "More tokens do not automatically mean better engineering discipline.",
            }),
            lessonId: "transformer-token-generation",
          },
          {
            id: "acs-rag",
            label: "Retrieval grounding",
            tooltip: tip({
              short: "Repo search and docs retrieval supply evidence for where to edit.",
              intuition: "Wrong file context is the dominant failure mode before bad syntax.",
              trap: "Fluent summaries of code the agent never opened look convincing.",
            }),
            lessonId: "rag",
          },
          {
            id: "acs-fine-tune",
            label: "Fine-tuning",
            tooltip: tip({
              short: "Specialized models learn patch formats, tool protocols, and review styles.",
              intuition: "Base LLM skill is not the same as repository workflow compliance.",
              trap: "Fine-tune on narrow tasks can overfit visible test patterns.",
            }),
            lessonId: "fine-tuning",
          },
          {
            id: "acs-debugging",
            label: "Model debugging mindset",
            tooltip: tip({
              short: "Localize whether failure is context, plan, patch, test, or deployment.",
              intuition: "Agent loops need the same systematic isolation as ML pipelines.",
              trap: "Retrying generation without diagnosing the stage wastes time and risk.",
            }),
            lessonId: "model-debugging",
          },
          {
            id: "acs-metrics",
            label: "Test outcome metrics",
            tooltip: tip({
              short: "Binary pass/fail on targeted and regression tests judge patch quality.",
              intuition: "SWE-bench style splits failing tests fixed vs passing tests preserved.",
              trap: "Passing visible tests alone can hide broken edge cases.",
            }),
            lessonId: "classification-metrics",
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: "acs-issue-intake",
            label: "Issue intake",
            tooltip: tip({
              short: "Parse bug report, stack trace, or feature request into a concrete goal.",
              intuition: "Ambiguous issues produce overbroad plans and wrong-file edits.",
              trap: "Skipping reproduction steps leads to patches that fix symptoms only.",
            }),
          },
          {
            id: "acs-repo-search",
            label: "Repo search",
            tooltip: tip({
              short: "Find symbols, callers, tests, and configs relevant to the issue.",
              intuition: "Search quality bounds plan quality—garbage context in, garbage patch out.",
              trap: "Keyword search without dependency awareness misses indirect callers.",
            }),
          },
          {
            id: "acs-plan",
            label: "Plan step",
            tooltip: tip({
              short: "Decompose the fix into files, functions, and test expectations before editing.",
              intuition: "Plans act as contracts that later diffs can be checked against.",
              trap: "Plans that skip test strategy often yield untestable or untested patches.",
            }),
          },
          {
            id: "acs-edit",
            label: "Edit step",
            tooltip: tip({
              short: "Apply minimal diffs to targeted locations with consistent style.",
              intuition: "Small patches reduce regression blast radius and review load.",
              trap: "Drive-by refactors obscure the actual fix in review.",
            }),
          },
          {
            id: "acs-verify",
            label: "Verify loop",
            tooltip: tip({
              short: "Run FAIL_TO_PASS and PASS_TO_PASS tests; retry on failure with new context.",
              intuition: "Verification is the proof obligation—not the model’s confidence.",
              trap: "Stopping after first green test without regression suite is unsafe.",
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: "acs-dev-loop",
            label: "Developer loop",
            tooltip: tip({
              short: "Mirror how engineers work: understand, locate, change, test, review.",
              intuition: "Agents succeed when workflow matches how repos actually evolve.",
              trap: "Skipping human review on risky commands assumes perfect model judgment.",
            }),
          },
          {
            id: "acs-scope-control",
            label: "Scope control",
            tooltip: tip({
              short: "Good agents touch few files and explain why each line changed.",
              intuition: "Scope drift is a leading cause of merge rejection and regressions.",
              trap: "“While I’m here” edits multiply failure modes.",
            }),
          },
          {
            id: "acs-evidence-bar",
            label: "Evidence bar",
            tooltip: tip({
              short: "A fix is real when specified tests flip from fail to pass without breaking others.",
              intuition: "Evidence beats narrative plausibility in software engineering.",
              trap: "Editing tests to match wrong behavior is reward hacking.",
            }),
          },
          {
            id: "acs-permission-gates",
            label: "Permission gates",
            tooltip: tip({
              short: "Dangerous commands, network access, and merges require explicit approval.",
              intuition: "Autonomy is graded—read-only search differs from arbitrary shell.",
              trap: "Auto-run shell on every retry invites destructive actions.",
            }),
          },
          {
            id: "acs-memory-limits",
            label: "Working memory limits",
            tooltip: tip({
              short: "Context windows force summarization, chunking, and re-search during long tasks.",
              intuition: "Checkpoints and structured notes compensate for lost verbatim history.",
              trap: "Stale summaries that drop failing test output mislead the next edit.",
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: "acs-swe-bench",
            label: "SWE-bench criterion",
            tooltip: tip({
              short: "Success requires targeted failing tests to pass and prior passing tests to stay green.",
              intuition: "FAIL_TO_PASS measures fix; PASS_TO_PASS measures regression safety.",
              trap: "Optimizing only FAIL_TO_PASS invites test gaming or narrow overfitting.",
            }),
          },
          {
            id: "acs-loop-diagram",
            label: "Control loop",
            tooltip: tip({
              short: "Observe repo state → act via tools → read test output → revise.",
              intuition: "Same sense–act–learn pattern as RL, but safety gates are stricter.",
              formula: "s_{t+1} = Env(s_t, tool_t)",
              trap: "Treating test output as optional logging breaks the loop.",
            }),
          },
          {
            id: "acs-diff-review",
            label: "Diff review artifact",
            tooltip: tip({
              short: "Human or automated review inspects unified diff before merge.",
              intuition: "Review catches scope creep, secret leaks, and logic holes tests miss.",
              trap: "Huge diffs defeat meaningful review even if CI is green.",
            }),
          },
          {
            id: "acs-checkpoint",
            label: "Checkpoint / rollback",
            tooltip: tip({
              short: "Save known-good tree state before risky edits or commands.",
              intuition: "Rollback turns exploratory agent behavior into reversible experiments.",
              trap: "No checkpoint means one bad command corrupts the workspace.",
            }),
          },
          {
            id: "acs-ci-integration",
            label: "CI integration",
            tooltip: tip({
              short: "Run project test suite in isolated environment matching production constraints.",
              intuition: "Local agent success must reproduce under CI images and env vars.",
              trap: "Missing dependencies in agent sandbox fake passes that CI rejects.",
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: "acs-plausible-patch",
            label: "Plausible patch trap",
            tooltip: tip({
              short: "Code looks idiomatic but fixes the wrong root cause.",
              intuition: "Always tie diff hunk to failing test assertion.",
              trap: "Narrative confidence substitutes for reproduction evidence.",
            }),
          },
          {
            id: "acs-wrong-file",
            label: "Wrong-file edit",
            tooltip: tip({
              short: "Similar symbol names in monorepos mislead retrieval.",
              intuition: "Verify package path, imports, and test module linkage.",
              trap: "Patching a homonym function silences symptoms elsewhere.",
            }),
          },
          {
            id: "acs-test-gaming",
            label: "Test gaming",
            tooltip: tip({
              short: "Weakening assertions or skipping tests to get green CI.",
              intuition: "Reward hacking appears when agents optimize pass rate not correctness.",
              trap: "Deleting failing tests is the extreme form of gaming.",
            }),
          },
          {
            id: "acs-unsafe-commands",
            label: "Unsafe commands",
            tooltip: tip({
              short: "rm -rf, credential exfiltration, or prod deploy without gates.",
              intuition: "Tool policies must default deny for destructive operations.",
              trap: "Prompt injection via issue text can steer shell tools.",
            }),
          },
          {
            id: "acs-scope-drift",
            label: "Scope drift",
            tooltip: tip({
              short: "Agent refactors unrelated modules while fixing a one-line bug.",
              intuition: "Keep plan and diff aligned; reject unplanned file touches.",
              trap: "Large context tempts “cleanup” that expands review surface.",
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: "acs-eval-safety",
            label: "Frontier evaluation",
            tooltip: tip({
              short: "Agent safety evals measure tool misuse, injection, and autonomy boundaries.",
              intuition: "Coding agents are a prime agentic risk surface.",
              trap: "Capability benchmarks ignore permission and rollback design.",
            }),
            lessonId: "frontier-evaluation-safety",
          },
          {
            id: "acs-monitoring",
            label: "Production monitoring",
            tooltip: tip({
              short: "Track agent success rate, retry count, test flakiness, and human override rate.",
              intuition: "Operational metrics detect drift in repo or model behavior.",
              trap: "Silent human fixes without telemetry hide systemic agent failures.",
            }),
            lessonId: "model-monitoring",
          },
          {
            id: "acs-ttc",
            label: "Test-time compute",
            tooltip: tip({
              short: "More thinking tokens or candidate patches can raise hard-issue solve rate.",
              intuition: "Best-of-N patches trade latency for SWE-bench-style success.",
              trap: "Extra samples without verification multiply unsafe diffs.",
            }),
            lessonId: "test-time-compute-thinking-budgets",
          },
          {
            id: "acs-uncertainty",
            label: "Uncertainty / defer",
            tooltip: tip({
              short: "Abstain or ask human when tests conflict or context is insufficient.",
              intuition: "Not every issue should be auto-merged.",
              trap: "Forced completion on ambiguous specs creates debt.",
            }),
            lessonId: "uncertainty-estimation",
          },
          {
            id: "acs-tool-using",
            label: "Tool policy design",
            tooltip: tip({
              short: "Which tools exist and how they are masked shapes agent capabilities.",
              intuition: "Coding agents inherit search/python/browser tool design choices.",
              trap: "Tool sprawl without scoping increases attack surface.",
            }),
            lessonId: "tool-using-reasoning-models",
          },
        ],
      },
    ],
  },
  "bag-of-words": {
    center: {
      id: "bag-of-words",
      label: "Bag of Words",
      type: 'current',
      tooltip: tip({
        short: "Bag-of-words represents a document as a vector of word counts (or weights), discarding grammar and word order.",
        intuition: "Each dimension asks “how often did this word appear?”—similar documents share heavy dimensions.",
        formula: "x_w = \\text{count}(w, D)",
        why: "BoW is the baseline text feature for search, spam filters, and topic models before dense embeddings.",
        trap: "Word order and negation disappear—“not bad” and “bad” can look alike if you only count “bad.”",
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: "bow-tokenization",
            label: "Tokenization",
            tooltip: tip({
              short: "Split raw text into tokens before counting.",
              intuition: "Token boundaries define what gets counted as one word.",
              trap: "Inconsistent tokenization breaks vocabulary alignment across docs.",
            }),
            lessonId: "tokenization",
          },
          {
            id: "bow-vocabulary",
            label: "Vocabulary",
            tooltip: tip({
              short: "Ordered list of distinct tokens the model knows.",
              intuition: "Only vocabulary words receive dimensions; others are dropped or UNK.",
              trap: "Training vocabulary must match inference vocabulary exactly.",
            }),
          },
          {
            id: "bow-vector",
            label: "Vector representation",
            tooltip: tip({
              short: "A document becomes a fixed-length numeric vector.",
              intuition: "Similar bags sit near each other in count space.",
              trap: "Vector length equals |V|, not document length in tokens.",
            }),
          },
          {
            id: "bow-counting",
            label: "Counting",
            tooltip: tip({
              short: "Increment dimension i when token w_i appears.",
              intuition: "Repeated words increase the same coordinate.",
              trap: "Double-counting from bad preprocessing inflates frequencies.",
            }),
          },
          {
            id: "bow-sparsity",
            label: "Sparsity",
            tooltip: tip({
              short: "Most entries are zero because each doc uses few words of a large vocab.",
              intuition: "Sparse storage makes large vocabs practical.",
              trap: "Dense materialization of huge vocabs wastes memory.",
            }),
          },
          {
            id: "bow-corpus",
            label: "Corpus",
            tooltip: tip({
              short: "Collection of documents from which vocabulary and statistics are built.",
              intuition: "Rare words and stopwords are corpus-dependent choices.",
              trap: "Vocabulary from one domain fails on another without adaptation.",
            }),
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: "bow-build-vocab",
            label: "Build vocabulary",
            tooltip: tip({
              short: "Scan training corpus; collect unique tokens (optionally filter rare/common).",
              intuition: "Vocabulary size trades expressiveness against noise and memory.",
              trap: "Including test-only words in vocab without UNK handling leaks information.",
            }),
          },
          {
            id: "bow-count-vector",
            label: "Count vector",
            tooltip: tip({
              short: "For each doc, set x[w]= occurrences of w.",
              intuition: "The bag is orderless—permuting words leaves x unchanged.",
              trap: "Phrases like “New York” split unless n-grams are added.",
            }),
          },
          {
            id: "bow-tfidf",
            label: "TF-IDF weighting",
            tooltip: tip({
              short: "Downweight common corpus words; upweight distinctive terms.",
              intuition: "IDF penalizes words that appear everywhere.",
              formula: "tfidf(w,D)=tf(w,D)\\cdot\\log\\frac{N}{df(w)}",
              trap: "IDF must be computed on training corpus only to avoid leakage.",
            }),
          },
          {
            id: "bow-normalize",
            label: "Normalization",
            tooltip: tip({
              short: "Scale vectors by L1/L2 norm or document length.",
              intuition: "Long documents otherwise look more similar to everything via raw counts.",
              trap: "Wrong norm makes cosine similarity misleading.",
            }),
          },
          {
            id: "bow-similarity",
            label: "Document similarity",
            tooltip: tip({
              short: "Compare bags with dot product or cosine after weighting.",
              intuition: "Shared rare words dominate similarity under TF-IDF.",
              trap: "Cosine on raw counts still favors length, not just topic.",
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: "bow-word-soup",
            label: "Word soup mental model",
            tooltip: tip({
              short: "Shake words out of a bag—only multiset remains.",
              intuition: "Syntax and discourse cues are intentionally discarded.",
              trap: "Tasks needing order need n-grams or sequence models.",
            }),
          },
          {
            id: "bow-topic-signal",
            label: "Topic signal",
            tooltip: tip({
              short: "Shared content words push documents together.",
              intuition: "“loan interest rate” docs share financial dimensions.",
              trap: "Function words alone rarely define topic without weighting.",
            }),
          },
          {
            id: "bow-high-dim",
            label: "High-dimensional space",
            tooltip: tip({
              short: "Each word is an axis; docs are points in ℝ^|V|.",
              intuition: "Curse of dimensionality motivates regularization and embeddings later.",
              trap: "Nearest neighbors in raw count space can be noisy at huge |V|.",
            }),
          },
          {
            id: "bow-baseline",
            label: "Strong baseline",
            tooltip: tip({
              short: "BoW + linear classifier often beats fancy models on small data.",
              intuition: "Simple features with good regularization are hard to beat for short text.",
              trap: "BoW caps performance on tasks needing compositionality.",
            }),
          },
          {
            id: "bow-negation-loss",
            label: "Negation loss",
            tooltip: tip({
              short: "“Not good” and “good” share the “good” dimension.",
              intuition: "Negation handling needs bigrams, parsing, or contextual embeddings.",
              trap: "Sentiment with BoW alone misses flipped polarity phrases.",
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: "bow-count-formula",
            label: "Count feature",
            tooltip: tip({
              short: "Raw term frequency in document D.",
              intuition: "Simplest BoW feature—interpretable and fast.",
              formula: "x_w=\\#(w,D)",
              trap: "Raw counts overweight long documents unless normalized.",
            }),
          },
          {
            id: "bow-binary",
            label: "Binary BoW",
            tooltip: tip({
              short: "x_w ∈ {0,1} indicating presence, not frequency.",
              intuition: "Reduces impact of repeated boilerplate.",
              trap: "Loses repetition signal useful for some spam detection tasks.",
            }),
          },
          {
            id: "bow-sklearn",
            label: "sklearn CountVectorizer",
            tooltip: tip({
              short: "Fit vocabulary on train; transform train and test consistently.",
              intuition: "Encapsulates tokenization, vocab, and sparse matrix output.",
              code: "cv = CountVectorizer(max_df=0.9)\nX_train = cv.fit_transform(train)\nX_test = cv.transform(test)",
              trap: "fit_transform on full data leaks test token statistics.",
            }),
          },
          {
            id: "bow-tfidf-code",
            label: "TfidfTransformer",
            tooltip: tip({
              short: "Apply IDF and normalization after counts.",
              intuition: "Pipeline: CountVectorizer → TfidfTransformer.",
              code: "tfidf = TfidfTransformer()\nX_train_tfidf = tfidf.fit_transform(X_train)",
              trap: "IDF fit on test documents leaks document frequency information.",
            }),
          },
          {
            id: "bow-cosine",
            label: "Cosine similarity",
            tooltip: tip({
              short: "Measure angle between sparse BoW vectors.",
              intuition: "Length-normalized comparison of word distribution shape.",
              trap: "Zero vectors from empty docs break similarity.",
            }),
            lessonId: "cosine-similarity",
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: "bow-order-blind",
            label: "Order blindness",
            tooltip: tip({
              short: "Permuting words leaves representation unchanged.",
              intuition: "Any task needing syntax needs richer features.",
              trap: "Expecting BoW to capture grammar is a category error.",
            }),
          },
          {
            id: "bow-vocab-leak",
            label: "Vocabulary leakage",
            tooltip: tip({
              short: "Building vocab on train+test inflates offline scores.",
              intuition: "Test-only tokens should not shape IDF or feature space fit.",
              trap: "Pipeline fit on all data is a classic leakage bug.",
            }),
          },
          {
            id: "bow-unk-handling",
            label: "UNK handling",
            tooltip: tip({
              short: "Unknown tokens at inference map to UNK or are dropped.",
              intuition: "OOV rate spikes on new domains.",
              trap: "Silently dropping OOV words erases critical rare entities.",
            }),
          },
          {
            id: "bow-stopwords",
            label: "Stopword choices",
            tooltip: tip({
              short: "Removing “the” helps some tasks, hurts others.",
              intuition: "Domain-specific stop lists beat one global list.",
              trap: "Aggressive stopword removal can delete discriminative short tokens.",
            }),
          },
          {
            id: "bow-sparsity-ml",
            label: "Sparse linear pitfalls",
            tooltip: tip({
              short: "Unregularized high-dim linear models overfit small corpora.",
              intuition: "Use regularization or dimensionality reduction.",
              trap: "Perfect train accuracy with huge vocab is often memorization.",
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: "bow-embeddings",
            label: "Dense embeddings",
            tooltip: tip({
              short: "Learn continuous word vectors that capture similarity beyond exact match.",
              intuition: "word2vec/GloVe replace sparse axes with shared geometry.",
              trap: "Embeddings still need context for polysemy—“bank” river vs money.",
            }),
            lessonId: "embeddings",
          },
          {
            id: "bow-word2vec",
            label: "Word2Vec",
            tooltip: tip({
              short: "Predict neighbors to place similar words nearby in vector space.",
              intuition: "BoW is the sparse precursor Word2Vec improves upon.",
              trap: "Word2Vec still ignores document-level order without extensions.",
            }),
            lessonId: "word2vec",
          },
          {
            id: "bow-rag",
            label: "RAG retrieval",
            tooltip: tip({
              short: "Sparse BM25/BoW retrieval remains competitive for keyword-heavy search.",
              intuition: "Hybrid sparse+dense retrieval uses BoW-style signals.",
              trap: "Pure dense retrieval can miss exact rare token matches.",
            }),
            lessonId: "rag",
          },
          {
            id: "bow-classifier",
            label: "Linear classifiers",
            tooltip: tip({
              short: "Logistic regression on BoW features is a classic NLP baseline.",
              intuition: "Interpretable weights per word for spam/topic tasks.",
              trap: "Linear boundary misses nonlinear phrase interactions.",
            }),
            lessonId: "logistic-regression",
          },
          {
            id: "bow-cosine-used",
            label: "Cosine similarity",
            tooltip: tip({
              short: "Compare BoW vectors for near-duplicate detection and clustering.",
              intuition: "Shared weighted terms imply topical nearness.",
              trap: "Cosine ignores magnitude of absolute confidence.",
            }),
            lessonId: "cosine-similarity",
          },
        ],
      },
    ],
  },
  "bayes-rule-ml": {
    center: {
      id: "bayes-rule-ml",
      label: "Bayes Rule for ML",
      type: 'current',
      tooltip: tip({
        short: "Bayes rule updates prior class beliefs into posterior probabilities after observing evidence.",
        intuition: "Posterior combines how likely the evidence is under each class with how common the class was beforehand.",
        formula: "P(y\\mid x)=\\frac{P(x\\mid y)P(y)}{P(x)}",
        why: "Bayes thinking underlies Naive Bayes, Bayesian optimization, uncertainty, and diagnosing false alarms in imbalanced ML.",
        trap: "P(y|x) and P(x|y) swap roles—likelihood is not posterior without priors and normalization.",
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: "br-conditional",
            label: "Conditional probability",
            tooltip: tip({
              short: "P(A|B) restricts the sample space to cases where B occurred.",
              intuition: "Posterior is conditional probability of class given features.",
              trap: "P(A|B) ≠ P(B|A) without Bayes linking them.",
            }),
            lessonId: "conditional-probability",
          },
          {
            id: "br-distributions",
            label: "Probability distributions",
            tooltip: tip({
              short: "Model uncertainty over outcomes with valid probabilities summing to one.",
              intuition: "Likelihoods and priors are distributions over parameters or classes.",
              trap: "Densities can exceed 1; only integrals must behave.",
            }),
            lessonId: "probability-distributions",
          },
          {
            id: "br-joint",
            label: "Joint and marginal",
            tooltip: tip({
              short: "P(x,y) decomposes; P(x) sums or integrates over y.",
              intuition: "Denominator P(x) ensures posteriors over classes sum to 1.",
              trap: "Forgetting normalization leaves numbers that are not probabilities.",
            }),
          },
          {
            id: "br-likelihood",
            label: "Likelihood P(x|y)",
            tooltip: tip({
              short: "How probable observed features are if class y were true.",
              intuition: "High likelihood pulls posterior toward that class—unless prior is tiny.",
              trap: "Likelihood is not “probability y is true given x.”",
            }),
          },
          {
            id: "br-prior",
            label: "Prior P(y)",
            tooltip: tip({
              short: "Belief about class before seeing this example—base rate.",
              intuition: "Rare diseases need strong evidence; common classes need less.",
              trap: "Ignoring priors is base-rate neglect.",
            }),
          },
          {
            id: "br-evidence",
            label: "Evidence P(x)",
            tooltip: tip({
              short: "Total probability of observing x across all classes.",
              intuition: "Normalizing constant scaling all posteriors comparably.",
              trap: "P(x) near zero with continuous x uses density—not a probability mass.",
            }),
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: "br-posterior",
            label: "Posterior P(y|x)",
            tooltip: tip({
              short: "Updated class probability after seeing features x.",
              intuition: "What you want for classification decisions and risk-aware thresholds.",
              formula: "P(y|x)=\\frac{P(x|y)P(y)}{P(x)}",
              trap: "Posterior depends on both signal quality and base rate.",
            }),
          },
          {
            id: "br-bayes-numerator",
            label: "Numerator product",
            tooltip: tip({
              short: "Likelihood times prior for each class before normalization.",
              intuition: "Unnormalized scores compare relative support for each y.",
              trap: "Comparing numerators across different x values is invalid.",
            }),
          },
          {
            id: "br-normalization",
            label: "Normalization",
            tooltip: tip({
              short: "Divide by P(x) so posteriors sum to 1 over classes.",
              intuition: "P(x)=Σ_y P(x|y)P(y) mixes all ways evidence could arise.",
              trap: "Skipping normalization when only comparing two classes on same x is OK; across x it is not.",
            }),
          },
          {
            id: "br-decision",
            label: "Decision threshold",
            tooltip: tip({
              short: "Pick class with highest posterior or compare to cost-sensitive threshold.",
              intuition: "0.5 threshold assumes symmetric errors and equal priors.",
              trap: "Accuracy-optimal threshold shifts with class imbalance and costs.",
            }),
          },
          {
            id: "br-continuous-x",
            label: "Continuous features",
            tooltip: tip({
              short: "Use likelihood models (Gaussian, Bernoulli) for P(x|y).",
              intuition: "Naive Bayes assumes conditional independence given y.",
              trap: "Wrong distributional assumption breaks likelihood ranking.",
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: "br-base-rate",
            label: "Base-rate story",
            tooltip: tip({
              short: "A positive test for a rare condition can still leave low posterior disease probability.",
              intuition: "False positives flood you when the condition is uncommon.",
              trap: "Sensitivity alone does not determine posterior confidence.",
            }),
          },
          {
            id: "br-medical-test",
            label: "Medical test analogy",
            tooltip: tip({
              short: "Hit rate and false alarm rate interact with prevalence.",
              intuition: "ML classifiers face the same arithmetic on imbalanced data.",
              trap: "95% accuracy sounds great until 99% negatives dominate.",
            }),
          },
          {
            id: "br-update-belief",
            label: "Belief updating",
            tooltip: tip({
              short: "Prior + evidence → posterior; repeat as new data arrives.",
              intuition: "Sequential Bayes treats yesterday’s posterior as today’s prior.",
              trap: "Non-independent duplicates double-count evidence.",
            }),
          },
          {
            id: "br-odds-form",
            label: "Odds form",
            tooltip: tip({
              short: "Posterior odds = prior odds × likelihood ratio.",
              intuition: "Likelihood ratio isolates evidence strength from base rate.",
              trap: "Odds and probabilities convert differently—do not mix formulas.",
            }),
          },
          {
            id: "br-imbalanced",
            label: "Imbalanced classes",
            tooltip: tip({
              short: "Prior reflects training prevalence or deployment base rate.",
              intuition: "Rebalancing training changes effective priors at inference.",
              trap: "SMOTE changes priors implicitly—recalibrate thresholds after.",
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: "br-formula",
            label: "Bayes formula",
            tooltip: tip({
              short: "Core identity linking inverse conditional probabilities.",
              intuition: "Every generative classifier evaluation uses this skeleton.",
              formula: "P(y|x)=\\frac{P(x|y)P(y)}{P(x)}",
              trap: "P(x|y) with high dimensions needs modeling assumptions.",
            }),
          },
          {
            id: "br-naive",
            label: "Naive Bayes assumption",
            tooltip: tip({
              short: "Features conditionally independent given class.",
              intuition: "Factorizes high-dimensional likelihood into per-feature terms.",
              formula: "P(x|y)=\\prod_i P(x_i|y)",
              trap: "Correlated features violate naive assumption and distort posteriors.",
            }),
          },
          {
            id: "br-log-space",
            label: "Log posteriors",
            tooltip: tip({
              short: "Compute log P(y|x) via log prior + log likelihood − log evidence.",
              intuition: "Prevents underflow when multiplying many small probabilities.",
              code: "log_post = log_prior + log_lik - log_evidence",
              trap: "Forgetting log-sum-exp for evidence over many classes causes -inf bugs.",
            }),
          },
          {
            id: "br-gaussian-nb",
            label: "Gaussian Naive Bayes",
            tooltip: tip({
              short: "Model each feature as Gaussian per class.",
              intuition: "Fast baseline for continuous tabular data.",
              trap: "Heavy tails and outliers break Gaussian likelihoods.",
            }),
          },
          {
            id: "br-sklearn-nb",
            label: "sklearn NaiveBayes",
            tooltip: tip({
              short: "Fit class priors and feature likelihoods; predict via argmax posterior.",
              intuition: "partial_fit supports streaming priors in some variants.",
              code: "clf = GaussianNB().fit(X_train, y_train)\nproba = clf.predict_proba(X_test)",
              trap: "predict_proba assumes naive model is well-specified.",
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: "br-inverse-fallacy",
            label: "Inverse fallacy",
            tooltip: tip({
              short: "Confusing P(y|x) with P(x|y).",
              intuition: "Likelihood answers a generative question; posterior answers discriminative.",
              trap: "“High P(x|y)” does not mean “high P(y|x)” without priors.",
            }),
          },
          {
            id: "br-base-rate-neglect",
            label: "Base-rate neglect",
            tooltip: tip({
              short: "Reporting sensitivity without prevalence misleads stakeholders.",
              intuition: "Always show posterior at realistic priors.",
              trap: "Leaderboard accuracy hides rare-class posterior collapse.",
            }),
          },
          {
            id: "br-double-count",
            label: "Double-counting evidence",
            tooltip: tip({
              short: "Duplicate correlated features act like repeated evidence.",
              intuition: "Naive Bayes overweight correlated spam words.",
              trap: "Feature selection should remove redundant likelihood factors.",
            }),
          },
          {
            id: "br-wrong-prior",
            label: "Wrong deployment prior",
            tooltip: tip({
              short: "Training prevalence ≠ production base rate.",
              intuition: "Recalibrate or adjust thresholds when prevalence shifts.",
              trap: "Undersampling positives without threshold fix breaks posteriors.",
            }),
          },
          {
            id: "br-calibration-gap",
            label: "Calibration gap",
            tooltip: tip({
              short: "Model scores may rank well but posteriors misstate frequency.",
              intuition: "Platt scaling or isotonic regression post-process outputs.",
              trap: "Treating softmax as calibrated Bayes posterior without checking.",
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: "br-mle",
            label: "Maximum likelihood",
            tooltip: tip({
              short: "MLE picks parameters maximizing P(data|θ)—related but not identical to full Bayes.",
              intuition: "Frequentist point estimate vs Bayesian posterior over θ.",
              trap: "MLE overfits without regularization or priors.",
            }),
            lessonId: "maximum-likelihood-estimation",
          },
          {
            id: "br-logistic",
            label: "Logistic regression",
            tooltip: tip({
              short: "Discriminative model estimates P(y|x) directly without modeling P(x|y).",
              intuition: "Contrasts generative Bayes with direct conditional modeling.",
              trap: "Logistic weights are not naive Bayes likelihood ratios.",
            }),
            lessonId: "logistic-regression",
          },
          {
            id: "br-nb-lesson",
            label: "Naive Bayes classifiers",
            tooltip: tip({
              short: "Classic linear-time baseline for text and tabular tasks.",
              intuition: "Bayes rule with independence assumption.",
              trap: "Violated independence still sometimes works surprisingly well on text.",
            }),
            lessonId: "knn-naive-bayes-svm",
          },
          {
            id: "br-calibration",
            label: "Calibration",
            tooltip: tip({
              short: "Check predicted probabilities against observed rates.",
              intuition: "Bayes posteriors should be frequency-honest at deployment.",
              trap: "Good ranking ≠ calibrated probability.",
            }),
            lessonId: "calibration",
          },
          {
            id: "br-uncertainty",
            label: "Uncertainty estimation",
            tooltip: tip({
              short: "Bayesian posteriors quantify belief; ensembles approximate it.",
              intuition: "Low posterior spread suggests confident classification.",
              trap: "Single-point NN softmax is not a full Bayesian posterior.",
            }),
            lessonId: "uncertainty-estimation",
          },
        ],
      },
    ],
  },
  "bloom-filter": {
    center: {
      id: "bloom-filter",
      label: "Bloom Filter",
      type: 'current',
      tooltip: tip({
        short: "A Bloom filter is a compact bit array plus hash functions that answers “possibly in set” or “definitely not in set.”",
        intuition: "Insertion sets k hash positions; membership checks whether all k bits are 1—shared bits create false positives but never false negatives.",
        formula: "p\\approx(1-e^{-kn/m})^k",
        why: "Bloom filters accelerate databases, caches, CDNs, and blockchain light clients by skipping expensive lookups for absent keys.",
        trap: "“Probably present” is not proof of membership; a zero bit proves absence.",
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: "bf-set-membership",
            label: "Set membership",
            tooltip: tip({
              short: "Decide whether an element belongs to a collection.",
              intuition: "Bloom filter approximates a set with one-sided error.",
              trap: "Approximate membership differs from exact hash set lookup.",
            }),
          },
          {
            id: "bf-hash-function",
            label: "Hash function",
            tooltip: tip({
              short: "Maps keys to integer indices in a bounded range.",
              intuition: "Multiple independent hashes reduce correlated collisions.",
              trap: "Weak or correlated hashes cluster bits and raise false positives.",
            }),
          },
          {
            id: "bf-bit-array",
            label: "Bit array",
            tooltip: tip({
              short: "Fixed-length array of 0/1 flags packed for space efficiency.",
              intuition: "m bits encode membership hints for n inserted items.",
              trap: "m too small saturates bits and destroys filter utility.",
            }),
          },
          {
            id: "bf-false-positive",
            label: "False positive",
            tooltip: tip({
              short: "Query returns “maybe present” for an absent key.",
              intuition: "All k bits happened to be 1 from other insertions.",
              trap: "False positives are expected; tune parameters to bound rate.",
            }),
          },
          {
            id: "bf-no-false-negative",
            label: "No false negatives",
            tooltip: tip({
              short: "After insert, query never returns absent if key was inserted.",
              intuition: "Bits only flip 0→1, never cleared on standard Bloom filter.",
              trap: "Deletions need counting Bloom or other variants—plain Bloom cannot delete safely.",
            }),
          },
          {
            id: "bf-probability",
            label: "Probability intuition",
            tooltip: tip({
              short: "Each bit is roughly Bernoulli after many inserts.",
              intuition: "Fill fraction drives false-positive formula.",
              trap: "Independence assumptions in formula are approximate but useful.",
            }),
            lessonId: "probability-distributions",
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: "bf-insert",
            label: "Insert operation",
            tooltip: tip({
              short: "Hash key to k indices; set each bit to 1.",
              intuition: "OR-ing bits records membership evidence.",
              trap: "Re-inserting same key is idempotent—only sets already-1 bits.",
            }),
          },
          {
            id: "bf-query",
            label: "Query operation",
            tooltip: tip({
              short: "Hash key; if any of k bits is 0, answer “definitely not.”",
              intuition: "One zero bit is proof the key was never inserted.",
              trap: "All ones only means “maybe”—not confirmation.",
            }),
          },
          {
            id: "bf-k-hashes",
            label: "Choice of k",
            tooltip: tip({
              short: "Number of hash functions per key.",
              intuition: "More hashes increase check strictness but fill bits faster.",
              trap: "k too large saturates array quickly; k too small raises collisions.",
            }),
          },
          {
            id: "bf-sizing",
            label: "Sizing m and n",
            tooltip: tip({
              short: "Pick bit array length m for expected insert count n.",
              intuition: "Optimal k often near (m/n) ln 2 for minimal false-positive rate.",
              formula: "k^\\*\\approx\\frac{m}{n}\\ln 2",
              trap: "Underestimating n blows false-positive budget.",
            }),
          },
          {
            id: "bf-fp-formula",
            label: "False-positive rate",
            tooltip: tip({
              short: "Approximate probability all k bits are 1 for absent key.",
              intuition: "Exponential fill term (1-e^{-kn/m}) raised to k.",
              formula: "p\\approx(1-e^{-kn/m})^k",
              trap: "Formula assumes ideal independent hashing—real hashes deviate slightly.",
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: "bf-safety-net",
            label: "Cheap prefilter",
            tooltip: tip({
              short: "Skip disk/network lookup when filter says absent.",
              intuition: "Most queries in negative-heavy workloads save latency.",
              trap: "Must still confirm positives with authoritative store.",
            }),
          },
          {
            id: "bf-shared-bits",
            label: "Shared bits",
            tooltip: tip({
              short: "Many keys OR into same positions—collisions are intentional tradeoff.",
              intuition: "Space savings come from aliasing bit patterns.",
              trap: "Assuming one bit maps to one key is wrong.",
            }),
          },
          {
            id: "bf-one-sided",
            label: "One-sided error",
            tooltip: tip({
              short: "Only positives are uncertain; negatives are certain.",
              intuition: "Design systems to tolerate occasional extra lookups.",
              trap: "Using Bloom output alone for security allowlists fails open on FP.",
            }),
          },
          {
            id: "bf-fill-ratio",
            label: "Fill ratio",
            tooltip: tip({
              short: "As array fills, false positives climb sharply.",
              intuition: "Monitor bit saturation in long-lived filters.",
              trap: "Never resizing a filter past planned n breaks guarantees.",
            }),
          },
          {
            id: "bf-vs-hashset",
            label: "Versus exact hash set",
            tooltip: tip({
              short: "Hash set stores keys; Bloom stores ~10 bits per item.",
              intuition: "Choose Bloom when memory dominates and FP acceptable.",
              trap: "Storing keys defeats Bloom’s space purpose.",
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: "bf-optimal-k",
            label: "Optimal k",
            tooltip: tip({
              short: "Minimize false-positive rate for given m and n.",
              intuition: "Derivative of p w.r.t. k yields ln 2 scaling.",
              formula: "k^\\*=(m/n)\\ln 2",
              trap: "Round k to integer and re-evaluate p—not continuous optimum.",
            }),
          },
          {
            id: "bf-optimal-m",
            label: "Optimal m",
            tooltip: tip({
              short: "Given n and target p, solve for required bits.",
              intuition: "m grows with n and with log(1/p).",
              formula: "m\\approx -\\frac{n\\ln p}{(\\ln 2)^2}",
              trap: "Underestimating p by order of magnitude undersizes m badly.",
            }),
          },
          {
            id: "bf-insert-code",
            label: "Insert pseudocode",
            tooltip: tip({
              short: "Set k bit positions modulo m.",
              intuition: "Use double hashing or distinct seeds for k functions.",
              code: "for i in range(k):\n  bits[hash(key,i) % m] = 1",
              trap: "Modulo bias if m not coprime with hash range—usually minor.",
            }),
          },
          {
            id: "bf-query-code",
            label: "Query pseudocode",
            tooltip: tip({
              short: "Return false on first zero bit.",
              intuition: "Short-circuit saves average query time.",
              code: "for i in range(k):\n  if not bits[hash(key,i) % m]:\n    return False\nreturn True  # maybe",
              trap: "Returning True without confirmation step violates safe API design.",
            }),
          },
          {
            id: "bf-py-bloom",
            label: "Python bloom filter",
            tooltip: tip({
              short: "Libraries implement sized filters with planned capacity and error rate.",
              intuition: "Constructor picks m and k from target n and p.",
              code: "from pybloom_live import BloomFilter\nbf = BloomFilter(capacity=1000, error_rate=0.01)",
              trap: "Exceeding capacity degrades error rate without resize support.",
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: "bf-fp-as-proof",
            label: "Treating maybe as yes",
            tooltip: tip({
              short: "Downstream must verify positives against source of truth.",
              intuition: "Bloom is a hint, not authoritative membership.",
              trap: "Security systems that skip verification on “maybe” are unsafe.",
            }),
          },
          {
            id: "bf-no-delete",
            label: "Deletion on plain Bloom",
            tooltip: tip({
              short: "Clearing bits breaks other keys sharing those bits.",
              intuition: "Use counting Bloom filters or rebuild to support removal.",
              trap: "Deleting by zeroing bits creates false negatives.",
            }),
          },
          {
            id: "bf-hash-correlation",
            label: "Correlated hashes",
            tooltip: tip({
              short: "Poor hash family makes bits clump non-uniformly.",
              intuition: "Independent-looking indices spread saturation evenly.",
              trap: "Using only low bits of one hash for all k indices fails.",
            }),
          },
          {
            id: "bf-capacity-exceed",
            label: "Capacity exceed",
            tooltip: tip({
              short: "Inserting far beyond planned n raises FP above target.",
              intuition: "Size filter for peak load plus headroom.",
              trap: "Dynamic sets without rebuild eventually lie statistically.",
            }),
          },
          {
            id: "bf-serialization",
            label: "Serialization drift",
            tooltip: tip({
              short: "Filter bits and parameters must match across services.",
              intuition: "Different m, k, or hash seeds desynchronize answers.",
              trap: "Deploying new hash seed without rebuild invalidates old filter.",
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: "bf-rag-index",
            label: "Vector index prefilter",
            tooltip: tip({
              short: "Cheaply exclude impossible buckets before ANN search.",
              intuition: "Negative-heavy retrieval benefits from one-sided filters.",
              trap: "False positives add extra ANN work—not fatal if bounded.",
            }),
            lessonId: "rag-vector-indexing",
          },
          {
            id: "bf-pagerank",
            label: "Large-scale graphs",
            tooltip: tip({
              short: "Probabilistic structures appear in web-scale algorithms alongside exact sets.",
              intuition: "Space–accuracy tradeoffs recur at billion-node scale.",
              trap: "Bloom does not replace graph traversal correctness checks.",
            }),
            lessonId: "pagerank",
          },
          {
            id: "bf-cache",
            label: "CDN / cache negative lookup",
            tooltip: tip({
              short: "Avoid origin fetch when object definitely not in edge cache.",
              intuition: "Absence proofs save bandwidth.",
              trap: "Positive cache still needs TTL and validation.",
            }),
          },
          {
            id: "bf-db",
            label: "Database indexing",
            tooltip: tip({
              short: "Skip disk pages known not to contain keys.",
              intuition: "Used in LSM and distributed storage systems.",
              trap: "Must combine with exact page search for positives.",
            }),
          },
          {
            id: "bf-probability-used",
            label: "Probability modeling",
            tooltip: tip({
              short: "Design m, k, n using approximate probability formulas.",
              intuition: "Connects to distribution intuition for bit fill.",
              trap: "Treat formula as design guide, not cryptographic guarantee.",
            }),
            lessonId: "probability-distributions",
          },
        ],
      },
    ],
  },
  "classifier-free-guidance": {
    center: {
      id: "classifier-free-guidance",
      label: "Classifier-Free Guidance",
      type: 'current',
      tooltip: tip({
        short: "Classifier-free guidance steers diffusion sampling by amplifying the gap between conditional and unconditional noise predictions.",
        intuition: "Ask the denoiser “with prompt” and “without prompt,” then push the update toward what the prompt adds.",
        formula: "\\hat{\\epsilon}=\\epsilon_{uncond}+s(\\epsilon_{cond}-\\epsilon_{uncond})",
        why: "CFG is the default prompt-strength knob in Stable Diffusion, SD3, and many text-to-image pipelines—no separate classifier required.",
        trap: "Higher guidance scale is not always better; extreme s oversaturates, distorts anatomy, and reduces diversity.",
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: "cfg-diffusion-basics",
            label: "Diffusion basics",
            tooltip: tip({
              short: "Models learn to predict noise added during forward corruption.",
              intuition: "Guidance modifies each denoising step’s noise estimate.",
              trap: "CFG does not replace training a denoiser—it steers an existing one.",
            }),
            lessonId: "diffusion-basics",
          },
          {
            id: "cfg-sampling",
            label: "Diffusion sampling",
            tooltip: tip({
              short: "Iteratively denoise from x_T toward a sample x_0.",
              intuition: "Guidance applies at every reverse step.",
              trap: "Broken sampler + strong CFG still fails.",
            }),
            lessonId: "diffusion-sampling",
          },
          {
            id: "cfg-conditioning",
            label: "Text conditioning",
            tooltip: tip({
              short: "Prompt embeddings enter the denoiser via cross-attention or adapters.",
              intuition: "Conditional branch uses prompt; unconditional drops or nulls it.",
              trap: "Tokenizer truncation silently weakens conditioning signal.",
            }),
            lessonId: "t5-encoder",
          },
          {
            id: "cfg-epsilon",
            label: "Noise prediction ε",
            tooltip: tip({
              short: "Denoiser outputs estimated noise to subtract at timestep t.",
              intuition: "CFG combines two ε predictions before the sampler update.",
              trap: "Some models predict x0 or v-parameterization—convert consistently.",
            }),
          },
          {
            id: "cfg-uncond-training",
            label: "Unconditional training",
            tooltip: tip({
              short: "Randomly drop prompts during training so model learns ε_uncond.",
              intuition: "Dropout probability (~10–20%) enables classifier-free guidance at inference.",
              trap: "Never dropping prompts at train time removes unconditional branch.",
            }),
          },
          {
            id: "cfg-scale-intuition",
            label: "Guidance scale s",
            tooltip: tip({
              short: "Scalar multiplying conditional-minus-unconditional direction.",
              intuition: "s=1 is no extra push; s>1 strengthens prompt adherence.",
              trap: "s=0 or wrong branch swap inverts intended steering.",
            }),
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: "cfg-two-forward",
            label: "Two forward passes",
            tooltip: tip({
              short: "Run denoiser once with prompt, once without (or null prompt).",
              intuition: "Difference vector points toward prompt-specific correction.",
              trap: "Batching both passes wrong shares weights incorrectly.",
            }),
          },
          {
            id: "cfg-combine",
            label: "Combine predictions",
            tooltip: tip({
              short: "Add unconditional estimate plus scaled conditional offset.",
              intuition: "Linear extrapolation in noise-prediction space.",
              formula: "\\hat{\\epsilon}=\\epsilon_u+s(\\epsilon_c-\\epsilon_u)",
              trap: "Using wrong ε parameterization breaks sampler math.",
            }),
          },
          {
            id: "cfg-sampler-step",
            label: "Sampler integration",
            tooltip: tip({
              short: "Guided ε feeds DDPM/DDIM/flow step to update x_t.",
              intuition: "Guidance affects every timestep trajectory.",
              trap: "Few-step samplers amplify CFG artifacts when mis-tuned.",
            }),
          },
          {
            id: "cfg-null-prompt",
            label: "Null / empty prompt",
            tooltip: tip({
              short: "Unconditional branch uses learned dropout representation.",
              intuition: "Not necessarily literal empty string—training defines null token.",
              trap: "Changing null prompt at inference shifts unconditional anchor.",
            }),
          },
          {
            id: "cfg-per-step",
            label: "Per-timestep effect",
            tooltip: tip({
              short: "Early steps set layout; late steps refine detail—CFG impact varies.",
              intuition: "Some schedules reduce guidance at low noise levels.",
              trap: "Uniform high s all timesteps often creates plastic look.",
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: "cfg-contrast",
            label: "Contrastive steering",
            tooltip: tip({
              short: "Amplify what prompt changes relative to generic generation.",
              intuition: "Like highlighting difference signal before update.",
              trap: "When ε_c ≈ ε_u, guidance does nothing useful.",
            }),
          },
          {
            id: "cfg-no-classifier",
            label: "No separate classifier",
            tooltip: tip({
              short: "Classic guidance used classifier gradients; CFG embeds contrast in denoiser.",
              intuition: "One network, two conditioning modes.",
              trap: "Name “classifier-free” still confuses newcomers expecting a CLIP classifier loop.",
            }),
          },
          {
            id: "cfg-diversity-trade",
            label: "Diversity tradeoff",
            tooltip: tip({
              short: "Higher s reduces sample variety for same prompt.",
              intuition: "Strong pull toward conditional mode collapses stochastic paths.",
              trap: "Production often sweeps s against human eval, not max s.",
            }),
          },
          {
            id: "cfg-artifact",
            label: "Artifact risk",
            tooltip: tip({
              short: "Extreme guidance yields oversaturated colors and warped anatomy.",
              intuition: "Linear extrapolation overshoots valid noise manifold.",
              trap: "Fixing bad prompts with s>12 often worsens image quality.",
            }),
          },
          {
            id: "cfg-negative-prompt",
            label: "Negative prompt interaction",
            tooltip: tip({
              short: "Some pipelines implement negatives via alternate conditioning—related but not identical to CFG math.",
              intuition: "Understand your framework’s actual implementation.",
              trap: "Assuming SD negative prompt equals CFG formula causes debugging confusion.",
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: "cfg-formula",
            label: "CFG formula",
            tooltip: tip({
              short: "Standard Ho & Salimans combination rule.",
              intuition: "s=0 recovers unconditional; s=1 uses conditional offset once added to uncond base.",
              formula: "\\hat{\\epsilon}=\\epsilon_u+s(\\epsilon_c-\\epsilon_u)",
              trap: "Some codebases use ε_c + s(ε_c − ε_u)—verify convention.",
            }),
          },
          {
            id: "cfg-implement",
            label: "Implementation sketch",
            tooltip: tip({
              short: "Compute both epsilons; blend; pass to scheduler step.",
              intuition: "Can batch [cond, uncond] in one forward with doubled batch dim.",
              code: "eps = eps_u + scale * (eps_c - eps_u)\nx_prev = scheduler.step(eps, t, x).prev_sample",
              trap: "Forgetting to duplicate latents for batched CFG doubles batch wrong.",
            }),
          },
          {
            id: "cfg-schedule",
            label: "Guidance schedule",
            tooltip: tip({
              short: "Scale s as function of timestep t.",
              intuition: "Lower s late preserves texture; higher s early locks composition.",
              trap: "Schedules are heuristic—not universal constants.",
            }),
          },
          {
            id: "cfg-training-drop",
            label: "Training dropout",
            tooltip: tip({
              short: "Replace prompt with null token at probability p_uncond.",
              intuition: "p_uncond trades unconditional quality vs guidance range.",
              trap: "Too little dropout makes ε_u poor and CFG unstable.",
            }),
          },
          {
            id: "cfg-sdxl-dual",
            label: "Dual text encoders",
            tooltip: tip({
              short: "Some models concatenate CLIP+T5 embeddings before denoiser.",
              intuition: "CFG still applies on combined conditioning tensor.",
              trap: "Partial encoder failure breaks both branches silently.",
            }),
            lessonId: "clip-encoder",
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: "cfg-max-scale",
            label: "Max scale myth",
            tooltip: tip({
              short: "s=15 is not “more creative”—often destructive.",
              intuition: "Tune s on validation prompts and human ratings.",
              trap: "Copying community “magic s” without model match fails.",
            }),
          },
          {
            id: "cfg-identical-eps",
            label: "Identical branches",
            tooltip: tip({
              short: "Broken null conditioning yields ε_c ≈ ε_u.",
              intuition: "Guidance knob appears dead; check training dropout.",
              trap: "Wrong prompt encoding duplicate makes branches identical.",
            }),
          },
          {
            id: "cfg-param-mismatch",
            label: "Parameterization mismatch",
            tooltip: tip({
              short: "ε vs x0 vs v predictions need consistent conversion before CFG.",
              intuition: "Scheduler and model head must agree on target.",
              trap: "Mixing v-pred with ε-CFG formula corrupts samples.",
            }),
          },
          {
            id: "cfg-step-count",
            label: "Too few steps + high s",
            tooltip: tip({
              short: "Low step count plus aggressive guidance amplifies errors.",
              intuition: "Quality needs enough denoising iterations for steering.",
              trap: "Distilled few-step models need re-tuned guidance ranges.",
            }),
          },
          {
            id: "cfg-prompt-length",
            label: "Truncated prompts",
            tooltip: tip({
              short: "Long prompts clip; CFG pushes toward partial conditioning.",
              intuition: "Important tokens beyond limit never enter ε_c.",
              trap: "Blaming CFG for ignoring words that never encoded.",
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: "cfg-sd3",
            label: "SD3 pipeline",
            tooltip: tip({
              short: "Modern pipelines stack encoders, latent diffusion, and CFG in production.",
              intuition: "CFG remains the user-facing prompt strength control.",
              trap: "Pipeline version changes default s and schedulers.",
            }),
            lessonId: "sd3-overview",
          },
          {
            id: "cfg-dit",
            label: "DiT backbone",
            tooltip: tip({
              short: "Transformer denoisers apply same CFG combination on patch tokens.",
              intuition: "Architecture changes; guidance recipe persists.",
              trap: "Memory doubles when running paired forwards naively.",
            }),
            lessonId: "dit",
          },
          {
            id: "cfg-unet",
            label: "U-Net vs DiT",
            tooltip: tip({
              short: "Compare convolutional and transformer denoisers steered identically by CFG.",
              intuition: "Backbone inductive bias interacts with guidance artifacts.",
              trap: "Hyperparameters do not transfer across backbones.",
            }),
            lessonId: "unet-vs-dit",
          },
          {
            id: "cfg-flow",
            label: "Flow matching samplers",
            tooltip: tip({
              short: "Continuous-time models may use analogous guidance on velocity fields.",
              intuition: "Steering idea generalizes beyond DDPM ε.",
              trap: "Copying DDPM CFG code into flow sampler without derivation fails.",
            }),
            lessonId: "flow-matching",
          },
          {
            id: "cfg-clip",
            label: "CLIP encoder",
            tooltip: tip({
              short: "Text embeddings feed conditional branch in many SD systems.",
              intuition: "Encoder quality caps what CFG can emphasize.",
              trap: "Weak text alignment limits guidance ceiling.",
            }),
            lessonId: "clip-encoder",
          },
        ],
      },
    ],
  },
  "clip-encoder": {
    center: {
      id: "clip-encoder",
      label: "CLIP Text Encoder",
      type: 'current',
      tooltip: tip({
        short: "CLIP aligns images and text in a shared embedding space by training on contrastive image–caption pairs.",
        intuition: "Matching pairs are pulled together; non-matching pairs are pushed apart—text and image become comparable vectors.",
        formula: "\\mathcal{L}=-\\log\\frac{\\exp(\\text{sim}(I,T)/\\tau)}{\\sum_j \\exp(\\text{sim}(I,T_j)/\\tau)}",
        why: "CLIP powers zero-shot classification, retrieval, diffusion text conditioning, and multimodal search.",
        trap: "CLIP similarity is broad alignment—not guaranteed fine-grained grounding of every object named in text.",
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: "clip-embeddings",
            label: "Embeddings",
            tooltip: tip({
              short: "Fixed-length vectors represent discrete or continuous objects.",
              intuition: "CLIP maps modalities into one vector space.",
              trap: "Embedding dimension is not interpretable axis by axis.",
            }),
            lessonId: "embeddings",
          },
          {
            id: "clip-cosine",
            label: "Cosine similarity",
            tooltip: tip({
              short: "Measures angle between vectors, ignoring magnitude.",
              intuition: "Contrastive loss uses normalized dot products.",
              trap: "Comparing unnormalized vectors skews similarity.",
            }),
            lessonId: "cosine-similarity",
          },
          {
            id: "clip-tokenization",
            label: "Text tokenization",
            tooltip: tip({
              short: "Captions become token sequences for the text tower.",
              intuition: "BPE/WordPiece limits length and OOV handling.",
              trap: "Truncated captions drop rare object words.",
            }),
            lessonId: "tokenizer-bpe",
          },
          {
            id: "clip-contrastive",
            label: "Contrastive learning",
            tooltip: tip({
              short: "Learn by comparing positives vs in-batch negatives.",
              intuition: "Each image should match its caption, not others in batch.",
              trap: "Small batch size weakens negative diversity.",
            }),
          },
          {
            id: "clip-dual-encoder",
            label: "Dual encoders",
            tooltip: tip({
              short: "Separate image and text networks with shared similarity metric.",
              intuition: "Encoders specialize per modality then meet in embedding space.",
              trap: "Asymmetric towers need balanced capacity and training.",
            }),
          },
          {
            id: "clip-softmax",
            label: "Softmax over similarities",
            tooltip: tip({
              short: "Turn similarity scores into a distribution over candidate texts.",
              intuition: "Temperature τ scales logits before softmax.",
              trap: "τ too small collapses gradients; too large washes signal.",
            }),
            lessonId: "softmax",
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: "clip-image-tower",
            label: "Image encoder",
            tooltip: tip({
              short: "CNN or ViT maps image to vector z_I.",
              intuition: "Visual patterns compress into embedding aligned with language.",
              trap: "Resolution and augmentations affect tower invariances.",
            }),
          },
          {
            id: "clip-text-tower",
            label: "Text encoder",
            tooltip: tip({
              short: "Transformer maps tokens to vector z_T.",
              intuition: "Captions become points comparable to images.",
              trap: "Text tower may differ from GPT-style causal LM training.",
            }),
          },
          {
            id: "clip-normalize",
            label: "L2 normalization",
            tooltip: tip({
              short: "Unit-length embeddings before dot product.",
              intuition: "Dot product equals cosine for normalized vectors.",
              trap: "Skipping norm breaks temperature-calibrated contrastive loss.",
            }),
          },
          {
            id: "clip-batch-loss",
            label: "InfoNCE / contrastive loss",
            tooltip: tip({
              short: "Cross-entropy over in-batch pairing matrix.",
              intuition: "Diagonal entries are positives; off-diagonal are negatives.",
              formula: "sim(I,T)=z_I^\\top z_T",
              trap: "Duplicate captions in batch create ambiguous positives.",
            }),
          },
          {
            id: "clip-zero-shot",
            label: "Zero-shot classification",
            tooltip: tip({
              short: "Embed class name prompts; pick highest similarity image–text pair.",
              intuition: "“A photo of a {label}” templates matter.",
              trap: "Prompt engineering shifts zero-shot accuracy significantly.",
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: "clip-shared-space",
            label: "Shared space picture",
            tooltip: tip({
              short: "Images and sentences as points in one globe—near means related.",
              intuition: "Enables cross-modal retrieval in both directions.",
              trap: "Nearness reflects training data co-occurrence biases.",
            }),
          },
          {
            id: "clip-caption-bias",
            label: "Web caption bias",
            tooltip: tip({
              short: "Training on alt-text pairs inherits web noise and stereotypes.",
              intuition: "Alignment quality mirrors dataset cleanliness.",
              trap: "Expecting perfect compositional reasoning from CLIP alone.",
            }),
          },
          {
            id: "clip-template",
            label: "Prompt templates",
            tooltip: tip({
              short: "Wrap labels in natural phrases for better zero-shot.",
              intuition: "Language prior activates relevant visual features.",
              trap: "Wrong template family hurts all classes uniformly.",
            }),
          },
          {
            id: "clip-not-detector",
            label: "Not an object detector",
            tooltip: tip({
              short: "Global embedding may miss small or occluded objects.",
              intuition: "Whole-image summary loses spatial detail.",
              trap: "Using CLIP similarity as grounding proof for every noun.",
            }),
          },
          {
            id: "clip-multimodal-bridge",
            label: "Multimodal bridge",
            tooltip: tip({
              short: "Same vectors connect vision models to language interfaces.",
              intuition: "Diffusion systems inject CLIP text embeddings as conditioning.",
              trap: "Encoder frozen while denoiser trains may limit prompt vocabulary.",
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: "clip-sim-formula",
            label: "Similarity score",
            tooltip: tip({
              short: "Dot product of normalized image and text embeddings.",
              intuition: "Higher score ⇒ model believes pair matches.",
              formula: "s=\\frac{z_I^\\top z_T}{\\|z_I\\|\\|z_T\\|}",
              trap: "Compare scores only within same model checkpoint.",
            }),
          },
          {
            id: "clip-loss-formula",
            label: "Symmetric contrastive loss",
            tooltip: tip({
              short: "Image-to-text and text-to-image cross-entropy averaged.",
              intuition: "Both directions prevent collapse of one tower.",
              trap: "Implementing only I→T loss weakens text retrieval.",
            }),
          },
          {
            id: "clip-temperature",
            label: "Temperature τ",
            tooltip: tip({
              short: "Scales logits before softmax; learned or fixed hyperparameter.",
              intuition: "Controls hardness of negative pushing.",
              trap: "Copying τ from paper without matching batch size mis-scales.",
            }),
          },
          {
            id: "clip-openai-api",
            label: "Inference pattern",
            tooltip: tip({
              short: "Encode image and candidate texts; argmax similarity.",
              intuition: "Production caches text embeddings for fixed label sets.",
              code: "img = model.encode_image(x)\ntexts = model.encode_text(tokens)\nprobs = (img @ texts.T).softmax(dim=-1)",
              trap: "Forgetting eval mode and consistent preprocessing distorts vectors.",
            }),
          },
          {
            id: "clip-diffusion-hook",
            label: "Diffusion conditioning",
            tooltip: tip({
              short: "Text embedding sequence feeds cross-attention in U-Net/DiT.",
              intuition: "CFG uses same embeddings with null variant.",
              trap: "Sequence length limits differ between CLIP and T5 stacks.",
            }),
            lessonId: "joint-attention",
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: "clip-grounding-trap",
            label: "False grounding confidence",
            tooltip: tip({
              short: "High similarity does not prove object presence at location.",
              intuition: "Use segmentation or specialized grounding models when needed.",
              trap: "RAG with CLIP alone for fine spatial claims.",
            }),
          },
          {
            id: "clip-domain-shift",
            label: "Domain shift",
            tooltip: tip({
              short: "Sketches, medical, or satellite images may embed poorly.",
              intuition: "Pretraining domain dominates geometry.",
              trap: "Zero-shot on far OOD data without finetune or adapters.",
            }),
          },
          {
            id: "clip-prompt-hack",
            label: "Prompt sensitivity",
            tooltip: tip({
              short: "Synonyms and articles change scores.",
              intuition: "Ensemble multiple templates for stability.",
              trap: "Single brittle prompt for production classification.",
            }),
          },
          {
            id: "clip-batch-neg",
            label: "Weak negatives",
            tooltip: tip({
              short: "Tiny batches provide easy negatives.",
              intuition: "Large batch contrastive training is key to CLIP quality.",
              trap: "Training CLIP-like models with batch=32 expecting OpenAI results.",
            }),
          },
          {
            id: "clip-checkpoint-mix",
            label: "Checkpoint mismatch",
            tooltip: tip({
              short: "Image/text towers and preprocess must match checkpoint.",
              intuition: "Mean/std and resize differ across ViT variants.",
              trap: "Swapping text tower weights without image tower breaks space.",
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: "clip-multimodal",
            label: "Multimodal LLM",
            tooltip: tip({
              short: "Vision-language models extend CLIP-style alignment to dialogue.",
              intuition: "Shared embeddings remain the retrieval glue.",
              trap: "Chat fine-tune can drift from contrastive geometry.",
            }),
            lessonId: "multimodal-llm",
          },
          {
            id: "clip-joint-attn",
            label: "Joint attention",
            tooltip: tip({
              short: "Cross-attention injects text into image generation models.",
              intuition: "CLIP embeddings are classic conditioning signal.",
              trap: "Attention masks must separate modalities correctly.",
            }),
            lessonId: "joint-attention",
          },
          {
            id: "clip-sd3",
            label: "SD3 pipeline",
            tooltip: tip({
              short: "Production diffusion stacks combine CLIP with other encoders.",
              intuition: "Know which encoder supplies which semantic band.",
              trap: "Assuming one encoder captures all prompt nuance.",
            }),
            lessonId: "sd3-overview",
          },
          {
            id: "clip-rag",
            label: "Image retrieval RAG",
            tooltip: tip({
              short: "Retrieve images or captions by embedding similarity.",
              intuition: "Same cosine machinery as text RAG.",
              trap: "Retrieval miss when visual domain differs from training.",
            }),
            lessonId: "rag",
          },
          {
            id: "clip-cfg",
            label: "Classifier-free guidance",
            tooltip: tip({
              short: "Null text embedding defines unconditional branch in diffusion.",
              intuition: "CLIP quality affects CFG ceiling.",
              trap: "Broken null embedding makes guidance ineffective.",
            }),
            lessonId: "classifier-free-guidance",
          },
        ],
      },
    ],
  },
  "condition-number": {
    center: {
      id: "condition-number",
      label: "Condition Number",
      type: 'current',
      tooltip: tip({
        short: "The condition number κ(A) measures how much relative input error can be amplified into relative output error by solving Ax=b or applying A.",
        intuition: "Large κ means some input directions are stretched hugely while others are nearly collapsed—numerical solutions become fragile.",
        formula: "\\kappa(A)=\\|A\\|\\,\\|A^{-1}\\|=\\frac{\\sigma_{\\max}}{\\sigma_{\\min}}",
        why: "Condition numbers explain why least squares, inversion, and PCA directions can be correct in theory yet unstable in floating point.",
        trap: "A mathematically correct algorithm can still fail when κ is enormous—even if Ax=b looks simple on paper.",
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: "cn-matrix-mult",
            label: "Matrix multiplication",
            tooltip: tip({
              short: "Linear map A transforms vectors x to Ax.",
              intuition: "Condition number quantifies sensitivity of this map.",
              trap: "Ill-conditioning is about the map, not one arbitrary x.",
            }),
            lessonId: "matrix-multiplication",
          },
          {
            id: "cn-norms",
            label: "Matrix norms",
            tooltip: tip({
              short: "Measure size of vectors and induced operator size of A.",
              intuition: "κ uses ||A|| and ||A^{-1}|| for relative error bounds.",
              trap: "Different norms give different κ values—compare consistently.",
            }),
          },
          {
            id: "cn-svd-preview",
            label: "Singular values",
            tooltip: tip({
              short: "σ_max and σ_min describe strongest and weakest stretch directions.",
              intuition: "κ is ratio of largest to smallest positive singular value for square invertible A.",
              trap: "Near-zero σ_min means κ→∞ even if det looks small but nonzero in float.",
            }),
            lessonId: "svd",
          },
          {
            id: "cn-inverse",
            label: "Matrix inverse",
            tooltip: tip({
              short: "A^{-1} solves Ax=b when A is square and invertible.",
              intuition: "Large ||A^{-1}|| amplifies input perturbations in output.",
              trap: "Pseudoinverse needed when A is rectangular or rank-deficient.",
            }),
            lessonId: "pseudoinverse",
          },
          {
            id: "cn-least-squares",
            label: "Least squares",
            tooltip: tip({
              short: "Overdetermined systems minimize ||Ax−b||.",
              intuition: "Normal equations involve A^T A whose κ squares condition issues.",
              trap: "Forming A^T A explicitly often worsens conditioning.",
            }),
            lessonId: "least-squares-projection",
          },
          {
            id: "cn-float",
            label: "Floating-point error",
            tooltip: tip({
              short: "Finite precision introduces tiny perturbations to A and b.",
              intuition: "High κ turns machine epsilon into large solution error.",
              trap: "Double precision helps but cannot fix infinite κ.",
            }),
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: "cn-definition",
            label: "Definition of κ",
            tooltip: tip({
              short: "Bound on worst-case relative error amplification.",
              intuition: "Answers “how much can output relative error exceed input relative error?”",
              formula: "\\kappa(A)=\\|A\\|\\|A^{-1}\\|",
              trap: "Defined for invertible square matrices; rectangular cases use SVD variants.",
            }),
          },
          {
            id: "cn-svd-ratio",
            label: "SVD ratio form",
            tooltip: tip({
              short: "For invertible A, κ_2(A)=σ_max/σ_min.",
              intuition: "Nearly collinear columns create tiny σ_min and huge κ.",
              trap: "Rank-deficient A has σ_min=0 and infinite condition in exact arithmetic.",
            }),
          },
          {
            id: "cn-error-bound",
            label: "Error bound",
            tooltip: tip({
              short: "Relative solution error ≤ κ × relative data error (up to constants).",
              intuition: "κ is the multiplier connecting input noise to output uncertainty.",
              formula: "\\frac{\\|\\Delta x\\|}{\\|x\\|}\\lesssim \\kappa(A)\\frac{\\|\\Delta b\\|}{\\|b\\|}",
              trap: "Bound is worst-case—typical errors may be smaller.",
            }),
          },
          {
            id: "cn-near-null",
            label: "Near-null direction",
            tooltip: tip({
              short: "Input direction aligned with smallest σ gets enormously amplified by A^{-1}.",
              intuition: "Tiny noise in b along that mode swings x.",
              trap: "Visually “nice” b can hide bad components in singular vector basis.",
            }),
          },
          {
            id: "cn-compute",
            label: "Computing κ",
            tooltip: tip({
              short: "Estimate via SVD or norm estimators in libraries.",
              intuition: "Full SVD is gold standard for teaching; partial estimates used at scale.",
              code: "kappa = np.linalg.cond(A, 2)  # uses svd",
              trap: "cond based on norm estimates can mis-rank badly conditioned matrices.",
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: "cn-stretch",
            label: "Stretching ellipse",
            tooltip: tip({
              short: "Unit circle maps to ellipse whose axis ratio reflects κ.",
              intuition: "Skinny ellipse ⇒ solving rotates noise into huge x errors.",
              trap: "2D pictures generalize to high-dimensional subspaces.",
            }),
          },
          {
            id: "cn-almost-singular",
            label: "Almost singular",
            tooltip: tip({
              short: "Nearly dependent columns make A almost non-invertible.",
              intuition: "Determinant near zero often co-occurs with large κ.",
              trap: "Small determinant alone is not κ—use singular value ratio.",
            }),
          },
          {
            id: "cn-well-ill",
            label: "Well vs ill posed",
            tooltip: tip({
              short: "Well-conditioned problems tolerate noise; ill-conditioned ones do not.",
              intuition: "Engineering goal: reformulate or regularize to shrink κ.",
              trap: "Exact arithmetic solution can look perfect while float solution fails.",
            }),
          },
          {
            id: "cn-regularization",
            label: "Regularization effect",
            tooltip: tip({
              short: "Adding λI or truncating SVD caps effective κ.",
              intuition: "Trade bias for stability—Tikhonov lifts tiny σ.",
              trap: "Too much regularization underfits real signal.",
            }),
          },
          {
            id: "cn-scaling",
            label: "Feature scaling",
            tooltip: tip({
              short: "Poorly scaled columns inflate κ of design matrices.",
              intuition: "Normalize features before regression when appropriate.",
              trap: "Scaling changes κ but not predictive model if done consistently—still helps numerics.",
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: "cn-2norm",
            label: "2-norm condition",
            tooltip: tip({
              short: "Spectral condition using largest and smallest singular values.",
              intuition: "Most common in numerical linear algebra texts.",
              formula: "\\kappa_2(A)=\\sigma_{\\max}/\\sigma_{\\min}",
              trap: "σ_min from noisy SVD can be dominated by roundoff floor.",
            }),
          },
          {
            id: "cn-normal-eq",
            label: "Normal equations trap",
            tooltip: tip({
              short: "κ(A^T A) ≈ κ(A)^2.",
              intuition: "Explains why QR/SVD beats A^T A formation.",
              formula: "\\kappa(A^T A)\\approx \\kappa(A)^2",
              trap: "Using normal equations silently squares conditioning problems.",
            }),
          },
          {
            id: "cn-python",
            label: "numpy cond",
            tooltip: tip({
              short: "Quick condition estimate in practice.",
              intuition: "Use before trusting inv or solve on messy data.",
              code: "import numpy as np\nk = np.linalg.cond(A)\nif k > 1e10: print(\"unstable\")",
              trap: "Threshold depends on problem scale and precision.",
            }),
          },
          {
            id: "cn-qr-fix",
            label: "Stable solve",
            tooltip: tip({
              short: "QR or SVD least squares avoids squaring κ.",
              intuition: "Orthogonal Q preserves norms; does not form A^T A.",
              trap: "Plain Gaussian elimination without pivoting can fail before κ explodes visibly.",
            }),
            lessonId: "qr-decomposition",
          },
          {
            id: "cn-truncated-svd",
            label: "Truncated SVD solve",
            tooltip: tip({
              short: "Drop tiny singular values to stabilize pseudoinverse.",
              intuition: "Regularized inverse caps effective κ.",
              trap: "Truncation removes real signal if cutoff too aggressive.",
            }),
            lessonId: "pseudoinverse",
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: "cn-ignore-kappa",
            label: "Ignoring κ",
            tooltip: tip({
              short: "Trusting inverse or normal equations because Ax=b is square.",
              intuition: "Always inspect σ_min or cond when results look noisy.",
              trap: "Huge residual with tiny residual norm in b direction still possible.",
            }),
          },
          {
            id: "cn-wrong-norm",
            label: "Mixing norms",
            tooltip: tip({
              short: "Compare κ values only under same norm definition.",
              intuition: "κ_1, κ_2, κ_∞ differ numerically.",
              trap: "Quoting κ without subscript miscommunicates severity.",
            }),
          },
          {
            id: "cn-feature-scale-trap",
            label: "Unscaled features",
            tooltip: tip({
              short: "Columns differing by 1e6 inflate κ artificially.",
              intuition: "Standardize features for numerical stability in regression.",
              trap: "Thinking scaling “changes the math” when it preserves solution with proper care.",
            }),
          },
          {
            id: "cn-rank-deficient",
            label: "Rank deficiency",
            tooltip: tip({
              short: "True σ_min=0 means infinite κ—needs pseudoinverse or constraints.",
              intuition: "Collinear features in regression are classic case.",
              trap: "Ridge fixes numerics but changes estimand—document λ choice.",
            }),
          },
          {
            id: "cn-pretty-residual",
            label: "Misleading residual",
            tooltip: tip({
              short: "Small ||Ax−b|| does not imply accurate x when κ large.",
              intuition: "b may lie mostly in well-measured subspace while x error hides in bad direction.",
              trap: "Reporting residual only without κ or confidence intervals.",
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: "cn-least-squares-used",
            label: "Least squares",
            tooltip: tip({
              short: "Conditioning determines trust in fitted coefficients.",
              intuition: "Inspect κ of design matrix before interpreting weights.",
              trap: "Statistically significant p-values with unstable coefficients coexist.",
            }),
            lessonId: "least-squares-projection",
          },
          {
            id: "cn-pca-used",
            label: "PCA",
            tooltip: tip({
              short: "Small eigenvalues of covariance mirror ill-conditioned variance directions.",
              intuition: "PCA components with tiny variance are noise-sensitive.",
              trap: "Keeping all components ignores conditioning reality.",
            }),
            lessonId: "pca",
          },
          {
            id: "cn-svd-used",
            label: "SVD",
            tooltip: tip({
              short: "Singular values make conditioning visible and actionable.",
              intuition: "Same σ ratio story as κ in diagnostic plots.",
              trap: "Confusing singular values of A with eigenvalues unless A is symmetric.",
            }),
            lessonId: "svd",
          },
          {
            id: "cn-pinv-used",
            label: "Pseudoinverse",
            tooltip: tip({
              short: "Cutoff on σ stabilizes inversion of ill-posed systems.",
              intuition: "Regularized κ tied to truncation threshold.",
              trap: "Cutoff too high biases solution toward zero.",
            }),
            lessonId: "pseudoinverse",
          },
          {
            id: "cn-determinant",
            label: "Determinant link",
            tooltip: tip({
              short: "Near-zero determinant signals volume collapse related to ill-conditioning.",
              intuition: "det alone insufficient in high dimensions.",
              trap: "|det| tiny with moderate κ possible in non-symmetric matrices.",
            }),
            lessonId: "determinant-volume",
          },
        ],
      },
    ],
  },
'diffusion-sampling': {
    center: {
      id: 'diffusion-sampling',
      label: 'Diffusion Sampling',
      type: 'current',
      tooltip: tip({
        short: 'Diffusion sampling turns a trained denoiser into a generation procedure—choosing how to walk from noise toward data with DDPM, DDIM, or flow-style update rules.',
        intuition: 'Start from pure noise, repeatedly ask the model what noise to remove, and pick how much randomness each reverse step keeps.',
        formula: 'x_{t-1} \\leftarrow S_\\phi(x_t, t, \\hat{\\epsilon}_\\theta)',
        why: 'The same trained denoiser supports many samplers; sampler choice controls speed, diversity, determinism, and sensitivity to prediction errors.',
        trap: 'A sampler is not a new image model—it is a procedure for using the trained denoiser.',
        example: 'Compare DDPM (stochastic), DDIM (eta=0 deterministic), and flow paths with the same denoiser but different step counts.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'diffusion-basics-prereq',
            label: 'Diffusion basics',
            tooltip: tip({
              short: 'Forward noise corruption and reverse denoising define what the model was trained to predict.',
              intuition: 'Sampling assumes you already have a denoiser trained on noisy inputs at many timesteps.',
              example: 'The model predicts noise ε̂ at timestep t given x_t.',
              trap: 'Sampling cannot fix a denoiser that never learned the reverse process.',
            }),
            lessonId: 'diffusion-basics',
          },
          {
            id: 'noise-schedule-prereq',
            label: 'Noise schedule',
            tooltip: tip({
              short: 'Betas or sigmas define how much noise exists at each timestep t.',
              intuition: 'The sampler must use the same schedule the model saw during training.',
              example: 'Linear or cosine schedules spread corruption across T steps.',
              trap: 'Mismatched inference schedule hurts sample quality even with a good model.',
            }),
          },
          {
            id: 'denoiser-output-prereq',
            label: 'Denoiser prediction',
            tooltip: tip({
              short: 'The network outputs predicted noise, clean sample, or velocity depending on parameterization.',
              intuition: 'Each sampler rewrites the update using that prediction and the current x_t.',
              example: 'ε-prediction, x0-prediction, and v-prediction are common training targets.',
              trap: 'Plugging the wrong prediction type into an update formula breaks the path.',
            }),
          },
          {
            id: 'gaussian-noise-prereq',
            label: 'Gaussian noise',
            tooltip: tip({
              short: 'DDPM reverse steps inject controlled Gaussian noise when eta > 0.',
              intuition: 'Stochasticity explores the generative distribution; zero noise yields a fixed path.',
              example: 'x_T ~ N(0, I) initializes generation.',
              trap: 'Deterministic samplers still need a good denoiser—they do not add free diversity.',
            }),
            lessonId: 'probability-distributions',
          },
          {
            id: 'timestep-index-prereq',
            label: 'Timestep indexing',
            tooltip: tip({
              short: 'Reverse loops count down t = T, T-1, …, 0 (or a subsampled subset).',
              intuition: 'Fewer steps skip timesteps and approximate the continuous limit.',
              example: '8-step sampling uses a coarser grid than 50-step training noise.',
              trap: 'Subsampling timesteps without a compatible solver increases error.',
            }),
          },
          {
            id: 'latent-space-note',
            label: 'Latent diffusion context',
            tooltip: tip({
              short: 'Production systems often sample in a VAE-compressed latent space, not raw pixels.',
              intuition: 'The sampler logic is the same; only the tensor shape and scaling change.',
              example: 'Stable Diffusion denoises 64×64 latents that decode to 512×512 images.',
              trap: 'Pixel-space and latent-space step counts are not interchangeable budgets.',
            }),
            lessonId: 'diffusion-vae',
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Sampler mechanisms',
        type: 'mechanism',
        children: [
          {
            id: 'ddpm-stochastic',
            label: 'DDPM stochastic steps',
            tooltip: tip({
              short: 'Each reverse step adds controlled random noise while removing predicted corruption.',
              intuition: 'Stochasticity helps explore the data manifold instead of one deterministic trajectory.',
              example: 'eta > 0 injects variance consistent with the forward process.',
              trap: 'More noise per step is not always better—it can blur fine detail.',
            }),
            highlightTarget: { panel: 'animation', type: 'ddpm' },
          },
          {
            id: 'ddim-deterministic',
            label: 'DDIM deterministic path',
            tooltip: tip({
              short: 'Set eta = 0 to follow a fixed noise-to-sample trajectory.',
              intuition: 'Same initial noise and steps yield the same image—useful for editing and reproducibility.',
              example: 'DDIM often allows fewer steps than vanilla DDPM at similar quality.',
              trap: 'Deterministic paths can look sharper but less diverse across seeds.',
            }),
            highlightTarget: { panel: 'animation', type: 'ddim' },
          },
          {
            id: 'flow-ode-sampler',
            label: 'Flow / ODE sampler',
            tooltip: tip({
              short: 'Treat generation as integrating a velocity field from noise toward data.',
              intuition: 'Continuous-time view connects diffusion to flow matching and straight-path transport.',
              example: 'ODE solvers step along a learned velocity instead of discrete beta schedules.',
              trap: 'Flow samplers still depend on accurate vector-field or denoiser estimates.',
            }),
            highlightTarget: { panel: 'animation', type: 'flow' },
          },
          {
            id: 'step-count-tradeoff',
            label: 'Step-count tradeoff',
            tooltip: tip({
              short: 'Fewer reverse steps reduce compute but amplify per-step denoising error.',
              intuition: 'Each skipped step must be compensated by a solver that approximates the continuous limit.',
              example: '8 steps can be fast yet brittle when prediction quality drops.',
              trap: 'Cutting steps without retuning the solver often adds blur or artifacts.',
            }),
            highlightTarget: { panel: 'animation', type: 'steps' },
          },
          {
            id: 'prediction-quality',
            label: 'Prediction quality sensitivity',
            tooltip: tip({
              short: 'Sampler trajectories amplify or absorb denoiser mistakes differently.',
              intuition: 'Deterministic few-step paths punish bad ε̂ more visibly than high-step stochastic paths.',
              example: 'Lower prediction quality leaves residual noise in the final sample clarity bar.',
              trap: 'Blaming the sampler alone misses a weak denoiser or schedule mismatch.',
            }),
            highlightTarget: { panel: 'animation', type: 'quality' },
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'sampler-not-model',
            label: 'Sampler ≠ model',
            tooltip: tip({
              short: 'One trained denoiser supports many reverse update rules.',
              intuition: 'Changing DDPM → DDIM does not retrain weights—it changes the walk from noise.',
              example: 'Same checkpoint, three sampler buttons, three generation behaviors.',
              trap: 'Calling DDIM a separate “model family” confuses training from inference.',
            }),
          },
          {
            id: 'speed-quality-pareto',
            label: 'Speed vs quality Pareto',
            tooltip: tip({
              short: 'Real-time generation pushes step count down until quality or stability breaks.',
              intuition: 'Distilled samplers and better solvers move the frontier, not magic step removal.',
              example: 'Production may use 20–30 steps; research demos sometimes use 4–8 with distillation.',
              trap: 'Benchmarking one step count on one prompt hides variance across seeds and prompts.',
            }),
          },
          {
            id: 'stochasticity-diversity',
            label: 'Stochasticity and diversity',
            tooltip: tip({
              short: 'Random reverse noise explores different modes of the learned distribution.',
              intuition: 'Deterministic samplers repeat; stochastic ones vary even with the same start noise.',
              example: 'DDPM with eta > 0 yields seed-sensitive diversity.',
              trap: 'Zero stochasticity plus fixed seed removes sample diversity entirely.',
            }),
          },
          {
            id: 'residual-noise-clarity',
            label: 'Residual noise vs clarity',
            tooltip: tip({
              short: 'Final image sharpness tracks how much structured noise remains at t → 0.',
              intuition: 'Watch remaining noise fall as steps accumulate successful denoising.',
              example: 'Trajectory end-point clarity rises when prediction quality is high.',
              trap: 'A crisp preview at low steps can still collapse with a bad final step.',
            }),
          },
          {
            id: 'conditioning-later',
            label: 'Prompt conditioning comes next',
            tooltip: tip({
              short: 'Unconditional sampling learns p(x); guidance steers conditional paths later.',
              intuition: 'Sampler choice still matters once classifier-free guidance blends predictions.',
              example: 'CFG amplifies conditional minus unconditional noise inside the same update rule.',
              trap: 'Heavy guidance can distort samples regardless of sampler elegance.',
            }),
            lessonId: 'classifier-free-guidance',
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'reverse-update-sketch',
            label: 'Reverse update sketch',
            tooltip: tip({
              short: 'Each step maps x_t and ε̂_θ to a less noisy x_{t-1}.',
              intuition: 'Exact formula depends on parameterization and solver.',
              formula: 'x_{t-1} = f(x_t, t, \\hat{\\epsilon}_\\theta, \\eta)',
              example: 'Plug toy x_t and ε̂ into the lesson update and inspect clarity.',
              trap: 'Copy-paste DDPM code for a v-prediction checkpoint without conversion.',
            }),
          },
          {
            id: 'ddim-eta-param',
            label: 'DDIM eta parameter',
            tooltip: tip({
              short: 'eta = 0 gives deterministic DDIM; eta = 1 recovers a DDPM-like stochastic update.',
              intuition: 'eta interpolates injected variance per step.',
              formula: '\\eta \\in [0, 1]',
              example: 'Slide eta from 0 to 0.35 and watch stochasticity rise.',
              trap: 'Assuming eta=0 always matches full DDPM quality at equal step count.',
            }),
          },
          {
            id: 'subsample-timesteps',
            label: 'Subsampled timesteps',
            tooltip: tip({
              short: 'Inference uses a subset {t_i} of the training grid.',
              intuition: 'Spacing must pair with a solver designed for stride > 1.',
              code: 'timesteps = torch.linspace(T, 0, steps).long()',
              example: '8 inference steps on a 1000-step training schedule.',
              trap: 'Uniform stride without solver retuning adds integration error.',
            }),
          },
          {
            id: 'init-noise-sample',
            label: 'Initialize from noise',
            tooltip: tip({
              short: 'Sample x_T ~ N(0, I) with fixed generator seed for reproducibility.',
              intuition: 'The initial tensor is the canvas; the sampler paints structure into it.',
              code: 'x = torch.randn(shape, generator=g)',
              example: 'Same seed + DDIM → identical output.',
              trap: 'Forgetting latent scaling factor when sampling in VAE space.',
            }),
          },
          {
            id: 'solver-loop',
            label: 'Sampler loop pattern',
            tooltip: tip({
              short: 'for t in timesteps: predict → update → optionally add noise.',
              intuition: 'Frameworks wrap this in pipeline classes (Diffusers, etc.).',
              code: 'for t in scheduler.timesteps:\n  eps = model(x, t)\n  x = scheduler.step(eps, t, x).prev_sample',
              example: 'Run the snippet on a toy latent tensor.',
              trap: 'Off-by-one on final t=0 step can leave blur or NaNs.',
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'confuse-sampler-training',
            label: 'Confusing sampler with training',
            tooltip: tip({
              short: 'Training learns denoising; sampling only applies the learned field at inference.',
              intuition: 'You cannot fix a broken model by switching DDIM → DDPM alone.',
              example: 'Concrete case: training learns denoising; sampling only applies the learned field at inference.',
              trap: 'Retraining is needed when the parameterization or schedule is wrong.',
            }),
          },
          {
            id: 'too-few-steps-trap',
            label: 'Too few steps',
            tooltip: tip({
              short: 'Aggressive step reduction without distillation or solver support hurts fidelity.',
              intuition: 'Each step must carry more denoising burden.',
              example: 'Concrete case: aggressive step reduction without distillation or solver support hurts fidelity.',
              trap: 'Chasing real-time latency before validating perceptual quality.',
            }),
          },
          {
            id: 'schedule-mismatch-trap',
            label: 'Schedule mismatch',
            tooltip: tip({
              short: 'Inference betas/sigmas must match training parameterization.',
              intuition: 'The model expects noise levels it was trained on.',
              example: 'Concrete case: inference betas/sigmas must match training parameterization.',
              trap: 'Swapping cosine for linear at inference silently degrades samples.',
            }),
          },
          {
            id: 'determinism-expectation-trap',
            label: 'Expecting DDPM reproducibility',
            tooltip: tip({
              short: 'Stochastic samplers differ run-to-run even with fixed weights.',
              intuition: 'Only eta=0 paths with fixed seed repeat exactly.',
              example: 'Concrete case: stochastic samplers differ run-to-run even with fixed weights.',
              trap: 'Debugging generative bugs without fixing the seed and eta.',
            }),
          },
          {
            id: 'pixel-latent-confusion',
            label: 'Pixel vs latent confusion',
            tooltip: tip({
              short: 'Latent diffusion applies the same sampler logic in compressed space.',
              intuition: 'Decode happens after the last denoising step.',
              example: 'Concrete case: latent diffusion applies the same sampler logic in compressed space.',
              trap: 'Tuning pixel step counts when the pipeline denoises latents.',
            }),
            lessonId: 'diffusion-vae',
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'cfg-lesson',
            label: 'Classifier-free guidance',
            tooltip: tip({
              short: 'Guidance blends conditional and unconditional denoiser outputs inside each sampler step.',
              intuition: 'Sampler choice and guidance scale interact on the same trajectory.',
              example: 'Concrete case: guidance blends conditional and unconditional denoiser outputs inside each sampler step.',
              trap: 'Extreme guidance can artifact regardless of DDIM vs DDPM.',
            }),
            lessonId: 'classifier-free-guidance',
          },
          {
            id: 'flow-matching-lesson',
            label: 'Flow matching',
            tooltip: tip({
              short: 'Continuous flow models share ODE-style sampling intuition with modern diffusion solvers.',
              intuition: 'Transport from noise to data can be learned as a velocity field.',
              example: 'Concrete case: continuous flow models share ODE-style sampling intuition with modern diffusion solvers.',
              trap: 'Flow matching still needs careful integration step size.',
            }),
            lessonId: 'flow-matching',
          },
          {
            id: 'sd3-lesson',
            label: 'SD3 architecture overview',
            tooltip: tip({
              short: 'Production pipelines combine VAE latents, DiT denoisers, text encoders, and fast samplers.',
              intuition: 'End-to-end latency is sampler steps × denoiser cost × resolution.',
              example: 'Concrete case: production pipelines combine VAE latents, DiT denoisers, text encoders, and fast samplers.',
              trap: 'Optimizing one stage ignores bottlenecks elsewhere.',
            }),
            lessonId: 'sd3-overview',
          },
          {
            id: 'distillation-note',
            label: 'Sampler distillation',
            tooltip: tip({
              short: 'Student models or distilled schedules mimic many-step teachers in fewer steps.',
              intuition: 'Distillation moves the speed–quality frontier without changing the base architecture.',
              example: 'Concrete case: student models or distilled schedules mimic many-step teachers in fewer steps.',
              trap: 'Distilled samplers can fail off the teacher’s prompt or domain.',
            }),
          },
          {
            id: 'unet-dit-backbone',
            label: 'U-Net vs DiT backbones',
            tooltip: tip({
              short: 'Sampler cost per step scales with denoiser FLOPs and memory, not sampler name.',
              intuition: 'DiT global attention changes per-step price independent of DDIM vs DDPM.',
              example: 'Concrete case: sampler cost per step scales with denoiser FLOPs and memory, not sampler name.',
              trap: 'Assuming DDIM always beats DDPM on wall-clock without measuring backbone cost.',
            }),
            lessonId: 'unet-vs-dit',
          },
        ],
      },
    ],
  },
  'diffusion-vae': {
    center: {
      id: 'diffusion-vae',
      label: 'VAE for Diffusion',
      type: 'current',
      tooltip: tip({
        short: 'A diffusion VAE compresses images into a lower-dimensional latent space where denoising is cheaper—then decodes latents back to pixels.',
        intuition: 'Train an encoder/decoder bottleneck so diffusion runs on small spatial tensors while preserving perceptual detail.',
        formula: 'z = E(x),\\quad \\hat{x} = D(z),\\quad \\mathcal{L} = \\|x-\\hat{x}\\| + \\beta\\, D_{KL}(q(z|x)\\|p(z))',
        why: 'Latent diffusion (Stable Diffusion, SD3) makes high-resolution generation tractable by denoising in VAE space instead of raw RGB grids.',
        trap: 'The VAE bottleneck limits fine detail—compression artifacts cap final fidelity.',
        example: 'Encode a 512×512 image to 64×64×4 latents, diffuse there, decode to pixels.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'vae-prereq',
            label: 'Variational autoencoder',
            tooltip: tip({
              short: 'VAEs learn probabilistic encoders q(z|x) and decoders p(x|z) with a KL regularizer.',
              intuition: 'Diffusion VAEs inherit reconstruction + regularization training before diffusion uses the latent.',
              example: 'Concrete case: VAEs learn probabilistic encoders q(z|x) and decoders p(x|z) with a KL regularizer.',
              trap: 'Treating the encoder as a deterministic compression without KL can hurt latent structure.',
            }),
            lessonId: 'vae',
          },
          {
            id: 'conv-autoencoder-prereq',
            label: 'Convolutional encoder/decoder',
            tooltip: tip({
              short: 'Strided convolutions downsample; transposed convolutions upsample spatial structure.',
              intuition: 'Spatial hierarchy captures local image patterns into a compact grid.',
              example: 'Concrete case: strided convolutions downsample; transposed convolutions upsample spatial structure.',
              trap: 'Checkerboard artifacts from careless transposed-conv strides.',
            }),
          },
          {
            id: 'kl-regularizer-prereq',
            label: 'KL to standard normal',
            tooltip: tip({
              short: 'Penalize q(z|x) drifting far from N(0, I) so latents stay diffuse and interpolatable.',
              intuition: 'Diffusion assumes latents live on a scale where Gaussian noise is meaningful.',
              example: 'Concrete case: penalize q(z|x) drifting far from N(0, I) so latents stay diffuse and interpolatable.',
              trap: 'KL collapse makes latents ignore input; too weak KL yields irregular latent geometry.',
            }),
            lessonId: 'probability-distributions',
          },
          {
            id: 'reconstruction-loss-prereq',
            label: 'Reconstruction loss',
            tooltip: tip({
              short: 'L1/L2 or perceptual losses push decoded images toward inputs.',
              intuition: 'The decoder must invert enough detail for downstream diffusion to refine.',
              example: 'Concrete case: L1/L2 or perceptual losses push decoded images toward inputs.',
              trap: 'Pure pixel MSE blurs textures; perceptual losses trade off differently.',
            }),
          },
          {
            id: 'latent-scaling-prereq',
            label: 'Latent scaling factor',
            tooltip: tip({
              short: 'Multiply latents by a constant so diffusion noise levels match training assumptions.',
              intuition: 'Stable Diffusion uses ~0.18215 scaling so latent variance aligns with the scheduler.',
              example: 'Concrete case: multiply latents by a constant so diffusion noise levels match training assumptions.',
              trap: 'Forgetting scaling when moving checkpoints between frameworks.',
            }),
          },
          {
            id: 'diffusion-basics-link',
            label: 'Diffusion on latents',
            tooltip: tip({
              short: 'Forward/reverse diffusion runs on z tensors instead of RGB pixels.',
              intuition: 'Same noise prediction objective, smaller spatial cost.',
              example: 'Concrete case: forward/reverse diffusion runs on z tensors instead of RGB pixels.',
              trap: 'Pixel-space diffusion intuition does not transfer step budgets 1:1.',
            }),
            lessonId: 'diffusion-basics',
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'encode-downsample',
            label: 'Encoder downsample',
            tooltip: tip({
              short: 'Map image x to latent mean/log-var or sampled z at lower resolution.',
              intuition: '8× spatial compression is typical (512 → 64).',
              example: 'Concrete case: map image x to latent mean/log-var or sampled z at lower resolution.',
              trap: 'Confusing channel depth with spatial compression factor.',
            }),
            highlightTarget: { panel: 'animation', type: 'encoder' },
          },
          {
            id: 'latent-bottleneck',
            label: 'Latent bottleneck',
            tooltip: tip({
              short: 'Information lost in z cannot be recovered by diffusion alone.',
              intuition: 'The VAE defines the canvas resolution and detail ceiling.',
              example: 'Concrete case: information lost in z cannot be recovered by diffusion alone.',
              trap: 'Expecting crisp text or micro-detail the VAE never encoded.',
            }),
            highlightTarget: { panel: 'animation', type: 'latent' },
          },
          {
            id: 'decode-upsample',
            label: 'Decoder upsample',
            tooltip: tip({
              short: 'D(z) reconstructs RGB from denoised latents after sampling completes.',
              intuition: 'Decoder quality sets baseline sharpness before diffusion refinements.',
              example: 'Concrete case: D(z) reconstructs RGB from denoised latents after sampling completes.',
              trap: 'Training decoder only on clean latents hides mismatch from noisy denoised z.',
            }),
            highlightTarget: { panel: 'animation', type: 'decoder' },
          },
          {
            id: 'elbo-training',
            label: 'ELBO training objective',
            tooltip: tip({
              short: 'Reconstruction term + KL balances fidelity against regular latent geometry.',
              intuition: 'Beta-VAE knob trades blur vs structured latent space.',
              example: 'Concrete case: reconstruction term + KL balances fidelity against regular latent geometry.',
              trap: 'Optimizing reconstruction alone without monitoring KL collapse.',
            }),
            highlightTarget: { panel: 'animation', type: 'loss' },
          },
          {
            id: 'two-stage-pipeline',
            label: 'Two-stage pipeline',
            tooltip: tip({
              short: 'Stage 1: train/freeze VAE. Stage 2: train diffusion on encoded latents.',
              intuition: 'Decoupling keeps diffusion training memory manageable.',
              example: 'Concrete case: stage 1: train/freeze VAE. Stage 2: train diffusion on encoded latents.',
              trap: 'Fine-tuning diffusion in pixel space on a frozen mismatched VAE.',
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'compute-savings',
            label: 'Compute savings',
            tooltip: tip({
              short: 'Denoising 64×64 latents costs far less than 512×512 RGB attention or conv stacks.',
              intuition: 'Quadratic attention on patches scales with token count squared.',
              example: 'Concrete case: denoising 64×64 latents costs far less than 512×512 RGB attention or conv stacks.',
              trap: 'Ignoring channel count—4-channel latents still add memory.',
            }),
          },
          {
            id: 'perceptual-compression',
            label: 'Perceptual compression',
            tooltip: tip({
              short: 'VAE removes imperceptible detail to keep semantically important structure.',
              intuition: 'Diffusion fills in texture conditioned on latent layout.',
              example: 'Concrete case: VAE removes imperceptible detail to keep semantically important structure.',
              trap: 'Small text and fingers often hit the bottleneck first.',
            }),
          },
          {
            id: 'frozen-vae-serving',
            label: 'Frozen VAE at serving',
            tooltip: tip({
              short: 'Production pipelines decode with a fixed pretrained VAE after diffusion.',
              intuition: 'VAE weights rarely change per prompt—only latents do.',
              example: 'Concrete case: production pipelines decode with a fixed pretrained VAE after diffusion.',
              trap: 'Swapping VAE checkpoints without retraining the denoiser breaks generation.',
            }),
          },
          {
            id: 'latent-interpolation',
            label: 'Latent interpolation',
            tooltip: tip({
              short: 'Smooth walks in z space morph images when the VAE is well regularized.',
              intuition: 'KL toward Gaussian encourages meaningful linear blends.',
              example: 'Concrete case: smooth walks in z space morph images when the VAE is well regularized.',
              trap: 'Collapsed latents interpolate poorly or ignore semantics.',
            }),
          },
          {
            id: 'detail-ceiling',
            label: 'Detail ceiling',
            tooltip: tip({
              short: 'Final sharpness is bounded by what encode→decode preserves.',
              intuition: 'Diffusion cannot invent high-frequency detail the VAE discarded.',
              example: 'Concrete case: final sharpness is bounded by what encode→decode preserves.',
              trap: 'Blaming the denoiser for VAE blur or color shift.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'vae-elbo-formula',
            label: 'VAE ELBO',
            tooltip: tip({
              short: 'Maximize ELBO ≈ reconstruction − KL.',
              intuition: 'Diffusion VAE training mirrors standard VAE with image-specific losses.',
              formula: '\\mathcal{L} = \\mathbb{E}_{q(z|x)}[\\log p(x|z)] - D_{KL}(q(z|x)\\|p(z))',
              example: 'Plug small numbers into the formula for maximize ELBO ≈ reconstruction − KL.',
              trap: 'Sign errors when implementing KL analytically for diagonal Gaussians.',
            }),
          },
          {
            id: 'reparameterize-trick',
            label: 'Reparameterization trick',
            tooltip: tip({
              short: 'Sample z = μ + σ ⊙ ε with ε ~ N(0, I) for backprop through stochastic encoder.',
              intuition: 'Makes encoder gradients flow through sampling.',
              code: 'z = mu + std * torch.randn_like(std)',
              example: 'Run the snippet on a toy batch and inspect one row of outputs.',
              trap: 'Using non-reparameterized sampling breaks encoder training.',
            }),
          },
          {
            id: 'latent-scale-code',
            label: 'Latent scaling in pipeline',
            tooltip: tip({
              short: 'Apply scaling factor when encoding/decoding around diffusion.',
              intuition: 'Matches scheduler noise variance to latent magnitude.',
              code: 'latents = 0.18215 * vae.encode(x).latent_dist.sample()\nimage = vae.decode(latents / 0.18215).sample',
              example: 'Run the snippet on a toy batch and inspect one row of outputs.',
              trap: 'Double-scaling when the checkpoint already bakes in constants.',
            }),
          },
          {
            id: 'spatial-compression-ratio',
            label: 'Spatial compression ratio',
            tooltip: tip({
              short: 'f-downsample factor f reduces H×W to H/f × W/f.',
              intuition: 'Token count for DiT drops by f².',
              formula: 'N_{tokens} = (H/f)(W/f) / p^2',
              example: 'Plug small numbers into the formula for f-downsample factor f reduces H×W to H/f × W/f.',
              trap: 'Patch size p further divides token grid—count both factors.',
            }),
          },
          {
            id: 'diffusers-vae-api',
            label: 'Diffusers VAE API',
            tooltip: tip({
              short: 'AutoencoderKL encode/decode wraps conv VAE used in latent diffusion.',
              intuition: 'Same object serves training preprocessing and inference decode.',
              code: 'from diffusers import AutoencoderKL\nvae = AutoencoderKL.from_pretrained(...)',
              example: 'Run the snippet on a toy batch and inspect one row of outputs.',
              trap: 'Mixing VAE revision with denoiser revision from different releases.',
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'pixel-diffusion-trap',
            label: 'Pixel diffusion at scale',
            tooltip: tip({
              short: 'Full-resolution RGB diffusion explodes memory for large images.',
              intuition: 'Latent space is the practical default for HD generation.',
              example: 'Concrete case: full-resolution RGB diffusion explodes memory for large images.',
              trap: 'Assuming research pixel models map directly to product cost.',
            }),
          },
          {
            id: 'kl-collapse-trap',
            label: 'KL collapse',
            tooltip: tip({
              short: 'Encoder ignores input when KL weight is too high or capacity too low.',
              intuition: 'Latents become uninformative noise for diffusion.',
              example: 'Concrete case: encoder ignores input when KL weight is too high or capacity too low.',
              trap: 'Good reconstruction loss can hide posterior collapse early.',
            }),
          },
          {
            id: 'scaling-factor-trap',
            label: 'Missing scaling factor',
            tooltip: tip({
              short: 'Wrong latent magnitude makes diffusion noise levels inconsistent.',
              intuition: 'Schedulers assume a calibrated variance range.',
              example: 'Concrete case: wrong latent magnitude makes diffusion noise levels inconsistent.',
              trap: 'Copying code without the checkpoint’s documented scale constant.',
            }),
          },
          {
            id: 'vae-denoiser-mismatch',
            label: 'VAE–denoiser mismatch',
            tooltip: tip({
              short: 'Denoiser must be trained on latents from the same VAE statistics.',
              intuition: 'Swapping either half of the pair breaks sample quality.',
              example: 'Concrete case: denoiser must be trained on latents from the same VAE statistics.',
              trap: 'Community merges that mix incompatible VAE and U-Net/DiT weights.',
            }),
          },
          {
            id: 'decode-before-finish',
            label: 'Decoding unfinished latents',
            tooltip: tip({
              short: 'Decoding mid-denoising shows blurry previews—not final quality.',
              intuition: 'Decode is meant for fully denoised z at t ≈ 0.',
              example: 'Concrete case: decoding mid-denoising shows blurry previews—not final quality.',
              trap: 'Judging diffusion quality from partial latent snapshots.',
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'unet-vs-dit-lesson',
            label: 'U-Net vs DiT',
            tooltip: tip({
              short: 'Both backbones denoise VAE latents; DiT patchifies the latent grid.',
              intuition: 'Architecture choice sits after the VAE compression decision.',
              example: 'Concrete case: both backbones denoise VAE latents; DiT patchifies the latent grid.',
              trap: 'Comparing FLOPs in pixel space when both run in latent space.',
            }),
            lessonId: 'unet-vs-dit',
          },
          {
            id: 'dit-lesson',
            label: 'DiT (Diffusion Transformer)',
            tooltip: tip({
              short: 'Patch tokens from latent feature maps feed transformer denoising blocks.',
              intuition: 'VAE spatial size sets DiT token count.',
              example: 'Concrete case: patch tokens from latent feature maps feed transformer denoising blocks.',
              trap: 'Huge patch counts if VAE compression is too weak.',
            }),
            lessonId: 'dit',
          },
          {
            id: 'sd3-pipeline-lesson',
            label: 'SD3 overview',
            tooltip: tip({
              short: 'Full pipeline chains VAE, text encoders, DiT denoiser, and sampler.',
              intuition: 'VAE is the fixed image codec for the generative stack.',
              example: 'Concrete case: full pipeline chains VAE, text encoders, DiT denoiser, and sampler.',
              trap: 'Optimizing denoiser steps while ignoring VAE decode cost.',
            }),
            lessonId: 'sd3-overview',
          },
          {
            id: 'diffusion-sampling-lesson',
            label: 'Diffusion sampling',
            tooltip: tip({
              short: 'Samplers operate on latent tensors produced by the VAE encoder at inference start.',
              intuition: 'Initial x_T noise matches latent shape, not RGB shape.',
              example: 'Concrete case: samplers operate on latent tensors produced by the VAE encoder at inference start.',
              trap: 'Shape mismatch when piping RGB noise into a latent denoiser.',
            }),
            lessonId: 'diffusion-sampling',
          },
          {
            id: 'joint-attention-lesson',
            label: 'Joint attention',
            tooltip: tip({
              short: 'Multimodal models attend across text and latent visual tokens jointly.',
              intuition: 'VAE latents become the visual token grid for cross-modal fusion.',
              example: 'Concrete case: multimodal models attend across text and latent visual tokens jointly.',
              trap: 'Forgetting that visual tokens are VAE patches, not raw pixels.',
            }),
            lessonId: 'joint-attention',
          },
        ],
      },
    ],
  },
  'dit': {
    center: {
      id: 'dit',
      label: 'DiT (Diffusion Transformer)',
      type: 'current',
      tooltip: tip({
        short: 'DiT replaces U-Net convolutions with transformer blocks on latent patches—each patch is a token denoised with global self-attention and timestep conditioning.',
        intuition: 'Patchify the latent feature map, run stacked transformer blocks with AdaLN conditioning, unpatchify to predict noise or velocity.',
        formula: '\\epsilon_\\theta = \\mathrm{Unpatchify}(\\mathrm{Transformer}(\\mathrm{Patchify}(z_t), t, c))',
        why: 'DiT scales diffusion like language models scale with depth and width—powering SD3-class systems with flexible compute–quality tradeoffs.',
        trap: 'DiT is not a bigger U-Net; representation and inductive bias change from local convolutions to global attention on patches.',
        example: '64×64 latent with patch size 2 → 32×32 = 1024 tokens per layer.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'self-attention-prereq',
            label: 'Self-attention',
            tooltip: tip({
              short: 'Each patch token attends to all other patches in the latent grid.',
              intuition: 'Global context helps layout, symmetry, and long-range structure.',
              example: 'Concrete case: each patch token attends to all other patches in the latent grid.',
              trap: 'Quadratic cost in token count limits resolution and patch fineness.',
            }),
            lessonId: 'self-attention',
          },
          {
            id: 'diffusion-vae-prereq',
            label: 'VAE latents',
            tooltip: tip({
              short: 'DiT denoises low-resolution latent tensors, not full RGB images.',
              intuition: 'Patch embedding starts from VAE-compressed feature maps.',
              example: 'Concrete case: DiT denoises low-resolution latent tensors, not full RGB images.',
              trap: 'Token budgeting must use latent spatial size, not pixel resolution.',
            }),
            lessonId: 'diffusion-vae',
          },
          {
            id: 'unet-vs-dit-prereq',
            label: 'U-Net vs DiT comparison',
            tooltip: tip({
              short: 'U-Nets use local conv pyramids; DiTs use patch tokens and attention.',
              intuition: 'Architecture choice trades local inductive bias for global mixing.',
              example: 'Concrete case: U-Nets use local conv pyramids; DiTs use patch tokens and attention.',
              trap: 'Assuming DiT always wins on small data or tiny resolutions.',
            }),
            lessonId: 'unet-vs-dit',
          },
          {
            id: 'timestep-embed-prereq',
            label: 'Timestep embedding',
            tooltip: tip({
              short: 'Sinusoidal or learned embeddings tell the network the noise level t.',
              intuition: 'Same patch tokens need different transforms at high vs low noise.',
              example: 'Concrete case: sinusoidal or learned embeddings tell the network the noise level t.',
              trap: 'Weak timestep conditioning makes all layers behave similarly across t.',
            }),
          },
          {
            id: 'layer-norm-prereq',
            label: 'Layer normalization',
            tooltip: tip({
              short: 'Normalization stabilizes deep transformer training.',
              intuition: 'AdaLN modulates norm scale/shift from timestep and class embeddings.',
              example: 'Concrete case: normalization stabilizes deep transformer training.',
              trap: 'Confusing AdaLN with plain pre-norm blocks from language models.',
            }),
            lessonId: 'layer-normalization',
          },
          {
            id: 'positional-info-prereq',
            label: 'Patch position information',
            tooltip: tip({
              short: 'Fixed sin-cos or learned positional embeddings encode where each patch lives.',
              intuition: 'Attention is permutation-invariant without explicit position.',
              example: 'Concrete case: fixed sin-cos or learned positional embeddings encode where each patch lives.',
              trap: 'Removing positional cues hurts spatial coherence in generated images.',
            }),
            lessonId: 'positional-encoding',
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'patchify-step',
            label: 'Patchify latents',
            tooltip: tip({
              short: 'Split latent feature map into p×p patches flattened to tokens.',
              intuition: 'Patch size p controls token count vs local granularity.',
              example: 'Concrete case: split latent feature map into p×p patches flattened to tokens.',
              trap: 'Smaller patches mean more tokens and much higher attention cost.',
            }),
            highlightTarget: { panel: 'animation', type: 'patchify' },
          },
          {
            id: 'transformer-blocks',
            label: 'Transformer blocks',
            tooltip: tip({
              short: 'Stack of multi-head self-attention + MLP with residual connections.',
              intuition: 'Depth and width follow scaling laws similar to LLMs.',
              example: 'Concrete case: stack of multi-head self-attention + MLP with residual connections.',
              trap: 'Depth without adequate data or conditioning yields diminishing returns.',
            }),
            highlightTarget: { panel: 'animation', type: 'block' },
          },
          {
            id: 'adaln-zero',
            label: 'AdaLN-Zero conditioning',
            tooltip: tip({
              short: 'Timestep (and class/text) embeddings produce scale, shift, and gate for each sublayer.',
              intuition: 'Zero-initialized gates let early training behave like identity—stable for diffusion.',
              example: 'Concrete case: timestep (and class/text) embeddings produce scale, shift, and gate for each sublayer.',
              trap: 'Plain LayerNorm without AdaLN weakens timestep-specific behavior.',
            }),
            highlightTarget: { panel: 'animation', type: 'adaln' },
          },
          {
            id: 'cross-modal-conditioning',
            label: 'Text / label conditioning',
            tooltip: tip({
              short: 'CLIP/T5 embeddings or class tokens modulate blocks alongside timestep.',
              intuition: 'Conditioning vectors enter AdaLN or cross-attention paths.',
              example: 'Concrete case: CLIP/T5 embeddings or class tokens modulate blocks alongside timestep.',
              trap: 'Undertrained text pathways yield prompt ignoring even with strong DiT depth.',
            }),
            highlightTarget: { panel: 'animation', type: 'conditioning' },
          },
          {
            id: 'unpatchify-output',
            label: 'Unpatchify prediction',
            tooltip: tip({
              short: 'Final linear layer maps tokens back to latent noise/residual shape.',
              intuition: 'Output tensor matches denoiser training target (ε, x0, or v).',
              example: 'Concrete case: final linear layer maps tokens back to latent noise/residual shape.',
              trap: 'Channel mismatch between patch embedding and latent depth breaks training.',
            }),
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'global-layout',
            label: 'Global layout mixing',
            tooltip: tip({
              short: 'Attention lets distant patches influence each other in one layer.',
              intuition: 'Useful for symmetry, object counts, and scene-level composition.',
              example: 'Concrete case: attention lets distant patches influence each other in one layer.',
              trap: 'Very local texture may still need sufficient depth or hybrid designs.',
            }),
          },
          {
            id: 'scaling-laws',
            label: 'Scaling laws',
            tooltip: tip({
              short: 'Larger DiT-S/B/L/XL models improve FID with compute similar to LLM scaling.',
              intuition: 'Depth/width/patch size form a compute–quality knob.',
              example: 'Concrete case: larger DiT-S/B/L/XL models improve FID with compute similar to LLM scaling.',
              trap: 'Scaling parameters without scaling data or training steps plateaus.',
            }),
            lessonId: 'frontier-llm-architecture-overview',
          },
          {
            id: 'patch-token-budget',
            label: 'Patch token budget',
            tooltip: tip({
              short: 'Tokens = (H/p)(W/p); attention memory grows ~ O(tokens²).',
              intuition: 'VAE compression and patch size jointly set serving cost.',
              example: 'Concrete case: tokens = (H/p)(W/p); attention memory grows ~ O(tokens²).',
              trap: 'High-res latents with tiny patches explode memory before RGB decode.',
            }),
          },
          {
            id: 'conv-inductive-bias',
            label: 'Lost conv inductive bias',
            tooltip: tip({
              short: 'DiT must learn locality from data instead of inheriting it from kernels.',
              intuition: 'Small datasets may favor U-Net locality; big data favors flexible attention.',
              example: 'Concrete case: DiT must learn locality from data instead of inheriting it from kernels.',
              trap: 'Expecting DiT to data-efficiently match CNNs on tiny image sets.',
            }),
          },
          {
            id: 'conditioning-strength',
            label: 'Conditioning strength',
            tooltip: tip({
              short: 'Timestep + text pathways must be strong enough to steer every block.',
              intuition: 'Weak conditioning yields mode averaging or prompt drift.',
              example: 'Concrete case: timestep + text pathways must be strong enough to steer every block.',
              trap: 'Blaming sampling alone when the DiT ignores caption embeddings.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'patch-token-count',
            label: 'Patch token count',
            tooltip: tip({
              short: 'N = (H/p)(W/p) for square patches on H×W latent grid.',
              intuition: 'Halving patch size quadruples tokens.',
              formula: 'N = \\frac{H}{p}\\cdot\\frac{W}{p}',
              example: 'Plug small numbers into the formula for n = (H/p)(W/p) for square patches on H×W latent grid.',
              trap: 'Forgetting channel dimension is separate from token count.',
            }),
          },
          {
            id: 'attention-cost',
            label: 'Attention FLOPs sketch',
            tooltip: tip({
              short: 'Self-attention cost scales with N² × head_dim per layer.',
              intuition: 'DiT serving bottlenecks often look like attention memory bandwidth.',
              formula: 'O(N^2 d)',
              example: 'Plug small numbers into the formula for self-attention cost scales with N² × head_dim per layer.',
              trap: 'FlashAttention reduces memory traffic but not token-count quadratic scaling fundamentally.',
            }),
            lessonId: 'flash-attention',
          },
          {
            id: 'adaln-sketch',
            label: 'AdaLN modulation sketch',
            tooltip: tip({
              short: 'Conditioning vector c produces (γ, β) that scale and shift normalized activations.',
              intuition: 'Separate modulation per attention and MLP sublayers.',
              code: 'h = norm(x) * (1 + gamma(c)) + beta(c)',
              example: 'Run the snippet on a toy batch and inspect one row of outputs.',
              trap: 'Applying the same γ,β to all timesteps collapses denoising specialization.',
            }),
          },
          {
            id: 'dit-forward-sketch',
            label: 'DiT forward sketch',
            tooltip: tip({
              short: 'tokens = patch_embed(z_t) + pos; for block in blocks: tokens = block(tokens, t, c); out = unpatchify(tokens).',
              intuition: 'Matches ViT-style pipelines with diffusion-specific conditioning.',
              code: 'x = patch_embed(latents) + pos_embed\nfor block in dit_blocks:\n  x = block(x, t_emb, cond_emb)\neps = unpatchify(final_layer(x))',
              example: 'Run the snippet on a toy batch and inspect one row of outputs.',
              trap: 'Shape bugs at unpatchify when patch size does not divide latent dims.',
            }),
          },
          {
            id: 'cfg-in-dit',
            label: 'Guidance in DiT inference',
            tooltip: tip({
              short: 'Run conditional and unconditional forward passes; combine noise predictions.',
              intuition: 'Same CFG math as U-Net diffusion, different backbone.',
              code: 'eps = eps_uncond + w * (eps_cond - eps_uncond)',
              example: 'Run the snippet on a toy batch and inspect one row of outputs.',
              trap: 'Guidance on wrong embedding (empty string vs null token) changes behavior.',
            }),
            lessonId: 'classifier-free-guidance',
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'dit-is-unet-trap',
            label: 'DiT is just a bigger U-Net',
            tooltip: tip({
              short: 'Representation shifts from feature maps to tokens; mixing ops change.',
              intuition: 'Scaling conv width ≠ scaling attention heads.',
              example: 'Concrete case: representation shifts from feature maps to tokens; mixing ops change.',
              trap: 'Porting U-Net hyperparameters directly to DiT depth/width.',
            }),
          },
          {
            id: 'token-explosion-trap',
            label: 'Token explosion',
            tooltip: tip({
              short: 'Small patch size at high latent resolution makes attention intractable.',
              intuition: 'Coarser patches or stronger VAE compression are required.',
              example: 'Concrete case: small patch size at high latent resolution makes attention intractable.',
              trap: 'Benchmarking DiT at pixel token counts meant for latents.',
            }),
          },
          {
            id: 'weak-pos-trap',
            label: 'Missing positional structure',
            tooltip: tip({
              short: 'Without position embeddings, patch sets are unordered bags to attention.',
              intuition: 'Spatial coherence degrades.',
              example: 'Concrete case: without position embeddings, patch sets are unordered bags to attention.',
              trap: 'Assuming latent grid order alone is enough without explicit pos embed.',
            }),
          },
          {
            id: 'text-conditioning-trap',
            label: 'Underpowered text conditioning',
            tooltip: tip({
              short: 'Deep DiT with weak caption pathway still ignores prompts.',
              intuition: 'Cross-attention or strong AdaLN text vectors must be trained jointly.',
              example: 'Concrete case: deep DiT with weak caption pathway still ignores prompts.',
              trap: 'Adding T5/CLIP at inference without matching training setup.',
            }),
            lessonId: 't5-encoder',
          },
          {
            id: 'scale-without-data-trap',
            label: 'Scale without data/compute',
            tooltip: tip({
              short: 'XL DiT needs proportional training budget and diverse images.',
              intuition: 'Scaling laws assume sufficient optimization steps.',
              example: 'Concrete case: XL DiT needs proportional training budget and diverse images.',
              trap: 'Buying parameters expecting instant SD3-class quality on small fine-tunes.',
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'sd3-dit-stack',
            label: 'SD3 architecture',
            tooltip: tip({
              short: 'SD3 combines multiple text encoders with a DiT denoiser on VAE latents.',
              intuition: 'End-to-end product stacks sampler + DiT + VAE + encoders.',
              example: 'Concrete case: SD3 combines multiple text encoders with a DiT denoiser on VAE latents.',
              trap: 'Ignoring multi-encoder prompt formatting when reproducing results.',
            }),
            lessonId: 'sd3-overview',
          },
          {
            id: 'joint-attention-dit',
            label: 'Joint attention',
            tooltip: tip({
              short: 'Multimodal diffusion fuses text and image tokens in shared attention blocks.',
              intuition: 'Extends DiT conditioning beyond AdaLN-only pathways.',
              example: 'Concrete case: multimodal diffusion fuses text and image tokens in shared attention blocks.',
              trap: 'Confusing joint attention with separate cross-attention-only U-Nets.',
            }),
            lessonId: 'joint-attention',
          },
          {
            id: 'efficient-serving-dit',
            label: 'Efficient inference',
            tooltip: tip({
              short: 'DiT serving cost is dominated by attention FLOPs and KV-like activations per layer.',
              intuition: 'FlashAttention, compilation, and step reduction matter for DiT products.',
              example: 'Concrete case: DiT serving cost is dominated by attention FLOPs and KV-like activations per layer.',
              trap: 'Quantizing only the VAE while leaving full-precision giant DiT.',
            }),
            lessonId: 'efficient-inference-compression-track',
          },
          {
            id: 'flow-matching-dit',
            label: 'Flow matching hybrids',
            tooltip: tip({
              short: 'Modern systems combine DiT backbones with flow/velocity training objectives.',
              intuition: 'Architecture and objective are separable design choices.',
              example: 'Concrete case: modern systems combine DiT backbones with flow/velocity training objectives.',
              trap: 'Assuming ε-prediction is the only DiT training target.',
            }),
            lessonId: 'flow-matching',
          },
          {
            id: 'frontier-arch-dit',
            label: 'Frontier architecture map',
            tooltip: tip({
              short: 'DiT represents the transformer diffusion family in frontier architecture surveys.',
              intuition: 'Compare against U-Net, MMDiT, and hybrid conv-attention designs.',
              example: 'Concrete case: DiT represents the transformer diffusion family in frontier architecture surveys.',
              trap: 'Treating all diffusion transformers as identical beyond patch size.',
            }),
            lessonId: 'frontier-llm-architecture-overview',
          },
        ],
      },
    ],
  },
  'efficient-inference-compression-track': {
    center: {
      id: 'efficient-inference-compression-track',
      label: 'Efficient Inference & Compression',
      type: 'current',
      tooltip: tip({
        short: 'Production inference optimizes a Pareto frontier—quantization, pruning, distillation, batching, KV memory, and decoding choices trade quality against latency, throughput, and memory.',
        intuition: 'Name the bottleneck first: weights, KV cache, prefill compute, decode bandwidth, or batch scheduling—then pick the lever that moves your SLO.',
        formula: 'latency \\approx TTFT + n_{out}\\times t_{decode},\\quad memory \\approx weights + KV',
        why: 'Frontier models are unusable in products without compression and serving engineering—chat, agents, and APIs all hit this wall.',
        trap: 'Maximizing tokens/sec alone can destroy time-to-first-token, per-token latency, or output quality.',
        example: 'Compare INT4 weights + paged KV + continuous batching against FP16 static batching on the same prompt load.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'kv-cache-prereq',
            label: 'KV cache',
            tooltip: tip({
              short: 'Decode reuses stored keys/values for prior tokens instead of recomputing.',
              intuition: 'KV bytes dominate memory for long contexts and large batches.',
              example: 'Concrete case: decode reuses stored keys/values for prior tokens instead of recomputing.',
              trap: 'Ignoring KV when budgeting GPU memory for serving.',
            }),
            lessonId: 'kv-cache',
          },
          {
            id: 'flash-attention-prereq',
            label: 'FlashAttention',
            tooltip: tip({
              short: 'Tiled attention reduces HBM traffic during long prefill.',
              intuition: 'Prefill is often compute-bound; decode is often memory-bound.',
              example: 'Concrete case: tiled attention reduces HBM traffic during long prefill.',
              trap: 'FlashAttention helps math bandwidth, not bad scheduling.',
            }),
            lessonId: 'flash-attention',
          },
          {
            id: 'gqa-prereq',
            label: 'Grouped-query attention',
            tooltip: tip({
              short: 'Share KV heads across query heads to shrink cache footprint.',
              intuition: 'Architectural compression complements runtime quantization.',
              example: 'Concrete case: share KV heads across query heads to shrink cache footprint.',
              trap: 'Assuming GQA alone removes need for paging or batching.',
            }),
            lessonId: 'grouped-query-attention',
          },
          {
            id: 'token-generation-prereq',
            label: 'Token generation loop',
            tooltip: tip({
              short: 'Autoregressive decode appends one token per step using growing KV state.',
              intuition: 'Serving systems interleave many such loops on one GPU.',
              example: 'Concrete case: autoregressive decode appends one token per step using growing KV state.',
              trap: 'Treating LLM inference like one-shot image batching.',
            }),
            lessonId: 'transformer-token-generation',
          },
          {
            id: 'sampling-prereq',
            label: 'Sampling strategies',
            tooltip: tip({
              short: 'Temperature, top-p, and beam settings change output length and diversity.',
              intuition: 'Longer outputs increase decode steps and KV growth.',
              example: 'Concrete case: temperature, top-p, and beam settings change output length and diversity.',
              trap: 'Optimizing hardware while users crank temperature and max tokens.',
            }),
            lessonId: 'sampling-strategies',
          },
          {
            id: 'transformer-prereq',
            label: 'Transformer architecture',
            tooltip: tip({
              short: 'Layers, hidden size, and head count set baseline FLOPs and memory.',
              intuition: 'Compression techniques target different parts of this stack.',
              example: 'Concrete case: layers, hidden size, and head count set baseline FLOPs and memory.',
              trap: 'Applying CNN quantization intuition blindly to transformer outliers.',
            }),
            lessonId: 'transformer',
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Optimization levers',
        type: 'mechanism',
        children: [
          {
            id: 'quantization-lever',
            label: 'Quantization',
            tooltip: tip({
              short: 'Store weights (and sometimes KV) in INT8/INT4 instead of FP16/BF16.',
              intuition: 'Cuts memory and bandwidth; may need calibration or QAT for quality.',
              example: 'Concrete case: store weights (and sometimes KV) in INT8/INT4 instead of FP16/BF16.',
              trap: 'Aggressive weight-only quant hurts reasoning or long-context tasks.',
            }),
            highlightTarget: { panel: 'animation', type: 'quantization' },
          },
          {
            id: 'pruning-distill-lever',
            label: 'Pruning & distillation',
            tooltip: tip({
              short: 'Remove parameters or train smaller student models to mimic teachers.',
              intuition: 'Structural sparsity changes FLOPs; distillation transfers behavior.',
              example: 'Concrete case: remove parameters or train smaller student models to mimic teachers.',
              trap: 'Pruning without retraining often damages quality disproportionately.',
            }),
            highlightTarget: { panel: 'animation', type: 'distillation' },
          },
          {
            id: 'continuous-batching-lever',
            label: 'Continuous batching',
            tooltip: tip({
              short: 'Dynamically admit/finish sequences instead of waiting for static batch padding.',
              intuition: 'Keeps GPU utilization high when output lengths differ.',
              example: 'Concrete case: dynamically admit/finish sequences instead of waiting for static batch padding.',
              trap: 'Static batching leaves slots idle as sequences finish at different times.',
            }),
            highlightTarget: { panel: 'animation', type: 'batching' },
          },
          {
            id: 'paged-kv-lever',
            label: 'Paged KV memory',
            tooltip: tip({
              short: 'Allocate KV in fixed blocks with per-request block tables.',
              intuition: 'Reduces fragmentation for variable-length chats.',
              example: 'Concrete case: allocate KV in fixed blocks with per-request block tables.',
              trap: 'Pre-allocating max-length KV for every slot wastes memory.',
            }),
            highlightTarget: { panel: 'animation', type: 'memory' },
          },
          {
            id: 'prefill-decode-split',
            label: 'Prefill vs decode',
            tooltip: tip({
              short: 'Prefill processes the prompt in parallel; decode generates tokens serially.',
              intuition: 'TTFT tracks prefill; TPOT tracks decode—optimize separately.',
              example: 'Concrete case: prefill processes the prompt in parallel; decode generates tokens serially.',
              trap: 'One metric hides which phase is the real bottleneck.',
            }),
            highlightTarget: { panel: 'animation', type: 'latency' },
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'pareto-frontier',
            label: 'Pareto frontier',
            tooltip: tip({
              short: 'You rarely maximize quality, throughput, and latency simultaneously.',
              intuition: 'Pick SLOs first, then move the frontier with the cheapest lever.',
              example: 'Concrete case: you rarely maximize quality, throughput, and latency simultaneously.',
              trap: 'Benchmarking offline throughput without interactive TTFT targets.',
            }),
          },
          {
            id: 'bottleneck-diagnosis',
            label: 'Name the bottleneck',
            tooltip: tip({
              short: 'Profile whether weights, KV, prefill, decode, or scheduling limits you.',
              intuition: 'Quantization helps bandwidth; batching helps utilization; paging helps memory.',
              example: 'Concrete case: profile whether weights, KV, prefill, decode, or scheduling limits you.',
              trap: 'Applying the wrong optimization lever wastes engineering time.',
            }),
          },
          {
            id: 'quality-latency-trade',
            label: 'Quality vs latency',
            tooltip: tip({
              short: 'Smaller/quantized models answer faster but may lose capability.',
              intuition: 'Route hard queries to large models and easy ones to compressed models.',
              example: 'Concrete case: smaller/quantized models answer faster but may lose capability.',
              trap: 'Ship INT4 everywhere without task-specific eval regressions.',
            }),
          },
          {
            id: 'workload-shape',
            label: 'Workload shape matters',
            tooltip: tip({
              short: 'Chat, agents, batch scoring, and embeddings stress different paths.',
              intuition: 'Long prompts stress prefill; long answers stress decode KV growth.',
              example: 'Concrete case: chat, agents, batch scoring, and embeddings stress different paths.',
              trap: 'One benchmark prompt length misrepresents production traffic.',
            }),
          },
          {
            id: 'goodput-not-peak',
            label: 'Goodput under SLO',
            tooltip: tip({
              short: 'Useful throughput counts only requests meeting latency targets.',
              intuition: 'Peak tokens/sec with violated TPOT misleads capacity planning.',
              example: 'Concrete case: useful throughput counts only requests meeting latency targets.',
              trap: 'Marketing peak TFLOPs without SLO-constrained goodput.',
            }),
            lessonId: 'efficient-llm-serving',
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'latency-formula',
            label: 'Latency decomposition',
            tooltip: tip({
              short: 'End-to-end latency ≈ TTFT + n_output × t_decode.',
              intuition: 'Long prompts inflate TTFT; long completions inflate decode term.',
              formula: 'latency \\approx TTFT + n_{out}\\cdot t_{decode}',
              example: 'Plug small numbers into the formula for end-to-end latency ≈ TTFT + n_output × t_decode.',
              trap: 'Ignoring queue wait time in multi-tenant servers.',
            }),
          },
          {
            id: 'kv-memory-formula',
            label: 'KV memory sketch',
            tooltip: tip({
              short: 'KV bytes scale with layers × heads × seq_len × head_dim × batch.',
              intuition: 'GQA/MLA reduce effective KV heads; paging reduces wasted reserved bytes.',
              formula: 'M_{KV} \\propto L \\cdot H_{kv} \\cdot T \\cdot d_h',
              example: 'Plug small numbers into the formula for KV bytes scale with layers × heads × seq_len × head_dim × batch.',
              trap: 'Forgetting bytes-per-element difference FP16 vs INT8 KV cache.',
            }),
          },
          {
            id: 'quant-config-code',
            label: 'Quantization config sketch',
            tooltip: tip({
              short: 'Load model in 4-bit/8-bit with calibration or GPTQ/AWQ schemes.',
              intuition: 'Weight-only quant is simpler than full KV quant.',
              code: 'model = AutoModelForCausalLM.from_pretrained(id, load_in_4bit=True)',
              example: 'Run the snippet on a toy batch and inspect one row of outputs.',
              trap: 'Quantizing without eval on your task-specific prompts.',
            }),
          },
          {
            id: 'batch-scheduler-sketch',
            label: 'Scheduler sketch',
            tooltip: tip({
              short: 'Admission control + continuous batching + KV block allocator.',
              intuition: 'Orchestration code is as important as kernel fusion.',
              code: 'while queue:\n  batch = assemble_running_sequences()\n  logits = model.forward(batch)\n  append_sampled_tokens()',
              example: 'Run the snippet on a toy batch and inspect one row of outputs.',
              trap: 'Micro-benchmarking forward() without scheduler contention.',
            }),
          },
          {
            id: 'speculative-decode-note',
            label: 'Speculative decoding',
            tooltip: tip({
              short: 'Draft model proposes tokens; target model verifies in parallel.',
              intuition: 'Trades extra compute for lower effective decode latency.',
              example: 'Concrete case: draft model proposes tokens; target model verifies in parallel.',
              trap: 'Draft/target mismatch yields no acceptance speedup.',
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'throughput-only-trap',
            label: 'Throughput-only optimization',
            tooltip: tip({
              short: 'Huge batch throughput can mean terrible single-user TTFT/TPOT.',
              intuition: 'Interactive chat cares about tail latency, not aggregate tokens/sec alone.',
              example: 'Concrete case: huge batch throughput can mean terrible single-user TTFT/TPOT.',
              trap: 'Publishing peak throughput without p95 latency.',
            }),
          },
          {
            id: 'quant-without-eval-trap',
            label: 'Quantize without eval',
            tooltip: tip({
              short: 'INT4 may look fine on perplexity yet fail reasoning or tool use.',
              intuition: 'Task-specific regression suites are mandatory.',
              example: 'Concrete case: INT4 may look fine on perplexity yet fail reasoning or tool use.',
              trap: 'Assuming MMLU alone covers production failure modes.',
            }),
          },
          {
            id: 'ignore-kv-trap',
            label: 'Ignoring KV cache',
            tooltip: tip({
              short: 'Weight quantization does not shrink growing decode KV for long chats.',
              intuition: 'Paging and GQA/MLA target a different term in the memory sum.',
              example: 'Concrete case: weight quantization does not shrink growing decode KV for long chats.',
              trap: 'OOM during long conversations despite 4-bit weights.',
            }),
          },
          {
            id: 'static-batch-trap',
            label: 'Static batch padding waste',
            tooltip: tip({
              short: 'Waiting for max-length batches leaves GPU idle.',
              intuition: 'Continuous batching exists to fix variable-length inefficiency.',
              example: 'Concrete case: waiting for max-length batches leaves GPU idle.',
              trap: 'Copying training batching code directly to serving.',
            }),
          },
          {
            id: 'one-size-serving-trap',
            label: 'One-size serving config',
            tooltip: tip({
              short: 'Agents, coding, and search need different max tokens, tools, and model sizes.',
              intuition: 'Routing and multi-model fleets beat one compressed model for everything.',
              example: 'Concrete case: agents, coding, and search need different max tokens, tools, and model sizes.',
              trap: 'Deploying a single INT4 7B for all workloads without cascades.',
            }),
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'efficient-llm-serving-lesson',
            label: 'Efficient LLM serving',
            tooltip: tip({
              short: 'Deep dive on schedulers, SLOs, KV paging, and production tradeoffs.',
              intuition: 'Track lesson extends compression into operational serving.',
              example: 'Concrete case: deep dive on schedulers, SLOs, KV paging, and production tradeoffs.',
              trap: 'Treating serving as purely a model-size problem.',
            }),
            lessonId: 'efficient-llm-serving',
          },
          {
            id: 'fine-tuning-quant-lesson',
            label: 'Fine-tuning (QLoRA)',
            tooltip: tip({
              short: 'Quantized base weights plus low-rank adapters enable cheap adaptation.',
              intuition: 'Compression interacts with downstream fine-tune memory budgets.',
              example: 'Concrete case: quantized base weights plus low-rank adapters enable cheap adaptation.',
              trap: 'Fine-tuning in FP16 while serving INT4 without alignment testing.',
            }),
            lessonId: 'fine-tuning',
          },
          {
            id: 'mla-lesson',
            label: 'Multi-head latent attention',
            tooltip: tip({
              short: 'Architectural KV compression via latent cache representations.',
              intuition: 'Frontier systems combine algorithmic and runtime compression.',
              example: 'Concrete case: architectural KV compression via latent cache representations.',
              trap: 'Assuming MLA removes need for quant or paging entirely.',
            }),
            lessonId: 'multi-head-latent-attention',
          },
          {
            id: 'frontier-moe-lesson',
            label: 'Frontier MoE systems',
            tooltip: tip({
              short: 'Sparse routing adds load-balancing and communication bottlenecks atop compression.',
              intuition: 'Active parameters ≠ memory footprint or bandwidth.',
              example: 'Concrete case: sparse routing adds load-balancing and communication bottlenecks atop compression.',
              trap: 'Counting only active FLOPs while ignoring expert duplication in memory.',
            }),
            lessonId: 'frontier-moe-systems',
          },
          {
            id: 'model-monitoring-serving',
            label: 'Model monitoring',
            tooltip: tip({
              short: 'Track latency, cost, and quality regressions after compression deploys.',
              intuition: 'Serving changes need the same monitoring discipline as model changes.',
              example: 'Concrete case: track latency, cost, and quality regressions after compression deploys.',
              trap: 'Shipping INT4 without online quality/latency dashboards.',
            }),
            lessonId: 'model-monitoring',
          },
        ],
      },
    ],
  },
  'entropy': {
    center: {
      id: 'entropy',
      label: 'Entropy',
      type: 'current',
      tooltip: tip({
        short: 'Entropy H(X) measures average surprise in bits when sampling from a distribution—high when outcomes are spread, zero when one outcome is certain.',
        intuition: 'Rare events carry more surprise; entropy averages that surprise weighted by how often each outcome occurs.',
        formula: 'H(X)=-\\sum_x p(x)\\log p(x)',
        why: 'Entropy quantifies uncertainty for compression, decision trees, softmax outputs, and the cross-entropy loss used in classification.',
        trap: 'Entropy is not the same as model error—cross-entropy compares predictions to labels; entropy describes the distribution itself.',
        example: 'Fair coin: H=1 bit; loaded coin p=0.9 heads: H≈0.47 bits; certain outcome: H=0.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'probability-mass-prereq',
            label: 'Probability mass',
            tooltip: tip({
              short: 'Discrete probabilities p(x) are non-negative and sum to 1.',
              intuition: 'Entropy is defined only on valid distributions.',
              example: 'Three-class softmax outputs [0.7, 0.2, 0.1] sum to 1.',
              trap: 'Using unnormalized scores instead of probabilities breaks H(X).',
            }),
            lessonId: 'probability-distributions',
          },
          {
            id: 'log-surprise-prereq',
            label: 'Logarithmic surprise',
            tooltip: tip({
              short: 'Surprise of an event is −log p(x)—rare events are more surprising.',
              intuition: 'Log turns multiplicative probabilities into additive information.',
              example: 'p=0.25 → surprise = 2 bits; p=0.5 → 1 bit.',
              trap: 'log(0) is undefined—entropy needs p(x)>0 for contributing outcomes.',
            }),
          },
          {
            id: 'expected-value-entropy-prereq',
            label: 'Expected value',
            tooltip: tip({
              short: 'Entropy is the expected surprise E[−log p(X)] over draws from p.',
              intuition: 'You weight each outcome’s surprise by how often it occurs.',
              example: 'H = 0.5·1 + 0.5·1 = 1 bit for a fair coin.',
              trap: 'Confusing E[X] (average outcome) with E[−log p(X)] (average information).',
            }),
            lessonId: 'expected-value-variance',
          },
          {
            id: 'uniform-vs-peaked-prereq',
            label: 'Uniform vs peaked',
            tooltip: tip({
              short: 'Spread-out distributions have higher entropy than peaked ones.',
              intuition: 'Certainty means zero surprise; equal odds maximize uncertainty.',
              example: 'Uniform over 4 symbols → H=2 bits; one symbol certain → H=0.',
              trap: 'Assuming “more random looking” data always means higher entropy without checking p(x).',
            }),
          },
          {
            id: 'bits-units-prereq',
            label: 'Bits vs nats',
            tooltip: tip({
              short: 'Base-2 log gives bits; natural log gives nats—same concept, different units.',
              intuition: 'Shannon entropy in ML texts usually uses log₂.',
              example: 'Fair coin H=1 bit = ln(2) nats.',
              trap: 'Mixing log bases when comparing entropies across papers or libraries.',
            }),
          },
          {
            id: 'joint-events-prereq',
            label: 'Joint outcomes',
            tooltip: tip({
              short: 'Entropy can be computed on marginals or conditionals once joint structure is known.',
              intuition: 'H(X|Y) measures remaining uncertainty after observing Y.',
              example: 'Label known → conditional entropy of class drops to 0.',
              trap: 'Treating conditional and unconditional entropy as interchangeable.',
            }),
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'surprise-per-outcome',
            label: 'Surprise per outcome',
            tooltip: tip({
              short: 'Each outcome contributes −log p(x) bits of surprise to the total.',
              intuition: 'Plot surprise bars—rare outcomes tower over frequent ones.',
              formula: 'I(x)=-\\log p(x)',
              example: 'p=0.01 → 6.64 bits of surprise if log₂.',
              trap: 'Forgetting that surprise depends on probability, not on the label name.',
            }),
            highlightTarget: { panel: 'animation', type: 'surprise' },
          },
          {
            id: 'weighted-average-entropy',
            label: 'Weighted average',
            tooltip: tip({
              short: 'H(X) = Σ p(x) · I(x)—average surprise under the distribution.',
              intuition: 'Frequent mild surprises and rare huge surprises both enter the sum.',
              formula: 'H(X)=\\sum_x p(x)\\,I(x)',
              example: 'Slide probabilities and watch the entropy meter move.',
              trap: 'Averaging surprises without multiplying by p(x) is wrong.',
            }),
            highlightTarget: { panel: 'animation', type: 'entropy' },
          },
          {
            id: 'entropy-vs-max',
            label: 'Maximum at uniform',
            tooltip: tip({
              short: 'For K equally likely outcomes, H = log₂(K)—the distribution is maximally uncertain.',
              intuition: 'Peaked distributions waste less average surprise per draw.',
              formula: 'H_{max}=\\log_2 K\\text{ when }p(x)=1/K',
              example: 'Four equal classes → H=2 bits.',
              trap: 'Assuming softmax always reaches maximum entropy—it often does not.',
            }),
            highlightTarget: { panel: 'animation', type: 'entropy' },
          },
          {
            id: 'zero-entropy-certain',
            label: 'Zero entropy = certainty',
            tooltip: tip({
              short: 'When one outcome has p=1, H(X)=0—no surprise on any draw.',
              intuition: 'Deterministic distributions carry no information per sample.',
              example: 'One-hot ground truth before any model uncertainty.',
              trap: 'Confusing zero entropy with “perfect model”—labels can be certain while predictions are not.',
            }),
          },
          {
            id: 'softmax-entropy-link',
            label: 'Softmax distribution entropy',
            tooltip: tip({
              short: 'Model softmax outputs define a predicted distribution whose entropy measures prediction spread.',
              intuition: 'Flat softmax → high entropy (uncertain); peaked softmax → low entropy (confident).',
              example: '[0.34, 0.33, 0.33] vs [0.95, 0.03, 0.02] on the same logits scale.',
              trap: 'Low entropy predictions can still be wrong—confidence ≠ accuracy.',
            }),
            lessonId: 'softmax',
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'coin-intuition',
            label: 'Coin flip intuition',
            tooltip: tip({
              short: 'Fair coin maximizes binary entropy; biased coins average less surprise.',
              intuition: 'You already “know” a loaded coin’s favorite side before flipping.',
              example: 'p=0.99 heads → almost no surprise when heads appears.',
              trap: 'Calling every stochastic process “high entropy” without checking p.',
            }),
          },
          {
            id: 'compression-intuition',
            label: 'Compression intuition',
            tooltip: tip({
              short: 'Lower entropy implies fewer bits needed on average to encode outcomes.',
              intuition: 'Shannon source coding: frequent symbols get short codes.',
              example: 'English letters are not uniform—entropy guides Huffman-style coding.',
              trap: 'Entropy is a lower bound—actual codes need integer bit lengths.',
            }),
          },
          {
            id: 'uncertainty-not-error',
            label: 'Uncertainty ≠ error',
            tooltip: tip({
              short: 'Entropy measures spread of a distribution, not distance to a target label.',
              intuition: 'Cross-entropy adds a reference distribution; entropy does not.',
              example: 'Uniform wrong predictions can have high entropy and high loss.',
              trap: 'Minimizing entropy alone does not make a classifier accurate.',
            }),
            lessonId: 'cross-entropy',
          },
          {
            id: 'tree-split-intuition',
            label: 'Tree split intuition',
            tooltip: tip({
              short: 'Decision trees pick splits that reduce label uncertainty (impurity).',
              intuition: 'Entropy and Gini both reward purer child nodes after a split.',
              example: 'Split that separates classes lowers average child entropy.',
              trap: 'Low training impurity can still overfit without regularization.',
            }),
            lessonId: 'tree-ensembles',
          },
          {
            id: 'conditional-entropy-intuition',
            label: 'Remaining uncertainty',
            tooltip: tip({
              short: 'After observing side information, conditional entropy H(X|Y) can drop.',
              intuition: 'Good features explain variance in the target and shrink uncertainty.',
              example: 'Knowing digit stroke style lowers entropy over digit class.',
              trap: 'Mutual information needs both H(X) and H(X|Y)—not just raw H(X).',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'shannon-formula',
            label: 'Shannon entropy',
            tooltip: tip({
              short: 'H(X) = −Σ p(x) log p(x) with base-2 logs for bits.',
              intuition: 'Sum over support where p(x)>0.',
              formula: 'H(X)=-\\sum_x p(x)\\log_2 p(x)',
              example: 'p=[0.5,0.5] → H=1 bit.',
              trap: 'Including terms where p(x)=0—convention is 0·log 0 = 0.',
            }),
          },
          {
            id: 'binary-entropy-formula',
            label: 'Binary entropy',
            tooltip: tip({
              short: 'Bernoulli entropy H(p) = −p log p − (1−p) log(1−p).',
              intuition: 'Symmetric around p=0.5; zero at p=0 or p=1.',
              formula: 'H(p)=-p\\log_2 p-(1-p)\\log_2(1-p)',
              example: 'p=0.9 → H≈0.47 bits.',
              trap: 'Using this for multi-class without reducing to Bernoulli per class incorrectly.',
            }),
          },
          {
            id: 'numpy-entropy-code',
            label: 'NumPy entropy sketch',
            tooltip: tip({
              short: 'Compute H from a probability vector with a small epsilon for numerical stability.',
              intuition: 'Clip zeros before log to avoid NaNs.',
              code: 'p = np.clip(p, 1e-12, 1.0)\nH = -np.sum(p * np.log2(p))',
              example: 'Run on softmax output from a toy classifier.',
              trap: 'Passing logits instead of softmax probabilities.',
            }),
          },
          {
            id: 'cross-entropy-contrast-formula',
            label: 'Cross-entropy contrast',
            tooltip: tip({
              short: 'H(p,q) = −Σ p(x) log q(x) uses true p and predicted q—entropy is H(p,p).',
              intuition: 'Cross-entropy = entropy plus KL divergence when p is fixed.',
              formula: 'H(p,q)=H(p)+D_{KL}(p\\|q)',
              example: 'One-hot label p with softmax q gives standard classification loss.',
              trap: 'Calling cross-entropy “entropy” in conversation confuses two quantities.',
            }),
            lessonId: 'cross-entropy',
          },
          {
            id: 'sklearn-entropy-impurity',
            label: 'Tree impurity',
            tooltip: tip({
              short: 'sklearn uses entropy or Gini to score split impurity on class frequencies.',
              intuition: 'Child weights average impurity after the split.',
              code: 'DecisionTreeClassifier(criterion="entropy")',
              example: 'Compare entropy vs gini on the same iris split.',
              trap: 'Identical argmax splits despite different impurity numbers.',
            }),
            lessonId: 'tree-ensembles',
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'entropy-is-loss-trap',
            label: 'Entropy = loss?',
            tooltip: tip({
              short: 'Training minimizes cross-entropy against labels, not Shannon entropy of labels alone.',
              intuition: 'Loss compares two distributions; entropy describes one.',
              example: 'Certain one-hot y has H(y)=0 but CE can still be large if q is wrong.',
              trap: 'Saying “minimize entropy” when you mean “minimize cross-entropy”.',
            }),
            lessonId: 'cross-entropy',
          },
          {
            id: 'log-base-trap',
            label: 'Wrong log base',
            tooltip: tip({
              short: 'Natural log vs log₂ changes numeric values without changing ordering.',
              intuition: 'Always state units when reporting entropy numbers.',
              example: 'H=1 bit = ln(2) nats ≈ 0.693.',
              trap: 'Comparing entropies computed with different bases as if equal.',
            }),
          },
          {
            id: 'logits-not-probs-trap',
            label: 'Logits not probabilities',
            tooltip: tip({
              short: 'Plugging raw logits into −p log p is invalid.',
              intuition: 'Apply softmax first to get a proper probability mass.',
              example: 'Large logits do not sum to 1 until normalized.',
              trap: 'Numerical overflow when exponentiating huge logits without stabilization.',
            }),
            lessonId: 'softmax',
          },
          {
            id: 'confident-wrong-trap',
            label: 'Confident but wrong',
            tooltip: tip({
              short: 'Low predictive entropy means peaked softmax—not necessarily correct.',
              intuition: 'Calibration checks whether confidence matches accuracy.',
              example: '99% wrong class probability → low entropy, high loss.',
              trap: 'Using entropy alone as a uncertainty score for deployment.',
            }),
            lessonId: 'calibration',
          },
          {
            id: 'continuous-entropy-trap',
            label: 'Continuous misuse',
            tooltip: tip({
              short: 'Differential entropy for continuous X differs from discrete Shannon H.',
              intuition: 'This lesson focuses on discrete distributions common in classification.',
              example: 'Gaussian differential entropy can be negative—discrete H cannot.',
              trap: 'Applying discrete entropy formulas to unbinned continuous densities.',
            }),
            lessonId: 'probability-distributions',
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'cross-entropy-lesson',
            label: 'Cross-entropy loss',
            tooltip: tip({
              short: 'Classification loss compares predicted q to true p via −Σ p log q.',
              intuition: 'Cross-entropy rewards probability mass on the correct class.',
              example: 'Softmax + cross-entropy is the default multiclass objective.',
              trap: 'Label smoothing changes effective p—entropy of targets is no longer zero.',
            }),
            lessonId: 'cross-entropy',
          },
          {
            id: 'softmax-lesson',
            label: 'Softmax',
            tooltip: tip({
              short: 'Maps logits to a simplex—entropy of the output tracks model confidence spread.',
              intuition: 'Temperature scales logits and therefore output entropy.',
              example: 'High temperature → softer, higher-entropy predictions.',
              trap: 'Every logit shift changes all probabilities—not independent scaling.',
            }),
            lessonId: 'softmax',
          },
          {
            id: 'tree-ensembles-lesson',
            label: 'Tree ensembles',
            tooltip: tip({
              short: 'Splits reduce impurity measured by entropy or Gini; forests average many trees.',
              intuition: 'Information gain = parent entropy minus weighted child entropy.',
              example: 'Random forest reduces variance of individual entropy-greedy trees.',
              trap: 'Deep unpruned trees can reach zero training entropy yet generalize poorly.',
            }),
            lessonId: 'tree-ensembles',
          },
          {
            id: 'mle-entropy-lesson',
            label: 'Maximum likelihood',
            tooltip: tip({
              short: 'Minimizing cross-entropy on one-hot labels equals maximizing log-likelihood.',
              intuition: 'Likelihood and information-theoretic views align for classification.',
              example: 'Categorical NLL in PyTorch implements the same objective.',
              trap: 'MLE does not automatically yield calibrated probabilities.',
            }),
            lessonId: 'maximum-likelihood-estimation',
          },
          {
            id: 'calibration-entropy-lesson',
            label: 'Calibration',
            tooltip: tip({
              short: 'Low entropy predictions should match empirical accuracy at that confidence.',
              intuition: 'Temperature scaling adjusts softmax sharpness post-training.',
              example: 'Reliability diagrams reveal overconfident low-entropy outputs.',
              trap: 'Sharp softmax (low entropy) is not trustworthy without calibration checks.',
            }),
            lessonId: 'calibration',
          },
        ],
      },
    ],
  },
  'expected-value-variance': {
    center: {
      id: 'expected-value-variance',
      label: 'Expected Value & Variance',
      type: 'current',
      tooltip: tip({
        short: 'Expected value E[X] is the probability-weighted average outcome; variance Var(X) measures spread around that mean—same average can hide very different risk.',
        intuition: 'Mean tells you where the distribution centers; variance tells you how wildly outcomes swing around it.',
        formula: 'E[X]=\\sum_x x\\,p(x),\\quad \\operatorname{Var}(X)=E[(X-\\mu)^2]',
        why: 'These moments underpin PCA (variance directions), RL value functions (expected return), bias-variance analysis, and honest risk comparisons.',
        trap: 'Same mean does not mean same risk—two bets with E[X]=0 can have wildly different variance.',
        example: 'Coin +$1/−$1 vs lottery ticket: both may center near zero, but variance differs enormously.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'discrete-rv-prereq',
            label: 'Discrete random variable',
            tooltip: tip({
              short: 'X takes countable outcomes x with probabilities p(x).',
              intuition: 'Expectation sums over the support weighted by chance.',
              example: 'Dice face 1–6 each with p=1/6.',
              trap: 'Using sample averages before defining population expectation.',
            }),
            lessonId: 'probability-distributions',
          },
          {
            id: 'probability-weighted-sum-prereq',
            label: 'Weighted sum',
            tooltip: tip({
              short: 'E[X] multiplies each outcome by its probability and adds.',
              intuition: 'Rare large values still contribute if p(x) is nonzero.',
              example: 'E[die]=3.5 even though 3.5 is not a rollable face.',
              trap: 'Expecting E[X] to always be an observable outcome.',
            }),
          },
          {
            id: 'deviation-from-mean-prereq',
            label: 'Deviation from mean',
            tooltip: tip({
              short: 'Variance squares (X − μ) so positive and negative deviations both count.',
              intuition: 'Squaring penalizes large swings more than small ones.',
              example: '(+3)² and (−3)² both add 9 to spread.',
              trap: 'Using |X−μ| without squaring gives mean absolute deviation—a different measure.',
            }),
          },
          {
            id: 'linearity-expectation-prereq',
            label: 'Linearity of expectation',
            tooltip: tip({
              short: 'E[aX + b] = aE[X] + b even when X and Y are dependent.',
              intuition: 'Expectation distributes over sums—variance does not without extra terms.',
              example: 'E[X+Y]=E[X]+E[Y] always.',
              trap: 'Assuming Var(X+Y)=Var(X)+Var(Y) when X and Y are correlated.',
            }),
          },
          {
            id: 'sample-vs-population-prereq',
            label: 'Sample vs population',
            tooltip: tip({
              short: 'Sample mean and variance estimate population moments with noise.',
              intuition: 'Small samples mislead—law of large numbers needs many draws.',
              example: 'Ten coin flips may not average exactly 0.5.',
              trap: 'Treating one dataset’s mean as the true E[X] without uncertainty.',
            }),
            lessonId: 'sampling-confidence-intervals',
          },
          {
            id: 'matrix-view-prereq',
            label: 'Vectors of features',
            tooltip: tip({
              short: 'Multivariate data has per-feature means and covariances across columns.',
              intuition: 'PCA and scaling build on variance structure in feature space.',
              example: 'Two features with very different scales affect covariance geometry.',
              trap: 'Computing variance on unscaled mixed units without context.',
            }),
            lessonId: 'matrix-multiplication',
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'compute-expectation',
            label: 'Compute E[X]',
            tooltip: tip({
              short: 'Multiply each outcome by p(x) and sum to get the center of mass.',
              intuition: 'Slide probabilities and watch the balance point move.',
              formula: 'E[X]=\\sum_x x\\,p(x)',
              example: 'Weighted average of payoff table entries.',
              trap: 'Forgetting weights must sum to 1.',
            }),
            highlightTarget: { panel: 'animation', type: 'expected' },
          },
          {
            id: 'compute-variance',
            label: 'Compute Var(X)',
            tooltip: tip({
              short: 'Square deviations from μ and take the probability-weighted average.',
              intuition: 'Spread grows when probability mass sits far from the mean.',
              formula: '\\operatorname{Var}(X)=E[(X-\\mu)^2]',
              example: 'Two distributions with μ=0 but different tail weights.',
              trap: 'Using sample n−1 vs n denominator inconsistently.',
            }),
            highlightTarget: { panel: 'animation', type: 'variance' },
          },
          {
            id: 'std-deviation',
            label: 'Standard deviation',
            tooltip: tip({
              short: 'σ = √Var(X) returns spread to the same units as X.',
              intuition: 'Easier to interpret than squared units.',
              formula: '\\sigma=\\sqrt{\\operatorname{Var}(X)}',
              example: 'Returns in dollars with σ also in dollars.',
              trap: 'Var and σ are not interchangeable numerically—σ is the square root.',
            }),
            highlightTarget: { panel: 'animation', type: 'variance' },
          },
          {
            id: 'same-mean-different-risk',
            label: 'Same mean, different risk',
            tooltip: tip({
              short: 'Compare two gambles with equal E[X] but different variance to see risk preference.',
              intuition: 'Risk-averse agents prefer lower variance at the same expected return.',
              example: 'Steady small swings vs rare huge jackpots.',
              trap: 'Choosing purely by highest E[X] ignores downside volatility.',
            }),
            highlightTarget: { panel: 'animation', type: 'decision' },
          },
          {
            id: 'law-of-total-expectation',
            label: 'Tower property sketch',
            tooltip: tip({
              short: 'E[X] = E[E[X|Y]]—average conditional expectations recover the marginal mean.',
              intuition: 'Used in RL backups: expected next-state value averages over transitions.',
              example: 'Bellman expectation backup is a conditional expectation over actions and next states.',
              trap: 'Applying tower rule without a well-defined conditional distribution.',
            }),
            lessonId: 'value-iteration',
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'balance-point-intuition',
            label: 'Balance point',
            tooltip: tip({
              short: 'E[X] is where a probability-weighted seesaw balances.',
              intuition: 'Outliers with small p still tilt the balance if x is extreme.',
              example: 'Lottery tail pulls mean up despite low probability.',
              trap: 'Median and mean diverge under skew—report both when skewed.',
            }),
          },
          {
            id: 'spread-intuition',
            label: 'Spread as risk',
            tooltip: tip({
              short: 'Variance captures how wide outcomes scatter—not just the typical value.',
              intuition: 'Finance and RL both care about downside volatility, not only average return.',
              example: 'Two policies same average reward—one wildly swings episode returns.',
              trap: 'Low sample variance on small data can hide true risk.',
            }),
          },
          {
            id: 'pca-variance-intuition',
            label: 'PCA variance directions',
            tooltip: tip({
              short: 'PCA picks axes where projected data retains maximum variance.',
              intuition: 'Leading eigenvectors of covariance point along spread-heavy directions.',
              example: 'Flatten 2D cloud onto axis with widest spread first.',
              trap: 'PCA maximizes input variance, not label separability.',
            }),
            lessonId: 'pca',
          },
          {
            id: 'bias-variance-intuition',
            label: 'Bias vs variance',
            tooltip: tip({
              short: 'Model error decomposes into systematic bias and sensitivity to sample noise.',
              intuition: 'High variance: predictions swing across train sets; high bias: always systematically wrong.',
              example: 'Deep unregularized fit: low bias, high variance on small data.',
              trap: 'More complex models always reduce bias but can inflate variance.',
            }),
            lessonId: 'bias-variance-tradeoff',
          },
          {
            id: 'rl-return-intuition',
            label: 'Expected return in RL',
            tooltip: tip({
              short: 'Value functions estimate expected cumulative reward from a state or state-action pair.',
              intuition: 'Bellman backups propagate expected future rewards backward.',
              example: 'V(s) averages over stochastic transitions and policies.',
              trap: 'Sample returns are noisy—value iteration averages over the model.',
            }),
            lessonId: 'value-iteration',
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'expectation-formula',
            label: 'Expectation formula',
            tooltip: tip({
              short: 'E[X] = Σ x p(x) for discrete X.',
              intuition: 'Weights are probabilities, not arbitrary scores.',
              formula: 'E[X]=\\sum_x x\\,p(x)',
              example: 'Fair die: E= (1+2+3+4+5+6)/6 = 3.5.',
              trap: 'Using frequencies without normalizing to probabilities.',
            }),
          },
          {
            id: 'variance-formula',
            label: 'Variance formula',
            tooltip: tip({
              short: 'Var(X) = E[X²] − (E[X])² is an equivalent computational form.',
              intuition: 'Sometimes faster than squaring deviations explicitly.',
              formula: '\\operatorname{Var}(X)=E[X^2]-(E[X])^2',
              example: 'Compute both forms on a toy pmf and compare.',
              trap: 'Catastrophic cancellation with floating point on nearly equal terms.',
            }),
          },
          {
            id: 'numpy-moments-code',
            label: 'NumPy moments',
            tooltip: tip({
              short: 'np.average and np.var with appropriate weights and ddof.',
              intuition: 'ddof=1 gives unbiased sample variance; ddof=0 population-style.',
              code: 'mu = np.average(x, weights=p)\nvar = np.average((x - mu)**2, weights=p)',
              example: 'Match animation pmf with weighted average calls.',
              trap: 'np.var default ddof=0 vs pandas ddof=1 confusion.',
            }),
          },
          {
            id: 'covariance-pca-link',
            label: 'Covariance matrix',
            tooltip: tip({
              short: 'Center data, compute cov, eigendecompose for PCA axes.',
              intuition: 'Diagonal entries are per-feature variances; off-diagonal co-movement.',
              formula: '\\Sigma=\\frac{1}{n}X_c^\\top X_c',
              example: 'First eigenvector = direction of max variance.',
              trap: 'PCA on uncentered data mixes means into directions.',
            }),
            lessonId: 'pca',
          },
          {
            id: 'bellman-expectation-code',
            label: 'Bellman expectation backup',
            tooltip: tip({
              short: 'V(s) ← Σ_s′ P(s′|s)[R + γ V(s′)] averages next-state values.',
              intuition: 'Expectation over stochastic transitions defines the backup target.',
              code: 'V[s] = sum(p * (r + gamma * V[s_next]) for p, r, s_next in transitions)',
              example: 'One sweep updates every state toward expected return.',
              trap: 'Using max instead of sum when environment is stochastic.',
            }),
            lessonId: 'value-iteration',
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'mean-is-typical-trap',
            label: 'Mean = typical?',
            tooltip: tip({
              short: 'Skewed distributions can have E[X] far from the most likely outcome.',
              intuition: 'Median and mode answer different “typical” questions.',
              example: 'Power-law wealth: mean exceeds what most people see.',
              trap: 'Reporting only mean for heavy-tailed metrics.',
            }),
          },
          {
            id: 'variance-units-trap',
            label: 'Squared units',
            tooltip: tip({
              short: 'Variance is in squared units—compare σ for interpretability.',
              intuition: 'Var(income) in dollars² is awkward to explain.',
              example: 'σ_income in dollars is clearer than Var in dollars².',
              trap: 'Adding variances of variables with different units meaninglessly.',
            }),
          },
          {
            id: 'ignore-correlation-trap',
            label: 'Independent variance sum',
            tooltip: tip({
              short: 'Var(X+Y)=Var(X)+Var(Y) only when X and Y are uncorrelated.',
              intuition: 'Covariance term 2Cov(X,Y) matters for portfolios and features.',
              example: 'Two perfectly correlated features double effective variance impact.',
              trap: 'Assuming feature variances add cleanly in PCA without centering/scaling.',
            }),
          },
          {
            id: 'sample-size-trap',
            label: 'Tiny sample variance',
            tooltip: tip({
              short: 'Sample variance on n=5 can wildly underestimate or overestimate risk.',
              intuition: 'Confidence intervals widen for small n.',
              example: 'Five RL episodes may not represent policy variance.',
              trap: 'Declaring winner policy from handful of noisy returns.',
            }),
            lessonId: 'sampling-confidence-intervals',
          },
          {
            id: 'optimize-mean-only-trap',
            label: 'Optimize mean only',
            tooltip: tip({
              short: 'Maximizing average reward alone ignores catastrophic tail outcomes.',
              intuition: 'Risk-sensitive RL and finance add variance or CVaR penalties.',
              example: 'Policy with great mean but occasional huge negative spikes.',
              trap: 'Benchmarking RL with mean return only on stochastic domains.',
            }),
            lessonId: 'reinforcement-learning',
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'value-iteration-lesson',
            label: 'Value iteration',
            tooltip: tip({
              short: 'Iterative Bellman backups compute expected returns under a policy.',
              intuition: 'Expectation over transitions is the computational core.',
              example: 'Sweep until V converges to expected discounted reward.',
              trap: 'Confusing planning with Q-learning from samples only.',
            }),
            lessonId: 'value-iteration',
          },
          {
            id: 'pca-lesson',
            label: 'PCA',
            tooltip: tip({
              short: 'Eigenvectors of covariance capture directions of maximal variance.',
              intuition: 'Explained variance ratio guides component count.',
              example: 'Keep components covering 95% cumulative variance.',
              trap: 'PCA components are linear—nonlinear manifolds need other methods.',
            }),
            lessonId: 'pca',
          },
          {
            id: 'bias-variance-lesson',
            label: 'Bias-variance tradeoff',
            tooltip: tip({
              short: 'Decompose generalization error into bias², variance, and irreducible noise.',
              intuition: 'Regularization increases bias to cut variance.',
              example: 'Ridge shrinks coefficients—smoother fit, less sample sensitivity.',
              trap: 'Complex models can interpolate train yet fail from variance.',
            }),
            lessonId: 'bias-variance-tradeoff',
          },
          {
            id: 'q-learning-lesson',
            label: 'Q-learning',
            tooltip: tip({
              short: 'Sample backups estimate expected return for state-action pairs from experience.',
              intuition: 'Expectation replaced by stochastic TD targets over time.',
              example: 'Q(s,a) ← Q + α[r + γ max Q(s′) − Q].',
              trap: 'Max operator introduces overestimation bias—not pure expectation.',
            }),
            lessonId: 'q-learning',
          },
          {
            id: 'feature-scaling-ev-lesson',
            label: 'Feature scaling',
            tooltip: tip({
              short: 'Standardization uses train mean and std—moments estimated safely from training rows.',
              intuition: 'Variance on raw mixed units distorts distance and gradients.',
              example: 'z = (x − μ_train) / σ_train per feature.',
              trap: 'Fitting scaler on all data leaks validation moments.',
            }),
            lessonId: 'feature-scaling-preprocessing',
          },
        ],
      },
    ],
  },
  'fasttext': {
    center: {
      id: 'fasttext',
      label: 'FastText',
      type: 'current',
      tooltip: tip({
        short: 'FastText represents words as sums of subword n-gram vectors—sharing pieces across words handles morphology and out-of-vocabulary tokens better than whole-word Word2Vec.',
        intuition: 'Break “apple” into character n-grams, learn a vector per n-gram, and add them up to get v(word).',
        formula: 'v(w)=\\sum_{g\\in G_w} v_g',
        why: 'Rare words, typos, and morphological variants borrow strength from shared substrings—critical for open-vocabulary text and low-resource languages.',
        trap: 'FastText is not magic—very unrelated strings sharing n-grams can get spuriously similar vectors.',
        example: '“where” and “there” share “here” n-grams; unseen “apples” composes from “apple” + “s” pieces.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'word2vec-prereq',
            label: 'Word2Vec basics',
            tooltip: tip({
              short: 'Word2Vec learns one dense vector per whole word from context prediction.',
              intuition: 'FastText extends the same objective with subword buckets.',
              example: 'Skip-gram predicts neighbors from center word embeddings.',
              trap: 'Whole-word models assign UNK to unseen tokens at inference.',
            }),
            lessonId: 'word2vec',
          },
          {
            id: 'embeddings-prereq',
            label: 'Embeddings',
            tooltip: tip({
              short: 'Dense vectors encode semantic similarity in geometric space.',
              intuition: 'Similar words/clusters end up with nearby vectors after training.',
              example: 'Cosine similarity ranks neighbors in embedding space.',
              trap: 'Static embeddings miss context—one vector per word type.',
            }),
            lessonId: 'embeddings',
          },
          {
            id: 'tokenization-prereq',
            label: 'Tokenization',
            tooltip: tip({
              short: 'Text splits into tokens before vector lookup or subword decomposition.',
              intuition: 'Whitespace and punctuation rules affect which n-grams appear.',
              example: '“don’t” may split differently across tokenizers.',
              trap: 'Train and serve tokenization mismatch breaks subword tables.',
            }),
            lessonId: 'tokenization',
          },
          {
            id: 'ngram-language-prereq',
            label: 'Character n-grams',
            tooltip: tip({
              short: 'Sliding windows of n characters (with boundary markers) form subword units.',
              intuition: 'Prefixes, suffixes, and stems recur across vocabulary.',
              example: '<where> decomposes into overlapping 3–6 character grams.',
              trap: 'Very short n alone miss word-level semantics; very long n overfit rare strings.',
            }),
          },
          {
            id: 'bag-of-ngrams-prereq',
            label: 'Bag-of-n-grams view',
            tooltip: tip({
              short: 'Word vector is the sum (or average) of its n-gram vectors—order within word partly lost.',
              intuition: 'Composition is additive, not recurrent.',
              example: 'v(apple) ≈ v(app) + v(ppl) + v(ple) + …',
              trap: 'Anagrams with same multiset of n-grams get identical vectors.',
            }),
          },
          {
            id: 'cosine-similarity-prereq',
            label: 'Cosine similarity',
            tooltip: tip({
              short: 'Compare embedding directions after summing n-gram vectors.',
              intuition: 'Magnitude effects partially cancel when using cosine on summed vectors.',
              example: 'Nearest neighbors in FastText space for word analogy queries.',
              trap: 'Summing many n-grams changes norm—cosine vs dot product rankings differ.',
            }),
            lessonId: 'cosine-similarity',
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'subword-decomposition',
            label: 'Subword decomposition',
            tooltip: tip({
              short: 'Each word maps to a set G_w of character n-grams bounded by min and max n.',
              intuition: 'Boundary symbols mark word edges for prefix/suffix capture.',
              example: '“cat” with n=3 might include <ca, cat, at>.',
              trap: 'Wrong min/max n settings truncate morphological cues.',
            }),
            highlightTarget: { panel: 'animation', type: 'subword' },
          },
          {
            id: 'ngram-vector-sum',
            label: 'Sum n-gram vectors',
            tooltip: tip({
              short: 'v(w) = Σ_{g∈G_w} v_g—word embedding is additive composition.',
              intuition: 'Shared n-grams tie related words and handle OOV by reusing pieces.',
              formula: 'v(w)=\\sum_{g\\in G_w} v_g',
              example: 'Rare “unhappiness” builds from “un”, “happi”, “ness” pieces if seen elsewhere.',
              trap: 'Identical n-gram multisets collapse distinct spellings.',
            }),
            highlightTarget: { panel: 'animation', type: 'ngram' },
          },
          {
            id: 'oov-inference',
            label: 'OOV inference',
            tooltip: tip({
              short: 'Unseen words still get vectors from their character n-grams at test time.',
              intuition: 'No UNK bucket required if subwords cover the string.',
              example: 'Misspelled “applle” shares most n-grams with “apple”.',
              trap: 'Random strings with accidental n-gram overlap look falsely related.',
            }),
            highlightTarget: { panel: 'animation', type: 'oov' },
          },
          {
            id: 'word2vec-objective-shared',
            label: 'Shared Word2Vec objective',
            tooltip: tip({
              short: 'Still optimize skip-gram/CBOW with negative sampling—only representation changes.',
              intuition: 'Context prediction pulls composed vectors toward neighbors.',
              example: 'Center “running” updates all n-gram vectors in its sum.',
              trap: 'Assuming FastText trains faster per epoch—vocabulary bucket count grows.',
            }),
            lessonId: 'word2vec',
          },
          {
            id: 'word-vs-subword-lookup',
            label: 'Whole word + subwords',
            tooltip: tip({
              short: 'Implementation often adds full word vector plus sum of n-gram vectors.',
              intuition: 'Whole-word bucket helps frequent tokens; subwords help rare/OOV.',
              example: 'v(w) = v_word + Σ v_ngram in official FastText.',
              trap: 'Comparing to Word2Vec without noting the extra whole-word term.',
            }),
            highlightTarget: { panel: 'animation', type: 'comparison' },
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'morphology-intuition',
            label: 'Morphology sharing',
            tooltip: tip({
              short: 'Prefixes and suffixes reuse vectors across word family members.',
              intuition: '“run”, “running”, “runner” share stem n-grams.',
              example: 'Agglutinative languages benefit strongly from subwords.',
              trap: 'English irregular forms may share misleading n-grams.',
            }),
          },
          {
            id: 'oov-not-unk-intuition',
            label: 'OOV ≠ UNK',
            tooltip: tip({
              short: 'Word2Vec maps unknown words to a single UNK vector; FastText composes fresh vectors.',
              intuition: 'Typos and new product names keep partial signal.',
              example: '“iphone15” from subwords even if whole token unseen in training.',
              trap: 'Completely novel character sequences still need overlapping n-grams to help.',
            }),
          },
          {
            id: 'spelling-robustness-intuition',
            label: 'Spelling robustness',
            tooltip: tip({
              short: 'Small edit distances often preserve many n-grams.',
              intuition: 'Useful for noisy user text and social media.',
              example: '“colour” vs “color” share overlapping character windows.',
              trap: 'Adversarial strings can exploit shared n-grams to spoof similarity.',
            }),
          },
          {
            id: 'static-context-intuition',
            label: 'Still static',
            tooltip: tip({
              short: 'One vector per word type—context polysemy remains unresolved.',
              intuition: 'Transformers later replace static lookup with contextual states.',
              example: '“bank” river vs finance shares one FastText vector.',
              trap: 'Expecting BERT-level sense disambiguation from FastText alone.',
            }),
            lessonId: 'bert',
          },
          {
            id: 'vocab-size-intuition',
            label: 'Bucket tradeoff',
            tooltip: tip({
              short: 'More n-gram buckets increase memory and training cost.',
              intuition: 'Hashing tricks compress bucket tables at collision cost.',
              example: 'minn=3, maxn=6 is a common starting range.',
              trap: 'Huge n ranges on tiny corpora overfit character noise.',
            }),
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'composition-formula',
            label: 'Vector composition',
            tooltip: tip({
              short: 'v(w) = Σ_{g∈G_w} v_g with optional whole-word term.',
              intuition: 'Training updates every n-gram vector appearing in the window.',
              formula: 'v(w)=v_{word}+\\sum_{g\\in G_w} v_g',
              example: 'Compare vector norm before/after summing many n-grams.',
              trap: 'Forgetting the whole-word term when reproducing paper results.',
            }),
          },
          {
            id: 'ngram-set-definition',
            label: 'N-gram set G_w',
            tooltip: tip({
              short: 'All character n-grams for n ∈ [minn, maxn] with boundary markers.',
              intuition: 'Boundaries distinguish prefix “un” from internal “un”.',
              formula: 'G_w=\\bigcup_{n=minn}^{maxn}\\text{ngrams}_n(w)',
              example: 'minn=3, maxn=5 on “cat” yields bounded gram list.',
              trap: 'Off-by-one n range drastically changes coverage.',
            }),
          },
          {
            id: 'fasttext-train-code',
            label: 'FastText training sketch',
            tooltip: tip({
              short: 'Official library exposes subword min/max and skip-gram/CBOW modes.',
              intuition: 'Same CLI as Word2Vec with subword flags enabled.',
              code: 'import fasttext\nmodel = fasttext.train_unsupervised(\n  "corpus.txt", model="skipgram", dim=100, minn=3, maxn=6\n)',
              example: 'Get vector for OOV word with model.get_word_vector("newword").',
              trap: 'Preprocessing must match training tokenization when calling get_word_vector.',
            }),
          },
          {
            id: 'word2vec-contrast-code',
            label: 'Word2Vec contrast',
            tooltip: tip({
              short: 'Standard Word2Vec has no subword buckets—OOV maps to UNK.',
              intuition: 'Side-by-side training shows FastText wins on rare-word similarity tasks.',
              code: '# Word2Vec: vector only if w in vocab\n# FastText: always compose from n-grams',
              example: 'Evaluate rare-word analogy subset separately from frequent words.',
              trap: 'Fair comparison requires matched dim, window, and corpus.',
            }),
            lessonId: 'word2vec',
          },
          {
            id: 'cosine-neighbors-code',
            label: 'Nearest neighbors',
            tooltip: tip({
              short: 'Rank vocabulary by cosine similarity to query vector from composed v(w).',
              intuition: 'OOV queries still participate if subwords exist.',
              code: 'v = model.get_word_vector(query)\n# rank cos(v, model.get_word_vector(w)) for w in vocab',
              example: 'Misspelled query still retrieves intended neighbor.',
              trap: 'Including training UNK token in neighbor lists skews results.',
            }),
            lessonId: 'cosine-similarity',
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'fasttext-equals-bert-trap',
            label: 'FastText = contextual?',
            tooltip: tip({
              short: 'FastText remains a static embedding method—no per-sentence context stack.',
              intuition: 'Polysemy and syntax need transformer representations.',
              example: 'Same “bank” vector in river and finance sentences.',
              trap: 'Replacing BERT with FastText for NLU tasks expecting context.',
            }),
            lessonId: 'bert',
          },
          {
            id: 'ngram-collision-trap',
            label: 'N-gram collisions',
            tooltip: tip({
              short: 'Hash bucket collisions merge unrelated n-grams in large-vocab settings.',
              intuition: 'Collisions add noise especially on small training data.',
              example: 'Two rare grams share bucket id → entangled vectors.',
              trap: 'Assuming collision-free subword tables at billion-token scale.',
            }),
          },
          {
            id: 'anagram-trap',
            label: 'Anagram equivalence',
            tooltip: tip({
              short: 'Pure sum-of-n-grams gives identical vectors for anagrams.',
              intuition: 'Character multiset determines composition—order within word lost.',
              example: '“listen” vs “silent” share n-gram multiset in char n-gram models.',
              trap: 'Using FastText vectors for tasks needing letter order.',
            }),
          },
          {
            id: 'train-serve-token-trap',
            label: 'Train/serve token mismatch',
            tooltip: tip({
              short: 'Different lowercasing or punctuation rules change n-gram sets at inference.',
              intuition: 'Subword lookup is brittle to preprocessing drift.',
              example: 'Train lowercased, serve cased → shifted n-grams and neighbors.',
              trap: 'Silently changing tokenizer between train and production index.',
            }),
            lessonId: 'tokenization',
          },
          {
            id: 'compare-unfair-word2vec-trap',
            label: 'Unfair Word2Vec comparison',
            tooltip: tip({
              short: 'Different dim, window, negatives, or corpus size invalidates benchmarks.',
              intuition: 'FastText wins most on rare/OOV slices—not always on frequent words.',
              example: 'Match dim=100, ws=5, minCount on same corpus before claiming superiority.',
              trap: 'Publishing full-vocab analogy scores hiding OOV-only gains.',
            }),
            lessonId: 'word2vec',
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'embeddings-retrieval-lesson',
            label: 'Embedding retrieval',
            tooltip: tip({
              short: 'FastText vectors power lexical search, dedup, and coarse semantic retrieval.',
              intuition: 'Subwords help short-text and noisy query matching.',
              example: 'Product name search with typo tolerance via subword overlap.',
              trap: 'Mixing FastText query vectors with BERT document vectors.',
            }),
            lessonId: 'embeddings',
          },
          {
            id: 'transformer-token-lesson',
            label: 'Transformer tokenization',
            tooltip: tip({
              short: 'Modern NLP uses subword BPE/SentencePiece—FastText foreshadows piece sharing.',
              intuition: 'Subword culture continues in LLM vocabularies.',
              example: 'GPT tokens are BPE merges, not whole words.',
              trap: 'Assuming WordPiece and char n-grams are identical mechanisms.',
            }),
            lessonId: 'tokenization',
          },
          {
            id: 'self-attention-lesson',
            label: 'Self-attention',
            tooltip: tip({
              short: 'Contextual models supersede static sums but inherit subword vocabulary ideas.',
              intuition: 'Embedding table rows align with subword tokens in transformers.',
              example: 'Same “ing” piece appears across verb forms in BPE vocab.',
              trap: 'Skipping static embedding foundations before attention stacks.',
            }),
            lessonId: 'self-attention',
          },
          {
            id: 'negative-sampling-lesson',
            label: 'Negative sampling',
            tooltip: tip({
              short: 'Efficient training signal for skip-gram—shared with Word2Vec training loops.',
              intuition: 'Noise contrastive updates pull true contexts together.',
              example: 'Sampled negatives approximate softmax denominator.',
              trap: 'Too few negatives underfit; too many slow training.',
            }),
            lessonId: 'word2vec',
          },
          {
            id: 'classification-features-lesson',
            label: 'Text classification features',
            tooltip: tip({
              short: 'Averaged FastText document vectors feed linear classifiers on small data.',
              intuition: 'Strong baseline before fine-tuning large language models.',
              example: 'Mean of word vectors for tweet sentiment baseline.',
              trap: 'Averaging ignores word order—bag-of-words ceiling applies.',
            }),
            lessonId: 'logistic-regression',
          },
        ],
      },
    ],
  },
  'feature-scaling-preprocessing': {
    center: {
      id: 'feature-scaling-preprocessing',
      label: 'Feature Scaling & Preprocessing',
      type: 'current',
      tooltip: tip({
        short: 'Feature scaling rescales inputs so distance-based models and optimizers treat dimensions fairly—fit statistics on training data only, then transform validation and test.',
        intuition: 'Without scaling, one large-magnitude feature dominates distance and gradient steps; with leakage-safe fitting, evaluation stays honest.',
        formula: 'z=\\frac{x-\\mu_{train}}{\\sigma_{train}},\\quad x_{mm}=\\frac{x-min_{train}}{max_{train}-min_{train}}',
        why: 'Scaling stabilizes kNN, SVM, neural nets, and regularized linear models while preprocessing pipelines must respect split boundaries.',
        trap: 'Fitting a scaler on all rows before splitting leaks validation statistics into training.',
        example: 'Income in dollars and age in years on raw axes make kNN vote almost entirely on income until both features share comparable scale.',
      }),
    },
    branches: [
      {
        id: 'prerequisites',
        label: 'Prerequisites',
        type: 'prerequisite',
        children: [
          {
            id: 'train-split-prereq',
            label: 'Train / validation / test split',
            tooltip: tip({
              short: 'Preprocessing parameters must be learned from training rows only.',
              intuition: 'Validation and test are transformed with frozen train statistics.',
              example: 'Fit StandardScaler on X_train, then transform X_val and X_test.',
              trap: 'fit_transform on the full dataset before splitting leaks.',
            }),
            lessonId: 'train-validation-test-split',
          },
          {
            id: 'leakage-prereq',
            label: 'Data leakage',
            tooltip: tip({
              short: 'Preprocessing leakage inflates validation scores like target leakage.',
              intuition: 'The scaler “sees” validation rows when fit on all data.',
              example: 'Global mean includes validation incomes when computing μ.',
              trap: 'Amazing validation after bad scaling often means leakage.',
            }),
            lessonId: 'data-leakage-deep-dive',
          },
          {
            id: 'feature-vector-prereq',
            label: 'Feature vector',
            tooltip: tip({
              short: 'Each example is a row of numeric (or encoded) inputs.',
              intuition: 'Scaling operates per feature column across training rows.',
              example: 'Age and income are two columns scaled independently.',
              trap: 'Scaling labels or IDs as if they were features is a mistake.',
            }),
          },
          {
            id: 'distance-prereq',
            label: 'Distance / similarity',
            tooltip: tip({
              short: 'kNN, clustering, and some kernels depend on feature magnitudes.',
              intuition: 'Unequal scales make one dimension dominate Euclidean distance.',
              example: 'Income 50,000 vs age 35 makes age invisible in raw distance.',
              trap: 'Tree models are scale-invariant; scaling them is usually unnecessary.',
            }),
            lessonId: 'knn-naive-bayes-svm',
          },
          {
            id: 'gradient-scale-prereq',
            label: 'Gradient scale',
            tooltip: tip({
              short: 'Optimizers step on each weight; feature scale affects effective learning rate.',
              intuition: 'Large input magnitudes produce large activations and gradients.',
              example: 'Unscaled pixels 0–255 need smaller LR than normalized 0–1 inputs.',
              trap: 'BatchNorm partly compensates but does not replace thoughtful scaling.',
            }),
            lessonId: 'gradient-descent',
          },
          {
            id: 'pipeline-prereq',
            label: 'ML pipeline',
            tooltip: tip({
              short: 'Scaler → model should be one fit object applied consistently at serve time.',
              intuition: 'Serving must replay the exact train-frozen transformation.',
              example: 'sklearn Pipeline([("scaler", StandardScaler()), ("clf", SVM())]).',
              trap: 'Refitting scaler in production on live batch statistics causes skew.',
            }),
          },
        ],
      },
      {
        id: 'mechanism',
        label: 'Core mechanism',
        type: 'mechanism',
        children: [
          {
            id: 'fit-train-stats',
            label: 'Fit on train statistics',
            tooltip: tip({
              short: 'Compute μ, σ, min, max, median, or IQR using training rows only.',
              intuition: 'These numbers become fixed parameters of the preprocessor.',
              example: 'mean_age and std_age from train split only.',
              trap: 'Including validation rows in fit() contaminates evaluation.',
            }),
            highlightTarget: { panel: 'animation', type: 'fit-train' },
          },
          {
            id: 'transform-all-splits',
            label: 'Transform all splits',
            tooltip: tip({
              short: 'Apply the same frozen scaler to train, validation, and test.',
              intuition: 'Validation simulates unseen data with train-learned scaling.',
              example: 'z_val = (x_val - μ_train) / σ_train.',
              trap: 'Refitting per split defeats the purpose of held-out evaluation.',
            }),
            highlightTarget: { panel: 'animation', type: 'transform' },
          },
          {
            id: 'standardize-method',
            label: 'Standardization (z-score)',
            tooltip: tip({
              short: 'Center by mean and divide by standard deviation per feature.',
              intuition: 'Typical values land near 0 with unit-ish spread.',
              formula: 'z=(x-\\mu_{train})/\\sigma_{train}',
              example: 'Age 42 with μ=35, σ=10 → z=0.7.',
              trap: 'σ=0 on constant columns causes division by zero—drop or impute.',
            }),
            highlightTarget: { panel: 'animation', type: 'standard' },
          },
          {
            id: 'minmax-method',
            label: 'Min-max scaling',
            tooltip: tip({
              short: 'Linearly map each feature to [0, 1] using train min and max.',
              intuition: 'Preserves relative order within the train range.',
              formula: 'x_{mm}=(x-min_{train})/(max_{train}-min_{train})',
              example: 'Income 76k with min 38k, max 112k → 0.63.',
              trap: 'One validation outlier beyond train max maps outside [0, 1].',
            }),
            highlightTarget: { panel: 'animation', type: 'minmax' },
          },
          {
            id: 'robust-method',
            label: 'Robust scaling',
            tooltip: tip({
              short: 'Use median and IQR instead of mean and std.',
              intuition: 'Heavy tails and outliers distort mean/std less in robust stats.',
              formula: 'x_r=(x-median_{train})/IQR_{train}',
              example: 'Director income outlier barely moves median-based scale.',
              trap: 'IQR=0 on sparse features still breaks division.',
            }),
            highlightTarget: { panel: 'animation', type: 'robust' },
          },
        ],
      },
      {
        id: 'intuitions',
        label: 'Intuitions',
        type: 'intuition',
        children: [
          {
            id: 'fair-distance-intuition',
            label: 'Fair distance',
            tooltip: tip({
              short: 'Scaling puts features on comparable rulers before measuring closeness.',
              intuition: 'kNN asks “who is nearest?”—raw units pick the loudest column.',
              example: 'After scaling, age and income both influence neighbor votes.',
              trap: 'Scaling cannot fix bad features or wrong distance metric.',
            }),
          },
          {
            id: 'optimizer-landscape-intuition',
            label: 'Optimizer landscape',
            tooltip: tip({
              short: 'Well-scaled inputs make loss contours less elongated.',
              intuition: 'Gradient descent takes more balanced steps across weights.',
              example: 'Neural nets often train faster on normalized inputs.',
              trap: 'Learning rate still needs tuning after scaling.',
            }),
          },
          {
            id: 'outlier-leverage-intuition',
            label: 'Outlier leverage',
            tooltip: tip({
              short: 'Mean and std are outlier-sensitive; median and IQR resist them.',
              intuition: 'One CEO salary can stretch min-max and standardization.',
              example: 'Robust scaler keeps validation outlier from warping everyone.',
              trap: 'Robust scaling does not remove outliers—only reduces their leverage on stats.',
            }),
          },
          {
            id: 'train-only-frozen-intuition',
            label: 'Train-only freeze',
            tooltip: tip({
              short: 'Think of scaler parameters as part of what the model learned from train.',
              intuition: 'Validation is the dress rehearsal with those frozen choices.',
              example: 'Changing μ at serve time is like changing learned weights.',
              trap: 'CV must refit scaler inside each training fold.',
            }),
            lessonId: 'cross-validation',
          },
          {
            id: 'when-not-to-scale-intuition',
            label: 'When not to scale',
            tooltip: tip({
              short: 'Tree splits on thresholds are scale-invariant; interpretable coefficients may want raw units.',
              intuition: 'Match preprocessing to model inductive bias and reporting needs.',
              example: 'Random forest on raw counts is often fine.',
              trap: 'Blindly scaling every pipeline adds complexity without benefit.',
            }),
            lessonId: 'tree-ensembles',
          },
        ],
      },
      {
        id: 'formula-code',
        label: 'Formula / Code',
        type: 'formula',
        children: [
          {
            id: 'zscore-formula',
            label: 'Z-score formula',
            tooltip: tip({
              short: 'z = (x − μ_train) / σ_train per feature column.',
              intuition: 'μ and σ are vectors with one entry per feature.',
              formula: 'z_j=(x_j-\\mu_j)/\\sigma_j',
              example: 'Two-feature row scaled column-wise.',
              trap: 'Use population or sample std consistently with library defaults.',
            }),
          },
          {
            id: 'minmax-formula',
            label: 'Min-max formula',
            tooltip: tip({
              short: 'Linear map from [min_train, max_train] to [0, 1].',
              intuition: 'Values below min or above max extrapolate outside the interval.',
              formula: 'x_{mm}=(x-min)/(max-min)',
              example: 'Image pixels often use min=0, max=255 from train set.',
              trap: 'Test outliers clip or extrapolate—consider robust or clip options.',
            }),
          },
          {
            id: 'sklearn-scaler-code',
            label: 'sklearn scalers',
            tooltip: tip({
              short: 'StandardScaler, MinMaxScaler, RobustScaler with fit on train only.',
              intuition: 'Pipeline.fit(X_train) learns; transform applies everywhere.',
              code: 'scaler = StandardScaler()\nscaler.fit(X_train)\nX_val_s = scaler.transform(X_val)',
              example: 'Never call fit on validation or test rows.',
              trap: 'fit_transform(X_train) then transform val is correct; fit_transform(all) is not.',
            }),
          },
          {
            id: 'cv-pipeline-code',
            label: 'Pipeline inside CV',
            tooltip: tip({
              short: 'Each fold refits scaler on that fold’s training indices only.',
              intuition: 'Cross-validation must mirror deployment boundaries.',
              code: 'Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())])\nGridSearchCV(pipe, cv=5)',
              example: 'Scaler statistics never see validation fold rows.',
              trap: 'Scaling before CV outside the pipeline leaks across folds.',
            }),
            lessonId: 'cross-validation',
          },
          {
            id: 'constant-feature-handling',
            label: 'Constant feature handling',
            tooltip: tip({
              short: 'Drop or skip scaling columns with zero variance.',
              intuition: 'σ=0 makes z-score undefined.',
              example: 'One-hot column always 0 for rare category in train.',
              trap: 'Libraries may silently set scale=1—verify constant columns.',
            }),
          },
        ],
      },
      {
        id: 'traps',
        label: 'Common traps',
        type: 'trap',
        children: [
          {
            id: 'global-fit-trap',
            label: 'Global fit before split',
            tooltip: tip({
              short: 'Computing normalization on the entire dataset before train/val/test split.',
              intuition: 'Validation statistics influence training preprocessing.',
              example: 'df.describe() on all rows then StandardScaler.fit(X).',
              trap: 'This is one of the most common silent leakage bugs.',
            }),
            lessonId: 'data-leakage-deep-dive',
          },
          {
            id: 'test-fit-trap',
            label: 'Accidental test fit',
            tooltip: tip({
              short: 'Calling fit or fit_transform on test data during experimentation.',
              intuition: 'Test rows should only ever be transformed.',
              example: 'scaler.fit(X_test) “just to see” still leaks test distribution.',
              trap: 'Notebooks that reuse the same scaler variable across splits confuse roles.',
            }),
          },
          {
            id: 'outlier-minmax-trap',
            label: 'Outlier breaks min-max',
            tooltip: tip({
              short: 'Single extreme train value compresses all other points.',
              intuition: 'Most mass squashes near 0 or 1.',
              example: 'One 235k income with min-max on salaries.',
              trap: 'Prefer robust scaling or winsorize when tails are heavy.',
            }),
          },
          {
            id: 'scale-labels-trap',
            label: 'Scaling labels or IDs',
            tooltip: tip({
              short: 'Treating user_id or target-encoded columns like continuous features.',
              intuition: 'Identifiers are not magnitudes to normalize.',
              example: 'StandardScaler on user_id adds nonsense z-scores.',
              trap: 'Target encoding itself needs leakage-safe fitting.',
            }),
          },
          {
            id: 'serve-skew-trap',
            label: 'Train/serve skew',
            tooltip: tip({
              short: 'Production uses different imputation, clipping, or scaler version than training.',
              intuition: 'Offline metrics no longer predict live behavior.',
              example: 'Training clipped at 99th percentile; serving uses raw values.',
              trap: 'Version scaler artifacts with the model bundle.',
            }),
            lessonId: 'model-monitoring',
          },
        ],
      },
      {
        id: 'used-later',
        label: 'Used later',
        type: 'application',
        children: [
          {
            id: 'knn-scaling-lesson',
            label: 'kNN / SVM / Naive Bayes',
            tooltip: tip({
              short: 'Distance and margin models especially benefit from comparable feature scales.',
              intuition: 'Gaussian NB variances also assume comparable input ranges in practice.',
              example: 'SVM with RBF kernel on unscaled mixed units often underperforms.',
              trap: 'Naive Bayes on counts may need different treatment than z-scoring.',
            }),
            lessonId: 'knn-naive-bayes-svm',
          },
          {
            id: 'pca-scaling-lesson',
            label: 'PCA',
            tooltip: tip({
              short: 'PCA on mixed-scale features skews toward large-variance columns in raw units.',
              intuition: 'Center and often standardize before eigendecomposition.',
              example: 'Income variance dominates age until features are scaled.',
              trap: 'PCA on unscaled data picks dollar scale over year scale.',
            }),
            lessonId: 'pca',
          },
          {
            id: 'neural-scaling-lesson',
            label: 'Neural network training',
            tooltip: tip({
              short: 'Input normalization pairs with weight init and learning rate.',
              intuition: 'Bad scaling plus deep stacks worsen gradient problems.',
              example: 'Image models often use channel mean/std normalization.',
              trap: 'BatchNorm does not remove need for sensible input scale.',
            }),
            lessonId: 'neural-network',
          },
          {
            id: 'regularization-scaling-lesson',
            label: 'Regularization',
            tooltip: tip({
              short: 'L1/L2 penalties interact with feature scale—coefficients shrink per dimension.',
              intuition: 'Unscaled features make penalty unfair across columns.',
              example: 'Lasso on raw income and age penalizes income coefficients differently.',
              trap: 'Standardize before L1 when comparing feature importance magnitudes.',
            }),
            lessonId: 'regularization',
          },
          {
            id: 'data-engineering-scaling-lesson',
            label: 'Data engineering for ML',
            tooltip: tip({
              short: 'Feature stores must persist scaler parameters with point-in-time correctness.',
              intuition: 'Serving pipelines replay train-frozen transforms.',
              example: 'Store μ, σ with feature version and training cutoff.',
              trap: 'Recomputing aggregates online without time bounds leaks future stats.',
            }),
            lessonId: 'data-engineering-for-ml-track',
          },
        ],
      },
    ],
  },

};

export function getConceptMap(animationId) {
  return CONCEPT_MAPS[animationId] || null;
}

export function isConceptMap(mindmap) {
  return Boolean(mindmap?.center && mindmap?.branches);
}
