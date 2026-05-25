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
};

export function getConceptMap(animationId) {
  return CONCEPT_MAPS[animationId] || null;
}

export function isConceptMap(mindmap) {
  return Boolean(mindmap?.center && mindmap?.branches);
}
