export const ALGEBRA_CODE_LABS = [
  {
    id: 'dot-product-first-pair',
    stepLabel: '1.1',
    title: 'First matching pair',
    concept: 'A dot product starts by multiplying entries with the same index.',
    objective: 'Replace one number with the first pair product.',
    difficulty: 'warmup',
    starterCode: `function firstPairProduct(a, b) {
  // TODO: replace 0 with the product of the first entries.
  return 0;
}`,
    testCode: `const results = [];

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: Object.is(actual, expected),
  });
}

check('first pair in [1, 2] dot [3, 4]', firstPairProduct([1, 2], [3, 4]), 3);
check('first pair in [0, 5] dot [10, 2]', firstPairProduct([0, 5], [10, 2]), 0);
check('first pair in [-1, 2] dot [3, 5]', firstPairProduct([-1, 2], [3, 5]), -3);

return results;`,
    hints: [
      'Use index 0 for the first entry of each vector.',
      'The first pair product is a[0] times b[0].',
      'return a[0] * b[0];',
    ],
    solution: `function firstPairProduct(a, b) {
  return a[0] * b[0];
}`,
    explanation: 'The first contribution to a dot product comes from multiplying the two index-0 entries.',
  },

  {
    id: 'dot-product-two-pairs',
    stepLabel: '1.2',
    title: 'Add two pair products',
    concept: 'A two-entry dot product adds the first pair product and the second pair product.',
    objective: 'Replace one expression with the missing second pair product.',
    difficulty: 'warmup',
    starterCode: `function dot2(a, b) {
  const first = a[0] * b[0];
  const second = 0; // TODO: replace 0.

  return first + second;
}`,
    testCode: `const results = [];

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: Object.is(actual, expected),
  });
}

check('dot2([1, 2], [3, 4])', dot2([1, 2], [3, 4]), 11);
check('dot2([0, 5], [10, 2])', dot2([0, 5], [10, 2]), 10);
check('dot2([-1, 2], [3, 5])', dot2([-1, 2], [3, 5]), 7);

return results;`,
    hints: [
      'The second pair uses index 1 in both arrays.',
      'Keep the existing return line. Only fix the value assigned to second.',
      'const second = a[1] * b[1];',
    ],
    solution: `function dot2(a, b) {
  const first = a[0] * b[0];
  const second = a[1] * b[1];

  return first + second;
}`,
    explanation: 'A two-entry dot product is a[0] * b[0] plus a[1] * b[1].',
  },

  {
    id: 'dot-product-loop-update',
    stepLabel: '1.3',
    title: 'Loop over every pair',
    concept: 'The loop repeats the same pair-product rule for vectors of any length.',
    objective: 'Complete the one accumulator update inside the loop.',
    difficulty: 'core',
    starterCode: `function dot(a, b) {
  let total = 0;

  for (let i = 0; i < a.length; i++) {
    // TODO: replace 0 with the current pair product.
    total += 0;
  }

  return total;
}`,
    testCode: `const results = [];

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: Object.is(actual, expected),
  });
}

check('dot([1, 2], [3, 4])', dot([1, 2], [3, 4]), 11);
check('dot([0, 5], [10, 2])', dot([0, 5], [10, 2]), 10);
check('dot([-1, 2], [3, 5])', dot([-1, 2], [3, 5]), 7);
check('dot([2, 2, 2], [1, 2, 3])', dot([2, 2, 2], [1, 2, 3]), 12);

return results;`,
    hints: [
      'Inside the loop, i points to the current matching pair.',
      'Add a[i] times b[i] into total.',
      'total += a[i] * b[i];',
    ],
    solution: `function dot(a, b) {
  let total = 0;

  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }

  return total;
}`,
    explanation: 'The loop version is the same rule as dot2, repeated until every matching pair has contributed.',
  },

  {
    id: 'matrix-cell-one-term',
    stepLabel: '2.1',
    title: 'One cell, first term',
    concept: 'One matrix-product cell begins with A[row][0] times B[0][col].',
    objective: 'Replace one expression with the first term of a row-column dot product.',
    difficulty: 'core',
    starterCode: `function firstCellTerm(A, B, row, col) {
  // TODO: replace 0 with the first row-column product.
  return 0;
}`,
    testCode: `const results = [];

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: Object.is(actual, expected),
  });
}

const A = [
  [1, 2],
  [3, 1],
];

const B = [
  [2, 1, 3],
  [1, 4, 2],
];

check('first term for C[0][0]', firstCellTerm(A, B, 0, 0), 2);
check('first term for C[0][2]', firstCellTerm(A, B, 0, 2), 3);
check('first term for C[1][1]', firstCellTerm(A, B, 1, 1), 3);

return results;`,
    hints: [
      'Use the selected row from A, the selected column from B, and k = 0.',
      'The first term is A[row][0] times B[0][col].',
      'return A[row][0] * B[0][col];',
    ],
    solution: `function firstCellTerm(A, B, row, col) {
  return A[row][0] * B[0][col];
}`,
    explanation: 'A matrix cell is a dot product; this is the first product in that dot product.',
  },

  {
    id: 'matrix-cell-loop-update',
    stepLabel: '2.2',
    title: 'One cell loop',
    concept: 'The index k moves across a row of A and down a column of B.',
    objective: 'Complete the one accumulator update for a matrix cell.',
    difficulty: 'core',
    starterCode: `function matrixCell(A, B, row, col) {
  let total = 0;

  for (let k = 0; k < B.length; k++) {
    // TODO: replace 0 with the current row-column product.
    total += 0;
  }

  return total;
}`,
    testCode: `const results = [];

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: Object.is(actual, expected),
  });
}

const A = [
  [1, 2],
  [3, 1],
];

const B = [
  [2, 1, 3],
  [1, 4, 2],
];

check('C[0][0]', matrixCell(A, B, 0, 0), 4);
check('C[0][1]', matrixCell(A, B, 0, 1), 9);
check('C[0][2]', matrixCell(A, B, 0, 2), 7);
check('C[1][0]', matrixCell(A, B, 1, 0), 7);
check('C[1][1]', matrixCell(A, B, 1, 1), 7);
check('C[1][2]', matrixCell(A, B, 1, 2), 11);

return results;`,
    hints: [
      'Use k as the shared index between A and B.',
      'A[row][k] chooses the next entry in the row. B[k][col] chooses the next entry in the column.',
      'total += A[row][k] * B[k][col];',
    ],
    solution: `function matrixCell(A, B, row, col) {
  let total = 0;

  for (let k = 0; k < B.length; k++) {
    total += A[row][k] * B[k][col];
  }

  return total;
}`,
    explanation: 'The complete cell is the sum of every row-column product for that row and column.',
  },

  {
    id: 'matrix-multiply-column-count',
    stepLabel: '3.1',
    title: 'Output column count',
    concept: 'The product A * B has one output column for each column in B.',
    objective: 'Replace one number so the inner loop visits every output column.',
    difficulty: 'challenge',
    starterCode: `function matrixCell(A, B, row, col) {
  let total = 0;
  for (let k = 0; k < B.length; k++) {
    total += A[row][k] * B[k][col];
  }
  return total;
}

function matmul(A, B) {
  const rows = A.length;
  const cols = 0; // TODO: replace 0 with the number of output columns.

  const C = [];

  for (let i = 0; i < rows; i++) {
    const row = [];

    for (let j = 0; j < cols; j++) {
      row.push(matrixCell(A, B, i, j));
    }

    C.push(row);
  }

  return C;
}`,
    testCode: `const results = [];

function sameMatrix(actual, expected) {
  return JSON.stringify(actual) === JSON.stringify(expected);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameMatrix(actual, expected),
  });
}

check(
  '2x2 times 2x3',
  matmul(
    [[1, 2], [3, 1]],
    [[2, 1, 3], [1, 4, 2]]
  ),
  [[4, 9, 7], [7, 7, 11]]
);

check(
  '2x3 times 3x1',
  matmul(
    [[1, 2, 3], [4, 5, 6]],
    [[1], [2], [3]]
  ),
  [[14], [32]]
);

return results;`,
    hints: [
      'The number of output columns comes from the first row of B.',
      'B[0] is the first row of B. Its length is the number of columns.',
      'const cols = B[0].length;',
    ],
    solution: `function matrixCell(A, B, row, col) {
  let total = 0;
  for (let k = 0; k < B.length; k++) {
    total += A[row][k] * B[k][col];
  }
  return total;
}

function matmul(A, B) {
  const rows = A.length;
  const cols = B[0].length;

  const C = [];

  for (let i = 0; i < rows; i++) {
    const row = [];

    for (let j = 0; j < cols; j++) {
      row.push(matrixCell(A, B, i, j));
    }

    C.push(row);
  }

  return C;
}`,
    explanation: 'The shape of A * B is rows of A by columns of B, so the inner loop must run once per column in B.',
  },

  {
    id: 'matrix-multiply-push-cell',
    stepLabel: '3.2',
    title: 'Push each computed cell',
    concept: 'The nested loops choose each output position; matrixCell computes the value for that position.',
    objective: 'Replace one argument so each row receives the computed cell value.',
    difficulty: 'challenge',
    starterCode: `function matrixCell(A, B, row, col) {
  let total = 0;
  for (let k = 0; k < B.length; k++) {
    total += A[row][k] * B[k][col];
  }
  return total;
}

function matmul(A, B) {
  const rows = A.length;
  const cols = B[0].length;

  const C = [];

  for (let i = 0; i < rows; i++) {
    const row = [];

    for (let j = 0; j < cols; j++) {
      // TODO: replace 0 with the computed C[i][j] value.
      row.push(0);
    }

    C.push(row);
  }

  return C;
}`,
    testCode: `const results = [];

function sameMatrix(actual, expected) {
  return JSON.stringify(actual) === JSON.stringify(expected);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameMatrix(actual, expected),
  });
}

check(
  '2x2 times 2x3',
  matmul(
    [[1, 2], [3, 1]],
    [[2, 1, 3], [1, 4, 2]]
  ),
  [[4, 9, 7], [7, 7, 11]]
);

check(
  'identity matrix',
  matmul(
    [[1, 0], [0, 1]],
    [[5, 6], [7, 8]]
  ),
  [[5, 6], [7, 8]]
);

check(
  '2x3 times 3x1',
  matmul(
    [[1, 2, 3], [4, 5, 6]],
    [[1], [2], [3]]
  ),
  [[14], [32]]
);

return results;`,
    hints: [
      'You already have matrixCell(A, B, i, j). Use it inside the nested loops.',
      'The outer loop chooses output row i. The inner loop chooses output column j.',
      'row.push(matrixCell(A, B, i, j));',
    ],
    solution: `function matrixCell(A, B, row, col) {
  let total = 0;
  for (let k = 0; k < B.length; k++) {
    total += A[row][k] * B[k][col];
  }
  return total;
}

function matmul(A, B) {
  const rows = A.length;
  const cols = B[0].length;

  const C = [];

  for (let i = 0; i < rows; i++) {
    const row = [];

    for (let j = 0; j < cols; j++) {
      row.push(matrixCell(A, B, i, j));
    }

    C.push(row);
  }

  return C;
}`,
    explanation: 'The full matrix product is the matrixCell rule repeated for every row and every column.',
  },
];
