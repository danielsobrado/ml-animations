export const ALGEBRA_CODE_LABS = [
  {
    id: 'dot-product-basic',
    title: 'Dot product',
    concept: 'Multiply matching entries, then add the products.',
    objective: 'Complete a dot product function.',
    difficulty: 'warmup',
    starterCode: `function dot(a, b) {
  // TODO: replace 0 with the dot product.
  // Example: dot([1, 2], [3, 4]) = 1*3 + 2*4 = 11
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

check('dot([1, 2], [3, 4])', dot([1, 2], [3, 4]), 11);
check('dot([0, 5], [10, 2])', dot([0, 5], [10, 2]), 10);
check('dot([-1, 2], [3, 5])', dot([-1, 2], [3, 5]), 7);
check('dot([2, 2, 2], [1, 2, 3])', dot([2, 2, 2], [1, 2, 3]), 12);

return results;`,
    hints: [
      'The dot product pairs entries by index: a[0] with b[0], a[1] with b[1], and so on.',
      'For two vectors, use a loop and keep a running total.',
      `let total = 0;
for (let i = 0; i < a.length; i++) {
  total += a[i] * b[i];
}
return total;`,
    ],
    solution: `function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}`,
    explanation: 'A dot product is a weighted sum: each entry of one vector weights the matching entry of the other vector.',
  },

  {
    id: 'matrix-cell',
    title: 'One matrix multiplication cell',
    concept: 'One output cell is one row-column dot product.',
    objective: 'Complete the function that computes C[row][col].',
    difficulty: 'core',
    starterCode: `function matrixCell(A, B, row, col) {
  // TODO: compute one cell of C = A * B.
  // Use row "row" from A and column "col" from B.
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

check('C[0][0]', matrixCell(A, B, 0, 0), 4);
check('C[0][1]', matrixCell(A, B, 0, 1), 9);
check('C[0][2]', matrixCell(A, B, 0, 2), 7);
check('C[1][0]', matrixCell(A, B, 1, 0), 7);
check('C[1][1]', matrixCell(A, B, 1, 1), 7);
check('C[1][2]', matrixCell(A, B, 1, 2), 11);

return results;`,
    hints: [
      'The number of terms in the dot product is the number of columns in A, which is also the number of rows in B.',
      'Use A[row][k] and B[k][col]. The index k moves across the row of A and down the column of B.',
      `let total = 0;
for (let k = 0; k < B.length; k++) {
  total += A[row][k] * B[k][col];
}
return total;`,
    ],
    solution: `function matrixCell(A, B, row, col) {
  let total = 0;
  for (let k = 0; k < B.length; k++) {
    total += A[row][k] * B[k][col];
  }
  return total;
}`,
    explanation: 'Matrix multiplication is repeated dot products. Each output cell C[row][col] is row row of A dotted with column col of B.',
  },

  {
    id: 'matrix-multiply-full',
    title: 'Full matrix multiplication',
    concept: 'Fill every output cell by reusing the row-column rule.',
    objective: 'Complete a full matrix multiplication function.',
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
      // TODO: push the correct cell value into row.
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
      'The outer loop chooses the output row i. The inner loop chooses the output column j.',
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
    explanation: 'The full matrix product is just the matrixCell rule repeated for every row and every column.',
  },
];
