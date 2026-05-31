import assert from 'node:assert/strict';
import test from 'node:test';

import { SVD_EXAMPLE } from './svdExample.js';

function multiply(left, right) {
  return left.map((row) => (
    right[0].map((_, col) => row.reduce((sum, value, index) => sum + value * right[index][col], 0))
  ));
}

function transpose(matrix) {
  return matrix[0].map((_, col) => matrix.map((row) => row[col]));
}

function assertMatrixClose(actual, expected, tolerance = 1e-10) {
  assert.equal(actual.length, expected.length);
  assert.equal(actual[0].length, expected[0].length);

  for (const [rowIndex, row] of actual.entries()) {
    for (const [colIndex, value] of row.entries()) {
      assert.ok(
        Math.abs(value - expected[rowIndex][colIndex]) <= tolerance,
        `entry ${rowIndex},${colIndex}: expected ${expected[rowIndex][colIndex]}, got ${value}`,
      );
    }
  }
}

test('svd animation example reconstructs its displayed matrix', () => {
  const { matrixA, matrixU, matrixSigma, matrixVT } = SVD_EXAMPLE;
  const reconstruction = multiply(multiply(matrixU, matrixSigma), matrixVT);

  assertMatrixClose(reconstruction, matrixA);
});

test('svd animation example keeps valid singular-value structure', () => {
  const { matrixU, matrixSigma, matrixVT } = SVD_EXAMPLE;
  const singularValues = [matrixSigma[0][0], matrixSigma[1][1]];

  assert.ok(singularValues.every((value) => value >= 0));
  assert.ok(singularValues[0] >= singularValues[1]);
  assert.equal(matrixSigma[2][0], 0);
  assert.equal(matrixSigma[2][1], 0);
  assertMatrixClose(multiply(transpose(matrixU), matrixU), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  assertMatrixClose(multiply(matrixVT, transpose(matrixVT)), [[1, 0], [0, 1]]);
});
