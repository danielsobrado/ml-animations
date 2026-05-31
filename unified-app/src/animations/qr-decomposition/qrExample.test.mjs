import assert from 'node:assert/strict';
import test from 'node:test';

import { QR_EXAMPLE } from './qrExample.js';

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

test('qr animation example reconstructs its displayed matrix', () => {
  const { matrixA, matrixQ, matrixR } = QR_EXAMPLE;

  assertMatrixClose(multiply(matrixQ, matrixR), matrixA);
  assertMatrixClose(multiply(transpose(matrixQ), matrixQ), [[1, 0], [0, 1]]);
  assert.equal(matrixR[1][0], 0);
});

test('qr animation highlights columns for Gram-Schmidt steps', () => {
  const { columnCellIndices, matrixA } = QR_EXAMPLE;
  const flat = matrixA.flat();

  assert.deepEqual(columnCellIndices.map((indices) => indices.map((index) => flat[index])), [
    [3, 0],
    [1, 2],
  ]);
});
