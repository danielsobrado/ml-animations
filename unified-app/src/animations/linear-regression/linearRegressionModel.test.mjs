import assert from 'node:assert/strict';
import test from 'node:test';

import {
  LINEAR_REGRESSION_DEMO_DATA,
  calculateMSE,
  calculateOLS,
  calculateResiduals,
  predict,
} from './linearRegressionModel.js';

test('linear regression demo model computes predictions, residuals, and MSE', () => {
  const model = { slope: 1, intercept: 0 };
  const { residuals, mse } = calculateResiduals(LINEAR_REGRESSION_DEMO_DATA, model);

  assert.equal(predict(model, 4), 4);
  assert.deepEqual(residuals.map((point) => point.error), [1, 1, 2, 0, 1]);
  assert.equal(mse, 7 / 5);
  assert.equal(calculateMSE(LINEAR_REGRESSION_DEMO_DATA, model), 7 / 5);
});

test('interactive OLS fitter handles ordinary and vertical point sets', () => {
  const model = calculateOLS([
    { x: 0, y: 1 },
    { x: 1, y: 3 },
    { x: 2, y: 5 },
  ]);

  assert.ok(model);
  assert.equal(model.slope, 2);
  assert.equal(model.intercept, 1);
  assert.equal(calculateOLS([{ x: 2, y: 1 }, { x: 2, y: 3 }]), null);
});
