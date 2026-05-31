export const LINEAR_REGRESSION_DEMO_DATA = Object.freeze([
  { x: 1, y: 2 },
  { x: 2, y: 3 },
  { x: 3, y: 5 },
  { x: 4, y: 4 },
  { x: 5, y: 6 },
]);

export function predict({ slope, intercept }, x) {
  return slope * x + intercept;
}

export function calculateResiduals(data, model) {
  let totalSquaredError = 0;
  const residuals = data.map((point) => {
    const predictedY = predict(model, point.x);
    const error = point.y - predictedY;
    totalSquaredError += error * error;
    return { ...point, predictedY, error };
  });

  return {
    residuals,
    mse: totalSquaredError / data.length,
  };
}

export function calculateMSE(data, model) {
  return calculateResiduals(data, model).mse;
}

export function calculateOLS(points) {
  if (points.length < 2) return null;

  const n = points.length;
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumXX = 0;

  for (const point of points) {
    sumX += point.x;
    sumY += point.y;
    sumXY += point.x * point.y;
    sumXX += point.x * point.x;
  }

  const denominator = n * sumXX - sumX * sumX;
  if (Math.abs(denominator) < 1e-12) return null;

  const slope = (n * sumXY - sumX * sumY) / denominator;
  const intercept = (sumY - slope * sumX) / n;

  return { slope, intercept };
}
