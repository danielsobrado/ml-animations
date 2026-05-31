export const BASE_POINTS = Object.freeze([
  { id: 'A', age: 23, income: 38000, split: 'train', label: 'student' },
  { id: 'B', age: 31, income: 52000, split: 'train', label: 'early career' },
  { id: 'C', age: 42, income: 76000, split: 'train', label: 'manager' },
  { id: 'D', age: 53, income: 94000, split: 'train', label: 'senior' },
  { id: 'E', age: 61, income: 112000, split: 'validation', label: 'director' },
  { id: 'F', age: 35, income: 58000, split: 'validation', label: 'analyst' },
]);

export const OUTLIER = Object.freeze({
  id: 'G',
  age: 64,
  income: 235000,
  split: 'validation',
  label: 'outlier',
});

export const METHODS = Object.freeze({
  raw: {
    label: 'Raw',
    formula: 'x',
    detail: 'No scaling: income units dominate distance and gradient magnitudes.',
  },
  standard: {
    label: 'Standardize',
    formula: '(x - mean) / std',
    detail: 'Centers each feature and measures values in standard deviations.',
  },
  minmax: {
    label: 'Min-max',
    formula: '(x - min) / (max - min)',
    detail: 'Compresses features into a 0-to-1 range using fitted endpoints.',
  },
  robust: {
    label: 'Robust',
    formula: '(x - median) / IQR',
    detail: 'Uses median and interquartile range, so one outlier has less leverage.',
  },
});

export function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function percentile(values, p) {
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * p;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

export function stats(values) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const q1 = percentile(values, 0.25);
  const q3 = percentile(values, 0.75);
  return {
    mean,
    std: Math.sqrt(variance) || 1,
    min: Math.min(...values),
    max: Math.max(...values),
    median: median(values),
    iqr: q3 - q1 || 1,
  };
}

export function fitScaler(points, fitOnAllData) {
  const fitPoints = fitOnAllData ? points : points.filter((point) => point.split === 'train');
  return {
    age: stats(fitPoints.map((point) => point.age)),
    income: stats(fitPoints.map((point) => point.income)),
  };
}

export function transformValue(value, featureStats, method) {
  if (method === 'standard') return (value - featureStats.mean) / featureStats.std;
  if (method === 'minmax') return (value - featureStats.min) / (featureStats.max - featureStats.min || 1);
  if (method === 'robust') return (value - featureStats.median) / featureStats.iqr;
  return value;
}

export function transformPoint(point, scaler, method) {
  return {
    ...point,
    x: transformValue(point.age, scaler.age, method),
    y: transformValue(point.income, scaler.income, method),
  };
}

export function distance(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

export function bounds(points) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
  };
}

export function project(point, box) {
  const pad = 34;
  const xRange = box.maxX - box.minX || 1;
  const yRange = box.maxY - box.minY || 1;
  return {
    cx: pad + ((point.x - box.minX) / xRange) * (360 - pad * 2),
    cy: 260 - pad - ((point.y - box.minY) / yRange) * (260 - pad * 2),
  };
}
