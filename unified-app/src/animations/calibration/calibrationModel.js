export const SCENARIOS = Object.freeze({
  calibrated: {
    label: 'Calibrated',
    detail: 'Predicted probabilities line up with observed frequencies.',
    bins: [
      { confidence: 0.1, observed: 0.09, count: 18 },
      { confidence: 0.3, observed: 0.31, count: 24 },
      { confidence: 0.5, observed: 0.49, count: 28 },
      { confidence: 0.7, observed: 0.72, count: 24 },
      { confidence: 0.9, observed: 0.88, count: 16 },
    ],
  },
  overconfident: {
    label: 'Overconfident',
    detail: 'Scores are too extreme: high-confidence buckets are right less often than promised.',
    bins: [
      { confidence: 0.1, observed: 0.22, count: 18 },
      { confidence: 0.3, observed: 0.38, count: 24 },
      { confidence: 0.5, observed: 0.51, count: 28 },
      { confidence: 0.7, observed: 0.61, count: 24 },
      { confidence: 0.9, observed: 0.74, count: 16 },
    ],
  },
  underconfident: {
    label: 'Underconfident',
    detail: 'Scores are too timid: low and high buckets sit closer to 0.5 than the outcomes justify.',
    bins: [
      { confidence: 0.1, observed: 0.03, count: 18 },
      { confidence: 0.3, observed: 0.18, count: 24 },
      { confidence: 0.5, observed: 0.5, count: 28 },
      { confidence: 0.7, observed: 0.82, count: 24 },
      { confidence: 0.9, observed: 0.96, count: 16 },
    ],
  },
});

export function totalCount(bins) {
  return bins.reduce((sum, bin) => sum + bin.count, 0);
}

export function expectedCalibrationError(bins) {
  const total = totalCount(bins);
  return bins.reduce((sum, bin) => sum + (bin.count / total) * Math.abs(bin.observed - bin.confidence), 0);
}

export function brierScore(bins) {
  const total = totalCount(bins);
  return bins.reduce((sum, bin) => {
    const positives = bin.count * bin.observed;
    const negatives = bin.count - positives;
    return sum + positives * (1 - bin.confidence) ** 2 + negatives * bin.confidence ** 2;
  }, 0) / total;
}

export function thresholdStats(bins, threshold) {
  const predictedPositive = bins.filter((bin) => bin.confidence >= threshold);
  const predictedNegative = bins.filter((bin) => bin.confidence < threshold);
  const tp = predictedPositive.reduce((sum, bin) => sum + bin.count * bin.observed, 0);
  const fp = predictedPositive.reduce((sum, bin) => sum + bin.count * (1 - bin.observed), 0);
  const fn = predictedNegative.reduce((sum, bin) => sum + bin.count * bin.observed, 0);
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  return {
    predictedPositive: predictedPositive.reduce((sum, bin) => sum + bin.count, 0),
    precision,
    recall,
  };
}

export function project(value) {
  return 260 - value * 220;
}
