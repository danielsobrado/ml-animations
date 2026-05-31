export const FEATURES = [
  { id: 'signalA', label: 'Signal A', base: 2.4, importance: 1.0, useful: true },
  { id: 'signalB', label: 'Signal B', base: -1.8, importance: 0.85, useful: true },
  { id: 'weakSignal', label: 'Weak signal', base: 0.8, importance: 0.45, useful: true },
  { id: 'noiseA', label: 'Noise A', base: 1.35, importance: 0.08, useful: false },
  { id: 'noiseB', label: 'Noise B', base: -1.1, importance: 0.05, useful: false },
  { id: 'noiseC', label: 'Noise C', base: 0.65, importance: 0.04, useful: false },
];

export const PENALTIES = {
  none: {
    label: 'None',
    detail: 'Weights only answer the training loss, so noisy features can stay large.',
    l1: 0,
    l2: 0,
  },
  l2: {
    label: 'L2 / ridge',
    detail: 'Shrinks weights smoothly while usually keeping all features active.',
    l1: 0,
    l2: 1,
  },
  l1: {
    label: 'L1 / lasso',
    detail: 'Can drive weak or noisy weights exactly to zero.',
    l1: 1,
    l2: 0,
  },
  elastic: {
    label: 'Elastic net',
    detail: 'Combines sparse selection with smooth shrinkage.',
    l1: 0.55,
    l2: 0.45,
  },
};

function effectiveLambda(penaltyId, lambda) {
  return penaltyId === 'none' ? 0 : lambda;
}

export function shrinkFeature(feature, penaltyId, lambda) {
  const penalty = PENALTIES[penaltyId];
  const appliedLambda = effectiveLambda(penaltyId, lambda);
  if (appliedLambda === 0) return { ...feature, weight: feature.base, removed: false };

  const l2Shrink = 1 / (1 + appliedLambda * penalty.l2 * 2.1);
  const afterL2 = feature.base * l2Shrink;
  const l1Cut = appliedLambda * penalty.l1 * (feature.useful ? 0.9 : 1.45);
  const sign = Math.sign(afterL2);
  const magnitude = Math.max(0, Math.abs(afterL2) - l1Cut);
  const weight = sign * magnitude;
  return { ...feature, weight, removed: Math.abs(weight) < 0.04 };
}

export function lossProfile(weights, lambda, penaltyId) {
  const penalty = PENALTIES[penaltyId];
  const appliedLambda = effectiveLambda(penaltyId, lambda);
  const signalLoss = weights.reduce((sum, feature) => {
    const lostUsefulSignal = feature.useful ? Math.abs(feature.base - feature.weight) * feature.importance * 5.5 : 0;
    const noisyVariance = feature.useful ? 0 : Math.abs(feature.weight) * 6.5;
    return sum + lostUsefulSignal + noisyVariance;
  }, 15);
  const l1Penalty = weights.reduce((sum, feature) => sum + Math.abs(feature.weight), 0) * appliedLambda * penalty.l1 * 2.5;
  const l2Penalty = weights.reduce((sum, feature) => sum + feature.weight ** 2, 0) * appliedLambda * penalty.l2 * 1.25;
  const train = signalLoss + l1Penalty * 0.2 + l2Penalty * 0.2;
  const validation = signalLoss + weights.filter((feature) => !feature.useful && !feature.removed).length * 3.5 + Math.max(0, appliedLambda - 0.55) * 18;
  return {
    dataLoss: signalLoss,
    penaltyLoss: l1Penalty + l2Penalty,
    train,
    validation,
    total: signalLoss + l1Penalty + l2Penalty,
  };
}

export function sweepProfile(penaltyId) {
  return Array.from({ length: 11 }, (_, index) => {
    const lambda = index / 10;
    const weights = FEATURES.map((feature) => shrinkFeature(feature, penaltyId, lambda));
    const losses = lossProfile(weights, lambda, penaltyId);
    return { lambda, ...losses };
  });
}

export function bestLambda(points) {
  return points.reduce((best, point) => (point.validation < best.validation ? point : best), points[0]);
}

export function linePath(points, key) {
  const max = Math.max(...points.flatMap((point) => [point.train, point.validation, point.total]), 1);
  return points.map((point, index) => {
    const x = 28 + index * 30;
    const y = 168 - (point[key] / max) * 130;
    return `${index === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(' ');
}

export function regularizationSummary(weights) {
  const removedCount = weights.filter((feature) => feature.removed).length;
  const noisyActive = weights.filter((feature) => !feature.useful && !feature.removed).length;
  const usefulMass = weights.filter((feature) => feature.useful).reduce((sum, feature) => sum + Math.abs(feature.weight), 0);
  const baseUsefulMass = FEATURES.filter((feature) => feature.useful).reduce((sum, feature) => sum + Math.abs(feature.base), 0);
  return {
    removedCount,
    noisyActive,
    usefulRetention: usefulMass / baseUsefulMass,
  };
}

export function diagnosisForState({ penaltyId, lambda, noisyActive, usefulRetention }) {
  if (penaltyId === 'none') {
    return 'No penalty: noisy weights remain active; compare a regularized setting on validation.';
  }
  if (lambda < 0.15) {
    return 'Too weak: noisy weights remain active and validation can suffer.';
  }
  if (lambda > 0.75) {
    return 'Too strong: useful signal is being shrunk enough to underfit.';
  }
  if (noisyActive <= 1 && usefulRetention > 0.55) {
    return 'Balanced: noisy weights are controlled while useful signal remains.';
  }
  return 'Tradeoff zone: compare validation loss before increasing lambda.';
}

export function percent(value) {
  return `${Math.round(value * 100)}%`;
}
