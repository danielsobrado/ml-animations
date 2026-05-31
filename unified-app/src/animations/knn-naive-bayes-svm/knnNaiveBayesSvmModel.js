export const POINTS = [
  { id: 'A', x: -2.4, y: 1.1, label: 'blue' },
  { id: 'B', x: -1.7, y: 0.4, label: 'blue' },
  { id: 'C', x: -1.1, y: 1.5, label: 'blue' },
  { id: 'D', x: -0.6, y: 0.2, label: 'blue' },
  { id: 'E', x: 0.7, y: -0.8, label: 'orange' },
  { id: 'F', x: 1.2, y: -1.5, label: 'orange' },
  { id: 'G', x: 1.8, y: -0.3, label: 'orange' },
  { id: 'H', x: 2.4, y: -1.1, label: 'orange' },
];

export const MODELS = {
  knn: {
    label: 'kNN',
    detail: 'Classifies by the labels of the nearest training points.',
  },
  naiveBayes: {
    label: 'Naive Bayes',
    detail: 'Multiplies per-feature likelihoods as if features were conditionally independent.',
  },
  svm: {
    label: 'SVM',
    detail: 'Chooses the side of a maximum-margin decision boundary.',
  },
};

export const SVM_PARAMS = {
  weight: [1.05, -0.9],
  bias: -0.05,
};

export function dist(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

export function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function variance(values) {
  const mu = mean(values);
  return values.reduce((sum, value) => sum + (value - mu) ** 2, 0) / values.length + 0.08;
}

export function gaussianLogPdf(value, mu, varValue) {
  return -0.5 * Math.log(2 * Math.PI * varValue) - ((value - mu) ** 2) / (2 * varValue);
}

export function classStats(label) {
  const classPoints = POINTS.filter((point) => point.label === label);
  const xs = classPoints.map((point) => point.x);
  const ys = classPoints.map((point) => point.y);
  return {
    prior: classPoints.length / POINTS.length,
    meanX: mean(xs),
    meanY: mean(ys),
    varX: variance(xs),
    varY: variance(ys),
  };
}

export function classifyKnn(query, k) {
  const neighbors = POINTS
    .map((point) => ({ ...point, distance: dist(point, query) }))
    .sort((a, b) => a.distance - b.distance);
  const votes = neighbors.slice(0, k).reduce((acc, point) => {
    acc[point.label] = (acc[point.label] || 0) + 1;
    return acc;
  }, {});
  const prediction = (votes.blue || 0) >= (votes.orange || 0) ? 'blue' : 'orange';
  return { prediction, neighbors, confidence: Math.max(votes.blue || 0, votes.orange || 0) / k };
}

export function classifyNaiveBayes(query) {
  const scores = Object.fromEntries(['blue', 'orange'].map((label) => {
    const stats = classStats(label);
    const score = Math.log(stats.prior)
      + gaussianLogPdf(query.x, stats.meanX, stats.varX)
      + gaussianLogPdf(query.y, stats.meanY, stats.varY);
    return [label, score];
  }));
  const prediction = scores.blue >= scores.orange ? 'blue' : 'orange';
  const expBlue = Math.exp(scores.blue - Math.max(scores.blue, scores.orange));
  const expOrange = Math.exp(scores.orange - Math.max(scores.blue, scores.orange));
  return {
    prediction,
    scores,
    confidence: prediction === 'blue'
      ? expBlue / (expBlue + expOrange)
      : expOrange / (expBlue + expOrange),
  };
}

export function svmMarginScore(query, params = SVM_PARAMS) {
  return params.weight[0] * query.x + params.weight[1] * query.y + params.bias;
}

export function classifySvm(query) {
  const marginScore = svmMarginScore(query);
  return {
    prediction: marginScore >= 0 ? 'orange' : 'blue',
    marginScore,
    confidence: Math.min(0.99, Math.abs(marginScore) / 2.4),
  };
}

export function project(point) {
  return {
    cx: 36 + ((point.x + 3) / 6) * 328,
    cy: 276 - ((point.y + 2.4) / 4.8) * 240,
  };
}

export function svmBoundarySegment(params = SVM_PARAMS) {
  const [wx, wy] = params.weight;
  const { bias } = params;
  const domain = { minX: -3, maxX: 3, minY: -2.4, maxY: 2.4 };
  const candidates = [];

  for (const x of [domain.minX, domain.maxX]) {
    const y = -(wx * x + bias) / wy;
    if (y >= domain.minY && y <= domain.maxY) candidates.push({ x, y });
  }

  for (const y of [domain.minY, domain.maxY]) {
    const x = -(wy * y + bias) / wx;
    if (x >= domain.minX && x <= domain.maxX) candidates.push({ x, y });
  }

  const unique = candidates.filter((point, index) => (
    candidates.findIndex((other) => Math.abs(other.x - point.x) < 1e-9 && Math.abs(other.y - point.y) < 1e-9) === index
  ));
  return unique.slice(0, 2).map(project);
}
