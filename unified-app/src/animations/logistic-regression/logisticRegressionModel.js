export const POINTS = Object.freeze([
  { id: 'A', risk: 16, engagement: 72, y: 0 },
  { id: 'B', risk: 22, engagement: 50, y: 0 },
  { id: 'C', risk: 26, engagement: 34, y: 0 },
  { id: 'D', risk: 31, engagement: 62, y: 0 },
  { id: 'E', risk: 35, engagement: 20, y: 0 },
  { id: 'F', risk: 39, engagement: 78, y: 1 },
  { id: 'G', risk: 45, engagement: 44, y: 0 },
  { id: 'H', risk: 49, engagement: 25, y: 1 },
  { id: 'I', risk: 52, engagement: 68, y: 1 },
  { id: 'J', risk: 58, engagement: 36, y: 1 },
  { id: 'K', risk: 63, engagement: 58, y: 1 },
  { id: 'L', risk: 68, engagement: 18, y: 1 },
  { id: 'M', risk: 72, engagement: 76, y: 1 },
  { id: 'N', risk: 78, engagement: 42, y: 1 },
  { id: 'O', risk: 83, engagement: 64, y: 1 },
  { id: 'P', risk: 89, engagement: 28, y: 1 },
]);

export const PRESETS = Object.freeze({
  balanced: {
    label: 'Balanced fit',
    detail: 'A useful linear separator with moderate probabilities.',
    weightRisk: 1.35,
    weightEngagement: -0.45,
    bias: 0.1,
    threshold: 0.5,
  },
  cautious: {
    label: 'Cautious positives',
    detail: 'A higher threshold reduces false alarms but can miss positives.',
    weightRisk: 1.35,
    weightEngagement: -0.45,
    bias: 0.1,
    threshold: 0.7,
  },
  underfit: {
    label: 'Underfit scores',
    detail: 'Small weights compress probabilities near 0.5.',
    weightRisk: 0.45,
    weightEngagement: -0.15,
    bias: 0,
    threshold: 0.5,
  },
});

export function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

export function logit(probability) {
  return Math.log(probability / (1 - probability));
}

export function scorePoint(point, weightRisk, weightEngagement, bias) {
  const centeredRisk = (point.risk - 50) / 18;
  const centeredEngagement = (point.engagement - 50) / 18;
  const z = weightRisk * centeredRisk + weightEngagement * centeredEngagement + bias;
  const probability = sigmoid(z);
  return { ...point, z, probability, predicted: probability >= 0.5 ? 1 : 0 };
}

export function classifyPoint(point, threshold) {
  return { ...point, predicted: point.probability >= threshold ? 1 : 0 };
}

export function summarize(scored) {
  return scored.reduce(
    (counts, point) => {
      if (point.y === 1 && point.predicted === 1) counts.tp += 1;
      if (point.y === 0 && point.predicted === 1) counts.fp += 1;
      if (point.y === 1 && point.predicted === 0) counts.fn += 1;
      if (point.y === 0 && point.predicted === 0) counts.tn += 1;
      return counts;
    },
    { tp: 0, fp: 0, fn: 0, tn: 0 },
  );
}

export function safeRatio(numerator, denominator) {
  return denominator === 0 ? 0 : numerator / denominator;
}

export function metricPercent(value) {
  return `${Math.round(value * 100)}%`;
}

export function boundaryLine(weightRisk, weightEngagement, bias, threshold) {
  const target = logit(threshold);
  const toSvgX = (risk) => 24 + risk * 3.12;
  const toSvgY = (engagement) => 336 - engagement * 3.12;

  if (Math.abs(weightRisk) < 0.05 && Math.abs(weightEngagement) < 0.05) {
    const x = toSvgX(50);
    return { x1: x, y1: 24, x2: x, y2: 336 };
  }

  if (Math.abs(weightEngagement) < 0.05) {
    const risk = 50 + ((target - bias) * 18) / weightRisk;
    const x = toSvgX(Math.max(0, Math.min(100, risk)));
    return { x1: x, y1: 24, x2: x, y2: 336 };
  }

  const yAt = (risk) => {
    const centeredRisk = (risk - 50) / 18;
    const centeredEngagement = (target - bias - weightRisk * centeredRisk) / weightEngagement;
    return 50 + centeredEngagement * 18;
  };

  const y0 = Math.max(-20, Math.min(120, yAt(0)));
  const y100 = Math.max(-20, Math.min(120, yAt(100)));
  return { x1: toSvgX(0), y1: toSvgY(y0), x2: toSvgX(100), y2: toSvgY(y100) };
}
