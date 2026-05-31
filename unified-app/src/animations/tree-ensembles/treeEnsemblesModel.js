export const POINTS = [
  { x: 0.12, y: 0.22, label: 0 },
  { x: 0.18, y: 0.36, label: 0 },
  { x: 0.25, y: 0.64, label: 0 },
  { x: 0.31, y: 0.79, label: 1 },
  { x: 0.42, y: 0.28, label: 0 },
  { x: 0.48, y: 0.58, label: 1 },
  { x: 0.54, y: 0.74, label: 1 },
  { x: 0.60, y: 0.34, label: 0 },
  { x: 0.67, y: 0.49, label: 1 },
  { x: 0.73, y: 0.71, label: 1 },
  { x: 0.81, y: 0.28, label: 1 },
  { x: 0.88, y: 0.54, label: 1 },
];

export const FOREST_RULES = [
  { feature: 'x', threshold: 0.52, polarity: 1 },
  { feature: 'y', threshold: 0.46, polarity: 1 },
  { feature: 'x', threshold: 0.74, polarity: -1 },
  { feature: 'y', threshold: 0.70, polarity: 1 },
  { feature: 'x', threshold: 0.35, polarity: 1 },
  { feature: 'y', threshold: 0.31, polarity: 1 },
  { feature: 'x', threshold: 0.62, polarity: 1 },
];

export const BOOSTING_STEPS = [
  { rule: 'x > 0.50', contribution: 0.42 },
  { rule: 'y > 0.55', contribution: 0.30 },
  { rule: 'x > 0.75', contribution: 0.18 },
  { rule: 'y < 0.32', contribution: -0.16 },
  { rule: 'x < 0.28', contribution: -0.14 },
];

export function predictTree(point, depth) {
  if (point.x < 0.52) {
    if (depth === 1) return 0;
    return point.y > 0.68 ? 1 : 0;
  }

  if (depth === 1) return 1;
  if (depth === 2) return point.y > 0.42 ? 1 : 0;
  return point.y > 0.42 || point.x > 0.78 ? 1 : 0;
}

export function ruleVote(point, rule) {
  const raw = point[rule.feature] >= rule.threshold ? 1 : 0;
  return rule.polarity === 1 ? raw : 1 - raw;
}

export function forestPrediction(point, treeCount) {
  const votes = FOREST_RULES.slice(0, treeCount).map((rule) => ruleVote(point, rule));
  const positiveVotes = votes.filter(Boolean).length;
  return {
    votes,
    positiveVotes,
    probability: positiveVotes / votes.length,
    label: positiveVotes >= Math.ceil(votes.length / 2) ? 1 : 0,
  };
}

export function boostedScore(point, rounds, learningRate) {
  let score = -0.15;
  const steps = BOOSTING_STEPS.slice(0, rounds).map((step) => {
    const matched =
      (step.rule === 'x > 0.50' && point.x > 0.5) ||
      (step.rule === 'y > 0.55' && point.y > 0.55) ||
      (step.rule === 'x > 0.75' && point.x > 0.75) ||
      (step.rule === 'y < 0.32' && point.y < 0.32) ||
      (step.rule === 'x < 0.28' && point.x < 0.28);
    const delta = matched ? step.contribution * learningRate : 0;
    score += delta;
    return { ...step, matched, delta, score };
  });
  return { score, probability: 1 / (1 + Math.exp(-score * 2.4)), steps };
}

export function accuracy(depth) {
  const correct = POINTS.filter((point) => predictTree(point, depth) === point.label).length;
  return correct / POINTS.length;
}

export function toScreen(point) {
  return [32 + point.x * 296, 328 - point.y * 296];
}
