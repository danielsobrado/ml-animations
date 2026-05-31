export const SCORED_EXAMPLES = Object.freeze([
  { id: 1, score: 0.96, label: 1 },
  { id: 2, score: 0.91, label: 1 },
  { id: 3, score: 0.86, label: 0 },
  { id: 4, score: 0.78, label: 1 },
  { id: 5, score: 0.72, label: 0 },
  { id: 6, score: 0.67, label: 1 },
  { id: 7, score: 0.61, label: 0 },
  { id: 8, score: 0.55, label: 1 },
  { id: 9, score: 0.48, label: 0 },
  { id: 10, score: 0.43, label: 0 },
  { id: 11, score: 0.37, label: 1 },
  { id: 12, score: 0.31, label: 0 },
  { id: 13, score: 0.25, label: 0 },
  { id: 14, score: 0.18, label: 1 },
  { id: 15, score: 0.12, label: 0 },
  { id: 16, score: 0.06, label: 0 },
]);

export const THRESHOLDS = Object.freeze(Array.from({ length: 21 }, (_, index) => index / 20));

export function confusionAt(threshold, examples = SCORED_EXAMPLES) {
  return examples.reduce((counts, example) => {
    const predicted = example.score >= threshold ? 1 : 0;
    if (predicted === 1 && example.label === 1) counts.tp += 1;
    if (predicted === 1 && example.label === 0) counts.fp += 1;
    if (predicted === 0 && example.label === 1) counts.fn += 1;
    if (predicted === 0 && example.label === 0) counts.tn += 1;
    return counts;
  }, { tp: 0, fp: 0, fn: 0, tn: 0 });
}

function ratio(numerator, denominator) {
  return denominator === 0 ? null : numerator / denominator;
}

export function metrics(counts) {
  const predictedPositives = counts.tp + counts.fp;
  const actualPositives = counts.tp + counts.fn;
  const actualNegatives = counts.fp + counts.tn;
  const precision = ratio(counts.tp, predictedPositives);
  const recall = ratio(counts.tp, actualPositives);
  const fpr = ratio(counts.fp, actualNegatives);
  const specificity = ratio(counts.tn, actualNegatives);

  return {
    precision,
    recall,
    fpr,
    specificity,
    tpr: recall,
    predictedPositives,
    actualPositives,
    actualNegatives,
  };
}

export function prPrecisionForPlot(point) {
  return point.precision ?? 1;
}

export function metricPercent(value) {
  return value === null ? 'N/A' : `${Math.round(value * 100)}%`;
}

export function curvePoints(thresholds = THRESHOLDS) {
  return thresholds
    .map((threshold) => {
      const counts = confusionAt(threshold);
      const summary = metrics(counts);
      return {
        threshold,
        ...counts,
        ...summary,
        precisionPlot: prPrecisionForPlot(summary),
      };
    })
    .sort((a, b) => (a.fpr ?? 0) - (b.fpr ?? 0) || (a.recall ?? 0) - (b.recall ?? 0));
}
