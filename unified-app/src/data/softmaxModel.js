export function computeSoftmax(logits, temperature = 1) {
  const safeTemperature = Math.max(Number(temperature) || 1, 1e-6);
  const scaled = logits.map((value) => value / safeTemperature);
  const maxLogit = Math.max(...scaled);
  const exps = scaled.map((value) => Math.exp(value - maxLogit));
  const total = exps.reduce((sum, value) => sum + value, 0);

  return exps.map((value) => value / total);
}

export function nudgeLogit(logits, index, delta) {
  return logits.map((value, currentIndex) => (
    currentIndex === index ? value + delta : value
  ));
}

export function softmaxMetrics(probabilities) {
  const entropy = -probabilities.reduce((sum, probability) => (
    probability > 0 ? sum + probability * Math.log2(probability) : sum
  ), 0);
  const maxProbability = Math.max(...probabilities);
  const margin = [...probabilities]
    .sort((a, b) => b - a)
    .slice(0, 2)
    .reduce((difference, value, index) => (index === 0 ? value : difference - value), 0);

  return {
    entropy,
    maxProbability,
    margin,
  };
}

export function classifySoftmaxSharpness(probabilities) {
  const { entropy, maxProbability } = softmaxMetrics(probabilities);

  if (maxProbability >= 0.8) {
    return {
      tone: 'sharp',
      label: 'Sharp',
      description: 'Most probability mass is concentrated on one class.',
    };
  }

  if (entropy >= Math.log2(probabilities.length) * 0.9) {
    return {
      tone: 'diffuse',
      label: 'Diffuse',
      description: 'Probability mass is spread fairly evenly across classes.',
    };
  }

  return {
    tone: 'balanced',
    label: 'Balanced',
    description: 'The distribution has a leader but still keeps alternatives alive.',
  };
}
