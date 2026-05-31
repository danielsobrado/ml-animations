export const SAMPLE_LEVELS = {
  small: { label: 'Small sample', count: 10, varianceRelief: 0.2, detail: 'Few examples make flexible models unstable.' },
  medium: { label: 'Medium sample', count: 22, varianceRelief: 0.45, detail: 'More examples reduce the penalty for flexibility.' },
  large: { label: 'Large sample', count: 42, varianceRelief: 0.7, detail: 'Many examples make variance easier to control.' },
};

export const MODEL_TYPES = {
  simple: { label: 'Simple', complexity: 1, detail: 'High bias: the model misses curved signal.' },
  balanced: { label: 'Balanced', complexity: 2, detail: 'Lower bias without chasing most sample noise.' },
  flexible: { label: 'Flexible', complexity: 3, detail: 'Low bias, but higher variance when data is noisy or scarce.' },
};

export function truth(x) {
  return 42 + 28 * Math.sin((x - 8) / 11) + 0.72 * x;
}

export function pseudoNoise(index) {
  const raw = Math.sin(index * 12.9898) * 43758.5453;
  return raw - Math.floor(raw);
}

export function makePoints(sampleLevel, noise) {
  const { count } = SAMPLE_LEVELS[sampleLevel];
  return Array.from({ length: count }, (_, index) => {
    const x = 4 + (index / Math.max(1, count - 1)) * 92;
    const centeredNoise = pseudoNoise(index + count * 3) - 0.5;
    return {
      id: index,
      x,
      y: truth(x) + centeredNoise * noise * 30,
    };
  });
}

export function predict(x, model, noise) {
  if (model === 'simple') {
    return 51 + 0.36 * x;
  }
  if (model === 'balanced') {
    return 42 + 23 * Math.sin((x - 8) / 13) + 0.58 * x;
  }
  return truth(x) + Math.sin(x * 0.55) * noise * 8 + Math.sin(x * 1.3) * noise * 3;
}

export function errorProfile(model, sampleLevel, noise) {
  const complexity = MODEL_TYPES[model].complexity;
  const varianceRelief = SAMPLE_LEVELS[sampleLevel].varianceRelief;
  const bias = model === 'simple' ? 34 : model === 'balanced' ? 13 : 6;
  const variance = model === 'simple'
    ? 5 + noise * 2
    : model === 'balanced'
      ? 9 + noise * 8 - varianceRelief * 5
      : 18 + noise * 20 - varianceRelief * 14;
  const irreducible = 8 + noise * 10;
  const train = Math.max(6, bias * 0.55 + variance * 0.22 + irreducible * 0.5 - complexity * 5);
  const validation = Math.max(8, bias + variance + irreducible);
  return {
    bias,
    variance: Math.max(4, variance),
    irreducible,
    train,
    validation,
    gap: Math.max(0, validation - train),
  };
}

export function recommendationForProfile(profile) {
  if (profile.bias > profile.variance) {
    return 'The model is underfitting: add useful flexibility or better features.';
  }
  if (profile.variance > profile.bias + 10) {
    return 'The model is variance-heavy: add data, regularize, simplify, or use averaging.';
  }
  return 'Bias and variance are reasonably balanced for this teaching setup.';
}

export function project(point) {
  return {
    cx: 34 + (point.x / 100) * 332,
    cy: 262 - ((point.y - 15) / 95) * 226,
  };
}

export function curvePath(model, noise, source = predict) {
  return Array.from({ length: 70 }, (_, index) => {
    const x = (index / 69) * 100;
    const y = source === truth ? truth(x) : predict(x, model, noise);
    const { cx, cy } = project({ x, y });
    return `${index === 0 ? 'M' : 'L'} ${cx.toFixed(1)} ${cy.toFixed(1)}`;
  }).join(' ');
}
