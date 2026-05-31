export const DEFAULT_LEARNING_RATE = 0.1;
export const DEFAULT_START_WEIGHT = 4;

export function loss(weight) {
  return weight * weight;
}

export function gradient(weight) {
  return 2 * weight;
}

export function nextWeight(weight, learningRate) {
  return weight - learningRate * gradient(weight);
}

export function learningRateStatus(learningRate) {
  if (learningRate < 0.05) return { text: 'Too slow', color: 'text-yellow-600' };
  if (learningRate > 0.5) return { text: 'May diverge!', color: 'text-red-600' };
  return { text: 'Good', color: 'text-green-600' };
}
