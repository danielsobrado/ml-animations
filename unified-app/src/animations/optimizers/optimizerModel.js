export const OPTIMIZERS = {
  sgd: { label: 'SGD', detail: 'Uses the current mini-batch gradient directly.' },
  momentum: { label: 'Momentum', detail: 'Builds velocity so repeated gradient directions compound.' },
  adam: { label: 'Adam', detail: 'Normalizes momentum by a running estimate of squared gradients.' },
};

export function loss([x, y]) {
  return 0.08 * (x + 3) ** 2 + 0.55 * (y - 1) ** 2;
}

export function trueGradient([x, y]) {
  return [0.16 * (x + 3), 1.1 * (y - 1)];
}

export function deterministicNoise(step, batchSize) {
  const scale = 0.42 / Math.sqrt(batchSize);
  return [
    Math.sin(step * 1.7 + batchSize * 0.11) * scale,
    Math.cos(step * 2.3 + batchSize * 0.07) * scale,
  ];
}

export function simulate({ optimizer, learningRate, momentum, batchSize, steps }) {
  let theta = [-4.8, 3.6];
  let velocity = [0, 0];
  let firstMoment = [0, 0];
  let secondMoment = [0, 0];
  const beta2 = 0.96;
  const path = [{ step: 0, theta, loss: loss(theta), grad: [0, 0] }];

  for (let step = 1; step <= steps; step += 1) {
    const exactGradient = trueGradient(theta);
    const noise = deterministicNoise(step, batchSize);
    const gradient = [exactGradient[0] + noise[0], exactGradient[1] + noise[1]];

    if (optimizer === 'momentum') {
      velocity = [
        momentum * velocity[0] + gradient[0],
        momentum * velocity[1] + gradient[1],
      ];
      theta = [
        theta[0] - learningRate * velocity[0],
        theta[1] - learningRate * velocity[1],
      ];
    } else if (optimizer === 'adam') {
      firstMoment = [
        momentum * firstMoment[0] + (1 - momentum) * gradient[0],
        momentum * firstMoment[1] + (1 - momentum) * gradient[1],
      ];
      secondMoment = [
        beta2 * secondMoment[0] + (1 - beta2) * gradient[0] ** 2,
        beta2 * secondMoment[1] + (1 - beta2) * gradient[1] ** 2,
      ];
      const correctedFirst = [
        firstMoment[0] / (1 - momentum ** step),
        firstMoment[1] / (1 - momentum ** step),
      ];
      const correctedSecond = [
        secondMoment[0] / (1 - beta2 ** step),
        secondMoment[1] / (1 - beta2 ** step),
      ];
      theta = [
        theta[0] - learningRate * correctedFirst[0] / (Math.sqrt(correctedSecond[0]) + 1e-6),
        theta[1] - learningRate * correctedFirst[1] / (Math.sqrt(correctedSecond[1]) + 1e-6),
      ];
    } else {
      theta = [
        theta[0] - learningRate * gradient[0],
        theta[1] - learningRate * gradient[1],
      ];
    }

    path.push({ step, theta, loss: loss(theta), grad: gradient });
  }

  return path;
}

export function project([x, y]) {
  return {
    cx: 60 + ((x + 5.5) / 5.5) * 420,
    cy: 320 - ((y + 0.5) / 4.5) * 260,
  };
}

export function lossColor(value) {
  if (value < 0.2) return '#ecfdf5';
  if (value < 0.6) return '#dbeafe';
  if (value < 1.4) return '#fef3c7';
  return '#fee2e2';
}
