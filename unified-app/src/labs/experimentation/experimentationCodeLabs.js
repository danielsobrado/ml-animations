export const EXPERIMENTATION_CODE_LABS = [
  {
    id: 'experiment-is-treated',
    stepLabel: '69.1',
    group: 'Treatment/control split',
    title: 'Identify treated user',
    concept: 'Experiments compare a treatment group against a control group.',
    objective: 'Return true when assignment equals "treatment".',
    difficulty: 'warmup',
    starterCode: `function isTreated(assignment) {
  // TODO: return whether this unit is in treatment.
  return false;
}`,
    testCode: `const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('treatment user', isTreated('treatment'), true);
check('control user', isTreated('control'), false);
check('other label', isTreated('holdout'), false);

return results;`,
    hints: [
      'Treatment is represented by the string "treatment".',
      'Use strict equality.',
      'return assignment === "treatment";',
    ],
    solution: `function isTreated(assignment) {
  return assignment === "treatment";
}`,
    explanation: 'A treatment indicator is the starting point for computing experiment outcomes by group.',
  },

  {
    id: 'experiment-count-treatment',
    stepLabel: '69.2',
    group: 'Treatment/control split',
    title: 'Count treatment units',
    concept: 'Before analyzing an experiment, check how many units landed in treatment.',
    objective: 'Count assignments equal to "treatment".',
    difficulty: 'core',
    starterCode: `function countTreatment(assignments) {
  let count = 0;

  for (let i = 0; i < assignments.length; i++) {
    // TODO: increment count for treatment assignment.
  }

  return count;
}`,
    testCode: `const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('mixed assignments', countTreatment(['treatment', 'control', 'treatment']), 2);
check('all control', countTreatment(['control', 'control']), 0);
check('all treatment', countTreatment(['treatment', 'treatment']), 2);

return results;`,
    hints: [
      'Check each assignment string.',
      'If assignments[i] === "treatment", add one.',
      'if (assignments[i] === "treatment") count += 1;',
    ],
    solution: `function countTreatment(assignments) {
  let count = 0;

  for (let i = 0; i < assignments.length; i++) {
    if (assignments[i] === "treatment") count += 1;
  }

  return count;
}`,
    explanation: 'Group counts help catch broken randomization or unexpected traffic allocation.',
  },

  {
    id: 'experiment-control-outcomes',
    stepLabel: '69.3',
    group: 'Treatment/control split',
    title: 'Collect control outcomes',
    concept: 'Control outcomes estimate what would happen without the intervention.',
    objective: 'Push outcomes whose matching assignment is "control".',
    difficulty: 'core',
    starterCode: `function controlOutcomes(assignments, outcomes) {
  const values = [];

  for (let i = 0; i < assignments.length; i++) {
    // TODO: collect outcomes for control units.
  }

  return values;
}`,
    testCode: `const results = [];

function sameArray(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({ name, actual: JSON.stringify(actual), expected: JSON.stringify(expected), passed: sameArray(actual, expected) });
}

check('mixed outcomes', controlOutcomes(['treatment', 'control', 'control'], [10, 20, 30]), [20, 30]);
check('no control', controlOutcomes(['treatment'], [10]), []);
check('all control', controlOutcomes(['control', 'control'], [1, 2]), [1, 2]);

return results;`,
    hints: [
      'Use the same index for assignments and outcomes.',
      'Control units have assignment "control".',
      'if (assignments[i] === "control") values.push(outcomes[i]);',
    ],
    solution: `function controlOutcomes(assignments, outcomes) {
  const values = [];

  for (let i = 0; i < assignments.length; i++) {
    if (assignments[i] === "control") values.push(outcomes[i]);
  }

  return values;
}`,
    explanation: 'Splitting outcomes by assignment is the first step toward estimating a treatment effect.',
  },

  {
    id: 'experiment-treatment-rate',
    stepLabel: '69.4',
    group: 'Treatment/control split',
    title: 'Treatment allocation rate',
    concept: 'The treatment rate is the fraction of units assigned to treatment.',
    objective: 'Return treatment count divided by total count.',
    difficulty: 'core',
    starterCode: `function treatmentRate(assignments) {
  let treated = 0;

  for (let i = 0; i < assignments.length; i++) {
    if (assignments[i] === "treatment") treated += 1;
  }

  // TODO: return the treatment allocation rate.
  return 0;
}`,
    testCode: `const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('half treated', treatmentRate(['treatment', 'control']), 0.5);
check('two thirds treated', treatmentRate(['treatment', 'control', 'treatment']), 2 / 3);
check('none treated', treatmentRate(['control', 'control']), 0);

return results;`,
    hints: [
      'treated is already counted.',
      'The denominator is assignments.length.',
      'return treated / assignments.length;',
    ],
    solution: `function treatmentRate(assignments) {
  let treated = 0;

  for (let i = 0; i < assignments.length; i++) {
    if (assignments[i] === "treatment") treated += 1;
  }

  return treated / assignments.length;
}`,
    explanation: 'A treatment allocation rate far from the planned split can signal assignment problems.',
  },

  {
    id: 'experiment-mean',
    stepLabel: '70.1',
    group: 'Difference in means',
    title: 'Mean outcome',
    concept: 'Difference-in-means starts by computing average outcome in each group.',
    objective: 'Return the average of values.',
    difficulty: 'warmup',
    starterCode: `function mean(values) {
  let total = 0;

  for (let i = 0; i < values.length; i++) {
    total += values[i];
  }

  // TODO: return average value.
  return total;
}`,
    testCode: `const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('mean [1,2,3]', mean([1, 2, 3]), 2);
check('mean [10,20]', mean([10, 20]), 15);
check('mean one value', mean([7]), 7);

return results;`,
    hints: [
      'Average means total divided by count.',
      'The count is values.length.',
      'return total / values.length;',
    ],
    solution: `function mean(values) {
  let total = 0;

  for (let i = 0; i < values.length; i++) {
    total += values[i];
  }

  return total / values.length;
}`,
    explanation: 'Group means summarize the outcome level for treatment and control.',
  },

  {
    id: 'experiment-difference-in-means',
    stepLabel: '70.2',
    group: 'Difference in means',
    title: 'Difference in means',
    concept: 'The simplest treatment effect estimate is treatment mean minus control mean.',
    objective: 'Return treatmentMean - controlMean.',
    difficulty: 'core',
    starterCode: `function differenceInMeans(treatmentMean, controlMean) {
  // TODO: return treatment minus control.
  return 0;
}`,
    testCode: `const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('positive lift', differenceInMeans(12, 10), 2);
check('negative lift', differenceInMeans(8, 10), -2);
check('no lift', differenceInMeans(10, 10), 0);

return results;`,
    hints: [
      'Treatment effect is treatment outcome minus control outcome.',
      'Keep the sign.',
      'return treatmentMean - controlMean;',
    ],
    solution: `function differenceInMeans(treatmentMean, controlMean) {
  return treatmentMean - controlMean;
}`,
    explanation: 'A positive difference means the treatment group had a higher average outcome.',
  },

  {
    id: 'experiment-group-mean',
    stepLabel: '70.3',
    group: 'Difference in means',
    title: 'Mean for one group',
    concept: 'Experiment analysis computes means conditional on assignment.',
    objective: 'Average outcomes whose assignment matches group.',
    difficulty: 'core',
    starterCode: `function groupMean(assignments, outcomes, group) {
  let total = 0;
  let count = 0;

  for (let i = 0; i < assignments.length; i++) {
    if (assignments[i] === group) {
      total += outcomes[i];
      count += 1;
    }
  }

  // TODO: return mean for this group.
  return 0;
}`,
    testCode: `const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('control mean', groupMean(['treatment', 'control', 'control'], [10, 20, 30], 'control'), 25);
check('treatment mean', groupMean(['treatment', 'control', 'treatment'], [10, 20, 40], 'treatment'), 25);
check('one unit mean', groupMean(['control'], [7], 'control'), 7);

return results;`,
    hints: [
      'total and count are already computed.',
      'Mean is total divided by count.',
      'return total / count;',
    ],
    solution: `function groupMean(assignments, outcomes, group) {
  let total = 0;
  let count = 0;

  for (let i = 0; i < assignments.length; i++) {
    if (assignments[i] === group) {
      total += outcomes[i];
      count += 1;
    }
  }

  return total / count;
}`,
    explanation: 'Conditional means let you compare treatment and control in one shared dataset.',
  },

  {
    id: 'experiment-ate-from-data',
    stepLabel: '70.4',
    group: 'Difference in means',
    title: 'ATE from experiment data',
    concept: 'A randomized experiment estimates average treatment effect by subtracting group means.',
    objective: 'Return treatment group mean minus control group mean.',
    difficulty: 'challenge',
    starterCode: `function groupMean(assignments, outcomes, group) {
  let total = 0;
  let count = 0;
  for (let i = 0; i < assignments.length; i++) {
    if (assignments[i] === group) {
      total += outcomes[i];
      count += 1;
    }
  }
  return total / count;
}

function averageTreatmentEffect(assignments, outcomes) {
  const treatmentMean = groupMean(assignments, outcomes, 'treatment');
  const controlMean = groupMean(assignments, outcomes, 'control');

  // TODO: return difference in means.
  return 0;
}`,
    testCode: `const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('positive effect', averageTreatmentEffect(['treatment', 'control', 'treatment', 'control'], [12, 10, 14, 8]), 4);
check('negative effect', averageTreatmentEffect(['treatment', 'control'], [7, 10]), -3);
check('zero effect', averageTreatmentEffect(['treatment', 'control'], [10, 10]), 0);

return results;`,
    hints: [
      'Both group means are already computed.',
      'ATE is treatmentMean - controlMean.',
      'return treatmentMean - controlMean;',
    ],
    solution: `function groupMean(assignments, outcomes, group) {
  let total = 0;
  let count = 0;
  for (let i = 0; i < assignments.length; i++) {
    if (assignments[i] === group) {
      total += outcomes[i];
      count += 1;
    }
  }
  return total / count;
}

function averageTreatmentEffect(assignments, outcomes) {
  const treatmentMean = groupMean(assignments, outcomes, 'treatment');
  const controlMean = groupMean(assignments, outcomes, 'control');
  return treatmentMean - controlMean;
}`,
    explanation: 'Randomization makes difference-in-means a credible estimate of causal effect.',
  },

  {
    id: 'experiment-sample-variance',
    stepLabel: '71.1',
    group: 'Standard error and confidence intervals',
    title: 'Sample variance',
    concept: 'Standard errors use sample variance to estimate outcome variability.',
    objective: 'Return sum of squared deviations divided by n - 1.',
    difficulty: 'core',
    starterCode: `function sampleVariance(values) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  let total = 0;

  for (let i = 0; i < values.length; i++) {
    const diff = values[i] - mean;
    // TODO: add squared deviation.
    total += 0;
  }

  return total / (values.length - 1);
}`,
    testCode: `const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('variance [1,2,3]', sampleVariance([1, 2, 3]), 1);
check('variance [10,20]', sampleVariance([10, 20]), 50);
check('constant values', sampleVariance([5, 5, 5]), 0);

return results;`,
    hints: [
      'diff is already centered.',
      'Squared deviation is diff * diff.',
      'total += diff * diff;',
    ],
    solution: `function sampleVariance(values) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  let total = 0;

  for (let i = 0; i < values.length; i++) {
    const diff = values[i] - mean;
    total += diff * diff;
  }

  return total / (values.length - 1);
}`,
    explanation: 'Sample variance estimates how noisy outcomes are around their mean.',
  },

  {
    id: 'experiment-standard-error-mean',
    stepLabel: '71.2',
    group: 'Standard error and confidence intervals',
    title: 'Standard error of mean',
    concept: 'The standard error of a mean shrinks as sample size grows.',
    objective: 'Return sqrt(variance / n).',
    difficulty: 'core',
    starterCode: `function standardErrorMean(variance, n) {
  // TODO: return standard error of one mean.
  return 0;
}`,
    testCode: `const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('variance 4 n 4', standardErrorMean(4, 4), 1);
check('variance 9 n 9', standardErrorMean(9, 9), 1);
check('variance 25 n 100', standardErrorMean(25, 100), 0.5);

return results;`,
    hints: [
      'Variance of a sample mean is variance / n.',
      'Standard error is the square root of that.',
      'return Math.sqrt(variance / n);',
    ],
    solution: `function standardErrorMean(variance, n) {
  return Math.sqrt(variance / n);
}`,
    explanation: 'More samples reduce uncertainty in the estimated mean.',
  },

  {
    id: 'experiment-standard-error-diff',
    stepLabel: '71.3',
    group: 'Standard error and confidence intervals',
    title: 'Standard error of difference',
    concept: 'For independent groups, variances of the two sample means add.',
    objective: 'Return sqrt(treatmentVariance / nTreatment + controlVariance / nControl).',
    difficulty: 'challenge',
    starterCode: `function standardErrorDifference(treatmentVariance, nTreatment, controlVariance, nControl) {
  // TODO: return standard error for difference in means.
  return 0;
}`,
    testCode: `const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('equal groups', standardErrorDifference(4, 4, 4, 4), Math.sqrt(2));
check('larger samples', standardErrorDifference(9, 9, 16, 16), Math.sqrt(2));
check('zero variance', standardErrorDifference(0, 10, 0, 10), 0);

return results;`,
    hints: [
      'Add variance / n for both groups.',
      'Then take Math.sqrt.',
      'return Math.sqrt(treatmentVariance / nTreatment + controlVariance / nControl);',
    ],
    solution: `function standardErrorDifference(treatmentVariance, nTreatment, controlVariance, nControl) {
  return Math.sqrt(treatmentVariance / nTreatment + controlVariance / nControl);
}`,
    explanation: 'The difference-in-means estimate is noisier when either group has high variance or low sample size.',
  },

  {
    id: 'experiment-confidence-interval',
    stepLabel: '71.4',
    group: 'Standard error and confidence intervals',
    title: 'Confidence interval bounds',
    concept: 'An approximate confidence interval is estimate plus or minus critical value times standard error.',
    objective: 'Return [estimate - z * se, estimate + z * se].',
    difficulty: 'core',
    starterCode: `function confidenceInterval(estimate, standardError, z = 1.96) {
  const margin = z * standardError;

  // TODO: return lower and upper bounds.
  return [];
}`,
    testCode: `const results = [];

function approxArray(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) <= tolerance);
}

function check(name, actual, expected) {
  results.push({ name, actual: JSON.stringify(actual), expected: JSON.stringify(expected), passed: approxArray(actual, expected) });
}

check('default z', confidenceInterval(10, 1), [8.04, 11.96]);
check('custom z', confidenceInterval(5, 2, 2), [1, 9]);
check('zero se', confidenceInterval(3, 0), [3, 3]);

return results;`,
    hints: [
      'The margin is already computed.',
      'Lower is estimate - margin, upper is estimate + margin.',
      'return [estimate - margin, estimate + margin];',
    ],
    solution: `function confidenceInterval(estimate, standardError, z = 1.96) {
  const margin = z * standardError;
  return [estimate - margin, estimate + margin];
}`,
    explanation: 'Confidence intervals communicate uncertainty around an estimated effect.',
  },

  {
    id: 'ab-z-statistic',
    stepLabel: '72.1',
    group: 'A/B test z-statistic',
    title: 'Z-statistic',
    concept: 'A z-statistic measures how many standard errors an estimate is away from zero.',
    objective: 'Return estimate / standardError.',
    difficulty: 'warmup',
    starterCode: `function zStatistic(estimate, standardError) {
  // TODO: return z-statistic.
  return 0;
}`,
    testCode: `const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('two standard errors', zStatistic(4, 2), 2);
check('negative estimate', zStatistic(-3, 1.5), -2);
check('zero estimate', zStatistic(0, 2), 0);

return results;`,
    hints: [
      'Divide effect estimate by its standard error.',
      'Keep the sign.',
      'return estimate / standardError;',
    ],
    solution: `function zStatistic(estimate, standardError) {
  return estimate / standardError;
}`,
    explanation: 'Large absolute z-statistics are less compatible with a zero-effect null hypothesis.',
  },

  {
    id: 'ab-significant-two-sided',
    stepLabel: '72.2',
    group: 'A/B test z-statistic',
    title: 'Two-sided significance',
    concept: 'A two-sided z-test flags effects far from zero in either direction.',
    objective: 'Return true when abs(z) exceeds critical value.',
    difficulty: 'core',
    starterCode: `function isSignificant(z, criticalValue = 1.96) {
  // TODO: compare absolute z with critical value.
  return false;
}`,
    testCode: `const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('large positive z', isSignificant(2.1), true);
check('large negative z', isSignificant(-2.1), true);
check('small z', isSignificant(1.5), false);
check('equal critical is not greater', isSignificant(1.96), false);

return results;`,
    hints: [
      'Use Math.abs(z).',
      'Compare with criticalValue.',
      'return Math.abs(z) > criticalValue;',
    ],
    solution: `function isSignificant(z, criticalValue = 1.96) {
  return Math.abs(z) > criticalValue;
}`,
    explanation: 'Two-sided tests detect changes in either direction.',
  },

  {
    id: 'ab-standard-error-proportion',
    stepLabel: '72.3',
    group: 'A/B test z-statistic',
    title: 'Proportion standard error',
    concept: 'Binary conversion-rate tests use p(1-p)/n variance for each group proportion.',
    objective: 'Return sqrt(p * (1 - p) / n).',
    difficulty: 'core',
    starterCode: `function proportionStandardError(p, n) {
  // TODO: return standard error for one conversion rate.
  return 0;
}`,
    testCode: `const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('p 0.5 n 100', proportionStandardError(0.5, 100), 0.05);
check('p 0.2 n 100', proportionStandardError(0.2, 100), 0.04);
check('p 0 n 100', proportionStandardError(0, 100), 0);

return results;`,
    hints: [
      'Proportion variance is p * (1 - p) / n.',
      'Take the square root.',
      'return Math.sqrt((p * (1 - p)) / n);',
    ],
    solution: `function proportionStandardError(p, n) {
  return Math.sqrt((p * (1 - p)) / n);
}`,
    explanation: 'Conversion-rate uncertainty is largest near 50% and smaller near 0% or 100%.',
  },

  {
    id: 'ab-conversion-z',
    stepLabel: '72.4',
    group: 'A/B test z-statistic',
    title: 'Conversion-rate z-statistic',
    concept: 'A/B conversion tests compare rate lift against the standard error of the difference.',
    objective: 'Return (treatmentRate - controlRate) / standard error.',
    difficulty: 'challenge',
    starterCode: `function conversionZ(treatmentRate, treatmentN, controlRate, controlN) {
  const se = Math.sqrt(
    (treatmentRate * (1 - treatmentRate)) / treatmentN +
    (controlRate * (1 - controlRate)) / controlN
  );

  // TODO: return z-statistic for conversion lift.
  return 0;
}`,
    testCode: `const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('positive lift', conversionZ(0.12, 1000, 0.10, 1000), (0.12 - 0.10) / Math.sqrt((0.12 * 0.88) / 1000 + (0.10 * 0.90) / 1000));
check('negative lift', conversionZ(0.08, 1000, 0.10, 1000), (0.08 - 0.10) / Math.sqrt((0.08 * 0.92) / 1000 + (0.10 * 0.90) / 1000));

return results;`,
    hints: [
      'The standard error is already computed as se.',
      'The lift is treatmentRate - controlRate.',
      'return (treatmentRate - controlRate) / se;',
    ],
    solution: `function conversionZ(treatmentRate, treatmentN, controlRate, controlN) {
  const se = Math.sqrt(
    (treatmentRate * (1 - treatmentRate)) / treatmentN +
    (controlRate * (1 - controlRate)) / controlN
  );

  return (treatmentRate - controlRate) / se;
}`,
    explanation: 'A conversion-rate z-statistic standardizes observed lift by its sampling uncertainty.',
  },

  {
    id: 'power-effect-to-noise',
    stepLabel: '73.1',
    group: 'Power and MDE intuition',
    title: 'Effect-to-noise ratio',
    concept: 'Power improves when the effect is large relative to standard error.',
    objective: 'Return effect / standardError.',
    difficulty: 'warmup',
    starterCode: `function effectToNoise(effect, standardError) {
  // TODO: return effect size in standard-error units.
  return 0;
}`,
    testCode: `const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('two se effect', effectToNoise(4, 2), 2);
check('half se effect', effectToNoise(1, 2), 0.5);
check('negative effect', effectToNoise(-3, 1.5), -2);

return results;`,
    hints: [
      'This is the same scaling idea as a z-statistic.',
      'Divide effect by standard error.',
      'return effect / standardError;',
    ],
    solution: `function effectToNoise(effect, standardError) {
  return effect / standardError;
}`,
    explanation: 'Small noisy effects are hard to detect reliably.',
  },

  {
    id: 'power-min-detectable-effect',
    stepLabel: '73.2',
    group: 'Power and MDE intuition',
    title: 'Minimum detectable effect',
    concept: 'A rough MDE multiplies standard error by the critical threshold needed for detection.',
    objective: 'Return multiplier * standardError.',
    difficulty: 'core',
    starterCode: `function minimumDetectableEffect(standardError, multiplier) {
  // TODO: return rough MDE.
  return 0;
}`,
    testCode: `const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('mde 2 se', minimumDetectableEffect(5, 2), 10);
check('mde 2.8 se', minimumDetectableEffect(10, 2.8), 28);
check('zero se', minimumDetectableEffect(0, 2), 0);

return results;`,
    hints: [
      'MDE is measured in outcome units.',
      'Multiply standardError by multiplier.',
      'return multiplier * standardError;',
    ],
    solution: `function minimumDetectableEffect(standardError, multiplier) {
  return multiplier * standardError;
}`,
    explanation: 'A smaller standard error lowers the effect size an experiment can reliably detect.',
  },

  {
    id: 'power-sample-size-scale',
    stepLabel: '73.3',
    group: 'Power and MDE intuition',
    title: 'Standard error from sample size',
    concept: 'For a fixed variance, standard error decreases with the square root of sample size.',
    objective: 'Return standardDeviation / sqrt(n).',
    difficulty: 'core',
    starterCode: `function standardErrorFromN(standardDeviation, n) {
  // TODO: return standard error from sample size.
  return 0;
}`,
    testCode: `const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('sd 10 n 100', standardErrorFromN(10, 100), 1);
check('sd 6 n 9', standardErrorFromN(6, 9), 2);
check('sd 5 n 25', standardErrorFromN(5, 25), 1);

return results;`,
    hints: [
      'Use Math.sqrt(n).',
      'Divide standard deviation by square root sample size.',
      'return standardDeviation / Math.sqrt(n);',
    ],
    solution: `function standardErrorFromN(standardDeviation, n) {
  return standardDeviation / Math.sqrt(n);
}`,
    explanation: 'Quadrupling sample size roughly halves standard error.',
  },

  {
    id: 'power-required-sample-size',
    stepLabel: '73.4',
    group: 'Power and MDE intuition',
    title: 'Required sample size intuition',
    concept: 'Required sample size grows with variance and shrinks with squared detectable effect.',
    objective: 'Return (multiplier * sd / mde)^2.',
    difficulty: 'challenge',
    starterCode: `function requiredSampleSize(standardDeviation, mde, multiplier) {
  // TODO: return rough sample size per group.
  return 0;
}`,
    testCode: `const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('basic size', requiredSampleSize(10, 2, 2), 100);
check('larger effect needs fewer samples', requiredSampleSize(10, 4, 2), 25);
check('larger sd needs more samples', requiredSampleSize(20, 2, 2), 400);

return results;`,
    hints: [
      'Compute multiplier * standardDeviation / mde.',
      'Then square it.',
      'return Math.pow((multiplier * standardDeviation) / mde, 2);',
    ],
    solution: `function requiredSampleSize(standardDeviation, mde, multiplier) {
  return Math.pow((multiplier * standardDeviation) / mde, 2);
}`,
    explanation: 'Detecting smaller effects requires much more data because sample size scales with one over MDE squared.',
  },

  {
    id: 'cuped-residual',
    stepLabel: '74.1',
    group: 'CUPED adjustment',
    title: 'CUPED residual',
    concept: 'CUPED removes predictable variation using a pre-experiment covariate.',
    objective: 'Return outcome - theta * covariate.',
    difficulty: 'warmup',
    starterCode: `function cupedResidual(outcome, covariate, theta) {
  // TODO: subtract theta times covariate.
  return outcome;
}`,
    testCode: `const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('simple residual', cupedResidual(10, 3, 2), 4);
check('zero theta', cupedResidual(10, 3, 0), 10);
check('negative covariate', cupedResidual(10, -2, 3), 16);

return results;`,
    hints: [
      'Adjustment is theta * covariate.',
      'Subtract the adjustment from outcome.',
      'return outcome - theta * covariate;',
    ],
    solution: `function cupedResidual(outcome, covariate, theta) {
  return outcome - theta * covariate;
}`,
    explanation: 'CUPED lowers variance by accounting for pre-existing outcome predictors.',
  },

  {
    id: 'cuped-theta',
    stepLabel: '74.2',
    group: 'CUPED adjustment',
    title: 'CUPED theta',
    concept: 'The CUPED coefficient is covariance(outcome, covariate) divided by variance(covariate).',
    objective: 'Return covariance / variance.',
    difficulty: 'core',
    starterCode: `function cupedTheta(covariance, covariateVariance) {
  // TODO: return CUPED coefficient.
  return 0;
}`,
    testCode: `const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('theta 2', cupedTheta(10, 5), 2);
check('theta half', cupedTheta(3, 6), 0.5);
check('zero covariance', cupedTheta(0, 5), 0);

return results;`,
    hints: [
      'Theta is a regression-style slope.',
      'Divide covariance by covariate variance.',
      'return covariance / covariateVariance;',
    ],
    solution: `function cupedTheta(covariance, covariateVariance) {
  return covariance / covariateVariance;
}`,
    explanation: 'A stronger covariate-outcome relationship gives CUPED more variance reduction potential.',
  },

  {
    id: 'cuped-adjust-vector',
    stepLabel: '74.3',
    group: 'CUPED adjustment',
    title: 'Adjust outcome vector',
    concept: 'CUPED applies the same residualization formula to every unit.',
    objective: 'Push outcomes[i] - theta * covariates[i].',
    difficulty: 'core',
    starterCode: `function cupedAdjust(outcomes, covariates, theta) {
  const adjusted = [];

  for (let i = 0; i < outcomes.length; i++) {
    // TODO: push CUPED-adjusted outcome.
    adjusted.push(outcomes[i]);
  }

  return adjusted;
}`,
    testCode: `const results = [];

function sameArray(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({ name, actual: JSON.stringify(actual), expected: JSON.stringify(expected), passed: sameArray(actual, expected) });
}

check('adjust two values', cupedAdjust([10, 20], [3, 4], 2), [4, 12]);
check('zero theta', cupedAdjust([10, 20], [3, 4], 0), [10, 20]);
check('negative covariate', cupedAdjust([10], [-2], 3), [16]);

return results;`,
    hints: [
      'Use matching outcome and covariate coordinates.',
      'Subtract theta * covariates[i].',
      'adjusted.push(outcomes[i] - theta * covariates[i]);',
    ],
    solution: `function cupedAdjust(outcomes, covariates, theta) {
  const adjusted = [];

  for (let i = 0; i < outcomes.length; i++) {
    adjusted.push(outcomes[i] - theta * covariates[i]);
  }

  return adjusted;
}`,
    explanation: 'After adjustment, the experiment can compare adjusted outcomes instead of raw outcomes.',
  },

  {
    id: 'cuped-centered-adjustment',
    stepLabel: '74.4',
    group: 'CUPED adjustment',
    title: 'Centered CUPED adjustment',
    concept: 'CUPED usually centers the covariate so the adjusted outcome remains on the original scale.',
    objective: 'Return outcome - theta * (covariate - covariateMean).',
    difficulty: 'challenge',
    starterCode: `function centeredCupedOutcome(outcome, covariate, covariateMean, theta) {
  // TODO: apply centered CUPED adjustment.
  return outcome;
}`,
    testCode: `const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('above mean covariate', centeredCupedOutcome(10, 5, 3, 2), 6);
check('at mean covariate', centeredCupedOutcome(10, 3, 3, 2), 10);
check('below mean covariate', centeredCupedOutcome(10, 1, 3, 2), 14);

return results;`,
    hints: [
      'First center the covariate: covariate - covariateMean.',
      'Then subtract theta times the centered covariate.',
      'return outcome - theta * (covariate - covariateMean);',
    ],
    solution: `function centeredCupedOutcome(outcome, covariate, covariateMean, theta) {
  return outcome - theta * (covariate - covariateMean);
}`,
    explanation: 'Centering preserves the average scale while reducing variance from predictable pre-period differences.',
  },

  {
    id: 'propensity-inverse-weight',
    stepLabel: '75.1',
    group: 'Propensity score weighting',
    title: 'Inverse propensity weight',
    concept: 'Propensity weighting upweights units that were unlikely to receive their observed assignment.',
    objective: 'Return 1 / propensity.',
    difficulty: 'warmup',
    starterCode: `function inversePropensityWeight(propensity) {
  // TODO: return inverse propensity weight.
  return 0;
}`,
    testCode: `const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('propensity half', inversePropensityWeight(0.5), 2);
check('propensity quarter', inversePropensityWeight(0.25), 4);
check('propensity one', inversePropensityWeight(1), 1);

return results;`,
    hints: [
      'Inverse means reciprocal.',
      'Use 1 / propensity.',
      'return 1 / propensity;',
    ],
    solution: `function inversePropensityWeight(propensity) {
  return 1 / propensity;
}`,
    explanation: 'Inverse propensity weights compensate for unequal assignment probabilities.',
  },

  {
    id: 'propensity-observed-weight',
    stepLabel: '75.2',
    group: 'Propensity score weighting',
    title: 'Observed assignment weight',
    concept: 'Treated units use 1 / p, control units use 1 / (1 - p).',
    objective: 'Return the inverse probability of the observed assignment.',
    difficulty: 'core',
    starterCode: `function observedAssignmentWeight(treated, propensity) {
  // TODO: return treated or control inverse probability weight.
  return 0;
}`,
    testCode: `const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('treated p half', observedAssignmentWeight(true, 0.5), 2);
check('control p half', observedAssignmentWeight(false, 0.5), 2);
check('control p 0.2', observedAssignmentWeight(false, 0.2), 1.25);

return results;`,
    hints: [
      'If treated, use 1 / propensity.',
      'If control, use 1 / (1 - propensity).',
      'return treated ? 1 / propensity : 1 / (1 - propensity);',
    ],
    solution: `function observedAssignmentWeight(treated, propensity) {
  return treated ? 1 / propensity : 1 / (1 - propensity);
}`,
    explanation: 'Observed-assignment weights make underrepresented assignment paths count more.',
  },

  {
    id: 'propensity-weighted-outcome',
    stepLabel: '75.3',
    group: 'Propensity score weighting',
    title: 'Weighted outcome',
    concept: 'Weighted estimators multiply each outcome by its inverse-propensity weight.',
    objective: 'Return outcome * weight.',
    difficulty: 'warmup',
    starterCode: `function weightedOutcome(outcome, weight) {
  // TODO: return weighted outcome.
  return outcome;
}`,
    testCode: `const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('double outcome', weightedOutcome(10, 2), 20);
check('zero outcome', weightedOutcome(0, 5), 0);
check('fractional weight', weightedOutcome(10, 0.5), 5);

return results;`,
    hints: [
      'Weighted outcome is a product.',
      'Multiply outcome by weight.',
      'return outcome * weight;',
    ],
    solution: `function weightedOutcome(outcome, weight) {
  return outcome * weight;
}`,
    explanation: 'Weighting changes how much each observed unit contributes to the estimator.',
  },

  {
    id: 'propensity-weighted-mean',
    stepLabel: '75.4',
    group: 'Propensity score weighting',
    title: 'Weighted mean',
    concept: 'A weighted mean divides weighted outcome sum by total weight.',
    objective: 'Return sum(outcome * weight) / sum(weight).',
    difficulty: 'challenge',
    starterCode: `function weightedMean(outcomes, weights) {
  let weightedTotal = 0;
  let weightTotal = 0;

  for (let i = 0; i < outcomes.length; i++) {
    weightedTotal += outcomes[i] * weights[i];
    weightTotal += weights[i];
  }

  // TODO: return weighted mean.
  return 0;
}`,
    testCode: `const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('equal weights', weightedMean([10, 20], [1, 1]), 15);
check('heavier first', weightedMean([10, 20], [3, 1]), 12.5);
check('one value', weightedMean([7], [10]), 7);

return results;`,
    hints: [
      'Both totals are already computed.',
      'Weighted mean is weightedTotal / weightTotal.',
      'return weightedTotal / weightTotal;',
    ],
    solution: `function weightedMean(outcomes, weights) {
  let weightedTotal = 0;
  let weightTotal = 0;

  for (let i = 0; i < outcomes.length; i++) {
    weightedTotal += outcomes[i] * weights[i];
    weightTotal += weights[i];
  }

  return weightedTotal / weightTotal;
}`,
    explanation: 'Propensity weighting estimates group means after correcting for assignment imbalance.',
  },

  {
    id: 'dag-has-edge',
    stepLabel: '76.1',
    group: 'DAG adjustment-set checks',
    title: 'Check DAG edge',
    concept: 'A DAG encodes causal assumptions as directed edges.',
    objective: 'Return true when an edge exists from fromNode to toNode.',
    difficulty: 'warmup',
    starterCode: `function hasEdge(edges, fromNode, toNode) {
  // TODO: return whether edges contains [fromNode, toNode].
  return false;
}`,
    testCode: `const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

const edges = [['X', 'Y'], ['Z', 'X']];

check('edge exists', hasEdge(edges, 'X', 'Y'), true);
check('reverse edge missing', hasEdge(edges, 'Y', 'X'), false);
check('different edge missing', hasEdge(edges, 'Z', 'Y'), false);

return results;`,
    hints: [
      'Loop through edge pairs.',
      'Check edge[0] and edge[1].',
      'if (edges[i][0] === fromNode && edges[i][1] === toNode) return true;',
    ],
    solution: `function hasEdge(edges, fromNode, toNode) {
  for (let i = 0; i < edges.length; i++) {
    if (edges[i][0] === fromNode && edges[i][1] === toNode) return true;
  }

  return false;
}`,
    explanation: 'DAG logic starts with knowing which direct causal arrows are present.',
  },

  {
    id: 'dag-is-parent',
    stepLabel: '76.2',
    group: 'DAG adjustment-set checks',
    title: 'Parent node check',
    concept: 'A parent of a node has a directed edge into that node.',
    objective: 'Return true when candidate -> node exists.',
    difficulty: 'core',
    starterCode: `function isParent(edges, candidate, node) {
  for (let i = 0; i < edges.length; i++) {
    const edge = edges[i];

    // TODO: return true if candidate points into node.
  }

  return false;
}`,
    testCode: `const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

const edges = [['Z', 'X'], ['X', 'Y'], ['W', 'Y']];

check('Z parent of X', isParent(edges, 'Z', 'X'), true);
check('W parent of Y', isParent(edges, 'W', 'Y'), true);
check('Y not parent of X', isParent(edges, 'Y', 'X'), false);

return results;`,
    hints: [
      'A parent edge is candidate -> node.',
      'Check edge[0] and edge[1].',
      'if (edge[0] === candidate && edge[1] === node) return true;',
    ],
    solution: `function isParent(edges, candidate, node) {
  for (let i = 0; i < edges.length; i++) {
    const edge = edges[i];

    if (edge[0] === candidate && edge[1] === node) return true;
  }

  return false;
}`,
    explanation: 'Parents are direct causes in the graph, according to the DAG assumptions.',
  },

  {
    id: 'dag-backdoor-candidate',
    stepLabel: '76.3',
    group: 'DAG adjustment-set checks',
    title: 'Backdoor candidate',
    concept: 'A common confounder is a variable that points into both treatment and outcome.',
    objective: 'Return true when z is parent of both treatment and outcome.',
    difficulty: 'challenge',
    starterCode: `function isParent(edges, candidate, node) {
  return edges.some((edge) => edge[0] === candidate && edge[1] === node);
}

function isCommonCause(edges, z, treatment, outcome) {
  // TODO: return whether z points into treatment and outcome.
  return false;
}`,
    testCode: `const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

const edges = [['Z', 'X'], ['Z', 'Y'], ['X', 'Y'], ['W', 'X']];

check('Z common cause', isCommonCause(edges, 'Z', 'X', 'Y'), true);
check('W not common cause', isCommonCause(edges, 'W', 'X', 'Y'), false);
check('X not common cause of itself and Y', isCommonCause(edges, 'X', 'X', 'Y'), false);

return results;`,
    hints: [
      'Use the isParent helper twice.',
      'z must point into both treatment and outcome.',
      'return isParent(edges, z, treatment) && isParent(edges, z, outcome);',
    ],
    solution: `function isParent(edges, candidate, node) {
  return edges.some((edge) => edge[0] === candidate && edge[1] === node);
}

function isCommonCause(edges, z, treatment, outcome) {
  return isParent(edges, z, treatment) && isParent(edges, z, outcome);
}`,
    explanation: 'Common causes are typical variables to consider adjusting for in backdoor paths.',
  },

  {
    id: 'dag-adjustment-set-covers-confounders',
    stepLabel: '76.4',
    group: 'DAG adjustment-set checks',
    title: 'Adjustment set covers confounders',
    concept: 'A basic adjustment check asks whether all known confounders are included.',
    objective: 'Return true when every confounder is in adjustmentSet.',
    difficulty: 'core',
    starterCode: `function coversConfounders(adjustmentSet, confounders) {
  for (let i = 0; i < confounders.length; i++) {
    // TODO: return false if a confounder is missing.
  }

  return true;
}`,
    testCode: `const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('covers all', coversConfounders(['Z', 'W'], ['Z', 'W']), true);
check('missing one', coversConfounders(['Z'], ['Z', 'W']), false);
check('no confounders', coversConfounders([], []), true);

return results;`,
    hints: [
      'Use adjustmentSet.includes(confounders[i]).',
      'If one is missing, return false.',
      'if (!adjustmentSet.includes(confounders[i])) return false;',
    ],
    solution: `function coversConfounders(adjustmentSet, confounders) {
  for (let i = 0; i < confounders.length; i++) {
    if (!adjustmentSet.includes(confounders[i])) return false;
  }

  return true;
}`,
    explanation: 'This toy check is not full d-separation, but it reinforces the adjustment-set idea.',
  },
];
