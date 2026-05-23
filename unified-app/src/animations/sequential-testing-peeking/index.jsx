import React from 'react';
import CausalConceptLesson from '../_shared/CausalConceptLesson';

const pct = (value) => `${value.toFixed(0)}%`;
const num = (value) => value.toLocaleString();

const config = {
  lessonId: 'sequential-testing-peeking',
  kicker: 'Experiment monitoring',
  title: 'Sequential Testing & Peeking',
  description: 'Repeatedly checking a fixed-horizon p-value gives noise many chances to look significant. Sequential testing plans the looks, stopping rules, and error budget before data arrives.',
  controls: [
    { id: 'looks', label: 'Interim looks', min: 1, max: 20, step: 1, defaultValue: 8, format: num, help: 'Each extra look gives a null result another chance to cross a naive threshold.' },
    { id: 'alpha', label: 'Nominal alpha', min: 1, max: 10, step: 1, defaultValue: 5, format: pct, help: 'The fixed-horizon false positive threshold used at every look.' },
    { id: 'spend', label: 'Alpha spent by design', min: 1, max: 10, step: 1, defaultValue: 5, format: pct, help: 'Sequential designs spend the total error budget across looks.' },
  ],
  compute(values) {
    const naiveFalsePositive = 1 - (1 - values.alpha / 100) ** values.looks;
    const sequentialFalsePositive = values.spend / 100;
    const inflation = naiveFalsePositive / sequentialFalsePositive;
    const boundary = Math.min(25, values.spend / Math.sqrt(values.looks));
    return {
      stats: [
        { label: 'Naive false positive', value: pct(naiveFalsePositive * 100), detail: 'Any look crosses alpha', tone: naiveFalsePositive > 0.12 ? 'rose' : 'amber' },
        { label: 'Planned error', value: pct(sequentialFalsePositive * 100), detail: 'Total alpha budget', tone: 'cyan' },
        { label: 'Inflation', value: `${inflation.toFixed(1)}x`, detail: 'Naive vs planned', tone: inflation > 2 ? 'rose' : 'amber' },
        { label: 'Early boundary', value: pct(boundary), detail: 'Illustrative stricter look', tone: 'emerald' },
      ],
      bars: [
        { label: 'Naive repeated peeking risk', value: pct(naiveFalsePositive * 100), width: naiveFalsePositive * 100, color: 'bg-rose-500' },
        { label: 'Sequential design risk', value: pct(sequentialFalsePositive * 100), width: sequentialFalsePositive * 100, color: 'bg-emerald-500' },
        { label: 'Single fixed-horizon risk', value: pct(values.alpha), width: values.alpha, color: 'bg-cyan-500' },
      ],
      formulaLines: [
        `naive any-look risk = 1 - (1 - alpha)^looks = ${(naiveFalsePositive * 100).toFixed(1)}%`,
        `planned sequential risk = alpha budget = ${values.spend}%`,
        `early looks need stricter boundaries than ${values.alpha}%`,
      ],
      readout: 'A dashboard that stops as soon as p < 0.05 is running many hidden tests. That changes the false positive rate.',
      steps: [
        { title: 'Declare looks', pass: values.looks <= 5, body: values.looks <= 5 ? 'A small number of planned analyses is easier to control.' : 'Many unplanned looks make a naive p-value threshold unreliable.' },
        { title: 'Spend alpha', pass: values.spend <= values.alpha, body: 'Sequential methods allocate the total Type I error budget across interim and final analyses.' },
        { title: 'Use stopping rules', pass: inflation < 2, body: inflation < 2 ? 'The false positive inflation is contained in this scenario.' : 'Naive peeking more than doubles the declared false positive risk.' },
      ],
      takeaway: 'If the team wants to monitor early, use a sequential design. If not, wait for the fixed horizon before making the decision.',
    };
  },
};

export default function SequentialTestingPeekingAnimation() {
  return <CausalConceptLesson config={config} />;
}
