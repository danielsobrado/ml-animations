import React from 'react';
import CausalConceptLesson from '../_shared/CausalConceptLesson';

const pct = (value) => `${value.toFixed(0)}%`;
const one = (value) => value.toFixed(1);

const config = {
  lessonId: 'cuped-variance-reduction',
  kicker: 'Experiment precision',
  title: 'CUPED / Variance Reduction',
  description: 'CUPED uses pre-treatment behavior to remove predictable baseline variation from the outcome, narrowing confidence intervals without changing the treatment effect target.',
  controls: [
    { id: 'correlation', label: 'Pre/post correlation', min: 0, max: 95, step: 5, defaultValue: 60, format: pct, help: 'Higher correlation means the covariate explains more outcome noise.' },
    { id: 'rawSe', label: 'Raw standard error', min: 1, max: 10, step: 0.5, defaultValue: 4, format: one, help: 'The original uncertainty before adjustment.' },
    { id: 'effect', label: 'Observed lift', min: 1, max: 12, step: 0.5, defaultValue: 5, format: one, help: 'The treatment-control difference being estimated.' },
  ],
  compute(values) {
    const rho = values.correlation / 100;
    const varianceLeft = 1 - rho ** 2;
    const adjustedSe = values.rawSe * Math.sqrt(varianceLeft);
    const rawZ = values.effect / values.rawSe;
    const adjustedZ = values.effect / adjustedSe;
    const sampleEquivalent = 1 / Math.max(0.05, varianceLeft);
    return {
      stats: [
        { label: 'Variance left', value: pct(varianceLeft * 100), detail: 'After CUPED adjustment', tone: varianceLeft < 0.7 ? 'emerald' : 'amber' },
        { label: 'Adjusted SE', value: one(adjustedSe), detail: `Raw SE: ${one(values.rawSe)}`, tone: 'cyan' },
        { label: 'Signal gain', value: `${(adjustedZ / rawZ).toFixed(1)}x`, detail: 'Z-score multiplier', tone: 'emerald' },
        { label: 'Sample equivalent', value: `${sampleEquivalent.toFixed(1)}x`, detail: 'Precision from same traffic', tone: sampleEquivalent > 1.5 ? 'emerald' : 'slate' },
      ],
      bars: [
        { label: 'Raw interval width', value: one(values.rawSe * 1.96 * 2), width: 100, color: 'bg-amber-500' },
        { label: 'CUPED interval width', value: one(adjustedSe * 1.96 * 2), width: (adjustedSe / values.rawSe) * 100, color: 'bg-emerald-500' },
        { label: 'Covariate signal', value: pct(values.correlation), width: values.correlation, color: 'bg-cyan-500' },
      ],
      formulaLines: [
        `adjusted outcome = Y - theta(X_pre - mean(X_pre))`,
        `variance multiplier = 1 - rho^2 = ${varianceLeft.toFixed(2)}`,
        `adjusted SE = raw SE * sqrt(1 - rho^2) = ${adjustedSe.toFixed(2)}`,
      ],
      readout: 'The adjustment removes predictable user-to-user noise that existed before treatment assignment.',
      steps: [
        { title: 'Use pre-treatment data', pass: true, body: 'The covariate must be measured before treatment so it cannot be affected by the experiment.' },
        { title: 'Reduce noise', pass: values.correlation >= 30, body: values.correlation >= 30 ? 'The covariate explains enough outcome variation to improve precision.' : 'Weak pre/post correlation gives little variance reduction.' },
        { title: 'Preserve estimand', pass: true, body: 'CUPED changes the estimator precision, not the causal question being asked.' },
      ],
      takeaway: 'Good pre-treatment covariates make experiments cheaper by narrowing intervals, but post-treatment covariates can bias the result.',
    };
  },
};

export default function CupedVarianceReductionAnimation() {
  return <CausalConceptLesson config={config} />;
}
