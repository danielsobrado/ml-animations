import React from 'react';
import CausalConceptLesson from '../_shared/CausalConceptLesson';

const pct = (value) => `${value.toFixed(0)}%`;

const config = {
  lessonId: 'propensity-scores',
  kicker: 'Observational adjustment',
  title: 'Propensity Scores',
  description: 'A propensity score estimates the probability of treatment from observed covariates. Matching or weighting uses it to make treated and control groups more comparable.',
  controls: [
    { id: 'overlap', label: 'Propensity overlap', min: 10, max: 100, step: 5, defaultValue: 65, format: pct, help: 'Common support: treated and control units with similar treatment probabilities.' },
    { id: 'imbalance', label: 'Initial imbalance', min: 0, max: 100, step: 5, defaultValue: 60, format: pct, help: 'Covariate difference before matching or weighting.' },
    { id: 'hiddenBias', label: 'Unobserved confounding', min: 0, max: 80, step: 5, defaultValue: 20, format: pct, help: 'Bias from variables not measured in the propensity model.' },
  ],
  compute(values) {
    const adjustedImbalance = values.imbalance * (1 - values.overlap / 120);
    const totalBias = Math.min(100, adjustedImbalance + values.hiddenBias);
    const usable = values.overlap >= 50 && totalBias < 45;
    return {
      stats: [
        { label: 'Overlap', value: pct(values.overlap), detail: 'Common support', tone: values.overlap >= 50 ? 'emerald' : 'rose' },
        { label: 'Balance left', value: pct(adjustedImbalance), detail: 'After adjustment', tone: adjustedImbalance < 30 ? 'emerald' : 'amber' },
        { label: 'Hidden bias', value: pct(values.hiddenBias), detail: 'Not fixed by propensity', tone: values.hiddenBias > 35 ? 'rose' : 'cyan' },
        { label: 'Use estimate?', value: usable ? 'Cautious' : 'No', detail: 'Design diagnostic', tone: usable ? 'emerald' : 'rose' },
      ],
      bars: [
        { label: 'Initial covariate imbalance', value: pct(values.imbalance), width: values.imbalance, color: 'bg-amber-500' },
        { label: 'Adjusted imbalance', value: pct(adjustedImbalance), width: adjustedImbalance, color: 'bg-cyan-500' },
        { label: 'Total residual bias risk', value: pct(totalBias), width: totalBias, color: totalBias > 45 ? 'bg-rose-500' : 'bg-emerald-500' },
      ],
      formulaLines: [
        'e(x) = P(T = 1 | X = x)',
        'IPW weight: treated 1/e(x), control 1/(1 - e(x))',
        `residual bias risk = ${totalBias.toFixed(0)}%`,
      ],
      readout: 'Propensity methods can balance observed covariates, but they cannot repair missing confounders or no-overlap regions.',
      steps: [
        { title: 'Check overlap', pass: values.overlap >= 50, body: values.overlap >= 50 ? 'Treated and control units share enough common support.' : 'Poor overlap means some treated units have no credible controls.' },
        { title: 'Improve balance', pass: adjustedImbalance < values.imbalance / 2, body: adjustedImbalance < values.imbalance / 2 ? 'Matching or weighting meaningfully reduces observed imbalance.' : 'The propensity adjustment leaves too much observed imbalance.' },
        { title: 'State limitation', pass: values.hiddenBias <= 30, body: values.hiddenBias <= 30 ? 'Unobserved confounding risk is moderate in this scenario.' : 'Unobserved confounding remains a major threat.' },
      ],
      takeaway: 'Propensity scores are design tools for observational data. They help most when overlap is strong and the important confounders are measured.',
    };
  },
};

export default function PropensityScoresAnimation() {
  return <CausalConceptLesson config={config} />;
}
