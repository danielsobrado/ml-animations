import React from 'react';
import CausalConceptLesson from '../_shared/CausalConceptLesson';

const pct = (value) => `${value.toFixed(0)}%`;

const config = {
  lessonId: 'treatment-effects',
  kicker: 'Effect heterogeneity',
  title: 'Treatment Effects',
  description: 'Average treatment effect summarizes the population, while conditional treatment effects reveal which segments benefit, are neutral, or are harmed.',
  controls: [
    { id: 'segmentShare', label: 'High-response segment', min: 10, max: 90, step: 5, defaultValue: 35, format: pct, help: 'Share of users with stronger treatment response.' },
    { id: 'highEffect', label: 'High segment effect', min: 0, max: 30, step: 1, defaultValue: 18, format: pct, help: 'Treatment lift for the responsive segment.' },
    { id: 'lowEffect', label: 'Low segment effect', min: -20, max: 15, step: 1, defaultValue: -4, format: pct, help: 'Treatment lift for everyone else.' },
  ],
  compute(values) {
    const share = values.segmentShare / 100;
    const ate = share * values.highEffect + (1 - share) * values.lowEffect;
    const heterogeneity = Math.abs(values.highEffect - values.lowEffect);
    const upliftUseful = values.highEffect > 0 && values.lowEffect < 0;
    return {
      stats: [
        { label: 'ATE', value: pct(ate), detail: 'Population average', tone: ate > 0 ? 'emerald' : 'rose' },
        { label: 'High CATE', value: pct(values.highEffect), detail: 'Responsive segment', tone: 'emerald' },
        { label: 'Low CATE', value: pct(values.lowEffect), detail: 'Remaining segment', tone: values.lowEffect < 0 ? 'rose' : 'cyan' },
        { label: 'Heterogeneity', value: pct(heterogeneity), detail: 'CATE spread', tone: heterogeneity > 15 ? 'amber' : 'slate' },
      ],
      bars: [
        { label: 'High-response segment effect', value: pct(values.highEffect), width: Math.abs(values.highEffect) * 3, color: 'bg-emerald-500' },
        { label: 'Low-response segment effect', value: pct(values.lowEffect), width: Math.abs(values.lowEffect) * 3, color: values.lowEffect < 0 ? 'bg-rose-500' : 'bg-cyan-500' },
        { label: 'Population ATE magnitude', value: pct(ate), width: Math.abs(ate) * 4, color: ate > 0 ? 'bg-emerald-500' : 'bg-rose-500' },
      ],
      formulaLines: [
        'ATE = average(Y(1) - Y(0))',
        'CATE(x) = E[Y(1) - Y(0) | X = x]',
        `ATE = ${values.segmentShare}% * ${values.highEffect}% + rest * ${values.lowEffect}% = ${ate.toFixed(1)}%`,
      ],
      readout: 'The average can hide a product opportunity or a harm concentrated in one segment.',
      steps: [
        { title: 'Estimate average effect', pass: ate > 0, body: ate > 0 ? 'The population average is positive.' : 'The population average is not positive.' },
        { title: 'Inspect heterogeneity', pass: heterogeneity <= 20, body: heterogeneity <= 20 ? 'Segment effects are fairly similar.' : 'Segment effects differ enough that a single rollout rule may be crude.' },
        { title: 'Target uplift', pass: upliftUseful, body: upliftUseful ? 'A targeted policy can help responders while avoiding harmed users.' : 'Targeting is less obvious under these segment effects.' },
      ],
      takeaway: 'ATE answers whether the treatment helps on average. CATE and uplift ask who should actually receive it.',
    };
  },
};

export default function TreatmentEffectsAnimation() {
  return <CausalConceptLesson config={config} />;
}
