import React from 'react';
import CausalConceptLesson from '../_shared/CausalConceptLesson';

const pct = (value) => `${value.toFixed(0)}%`;
const one = (value) => value.toFixed(1);

const config = {
  lessonId: 'confounding-simpsons-paradox',
  kicker: 'Observational bias',
  title: "Confounding & Simpson's Paradox",
  description: 'Aggregate effects can reverse within groups when treatment exposure is tied to a confounder. Simpson reversals are a warning that the comparison mix changed.',
  controls: [
    { id: 'highRiskTreatmentShare', label: 'High-risk in treatment', min: 10, max: 90, step: 5, defaultValue: 70, format: pct, help: 'Treatment gets more difficult users when assignment is confounded.' },
    { id: 'segmentGap', label: 'Segment baseline gap', min: 5, max: 50, step: 5, defaultValue: 30, format: pct, help: 'How different the low-risk and high-risk baselines are.' },
    { id: 'withinLift', label: 'Within-segment lift', min: -10, max: 20, step: 1, defaultValue: 6, format: pct, help: 'Treatment effect inside each comparable segment.' },
  ],
  compute(values) {
    const lowBase = 0.55;
    const highBase = lowBase - values.segmentGap / 100;
    const lift = values.withinLift / 100;
    const highTreat = values.highRiskTreatmentShare / 100;
    const highControl = 1 - highTreat;
    const treatRate = highTreat * (highBase + lift) + (1 - highTreat) * (lowBase + lift);
    const controlRate = highControl * highBase + (1 - highControl) * lowBase;
    const aggregateEffect = treatRate - controlRate;
    const reversal = Math.sign(aggregateEffect) !== Math.sign(lift) && lift !== 0;
    return {
      stats: [
        { label: 'Within effect', value: pct(lift * 100), detail: 'Inside each segment', tone: lift >= 0 ? 'emerald' : 'rose' },
        { label: 'Aggregate effect', value: pct(aggregateEffect * 100), detail: 'Mixed population readout', tone: reversal ? 'rose' : 'cyan' },
        { label: 'Mix imbalance', value: pct(Math.abs(highTreat - highControl) * 100), detail: 'High-risk share gap', tone: Math.abs(highTreat - highControl) > 0.4 ? 'rose' : 'amber' },
        { label: 'Reversal', value: reversal ? 'Yes' : 'No', detail: 'Simpson warning', tone: reversal ? 'rose' : 'emerald' },
      ],
      bars: [
        { label: 'Treatment aggregate', value: pct(treatRate * 100), width: treatRate * 100, color: 'bg-emerald-500' },
        { label: 'Control aggregate', value: pct(controlRate * 100), width: controlRate * 100, color: 'bg-cyan-500' },
        { label: 'High-risk treatment mix', value: pct(highTreat * 100), width: highTreat * 100, color: 'bg-amber-500' },
      ],
      formulaLines: [
        `aggregate = sum(segment rate * segment mix)`,
        `within lift = ${values.withinLift} points in each segment`,
        `mixed aggregate lift = ${(aggregateEffect * 100).toFixed(1)} points`,
      ],
      readout: 'The aggregate comparison blends outcome differences with a population-mix difference.',
      steps: [
        { title: 'Compare like with like', pass: Math.abs(highTreat - highControl) <= 0.3, body: Math.abs(highTreat - highControl) <= 0.3 ? 'Segment mix is fairly balanced.' : 'Treatment and control contain very different user mixes.' },
        { title: 'Inspect slices', pass: true, body: 'Segment-level effects reveal whether the aggregate is hiding a reversal.' },
        { title: 'Block confounding', pass: !reversal, body: reversal ? 'The aggregate points in the wrong direction for the within-segment effect.' : 'No reversal appears under these settings.' },
      ],
      takeaway: 'When exposure is not randomized, always ask which variable affects both treatment assignment and the outcome.',
    };
  },
};

export default function ConfoundingSimpsonsParadoxAnimation() {
  return <CausalConceptLesson config={config} />;
}
