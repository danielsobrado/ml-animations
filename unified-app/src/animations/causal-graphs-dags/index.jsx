import React from 'react';
import CausalConceptLesson from '../_shared/CausalConceptLesson';

const pct = (value) => `${value.toFixed(0)}%`;

const config = {
  lessonId: 'causal-graphs-dags',
  kicker: 'Causal structure',
  title: 'Causal Graphs / DAGs',
  description: 'Directed acyclic graphs make assumptions explicit: confounders open backdoor paths, mediators carry part of the effect, and colliders can create bias when conditioned on.',
  controls: [
    { id: 'confounding', label: 'Backdoor strength', min: 0, max: 100, step: 5, defaultValue: 55, format: pct, help: 'A common cause of treatment and outcome opens a non-causal path.' },
    { id: 'adjustment', label: 'Confounder adjustment', min: 0, max: 100, step: 5, defaultValue: 70, format: pct, help: 'Adjustment closes the backdoor path when the variable is a true confounder.' },
    { id: 'collider', label: 'Collider conditioning', min: 0, max: 100, step: 5, defaultValue: 20, format: pct, help: 'Conditioning on a common effect can open a path that was blocked.' },
  ],
  compute(values) {
    const openBackdoor = values.confounding * (1 - values.adjustment / 100);
    const colliderBias = values.collider * 0.8;
    const bias = Math.min(100, openBackdoor + colliderBias);
    const valid = bias < 30;
    return {
      stats: [
        { label: 'Backdoor left', value: pct(openBackdoor), detail: 'Confounding after adjustment', tone: openBackdoor < 25 ? 'emerald' : 'amber' },
        { label: 'Collider bias', value: pct(colliderBias), detail: 'Opened by conditioning', tone: colliderBias > 30 ? 'rose' : 'cyan' },
        { label: 'Total bias risk', value: pct(bias), detail: 'Qualitative DAG score', tone: valid ? 'emerald' : 'rose' },
        { label: 'Adjustment set', value: valid ? 'Plausible' : 'Revise', detail: 'Graph-based decision', tone: valid ? 'emerald' : 'amber' },
      ],
      bars: [
        { label: 'Open backdoor path', value: pct(openBackdoor), width: openBackdoor, color: 'bg-amber-500' },
        { label: 'Collider path opened', value: pct(colliderBias), width: colliderBias, color: 'bg-rose-500' },
        { label: 'Closed path share', value: pct(Math.max(0, 100 - bias)), width: 100 - bias, color: 'bg-emerald-500' },
      ],
      formulaLines: [
        'confounder: C -> T and C -> Y',
        'collider: T -> S <- U, conditioning on S opens T <-> U',
        `bias risk = open backdoor + collider bias = ${bias.toFixed(0)}%`,
      ],
      readout: 'The graph tells you what to adjust for, and just as importantly what not to adjust for.',
      steps: [
        { title: 'Close backdoors', pass: openBackdoor < 30, body: openBackdoor < 30 ? 'The confounding path is mostly controlled.' : 'A strong backdoor path remains open.' },
        { title: 'Avoid colliders', pass: colliderBias < 30, body: colliderBias < 30 ? 'Collider conditioning is limited.' : 'Conditioning on a collider can create a false association.' },
        { title: 'Preserve causal target', pass: valid, body: valid ? 'This adjustment strategy is plausible for the total effect.' : 'Revise the adjustment set before estimating the effect.' },
      ],
      takeaway: 'DAGs are not decorations. They are a compact way to state assumptions before deciding which variables belong in an adjustment set.',
    };
  },
};

export default function CausalGraphsDagsAnimation() {
  return <CausalConceptLesson config={config} />;
}
