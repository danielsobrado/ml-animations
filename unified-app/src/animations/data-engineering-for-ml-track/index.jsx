import React from 'react';
import CausalConceptLesson from '../_shared/CausalConceptLesson';

const pct = (value) => `${value.toFixed(0)}%`;

const config = {
  lessonId: 'data-engineering-for-ml-track',
  kicker: 'ML data systems',
  title: 'Data Engineering for ML',
  description: 'Applied ML depends on feature stores, label windows, point-in-time correctness, missing-data mechanisms, categorical encoding, target encoding safety, data contracts, and train/serve skew controls.',
  controls: [
    { id: 'freshness', label: 'Feature freshness', min: 0, max: 100, step: 5, defaultValue: 70, format: pct, help: 'How reliably production features match the intended timestamp.' },
    { id: 'leakage', label: 'Label leakage risk', min: 0, max: 100, step: 5, defaultValue: 30, format: pct, help: 'Future labels, target encodings, or aggregates leaking into training rows.' },
    { id: 'contracts', label: 'Data contract coverage', min: 0, max: 100, step: 5, defaultValue: 45, format: pct, help: 'Schema, freshness, null-rate, range, and ownership checks.' },
  ],
  compute(values) {
    const skewRisk = Math.max(0, 80 - values.freshness * 0.45 - values.contracts * 0.35 + values.leakage * 0.4);
    const readiness = Math.max(0, 100 - skewRisk - values.leakage * 0.35);
    return {
      stats: [
        { label: 'Point-in-time health', value: pct(values.freshness), detail: 'Feature timestamp fit', tone: values.freshness > 65 ? 'emerald' : 'amber' },
        { label: 'Leakage risk', value: pct(values.leakage), detail: 'Label and target encoding', tone: values.leakage > 35 ? 'rose' : 'cyan' },
        { label: 'Train/serve skew', value: pct(skewRisk), detail: 'Production mismatch risk', tone: skewRisk > 40 ? 'rose' : 'emerald' },
        { label: 'Pipeline readiness', value: pct(readiness), detail: 'Operational score', tone: readiness > 55 ? 'emerald' : 'amber' },
      ],
      bars: [
        { label: 'Feature freshness', value: pct(values.freshness), width: values.freshness, color: 'bg-emerald-500' },
        { label: 'Data contract coverage', value: pct(values.contracts), width: values.contracts, color: 'bg-cyan-500' },
        { label: 'Train/serve skew risk', value: pct(skewRisk), width: skewRisk, color: 'bg-rose-500' },
      ],
      formulaLines: [
        'label window: predict at t, observe target over [t, t+h]',
        'point-in-time join: feature_timestamp <= prediction_timestamp',
        'contracts: schema, ranges, nulls, freshness, ownership',
      ],
      readout: 'Most production ML failures are data boundary failures: wrong timestamp, wrong label window, or wrong production feature behavior.',
      steps: [
        { title: 'Protect time boundaries', pass: values.leakage <= 30, body: values.leakage <= 30 ? 'Label and feature windows are mostly separated.' : 'Leakage risk is high; audit target encodings and aggregates.' },
        { title: 'Match train and serve', pass: skewRisk <= 40, body: skewRisk <= 40 ? 'Production feature behavior is close to training assumptions.' : 'Train/serve skew needs feature parity and freshness checks.' },
        { title: 'Enforce contracts', pass: values.contracts >= 60, body: values.contracts >= 60 ? 'Data contracts can catch common pipeline regressions.' : 'Contract coverage is thin for a production ML workflow.' },
      ],
      takeaway: 'Data engineering for ML is about preserving what was knowable, measurable, and reproducible at prediction time.',
    };
  },
};

export default function DataEngineeringForMlTrackAnimation() {
  return <CausalConceptLesson config={config} />;
}
