import React from 'react';
import CausalConceptLesson from '../_shared/CausalConceptLesson';

const pct = (value) => `${value.toFixed(0)}%`;

const config = {
  lessonId: 'ml-security-robustness-track',
  kicker: 'Reliability under attack',
  title: 'ML Security & Robustness',
  description: 'Modern ML systems need defenses for prompt injection, retrieval poisoning, data poisoning, adversarial inputs, membership inference, PII leakage, tool-call safety, and jailbreak evaluation.',
  controls: [
    { id: 'attackSurface', label: 'Attack surface', min: 0, max: 100, step: 5, defaultValue: 65, format: pct, help: 'External content, tools, retrieval, user uploads, or model outputs exposed to attackers.' },
    { id: 'defense', label: 'Defense coverage', min: 0, max: 100, step: 5, defaultValue: 45, format: pct, help: 'Input filtering, retrieval isolation, evals, policy checks, and monitoring.' },
    { id: 'sensitivity', label: 'Data sensitivity', min: 0, max: 100, step: 5, defaultValue: 55, format: pct, help: 'PII, secrets, proprietary data, or regulated content at risk.' },
  ],
  compute(values) {
    const residualRisk = Math.max(0, values.attackSurface * 0.45 + values.sensitivity * 0.45 - values.defense * 0.55);
    const readiness = Math.max(0, 100 - residualRisk);
    return {
      stats: [
        { label: 'Residual risk', value: pct(residualRisk), detail: 'After defenses', tone: residualRisk > 45 ? 'rose' : 'amber' },
        { label: 'Defense coverage', value: pct(values.defense), detail: 'Controls and evals', tone: values.defense > 60 ? 'emerald' : 'amber' },
        { label: 'Sensitive data', value: pct(values.sensitivity), detail: 'Leakage impact', tone: values.sensitivity > 50 ? 'rose' : 'cyan' },
        { label: 'Release readiness', value: pct(readiness), detail: 'Security posture', tone: readiness > 60 ? 'emerald' : 'amber' },
      ],
      bars: [
        { label: 'Prompt and tool attack surface', value: pct(values.attackSurface), width: values.attackSurface, color: 'bg-rose-500' },
        { label: 'Defense and evaluation coverage', value: pct(values.defense), width: values.defense, color: 'bg-emerald-500' },
        { label: 'Residual security risk', value: pct(residualRisk), width: residualRisk, color: 'bg-amber-500' },
      ],
      formulaLines: [
        'threats: prompt injection, poisoning, adversarial examples',
        'privacy: membership inference, PII leakage',
        'controls: isolation, evals, policy gates, monitoring',
      ],
      readout: 'Security risk grows when untrusted inputs can affect retrieval, tools, or sensitive outputs without isolation and evaluation.',
      steps: [
        { title: 'Map attack surface', pass: values.attackSurface <= 50, body: values.attackSurface <= 50 ? 'Exposure is limited.' : 'The system has many places where untrusted content can steer behavior.' },
        { title: 'Cover defenses', pass: values.defense >= 60, body: values.defense >= 60 ? 'Controls and adversarial evals are reasonably broad.' : 'Defense coverage is thin for this attack surface.' },
        { title: 'Protect sensitive data', pass: values.sensitivity <= 40 || values.defense >= 70, body: values.sensitivity <= 40 || values.defense >= 70 ? 'Data sensitivity is matched by controls.' : 'Sensitive data requires stronger leakage and access controls.' },
      ],
      takeaway: 'Treat ML security as a system property: model behavior, retrieval, tools, data access, and evals must be designed together.',
    };
  },
};

export default function MlSecurityRobustnessTrackAnimation() {
  return <CausalConceptLesson config={config} />;
}
