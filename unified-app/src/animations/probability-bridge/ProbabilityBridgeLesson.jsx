import React, { useMemo, useState } from 'react';
import { BarChart3, Calculator, RotateCcw, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const LESSONS = {
  'bayes-rule-ml': {
    eyebrow: 'Probability updates',
    title: 'Bayes Rule For ML',
    description:
      'Bayes rule updates a prior belief after evidence arrives. In ML, it explains how base rates, evidence quality, and false alarms combine into a posterior probability.',
    controls: [
      { id: 'prior', label: 'Base rate', min: 1, max: 50, step: 1, defaultValue: 12, suffix: '%' },
      { id: 'sensitivity', label: 'Evidence hit rate', min: 50, max: 99, step: 1, defaultValue: 86, suffix: '%' },
      { id: 'falsePositive', label: 'False alarm rate', min: 1, max: 50, step: 1, defaultValue: 14, suffix: '%' },
    ],
    compute(values) {
      const prior = values.prior / 100;
      const sensitivity = values.sensitivity / 100;
      const falsePositive = values.falsePositive / 100;
      const numerator = sensitivity * prior;
      const denominator = numerator + falsePositive * (1 - prior);
      const posterior = denominator === 0 ? 0 : numerator / denominator;
      return {
        primary: `${Math.round(posterior * 100)}%`,
        primaryLabel: 'posterior probability',
        secondary: `${Math.round(numerator * 1000)} / ${Math.round(denominator * 1000)}`,
        secondaryLabel: 'true evidence among all positives',
        bars: [
          { label: 'Prior', value: prior, color: '#0891b2' },
          { label: 'Hit rate', value: sensitivity, color: '#10b981' },
          { label: 'False alarm', value: falsePositive, color: '#f97316' },
          { label: 'Posterior', value: posterior, color: '#7c3aed' },
        ],
        insight:
          'Even strong evidence can produce a modest posterior when the base rate is low or the false alarm rate is high.',
      };
    },
    formula: 'P(class|evidence) = P(evidence|class)P(class) / P(evidence)',
    mistake: 'Ignoring the base rate makes rare classes look more likely than the evidence supports.',
    check: 'Raise the base rate and explain why the same evidence now produces a higher posterior.',
  },
  'sampling-confidence-intervals': {
    eyebrow: 'Uncertainty from samples',
    title: 'Sampling And Confidence Intervals',
    description:
      'A sample estimate is not the population truth. Confidence intervals show a plausible range for the population value given sample size and observed variability.',
    controls: [
      { id: 'estimate', label: 'Observed rate', min: 5, max: 95, step: 1, defaultValue: 64, suffix: '%' },
      { id: 'sampleSize', label: 'Sample size', min: 25, max: 1000, step: 25, defaultValue: 200, suffix: '' },
      { id: 'confidence', label: 'Confidence level', min: 80, max: 99, step: 1, defaultValue: 95, suffix: '%' },
    ],
    compute(values) {
      const p = values.estimate / 100;
      const n = values.sampleSize;
      const z = values.confidence >= 99 ? 2.58 : values.confidence >= 95 ? 1.96 : 1.28;
      const margin = z * Math.sqrt((p * (1 - p)) / n);
      const low = Math.max(0, p - margin);
      const high = Math.min(1, p + margin);
      return {
        primary: `${Math.round(low * 100)}-${Math.round(high * 100)}%`,
        primaryLabel: 'interval estimate',
        secondary: `+/- ${(margin * 100).toFixed(1)} pts`,
        secondaryLabel: 'sampling margin',
        bars: [
          { label: 'Lower', value: low, color: '#f97316' },
          { label: 'Estimate', value: p, color: '#0891b2' },
          { label: 'Upper', value: high, color: '#10b981' },
          { label: 'Sample strength', value: Math.min(1, n / 1000), color: '#7c3aed' },
        ],
        insight:
          'Larger samples narrow the interval. Higher confidence widens it because the range must cover more repeated-sampling outcomes.',
      };
    },
    formula: 'estimate +/- z * sqrt(p(1-p)/n)',
    mistake: 'A 95% confidence interval is not a 95% probability that this fixed population value is inside one computed interval.',
    check: 'Double the sample size and predict whether the interval width halves or only shrinks by a square-root factor.',
  },
  'hypothesis-testing-intuition': {
    eyebrow: 'Signal versus noise',
    title: 'Hypothesis Testing Intuition',
    description:
      'Hypothesis tests ask whether an observed effect is surprising under a no-effect baseline. They are a noise check, not a proof that an effect matters in practice.',
    controls: [
      { id: 'effect', label: 'Observed effect', min: 0, max: 40, step: 1, defaultValue: 12, suffix: ' pts' },
      { id: 'noise', label: 'Outcome noise', min: 5, max: 35, step: 1, defaultValue: 18, suffix: ' pts' },
      { id: 'sampleSize', label: 'Sample size', min: 20, max: 1000, step: 20, defaultValue: 240, suffix: '' },
    ],
    compute(values) {
      const standardError = values.noise / Math.sqrt(values.sampleSize);
      const z = standardError === 0 ? 0 : values.effect / standardError;
      const strength = Math.min(1, z / 4);
      return {
        primary: z.toFixed(2),
        primaryLabel: 'test statistic',
        secondary: z >= 1.96 ? 'statistically unusual' : 'not unusual enough',
        secondaryLabel: 'against no-effect baseline',
        bars: [
          { label: 'Effect', value: values.effect / 40, color: '#0891b2' },
          { label: 'Noise', value: values.noise / 35, color: '#f97316' },
          { label: 'Sample strength', value: Math.min(1, values.sampleSize / 1000), color: '#10b981' },
          { label: 'Evidence', value: strength, color: '#7c3aed' },
        ],
        insight:
          'The same observed effect becomes easier to distinguish from noise when sample size rises or outcome variability falls.',
      };
    },
    formula: 'test statistic = observed effect / standard error',
    mistake: 'Statistical significance is not the same as business or practical importance.',
    check: 'Increase sample size while holding effect fixed and explain why the test statistic grows.',
  },
  'maximum-likelihood-estimation': {
    eyebrow: 'Choose parameters from data',
    title: 'Maximum Likelihood Estimation',
    description:
      'Maximum likelihood chooses the parameter value that makes the observed data most probable under the model family.',
    controls: [
      { id: 'successes', label: 'Observed successes', min: 0, max: 100, step: 1, defaultValue: 62, suffix: '' },
      { id: 'trials', label: 'Trials', min: 20, max: 200, step: 5, defaultValue: 100, suffix: '' },
      { id: 'candidate', label: 'Candidate probability', min: 1, max: 99, step: 1, defaultValue: 50, suffix: '%' },
    ],
    compute(values) {
      const trials = Math.max(values.trials, values.successes);
      const observed = values.successes / trials;
      const candidate = values.candidate / 100;
      const logLikelihood =
        values.successes * Math.log(Math.max(candidate, 0.001)) +
        (trials - values.successes) * Math.log(Math.max(1 - candidate, 0.001));
      const bestLogLikelihood =
        values.successes * Math.log(Math.max(observed, 0.001)) +
        (trials - values.successes) * Math.log(Math.max(1 - observed, 0.001));
      const fit = Math.exp(Math.max(-8, logLikelihood - bestLogLikelihood));
      return {
        primary: `${Math.round(observed * 100)}%`,
        primaryLabel: 'MLE parameter',
        secondary: `${Math.round(fit * 100)}%`,
        secondaryLabel: 'candidate relative likelihood',
        bars: [
          { label: 'Observed rate', value: observed, color: '#0891b2' },
          { label: 'Candidate', value: candidate, color: '#f97316' },
          { label: 'Relative fit', value: fit, color: '#7c3aed' },
          { label: 'Data amount', value: Math.min(1, trials / 200), color: '#10b981' },
        ],
        insight:
          'For Bernoulli data, the likelihood peaks at the observed success rate. More trials make bad candidate parameters fall off faster.',
      };
    },
    formula: 'theta_hat = argmax_theta P(data | theta)',
    mistake: 'MLE does not say the chosen model family is true; it only finds the best parameter inside that family.',
    check: 'Move the candidate probability toward the observed rate and watch relative likelihood recover.',
  },
  'loss-functions-likelihoods': {
    eyebrow: 'Why losses have this shape',
    title: 'Loss Functions As Likelihoods',
    description:
      'Many ML losses are negative log-likelihoods. Squared error fits Gaussian noise; cross-entropy fits categorical or Bernoulli outcomes.',
    controls: [
      { id: 'error', label: 'Regression error', min: 0, max: 5, step: 0.1, defaultValue: 1.2, suffix: '' },
      { id: 'sigma', label: 'Noise scale', min: 0.5, max: 4, step: 0.1, defaultValue: 1.5, suffix: '' },
      { id: 'probability', label: 'True-class probability', min: 1, max: 99, step: 1, defaultValue: 72, suffix: '%' },
    ],
    compute(values) {
      const gaussianLoss = (values.error ** 2) / (2 * values.sigma ** 2);
      const probability = values.probability / 100;
      const crossEntropy = -Math.log(Math.max(probability, 0.001));
      return {
        primary: gaussianLoss.toFixed(2),
        primaryLabel: 'Gaussian NLL term',
        secondary: crossEntropy.toFixed(2),
        secondaryLabel: 'classification NLL',
        bars: [
          { label: 'Error size', value: values.error / 5, color: '#f97316' },
          { label: 'Noise tolerance', value: values.sigma / 4, color: '#10b981' },
          { label: 'True-class prob', value: probability, color: '#0891b2' },
          { label: 'CE pressure', value: Math.min(1, crossEntropy / 4), color: '#7c3aed' },
        ],
        insight:
          'Loss shape reflects an assumption about noise. Larger regression errors or lower true-class probabilities become less likely and therefore more costly.',
      };
    },
    formula: 'loss = -log P(observed target | prediction)',
    mistake: 'A loss function is not arbitrary scoring paint; it often encodes a noise model and target distribution.',
    check: 'Lower the true-class probability and explain why cross-entropy rises sharply.',
  },
};

function makeDefaults(controls) {
  return controls.reduce((acc, control) => {
    acc[control.id] = control.defaultValue;
    return acc;
  }, {});
}

function Stat({ label, value, detail }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <p className="text-xs font-black uppercase tracking-wide text-slate-500">{label}</p>
      <strong className="mt-1 block text-2xl font-black text-slate-950">{value}</strong>
      <span className="text-sm text-slate-600">{detail}</span>
    </div>
  );
}

export default function ProbabilityBridgeLesson({ lessonId }) {
  const lesson = LESSONS[lessonId];
  const [values, setValues] = useState(() => makeDefaults(lesson.controls));
  const result = useMemo(() => lesson.compute(values), [lesson, values]);

  const reset = () => setValues(makeDefaults(lesson.controls));
  const setControl = (id) => (event) => {
    setValues((current) => ({ ...current, [id]: Number(event.target.value) }));
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">{lesson.eyebrow}</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">{lesson.title}</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">{lesson.description}</p>
          </div>
          <button
            type="button"
            onClick={reset}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-bold text-slate-800"
          >
            <RotateCcw size={16} />
            Reset
          </button>
        </div>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <SlidersHorizontal size={16} />
          Controls
        </div>
        <div className="grid gap-4 lg:grid-cols-3">
          {lesson.controls.map((control) => (
            <label key={control.id} className="grid gap-2 text-sm font-bold text-slate-700">
              {control.label}: {values[control.id]}
              {control.suffix}
              <input
                type="range"
                min={control.min}
                max={control.max}
                step={control.step}
                value={values[control.id]}
                onChange={setControl(control.id)}
              />
            </label>
          ))}
        </div>
      </section>

      <div className="grid gap-3 md:grid-cols-2">
        <Stat label={result.primaryLabel} value={result.primary} detail="current setup" />
        <Stat label={result.secondaryLabel} value={result.secondary} detail="diagnostic signal" />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <BarChart3 size={16} />
            Interactive comparison
          </h3>
          <div className="mt-5 space-y-4">
            {result.bars.map((bar) => (
              <div key={bar.label}>
                <div className="mb-1 flex items-center justify-between text-sm">
                  <strong className="text-slate-800">{bar.label}</strong>
                  <span className="font-bold text-slate-500">{Math.round(bar.value * 100)}%</span>
                </div>
                <div className="h-3 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${Math.min(100, Math.max(0, bar.value * 100))}%`, background: bar.color }}
                  />
                </div>
              </div>
            ))}
          </div>
          <p className="mt-5 rounded-lg bg-slate-50 p-3 text-sm leading-6 text-slate-700">{result.insight}</p>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <Calculator size={16} />
            Math lens
          </h3>
          <div className="mt-4 rounded-lg border border-slate-200 bg-slate-950 p-4 font-mono text-sm text-cyan-100">
            {lesson.formula}
          </div>
          <div className="mt-4 grid gap-3">
            <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
              <p className="text-xs font-black uppercase tracking-wide text-amber-700">Mistake to avoid</p>
              <p className="mt-2 text-sm leading-6 text-amber-950">{lesson.mistake}</p>
            </div>
            <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4">
              <p className="text-xs font-black uppercase tracking-wide text-emerald-700">Understanding check</p>
              <p className="mt-2 text-sm leading-6 text-emerald-950">{lesson.check}</p>
            </div>
          </div>
        </section>
      </div>

      <AssessmentPanel lessonId={lessonId} title={`${lesson.title} check`} />
    </div>
  );
}
