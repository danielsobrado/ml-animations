import React, { useMemo, useState } from 'react';
import { BarChart3, RefreshCw, Search, Target, TrendingDown, TrendingUp } from 'lucide-react';

const CASES = [
  { id: 'A', age: 24, score: 42, tenure: 2.2, label: 0 },
  { id: 'B', age: 31, score: 58, tenure: 3.6, label: 0 },
  { id: 'C', age: 46, score: 73, tenure: 5.0, label: 1 },
  { id: 'D', age: 29, score: 64, tenure: 4.1, label: 0 },
  { id: 'E', age: 52, score: 81, tenure: 6.9, label: 1 },
  { id: 'F', age: 61, score: 90, tenure: 7.8, label: 1 },
  { id: 'G', age: 38, score: 55, tenure: 1.8, label: 0 },
  { id: 'H', age: 27, score: 49, tenure: 2.4, label: 0 },
];

const FEATURES = [
  { id: 'age', label: 'Age', min: 18, max: 80 },
  { id: 'score', label: 'Risk score', min: 20, max: 100 },
  { id: 'tenure', label: 'Tenure', min: 1, max: 10 },
];

const MODELS = {
  linear: {
    label: 'Simple linear rule',
    desc: 'Transparent coefficients. Good first pass, easy to inspect.',
    weights: { age: -0.018, score: 0.042, tenure: 0.18 },
    bias: -0.4,
  },
  correlated: {
    label: 'Correlated explanation model',
    desc: 'Looks plausible, but two inputs become entangled.',
    weights: { age: -0.026, score: 0.030, tenure: 0.20 },
    bias: -0.1,
  },
  sparse: {
    label: 'Sparse production model',
    desc: 'One strong feature dominates; easier but less fair to nuanced cohorts.',
    weights: { age: -0.012, score: 0.055, tenure: 0.09 },
    bias: -1.4,
  },
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function meanAbs(values) {
  return values.reduce((sum, value) => sum + Math.abs(value), 0) / values.length;
}

function toRange(values, key) {
  const vals = values.map((sample) => sample[key]);
  return { min: Math.min(...vals), max: Math.max(...vals) };
}

function normalize(sample, ranges) {
  const next = { ...sample };
  for (const feature of Object.keys(ranges)) {
    const range = ranges[feature];
    const width = range.max - range.min || 1;
    next[feature] = (sample[feature] - range.min) / width;
  }
  return next;
}

function predict(features, model) {
  const score = model.bias
    + model.weights.age * features.age
    + model.weights.score * features.score
    + model.weights.tenure * features.tenure;
  const probability = sigmoid(score);
  return { score, probability };
}

function accuracy(samples, model, ranges) {
  const base = meanRanges(samples);
  const total = samples.length;
  const correct = samples.filter((sample) => {
    const normalized = normalize(sample, ranges);
    return (predict(normalized, model).probability >= 0.5 ? 1 : 0) === sample.label;
  }).length;
  return correct / total;
}

function meanRanges(samples) {
  const ranges = {};
  for (const feature of Object.keys(samples[0])) {
    if (feature === 'label' || feature === 'id') continue;
    ranges[feature] = toRange(samples, feature);
  }
  return ranges;
}

function globalContributions(samples, model, baseline, ranges) {
  const normalized = samples.map((sample) => normalize(sample, ranges));
  const baselinePred = predict(baseline, model).probability;
  const baseAcc = accuracy(samples, model, ranges);
  return Object.keys(model.weights).map((feature) => {
    const replaced = normalized.map((sample) => ({ ...sample, [feature]: baseline[feature] }));
    const keep = replaced.filter((sample, index) => {
      const predicted = predict(sample, model).probability >= 0.5 ? 1 : 0;
      return predicted === CASES[index].label;
    }).length;
    const drop = baseAcc - keep / samples.length;
    return {
      feature,
      contribution: Math.max(0, drop * 1.4 + 0.04),
    };
  });
}

function localContribution(sample, baseline, model) {
  const baseContrib = {};
  for (const feature of Object.keys(model.weights)) {
    baseContrib[feature] = model.weights[feature] * (sample[feature] - baseline[feature]);
  }
  const total = Object.values(baseContrib).reduce((sum, value) => sum + value, 0);
  const sorted = Object.entries(baseContrib).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  return { baseContrib, total, sorted };
}

function barWidth(weight, maxWeight) {
  return `${clamp((Math.abs(weight) / maxWeight) * 100, 0, 100)}%`;
}

export default function ModelInterpretability() {
  const [modelId, setModelId] = useState('linear');
  const [showCorrelated, setShowCorrelated] = useState(false);
  const [sampleId, setSampleId] = useState('C');
  const [counterFeature, setCounterFeature] = useState('score');
  const [counterShift, setCounterShift] = useState(20);

  const ranges = useMemo(() => meanRanges(CASES), []);
  const normalized = useMemo(() => CASES.map((sample) => normalize(sample, ranges)), [ranges]);
  const model = MODELS[modelId];
  const baseline = useMemo(() => {
    const means = Object.fromEntries(
      Object.keys(model.weights).map((key) => [key, mean(normalized.map((point) => point[key]))]),
    );
    return means;
  }, [normalized]);

  const effectiveModel = useMemo(() => {
    if (!showCorrelated) return model;
    return {
      ...model,
      bias: model.bias + 0.06,
      weights: {
        ...model.weights,
        age: model.weights.age - 0.004,
        score: model.weights.score + 0.006,
      },
      label: `${model.label || model.label} (correlated)`,
    };
  }, [model, showCorrelated]);

  const selected = normalized.find((sample) => sample.id === sampleId) || normalized[0];
  const current = predict(selected, effectiveModel);
  const adjusted = useMemo(() => {
    const shifted = {
      ...selected,
      [counterFeature]: clamp(selected[counterFeature] + (counterShift / 100), 0, 1),
    };
    return {
      sample: shifted,
      pred: predict(shifted, effectiveModel),
    };
  }, [counterFeature, counterShift, selected, effectiveModel]);

  const contributions = useMemo(() => {
    const raw = globalContributions(normalized, effectiveModel, baseline, ranges);
    const maxWeight = raw.reduce((max, item) => Math.max(max, item.contribution), 0) || 1;
    return raw.map((entry) => ({ ...entry, width: barWidth(entry.contribution, maxWeight) }));
  }, [effectiveModel, normalized, baseline, ranges]);

  const local = useMemo(() => localContribution(selected, baseline, effectiveModel), [selected, baseline, effectiveModel]);
  const sortedByAbs = [...local.sorted].sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

  const localFlip = current.probability >= 0.5;
  const adjustedFlip = adjusted.pred.probability >= 0.5;
  const globalAcc = accuracy(CASES, effectiveModel, ranges);
  const baseCase = toCaseSummary(CASES, normalized, effectiveModel, ranges);

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Explainability</p>
            <h1 className="mt-1 text-2xl font-black text-slate-950">Model Interpretability</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Use attribution-style views to separate intuition from proof. This lesson shows local decomposition, global ranking,
              and sensitivity under correlated-feature conditions.
            </p>
          </div>
          <button
            type="button"
            onClick={() => {
              setModelId('linear');
              setShowCorrelated(false);
              setSampleId('C');
              setCounterFeature('score');
              setCounterShift(20);
            }}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-bold text-slate-800"
          >
            <RefreshCw size={16} />
            Reset
          </button>
        </div>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-3 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <Target size={16} />
          Global model and data controls
        </div>
        <div className="grid gap-4 lg:grid-cols-[1.4fr_0.8fr_0.8fr]">
          <div className="grid gap-2">
            <span className="text-sm font-bold text-slate-700">Explanation view</span>
            <div className="grid gap-2 sm:grid-cols-3">
              {Object.entries(MODELS).map(([key, settings]) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => setModelId(key)}
                  className={`rounded-lg border px-3 py-2 text-left text-sm font-black transition ${modelId === key ? 'border-cyan-600 bg-cyan-50 text-cyan-900' : 'border-slate-200 bg-slate-50 text-slate-700'}`}
                >
                  {settings.label}
                </button>
              ))}
            </div>
            <p className="text-sm leading-6 text-slate-700">{effectiveModel.desc}</p>
          </div>
          <label className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-bold text-slate-700">
            Correlated features mode
            <input type="checkbox" className="ml-2" checked={showCorrelated} onChange={(event) => setShowCorrelated(event.target.checked)} />
          </label>
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <p className="text-sm font-black uppercase tracking-wide text-slate-600">Toy model confidence</p>
            <p className="mt-2 text-sm leading-6 text-slate-700">
              Current split accuracy is <strong>{(globalAcc * 100).toFixed(0)}%</strong>.
            </p>
            <p className="mt-2 text-xs leading-6 text-slate-600">
              This is a simplified surrogate. Attributions are directional signals, not causal proofs.
            </p>
          </div>
        </div>
      </section>

      <div className="grid gap-3 xl:grid-cols-2">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-3 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <BarChart3 size={16} />
            Global feature importance (ablation approximation)
          </div>
          <div className="space-y-3">
            {contributions.map((entry) => {
              const feature = FEATURES.find((item) => item.id === entry.feature);
              return (
                <div key={entry.feature}>
                  <div className="mb-1 flex items-center justify-between text-sm">
                    <span className="font-semibold text-slate-700">{feature?.label}</span>
                    <span className="font-black text-slate-900">{(entry.contribution * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-3 rounded bg-slate-100">
                    <div className="h-3 rounded bg-cyan-600" style={{ width: entry.width }} />
                  </div>
                </div>
              );
            })}
          </div>
          <p className="mt-4 text-sm leading-6 text-slate-700">
            {baseCase.notice}
          </p>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-3 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <Search size={16} />
            Local explanation for one sample
          </div>
          <div className="grid gap-2">
            <label className="text-sm font-bold text-slate-700">
              Sample
              <select value={sampleId} onChange={(event) => setSampleId(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-3 py-2">
                {normalized.map((sample) => (
                  <option key={sample.id} value={sample.id}>
                    {sample.id} (label {sample.label})
                  </option>
                ))}
              </select>
            </label>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <p className="text-sm text-slate-600">
                Predicted probability: <strong className="text-slate-900">{(current.probability * 100).toFixed(1)}%</strong>
              </p>
              <p className="text-xs mt-1">
                Raw logit: <strong>{current.score.toFixed(2)}</strong> ({current.probability >= 0.5 ? 'positive' : 'negative'})
              </p>
            </div>
            <div className="space-y-2">
              {sortedByAbs.map(([feature, contribution]) => {
                const featureMeta = FEATURES.find((item) => item.id === feature);
                return (
                  <div key={feature} className="rounded-lg border border-slate-200 bg-slate-50 p-2">
                    <div className="text-xs uppercase tracking-wide text-slate-500">{featureMeta?.label}</div>
                    <div className="text-sm font-black text-slate-900">
                      {contribution >= 0 ? `+${contribution.toFixed(3)}` : contribution.toFixed(3)} in logit
                    </div>
                  </div>
                );
              })}
              <p className="text-xs text-slate-600">Sum of displayed contributions: {local.total.toFixed(3)} in shifted logit space.</p>
            </div>
          </div>
        </section>
      </div>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <TrendingUp size={16} />
          Counterfactual and sensitivity check
        </div>
        <div className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
          <div className="space-y-4">
            <label className="block text-sm font-bold text-slate-700">
              Feature to perturb
              <select
                value={counterFeature}
                onChange={(event) => setCounterFeature(event.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-3 py-2"
              >
                {FEATURES.map((feature) => (
                  <option key={feature.id} value={feature.id}>{feature.label}</option>
                ))}
              </select>
            </label>
            <label className="block text-sm font-bold text-slate-700">
              +{counterShift}%
              <input
                type="range"
                min={-35}
                max={35}
                value={counterShift}
                onChange={(event) => setCounterShift(Number(event.target.value))}
                className="mt-2 w-full accent-cyan-700"
              />
            </label>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm leading-6 text-slate-700">
              <p>
                Before: <strong>{(current.probability * 100).toFixed(1)}%</strong>
              </p>
              <p>
                After perturbation: <strong>{(adjusted.pred.probability * 100).toFixed(1)}%</strong>
              </p>
              <p className={localFlip === adjustedFlip ? 'text-amber-800 mt-2' : 'text-emerald-800 mt-2'}>
                {localFlip !== adjustedFlip ? 'Decision flips under counterfactual perturbation.' : 'Decision is stable under this perturbation.'}
              </p>
            </div>
          </div>
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
            <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Interpretation notes</h3>
            <p className="mt-3 text-sm leading-6 text-slate-700">
              Attribution methods answer: <strong>what drives this example</strong>.
              They do not certify that increasing one feature causes the outcome.
            </p>
            <p className="mt-3 text-sm leading-6 text-slate-700">
              In this toy setting, we estimate feature impact from shifted logit terms and feature-mean ablations.
            </p>
            <p className="mt-3 text-sm leading-6 text-slate-700">
              With correlated features active, these checks become less reliable because feature swaps are no longer independent.
            </p>
          </div>
        </div>
      </section>

      <section className="rounded-lg border border-red-200 bg-red-50 p-5">
        <div className="mb-3 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-red-700">
          <TrendingDown size={16} />
          Failure-mode reminder
        </div>
        <p className="text-sm leading-6 text-red-950">
          A single high-attribution feature can be an explanation artifact if the model or data is misspecified.
          Use multiple lenses (global, local, and counterfactual) before relying on any one story.
        </p>
      </section>
    </div>
  );
}

function toCaseSummary(samples, normalizedSamples, model, ranges) {
  const score = accuracy(samples, model, ranges);
  const label = score > 0.7 ? 'stable and reasonably balanced' : 'not very stable';
  return {
    notice: `Current setup is ${label}; predictions were computed after feature normalization and the selected model coefficients.`,
  };
}

