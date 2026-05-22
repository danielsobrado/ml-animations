import React, { useMemo, useState } from 'react';
import { Activity, BarChart3, GitBranch, Scale, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const TOKENS = {
  calm: {
    label: 'Balanced token',
    values: [0.8, 1.1, 0.9, 1.2, 1.0, 0.7],
    branch: [0.2, -0.1, 0.1, 0.3, -0.2, 0.1],
  },
  spiky: {
    label: 'Spiky token',
    values: [0.2, 4.5, -2.8, 1.1, 0.7, 3.6],
    branch: [0.3, 1.2, -1.1, 0.2, 0.1, 0.9],
  },
  shifted: {
    label: 'Shifted token',
    values: [3.4, 3.9, 4.2, 3.6, 4.5, 4.0],
    branch: [-0.4, 0.2, 0.1, -0.2, 0.3, 0.1],
  },
};

const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
const variance = (values, mu) => values.reduce((sum, value) => sum + (value - mu) ** 2, 0) / values.length;
const add = (left, right) => left.map((value, index) => value + right[index]);
const norm = (values) => Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));

function layerNorm(values, gamma, beta) {
  const mu = mean(values);
  const varValue = variance(values, mu);
  const std = Math.sqrt(varValue + 0.00001);
  return {
    mu,
    varValue,
    std,
    normalized: values.map((value) => (value - mu) / std),
    output: values.map((value) => ((value - mu) / std) * gamma + beta),
  };
}

function Bar({ value, maxAbs = 4, color = '#2563eb' }) {
  const width = `${Math.min(50, (Math.abs(value) / maxAbs) * 50)}%`;
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 text-xs font-semibold text-slate-500">{value.toFixed(2)}</div>
      <div className="relative h-3 flex-1 rounded-full bg-slate-100">
        <div
          className={`absolute top-0 h-3 rounded-full ${value >= 0 ? 'left-1/2' : 'right-1/2'}`}
          style={{ width, backgroundColor: color }}
        />
        <div className="absolute left-1/2 top-[-2px] h-5 w-px bg-slate-300" />
      </div>
    </div>
  );
}

function Metric({ icon: Icon, label, value, helper }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
        <Icon size={16} />
        {label}
      </div>
      <div className="mt-2 text-2xl font-bold text-slate-950">{value}</div>
      <p className="mt-1 text-sm text-slate-600">{helper}</p>
    </div>
  );
}

function VectorPanel({ title, values, color }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <h3 className="font-bold text-slate-950">{title}</h3>
      <div className="mt-3 space-y-2">
        {values.map((value, index) => (
          <Bar key={index} value={value} color={color} />
        ))}
      </div>
    </div>
  );
}

export default function LayerNormalizationAnimation() {
  const [tokenId, setTokenId] = useState('spiky');
  const [gamma, setGamma] = useState(1);
  const [beta, setBeta] = useState(0);
  const [branchScale, setBranchScale] = useState(1);
  const [mode, setMode] = useState('pre');

  const token = TOKENS[tokenId];
  const branch = token.branch.map((value) => value * branchScale);
  const normalizedInput = useMemo(() => layerNorm(token.values, gamma, beta), [beta, gamma, token.values]);
  const postResidual = add(token.values, branch);
  const normalizedPostResidual = useMemo(() => layerNorm(postResidual, gamma, beta), [beta, gamma, postResidual]);
  const finalOutput = mode === 'pre' ? add(token.values, branch) : normalizedPostResidual.output;
  const displayedNorm = mode === 'pre' ? normalizedInput : normalizedPostResidual;
  const stabilityRatio = norm(finalOutput) / Math.max(norm(token.values), 0.001);

  return (
    <div className="min-h-full bg-slate-50">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 p-4 md:p-6">
        <header className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-violet-700">
            <Scale size={17} />
            Transformer stabilization
          </div>
          <h1 className="mt-2 text-2xl font-bold text-slate-950 md:text-3xl">Layer Normalization</h1>
          <p className="mt-2 max-w-3xl text-slate-700">
            LayerNorm normalizes the feature dimensions inside one token representation. In transformer blocks it keeps
            residual updates on a usable scale even when activations are shifted or spiky.
          </p>
        </header>

        <section className="grid gap-4 lg:grid-cols-[320px_1fr]">
          <aside className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex items-center gap-2 font-semibold text-slate-950">
              <SlidersHorizontal size={18} />
              Normalization controls
            </div>
            <div className="mt-5 space-y-4">
              <label className="block">
                <div className="mb-2 text-sm font-semibold text-slate-700">Token case</div>
                <select
                  value={tokenId}
                  onChange={(event) => setTokenId(event.target.value)}
                  className="w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900"
                >
                  {Object.entries(TOKENS).map(([id, item]) => (
                    <option key={id} value={id}>{item.label}</option>
                  ))}
                </select>
              </label>
              <label className="block">
                <div className="mb-2 text-sm font-semibold text-slate-700">Block placement</div>
                <select
                  value={mode}
                  onChange={(event) => setMode(event.target.value)}
                  className="w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900"
                >
                  <option value="pre">Pre-norm: normalize before branch</option>
                  <option value="post">Post-norm: normalize after residual add</option>
                </select>
              </label>
              {[
                ['Gamma scale', gamma, 0.4, 2, 0.05, setGamma],
                ['Beta shift', beta, -1.5, 1.5, 0.05, setBeta],
                ['Branch strength', branchScale, 0, 2, 0.05, setBranchScale],
              ].map(([label, value, min, max, step, setter]) => (
                <label key={label} className="block">
                  <div className="mb-2 flex justify-between text-sm font-semibold text-slate-700">
                    <span>{label}</span>
                    <span>{Number(value).toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={min}
                    max={max}
                    step={step}
                    value={value}
                    onChange={(event) => setter(Number(event.target.value))}
                    className="w-full accent-violet-700"
                  />
                </label>
              ))}
            </div>
          </aside>

          <main className="space-y-4">
            <div className="grid gap-4 md:grid-cols-4">
              <Metric icon={BarChart3} label="Mean" value={displayedNorm.mu.toFixed(2)} helper="Computed over features in one token." />
              <Metric icon={Activity} label="Std dev" value={displayedNorm.std.toFixed(2)} helper="Scale removed before gamma/beta." />
              <Metric icon={GitBranch} label="Final norm ratio" value={`${stabilityRatio.toFixed(2)}x`} helper="Output magnitude versus input." />
              <Metric icon={Scale} label="Placement" value={mode === 'pre' ? 'pre-norm' : 'post-norm'} helper="Where LayerNorm sits around residual add." />
            </div>

            <section className="grid gap-4 xl:grid-cols-3">
              <VectorPanel title="Input token features" values={token.values} color="#64748b" />
              <VectorPanel title="Normalized features" values={displayedNorm.output} color="#7c3aed" />
              <VectorPanel title="Final residual output" values={finalOutput} color="#0f766e" />
            </section>

            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <h2 className="text-lg font-bold text-slate-950">Feature-by-feature transformation</h2>
              <p className="text-sm text-slate-600">
                LayerNorm subtracts the token mean, divides by token standard deviation, then applies learned gamma and
                beta. Batch statistics from other examples are not used.
              </p>
              <div className="mt-4 overflow-auto rounded-lg border border-slate-200">
                <table className="w-full min-w-[680px] border-collapse text-sm">
                  <thead className="bg-slate-100 text-left text-xs uppercase tracking-wide text-slate-500">
                    <tr>
                      <th className="px-3 py-2">Feature</th>
                      <th className="px-3 py-2">Raw value</th>
                      <th className="px-3 py-2">Centered</th>
                      <th className="px-3 py-2">Normalized</th>
                      <th className="px-3 py-2">Gamma/beta output</th>
                    </tr>
                  </thead>
                  <tbody>
                    {token.values.map((value, index) => (
                      <tr key={index} className="border-t border-slate-200">
                        <td className="px-3 py-2 font-bold text-slate-950">h{index + 1}</td>
                        <td className="px-3 py-2 font-mono">{value.toFixed(2)}</td>
                        <td className="px-3 py-2 font-mono">{(value - normalizedInput.mu).toFixed(2)}</td>
                        <td className="px-3 py-2 font-mono">{normalizedInput.normalized[index].toFixed(2)}</td>
                        <td className="px-3 py-2 font-mono">{normalizedInput.output[index].toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="grid gap-4 md:grid-cols-3">
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">Predict before running</h3>
                <p className="mt-2 text-sm text-slate-700">
                  Switch from balanced to shifted input and predict whether the normalized output keeps the large mean.
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">Transformer link</h3>
                <p className="mt-2 text-sm text-slate-700">
                  Pre-norm improves gradient flow because the transformation branch receives normalized input while the
                  residual path remains direct.
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">Mistake to avoid</h3>
                <p className="mt-2 text-sm text-slate-700">
                  LayerNorm is not BatchNorm. It normalizes features within each token, so it works naturally at batch
                  size one during autoregressive decoding.
                </p>
              </div>
            </section>
          </main>
        </section>

        <AssessmentPanel lessonId="layer-normalization" />
      </div>
    </div>
  );
}
