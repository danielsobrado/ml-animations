import React, { useMemo, useState } from 'react';
import { AlertTriangle, BarChart3, Gauge, ShieldCheck, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const BATCH_TEMPLATE = [
  { id: 'A', state: 'near goal', action: 'continue', ratioOffset: 0, advantageScale: 1 },
  { id: 'B', state: 'near goal', action: 'detour', ratioOffset: 0.28, advantageScale: 0.65 },
  { id: 'C', state: 'trap edge', action: 'step back', ratioOffset: -0.34, advantageScale: -0.8 },
  { id: 'D', state: 'open room', action: 'explore', ratioOffset: 0.14, advantageScale: -0.45 },
  { id: 'E', state: 'bottleneck', action: 'wait', ratioOffset: -0.18, advantageScale: 0.5 },
];

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function ppoObjective(ratio, advantage, epsilon) {
  const clippedRatio = clamp(ratio, 1 - epsilon, 1 + epsilon);
  const unclipped = ratio * advantage;
  const clipped = clippedRatio * advantage;
  const objective = Math.min(unclipped, clipped);
  return {
    ratio,
    clippedRatio,
    unclipped,
    clipped,
    objective,
    clippedActive: Math.abs(objective - unclipped) > 1e-9,
  };
}

function safeLog(value) {
  return Math.log(Math.max(value, 1e-8));
}

function binaryEntropy(probability) {
  const p = clamp(probability, 1e-6, 1 - 1e-6);
  return -(p * safeLog(p) + (1 - p) * safeLog(1 - p));
}

function binaryKl(oldProbability, newProbability) {
  const p = clamp(oldProbability, 1e-6, 1 - 1e-6);
  const q = clamp(newProbability, 1e-6, 1 - 1e-6);
  return p * safeLog(p / q) + (1 - p) * safeLog((1 - p) / (1 - q));
}

function Metric({ label, value, helper }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <p className="text-xs font-black uppercase tracking-wide text-slate-500">{label}</p>
      <strong className="mt-1 block text-2xl font-black text-slate-950">{value}</strong>
      <span className="text-sm text-slate-600">{helper}</span>
    </div>
  );
}

function Control({ label, value, children }) {
  return (
    <label className="grid gap-2 text-sm font-bold text-slate-700">
      <span className="flex items-center justify-between gap-3">
        {label}
        <strong className="text-slate-950">{value}</strong>
      </span>
      {children}
    </label>
  );
}

function SurrogateChart({ advantage, epsilon }) {
  const xMin = 0.4;
  const xMax = 1.8;
  const values = Array.from({ length: 90 }, (_, index) => {
    const ratio = xMin + (index / 89) * (xMax - xMin);
    return { ratio, objective: ppoObjective(ratio, advantage, epsilon).objective };
  });
  const yValues = values.map((point) => point.objective);
  const yMin = Math.min(...yValues, -0.5);
  const yMax = Math.max(...yValues, 0.5);
  const xScale = (ratio) => 42 + ((ratio - xMin) / (xMax - xMin)) * 436;
  const yScale = (value) => 250 - ((value - yMin) / (yMax - yMin || 1)) * 190;
  const path = values.map((point, index) => (
    `${index === 0 ? 'M' : 'L'} ${xScale(point.ratio).toFixed(1)} ${yScale(point.objective).toFixed(1)}`
  )).join(' ');
  const lowerX = xScale(1 - epsilon);
  const upperX = xScale(1 + epsilon);

  return (
    <svg viewBox="0 0 520 290" className="block h-auto w-full max-w-full rounded-lg border border-slate-200 bg-slate-50" role="img" aria-label="PPO clipped surrogate objective by policy ratio">
      <rect x="42" y="42" width="436" height="208" fill="#fff" stroke="#e2e8f0" />
      <line x1="42" x2="478" y1={yScale(0)} y2={yScale(0)} stroke="#cbd5e1" strokeDasharray="4 4" />
      <rect x={lowerX} y="42" width={upperX - lowerX} height="208" fill="#dcfce7" opacity="0.45" />
      <line x1={lowerX} x2={lowerX} y1="42" y2="250" stroke="#16a34a" strokeWidth="2" />
      <line x1={upperX} x2={upperX} y1="42" y2="250" stroke="#16a34a" strokeWidth="2" />
      <path d={path} fill="none" stroke="#2563eb" strokeWidth="4" strokeLinecap="round" />
      <text x="42" y="276" className="fill-slate-600 text-xs font-bold">ratio 0.4</text>
      <text x="217" y="276" className="fill-slate-600 text-xs font-bold">old policy 1.0</text>
      <text x="420" y="276" className="fill-slate-600 text-xs font-bold">ratio 1.8</text>
      <text x={lowerX - 24} y="36" className="fill-emerald-700 text-xs font-bold">1-eps</text>
      <text x={upperX - 10} y="36" className="fill-emerald-700 text-xs font-bold">1+eps</text>
    </svg>
  );
}

export default function PpoClippedPolicyGradientAnimation() {
  const [ratio, setRatio] = useState(1.24);
  const [advantage, setAdvantage] = useState(1.8);
  const [epsilon, setEpsilon] = useState(0.2);
  const [oldProbability, setOldProbability] = useState(0.32);

  const controlled = useMemo(() => ppoObjective(ratio, advantage, epsilon), [ratio, advantage, epsilon]);
  const newProbability = clamp(oldProbability * ratio, 0.01, 0.99);
  const kl = binaryKl(oldProbability, newProbability);
  const entropy = binaryEntropy(newProbability);
  const batchRows = useMemo(() => BATCH_TEMPLATE.map((sample) => {
    const sampleRatio = clamp(ratio + sample.ratioOffset, 0.2, 2.2);
    const sampleAdvantage = advantage * sample.advantageScale;
    return {
      ...sample,
      advantage: sampleAdvantage,
      ...ppoObjective(sampleRatio, sampleAdvantage, epsilon),
    };
  }), [ratio, advantage, epsilon]);
  const clippedCount = batchRows.filter((row) => row.clippedActive).length;
  const meanObjective = batchRows.reduce((sum, row) => sum + row.objective, 0) / batchRows.length;

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Policy optimization</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">PPO: Clipped Policy Gradient</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              PPO keeps a policy-gradient update from moving too far from the behavior policy used to collect the
              batch. The clipping term limits probability-ratio gains when the update already moved far enough.
            </p>
          </div>
          <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-bold text-slate-700">
            objective = min(ratio * A, clip(ratio) * A)
          </div>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-[0.85fr_1.15fr]">
        <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <SlidersHorizontal size={16} />
            Update controls
          </h3>
          <div className="grid gap-5">
            <Control label="Policy ratio r_t" value={ratio.toFixed(2)}>
              <input min="0.4" max="1.8" step="0.02" type="range" value={ratio} onChange={(event) => setRatio(Number(event.target.value))} />
            </Control>
            <Control label="Advantage A_t" value={advantage.toFixed(1)}>
              <input min="-3" max="3" step="0.1" type="range" value={advantage} onChange={(event) => setAdvantage(Number(event.target.value))} />
            </Control>
            <Control label="Clip epsilon" value={epsilon.toFixed(2)}>
              <input min="0.05" max="0.4" step="0.01" type="range" value={epsilon} onChange={(event) => setEpsilon(Number(event.target.value))} />
            </Control>
            <Control label="Old action probability" value={(oldProbability * 100).toFixed(0) + '%'}>
              <input min="0.08" max="0.72" step="0.01" type="range" value={oldProbability} onChange={(event) => setOldProbability(Number(event.target.value))} />
            </Control>
          </div>
        </div>

        <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <Gauge size={16} />
            Single-sample calculation
          </h3>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <p className="text-xs font-black uppercase tracking-wide text-slate-500">Unclipped</p>
              <p className="mt-2 font-mono text-lg font-black text-slate-950">{controlled.unclipped.toFixed(3)}</p>
              <p className="mt-1 text-sm text-slate-600">ratio * advantage</p>
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <p className="text-xs font-black uppercase tracking-wide text-slate-500">Clipped candidate</p>
              <p className="mt-2 font-mono text-lg font-black text-slate-950">{controlled.clipped.toFixed(3)}</p>
              <p className="mt-1 text-sm text-slate-600">clip ratio = {controlled.clippedRatio.toFixed(2)}</p>
            </div>
          </div>
          <div className={`mt-4 rounded-lg border p-4 ${controlled.clippedActive ? 'border-amber-200 bg-amber-50' : 'border-emerald-200 bg-emerald-50'}`}>
            <p className="text-xs font-black uppercase tracking-wide text-slate-600">Selected PPO objective</p>
            <p className="mt-1 font-mono text-3xl font-black text-slate-950">{controlled.objective.toFixed(3)}</p>
            <p className="mt-2 text-sm leading-6 text-slate-700">
              {controlled.clippedActive
                ? 'The clipped term is active for this advantage sign and ratio.'
                : 'The unclipped term still determines the update for this sample.'}
            </p>
          </div>
        </div>
      </section>

      <div className="grid gap-3 md:grid-cols-4">
        <Metric label="New action probability" value={(newProbability * 100).toFixed(1) + '%'} helper="old probability times ratio" />
        <Metric label="Approx KL" value={kl.toFixed(3)} helper="policy drift proxy" />
        <Metric label="Entropy" value={entropy.toFixed(3)} helper="remaining action uncertainty" />
        <Metric label="Clipped rows" value={`${clippedCount}/${batchRows.length}`} helper="minibatch samples capped" />
      </div>

      <section className="grid min-w-0 gap-4 xl:grid-cols-[0.95fr_1.05fr]">
        <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <BarChart3 size={16} />
            Clipped surrogate curve
          </h3>
          <SurrogateChart advantage={advantage || 0.1} epsilon={epsilon} />
        </div>

        <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <ShieldCheck size={16} />
            Minibatch clipping table
          </h3>
          <div className="max-w-full overflow-x-auto">
            <table className="w-full min-w-[620px] text-left text-sm">
              <thead className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="py-2 pr-3">Sample</th>
                  <th className="py-2 pr-3">Action</th>
                  <th className="py-2 pr-3">Ratio</th>
                  <th className="py-2 pr-3">Advantage</th>
                  <th className="py-2 pr-3">Objective</th>
                  <th className="py-2 pr-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {batchRows.map((row) => (
                  <tr key={row.id}>
                    <td className="py-3 pr-3 font-bold text-slate-950">{row.id}. {row.state}</td>
                    <td className="py-3 pr-3 text-slate-700">{row.action}</td>
                    <td className="py-3 pr-3 font-mono text-slate-950">{row.ratio.toFixed(2)}</td>
                    <td className={`py-3 pr-3 font-mono ${row.advantage >= 0 ? 'text-emerald-700' : 'text-rose-700'}`}>{row.advantage.toFixed(2)}</td>
                    <td className="py-3 pr-3 font-mono text-slate-950">{row.objective.toFixed(3)}</td>
                    <td className="py-3 pr-3">
                      <span className={`rounded px-2 py-1 text-xs font-black uppercase tracking-wide ${row.clippedActive ? 'bg-amber-100 text-amber-800' : 'bg-emerald-100 text-emerald-800'}`}>
                        {row.clippedActive ? 'clipped' : 'unclipped'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 rounded-lg bg-slate-50 p-3 text-sm leading-6 text-slate-700">
            Mean clipped objective: <strong className="text-slate-950">{meanObjective.toFixed(3)}</strong>. Positive
            advantages clip high ratios; negative advantages clip low ratios because the minimum chooses the more
            conservative pressure.
          </p>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-blue-200 bg-blue-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-blue-700">Trust-region intuition</h3>
          <p className="mt-3 text-sm leading-6 text-blue-950">
            PPO approximates a trust region by refusing extra objective gain once the new policy probability ratio is
            outside the epsilon band.
          </p>
        </div>
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-amber-700">
            <AlertTriangle size={16} />
            KL caution
          </h3>
          <p className="mt-3 text-sm leading-6 text-amber-950">
            Clipping is not a proof of monotonic improvement. Real PPO still monitors KL, entropy, reward scale, and
            value-function fit.
          </p>
        </div>
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-emerald-700">Understanding check</h3>
          <p className="mt-3 text-sm leading-6 text-emerald-950">
            Make the advantage negative, then lower the ratio below 1 - epsilon. The clipped candidate should become
            the selected objective.
          </p>
        </div>
      </section>

      <AssessmentPanel lessonId="ppo-clipped-policy-gradient" title="PPO clipped policy-gradient check" />
    </div>
  );
}
