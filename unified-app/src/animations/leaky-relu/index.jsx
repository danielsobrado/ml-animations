import React, { useMemo, useState } from 'react';
import { Activity, Gauge, LineChart, SlidersHorizontal, Zap } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const EXAMPLES = [
  { id: 'negative', label: 'Negative pre-activation', z: -3.2, upstream: 1.4 },
  { id: 'near-zero', label: 'Near the kink', z: -0.4, upstream: 1.4 },
  { id: 'positive', label: 'Active neuron', z: 2.1, upstream: 1.4 },
];

function leakyRelu(z, alpha) {
  return z >= 0 ? z : alpha * z;
}

function derivative(z, alpha) {
  return z >= 0 ? 1 : alpha;
}

function formatNumber(value) {
  return Number(value.toFixed(3)).toString();
}

function ActivationGraph({ alpha, z }) {
  const points = useMemo(() => {
    const xs = [-4, -3, -2, -1, 0, 1, 2, 3, 4];
    return xs.map((x) => ({ x, y: leakyRelu(x, alpha) }));
  }, [alpha]);

  const xToPct = (x) => 50 + x * 11;
  const yToPct = (y) => 70 - y * 14;
  const path = points.map((p, index) => `${index === 0 ? 'M' : 'L'} ${xToPct(p.x)} ${yToPct(p.y)}`).join(' ');
  const reluPath = points.map((p, index) => `${index === 0 ? 'M' : 'L'} ${xToPct(p.x)} ${yToPct(p.x >= 0 ? p.x : 0)}`).join(' ');
  const active = { x: xToPct(z), y: yToPct(leakyRelu(z, alpha)) };

  return (
    <svg viewBox="0 0 100 100" className="h-72 w-full rounded-lg border border-slate-200 bg-white">
      <line x1="6" y1="70" x2="94" y2="70" stroke="#cbd5e1" strokeWidth="0.8" />
      <line x1="50" y1="8" x2="50" y2="92" stroke="#cbd5e1" strokeWidth="0.8" />
      <path d={reluPath} fill="none" stroke="#94a3b8" strokeWidth="1.6" strokeDasharray="3 3" />
      <path d={path} fill="none" stroke="#2563eb" strokeWidth="2.4" />
      <circle cx={active.x} cy={active.y} r="3" fill="#f97316" stroke="white" strokeWidth="1.5" />
      <text x="8" y="14" className="fill-slate-500 text-[4px]">dashed = ReLU baseline</text>
      <text x="78" y="66" className="fill-slate-500 text-[4px]">z</text>
      <text x="53" y="12" className="fill-slate-500 text-[4px]">f(z)</text>
    </svg>
  );
}

function StatCard({ icon: Icon, label, value, note }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex items-center gap-2 text-sm font-medium text-slate-600">
        <Icon size={16} />
        {label}
      </div>
      <div className="mt-2 text-2xl font-semibold text-slate-950">{value}</div>
      <p className="mt-1 text-sm text-slate-600">{note}</p>
    </div>
  );
}

export default function LeakyReluAnimation() {
  const [exampleId, setExampleId] = useState('negative');
  const [alpha, setAlpha] = useState(0.05);
  const [biasShift, setBiasShift] = useState(0);
  const [upstream, setUpstream] = useState(1.4);

  const baseExample = EXAMPLES.find((item) => item.id === exampleId) ?? EXAMPLES[0];
  const z = baseExample.z + biasShift;
  const output = leakyRelu(z, alpha);
  const localSlope = derivative(z, alpha);
  const backwardGradient = upstream * localSlope;
  const reluOutput = Math.max(0, z);
  const reluGradient = z >= 0 ? upstream : 0;
  const leakLift = Math.abs(backwardGradient - reluGradient);

  const rows = [-3, -1, -0.2, 0, 0.8, 2].map((value) => ({
    z: value,
    relu: Math.max(0, value),
    leaky: leakyRelu(value, alpha),
    slope: derivative(value, alpha),
  }));

  return (
    <div className="min-h-full bg-slate-50 text-slate-900">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-6">
        <header className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <p className="text-sm font-semibold uppercase tracking-wide text-blue-700">Activation functions</p>
              <h1 className="mt-1 text-3xl font-semibold tracking-normal text-slate-950">Leaky ReLU</h1>
              <p className="mt-2 max-w-3xl text-slate-600">
                Compare a hard ReLU gate with a small negative-side slope. The key learning signal is the local derivative:
                negative pre-activations are damped, not completely disconnected.
              </p>
            </div>
            <div className="rounded-lg bg-blue-50 px-4 py-3 text-sm text-blue-950">
              <div className="font-semibold">f(z) = max(alpha z, z)</div>
              <div className="mt-1 text-blue-800">slope is alpha for z below 0 and 1 for z at or above 0</div>
            </div>
          </div>
        </header>

        <section className="grid gap-4 lg:grid-cols-[320px_1fr]">
          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <div className="mb-4 flex items-center gap-2 font-semibold text-slate-800">
              <SlidersHorizontal size={18} />
              Controls
            </div>
            <label className="text-sm font-medium text-slate-700" htmlFor="example">Scenario</label>
            <select
              id="example"
              value={exampleId}
              onChange={(event) => setExampleId(event.target.value)}
              className="mt-1 w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm"
            >
              {EXAMPLES.map((example) => (
                <option key={example.id} value={example.id}>{example.label}</option>
              ))}
            </select>

            <label className="mt-4 block text-sm font-medium text-slate-700" htmlFor="alpha">Negative slope alpha: {alpha.toFixed(2)}</label>
            <input
              id="alpha"
              type="range"
              min="0"
              max="0.3"
              step="0.01"
              value={alpha}
              onChange={(event) => setAlpha(Number(event.target.value))}
              className="mt-2 w-full"
            />

            <label className="mt-4 block text-sm font-medium text-slate-700" htmlFor="bias">Bias shift: {biasShift.toFixed(1)}</label>
            <input
              id="bias"
              type="range"
              min="-2"
              max="2"
              step="0.1"
              value={biasShift}
              onChange={(event) => setBiasShift(Number(event.target.value))}
              className="mt-2 w-full"
            />

            <label className="mt-4 block text-sm font-medium text-slate-700" htmlFor="upstream">Upstream gradient: {upstream.toFixed(1)}</label>
            <input
              id="upstream"
              type="range"
              min="0.2"
              max="3"
              step="0.1"
              value={upstream}
              onChange={(event) => setUpstream(Number(event.target.value))}
              className="mt-2 w-full"
            />
          </div>

          <div className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
            <ActivationGraph alpha={alpha} z={z} />
            <div className="grid gap-4 sm:grid-cols-2">
              <StatCard icon={Gauge} label="Pre-activation z" value={formatNumber(z)} note="Bias shifts decide which side of the kink the neuron sees." />
              <StatCard icon={Activity} label="Forward output" value={formatNumber(output)} note={`Hard ReLU would output ${formatNumber(reluOutput)} for the same z.`} />
              <StatCard icon={LineChart} label="Local slope" value={formatNumber(localSlope)} note="This is the multiplier used by backprop at this activation." />
              <StatCard icon={Zap} label="Backward gradient" value={formatNumber(backwardGradient)} note={`Leaky path preserves ${formatNumber(leakLift)} more gradient than ReLU here.`} />
            </div>
          </div>
        </section>

        <section className="grid gap-4 lg:grid-cols-[1fr_360px]">
          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <h2 className="text-lg font-semibold text-slate-950">Forward and backward ledger</h2>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full min-w-[620px] text-left text-sm">
                <thead className="border-b border-slate-200 text-slate-600">
                  <tr>
                    <th className="py-2 pr-3">z</th>
                    <th className="py-2 pr-3">ReLU output</th>
                    <th className="py-2 pr-3">Leaky output</th>
                    <th className="py-2 pr-3">Leaky slope</th>
                    <th className="py-2 pr-3">Gradient if upstream = {upstream.toFixed(1)}</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={row.z} className="border-b border-slate-100">
                      <td className="py-2 pr-3 font-mono">{row.z}</td>
                      <td className="py-2 pr-3 font-mono">{formatNumber(row.relu)}</td>
                      <td className="py-2 pr-3 font-mono">{formatNumber(row.leaky)}</td>
                      <td className="py-2 pr-3 font-mono">{formatNumber(row.slope)}</td>
                      <td className="py-2 pr-3 font-mono">{formatNumber(upstream * row.slope)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="space-y-4">
            <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
              <h2 className="font-semibold text-amber-950">Predict before running</h2>
              <p className="mt-2 text-sm text-amber-900">
                Set alpha to 0, then move z below zero. The output and gradient match ReLU. Raise alpha and predict which value
                changes more visibly: the forward activation or the backward gradient.
              </p>
            </div>
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <h2 className="font-semibold text-slate-950">Mistake to avoid</h2>
              <p className="mt-2 text-sm text-slate-600">
                Leaky ReLU does not make negative evidence positive. It keeps the sign in the forward pass and keeps a small
                nonzero slope for learning.
              </p>
            </div>
          </div>
        </section>

        <AssessmentPanel lessonId="leaky-relu" />
      </div>
    </div>
  );
}
