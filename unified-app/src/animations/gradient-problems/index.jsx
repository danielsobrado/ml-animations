import React, { useMemo, useState } from 'react';
import { Activity, GitBranch, Scissors, SlidersHorizontal, TrendingDown, TrendingUp } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

function buildGradientFlow({ depth, weightScale, activationSlope, residualMix, clipAt }) {
  let gradient = 1;
  const rows = [];
  for (let layer = depth; layer >= 1; layer -= 1) {
    const localDerivative = weightScale * activationSlope + residualMix;
    const unclipped = gradient * localDerivative;
    const clipped = clipAt > 0 ? clamp(unclipped, -clipAt, clipAt) : unclipped;
    rows.push({
      layer,
      incoming: gradient,
      localDerivative,
      unclipped,
      outgoing: clipped,
      clipped: clipped !== unclipped,
    });
    gradient = clipped;
  }
  return rows;
}

function format(value) {
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) return value.toExponential(2);
  return value.toFixed(4);
}

function Metric({ icon: Icon, label, value, helper, tone = 'slate' }) {
  const toneClass = {
    slate: 'border-slate-200 bg-white',
    amber: 'border-amber-200 bg-amber-50',
    rose: 'border-rose-200 bg-rose-50',
    emerald: 'border-emerald-200 bg-emerald-50',
  }[tone];
  return (
    <div className={`rounded-lg border p-4 shadow-sm ${toneClass}`}>
      <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
        <Icon size={16} />
        {label}
      </div>
      <div className="mt-2 text-2xl font-bold text-slate-950">{value}</div>
      <p className="mt-1 text-sm text-slate-600">{helper}</p>
    </div>
  );
}

export default function GradientProblemsAnimation() {
  const [depth, setDepth] = useState(12);
  const [weightScale, setWeightScale] = useState(0.75);
  const [activationSlope, setActivationSlope] = useState(0.7);
  const [residualMix, setResidualMix] = useState(0);
  const [clipAt, setClipAt] = useState(0);

  const rows = useMemo(
    () => buildGradientFlow({ depth, weightScale, activationSlope, residualMix, clipAt }),
    [depth, weightScale, activationSlope, residualMix, clipAt],
  );
  const finalGradient = rows[rows.length - 1].outgoing;
  const localDerivative = rows[0].localDerivative;
  const clippedCount = rows.filter((row) => row.clipped).length;
  const diagnosis = Math.abs(finalGradient) < 0.01
    ? 'vanishing'
    : Math.abs(finalGradient) > 10
      ? 'exploding'
      : 'stable';
  const diagnosisTone = diagnosis === 'stable' ? 'emerald' : diagnosis === 'vanishing' ? 'amber' : 'rose';

  return (
    <div className="min-h-full bg-slate-50">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 p-4 md:p-6">
        <header className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-rose-700">
            <TrendingDown size={17} />
            Backpropagation stability
          </div>
          <h1 className="mt-2 text-2xl font-bold text-slate-950 md:text-3xl">Gradient Problems</h1>
          <p className="mt-2 max-w-3xl text-slate-700">
            Deep networks multiply many local derivatives during backpropagation. Products below one can erase signal;
            products above one can amplify it until updates become unstable.
          </p>
        </header>

        <section className="grid gap-4 lg:grid-cols-[320px_1fr]">
          <aside className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex items-center gap-2 font-semibold text-slate-950">
              <SlidersHorizontal size={18} />
              Chain-rule controls
            </div>
            <div className="mt-5 space-y-4">
              {[
                ['Depth', depth, 4, 30, 1, setDepth],
                ['Weight scale', weightScale, 0.2, 1.8, 0.05, setWeightScale],
                ['Activation slope', activationSlope, 0.1, 1.2, 0.05, setActivationSlope],
                ['Residual path', residualMix, 0, 1, 0.05, setResidualMix],
                ['Clip absolute value', clipAt, 0, 5, 0.25, setClipAt],
              ].map(([label, value, min, max, step, setter]) => (
                <label key={label} className="block">
                  <div className="mb-2 flex justify-between text-sm font-semibold text-slate-700">
                    <span>{label}</span>
                    <span>{Number(value).toFixed(step === 1 ? 0 : 2)}</span>
                  </div>
                  <input
                    type="range"
                    min={min}
                    max={max}
                    step={step}
                    value={value}
                    onChange={(event) => setter(Number(event.target.value))}
                    className="w-full accent-rose-700"
                  />
                </label>
              ))}
            </div>
          </aside>

          <main className="space-y-4">
            <div className="grid gap-4 md:grid-cols-4">
              <Metric icon={Activity} label="Final gradient" value={format(finalGradient)} helper="Signal reaching the earliest layer." tone={diagnosisTone} />
              <Metric icon={GitBranch} label="Local multiplier" value={localDerivative.toFixed(2)} helper="Approximate derivative per layer." />
              <Metric icon={Scissors} label="Clipped layers" value={clippedCount} helper="Layers where clipping changed the value." />
              <Metric icon={diagnosis === 'exploding' ? TrendingUp : TrendingDown} label="Diagnosis" value={diagnosis} helper="Based on the final gradient magnitude." tone={diagnosisTone} />
            </div>

            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <h2 className="text-lg font-bold text-slate-950">Gradient chain ledger</h2>
              <p className="text-sm text-slate-600">
                Start from output gradient 1.0 and multiply backward through each layer. Residual paths add a direct
                route, while clipping limits extreme values after multiplication.
              </p>
              <div className="mt-4 max-h-[520px] overflow-auto rounded-lg border border-slate-200">
                <table className="w-full min-w-[720px] border-collapse text-sm">
                  <thead className="sticky top-0 bg-slate-100 text-left text-xs uppercase tracking-wide text-slate-500">
                    <tr>
                      <th className="px-3 py-2">Layer</th>
                      <th className="px-3 py-2">Incoming grad</th>
                      <th className="px-3 py-2">Local derivative</th>
                      <th className="px-3 py-2">Unclipped</th>
                      <th className="px-3 py-2">Outgoing grad</th>
                      <th className="px-3 py-2">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row) => (
                      <tr key={row.layer} className="border-t border-slate-200">
                        <td className="px-3 py-2 font-bold text-slate-950">{row.layer}</td>
                        <td className="px-3 py-2 font-mono">{format(row.incoming)}</td>
                        <td className="px-3 py-2 font-mono">{format(row.localDerivative)}</td>
                        <td className="px-3 py-2 font-mono">{format(row.unclipped)}</td>
                        <td className="px-3 py-2 font-mono">{format(row.outgoing)}</td>
                        <td className="px-3 py-2">
                          <span className={`rounded-full px-2 py-1 text-xs font-bold ${
                            row.clipped ? 'bg-amber-100 text-amber-800' : 'bg-slate-100 text-slate-700'
                          }`}>
                            {row.clipped ? 'clipped' : 'passed'}
                          </span>
                        </td>
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
                  Set the local multiplier below one, increase depth, and predict how quickly the earliest-layer
                  gradient shrinks.
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">Residual fix</h3>
                <p className="mt-2 text-sm text-slate-700">
                  A residual path gives gradients a direct additive route, so the backward signal is not forced through
                  only weak nonlinear derivatives.
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">Failure mode</h3>
                <p className="mt-2 text-sm text-slate-700">
                  Clipping controls explosion after it appears; it does not fix bad scale, saturated activations, or poor
                  initialization by itself.
                </p>
              </div>
            </section>
          </main>
        </section>

        <AssessmentPanel lessonId="gradient-problems" />
      </div>
    </div>
  );
}
