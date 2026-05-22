import React, { useMemo, useState } from 'react';
import { CalendarRange, RefreshCw, Shield, ShieldCheck, ToggleLeft, ToggleRight } from 'lucide-react';

const OBSERVATIONS = [
  { id: 'traffic', y: 72, truth: 66, aleatoric: 0.22, epistemic: 0.09 },
  { id: 'support', y: 58, truth: 63, aleatoric: 0.17, epistemic: 0.11 },
  { id: 'conversion', y: 38, truth: 32, aleatoric: 0.19, epistemic: 0.08 },
  { id: 'fraud', y: 89, truth: 93, aleatoric: 0.24, epistemic: 0.14 },
  { id: 'churn', y: 44, truth: 40, aleatoric: 0.13, epistemic: 0.06 },
  { id: 'risk', y: 61, truth: 55, aleatoric: 0.15, epistemic: 0.13 },
];

const COVERAGE_Z = {
  0.7: 0.84,
  0.75: 1.15,
  0.8: 1.28,
  0.85: 1.44,
  0.9: 1.64,
  0.95: 1.96,
};

const MODES = {
  inDist: {
    label: 'In distribution',
    ood: 0.0,
    description: 'Inputs are similar to training conditions.',
  },
  domainShift: {
    label: 'Domain shift',
    ood: 0.14,
    description: 'Feature patterns moved, but labels keep the same meaning.',
  },
  farOOD: {
    label: 'Out-of-domain',
    ood: 0.25,
    description: 'New context; predictions can remain confidently wrong.',
  },
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function predictInterval(prediction, mode, alpha, epistemic, aleatoric, ood, abstainLimit) {
  const z = COVERAGE_Z[alpha] || COVERAGE_Z[0.9];
  const sigma = clamp((aleatoric * 0.95 + epistemic * 0.85 + mode.ood * ood * 0.6), 0.6, 12);
  const half = sigma * z * (1 + abstainLimit / 100);
  const lower = clamp(prediction - half, 0, 100);
  const upper = clamp(prediction + half, 0, 100);
  return { lower, upper, width: upper - lower, sigma, half, confidence: 1 / (1 + sigma) };
}

function riskColor(risk, min, max) {
  if (risk < min) return 'bg-emerald-100 border-emerald-400 text-emerald-950';
  if (risk < max) return 'bg-amber-100 border-amber-400 text-amber-950';
  return 'bg-rose-100 border-rose-400 text-rose-950';
}

function Stat({ label, value, detail }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <p className="text-xs font-black uppercase tracking-wide text-slate-500">{label}</p>
      <strong className="mt-1 block text-2xl font-black text-slate-900">{value}</strong>
      <span className="mt-2 block text-xs text-slate-600">{detail}</span>
    </div>
  );
}

function IntervalRow({ item, prediction, interval, covered, deferred }) {
  const style = deferred ? 'border-rose-200 bg-rose-50 text-rose-900' : 'border-slate-200 bg-slate-50 text-slate-900';
  return (
    <div className={`rounded-lg border ${style} p-3 text-sm`}>
      <div className="mb-1 flex items-center justify-between">
        <strong>{item.id}</strong>
        <span className="font-black">{prediction.toFixed(1)} ± {interval.half.toFixed(1)}</span>
      </div>
      <div className="h-2 rounded bg-slate-200 mb-2 overflow-hidden">
        <div
          className="h-2 rounded bg-cyan-500"
          style={{
            width: `${prediction}%`,
            marginLeft: `${prediction - Math.max(interval.half, 1)}%`,
          }}
        />
      </div>
      <p className="text-xs leading-6">
        True value: <strong>{item.truth}</strong> | covered: <strong>{covered ? 'yes' : 'no'}</strong> | status: <strong>{deferred ? 'deferred' : 'served'}</strong>
      </p>
    </div>
  );
}

export default function UncertaintyEstimation() {
  const [modeId, setModeId] = useState('inDist');
  const [coverage, setCoverage] = useState(0.9);
  const [epistemicWeight, setEpistemicWeight] = useState(1);
  const [aleatoricWeight, setAleatoricWeight] = useState(1);
  const [oodWeight, setOodWeight] = useState(1);
  const [abstainLimit, setAbstainLimit] = useState(60);
  const [useConformal, setUseConformal] = useState(true);

  const mode = MODES[modeId];
  const rows = useMemo(() => {
    return OBSERVATIONS.map((item) => {
      const interval = predictInterval(
        item.y,
        mode,
        coverage,
        item.epistemic * epistemicWeight,
        item.aleatoric * aleatoricWeight,
        oodWeight,
        abstainLimit,
      );
      const covered = item.truth >= interval.lower && item.truth <= interval.upper;
      const deferred = useConformal && interval.width > abstainLimit;
      return {
        id: item.id,
        truth: item.truth,
        prediction: item.y,
        interval,
        covered,
        deferred,
        width: interval.width,
      };
    });
  }, [modeId, mode, coverage, epistemicWeight, aleatoricWeight, oodWeight, abstainLimit, useConformal]);

  const coverageRate = (rows.filter((row) => row.covered).length / rows.length) * 100;
  const deferRate = (rows.filter((row) => row.deferred).length / rows.length) * 100;
  const meanWidth = rows.reduce((acc, row) => acc + row.width, 0) / rows.length;
  const avgConfidence = rows.reduce((acc, row) => acc + row.interval.confidence, 0) / rows.length;
  const unstableRows = rows.filter((row) => row.deferred || row.width > 25).length;

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Decision risk</p>
            <h1 className="mt-1 text-2xl font-black text-slate-950">Uncertainty Estimation</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Separate uncertainty sources and tune policies so confident decisions are made only when the signal is reliable.
            </p>
          </div>
          <button
            type="button"
            onClick={() => {
              setModeId('inDist');
              setCoverage(0.9);
              setEpistemicWeight(1);
              setAleatoricWeight(1);
              setOodWeight(1);
              setAbstainLimit(60);
              setUseConformal(true);
            }}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-bold text-slate-800"
          >
            <RefreshCw size={16} />
            Reset
          </button>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Sources and controls</h3>
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <label className="text-sm font-bold text-slate-700">
              Deployment regime
              <select value={modeId} onChange={(event) => setModeId(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-3 py-2">
                {Object.entries(MODES).map(([id, current]) => (
                  <option key={id} value={id}>{current.label}</option>
                ))}
              </select>
            </label>
            <label className="text-sm font-bold text-slate-700">
              Target coverage
              <span className="ml-2 text-slate-500">{(coverage * 100).toFixed(0)}%</span>
              <input
                type="range"
                min={0.7}
                max={0.95}
                step={0.05}
                value={coverage}
                onChange={(event) => setCoverage(Number(event.target.value))}
                className="mt-2 w-full accent-cyan-700"
              />
            </label>
            <label className="text-sm font-bold text-slate-700">
              Epistemic multiplier
              <span className="ml-2 text-slate-500">{epistemicWeight.toFixed(1)}x</span>
              <input
                type="range"
                min={0.2}
                max={2.5}
                step={0.1}
                value={epistemicWeight}
                onChange={(event) => setEpistemicWeight(Number(event.target.value))}
                className="mt-2 w-full accent-cyan-700"
              />
            </label>
            <label className="text-sm font-bold text-slate-700">
              Aleatoric multiplier
              <span className="ml-2 text-slate-500">{aleatoricWeight.toFixed(1)}x</span>
              <input
                type="range"
                min={0.2}
                max={2.0}
                step={0.1}
                value={aleatoricWeight}
                onChange={(event) => setAleatoricWeight(Number(event.target.value))}
                className="mt-2 w-full accent-cyan-700"
              />
            </label>
          </div>
          <p className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm leading-6 text-slate-700">
            {mode.description}
          </p>
        </div>

        <div className="grid gap-3">
          <Stat label="Observed coverage" value={`${coverageRate.toFixed(1)}%`} detail="fraction where true value is inside interval" />
          <Stat label="Mean interval width" value={`${meanWidth.toFixed(1)} pts`} detail="narrower may under-cover" />
          <Stat label="Deferred proportion" value={`${deferRate.toFixed(1)}%`} detail={useConformal ? 'defer-if-wide policy' : 'policy off'} />
          <div className="grid gap-3">
            <label className="flex items-center justify-between rounded-lg border border-slate-200 bg-white p-3">
              <span className="text-sm font-bold text-slate-700">Use abstain policy</span>
              <button type="button" onClick={() => setUseConformal((value) => !value)} className="text-slate-700">
                {useConformal ? <ToggleRight size={18} /> : <ToggleLeft size={18} />}
              </button>
            </label>
            <label className="text-sm font-bold text-slate-700">
              OOD multiplier
              <span className="ml-2 text-slate-500">{oodWeight.toFixed(2)}x</span>
              <input
                type="range"
                min={0.4}
                max={2.2}
                step={0.1}
                value={oodWeight}
                onChange={(event) => setOodWeight(Number(event.target.value))}
                className="mt-2 w-full accent-cyan-700"
              />
            </label>
            <label className="text-sm font-bold text-slate-700">
              Deferral threshold
              <span className="ml-2 text-slate-500">{abstainLimit.toFixed(0)}</span>
              <input
                type="range"
                min={40}
                max={120}
                step={5}
                value={abstainLimit}
                onChange={(event) => setAbstainLimit(Number(event.target.value))}
                className="mt-2 w-full accent-cyan-700"
              />
            </label>
          </div>
        </div>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <CalendarRange size={16} />
          Toy prediction intervals and truth checks
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          {rows.map((row) => (
            <IntervalRow
              key={row.id}
              item={row}
              prediction={row.prediction}
              interval={row.interval}
              covered={row.covered}
              deferred={row.deferred}
            />
          ))}
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-3">
        <div className={`rounded-lg border p-4 ${riskColor(unstableRows, 1, 3)}`}>
          <h3 className="text-sm font-black uppercase tracking-wide">Signal health</h3>
          <p className="mt-2 text-sm leading-6">
            Confidence is approximately <strong>{(avgConfidence * 100).toFixed(0)}%</strong>.
            This combines modeled aleatoric and epistemic spread in a toy way.
          </p>
        </div>
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
          <h3 className="text-sm font-black uppercase tracking-wide text-slate-700">Interpretation guide</h3>
          <p className="mt-2 text-sm leading-6 text-slate-700">
            High confidence means low spread around the prediction.
            Wide intervals are not a model weakness by themselves—they are often a warning that the data regime changed.
          </p>
        </div>
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
          <h3 className="text-sm font-black uppercase tracking-wide text-slate-700">Deployment policy</h3>
          <p className="mt-2 text-sm leading-6 text-slate-700">
            Use an abstain policy when interval width exceeds your reliability budget.
            Better to defer with a clear fallback than ship a high-risk prediction.
          </p>
        </div>
      </section>

      <section className="rounded-lg border border-cyan-200 bg-cyan-50 p-4 text-sm leading-6 text-cyan-950">
        <div className="flex items-center gap-2 font-black uppercase tracking-wide text-cyan-700">
          <Shield size={16} />
          Quick check
        </div>
        <p className="mt-2">
          {coverageRate >= 70
            ? 'Coverage is acceptable for this toy setup, but monitor failures where the signal was deferred or missed entirely.'
            : 'Coverage is low. Increase target coverage, improve calibration, or tighten feature scope.'}
        </p>
        <p className="mt-2">
          <strong className="font-black">Conformal-style reminder:</strong> empirical coverage depends on calibration data and exchangeability, not only interval width.
        </p>
      </section>

      <section className="rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm leading-6 text-emerald-950">
        <div className="flex items-center gap-2 font-black uppercase tracking-wide text-emerald-700">
          <ShieldCheck size={16} />
          Lab hint
        </div>
        <p className="mt-2">
          Try moving to domain shift, set aleatoric to 2.0, and adjust abstain threshold to keep unstable cases deferred while preserving enough throughput.
        </p>
      </section>
    </div>
  );
}

