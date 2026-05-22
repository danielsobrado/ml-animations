import React, { useMemo, useState } from 'react';
import { AlertTriangle, BarChart3, Calculator, RotateCcw, ShieldCheck, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const POPULATION = 1000;

function pct(value) {
  return `${Math.round(value * 100)}%`;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function computeBayes(priorPct, sensitivityPct, falsePositivePct) {
  const prior = priorPct / 100;
  const sensitivity = sensitivityPct / 100;
  const falsePositive = falsePositivePct / 100;
  const classCount = POPULATION * prior;
  const otherCount = POPULATION - classCount;
  const truePositive = classCount * sensitivity;
  const falseAlarm = otherCount * falsePositive;
  const positiveTotal = truePositive + falseAlarm;
  const posterior = positiveTotal === 0 ? 0 : truePositive / positiveTotal;

  return {
    prior,
    sensitivity,
    falsePositive,
    classCount,
    otherCount,
    truePositive,
    falseAlarm,
    positiveTotal,
    posterior,
  };
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

function Segment({ label, value, color, total }) {
  const width = total === 0 ? 0 : clamp((value / total) * 100, 0, 100);
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-sm font-bold text-slate-700">
        <span>{label}</span>
        <span>{Math.round(value)} of {Math.round(total)}</span>
      </div>
      <div className="h-3 overflow-hidden rounded-full bg-slate-100">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${width}%` }} />
      </div>
    </div>
  );
}

function PersonGrid({ stats }) {
  const cells = Array.from({ length: 100 }, (_, index) => {
    const person = (index + 0.5) * 10;
    if (person <= stats.truePositive) return 'bg-emerald-500';
    if (person <= stats.truePositive + stats.classCount - stats.truePositive) return 'bg-emerald-100';
    if (person <= stats.classCount + stats.falseAlarm) return 'bg-amber-500';
    return 'bg-slate-200';
  });

  return (
    <div className="grid grid-cols-10 gap-1 rounded-lg bg-slate-50 p-3" aria-label="Population evidence grid">
      {cells.map((className, index) => (
        <span key={index} className={`h-4 rounded-sm ${className}`} />
      ))}
    </div>
  );
}

export default function BayesRuleMLAnimation() {
  const [prior, setPrior] = useState(8);
  const [sensitivity, setSensitivity] = useState(86);
  const [falsePositive, setFalsePositive] = useState(12);
  const [threshold, setThreshold] = useState(70);

  const stats = useMemo(() => computeBayes(prior, sensitivity, falsePositive), [prior, sensitivity, falsePositive]);
  const sweep = useMemo(
    () => Array.from({ length: 12 }, (_, index) => {
      const rate = 2 + index * 4;
      return { rate, posterior: computeBayes(prior, sensitivity, rate).posterior };
    }),
    [prior, sensitivity],
  );
  const firstUseful = sweep.find((point) => point.posterior * 100 >= threshold);

  const reset = () => {
    setPrior(8);
    setSensitivity(86);
    setFalsePositive(12);
    setThreshold(70);
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Probability updates</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Bayes Rule for ML</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Bayes rule turns a model signal into a posterior by accounting for the base rate, true positive rate,
              and false alarms. The normalizer is all cases that could have produced the same evidence.
            </p>
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
          Evidence controls
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Base rate: {prior}%
            <input type="range" min="1" max="50" value={prior} onChange={(event) => setPrior(Number(event.target.value))} />
            <span className="text-xs font-semibold text-slate-500">How common the class is before evidence.</span>
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Hit rate: {sensitivity}%
            <input type="range" min="45" max="99" value={sensitivity} onChange={(event) => setSensitivity(Number(event.target.value))} />
            <span className="text-xs font-semibold text-slate-500">P(signal | class).</span>
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            False alarm: {falsePositive}%
            <input type="range" min="1" max="45" value={falsePositive} onChange={(event) => setFalsePositive(Number(event.target.value))} />
            <span className="text-xs font-semibold text-slate-500">P(signal | not class).</span>
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Useful threshold: {threshold}%
            <input type="range" min="40" max="95" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
            <span className="text-xs font-semibold text-slate-500">Posterior needed for action.</span>
          </label>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <Stat label="Prior" value={pct(stats.prior)} detail={`${Math.round(stats.classCount)} in ${POPULATION} before evidence.`} />
        <Stat label="Posterior" value={pct(stats.posterior)} detail="P(class | positive signal)." />
        <Stat label="Evidence mix" value={`${Math.round(stats.truePositive)} / ${Math.round(stats.positiveTotal)}`} detail="True positives among all positive signals." />
      </section>

      <section className="grid gap-6 xl:grid-cols-[1fr_1fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <BarChart3 size={16} />
            Where positive evidence comes from
          </div>
          <PersonGrid stats={stats} />
          <div className="mt-4 grid gap-3">
            <Segment label="True positives" value={stats.truePositive} total={stats.positiveTotal} color="bg-emerald-500" />
            <Segment label="False alarms" value={stats.falseAlarm} total={stats.positiveTotal} color="bg-amber-500" />
            <Segment label="All class cases" value={stats.classCount} total={POPULATION} color="bg-emerald-300" />
          </div>
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <Calculator size={16} />
            Posterior calculation
          </div>
          <div className="rounded-lg bg-slate-950 p-4 font-mono text-sm leading-7 text-cyan-100">
            numerator = P(signal|class) * P(class)<br />
            = {stats.sensitivity.toFixed(2)} * {stats.prior.toFixed(2)} = {(stats.sensitivity * stats.prior).toFixed(3)}<br />
            denominator = numerator + P(signal|not class) * P(not class)<br />
            posterior = {pct(stats.posterior)}
          </div>
          <p className="mt-4 text-sm leading-6 text-slate-700">
            The denominator is the competition: true positives plus false alarms. Rare classes need very low false
            alarm rates before a positive signal becomes decisive.
          </p>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <BarChart3 size={16} />
            False-alarm sensitivity sweep
          </div>
          <svg viewBox="0 0 430 220" role="img" aria-label="Posterior across false positive rates" className="h-auto w-full rounded-lg bg-slate-50">
            <rect x="38" y="24" width="360" height="148" rx="8" fill="#ffffff" stroke="#cbd5e1" />
            <line x1="38" x2="398" y1={172 - threshold * 1.48} y2={172 - threshold * 1.48} stroke="#f97316" strokeWidth="3" strokeDasharray="6 6" />
            <polyline
              fill="none"
              stroke="#0891b2"
              strokeWidth="4"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={sweep.map((point, index) => `${44 + index * 31},${172 - point.posterior * 148}`).join(' ')}
            />
            {sweep.map((point, index) => (
              <circle key={point.rate} cx={44 + index * 31} cy={172 - point.posterior * 148} r="4" fill={point.rate === falsePositive ? '#0f172a' : '#0891b2'} />
            ))}
            <text x="218" y="205" textAnchor="middle" fontSize="12" fontWeight="800" fill="#475569">false positive rate rises left to right</text>
            <text x="20" y="100" transform="rotate(-90 20 100)" textAnchor="middle" fontSize="12" fontWeight="800" fill="#475569">posterior</text>
          </svg>
        </div>

        <div className="space-y-4">
          <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-4">
            <p className="text-xs font-black uppercase tracking-wide text-cyan-700">Predict before running</p>
            <p className="mt-2 text-sm leading-6 text-cyan-950">
              Lower the base rate, then predict whether increasing hit rate or lowering false alarms moves the posterior more.
            </p>
          </div>
          <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
            <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-amber-700">
              <AlertTriangle size={14} />
              Failure mode
            </p>
            <p className="mt-2 text-sm leading-6 text-amber-950">
              A high hit rate does not guarantee a high posterior when the class is rare and false positives are common.
            </p>
          </div>
          <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4">
            <p className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-emerald-700">
              <ShieldCheck size={14} />
              Action check
            </p>
            <p className="mt-2 text-sm leading-6 text-emerald-950">
              {firstUseful
                ? `At this base rate and hit rate, posterior reaches ${threshold}% only when false alarms are about ${firstUseful.rate}% or lower.`
                : `No false-alarm setting in this sweep reaches the ${threshold}% action threshold.`}
            </p>
          </div>
        </div>
      </section>

      <AssessmentPanel lessonId="bayes-rule-ml" />
    </div>
  );
}
