import React, { useMemo, useState } from 'react';
import { AlertTriangle, BarChart3, CheckCircle2, Gauge, RotateCcw, Scale, SlidersHorizontal, Target } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';
import {
  LessonCallout,
  LessonEquation,
  LessonKicker,
  LessonPanel,
  LessonResetButton,
  LessonStage,
  LessonStat,
} from '../../components/animation-shell/LessonUi';

function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return sign * y;
}

function normalCdf(x) {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

function zForPower(powerPct) {
  if (powerPct >= 95) return 1.645;
  if (powerPct >= 90) return 1.282;
  if (powerPct >= 85) return 1.036;
  if (powerPct >= 80) return 0.842;
  if (powerPct >= 75) return 0.674;
  return 0.524;
}

function zForAlpha(alphaPct) {
  if (alphaPct <= 1) return 2.576;
  if (alphaPct <= 2) return 2.326;
  if (alphaPct <= 5) return 1.96;
  if (alphaPct <= 10) return 1.645;
  return 1.44;
}

function PowerCurve({ requiredN, plannedN, powerPct }) {
  const points = Array.from({ length: 18 }, (_, index) => {
    const n = 1000 + index * 5000;
    const ratio = Math.sqrt(n / Math.max(1, requiredN));
    const power = normalCdf((zForPower(powerPct) + 1.96) * ratio - 1.96);
    return { n, power };
  });
  const maxN = points[points.length - 1].n;

  return (
    <LessonPanel>
      <LessonKicker icon={BarChart3}>Power curve</LessonKicker>
      <div className="ua-lesson-power-bars">
        {points.map((point) => {
          const isPlannedZone = point.n <= plannedN;
          return (
            <div key={point.n} className="flex flex-1 flex-col items-center justify-end gap-1">
              <div
                className={`w-full rounded-t ${isPlannedZone ? 'bg-emerald-500' : 'bg-cyan-300'}`}
                style={{ height: `${Math.max(6, point.power * 100)}%` }}
                title={`${point.n.toLocaleString()} users: ${(point.power * 100).toFixed(0)}% power`}
              />
            </div>
          );
        })}
      </div>
      <div className="ua-lesson-power-axis">
        <span>1k users</span>
        <span>planned: {plannedN.toLocaleString()}</span>
        <span>{maxN.toLocaleString()} users</span>
      </div>
      <p className="ua-lesson-footnote">
        Power rises with sample size, but by square-root math: halving the detectable effect usually needs about four
        times as many observations.
      </p>
    </LessonPanel>
  );
}

export default function PowerSampleSizeAnimation() {
  const [baselinePct, setBaselinePct] = useState(12);
  const [mdePct, setMdePct] = useState(6);
  const [plannedN, setPlannedN] = useState(16000);
  const [alphaPct, setAlphaPct] = useState(5);
  const [targetPowerPct, setTargetPowerPct] = useState(80);
  const [varianceMultiplier, setVarianceMultiplier] = useState(1);

  const metrics = useMemo(() => {
    const baseline = baselinePct / 100;
    const treatment = Math.min(0.999, baseline * (1 + mdePct / 100));
    const absoluteEffect = treatment - baseline;
    const pooledVariance = varianceMultiplier * (baseline * (1 - baseline) + treatment * (1 - treatment));
    const zAlpha = zForAlpha(alphaPct);
    const zBeta = zForPower(targetPowerPct);
    const perGroupRequired = Math.ceil(((zAlpha + zBeta) ** 2 * pooledVariance) / Math.max(0.000001, absoluteEffect ** 2));
    const totalRequired = perGroupRequired * 2;
    const perGroupPlanned = plannedN / 2;
    const observedSe = Math.sqrt(pooledVariance / Math.max(1, perGroupPlanned));
    const detectableEffect = (zAlpha + zBeta) * observedSe;
    const detectableRelative = baseline === 0 ? 0 : detectableEffect / baseline;
    const effectZ = absoluteEffect / observedSe;
    const achievedPower = normalCdf(effectZ - zAlpha);
    const falseNegativePct = Math.max(0, Math.min(100, (1 - achievedPower) * 100));
    const underpowered = plannedN < totalRequired;

    return {
      absoluteEffect,
      totalRequired,
      perGroupRequired,
      detectableRelative,
      achievedPower,
      falsePositivePct: alphaPct,
      falseNegativePct,
      underpowered,
      varianceInflation: Math.sqrt(varianceMultiplier),
    };
  }, [alphaPct, baselinePct, mdePct, plannedN, targetPowerPct, varianceMultiplier]);

  const reset = () => {
    setBaselinePct(12);
    setMdePct(6);
    setPlannedN(16000);
    setAlphaPct(5);
    setTargetPowerPct(80);
    setVarianceMultiplier(1);
  };

  const plannedRatio = Math.min(100, (plannedN / metrics.totalRequired) * 100);
  const powerRatio = Math.min(100, metrics.achievedPower * 100);

  return (
    <LessonStage>
      <LessonPanel>
        <div className="ua-lesson-head">
          <div>
            <LessonKicker>Experiment planning</LessonKicker>
            <h2>Power & Sample Size</h2>
            <p>
              Power analysis asks whether an experiment has enough observations to detect the smallest effect worth
              acting on while controlling false positives and false negatives.
            </p>
          </div>
          <LessonResetButton onClick={reset}>
            <RotateCcw size={16} />
            Reset
          </LessonResetButton>
        </div>
      </LessonPanel>

      <LessonPanel>
        <LessonKicker icon={SlidersHorizontal}>Planning controls</LessonKicker>
        <div className="ua-lesson-control-grid">
          <label>
            Baseline conversion: {baselinePct}%
            <input type="range" min="2" max="40" value={baselinePct} onChange={(event) => setBaselinePct(Number(event.target.value))} />
            <span>Lower baselines often need more users for the same relative lift.</span>
          </label>
          <label>
            Minimum detectable effect: {mdePct}%
            <input type="range" min="1" max="20" value={mdePct} onChange={(event) => setMdePct(Number(event.target.value))} />
            <span>Small effects require large experiments.</span>
          </label>
          <label>
            Planned total sample: {plannedN.toLocaleString()}
            <input type="range" min="2000" max="90000" step="1000" value={plannedN} onChange={(event) => setPlannedN(Number(event.target.value))} />
            <span>Assumes a balanced 50/50 split.</span>
          </label>
          <label>
            Alpha: {alphaPct}%
            <input type="range" min="1" max="15" value={alphaPct} onChange={(event) => setAlphaPct(Number(event.target.value))} />
            <span>False positive tolerance under no real effect.</span>
          </label>
          <label>
            Target power: {targetPowerPct}%
            <input type="range" min="70" max="95" step="5" value={targetPowerPct} onChange={(event) => setTargetPowerPct(Number(event.target.value))} />
            <span>Chance to detect the MDE when it is real.</span>
          </label>
          <label>
            Variance multiplier: {varianceMultiplier.toFixed(1)}x
            <input type="range" min="0.5" max="3" step="0.1" value={varianceMultiplier} onChange={(event) => setVarianceMultiplier(Number(event.target.value))} />
            <span>Noisier metrics widen standard errors.</span>
          </label>
        </div>
      </LessonPanel>

      <section className="ua-lesson-stat-grid">
        <LessonStat label="Required sample" value={metrics.totalRequired.toLocaleString()} detail={`${metrics.perGroupRequired.toLocaleString()} per group`} tone={metrics.underpowered ? 'amber' : 'emerald'} />
        <LessonStat label="Achieved power" value={`${(metrics.achievedPower * 100).toFixed(0)}%`} detail={`Target: ${targetPowerPct}%`} tone={metrics.achievedPower * 100 >= targetPowerPct ? 'emerald' : 'amber'} />
        <LessonStat label="False positive risk" value={`${metrics.falsePositivePct}%`} detail="Controlled by alpha" tone={alphaPct <= 5 ? 'cyan' : 'amber'} />
        <LessonStat label="False negative risk" value={`${metrics.falseNegativePct.toFixed(0)}%`} detail="Miss a real MDE" tone={metrics.falseNegativePct <= 20 ? 'emerald' : 'rose'} />
      </section>

      <section className="ua-lesson-split-grid">
        <PowerCurve requiredN={metrics.totalRequired} plannedN={plannedN} powerPct={targetPowerPct} />

        <LessonPanel>
          <LessonKicker icon={Gauge}>Design readout</LessonKicker>
          <div className="ua-lesson-bar-stack">
            <div>
              <div className="ua-lesson-bar-label">
                <span>Planned vs required sample</span>
                <span>{plannedRatio.toFixed(0)}%</span>
              </div>
              <div className="ua-lesson-bar-track">
                <div className={`ua-lesson-bar-fill ${metrics.underpowered ? 'ua-lesson-bar-fill-warn' : 'ua-lesson-bar-fill-treatment'}`} style={{ width: `${plannedRatio}%` }} />
              </div>
            </div>
            <div>
              <div className="ua-lesson-bar-label">
                <span>Achieved power</span>
                <span>{(metrics.achievedPower * 100).toFixed(0)}%</span>
              </div>
              <div className="ua-lesson-bar-track">
                <div className={`ua-lesson-bar-fill ${metrics.achievedPower * 100 >= targetPowerPct ? 'ua-lesson-bar-fill-treatment' : 'ua-lesson-bar-fill-risk'}`} style={{ width: `${powerRatio}%` }} />
              </div>
            </div>
          </div>
          <LessonEquation>
            MDE = {(metrics.absoluteEffect * 100).toFixed(2)} percentage points<br />
            detectable with planned N = {(metrics.detectableRelative * 100).toFixed(1)}% relative lift<br />
            noise scale = {metrics.varianceInflation.toFixed(2)}x standard error
          </LessonEquation>
          <p className="ua-lesson-footnote">
            An underpowered experiment can return “not significant” even when a useful effect exists. That is a false
            negative, not proof that the treatment does nothing.
          </p>
        </LessonPanel>
      </section>

      <section className="ua-lesson-callout-grid">
        <LessonCallout tone={metrics.underpowered ? 'warn' : 'good'}>
          <p className="ua-lesson-callout-title">
            {metrics.underpowered ? <AlertTriangle size={14} /> : <CheckCircle2 size={14} />}
            Sample plan
          </p>
          <p>
            {metrics.underpowered ? 'The planned sample is too small for the declared MDE and power target.' : 'The planned sample can support the declared MDE and power target.'}
          </p>
        </LessonCallout>
        <LessonCallout tone="neutral">
          <p className="ua-lesson-callout-title">
            <Scale size={14} />
            Error tradeoff
          </p>
          <p>
            Lower alpha reduces false positives, but usually demands more sample or accepts lower power.
          </p>
        </LessonCallout>
        <LessonCallout tone="neutral">
          <p className="ua-lesson-callout-title">
            <Target size={14} />
            Practical target
          </p>
          <p>
            Choose the MDE from product value before the experiment, not from the observed result afterward.
          </p>
        </LessonCallout>
      </section>

      <AssessmentPanel lessonId="power-sample-size" />
    </LessonStage>
  );
}
