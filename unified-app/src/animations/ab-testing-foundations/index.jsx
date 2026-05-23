import React, { useMemo, useState } from 'react';
import { AlertTriangle, BarChart3, CheckCircle2, GitBranch, RotateCcw, ShieldCheck, SlidersHorizontal } from 'lucide-react';
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

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function AssignmentDiagram({ treatmentShare }) {
  const users = Array.from({ length: 40 }, (_, index) => {
    const threshold = Math.round((treatmentShare / 100) * 40);
    return index < threshold ? 'treatment' : 'control';
  });

  return (
    <LessonPanel>
      <LessonKicker icon={GitBranch}>Random assignment</LessonKicker>
      <div className="ua-lesson-assignment-grid" aria-label="Assigned users">
        {users.map((group, index) => (
          <div
            key={index}
            className={group === 'treatment' ? 'ua-lesson-cell-treatment' : 'ua-lesson-cell-control'}
            title={group}
          />
        ))}
      </div>
      <div className="ua-lesson-assignment-legend">
        <div className="ua-lesson-assignment-note ua-lesson-assignment-note-control">
          <strong>Control</strong>
          <p>Receives the current product or policy.</p>
        </div>
        <div className="ua-lesson-assignment-note ua-lesson-assignment-note-treatment">
          <strong>Treatment</strong>
          <p>Receives the proposed change being tested.</p>
        </div>
      </div>
      <p className="ua-lesson-footnote">
        Randomization is the key design move: it makes the groups comparable before the product change acts.
      </p>
    </LessonPanel>
  );
}

export default function AbTestingFoundationsAnimation() {
  const [baselinePct, setBaselinePct] = useState(12);
  const [liftPct, setLiftPct] = useState(8);
  const [sampleSize, setSampleSize] = useState(12000);
  const [treatmentShare, setTreatmentShare] = useState(50);
  const [mdePct, setMdePct] = useState(5);
  const [guardrailImpactPct, setGuardrailImpactPct] = useState(-1.5);
  const [guardrailThresholdPct, setGuardrailThresholdPct] = useState(-2);

  const metrics = useMemo(() => {
    const treatmentN = Math.round(sampleSize * (treatmentShare / 100));
    const controlN = sampleSize - treatmentN;
    const controlRate = baselinePct / 100;
    const treatmentRate = clamp(controlRate * (1 + liftPct / 100), 0.001, 0.999);
    const diff = treatmentRate - controlRate;
    const pooled = (controlRate * controlN + treatmentRate * treatmentN) / sampleSize;
    const se = Math.sqrt(pooled * (1 - pooled) * ((1 / Math.max(1, controlN)) + (1 / Math.max(1, treatmentN))));
    const z = se === 0 ? 0 : diff / se;
    const pValue = Math.min(1, 2 * (1 - normalCdf(Math.abs(z))));
    const ciLow = diff - 1.96 * se;
    const ciHigh = diff + 1.96 * se;
    const relativeLift = controlRate === 0 ? 0 : diff / controlRate;
    const practical = Math.abs(relativeLift) * 100 >= mdePct;
    const significant = pValue < 0.05;
    const guardrailPass = guardrailImpactPct >= guardrailThresholdPct;
    const allocationRisk = Math.min(treatmentN, controlN) / Math.max(treatmentN, controlN) < 0.35;

    return {
      treatmentN,
      controlN,
      controlRate,
      treatmentRate,
      diff,
      se,
      z,
      pValue,
      ciLow,
      ciHigh,
      relativeLift,
      practical,
      significant,
      guardrailPass,
      allocationRisk,
      decisionReady: significant && practical && guardrailPass && !allocationRisk,
    };
  }, [baselinePct, guardrailImpactPct, guardrailThresholdPct, liftPct, mdePct, sampleSize, treatmentShare]);

  const reset = () => {
    setBaselinePct(12);
    setLiftPct(8);
    setSampleSize(12000);
    setTreatmentShare(50);
    setMdePct(5);
    setGuardrailImpactPct(-1.5);
    setGuardrailThresholdPct(-2);
  };

  const barMax = Math.max(metrics.controlRate, metrics.treatmentRate, 0.02);
  const controlBar = (metrics.controlRate / barMax) * 100;
  const treatmentBar = (metrics.treatmentRate / barMax) * 100;

  return (
    <LessonStage>
      <LessonPanel>
        <div className="ua-lesson-head">
          <div>
            <LessonKicker>Experiment design</LessonKicker>
            <h2>A/B Testing Foundations</h2>
            <p>
              An A/B test estimates the causal effect of a change by randomly assigning comparable users to a
              control or treatment group, then reading a pre-declared metric and guardrails together.
            </p>
          </div>
          <LessonResetButton onClick={reset}>
            <RotateCcw size={16} />
            Reset
          </LessonResetButton>
        </div>
      </LessonPanel>

      <LessonPanel>
        <LessonKicker icon={SlidersHorizontal}>Experiment controls</LessonKicker>
        <div className="ua-lesson-control-grid">
          <label>
            Baseline conversion: {baselinePct}%
            <input type="range" min="1" max="40" value={baselinePct} onChange={(event) => setBaselinePct(Number(event.target.value))} />
            <span>Expected control-group success rate.</span>
          </label>
          <label>
            Treatment lift: {liftPct}%
            <input type="range" min="-20" max="30" value={liftPct} onChange={(event) => setLiftPct(Number(event.target.value))} />
            <span>Relative change caused by the variant.</span>
          </label>
          <label>
            Total sample size: {sampleSize.toLocaleString()}
            <input type="range" min="1000" max="60000" step="1000" value={sampleSize} onChange={(event) => setSampleSize(Number(event.target.value))} />
            <span>More users reduce standard error.</span>
          </label>
          <label>
            Treatment allocation: {treatmentShare}%
            <input type="range" min="10" max="90" step="5" value={treatmentShare} onChange={(event) => setTreatmentShare(Number(event.target.value))} />
            <span>Very uneven splits waste precision.</span>
          </label>
          <label>
            Practical MDE: {mdePct}%
            <input type="range" min="1" max="15" value={mdePct} onChange={(event) => setMdePct(Number(event.target.value))} />
            <span>Minimum relative lift worth acting on.</span>
          </label>
          <label>
            Guardrail impact: {guardrailImpactPct.toFixed(1)}%
            <input type="range" min="-8" max="5" step="0.5" value={guardrailImpactPct} onChange={(event) => setGuardrailImpactPct(Number(event.target.value))} />
            <span>Observed change on a secondary metric such as latency, refunds, or churn.</span>
          </label>
          <label>
            Guardrail breach threshold: {guardrailThresholdPct.toFixed(1)}%
            <input type="range" min="-8" max="0" step="0.5" value={guardrailThresholdPct} onChange={(event) => setGuardrailThresholdPct(Number(event.target.value))} />
            <span>Pre-declared maximum acceptable degradation before blocking launch.</span>
          </label>
        </div>
      </LessonPanel>

      <section className="ua-lesson-stat-grid">
        <LessonStat label="Control group" value={metrics.controlN.toLocaleString()} detail={`${(metrics.controlRate * 100).toFixed(1)}% conversion`} tone="cyan" />
        <LessonStat label="Treatment group" value={metrics.treatmentN.toLocaleString()} detail={`${(metrics.treatmentRate * 100).toFixed(1)}% conversion`} tone="emerald" />
        <LessonStat label="Relative lift" value={`${(metrics.relativeLift * 100).toFixed(1)}%`} detail={`MDE target: ${mdePct}%`} tone={metrics.practical ? 'emerald' : 'amber'} />
        <LessonStat label="p-value" value={`${(metrics.pValue * 100).toFixed(1)}%`} detail={metrics.significant ? 'Below 5% alpha' : 'Not significant yet'} tone={metrics.significant ? 'emerald' : 'amber'} />
      </section>

      <section className="ua-lesson-split-grid">
        <AssignmentDiagram treatmentShare={treatmentShare} />

        <LessonPanel>
          <LessonKicker icon={BarChart3}>Metric readout</LessonKicker>
          <div className="ua-lesson-bar-stack">
            <div>
              <div className="ua-lesson-bar-label">
                <span>Control</span>
                <span>{(metrics.controlRate * 100).toFixed(2)}%</span>
              </div>
              <div className="ua-lesson-bar-track">
                <div className="ua-lesson-bar-fill ua-lesson-bar-fill-control" style={{ width: `${controlBar}%` }} />
              </div>
            </div>
            <div>
              <div className="ua-lesson-bar-label">
                <span>Treatment</span>
                <span>{(metrics.treatmentRate * 100).toFixed(2)}%</span>
              </div>
              <div className="ua-lesson-bar-track">
                <div className="ua-lesson-bar-fill ua-lesson-bar-fill-treatment" style={{ width: `${treatmentBar}%` }} />
              </div>
            </div>
          </div>
          <LessonEquation>
            absolute lift = {(metrics.diff * 100).toFixed(2)} pp<br />
            95% CI = {(metrics.ciLow * 100).toFixed(2)} pp to {(metrics.ciHigh * 100).toFixed(2)} pp<br />
            z = {metrics.z.toFixed(2)}, p = {metrics.pValue.toFixed(3)}
          </LessonEquation>
          <p className="ua-lesson-footnote">
            The same result needs both statistical evidence and practical size. A tiny but significant lift may still
            be too small to ship.
          </p>
        </LessonPanel>
      </section>

      <section className="ua-lesson-callout-grid">
        <LessonCallout tone={metrics.significant ? 'good' : 'warn'}>
          <p className="ua-lesson-callout-title">
            {metrics.significant ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
            Statistical signal
          </p>
          <p>
            {metrics.significant ? 'The observed gap is unlikely under a no-effect baseline.' : 'The observed gap is still plausible under noise.'}
          </p>
        </LessonCallout>
        <LessonCallout tone={metrics.practical ? 'good' : 'warn'}>
          <p className="ua-lesson-callout-title">
            {metrics.practical ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
            Practical size
          </p>
          <p>
            {metrics.practical ? 'The lift clears the minimum effect worth acting on.' : 'The lift is below the pre-declared practical threshold.'}
          </p>
        </LessonCallout>
        <LessonCallout tone={metrics.guardrailPass ? 'good' : 'warn'}>
          <p className="ua-lesson-callout-title">
            <ShieldCheck size={14} />
            Guardrail
          </p>
          <p>
            {metrics.guardrailPass
              ? `Impact ${guardrailImpactPct.toFixed(1)}% stays above the ${guardrailThresholdPct.toFixed(1)}% breach threshold.`
              : `Impact ${guardrailImpactPct.toFixed(1)}% breaches the ${guardrailThresholdPct.toFixed(1)}% guardrail threshold.`}
          </p>
        </LessonCallout>
        <LessonCallout tone={metrics.decisionReady ? 'good' : 'neutral'}>
          <p className="ua-lesson-callout-title">Decision</p>
          <p>
            {metrics.decisionReady
              ? 'Ship candidate: signal, size, guardrails, and allocation all pass.'
              : 'Do not ship automatically. Fix the design, collect more data, or document the tradeoff.'}
          </p>
        </LessonCallout>
      </section>

      <AssessmentPanel lessonId="ab-testing-foundations" />
    </LessonStage>
  );
}
