import React from 'react';
import CausalConceptLesson from '../_shared/CausalConceptLesson';

const pct = (value) => `${value.toFixed(0)}%`;

const config = {
  lessonId: 'time-series-forecasting-track',
  kicker: 'Temporal ML',
  title: 'Time Series & Forecasting',
  description: 'Forecasting needs time-aware validation, lagged features, seasonality checks, leakage controls, exogenous variables, and backtests that mimic future deployment.',
  controls: [
    { id: 'seasonality', label: 'Seasonality strength', min: 0, max: 100, step: 5, defaultValue: 60, format: pct, help: 'Recurring patterns that lag features or seasonal terms should capture.' },
    { id: 'leakage', label: 'Future leakage risk', min: 0, max: 100, step: 5, defaultValue: 25, format: pct, help: 'How much future information sneaks into features or validation.' },
    { id: 'backtests', label: 'Rolling backtests', min: 1, max: 12, step: 1, defaultValue: 5, format: (value) => value.toLocaleString(), help: 'More rolling windows stress-test stability across time.' },
  ],
  compute(values) {
    const coverage = Math.min(100, values.backtests * 9 + values.seasonality * 0.35);
    const reliability = Math.max(0, coverage - values.leakage * 0.8);
    return {
      stats: [
        { label: 'Backtest coverage', value: pct(coverage), detail: 'Rolling-window evidence', tone: coverage > 60 ? 'emerald' : 'amber' },
        { label: 'Leakage penalty', value: pct(values.leakage), detail: 'Future data exposure', tone: values.leakage > 35 ? 'rose' : 'cyan' },
        { label: 'Seasonal signal', value: pct(values.seasonality), detail: 'Pattern to model', tone: 'cyan' },
        { label: 'Forecast readiness', value: pct(reliability), detail: 'Qualitative score', tone: reliability > 55 ? 'emerald' : 'amber' },
      ],
      bars: [
        { label: 'Rolling split coverage', value: pct(coverage), width: coverage, color: 'bg-emerald-500' },
        { label: 'Future leakage risk', value: pct(values.leakage), width: values.leakage, color: 'bg-rose-500' },
        { label: 'Seasonality captured', value: pct(values.seasonality), width: values.seasonality, color: 'bg-cyan-500' },
      ],
      formulaLines: [
        'rolling split: train[0:t] -> validate[t+1:t+h]',
        'lag feature: y[t-k], not y[t+h]',
        'metrics: MAE, RMSE, MAPE, pinball loss',
      ],
      readout: 'A random row split can look excellent while leaking the future. Rolling backtests make the evaluation chronological.',
      steps: [
        { title: 'Respect time order', pass: values.leakage <= 25, body: values.leakage <= 25 ? 'Future information is mostly blocked.' : 'Leakage risk is high; rebuild features and splits by timestamp.' },
        { title: 'Backtest enough windows', pass: values.backtests >= 4, body: values.backtests >= 4 ? 'Multiple cutoffs test stability across regimes.' : 'One cutoff is too brittle for a forecasting workflow.' },
        { title: 'Model temporal structure', pass: values.seasonality >= 30, body: values.seasonality >= 30 ? 'Seasonality and lag features matter in this scenario.' : 'Simple baselines may be competitive when temporal pattern is weak.' },
      ],
      takeaway: 'Forecasting is mostly about time discipline: features, labels, splits, and metrics must all be aligned to what was knowable at prediction time.',
    };
  },
};

export default function TimeSeriesForecastingTrackAnimation() {
  return <CausalConceptLesson config={config} />;
}
