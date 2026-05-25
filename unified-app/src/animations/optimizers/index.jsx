import React, { useMemo, useState } from 'react';
import { CheckCircle2, RotateCcw, SlidersHorizontal, TrendingDown, XCircle } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';
import OptimizerLandscape3D from './OptimizerLandscape3D';
import { OPTIMIZERS, loss, lossColor, project, simulate } from './optimizerModel';

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

function Stat({ label, value, detail }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <p className="text-xs font-black uppercase tracking-wide text-slate-500">{label}</p>
      <strong className="mt-1 block text-2xl font-black text-slate-950">{value}</strong>
      <span className="text-sm text-slate-600">{detail}</span>
    </div>
  );
}

function AnswerBadge({ selected, correct }) {
  if (!selected) return null;
  const isCorrect = selected === correct;
  const Icon = isCorrect ? CheckCircle2 : XCircle;
  return (
    <p className={`mt-3 inline-flex items-center gap-2 rounded border px-3 py-2 text-sm font-bold ${isCorrect ? 'border-emerald-200 bg-emerald-50 text-emerald-800' : 'border-rose-200 bg-rose-50 text-rose-800'}`}>
      <Icon size={16} />
      {isCorrect ? 'Matches the simulated update.' : 'Not for the current settings.'}
    </p>
  );
}

export default function OptimizersAnimation() {
  const [optimizer, setOptimizer] = useState('adam');
  const [learningRate, setLearningRate] = useState(0.18);
  const [momentum, setMomentum] = useState(0.85);
  const [batchSize, setBatchSize] = useState(8);
  const [steps, setSteps] = useState(18);
  const [movePrediction, setMovePrediction] = useState(null);
  const [lossPrediction, setLossPrediction] = useState(null);

  const allPaths = useMemo(
    () => Object.fromEntries(
      Object.keys(OPTIMIZERS).map((id) => [
        id,
        simulate({ optimizer: id, learningRate, momentum, batchSize, steps }),
      ]),
    ),
    [learningRate, momentum, batchSize, steps],
  );
  const path = allPaths[optimizer];
  const finalPoint = path[path.length - 1];
  const bestPoint = path.reduce((best, point) => (point.loss < best.loss ? point : best), path[0]);
  const startLoss = path[0].loss;
  const improvement = Math.max(0, 1 - finalPoint.loss / startLoss);
  const firstStep = path[1] || path[0];
  const firstDelta = [
    firstStep.theta[0] - path[0].theta[0],
    firstStep.theta[1] - path[0].theta[1],
  ];
  const moveAnswer = `${firstDelta[0] >= 0 ? 'right' : 'left'}-${firstDelta[1] >= 0 ? 'up' : 'down'}`;
  const lossAnswer = finalPoint.loss < startLoss - 0.02 ? 'lower' : finalPoint.loss > startLoss + 0.02 ? 'higher' : 'similar';

  const reset = () => {
    setOptimizer('adam');
    setLearningRate(0.18);
    setMomentum(0.85);
    setBatchSize(8);
    setSteps(18);
    setMovePrediction(null);
    setLossPrediction(null);
  };

  const contourCells = [];
  for (let row = 0; row < 7; row += 1) {
    for (let col = 0; col < 11; col += 1) {
      const x = -5.5 + col * 0.55;
      const y = -0.2 + row * 0.65;
      contourCells.push({ x: 60 + col * 42, y: 62 + row * 38, color: lossColor(loss([x, y])) });
    }
  }

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Training dynamics</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Optimizers: SGD, Momentum, and Adam</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              Optimization turns gradients into parameter updates. Compare plain mini-batch SGD, velocity-based
              momentum, and Adam on the same curved loss surface.
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
          Optimizer controls
        </div>
        <div className="grid gap-4 lg:grid-cols-[1.2fr_1fr_1fr_1fr]">
          <div className="grid gap-2">
            <span className="text-sm font-bold text-slate-700">Update rule</span>
            <div className="grid grid-cols-3 gap-2">
              {Object.entries(OPTIMIZERS).map(([id, config]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setOptimizer(id)}
                  className={`rounded-lg border px-3 py-2 text-sm font-black transition ${optimizer === id ? 'border-blue-500 bg-blue-600 text-white' : 'border-slate-200 bg-slate-50 text-slate-700'}`}
                >
                  {config.label}
                </button>
              ))}
            </div>
          </div>
          <Control label="Learning rate" value={learningRate.toFixed(2)}>
            <input min="0.04" max="0.36" step="0.02" type="range" value={learningRate} onChange={(event) => setLearningRate(Number(event.target.value))} />
          </Control>
          <Control label="Momentum beta1" value={momentum.toFixed(2)}>
            <input min="0" max="0.95" step="0.05" type="range" value={momentum} onChange={(event) => setMomentum(Number(event.target.value))} />
          </Control>
          <Control label="Mini-batch size" value={batchSize}>
            <input min="1" max="64" step="1" type="range" value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} />
          </Control>
        </div>
        <div className="mt-4 grid gap-4 lg:grid-cols-[1fr_3fr]">
          <Control label="Training steps" value={steps}>
            <input min="4" max="32" step="1" type="range" value={steps} onChange={(event) => setSteps(Number(event.target.value))} />
          </Control>
          <p className="rounded-lg bg-slate-50 p-3 text-sm leading-6 text-slate-700">
            <strong className="text-slate-950">{OPTIMIZERS[optimizer].label}:</strong> {OPTIMIZERS[optimizer].detail}
            {' '}Small batches add noisy gradients; larger batches smooth the path but cost more computation per step.
          </p>
        </div>
      </section>

      <div className="grid gap-3 md:grid-cols-4">
        <Stat label="Final loss" value={finalPoint.loss.toFixed(3)} detail={`started at ${startLoss.toFixed(3)}`} />
        <Stat label="Best loss" value={bestPoint.loss.toFixed(3)} detail={`at step ${bestPoint.step}`} />
        <Stat label="Improvement" value={`${Math.round(improvement * 100)}%`} detail="from starting loss" />
        <Stat label="Noise scale" value={(0.42 / Math.sqrt(batchSize)).toFixed(2)} detail="mini-batch gradient jitter" />
      </div>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Predict the update</h3>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
          Before reading the path, predict the first parameter movement and the final loss trend for the selected
          optimizer. The check uses the same gradient, noise, and update rule as the visualizations.
        </p>
        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">First step direction</p>
            <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
              {[
                ['right-down', 'right + down'],
                ['right-up', 'right + up'],
                ['left-down', 'left + down'],
                ['left-up', 'left + up'],
              ].map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setMovePrediction(id)}
                  className={`min-h-[44px] rounded border px-3 py-2 text-sm font-bold ${movePrediction === id ? 'border-slate-900 bg-slate-900 text-white' : 'border-slate-200 bg-white text-slate-700'}`}
                >
                  {label}
                </button>
              ))}
            </div>
            <AnswerBadge selected={movePrediction} correct={moveAnswer} />
          </div>

          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">Final loss after selected steps</p>
            <div className="mt-3 grid grid-cols-3 gap-2">
              {[
                ['lower', 'lower'],
                ['similar', 'about same'],
                ['higher', 'higher'],
              ].map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setLossPrediction(id)}
                  className={`min-h-[44px] rounded border px-3 py-2 text-sm font-bold ${lossPrediction === id ? 'border-slate-900 bg-slate-900 text-white' : 'border-slate-200 bg-white text-slate-700'}`}
                >
                  {label}
                </button>
              ))}
            </div>
            <AnswerBadge selected={lossPrediction} correct={lossAnswer} />
          </div>
        </div>
      </section>

      <OptimizerLandscape3D paths={allPaths} activeOptimizer={optimizer} />

      <div className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <TrendingDown size={16} />
            Parameter path on the loss surface
          </h3>
          <svg viewBox="0 0 540 360" className="mt-4 h-auto w-full rounded-lg border border-slate-200 bg-slate-50" role="img" aria-label="Optimizer path on a loss surface">
            {contourCells.map((cell) => (
              <rect key={`${cell.x}-${cell.y}`} x={cell.x} y={cell.y} width="42" height="38" fill={cell.color} opacity="0.8" />
            ))}
            <path
              d={path.map((point, index) => {
                const { cx, cy } = project(point.theta);
                return `${index === 0 ? 'M' : 'L'} ${cx.toFixed(1)} ${cy.toFixed(1)}`;
              }).join(' ')}
              fill="none"
              stroke="#2563eb"
              strokeWidth="4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            {path.map((point, index) => {
              const { cx, cy } = project(point.theta);
              return (
                <circle
                  key={point.step}
                  cx={cx}
                  cy={cy}
                  r={index === path.length - 1 ? 7 : 4}
                  fill={index === path.length - 1 ? '#0f172a' : '#2563eb'}
                  opacity={index === 0 ? 0.7 : 1}
                />
              );
            })}
            <circle cx={project([-3, 1]).cx} cy={project([-3, 1]).cy} r="8" fill="#059669" />
            <text x={project([-3, 1]).cx + 12} y={project([-3, 1]).cy + 4} className="fill-slate-700 text-xs font-bold">minimum</text>
          </svg>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Update rule</h3>
          <div className="mt-4 space-y-3">
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <p className="font-mono text-sm text-slate-950">SGD: theta = theta - eta * g_batch</p>
              <p className="mt-2 text-sm leading-6 text-slate-700">Fast and simple, but mini-batch noise can zigzag through narrow valleys.</p>
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <p className="font-mono text-sm text-slate-950">Momentum: v = beta v + g</p>
              <p className="mt-2 text-sm leading-6 text-slate-700">Velocity damps direction changes and accelerates when gradients agree.</p>
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <p className="font-mono text-sm text-slate-950">Adam: m / sqrt(v)</p>
              <p className="mt-2 text-sm leading-6 text-slate-700">Adam combines momentum with per-parameter step scaling, which helps uneven curvature.</p>
            </div>
          </div>
        </section>
      </div>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-blue-200 bg-blue-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-blue-700">What the learner manipulates</h3>
          <p className="mt-3 text-sm leading-6 text-blue-950">
            Change the update rule, learning rate, momentum, batch size, and number of steps, then compare the path and
            loss trace.
          </p>
        </div>
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-amber-700">Mistake to avoid</h3>
          <p className="mt-3 text-sm leading-6 text-amber-950">
            Adam is not automatically best. A large learning rate can still overshoot, and small batches can make any
            optimizer noisy.
          </p>
        </div>
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-5">
          <h3 className="text-sm font-black uppercase tracking-wide text-emerald-700">Understanding check</h3>
          <p className="mt-3 text-sm leading-6 text-emerald-950">
            Predict whether increasing batch size should make the path smoother before moving the slider.
          </p>
        </div>
      </section>

      <AssessmentPanel lessonId="optimizers" title="Optimizers check" />
    </div>
  );
}
