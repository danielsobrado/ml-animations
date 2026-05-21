import React, { useMemo, useState } from 'react';
import { RotateCcw, SlidersHorizontal, TrendingDown } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const OPTIMIZERS = {
  sgd: { label: 'SGD', detail: 'Uses the current mini-batch gradient directly.' },
  momentum: { label: 'Momentum', detail: 'Builds velocity so repeated gradient directions compound.' },
  adam: { label: 'Adam', detail: 'Normalizes momentum by a running estimate of squared gradients.' },
};

function loss([x, y]) {
  return 0.08 * (x + 3) ** 2 + 0.55 * (y - 1) ** 2;
}

function trueGradient([x, y]) {
  return [0.16 * (x + 3), 1.1 * (y - 1)];
}

function deterministicNoise(step, batchSize) {
  const scale = 0.42 / Math.sqrt(batchSize);
  return [
    Math.sin(step * 1.7 + batchSize * 0.11) * scale,
    Math.cos(step * 2.3 + batchSize * 0.07) * scale,
  ];
}

function simulate({ optimizer, learningRate, momentum, batchSize, steps }) {
  let theta = [-4.8, 3.6];
  let velocity = [0, 0];
  let firstMoment = [0, 0];
  let secondMoment = [0, 0];
  const beta2 = 0.96;
  const path = [{ step: 0, theta, loss: loss(theta), grad: [0, 0] }];

  for (let step = 1; step <= steps; step += 1) {
    const exactGradient = trueGradient(theta);
    const noise = deterministicNoise(step, batchSize);
    const gradient = [exactGradient[0] + noise[0], exactGradient[1] + noise[1]];

    if (optimizer === 'momentum') {
      velocity = [
        momentum * velocity[0] + gradient[0],
        momentum * velocity[1] + gradient[1],
      ];
      theta = [
        theta[0] - learningRate * velocity[0],
        theta[1] - learningRate * velocity[1],
      ];
    } else if (optimizer === 'adam') {
      firstMoment = [
        momentum * firstMoment[0] + (1 - momentum) * gradient[0],
        momentum * firstMoment[1] + (1 - momentum) * gradient[1],
      ];
      secondMoment = [
        beta2 * secondMoment[0] + (1 - beta2) * gradient[0] ** 2,
        beta2 * secondMoment[1] + (1 - beta2) * gradient[1] ** 2,
      ];
      const correctedFirst = [
        firstMoment[0] / (1 - momentum ** step),
        firstMoment[1] / (1 - momentum ** step),
      ];
      const correctedSecond = [
        secondMoment[0] / (1 - beta2 ** step),
        secondMoment[1] / (1 - beta2 ** step),
      ];
      theta = [
        theta[0] - learningRate * correctedFirst[0] / (Math.sqrt(correctedSecond[0]) + 1e-6),
        theta[1] - learningRate * correctedFirst[1] / (Math.sqrt(correctedSecond[1]) + 1e-6),
      ];
    } else {
      theta = [
        theta[0] - learningRate * gradient[0],
        theta[1] - learningRate * gradient[1],
      ];
    }

    path.push({ step, theta, loss: loss(theta), grad: gradient });
  }

  return path;
}

function project([x, y]) {
  return {
    cx: 60 + ((x + 5.5) / 5.5) * 420,
    cy: 320 - ((y + 0.5) / 4.5) * 260,
  };
}

function lossColor(value) {
  if (value < 0.2) return '#ecfdf5';
  if (value < 0.6) return '#dbeafe';
  if (value < 1.4) return '#fef3c7';
  return '#fee2e2';
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

function Stat({ label, value, detail }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <p className="text-xs font-black uppercase tracking-wide text-slate-500">{label}</p>
      <strong className="mt-1 block text-2xl font-black text-slate-950">{value}</strong>
      <span className="text-sm text-slate-600">{detail}</span>
    </div>
  );
}

export default function OptimizersAnimation() {
  const [optimizer, setOptimizer] = useState('adam');
  const [learningRate, setLearningRate] = useState(0.18);
  const [momentum, setMomentum] = useState(0.85);
  const [batchSize, setBatchSize] = useState(8);
  const [steps, setSteps] = useState(18);

  const path = useMemo(
    () => simulate({ optimizer, learningRate, momentum, batchSize, steps }),
    [optimizer, learningRate, momentum, batchSize, steps],
  );
  const finalPoint = path[path.length - 1];
  const bestPoint = path.reduce((best, point) => (point.loss < best.loss ? point : best), path[0]);
  const startLoss = path[0].loss;
  const improvement = Math.max(0, 1 - finalPoint.loss / startLoss);

  const reset = () => {
    setOptimizer('adam');
    setLearningRate(0.18);
    setMomentum(0.85);
    setBatchSize(8);
    setSteps(18);
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
