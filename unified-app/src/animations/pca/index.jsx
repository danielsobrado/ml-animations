import React, { useMemo, useState } from 'react';
import { ArrowRight, RotateCcw, ScatterChart, SlidersHorizontal } from 'lucide-react';

const RAW_POINTS = [
  [-2.4, -1.8], [-2.0, -1.1], [-1.7, -1.4], [-1.4, -0.6], [-1.0, -0.9],
  [-0.7, -0.2], [-0.3, -0.4], [0.0, 0.1], [0.4, 0.3], [0.8, 0.5],
  [1.1, 1.0], [1.4, 0.7], [1.8, 1.5], [2.1, 1.2], [2.5, 2.0],
];

function rotate([x, y], angle) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return [x * cos - y * sin, x * sin + y * cos];
}

function makePoints(correlation, noise) {
  const angle = correlation * 0.65;
  const spreadY = 0.35 + (1 - Math.abs(correlation)) * 0.75 + noise * 0.35;

  return RAW_POINTS.map(([x, y], index) => {
    const jitter = Math.sin(index * 2.3) * noise;
    return rotate([x, y * spreadY + jitter], angle);
  });
}

function covariance(points) {
  const mean = points.reduce((sum, point) => [sum[0] + point[0], sum[1] + point[1]], [0, 0]).map((value) => value / points.length);
  const centered = points.map(([x, y]) => [x - mean[0], y - mean[1]]);
  const [xx, xy, yy] = centered.reduce(
    (acc, [x, y]) => [acc[0] + x * x, acc[1] + x * y, acc[2] + y * y],
    [0, 0, 0],
  ).map((value) => value / (points.length - 1));
  const trace = xx + yy;
  const determinant = xx * yy - xy * xy;
  const root = Math.sqrt(Math.max(0, trace * trace - 4 * determinant));
  const lambda1 = (trace + root) / 2;
  const lambda2 = (trace - root) / 2;
  const angle = 0.5 * Math.atan2(2 * xy, xx - yy);

  return { mean, centered, lambda1, lambda2, angle };
}

function toScreen([x, y], size = 360) {
  const scale = 58;
  return [size / 2 + x * scale, size / 2 - y * scale];
}

function project(point, mean, angle) {
  const unit = [Math.cos(angle), Math.sin(angle)];
  const centered = [point[0] - mean[0], point[1] - mean[1]];
  const score = centered[0] * unit[0] + centered[1] * unit[1];
  return [mean[0] + score * unit[0], mean[1] + score * unit[1]];
}

function Axis({ mean, angle, length, className }) {
  const unit = [Math.cos(angle), Math.sin(angle)];
  const start = toScreen([mean[0] - unit[0] * length, mean[1] - unit[1] * length]);
  const end = toScreen([mean[0] + unit[0] * length, mean[1] + unit[1] * length]);

  return <line x1={start[0]} y1={start[1]} x2={end[0]} y2={end[1]} className={className} strokeWidth="4" strokeLinecap="round" />;
}

function Stat({ label, value, detail }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <p className="text-xs font-bold uppercase tracking-wide text-slate-500">{label}</p>
      <strong className="mt-1 block text-2xl font-black text-slate-900">{value}</strong>
      <span className="text-sm text-slate-600">{detail}</span>
    </div>
  );
}

export default function PCAAnimation() {
  const [correlation, setCorrelation] = useState(0.65);
  const [noise, setNoise] = useState(0.25);
  const [components, setComponents] = useState(1);
  const points = useMemo(() => makePoints(correlation, noise), [correlation, noise]);
  const pca = useMemo(() => covariance(points), [points]);
  const explained = pca.lambda1 / (pca.lambda1 + pca.lambda2);
  const reconstructed = points.map((point) => (
    components === 1 ? project(point, pca.mean, pca.angle) : point
  ));

  const reset = () => {
    setCorrelation(0.65);
    setNoise(0.25);
    setComponents(1);
  };

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-bold uppercase tracking-wide text-slate-500">Variance-preserving projection</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Principal Component Analysis</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
              PCA centers the data, finds the directions where points vary most, then projects onto the strongest
              components so lower-dimensional coordinates keep as much signal as possible.
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

      <div className="grid gap-3 md:grid-cols-3">
        <Stat label="PC1 explained" value={`${Math.round(explained * 100)}%`} detail="variance retained by one axis" />
        <Stat label="PC2 explained" value={`${Math.round((1 - explained) * 100)}%`} detail="remaining variance" />
        <Stat label="Projection" value={`${components}D`} detail={components === 1 ? 'compressed view' : 'full two-axis view'} />
      </div>

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
          <SlidersHorizontal size={16} />
          Controls
        </div>
        <div className="grid gap-4 lg:grid-cols-3">
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Correlation: {correlation.toFixed(2)}
            <input
              type="range"
              min="-0.9"
              max="0.9"
              step="0.05"
              value={correlation}
              onChange={(event) => setCorrelation(Number(event.target.value))}
            />
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Noise: {noise.toFixed(2)}
            <input
              type="range"
              min="0"
              max="0.9"
              step="0.05"
              value={noise}
              onChange={(event) => setNoise(Number(event.target.value))}
            />
          </label>
          <div className="grid grid-cols-2 gap-2 self-end">
            {[1, 2].map((count) => (
              <button
                key={count}
                type="button"
                onClick={() => setComponents(count)}
                className={`rounded-lg border px-3 py-2 text-sm font-bold ${
                  components === count ? 'border-blue-600 bg-blue-600 text-white' : 'border-slate-300 bg-white text-slate-700'
                }`}
              >
                {count} component{count === 1 ? '' : 's'}
              </button>
            ))}
          </div>
        </div>
      </section>

      <div className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
        <section className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-4 flex items-center gap-2 text-sm font-black uppercase tracking-wide text-slate-600">
            <ScatterChart size={16} />
            Data, components, and reconstruction
          </div>
          <svg viewBox="0 0 360 360" className="h-auto w-full rounded-lg border border-slate-200 bg-slate-50">
            <line x1="180" y1="20" x2="180" y2="340" stroke="#e2e8f0" strokeWidth="1" />
            <line x1="20" y1="180" x2="340" y2="180" stroke="#e2e8f0" strokeWidth="1" />
            <Axis mean={pca.mean} angle={pca.angle + Math.PI / 2} length={2.4} className="stroke-slate-400" />
            <Axis mean={pca.mean} angle={pca.angle} length={3.1} className="stroke-blue-600" />
            {points.map((point, index) => {
              const source = toScreen(point);
              const target = toScreen(reconstructed[index]);
              return (
                <g key={index}>
                  {components === 1 && (
                    <line x1={source[0]} y1={source[1]} x2={target[0]} y2={target[1]} stroke="#cbd5e1" strokeDasharray="4 4" />
                  )}
                  <circle cx={target[0]} cy={target[1]} r="4" fill={components === 1 ? '#1d4ed8' : '#0f172a'} opacity="0.9" />
                  {components === 1 && <circle cx={source[0]} cy={source[1]} r="3" fill="#94a3b8" opacity="0.6" />}
                </g>
              );
            })}
          </svg>
        </section>

        <aside className="grid gap-4">
          <section className="rounded-lg border border-slate-200 bg-white p-5">
            <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">What PCA does</h3>
            <ol className="mt-4 space-y-3 text-sm leading-6 text-slate-700">
              <li className="flex gap-3"><strong>1</strong><span>Center the data so directions describe variation around the mean.</span></li>
              <li className="flex gap-3"><strong>2</strong><span>Find covariance eigenvectors, ordered by eigenvalue size.</span></li>
              <li className="flex gap-3"><strong>3</strong><span>Keep the first k directions and project every point onto them.</span></li>
            </ol>
          </section>
          <section className="rounded-lg border border-blue-200 bg-blue-50 p-5">
            <h3 className="text-sm font-black uppercase tracking-wide text-blue-700">Read the plot</h3>
            <p className="mt-3 text-sm leading-6 text-blue-950">
              The blue line is PC1. Gray points are original data; blue points are the 1D reconstruction when one
              component is kept. More noise spreads points away from PC1 and lowers explained variance.
            </p>
          </section>
          <section className="rounded-lg border border-slate-200 bg-white p-5">
            <h3 className="text-sm font-black uppercase tracking-wide text-slate-600">Formula</h3>
            <p className="mt-3 flex flex-wrap items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 p-3 font-mono text-sm text-slate-900">
              <span>X_centered</span>
              <ArrowRight className="inline text-slate-500" size={14} />
              <span>covariance</span>
              <ArrowRight className="inline text-slate-500" size={14} />
              <span>eigenvectors</span>
            </p>
          </section>
        </aside>
      </div>
    </div>
  );
}
