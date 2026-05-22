import React, { useMemo, useState } from 'react';
import { Compass, Grid3X3, RotateCcw, Sigma, Target } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const DIMENSIONS = [32, 64, 128];
const BASES = [1000, 10000, 100000];

const toDegrees = (radians) => ((radians * 180) / Math.PI) % 360;
const formatAngle = (radians) => `${toDegrees(radians).toFixed(1)} deg`;

function rotatePair([x, y], angle) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return [x * cos - y * sin, x * sin + y * cos];
}

function dot(left, right) {
  return left[0] * right[0] + left[1] * right[1];
}

function ControlButton({ active, children, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-lg border px-3 py-2 text-sm font-semibold transition ${
        active
          ? 'border-violet-800 bg-violet-700 text-white shadow-sm'
          : 'border-slate-200 bg-white text-slate-700 hover:border-violet-400'
      }`}
    >
      {children}
    </button>
  );
}

function Metric({ icon: Icon, label, value, helper }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
        <Icon size={16} />
        {label}
      </div>
      <div className="mt-2 text-2xl font-bold text-slate-950">{value}</div>
      <p className="mt-1 text-sm text-slate-600">{helper}</p>
    </div>
  );
}

function VectorPlot({ vector, color, label }) {
  const size = 120;
  const center = size / 2;
  const endX = center + vector[0] * 42;
  const endY = center - vector[1] * 42;

  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
      <svg viewBox={`0 0 ${size} ${size}`} className="h-28 w-full">
        <circle cx={center} cy={center} r="42" fill="white" stroke="#cbd5e1" />
        <line x1="18" y1={center} x2="102" y2={center} stroke="#e2e8f0" />
        <line x1={center} y1="18" x2={center} y2="102" stroke="#e2e8f0" />
        <line x1={center} y1={center} x2={endX} y2={endY} stroke={color} strokeWidth="4" strokeLinecap="round" />
        <circle cx={endX} cy={endY} r="5" fill={color} />
      </svg>
      <div className="text-center text-sm font-semibold text-slate-700">{label}</div>
    </div>
  );
}

export default function RoPEAnimation() {
  const [queryPos, setQueryPos] = useState(8);
  const [keyPos, setKeyPos] = useState(3);
  const [dimension, setDimension] = useState(64);
  const [base, setBase] = useState(10000);
  const [pairIndex, setPairIndex] = useState(0);

  const baseQuery = [0.92, 0.38];
  const baseKey = [0.86, -0.22];

  const stats = useMemo(() => {
    const theta = Math.pow(base, (-2 * pairIndex) / dimension);
    const queryAngle = queryPos * theta;
    const keyAngle = keyPos * theta;
    const relativeAngle = (queryPos - keyPos) * theta;
    const rotatedQuery = rotatePair(baseQuery, queryAngle);
    const rotatedKey = rotatePair(baseKey, keyAngle);
    const directScore = dot(rotatedQuery, rotatedKey);
    const relativeKey = rotatePair(baseKey, -relativeAngle);
    const relativeScore = dot(baseQuery, relativeKey);
    const unrotatedScore = dot(baseQuery, baseKey);

    const rows = Array.from({ length: Math.min(8, dimension / 2) }, (_, index) => {
      const rowTheta = Math.pow(base, (-2 * index) / dimension);
      return {
        pair: index,
        theta: rowTheta,
        queryAngle: queryPos * rowTheta,
        keyAngle: keyPos * rowTheta,
        relativeAngle: (queryPos - keyPos) * rowTheta,
      };
    });

    return {
      theta,
      queryAngle,
      keyAngle,
      relativeAngle,
      rotatedQuery,
      rotatedKey,
      directScore,
      relativeScore,
      unrotatedScore,
      rows,
    };
  }, [base, dimension, keyPos, pairIndex, queryPos]);

  const relativeDistance = queryPos - keyPos;
  const scoreShift = stats.directScore - stats.unrotatedScore;

  return (
    <div className="min-h-full bg-slate-50">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 p-4 md:p-6">
        <header className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
            <div>
              <div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-violet-700">
                <RotateCcw size={17} />
                Rotary position embeddings
              </div>
              <h1 className="mt-2 text-2xl font-bold text-slate-950 md:text-3xl">RoPE</h1>
              <p className="mt-2 max-w-3xl text-slate-700">
                RoPE injects position by rotating query and key dimension pairs. When the rotated vectors take a dot
                product, the score depends on relative distance, not just absolute token identities.
              </p>
            </div>
            <div className="rounded-lg border border-violet-200 bg-violet-50 px-4 py-3 text-sm text-violet-950">
              <div className="font-bold">Core effect</div>
              <div>Q at position m meets K at position n through m - n.</div>
            </div>
          </div>
        </header>

        <section className="grid gap-4 lg:grid-cols-[320px_1fr]">
          <aside className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex items-center gap-2 font-semibold text-slate-950">
              <Grid3X3 size={18} />
              Controls
            </div>

            <div className="mt-5 space-y-5">
              <label className="block">
                <div className="mb-2 flex items-center justify-between text-sm font-semibold text-slate-700">
                  <span>Query position m</span>
                  <span>{queryPos}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="32"
                  step="1"
                  value={queryPos}
                  onChange={(event) => setQueryPos(Number(event.target.value))}
                  className="w-full accent-violet-700"
                />
              </label>

              <label className="block">
                <div className="mb-2 flex items-center justify-between text-sm font-semibold text-slate-700">
                  <span>Key position n</span>
                  <span>{keyPos}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="32"
                  step="1"
                  value={keyPos}
                  onChange={(event) => setKeyPos(Number(event.target.value))}
                  className="w-full accent-fuchsia-700"
                />
              </label>

              <div>
                <div className="mb-2 text-sm font-semibold text-slate-700">Model dimension</div>
                <div className="grid grid-cols-3 gap-2">
                  {DIMENSIONS.map((dim) => (
                    <ControlButton key={dim} active={dimension === dim} onClick={() => setDimension(dim)}>
                      {dim}
                    </ControlButton>
                  ))}
                </div>
              </div>

              <div>
                <div className="mb-2 text-sm font-semibold text-slate-700">RoPE base</div>
                <div className="grid grid-cols-1 gap-2">
                  {BASES.map((nextBase) => (
                    <ControlButton key={nextBase} active={base === nextBase} onClick={() => setBase(nextBase)}>
                      {nextBase}
                    </ControlButton>
                  ))}
                </div>
              </div>

              <label className="block">
                <div className="mb-2 flex items-center justify-between text-sm font-semibold text-slate-700">
                  <span>Dimension pair</span>
                  <span>{pairIndex}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max={Math.min(7, dimension / 2 - 1)}
                  step="1"
                  value={pairIndex}
                  onChange={(event) => setPairIndex(Number(event.target.value))}
                  className="w-full accent-slate-900"
                />
              </label>
            </div>
          </aside>

          <main className="space-y-4">
            <div className="grid gap-4 md:grid-cols-4">
              <Metric icon={Compass} label="Relative distance" value={relativeDistance} helper="The attention score can depend on m - n." />
              <Metric icon={RotateCcw} label="Pair frequency theta" value={stats.theta.toExponential(2)} helper="Higher pairs rotate more slowly." />
              <Metric icon={Target} label="Rotated score" value={stats.directScore.toFixed(3)} helper={`Changed by ${scoreShift.toFixed(3)} from no rotation.`} />
              <Metric icon={Sigma} label="Relative check" value={Math.abs(stats.directScore - stats.relativeScore).toExponential(1)} helper="Direct rotation and relative form should match." />
            </div>

            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div>
                  <h2 className="text-lg font-bold text-slate-950">Rotation in one dimension pair</h2>
                  <p className="text-sm text-slate-600">
                    RoPE rotates 2D slices of Q and K. The selected pair uses angle = position * theta.
                  </p>
                </div>
                <div className="rounded-md bg-slate-100 px-3 py-2 text-sm font-semibold text-slate-700">
                  q angle {formatAngle(stats.queryAngle)} | k angle {formatAngle(stats.keyAngle)}
                </div>
              </div>

              <div className="mt-4 grid gap-4 md:grid-cols-3">
                <VectorPlot vector={baseQuery} color="#64748b" label="Base query pair" />
                <VectorPlot vector={stats.rotatedQuery} color="#7c3aed" label={`Rotated Q at m=${queryPos}`} />
                <VectorPlot vector={stats.rotatedKey} color="#c026d3" label={`Rotated K at n=${keyPos}`} />
              </div>
            </section>

            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <h2 className="text-lg font-bold text-slate-950">Multi-frequency schedule</h2>
              <p className="text-sm text-slate-600">
                Early pairs rotate quickly; later pairs rotate slowly. Together they encode short and long distance
                patterns.
              </p>
              <div className="mt-4 overflow-x-auto">
                <table className="w-full min-w-[650px] text-left text-sm">
                  <thead className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
                    <tr>
                      <th className="py-2">Pair</th>
                      <th className="py-2">Theta</th>
                      <th className="py-2">Q angle</th>
                      <th className="py-2">K angle</th>
                      <th className="py-2">Relative angle</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stats.rows.map((row) => (
                      <tr
                        key={row.pair}
                        className={`border-b border-slate-100 ${row.pair === pairIndex ? 'bg-violet-50' : ''}`}
                      >
                        <td className="py-2 font-semibold text-slate-900">{row.pair}</td>
                        <td className="py-2 font-mono text-slate-700">{row.theta.toExponential(2)}</td>
                        <td className="py-2">{formatAngle(row.queryAngle)}</td>
                        <td className="py-2">{formatAngle(row.keyAngle)}</td>
                        <td className="py-2 font-semibold text-violet-700">{formatAngle(row.relativeAngle)}</td>
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
                  Move Q and K together by the same amount. The absolute angles change, but the relative distance stays
                  fixed, so the relative part of the score is preserved.
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">What RoPE changes</h3>
                <p className="mt-2 text-sm text-slate-700">
                  RoPE is applied to query and key vectors before attention scoring. It does not rotate value vectors or
                  replace the causal mask.
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">Failure mode</h3>
                <p className="mt-2 text-sm text-slate-700">
                  Extrapolating to much longer contexts can distort high-frequency rotations, so practical systems tune
                  base values or apply scaling methods.
                </p>
              </div>
            </section>
          </main>
        </section>

        <AssessmentPanel lessonId="rope" />
      </div>
    </div>
  );
}
