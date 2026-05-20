import React, { useMemo, useState } from 'react';
import { CheckCircle2, Layers, Search, Target } from 'lucide-react';

const DECOMPOSITIONS = [
  {
    id: 'lu',
    name: 'LU',
    formula: 'A = L U',
    factors: 'lower triangular L, upper triangular U',
    requirement: 'square matrix; pivoting often gives P A = L U',
    preserves: 'fast solves by forward and back substitution',
    use: 'linear systems, determinants, repeated solves',
    warning: 'unstable without pivoting when pivots are small',
    task: 'solve',
    color: 'border-sky-300 bg-sky-50 text-sky-900',
  },
  {
    id: 'qr',
    name: 'QR',
    formula: 'A = Q R',
    factors: 'orthonormal Q, upper triangular R',
    requirement: 'works for rectangular matrices; full column rank helps least squares',
    preserves: 'lengths and angles through Q',
    use: 'least squares, orthogonal bases, numerical stability',
    warning: 'more expensive than normal equations but usually safer',
    task: 'basis',
    color: 'border-emerald-300 bg-emerald-50 text-emerald-900',
  },
  {
    id: 'eigen',
    name: 'Eigen',
    formula: 'A = V Lambda V^-1',
    factors: 'eigenvectors V, eigenvalues Lambda',
    requirement: 'square and diagonalizable; symmetric gives A = Q Lambda Q^T',
    preserves: 'directions stretched by the matrix',
    use: 'dynamics, spectral graph methods, covariance intuition',
    warning: 'not every square matrix has a stable eigenbasis',
    task: 'spectrum',
    color: 'border-violet-300 bg-violet-50 text-violet-900',
  },
  {
    id: 'svd',
    name: 'SVD',
    formula: 'A = U Sigma V^T',
    factors: 'left singular vectors U, singular values Sigma, right singular vectors V',
    requirement: 'works for any real m x n matrix',
    preserves: 'best orthogonal input and output axes',
    use: 'compression, rank, pseudoinverse, PCA foundation',
    warning: 'general and robust, but usually more costly',
    task: 'compress',
    color: 'border-orange-300 bg-orange-50 text-orange-900',
  },
  {
    id: 'cholesky',
    name: 'Cholesky',
    formula: 'A = L L^T',
    factors: 'lower triangular L and its transpose',
    requirement: 'symmetric positive definite matrix',
    preserves: 'positive curvature structure',
    use: 'Gaussian models, covariance matrices, fast SPD solves',
    warning: 'fails if A is not positive definite',
    task: 'solve',
    color: 'border-amber-300 bg-amber-50 text-amber-900',
  },
  {
    id: 'pca',
    name: 'PCA / Truncated SVD',
    formula: 'A approx U_k Sigma_k V_k^T',
    factors: 'top k singular directions',
    requirement: 'choose k by variance, error, or budget',
    preserves: 'dominant variance and low-rank structure',
    use: 'dimensionality reduction, denoising, visualization',
    warning: 'linear summary can hide small but important signals',
    task: 'compress',
    color: 'border-pink-300 bg-pink-50 text-pink-900',
  },
  {
    id: 'nmf',
    name: 'NMF',
    formula: 'A approx W H',
    factors: 'nonnegative parts W and activations H',
    requirement: 'A should be nonnegative; rank k is chosen',
    preserves: 'additive parts-based structure',
    use: 'topic models, parts of images, interpretable factors',
    warning: 'non-convex objective, so initialization matters',
    task: 'interpret',
    color: 'border-teal-300 bg-teal-50 text-teal-900',
  },
];

const TASKS = [
  { id: 'all', label: 'All' },
  { id: 'solve', label: 'Solves' },
  { id: 'basis', label: 'Bases' },
  { id: 'spectrum', label: 'Spectrum' },
  { id: 'compress', label: 'Compression' },
  { id: 'interpret', label: 'Interpretability' },
];

const FLOW = [
  { ask: 'Need to solve Ax = b repeatedly?', pick: 'LU, or Cholesky when A is SPD' },
  { ask: 'Need stable least squares?', pick: 'QR' },
  { ask: 'Need directions that A only stretches?', pick: 'Eigen decomposition' },
  { ask: 'Need the most general low-rank view?', pick: 'SVD' },
  { ask: 'Need additive nonnegative parts?', pick: 'NMF' },
];

function DecompositionCard({ item, selected, onSelect }) {
  return (
    <button
      onClick={() => onSelect(item)}
      className={`rounded-lg border p-4 text-left shadow-sm transition hover:-translate-y-0.5 hover:shadow-md ${
        selected ? `${item.color} ring-2 ring-offset-2 ring-offset-white` : 'border-slate-200 bg-white'
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-bold text-slate-950">{item.name}</h3>
          <p className="mt-1 font-mono text-sm text-slate-700">{item.formula}</p>
        </div>
        {selected && <CheckCircle2 className="shrink-0 text-cyan-600" size={20} />}
      </div>
      <p className="mt-3 text-sm text-slate-600">{item.use}</p>
    </button>
  );
}

export default function OneSheetPanel() {
  const [task, setTask] = useState('all');
  const [selectedId, setSelectedId] = useState('svd');

  const filtered = useMemo(() => (
    task === 'all' ? DECOMPOSITIONS : DECOMPOSITIONS.filter((item) => item.task === task)
  ), [task]);

  const selected = DECOMPOSITIONS.find((item) => item.id === selectedId) ?? DECOMPOSITIONS[0];

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-5 p-4 text-slate-900">
      <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-cyan-600">Linear algebra</p>
            <h2 className="mt-1 text-2xl font-bold">Matrix Decompositions One-Sheet</h2>
            <p className="mt-2 max-w-3xl text-sm text-slate-600">
              A compact guide to what each factorization is for, when it works, and the fastest way to choose one.
            </p>
          </div>

          <label className="flex items-center gap-2 rounded-lg bg-slate-100 px-3 py-2 text-sm font-semibold">
            <Search size={16} />
            <select
              value={task}
              onChange={(event) => setTask(event.target.value)}
              className="bg-transparent text-slate-900 outline-none"
            >
              {TASKS.map((item) => (
                <option key={item.id} value={item.id}>{item.label}</option>
              ))}
            </select>
          </label>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-[1.2fr_0.8fr]">
        <div className="grid gap-3 md:grid-cols-2">
          {filtered.map((item) => (
            <DecompositionCard
              key={item.id}
              item={item}
              selected={selected.id === item.id}
              onSelect={(next) => setSelectedId(next.id)}
            />
          ))}
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-cyan-100 text-cyan-700">
              <Layers size={20} />
            </div>
            <div>
              <h3 className="text-xl font-bold">{selected.name}</h3>
              <p className="font-mono text-sm text-slate-600">{selected.formula}</p>
            </div>
          </div>

          <dl className="space-y-4 text-sm">
            <div>
              <dt className="font-bold text-slate-950">Factors</dt>
              <dd className="mt-1 text-slate-600">{selected.factors}</dd>
            </div>
            <div>
              <dt className="font-bold text-slate-950">Requirement</dt>
              <dd className="mt-1 text-slate-600">{selected.requirement}</dd>
            </div>
            <div>
              <dt className="font-bold text-slate-950">Preserves</dt>
              <dd className="mt-1 text-slate-600">{selected.preserves}</dd>
            </div>
            <div>
              <dt className="font-bold text-slate-950">Watch out</dt>
              <dd className="mt-1 text-slate-600">{selected.warning}</dd>
            </div>
          </dl>
        </div>
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-rose-100 text-rose-700">
            <Target size={20} />
          </div>
          <div>
            <h3 className="text-lg font-bold">Choose by goal</h3>
            <p className="text-sm text-slate-600">Start from the job, then pick the factorization.</p>
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-5">
          {FLOW.map((item) => (
            <div key={item.ask} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <p className="text-sm font-bold text-slate-950">{item.ask}</p>
              <p className="mt-2 text-xs text-slate-600">{item.pick}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
