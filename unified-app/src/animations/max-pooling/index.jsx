import React, { useMemo, useState } from 'react';
import { ArrowDownRight, Grid3X3, Maximize, MousePointer2, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const MATRICES = {
  edges: {
    label: 'Edge-like activations',
    values: [
      [1, 2, 1, 0, 1],
      [2, 9, 3, 1, 0],
      [1, 4, 8, 2, 1],
      [0, 1, 3, 7, 2],
      [1, 0, 2, 3, 6],
    ],
  },
  sparse: {
    label: 'Sparse detections',
    values: [
      [0, 0, 6, 1, 0],
      [1, 0, 2, 0, 0],
      [0, 8, 1, 0, 3],
      [0, 2, 0, 9, 1],
      [4, 0, 1, 0, 0],
    ],
  },
  texture: {
    label: 'Textured feature map',
    values: [
      [3, 4, 2, 5, 3],
      [4, 5, 3, 6, 2],
      [2, 3, 4, 4, 5],
      [5, 2, 6, 3, 4],
      [3, 4, 2, 5, 6],
    ],
  },
};

function poolMatrix(matrix, poolSize, stride) {
  const outSize = Math.floor((matrix.length - poolSize) / stride) + 1;
  return Array.from({ length: outSize }, (_, row) =>
    Array.from({ length: outSize }, (_, col) => {
      const startRow = row * stride;
      const startCol = col * stride;
      const cells = [];
      for (let r = 0; r < poolSize; r += 1) {
        for (let c = 0; c < poolSize; c += 1) {
          cells.push({ row: startRow + r, col: startCol + c, value: matrix[startRow + r][startCol + c] });
        }
      }
      const winner = cells.reduce((best, cell) => (cell.value > best.value ? cell : best), cells[0]);
      const average = cells.reduce((sum, cell) => sum + cell.value, 0) / cells.length;
      return { value: winner.value, winner, cells, average };
    }),
  );
}

function Matrix({ matrix, selected, poolSize, title }) {
  const inWindow = (row, col) => (
    row >= selected.startRow
    && row < selected.startRow + poolSize
    && col >= selected.startCol
    && col < selected.startCol + poolSize
  );

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="mb-3 text-lg font-semibold text-slate-950">{title}</h2>
      <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${matrix.length}, minmax(0, 1fr))` }}>
        {matrix.flatMap((row, rowIndex) => row.map((value, colIndex) => {
          const active = inWindow(rowIndex, colIndex);
          const isWinner = selected.winner.row === rowIndex && selected.winner.col === colIndex;
          return (
            <div
              key={`${rowIndex}-${colIndex}`}
              className={`flex aspect-square items-center justify-center rounded-md border text-lg font-semibold ${
                isWinner
                  ? 'border-orange-500 bg-orange-100 text-orange-950 ring-2 ring-orange-300'
                  : active
                    ? 'border-blue-300 bg-blue-50 text-blue-950'
                    : 'border-slate-200 bg-slate-50 text-slate-700'
              }`}
            >
              {value}
            </div>
          );
        }))}
      </div>
    </div>
  );
}

function OutputGrid({ pooled, selectedCell, setSelectedCell }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="mb-3 text-lg font-semibold text-slate-950">Pooled output</h2>
      <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${pooled.length}, minmax(0, 1fr))` }}>
        {pooled.flatMap((row, rowIndex) => row.map((cell, colIndex) => {
          const active = rowIndex === selectedCell.row && colIndex === selectedCell.col;
          return (
            <button
              key={`${rowIndex}-${colIndex}`}
              type="button"
              onClick={() => setSelectedCell({ row: rowIndex, col: colIndex })}
              className={`flex aspect-square items-center justify-center rounded-md border text-lg font-semibold transition ${
                active ? 'border-blue-600 bg-blue-600 text-white shadow-sm' : 'border-slate-200 bg-slate-50 text-slate-800 hover:border-blue-300'
              }`}
            >
              {cell.value}
            </button>
          );
        }))}
      </div>
      <p className="mt-3 text-sm text-slate-600">Click an output cell to inspect the input window that produced it.</p>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, note }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex items-center gap-2 text-sm font-medium text-slate-600">
        <Icon size={16} />
        {label}
      </div>
      <div className="mt-2 text-2xl font-semibold text-slate-950">{value}</div>
      <p className="mt-1 text-sm text-slate-600">{note}</p>
    </div>
  );
}

export default function MaxPoolingAnimation() {
  const [matrixId, setMatrixId] = useState('edges');
  const [poolSize, setPoolSize] = useState(2);
  const [stride, setStride] = useState(2);
  const [selectedCell, setSelectedCell] = useState({ row: 0, col: 0 });

  const matrix = MATRICES[matrixId].values;
  const pooled = useMemo(() => poolMatrix(matrix, poolSize, stride), [matrix, poolSize, stride]);
  const clampedCell = {
    row: Math.min(selectedCell.row, pooled.length - 1),
    col: Math.min(selectedCell.col, pooled.length - 1),
  };
  const selected = pooled[clampedCell.row][clampedCell.col];
  const selectedWindow = {
    ...selected,
    startRow: clampedCell.row * stride,
    startCol: clampedCell.col * stride,
  };
  const compression = ((1 - (pooled.length * pooled.length) / (matrix.length * matrix.length)) * 100).toFixed(0);
  const maxMinusAverage = selected.value - selected.average;

  return (
    <div className="min-h-full bg-slate-50 text-slate-900">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-6">
        <header className="rounded-lg border border-slate-200 bg-white p-5">
          <p className="text-sm font-semibold uppercase tracking-wide text-blue-700">CNN downsampling</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-normal text-slate-950">Max Pooling</h1>
          <p className="mt-2 max-w-3xl text-slate-600">
            Max pooling downsamples a feature map by keeping the strongest activation inside each local window. It adds
            translation tolerance, but discards the exact locations and non-maximum values inside the window.
          </p>
        </header>

        <section className="grid gap-4 lg:grid-cols-[320px_1fr]">
          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <div className="mb-4 flex items-center gap-2 font-semibold text-slate-800">
              <SlidersHorizontal size={18} />
              Controls
            </div>
            <label htmlFor="matrix" className="text-sm font-medium text-slate-700">Feature map</label>
            <select
              id="matrix"
              value={matrixId}
              onChange={(event) => {
                setMatrixId(event.target.value);
                setSelectedCell({ row: 0, col: 0 });
              }}
              className="mt-1 w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm"
            >
              {Object.entries(MATRICES).map(([id, item]) => (
                <option key={id} value={id}>{item.label}</option>
              ))}
            </select>

            <label htmlFor="pool" className="mt-4 block text-sm font-medium text-slate-700">Pool window: {poolSize}x{poolSize}</label>
            <input
              id="pool"
              type="range"
              min="2"
              max="3"
              step="1"
              value={poolSize}
              onChange={(event) => {
                setPoolSize(Number(event.target.value));
                setSelectedCell({ row: 0, col: 0 });
              }}
              className="mt-2 w-full"
            />

            <label htmlFor="stride" className="mt-4 block text-sm font-medium text-slate-700">Stride: {stride}</label>
            <input
              id="stride"
              type="range"
              min="1"
              max="3"
              step="1"
              value={stride}
              onChange={(event) => {
                setStride(Number(event.target.value));
                setSelectedCell({ row: 0, col: 0 });
              }}
              className="mt-2 w-full"
            />

            <div className="mt-5 rounded-lg bg-blue-50 p-3 text-sm text-blue-950">
              <div className="font-semibold">Output size</div>
              <div className="mt-1 font-mono">floor((5 - {poolSize}) / {stride}) + 1 = {pooled.length}</div>
            </div>
          </div>

          <div className="grid gap-4 xl:grid-cols-[1fr_0.7fr]">
            <Matrix matrix={matrix} selected={selectedWindow} poolSize={poolSize} title="Input feature map" />
            <OutputGrid pooled={pooled} selectedCell={clampedCell} setSelectedCell={setSelectedCell} />
          </div>
        </section>

        <section className="grid gap-4 md:grid-cols-4">
          <StatCard icon={MousePointer2} label="Selected window" value={`r${selectedWindow.startRow}, c${selectedWindow.startCol}`} note="The pooling window is highlighted on the input map." />
          <StatCard icon={Maximize} label="Argmax kept" value={selected.value} note={`Winner came from row ${selected.winner.row}, column ${selected.winner.col}.`} />
          <StatCard icon={Grid3X3} label="Compression" value={`${compression}%`} note="Fewer activations pass to the next layer." />
          <StatCard icon={ArrowDownRight} label="Max minus average" value={maxMinusAverage.toFixed(2)} note="Large gaps show how much non-maximum evidence was discarded." />
        </section>

        <section className="grid gap-4 lg:grid-cols-[1fr_360px]">
          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <h2 className="text-lg font-semibold text-slate-950">Window ledger</h2>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full min-w-[560px] text-left text-sm">
                <thead className="border-b border-slate-200 text-slate-600">
                  <tr>
                    <th className="py-2 pr-3">Cell</th>
                    <th className="py-2 pr-3">Input coordinate</th>
                    <th className="py-2 pr-3">Value</th>
                    <th className="py-2 pr-3">Kept?</th>
                  </tr>
                </thead>
                <tbody>
                  {selected.cells.map((cell, index) => {
                    const kept = cell.row === selected.winner.row && cell.col === selected.winner.col;
                    return (
                      <tr key={`${cell.row}-${cell.col}`} className="border-b border-slate-100">
                        <td className="py-2 pr-3 font-mono">{index + 1}</td>
                        <td className="py-2 pr-3 font-mono">({cell.row}, {cell.col})</td>
                        <td className="py-2 pr-3 font-mono">{cell.value}</td>
                        <td className={`py-2 pr-3 font-medium ${kept ? 'text-orange-700' : 'text-slate-500'}`}>{kept ? 'yes' : 'no'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="space-y-4">
            <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
              <h2 className="font-semibold text-amber-950">Predict before running</h2>
              <p className="mt-2 text-sm text-amber-900">
                Pick one output cell, inspect its highlighted window, and predict the max before clicking another output cell.
                Then increase stride and predict which windows stop being visited.
              </p>
            </div>
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <h2 className="font-semibold text-slate-950">Mistake to avoid</h2>
              <p className="mt-2 text-sm text-slate-600">
                Pooling is not a learned filter. It is a fixed summary operation, so the network cannot recover discarded
                within-window detail later unless another path preserved it.
              </p>
            </div>
          </div>
        </section>

        <AssessmentPanel lessonId="max-pooling" />
      </div>
    </div>
  );
}
