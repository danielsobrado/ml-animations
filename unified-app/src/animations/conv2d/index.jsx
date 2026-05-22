import React, { useMemo, useState } from 'react';
import { Calculator, Grid3X3, MoveRight, ScanLine, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const INPUT = [
  [1, 2, 0, 1, 2],
  [0, 1, 3, 2, 1],
  [2, 3, 1, 0, 2],
  [1, 0, 2, 3, 1],
  [2, 1, 0, 1, 3],
];

const KERNELS = {
  vertical: {
    label: 'Vertical edge',
    values: [
      [1, 0, -1],
      [1, 0, -1],
      [1, 0, -1],
    ],
  },
  horizontal: {
    label: 'Horizontal edge',
    values: [
      [1, 1, 1],
      [0, 0, 0],
      [-1, -1, -1],
    ],
  },
  sharpen: {
    label: 'Sharpen',
    values: [
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0],
    ],
  },
};

function padInput(input, padding) {
  if (padding === 0) return input;
  const width = input[0].length + padding * 2;
  const topBottom = Array.from({ length: padding }, () => Array(width).fill(0));
  const middle = input.map((row) => [
    ...Array(padding).fill(0),
    ...row,
    ...Array(padding).fill(0),
  ]);
  return [...topBottom, ...middle, ...topBottom];
}

function convolve(input, kernel, stride) {
  const outputSize = Math.floor((input.length - 3) / stride) + 1;
  const output = [];
  for (let row = 0; row < outputSize; row += 1) {
    const outputRow = [];
    for (let col = 0; col < outputSize; col += 1) {
      const startRow = row * stride;
      const startCol = col * stride;
      let sum = 0;
      for (let kr = 0; kr < 3; kr += 1) {
        for (let kc = 0; kc < 3; kc += 1) {
          sum += input[startRow + kr][startCol + kc] * kernel[kr][kc];
        }
      }
      outputRow.push(sum);
    }
    output.push(outputRow);
  }
  return output;
}

function clampCell(cell, outputSize) {
  return {
    row: Math.min(cell.row, outputSize - 1),
    col: Math.min(cell.col, outputSize - 1),
  };
}

function Matrix({ matrix, title, windowStart, tone = 'blue' }) {
  const maxAbs = Math.max(...matrix.flat().map((value) => Math.abs(value)), 1);
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <h3 className="text-sm font-bold uppercase tracking-wide text-slate-600">{title}</h3>
      <div className="mt-3 grid gap-1" style={{ gridTemplateColumns: `repeat(${matrix[0].length}, minmax(32px, 1fr))` }}>
        {matrix.map((row, rowIndex) =>
          row.map((value, colIndex) => {
            const inWindow = windowStart
              && rowIndex >= windowStart.row
              && rowIndex < windowStart.row + 3
              && colIndex >= windowStart.col
              && colIndex < windowStart.col + 3;
            const alpha = 0.12 + Math.min(0.5, Math.abs(value) / maxAbs * 0.5);
            const background = value >= 0
              ? `rgba(14, 116, 144, ${alpha})`
              : `rgba(220, 38, 38, ${alpha})`;
            return (
              <div
                key={`${rowIndex}-${colIndex}`}
                className={`flex aspect-square items-center justify-center rounded-md border text-sm font-bold ${
                  inWindow ? 'border-amber-500 ring-2 ring-amber-300' : 'border-slate-200'
                } ${tone === 'kernel' && value === 0 ? 'text-slate-400' : 'text-slate-950'}`}
                style={{ background }}
              >
                {value}
              </div>
            );
          }),
        )}
      </div>
    </div>
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

export default function Conv2dAnimation() {
  const [kernelId, setKernelId] = useState('vertical');
  const [stride, setStride] = useState(1);
  const [padding, setPadding] = useState(0);
  const [selectedCell, setSelectedCell] = useState({ row: 0, col: 0 });

  const kernel = KERNELS[kernelId].values;
  const padded = useMemo(() => padInput(INPUT, padding), [padding]);
  const output = useMemo(() => convolve(padded, kernel, stride), [kernel, padded, stride]);
  const activeCell = clampCell(selectedCell, output.length);
  const windowStart = { row: activeCell.row * stride, col: activeCell.col * stride };
  const outputSizeFormula = `floor((${INPUT.length} + 2*${padding} - 3) / ${stride}) + 1`;
  const products = [];
  let selectedSum = 0;

  for (let kr = 0; kr < 3; kr += 1) {
    for (let kc = 0; kc < 3; kc += 1) {
      const inputValue = padded[windowStart.row + kr][windowStart.col + kc];
      const kernelValue = kernel[kr][kc];
      const product = inputValue * kernelValue;
      selectedSum += product;
      products.push({ inputValue, kernelValue, product });
    }
  }

  return (
    <div className="min-h-full bg-slate-50">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 p-4 md:p-6">
        <header className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-cyan-700">
            <Grid3X3 size={17} />
            CNN local operator
          </div>
          <h1 className="mt-2 text-2xl font-bold text-slate-950 md:text-3xl">Conv2D</h1>
          <p className="mt-2 max-w-3xl text-slate-700">
            A 2D convolution slides one small kernel over local image windows. Each output cell is one dot product
            between the kernel weights and the aligned input patch.
          </p>
        </header>

        <section className="grid gap-4 lg:grid-cols-[320px_1fr]">
          <aside className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex items-center gap-2 font-semibold text-slate-950">
              <SlidersHorizontal size={18} />
              Convolution controls
            </div>
            <div className="mt-5 space-y-4">
              <label className="block">
                <div className="mb-2 text-sm font-semibold text-slate-700">Kernel</div>
                <select
                  value={kernelId}
                  onChange={(event) => setKernelId(event.target.value)}
                  className="w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900"
                >
                  {Object.entries(KERNELS).map(([id, item]) => (
                    <option key={id} value={id}>{item.label}</option>
                  ))}
                </select>
              </label>
              <label className="block">
                <div className="mb-2 flex justify-between text-sm font-semibold text-slate-700">
                  <span>Stride</span>
                  <span>{stride}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="2"
                  step="1"
                  value={stride}
                  onChange={(event) => {
                    setStride(Number(event.target.value));
                    setSelectedCell({ row: 0, col: 0 });
                  }}
                  className="w-full accent-cyan-700"
                />
              </label>
              <label className="block">
                <div className="mb-2 flex justify-between text-sm font-semibold text-slate-700">
                  <span>Zero padding</span>
                  <span>{padding}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="1"
                  value={padding}
                  onChange={(event) => {
                    setPadding(Number(event.target.value));
                    setSelectedCell({ row: 0, col: 0 });
                  }}
                  className="w-full accent-cyan-700"
                />
              </label>
            </div>

            <div className="mt-5 rounded-lg bg-slate-50 p-3">
              <div className="text-sm font-bold text-slate-950">Current kernel</div>
              <div className="mt-3 grid grid-cols-3 gap-1">
                {kernel.flat().map((value, index) => (
                  <div
                    key={index}
                    className={`flex h-10 items-center justify-center rounded-md border border-slate-200 bg-white text-sm font-bold ${
                      value === 0 ? 'text-slate-400' : 'text-slate-950'
                    }`}
                  >
                    {value}
                  </div>
                ))}
              </div>
            </div>
          </aside>

          <main className="space-y-4">
            <div className="grid gap-4 md:grid-cols-4">
              <Metric icon={Calculator} label="Output size" value={`${output.length} x ${output.length}`} helper={outputSizeFormula} />
              <Metric icon={MoveRight} label="Stride" value={stride} helper="How far the kernel jumps each step." />
              <Metric icon={ScanLine} label="Padding" value={padding} helper="Zeros added around the input border." />
              <Metric icon={Grid3X3} label="Selected value" value={selectedSum} helper={`Output [${activeCell.row + 1},${activeCell.col + 1}]`} />
            </div>

            <section className="grid gap-4 xl:grid-cols-[1fr_1fr]">
              <Matrix matrix={padded} title="Padded input" windowStart={windowStart} />
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="text-sm font-bold uppercase tracking-wide text-slate-600">Output map</h3>
                <div className="mt-3 grid gap-2" style={{ gridTemplateColumns: `repeat(${output.length}, minmax(70px, 1fr))` }}>
                  {output.map((row, rowIndex) =>
                    row.map((value, colIndex) => (
                      <button
                        key={`${rowIndex}-${colIndex}`}
                        type="button"
                        onClick={() => setSelectedCell({ row: rowIndex, col: colIndex })}
                        className={`rounded-lg border p-3 text-left ${
                          activeCell.row === rowIndex && activeCell.col === colIndex
                            ? 'border-cyan-500 bg-cyan-50'
                            : 'border-slate-200 bg-slate-50 hover:border-cyan-300'
                        }`}
                      >
                        <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                          [{rowIndex + 1},{colIndex + 1}]
                        </div>
                        <div className="mt-1 text-2xl font-bold text-slate-950">{value}</div>
                      </button>
                    )),
                  )}
                </div>
              </div>
            </section>

            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <h2 className="text-lg font-bold text-slate-950">Selected-cell dot product</h2>
              <p className="text-sm text-slate-600">
                Each term multiplies one highlighted input value by the aligned kernel weight. The nine products are
                summed into exactly one output cell.
              </p>
              <div className="mt-4 grid gap-2 md:grid-cols-3">
                {products.map((item, index) => (
                  <div key={index} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                    <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">term {index + 1}</div>
                    <div className="mt-1 text-lg font-bold text-slate-950">
                      {item.inputValue} x {item.kernelValue} = {item.product}
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <section className="grid gap-4 md:grid-cols-3">
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">Predict before running</h3>
                <p className="mt-2 text-sm text-slate-700">
                  Increase stride and predict whether the output gets larger or smaller before reading the size metric.
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">Causal order</h3>
                <p className="mt-2 text-sm text-slate-700">
                  Padding changes the input grid first, stride changes which windows are visited, and the kernel creates
                  each output value.
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">Mistake to avoid</h3>
                <p className="mt-2 text-sm text-slate-700">
                  This is not elementwise multiplication of two same-sized images. The same kernel is reused at every
                  location.
                </p>
              </div>
            </section>
          </main>
        </section>

        <AssessmentPanel lessonId="conv2d" />
      </div>
    </div>
  );
}
