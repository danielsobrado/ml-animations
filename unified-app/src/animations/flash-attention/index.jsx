import React, { useMemo, useState } from 'react';
import { Cpu, Database, Grid3X3, HardDrive, Layers3, Zap } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const SEQ_OPTIONS = [1024, 2048, 4096, 8192];
const TILE_OPTIONS = [32, 64, 128, 256];
const DTYPE_BYTES = {
  fp32: 4,
  fp16: 2,
  bf16: 2,
};

const formatNumber = (value) => new Intl.NumberFormat('en-US').format(Math.round(value));
const formatBytes = (bytes) => {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024).toFixed(1)} KB`;
};

function ButtonGroup({ label, options, value, onChange, format = (item) => item }) {
  return (
    <div>
      <div className="mb-2 text-sm font-semibold text-slate-700">{label}</div>
      <div className="grid grid-cols-2 gap-2">
        {options.map((option) => (
          <button
            key={option}
            type="button"
            onClick={() => onChange(option)}
            className={`rounded-lg border px-3 py-2 text-sm font-semibold transition ${
              value === option
                ? 'border-amber-700 bg-amber-600 text-white shadow-sm'
                : 'border-slate-200 bg-white text-slate-700 hover:border-amber-400'
            }`}
          >
            {format(option)}
          </button>
        ))}
      </div>
    </div>
  );
}

function Metric({ icon: Icon, label, value, detail }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
        <Icon size={16} />
        {label}
      </div>
      <div className="mt-2 text-2xl font-bold text-slate-950">{value}</div>
      <p className="mt-1 text-sm text-slate-600">{detail}</p>
    </div>
  );
}

export default function FlashAttentionAnimation() {
  const [seqLength, setSeqLength] = useState(4096);
  const [tileSize, setTileSize] = useState(128);
  const [headDim, setHeadDim] = useState(128);
  const [dtype, setDtype] = useState('fp16');
  const [activeTile, setActiveTile] = useState(5);

  const stats = useMemo(() => {
    const bytes = DTYPE_BYTES[dtype];
    const blocks = Math.ceil(seqLength / tileSize);
    const tileCount = blocks * blocks;
    const fullScores = seqLength * seqLength * bytes;
    const tileScores = tileSize * tileSize * bytes;
    const qkvTile = tileSize * headDim * bytes * 3;
    const runningState = tileSize * bytes * 3;
    const workingSet = tileScores + qkvTile + runningState;
    const memoryRatio = workingSet / fullScores;
    const savedPercent = (1 - memoryRatio) * 100;
    const flops = 2 * seqLength * seqLength * headDim;
    const ioProxyStandard = fullScores * 3;
    const ioProxyFlash = seqLength * headDim * bytes * blocks * 2 + seqLength * headDim * bytes;
    const ioSaved = (1 - ioProxyFlash / ioProxyStandard) * 100;

    return {
      blocks,
      tileCount,
      fullScores,
      tileScores,
      workingSet,
      savedPercent,
      flops,
      ioSaved: Math.max(0, ioSaved),
    };
  }, [dtype, headDim, seqLength, tileSize]);

  const currentTile = activeTile % stats.tileCount;
  const activeRow = Math.floor(currentTile / stats.blocks);
  const activeCol = currentTile % stats.blocks;

  const visibleBlocks = Array.from({ length: Math.min(stats.blocks, 8) }, (_, index) => index);
  const scaleLabel = stats.blocks > 8 ? `showing 8 of ${stats.blocks} blocks` : `${stats.blocks} blocks`;

  return (
    <div className="min-h-full bg-slate-50">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 p-4 md:p-6">
        <header className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
            <div>
              <div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-amber-700">
                <Zap size={17} />
                Hardware-aware attention
              </div>
              <h1 className="mt-2 text-2xl font-bold text-slate-950 md:text-3xl">Flash Attention</h1>
              <p className="mt-2 max-w-3xl text-slate-700">
                Flash Attention computes exact attention while streaming score tiles through fast memory. It avoids
                writing the full attention matrix to high-bandwidth memory by keeping online softmax statistics per row.
              </p>
            </div>
            <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
              <div className="font-bold">Exact result, different schedule</div>
              <div>Same attention math, less memory traffic.</div>
            </div>
          </div>
        </header>

        <section className="grid gap-4 lg:grid-cols-[320px_1fr]">
          <aside className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex items-center gap-2 font-semibold text-slate-950">
              <Grid3X3 size={18} />
              Workbench
            </div>
            <div className="mt-5 space-y-5">
              <ButtonGroup
                label="Sequence length"
                options={SEQ_OPTIONS}
                value={seqLength}
                onChange={(next) => {
                  setSeqLength(next);
                  setActiveTile(0);
                }}
                format={(item) => formatNumber(item)}
              />
              <ButtonGroup
                label="Tile size"
                options={TILE_OPTIONS}
                value={tileSize}
                onChange={(next) => {
                  setTileSize(next);
                  setActiveTile(0);
                }}
              />
              <label className="block">
                <div className="mb-2 flex items-center justify-between text-sm font-semibold text-slate-700">
                  <span>Head dimension</span>
                  <span>{headDim}</span>
                </div>
                <input
                  type="range"
                  min="64"
                  max="256"
                  step="64"
                  value={headDim}
                  onChange={(event) => setHeadDim(Number(event.target.value))}
                  className="w-full accent-amber-600"
                />
              </label>
              <ButtonGroup
                label="Element type"
                options={Object.keys(DTYPE_BYTES)}
                value={dtype}
                onChange={setDtype}
                format={(item) => item.toUpperCase()}
              />
            </div>
          </aside>

          <main className="space-y-4">
            <div className="grid gap-4 md:grid-cols-4">
              <Metric
                icon={HardDrive}
                label="Full score matrix"
                value={formatBytes(stats.fullScores)}
                detail="Materializing QK^T scales with sequence length squared."
              />
              <Metric
                icon={Cpu}
                label="Tile working set"
                value={formatBytes(stats.workingSet)}
                detail="Approximate fast-memory footprint for one tile and row state."
              />
              <Metric
                icon={Database}
                label="Memory avoided"
                value={`${stats.savedPercent.toFixed(1)}%`}
                detail="Relative to storing the full attention score matrix."
              />
              <Metric
                icon={Layers3}
                label="Tiles streamed"
                value={formatNumber(stats.tileCount)}
                detail={`${stats.blocks} by ${stats.blocks} block schedule.`}
              />
            </div>

            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div>
                  <h2 className="text-lg font-bold text-slate-950">Tiled attention schedule</h2>
                  <p className="text-sm text-slate-600">
                    Move across key/value tiles for each query block, updating the row max, denominator, and output
                    accumulator without storing every score.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setActiveTile((tile) => (tile + 1) % stats.tileCount)}
                  className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800"
                >
                  Step tile
                </button>
              </div>

              <div className="mt-4 overflow-x-auto">
                <div className="inline-grid gap-1" style={{ gridTemplateColumns: `repeat(${visibleBlocks.length}, minmax(38px, 1fr))` }}>
                  {visibleBlocks.flatMap((row) =>
                    visibleBlocks.map((col) => {
                      const active = row === activeRow && col === activeCol;
                      const processed = row * stats.blocks + col < currentTile;
                      return (
                        <div
                          key={`${row}-${col}`}
                          className={`flex h-10 min-w-10 items-center justify-center rounded-md border text-xs font-bold ${
                            active
                              ? 'border-amber-700 bg-amber-500 text-white'
                              : processed
                                ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
                                : 'border-slate-200 bg-slate-50 text-slate-500'
                          }`}
                        >
                          {row},{col}
                        </div>
                      );
                    }),
                  )}
                </div>
              </div>
              <div className="mt-3 text-sm text-slate-600">
                Active tile: query block {activeRow + 1}, key/value block {activeCol + 1}; {scaleLabel}.
              </div>
            </section>

            <section className="grid gap-4 md:grid-cols-3">
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">Online softmax state</h3>
                <p className="mt-2 text-sm text-slate-700">
                  Each query row keeps a running max, normalization denominator, and output accumulator. New tiles
                  rescale that state before adding their contribution.
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">What does not change</h3>
                <p className="mt-2 text-sm text-slate-700">
                  The algorithm still computes exact scaled dot-product attention. The savings come from memory
                  scheduling, not from approximating attention probabilities.
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">Bottleneck it targets</h3>
                <p className="mt-2 text-sm text-slate-700">
                  Long-context attention is often limited by high-bandwidth-memory reads and writes. Streaming tiles
                  reduces traffic by about {stats.ioSaved.toFixed(0)}% in this simplified proxy.
                </p>
              </div>
            </section>
          </main>
        </section>

        <AssessmentPanel lessonId="flash-attention" />
      </div>
    </div>
  );
}
