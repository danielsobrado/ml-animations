import React, { useMemo, useState } from 'react';
import { AlertCircle, ArrowRight, Braces, Grid3X3, Waves } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const SENTENCES = [
  ['the', 'model', 'reads', 'tokens', 'left', 'to', 'right'],
  ['dog', 'bites', 'man'],
  ['man', 'bites', 'dog'],
];

const ENCODING_TYPES = [
  { id: 'sinusoidal', label: 'Sinusoidal' },
  { id: 'learned', label: 'Learned absolute' },
  { id: 'none', label: 'No position signal' },
];

const DIMENSIONS = [16, 32, 64, 128];

function positionalValue(type, position, dimIndex, dimension) {
  if (type === 'none') return 0;
  if (type === 'learned') {
    const seed = (position + 1) * 97 + (dimIndex + 3) * 53;
    return Math.sin(seed) * 0.65 + Math.cos(seed * 0.37) * 0.35;
  }
  const angle = position / Math.pow(10000, (2 * Math.floor(dimIndex / 2)) / dimension);
  return dimIndex % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
}

function colorFor(value) {
  const normalized = (value + 1) / 2;
  const blue = Math.round(220 - normalized * 120);
  const green = Math.round(130 + normalized * 80);
  const red = Math.round(80 + normalized * 150);
  return `rgb(${red}, ${green}, ${blue})`;
}

function ControlButton({ active, children, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-lg border px-3 py-2 text-sm font-semibold transition ${
        active
          ? 'border-cyan-800 bg-cyan-700 text-white shadow-sm'
          : 'border-slate-200 bg-white text-slate-700 hover:border-cyan-400'
      }`}
    >
      {children}
    </button>
  );
}

function Metric({ label, value, helper }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-2 text-2xl font-bold text-slate-950">{value}</div>
      <p className="mt-1 text-sm text-slate-600">{helper}</p>
    </div>
  );
}

export default function PositionalEncodingAnimation() {
  const [sentenceIndex, setSentenceIndex] = useState(0);
  const [encodingType, setEncodingType] = useState('sinusoidal');
  const [dimension, setDimension] = useState(64);
  const [selectedPosition, setSelectedPosition] = useState(3);
  const [probePosition, setProbePosition] = useState(32);

  const tokens = SENTENCES[sentenceIndex];
  const visibleDims = 16;

  const encodingRows = useMemo(
    () =>
      tokens.map((token, position) => ({
        token,
        position,
        values: Array.from({ length: visibleDims }, (_, dimIndex) =>
          positionalValue(encodingType, position, dimIndex, dimension),
        ),
      })),
    [dimension, encodingType, tokens],
  );

  const selectedValues = useMemo(
    () => Array.from({ length: 12 }, (_, dimIndex) => positionalValue(encodingType, selectedPosition, dimIndex, dimension)),
    [dimension, encodingType, selectedPosition],
  );

  const similarity = useMemo(() => {
    const current = Array.from({ length: dimension }, (_, dimIndex) =>
      positionalValue(encodingType, selectedPosition, dimIndex, dimension),
    );
    const probe = Array.from({ length: dimension }, (_, dimIndex) =>
      positionalValue(encodingType, probePosition, dimIndex, dimension),
    );
    const dot = current.reduce((sum, value, index) => sum + value * probe[index], 0);
    const currentNorm = Math.sqrt(current.reduce((sum, value) => sum + value * value, 0));
    const probeNorm = Math.sqrt(probe.reduce((sum, value) => sum + value * value, 0));
    if (currentNorm === 0 || probeNorm === 0) return 0;
    return dot / (currentNorm * probeNorm);
  }, [dimension, encodingType, probePosition, selectedPosition]);

  const orderAmbiguity = encodingType === 'none' ? 'High' : 'Reduced';
  const extrapolation =
    encodingType === 'sinusoidal' ? 'Formula extends beyond trained positions' : encodingType === 'learned' ? 'Needs learned rows for new positions' : 'No ordering signal';

  return (
    <div className="min-h-full bg-slate-50">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 p-4 md:p-6">
        <header className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
            <div>
              <div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-cyan-700">
                <Waves size={17} />
                Transformer input order
              </div>
              <h1 className="mt-2 text-2xl font-bold text-slate-950 md:text-3xl">Positional encoding</h1>
              <p className="mt-2 max-w-3xl text-slate-700">
                Self-attention sees token interactions but has no built-in left-to-right order. Positional encodings add
                a position-dependent vector so identical words at different places become distinguishable.
              </p>
            </div>
            <div className="rounded-lg border border-cyan-200 bg-cyan-50 px-4 py-3 text-sm text-cyan-950">
              <div className="font-bold">Key question</div>
              <div>Can the model tell which token came first?</div>
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
              <div>
                <div className="mb-2 text-sm font-semibold text-slate-700">Sentence</div>
                <div className="space-y-2">
                  {SENTENCES.map((sentence, index) => (
                    <ControlButton
                      key={sentence.join('-')}
                      active={sentenceIndex === index}
                      onClick={() => {
                        setSentenceIndex(index);
                        setSelectedPosition(Math.min(selectedPosition, sentence.length - 1));
                      }}
                    >
                      {sentence.join(' ')}
                    </ControlButton>
                  ))}
                </div>
              </div>

              <div>
                <div className="mb-2 text-sm font-semibold text-slate-700">Position signal</div>
                <div className="grid grid-cols-1 gap-2">
                  {ENCODING_TYPES.map((type) => (
                    <ControlButton key={type.id} active={encodingType === type.id} onClick={() => setEncodingType(type.id)}>
                      {type.label}
                    </ControlButton>
                  ))}
                </div>
              </div>

              <div>
                <div className="mb-2 text-sm font-semibold text-slate-700">Model dimension</div>
                <div className="grid grid-cols-2 gap-2">
                  {DIMENSIONS.map((dim) => (
                    <ControlButton key={dim} active={dimension === dim} onClick={() => setDimension(dim)}>
                      {dim}
                    </ControlButton>
                  ))}
                </div>
              </div>

              <label className="block">
                <div className="mb-2 flex items-center justify-between text-sm font-semibold text-slate-700">
                  <span>Probe position</span>
                  <span>{probePosition}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="64"
                  step="1"
                  value={probePosition}
                  onChange={(event) => setProbePosition(Number(event.target.value))}
                  className="w-full accent-cyan-700"
                />
              </label>
            </div>
          </aside>

          <main className="space-y-4">
            <div className="grid gap-4 md:grid-cols-3">
              <Metric label="Order ambiguity" value={orderAmbiguity} helper="Without position, token order can collapse under attention." />
              <Metric label="Probe similarity" value={similarity.toFixed(2)} helper={`Position ${selectedPosition} compared with position ${probePosition}.`} />
              <Metric label="Extrapolation" value={encodingType === 'sinusoidal' ? 'Strong' : encodingType === 'learned' ? 'Limited' : 'None'} helper={extrapolation} />
            </div>

            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div>
                  <h2 className="text-lg font-bold text-slate-950">Token plus position</h2>
                  <p className="text-sm text-slate-600">
                    Click a token to inspect its position vector. The heatmap shows the first 16 position dimensions.
                  </p>
                </div>
                <div className="flex items-center gap-2 rounded-md bg-slate-100 px-3 py-2 text-sm font-semibold text-slate-700">
                  token embedding <ArrowRight size={14} /> token + position
                </div>
              </div>

              <div className="mt-4 space-y-3">
                {encodingRows.map((row) => (
                  <button
                    key={`${row.token}-${row.position}`}
                    type="button"
                    onClick={() => setSelectedPosition(row.position)}
                    className={`grid w-full gap-3 rounded-lg border p-3 text-left transition md:grid-cols-[110px_1fr] ${
                      selectedPosition === row.position
                        ? 'border-cyan-500 bg-cyan-50'
                        : 'border-slate-200 bg-white hover:border-cyan-300'
                    }`}
                  >
                    <div>
                      <div className="text-sm font-bold text-slate-950">{row.token}</div>
                      <div className="text-xs text-slate-500">position {row.position}</div>
                    </div>
                    <div className="grid grid-cols-8 gap-1 md:grid-cols-[repeat(16,minmax(0,1fr))]">
                      {row.values.map((value, dimIndex) => (
                        <div
                          key={dimIndex}
                          className="h-7 rounded-sm border border-white"
                          style={{ backgroundColor: colorFor(value) }}
                          title={`dim ${dimIndex}: ${value.toFixed(2)}`}
                        />
                      ))}
                    </div>
                  </button>
                ))}
              </div>
            </section>

            <section className="grid gap-4 md:grid-cols-3">
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <div className="flex items-center gap-2 font-bold text-slate-950">
                  <AlertCircle size={17} />
                  Predict before running
                </div>
                <p className="mt-2 text-sm text-slate-700">
                  Switch between "dog bites man" and "man bites dog". With no position signal, the bag of tokens is the
                  same, so order-sensitive meaning becomes hard to represent.
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <div className="flex items-center gap-2 font-bold text-slate-950">
                  <Braces size={17} />
                  Sinusoidal formula
                </div>
                <p className="mt-2 text-sm text-slate-700">
                  Even dimensions use sine and odd dimensions use cosine at different frequencies, giving each position
                  a repeatable multi-scale fingerprint.
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">Failure mode</h3>
                <p className="mt-2 text-sm text-slate-700">
                  Learned absolute positions can work well inside the trained context window, but unseen positions need
                  learned rows or another extrapolation strategy.
                </p>
              </div>
            </section>

            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <h2 className="text-lg font-bold text-slate-950">Selected vector slice</h2>
              <div className="mt-4 grid gap-2 md:grid-cols-12">
                {selectedValues.map((value, index) => (
                  <div key={index} className="rounded-md border border-slate-200 bg-slate-50 p-2 text-center">
                    <div className="text-xs font-semibold text-slate-500">d{index}</div>
                    <div className="text-sm font-bold text-slate-900">{value.toFixed(2)}</div>
                  </div>
                ))}
              </div>
            </section>
          </main>
        </section>

        <AssessmentPanel lessonId="positional-encoding" />
      </div>
    </div>
  );
}
