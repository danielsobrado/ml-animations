import React, { useMemo, useState } from 'react';
import { CheckCircle2, FlaskConical, MousePointer2, Target } from 'lucide-react';

const cases = [
  { id: 'near', name: 'Nearly in Col(A)', x: 270, y: 120, residual: 36, note: 'Small residual, close fit.' },
  { id: 'far', name: 'Far from Col(A)', x: 300, y: 62, residual: 90, note: 'Large residual, weak fit.' },
  { id: 'exact', name: 'Exactly consistent', x: 230, y: 166, residual: 0, note: 'Residual is zero, so b is already in Col(A).' },
];

const questions = [
  {
    prompt: 'What makes the least-squares residual special?',
    choices: ['It is orthogonal to Col(A).', 'It is always inside Col(A).', 'It maximizes ||Ax - b||.'],
    answer: 0,
  },
  {
    prompt: 'Which equation characterizes the least-squares solution?',
    choices: ['A^T A x = A^T b', 'A x = 0', 'A^T y = b'],
    answer: 0,
  },
];

export default function LeastSquaresProjection() {
  const [caseId, setCaseId] = useState('near');
  const [selected, setSelected] = useState(null);
  const [questionIndex, setQuestionIndex] = useState(0);
  const active = cases.find((item) => item.id === caseId) || cases[0];
  const q = questions[questionIndex];

  const projection = useMemo(() => {
    const t = Math.max(0, Math.min(1, (active.x - 90) / 245));
    return { x: 92 + t * 255, y: 226 - t * 104 };
  }, [active]);

  return (
    <div className="min-h-full bg-slate-950 p-4 text-slate-100">
      <div className="mx-auto grid max-w-7xl gap-4 lg:grid-cols-[1.1fr_0.9fr]">
        <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h1 className="text-2xl font-bold">Least Squares Projection</h1>
              <p className="text-sm text-slate-400">Project b onto Col(A), then measure the orthogonal residual.</p>
            </div>
            <div className="flex flex-wrap gap-2">
              {cases.map((item) => (
                <button
                  key={item.id}
                  onClick={() => setCaseId(item.id)}
                  className={`rounded-md px-3 py-2 text-sm font-semibold transition ${
                    caseId === item.id ? 'bg-cyan-400 text-slate-950' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  {item.name}
                </button>
              ))}
            </div>
          </div>

          <svg viewBox="0 0 420 290" className="h-[360px] w-full rounded-md bg-slate-950">
            <defs>
              <marker id="leastSquaresArrow" markerHeight="8" markerWidth="8" orient="auto" refX="8" refY="4">
                <path d="M0,0 L8,4 L0,8 Z" fill="#22d3ee" />
              </marker>
            </defs>
            <line x1="52" y1="242" x2="372" y2="112" stroke="#334155" strokeWidth="18" strokeLinecap="round" />
            <line x1="52" y1="242" x2="372" y2="112" stroke="#38bdf8" strokeWidth="3" />
            <text x="275" y="95" fill="#67e8f9" fontSize="15">Col(A)</text>
            <line x1="65" y1="245" x2={active.x} y2={active.y} stroke="#94a3b8" strokeWidth="2" markerEnd="url(#leastSquaresArrow)" />
            <line x1="65" y1="245" x2={projection.x} y2={projection.y} stroke="#22d3ee" strokeWidth="4" markerEnd="url(#leastSquaresArrow)" />
            <line x1={projection.x} y1={projection.y} x2={active.x} y2={active.y} stroke="#fb7185" strokeDasharray="7 6" strokeWidth="4" />
            <circle cx={active.x} cy={active.y} r="8" fill="#fb7185" />
            <circle cx={projection.x} cy={projection.y} r="8" fill="#22d3ee" />
            <text x={active.x + 12} y={active.y - 8} fill="#fecdd3" fontSize="16">b</text>
            <text x={projection.x + 10} y={projection.y + 22} fill="#cffafe" fontSize="16">b_hat = Ax_hat</text>
            <text x={(projection.x + active.x) / 2 + 8} y={(projection.y + active.y) / 2} fill="#fecdd3" fontSize="15">e</text>
          </svg>
        </section>

        <aside className="grid gap-4">
          <div className="rounded-lg border border-cyan-400/30 bg-cyan-400/10 p-5">
            <div className="mb-3 flex items-center gap-2 text-cyan-200">
              <Target size={20} />
              <h2 className="text-lg font-bold">Core Picture</h2>
            </div>
            <div className="space-y-3 text-sm text-slate-200">
              <p><span className="font-semibold text-white">b = b_hat + e</span>, where b_hat lives in Col(A).</p>
              <p>The residual e = b - A x_hat is perpendicular to every column of A.</p>
              <p className="rounded-md bg-slate-950 p-3 font-mono text-cyan-200">{'A^T e = 0  =>  A^T A x_hat = A^T b'}</p>
              <p className="text-slate-300">{active.note}</p>
            </div>
          </div>

          <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
            <div className="mb-3 flex items-center gap-2">
              <FlaskConical size={20} className="text-rose-300" />
              <h2 className="text-lg font-bold">Practice Lab</h2>
            </div>
            <p className="mb-3 text-sm text-slate-300">{q.prompt}</p>
            <div className="space-y-2">
              {q.choices.map((choice, index) => (
                <button
                  key={choice}
                  onClick={() => setSelected(index)}
                  className={`flex w-full items-center gap-2 rounded-md border px-3 py-2 text-left text-sm ${
                    selected === index
                      ? index === q.answer
                        ? 'border-emerald-400 bg-emerald-400/10 text-emerald-100'
                        : 'border-rose-400 bg-rose-400/10 text-rose-100'
                      : 'border-slate-700 bg-slate-950 text-slate-300 hover:border-slate-500'
                  }`}
                >
                  {selected === index ? <CheckCircle2 size={16} /> : <MousePointer2 size={16} />}
                  {choice}
                </button>
              ))}
            </div>
            <button
              onClick={() => {
                setQuestionIndex((questionIndex + 1) % questions.length);
                setSelected(null);
              }}
              className="mt-4 rounded-md bg-slate-100 px-4 py-2 text-sm font-semibold text-slate-950"
            >
              Next prompt
            </button>
          </div>
        </aside>
      </div>
    </div>
  );
}
