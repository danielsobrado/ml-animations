import React, { useMemo, useState } from 'react';
import { CheckCircle2, HelpCircle, RotateCcw, XCircle } from 'lucide-react';

const QUESTIONS = [
  {
    prompt: 'Which subspace contains all solutions to Ax = 0?',
    choices: ['Null(A)', 'Row(A)', 'Col(A)', 'Null(A^T)'],
    answer: 'Null(A)',
    explanation: 'The null space is defined as all vectors x in the domain satisfying Ax = 0.',
  },
  {
    prompt: 'For A as an m x n matrix with rank r, what is dim Null(A)?',
    choices: ['n - r', 'm - r', 'r', 'm + n - r'],
    answer: 'n - r',
    explanation: 'Null(A) lives in R^n, so rank-nullity gives dim Null(A) = n - r.',
  },
  {
    prompt: 'What condition decides whether Ax = b is consistent?',
    choices: ['b is in Col(A)', 'b is in Row(A)', 'x is in Null(A)', 'A has no rows'],
    answer: 'b is in Col(A)',
    explanation: 'The column space is the set of all reachable outputs Ax.',
  },
  {
    prompt: 'Which orthogonality statement is correct?',
    choices: ['Row(A) = Null(A)^perp', 'Col(A) = Row(A)^perp', 'Null(A) = Null(A^T)^perp', 'Row(A) = Col(A)^perp'],
    answer: 'Row(A) = Null(A)^perp',
    explanation: 'Inside the domain R^n, row space and null space are orthogonal complements.',
  },
  {
    prompt: 'What does a vector y in Null(A^T) test?',
    choices: ['Compatibility y^T b = 0', 'The diagonal of A', 'The norm of x', 'The determinant of every submatrix'],
    answer: 'Compatibility y^T b = 0',
    explanation: 'Left-null vectors are orthogonal to Col(A), so a consistent b must be orthogonal to every such y.',
  },
];

const MATCHING = [
  { label: 'Row(A)', value: 'Subspace of R^n with dimension r' },
  { label: 'Null(A)', value: 'Subspace of R^n with dimension n - r' },
  { label: 'Col(A)', value: 'Subspace of R^m with dimension r' },
  { label: 'Null(A^T)', value: 'Subspace of R^m with dimension m - r' },
];

function MatrixMiniTable() {
  return (
    <div className="grid gap-2 rounded-lg border border-slate-200 bg-white p-3 text-sm shadow-sm md:grid-cols-4">
      {MATCHING.map((item) => (
        <div key={item.label} className="rounded-lg bg-slate-100 p-3">
          <p className="font-mono font-bold text-slate-950">{item.label}</p>
          <p className="mt-1 text-xs text-slate-600">{item.value}</p>
        </div>
      ))}
    </div>
  );
}

export default function PracticePanel() {
  const [current, setCurrent] = useState(0);
  const [selected, setSelected] = useState(null);
  const [answered, setAnswered] = useState(false);
  const [score, setScore] = useState(0);
  const [showHint, setShowHint] = useState(false);

  const question = QUESTIONS[current];
  const isCorrect = selected === question.answer;
  const complete = current === QUESTIONS.length - 1 && answered;

  const progress = useMemo(() => {
    const answeredCount = answered ? current + 1 : current;
    return Math.round((answeredCount / QUESTIONS.length) * 100);
  }, [answered, current]);

  const submit = () => {
    if (!selected || answered) return;
    setAnswered(true);
    if (selected === question.answer) setScore((value) => value + 1);
  };

  const next = () => {
    if (current >= QUESTIONS.length - 1) return;
    setCurrent((value) => value + 1);
    setSelected(null);
    setAnswered(false);
    setShowHint(false);
  };

  const reset = () => {
    setCurrent(0);
    setSelected(null);
    setAnswered(false);
    setScore(0);
    setShowHint(false);
  };

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-5 p-4 text-slate-900">
      <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-rose-600">Practice lab</p>
            <h2 className="mt-1 text-2xl font-bold">Subspace Quiz</h2>
            <p className="mt-2 text-sm text-slate-600">
              Identify the subspace, dimension, and consistency rule that belongs to each statement.
            </p>
          </div>
          <div className="rounded-lg bg-slate-100 px-4 py-3 text-center">
            <p className="text-xs uppercase tracking-wide text-slate-500">Score</p>
            <p className="mt-1 text-xl font-bold">{score} / {QUESTIONS.length}</p>
          </div>
        </div>

        <div className="mt-5 h-2 overflow-hidden rounded-full bg-slate-100">
          <div className="h-full bg-rose-500 transition-all" style={{ width: `${progress}%` }} />
        </div>
      </div>

      <MatrixMiniTable />

      <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <p className="text-sm font-semibold text-slate-500">
              Question {current + 1} of {QUESTIONS.length}
            </p>
            <h3 className="mt-2 text-xl font-bold">{question.prompt}</h3>
          </div>
          <button
            onClick={() => setShowHint((value) => !value)}
            className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-amber-100 text-amber-700 transition hover:bg-amber-200"
            title="Show hint"
          >
            <HelpCircle size={18} />
          </button>
        </div>

        {showHint && (
          <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
            Use the ambient space first: Row(A) and Null(A) live in R^n; Col(A) and Null(A^T) live in R^m.
          </div>
        )}

        <div className="grid gap-3 md:grid-cols-2">
          {question.choices.map((choice) => {
            const chosen = selected === choice;
            const correctChoice = answered && choice === question.answer;
            const wrongChoice = answered && chosen && choice !== question.answer;

            return (
              <button
                key={choice}
                onClick={() => !answered && setSelected(choice)}
                className={`flex min-h-[64px] items-center justify-between rounded-lg border px-4 py-3 text-left text-sm font-semibold transition ${
                  correctChoice
                    ? 'border-emerald-300 bg-emerald-50 text-emerald-900'
                    : wrongChoice
                      ? 'border-red-300 bg-red-50 text-red-900'
                      : chosen
                        ? 'border-rose-300 bg-rose-50 text-rose-900'
                        : 'border-slate-200 bg-slate-50 text-slate-700 hover:border-rose-200 hover:bg-rose-50'
                }`}
              >
                <span>{choice}</span>
                {correctChoice && <CheckCircle2 size={18} />}
                {wrongChoice && <XCircle size={18} />}
              </button>
            );
          })}
        </div>

        {answered && (
          <div
            className={`mt-4 rounded-lg border p-4 text-sm ${
              isCorrect
                ? 'border-emerald-200 bg-emerald-50 text-emerald-900'
                : 'border-red-200 bg-red-50 text-red-900'
            }`}
          >
            <p className="font-bold">{isCorrect ? 'Correct' : 'Not quite'}</p>
            <p className="mt-1">{question.explanation}</p>
          </div>
        )}

        <div className="mt-5 flex flex-wrap items-center gap-3">
          {!complete && (
            <>
              <button
                onClick={submit}
                disabled={!selected || answered}
                className="rounded-lg bg-rose-600 px-4 py-2 text-sm font-bold text-white transition hover:bg-rose-700 disabled:cursor-not-allowed disabled:bg-slate-400"
              >
                Submit
              </button>
              <button
                onClick={next}
                disabled={!answered || current >= QUESTIONS.length - 1}
                className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-bold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
              >
                Next
              </button>
            </>
          )}
          <button
            onClick={reset}
            className="inline-flex items-center gap-2 rounded-lg bg-slate-100 px-4 py-2 text-sm font-bold text-slate-700 transition hover:bg-slate-200"
          >
            <RotateCcw size={16} />
            Reset
          </button>
        </div>

        {complete && (
          <div className="mt-5 rounded-lg border border-indigo-200 bg-indigo-50 p-4 text-center text-indigo-900">
            <p className="text-lg font-bold">Quiz complete</p>
            <p className="mt-1 text-sm">Final score: {score} / {QUESTIONS.length}</p>
          </div>
        )}
      </div>
    </div>
  );
}
