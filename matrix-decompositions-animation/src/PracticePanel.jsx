import React, { useMemo, useState } from 'react';
import { CheckCircle2, RotateCcw, XCircle } from 'lucide-react';

const QUESTIONS = [
  {
    scenario: 'You need to solve many systems Ax = b with the same dense square A.',
    choices: ['LU', 'NMF', 'PCA', 'Eigen'],
    answer: 'LU',
    why: 'LU is built for repeated direct solves once the matrix has been factored.',
  },
  {
    scenario: 'A is symmetric positive definite and you want the fastest stable direct solve.',
    choices: ['Cholesky', 'QR', 'NMF', 'Truncated SVD'],
    answer: 'Cholesky',
    why: 'Cholesky uses A = L L^T and exploits SPD structure.',
  },
  {
    scenario: 'You want a numerically stable least-squares solution for a tall matrix.',
    choices: ['QR', 'LU', 'NMF', 'Eigen'],
    answer: 'QR',
    why: 'QR builds an orthonormal basis for the column space and avoids normal-equation instability.',
  },
  {
    scenario: 'You need the best rank-k linear approximation to a general rectangular matrix.',
    choices: ['Truncated SVD', 'LU', 'Cholesky', 'Eigen'],
    answer: 'Truncated SVD',
    why: 'The top k singular values and vectors give the best rank-k approximation in common matrix norms.',
  },
  {
    scenario: 'You want interpretable additive parts for a nonnegative document-term matrix.',
    choices: ['NMF', 'QR', 'Cholesky', 'LU'],
    answer: 'NMF',
    why: 'NMF keeps factors nonnegative, making parts and activations easier to interpret.',
  },
];

export default function PracticePanel() {
  const [current, setCurrent] = useState(0);
  const [selected, setSelected] = useState(null);
  const [answered, setAnswered] = useState(false);
  const [score, setScore] = useState(0);

  const question = QUESTIONS[current];
  const complete = current === QUESTIONS.length - 1 && answered;
  const isCorrect = selected === question.answer;

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
  };

  const reset = () => {
    setCurrent(0);
    setSelected(null);
    setAnswered(false);
    setScore(0);
  };

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-5 p-4 text-slate-900">
      <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-rose-600">Practice lab</p>
            <h2 className="mt-1 text-2xl font-bold">Choose the Factorization</h2>
            <p className="mt-2 text-sm text-slate-600">
              Pick the matrix decomposition that best fits the scenario.
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

      <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <p className="text-sm font-semibold text-slate-500">
          Question {current + 1} of {QUESTIONS.length}
        </p>
        <h3 className="mt-2 text-xl font-bold">{question.scenario}</h3>

        <div className="mt-5 grid gap-3 md:grid-cols-2">
          {question.choices.map((choice) => {
            const chosen = selected === choice;
            const correctChoice = answered && choice === question.answer;
            const wrongChoice = answered && chosen && choice !== question.answer;

            return (
              <button
                key={choice}
                onClick={() => !answered && setSelected(choice)}
                className={`flex min-h-[62px] items-center justify-between rounded-lg border px-4 py-3 text-left text-sm font-bold transition ${
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
            className={`mt-5 rounded-lg border p-4 text-sm ${
              isCorrect
                ? 'border-emerald-200 bg-emerald-50 text-emerald-900'
                : 'border-red-200 bg-red-50 text-red-900'
            }`}
          >
            <p className="font-bold">{isCorrect ? 'Correct' : 'Not quite'}</p>
            <p className="mt-1">{question.why}</p>
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
          <div className="mt-5 rounded-lg border border-cyan-200 bg-cyan-50 p-4 text-center text-cyan-900">
            <p className="text-lg font-bold">Practice complete</p>
            <p className="mt-1 text-sm">Final score: {score} / {QUESTIONS.length}</p>
          </div>
        )}
      </div>
    </div>
  );
}
