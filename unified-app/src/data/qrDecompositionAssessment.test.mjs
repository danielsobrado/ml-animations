import test from 'node:test';
import assert from 'node:assert/strict';

import { getLessonAssessment } from './lessonAssessments.js';

const LEVEL_ORDER = {
  Foundation: 0,
  Mechanism: 1,
  Application: 2,
  Tricky: 3,
  Interview: 4,
};

const LEVELS = new Set(Object.keys(LEVEL_ORDER));

function normalized(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function correctAnswer(question) {
  return question.choices[question.answerIndex];
}

test('qr decomposition has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('qr-decomposition');
  const ids = new Set(quiz.map((question) => question.id));
  const globalCounts = [0, 0, 0];

  assert.equal(quiz.length, 100);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'trace-projection-removal',
    'verify-qr-structure',
    'connect-qr-least-squares',
  ]);
  assert.equal(ids.size, 100);
  assert.ok(quiz.every((question) => !question.id.startsWith('generated-')));

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^qr-\d{3}-[a-z0-9-]+$/, `${question.id} should use the curated id format`);
    assert.equal(Number(question.id.slice(3, 6)), index + 1, `${question.id} should stay in numeric order`);
    assert.ok(LEVELS.has(question.level), `${question.id} should use a known level`);
    assert.ok(question.prompt && question.prompt.length > 20, `${question.id} should have a substantial prompt`);
    assert.equal(question.choices.length, 3, `${question.id} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, 3, `${question.id} should not repeat a choice`);
    assert.ok(Number.isInteger(question.answerIndex), `${question.id} should have an integer answer index`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `${question.id} has invalid answer index`);
    assert.ok(question.explanation && question.explanation.length > 30, `${question.id} should explain the answer`);
    globalCounts[question.answerIndex] += 1;
  }

  assert.ok(
    Math.max(...globalCounts) - Math.min(...globalCounts) <= 1,
    `answer positions should be globally balanced, got ${globalCounts.join(', ')}`,
  );
});

test('qr decomposition assessment avoids duplicate prompts and exact correct answers', () => {
  const { quiz } = getLessonAssessment('qr-decomposition');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length, 'prompts should be unique');
  assert.equal(new Set(answers).size, answers.length, 'exact correct answers should be unique');
});

test('qr decomposition assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('qr-decomposition');

  const expectedBands = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of expectedBands) {
    const band = quiz.slice(start, end);
    assert.ok(band.every((question) => question.level === level), `${level} band should occupy questions ${start + 1}-${end}`);
  }

  for (let index = 1; index < quiz.length; index += 1) {
    assert.ok(
      LEVEL_ORDER[quiz[index].level] >= LEVEL_ORDER[quiz[index - 1].level],
      `question ${index + 1} should not regress from ${quiz[index - 1].level} to ${quiz[index].level}`,
    );
  }
});

test('qr decomposition assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('qr-decomposition');
  const milestones = [
    [/main purpose of qr decomposition/, 0, 10],
    [/a qr/, 0, 10],
    [/columns of q/, 0, 12],
    [/r in qr/, 0, 12],
    [/gram schmidt/, 0, 20],
    [/least squares/, 10, 25],
    [/projection of a j onto q i/, 20, 35],
    [/what least squares system do we solve/, 20, 40],
    [/pivoted qr/, 30, 45],
    [/compare with normal equations on conditioning/, 35, 50],
    [/many rows fewer columns/, 50, 60],
    [/forming x t x/, 50, 65],
    [/trap/, 75, 90],
    [/summarize qr decomposition in an interview/, 90, 100],
  ];

  for (const [pattern, minIndex, maxIndex] of milestones) {
    const matchIndex = quiz.findIndex((question) => pattern.test(normalized(`${question.prompt} ${question.explanation}`)));
    assert.notEqual(matchIndex, -1, `missing learning point ${pattern}`);
    assert.ok(
      matchIndex >= minIndex && matchIndex < maxIndex,
      `${pattern} appears at question ${matchIndex + 1}, outside expected range ${minIndex + 1}-${maxIndex}`,
    );
  }
});

test('qr decomposition assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('qr-decomposition');
  const falseClaimPatterns = [
    /q is just a with scaled columns/,
    /always has zero residual/,
    /classical gram-schmidt is always production safe/,
    /r is always safely invertible/,
    /orthogonality fixes every numerical problem/,
    /forming normal equations is just as safe/,
    /qr factorization guarantees stable coefficients/,
    /pivoting destroys every qr product/,
    /guarantees perfect prediction/,
  ];

  for (const [index, question] of quiz.entries()) {
    const prompt = normalized(question.prompt);
    const answer = normalized(correctAnswer(question));
    const explicitTrapPrompt = /trap|misconception|what is wrong|risky|avoid|interview|contrast|why is this unsafe/.test(prompt);
    const falseClaimKeyed = falseClaimPatterns.some((pattern) => pattern.test(answer));

    assert.ok(
      !falseClaimKeyed || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('qr decomposition assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('qr-decomposition');
  const misconceptionPatterns = [
    /Q is just A with scaled columns/i,
    /QR least squares always has zero residual/i,
    /classical Gram-Schmidt risky/i,
    /normal equations not equivalent to QR/i,
    /R is always safely invertible/i,
    /snapshot-testing QR factors/i,
    /full Q when only reduced Q is needed/i,
    /ignoring pivoting in rank-revealing work/i,
    /Q entries as feature importance/i,
    /tall full-rank formula to every matrix shape/i,
    /orthogonality fixes every numerical problem/i,
    /computing R inverse explicitly/i,
    /ignore P in pivoted QR/i,
    /fixed zero threshold for R diagonals/i,
    /memorizing only A = QR/i,
  ];
  const trapPrompt = /trap|wrong|risky|avoid|what can go wrong|not equivalent|not imply|not automatically/i;

  for (const [index, question] of quiz.entries()) {
    const text = `${question.prompt} ${question.choices.join(' ')} ${question.explanation}`;
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('qr decomposition assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('qr-decomposition');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [questionIndex, question] of page.entries()) {
      const prompt = normalized(question.prompt);

      for (const [answerIndex, answer] of answers.entries()) {
        if (questionIndex === answerIndex) continue;

        assert.ok(
          !prompt.includes(answer),
          `question ${pageStart + questionIndex + 1} prompt should not reveal answer from question ${pageStart + answerIndex + 1}`,
        );
      }
    }
  }
});

test('qr decomposition assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('qr-decomposition');
  const pageSize = 10;

  assert.ok(new Set(quiz.map((question) => question.answerIndex)).size > 1);

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 1, 2].map((slot) => page.filter((question) => question.answerIndex === slot).length);
    const maxSameSlot = Math.max(...counts);
    const minSameSlot = Math.min(...counts);

    assert.ok(
      maxSameSlot - minSameSlot <= 1,
      `page starting at question ${pageStart + 1} should balance correct option slots, got ${counts.join('/')}`,
    );
  }
});
