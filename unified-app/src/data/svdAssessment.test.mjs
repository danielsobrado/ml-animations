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

test('svd has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('svd');
  const ids = new Set(quiz.map((question) => question.id));
  const globalCounts = [0, 0, 0];

  assert.equal(quiz.length, 100);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'trace-svd-factor-flow',
    'verify-singular-values',
    'connect-svd-uses',
  ]);
  assert.equal(ids.size, 100);
  assert.ok(quiz.every((question) => !question.id.startsWith('generated-')));

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^svd-\d{3}-[a-z0-9-]+$/, `${question.id} should use the curated id format`);
    assert.equal(Number(question.id.slice(4, 7)), index + 1, `${question.id} should stay in numeric order`);
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

test('svd assessment avoids duplicate prompts and exact correct answers', () => {
  const { quiz } = getLessonAssessment('svd');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length, 'prompts should be unique');
  assert.equal(new Set(answers).size, answers.length, 'exact correct answers should be unique');
});

test('svd assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('svd');

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

test('svd assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('svd');
  const milestones = [
    [/main purpose of svd/, 0, 10],
    [/a u sigma v t/, 0, 12],
    [/what does u contain/, 0, 12],
    [/what does sigma contain/, 0, 15],
    [/broadly useful/, 0, 20],
    [/reconstructs a/, 5, 25],
    [/which equation defines a singular triplet/, 20, 35],
    [/a t a/, 20, 40],
    [/eckart young/, 25, 45],
    [/low rank image approximation/, 50, 60],
    [/pca directions/, 50, 65],
    [/rank deficient/, 50, 70],
    [/trap/, 75, 90],
    [/summarize svd in an interview/, 90, 100],
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

test('svd assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('svd');
  const falseClaimPatterns = [
    /singular vectors are just eigenvectors of a/,
    /blindly inverting every singular value/,
    /singular values can be negative/,
    /rectangular matrices have no singular values/,
    /a t a is always safer/,
    /always the best production method/,
    /truncated svd to reconstruct every entry exactly/,
    /always gives human-interpretable topics/,
    /only works for symmetric matrices/,
  ];

  for (const [index, question] of quiz.entries()) {
    const prompt = normalized(question.prompt);
    const answer = normalized(correctAnswer(question));
    const explicitTrapPrompt = /trap|misconception|what is wrong|dangerous|risky|interview|contrast|why is this unsafe/.test(prompt);
    const falseClaimKeyed = falseClaimPatterns.some((pattern) => pattern.test(answer));

    assert.ok(
      !falseClaimKeyed || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('svd assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('svd');
  const trapIds = [
    'svd-076-eigen-trap',
    'svd-077-small-values-trap',
    'svd-078-k-trap',
    'svd-079-sign-trap',
    'svd-080-feature-importance-trap',
    'svd-081-mean-trap',
    'svd-082-compression-trap',
    'svd-083-uniqueness-trap',
    'svd-084-rectangular-trap',
    'svd-085-cost-trap',
    'svd-086-scale-trap',
    'svd-087-ata-trap',
    'svd-088-exact-truncation-trap',
    'svd-089-positive-components-trap',
    'svd-090-formula-only-trap',
  ];
  const misconceptionPatterns = [
    /singular vectors are just eigenvectors of A/i,
    /blindly inverting every singular value/i,
    /choosing k only because it looks small/i,
    /comparisons between SVD runs/i,
    /direct feature importance/i,
    /uncentered PCA via SVD mislead/i,
    /low-rank compression hide/i,
    /singular values are tied/i,
    /forcing rectangular A into ordinary eigendecomposition/i,
    /SVD is always the best production method/i,
    /raw SVD on mixed-unit features/i,
    /eigendecomposing A\^T A risky/i,
    /truncated SVD to reconstruct every entry exactly/i,
    /expecting SVD components to be nonnegative parts/i,
    /memorizing only A = U Sigma V\^T/i,
  ];
  const trapPrompt = /trap|wrong|dangerous|mislead|risky|what can .* hide|non-uniqueness/i;

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

  for (const [index, question] of quiz.entries()) {
    const text = `${question.prompt} ${question.choices.join(' ')} ${question.explanation}`;
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('svd assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('svd');
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

test('svd assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('svd');
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
