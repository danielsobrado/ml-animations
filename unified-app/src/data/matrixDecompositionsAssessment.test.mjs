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

function normalized(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim();
}

function correctAnswer(question) {
  return question.choices[question.answerIndex];
}

test('matrix decompositions has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('matrix-decompositions');
  const ids = new Set(quiz.map((question) => question.id));
  const globalCounts = [0, 0, 0];

  assert.deepEqual(labs.map((lab) => lab.id), ['choose-by-goal']);

  assert.equal(quiz.length, 100);
  assert.equal(ids.size, 100);
  assert.ok(quiz.every((question) => !question.id.startsWith('generated-')));

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^md-\d{3}-[a-z0-9-]+$/);
    assert.ok(question.id.startsWith(`md-${String(index + 1).padStart(3, '0')}-`), `${question.id} should match question order`);
    assert.ok(question.prompt && /\S/.test(question.prompt), `${question.id} should have a prompt`);
    assert.equal(question.choices.length, 3, `${question.id} should have three choices`);
    assert.ok(Number.isInteger(question.answerIndex), `${question.id} should have an integer answer index`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `${question.id} has invalid answer index`);
    assert.ok(question.explanation && /\S/.test(question.explanation), `${question.id} should explain the answer`);
    assert.equal(new Set(question.choices.map(normalized)).size, 3, `${question.id} should not repeat choices`);
    globalCounts[question.answerIndex] += 1;
  }

  assert.ok(
    Math.max(...globalCounts) - Math.min(...globalCounts) <= 1,
    `answer positions should be globally balanced, got ${globalCounts.join(', ')}`,
  );
});

test('matrix decompositions assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('matrix-decompositions');

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

test('matrix decompositions assessment covers chooser learning points in order', () => {
  const { quiz } = getLessonAssessment('matrix-decompositions');
  const sections = [
    quiz.slice(0, 20).map((question) => normalized(`${question.prompt} ${question.explanation}`)).join(' '),
    quiz.slice(20, 50).map((question) => normalized(`${question.prompt} ${question.explanation}`)).join(' '),
    quiz.slice(50, 75).map((question) => normalized(`${question.prompt} ${question.explanation}`)).join(' '),
    quiz.slice(75).map((question) => normalized(`${question.prompt} ${question.explanation}`)).join(' '),
  ];

  assert.match(sections[0], /lu/);
  assert.match(sections[0], /qr/);
  assert.match(sections[0], /svd/);
  assert.match(sections[0], /cholesky/);
  assert.match(sections[0], /nmf/);
  assert.match(sections[1], /stability|condition|pivot|rank/);
  assert.match(sections[1], /rectangular|spd|orthogonal/);
  assert.match(sections[2], /scenario|production|solve|compression|least squares/);
  assert.match(sections[3], /trap|interview|misconception|tradeoff/);
});

test('matrix decompositions assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('matrix-decompositions');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const correctAnswers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(correctAnswers).size, correctAnswers.length);
});

test('matrix decompositions assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('matrix-decompositions');
  const falseClaimPatterns = [
    /always the best/,
    /every square matrix has a full stable/,
    /always exact/,
    /all decompositions are interchangeable/,
    /cholesky because every matrix is spd/,
    /use explicit inverse/,
  ];

  for (const [index, question] of quiz.entries()) {
    const prompt = normalized(question.prompt);
    const answer = normalized(correctAnswer(question));
    const explicitTrapPrompt = /trap|misconception|what is wrong|dangerous|risky|interview|contrast/.test(prompt);
    const falseClaimKeyed = falseClaimPatterns.some((pattern) => pattern.test(answer));

    assert.ok(
      !falseClaimKeyed || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('matrix decompositions assessment marks misconceptions as traps after setup', () => {
  const { quiz } = getLessonAssessment('matrix-decompositions');
  const misconceptionPatterns = [
    /SVD is always the best/i,
    /just because A is square/i,
    /general rectangular A/i,
    /normal equations look attractive but be risky/i,
    /arbitrary signed data/i,
    /low-rank approximation hide/i,
    /LU without pivoting/i,
    /Every square matrix has a full stable eigenvector basis/i,
    /choosing k only because it looks small/i,
    /Q being orthonormal not imply/i,
    /A\^T A not automatically harmless/i,
    /not assume about NMF factors/i,
    /avoid forming an explicit inverse/i,
    /memorizing only formulas/i,
  ];
  const trapPrompt = /trap|wrong|dangerous|risky|not imply|not automatically|not assume|avoid|misunderstands|what can fail|what can .* hide/i;

  for (const [index, question] of quiz.entries()) {
    const text = `${question.prompt} ${question.choices.join(' ')} ${question.explanation}`;
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('matrix decompositions assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('matrix-decompositions');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact correct answers`);

    for (const [questionIndex, question] of page.entries()) {
      const prompt = normalized(question.prompt);

      for (const [answerIndex, answer] of answers.entries()) {
        if (questionIndex === answerIndex || answer.length < 8) continue;

        assert.ok(
          !prompt.includes(answer),
          `question ${pageStart + questionIndex + 1} prompt should not reveal answer from question ${pageStart + answerIndex + 1}`,
        );
      }
    }
  }
});

test('matrix decompositions assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('matrix-decompositions');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const question of page) counts[question.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
