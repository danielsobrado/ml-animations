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

test('k-means has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('k-means');
  const ids = new Set(quiz.map((question) => question.id));
  const globalCounts = [0, 0, 0];

  assert.equal(quiz.length, 100);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'k-vs-inertia',
    'trace-assign-update-loop',
    'stress-test-cluster-assumptions',
  ]);
  assert.equal(ids.size, 100);
  assert.ok(quiz.every((question) => !question.id.startsWith('generated-')));

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^km-\d{3}-[a-z0-9-]+$/, `${question.id} should use the curated id format`);
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

test('k-means assessment avoids duplicate prompts and exact correct answers', () => {
  const { quiz } = getLessonAssessment('k-means');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length, 'prompts should be unique');
  assert.equal(new Set(answers).size, answers.length, 'exact correct answers should be unique');
});

test('k-means assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('k-means');

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

test('k-means assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('k-means');
  const milestones = [
    [/problem does k means solve/, 0, 8],
    [/what does k mean/, 0, 8],
    [/assignment step/, 0, 10],
    [/centroid updated/, 0, 12],
    [/objective does k means try to reduce/, 0, 15],
    [/feature scaling matter/, 10, 22],
    [/sum i x i c ai 2/, 20, 30],
    [/centroid update the mean/, 20, 32],
    [/k means starts/, 25, 40],
    [/elbow heuristic/, 30, 45],
    [/inertia keeps dropping as k rises/, 50, 62],
    [/two random seeds produce different segmentations/, 50, 66],
    [/trap/, 75, 90],
    [/summarize k means in an interview/, 90, 100],
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

test('k-means assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('k-means');
  const falseClaimPatterns = [
    /lower inertia as always better/,
    /clusters equal classes/,
    /trusting one random initialization/,
    /scaling has no effect/,
    /k means outputs calibrated probabilities/,
    /outliers are always removed automatically/,
    /always discovers ground truth/,
    /cluster ids are ordered scores/,
  ];

  for (const [index, question] of quiz.entries()) {
    const prompt = normalized(question.prompt);
    const answer = normalized(correctAnswer(question));
    const explicitTrapPrompt = /trap|misconception|what is wrong|risky|interview|contrast|why is this unsafe/.test(prompt);
    const falseClaimKeyed = falseClaimPatterns.some((pattern) => pattern.test(answer));

    assert.ok(
      !falseClaimKeyed || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('k-means assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('k-means');
  const misconceptionPatterns = [
    /lower inertia as always better/i,
    /clusters equal classes/i,
    /raw mixed-unit features/i,
    /one random initialization/i,
    /forcing k-means on nonconvex shapes/i,
    /outlier trap/i,
    /cluster id 2 as greater/i,
    /ignoring empty clusters/i,
    /elbow method not a theorem/i,
    /limitation of silhouette/i,
    /high-dimensional clustering/i,
    /leakage risk/i,
    /one-hot k-means/i,
    /without monitoring/i,
    /memorizing only the k-means objective/i,
  ];
  const trapPrompt = /trap|wrong|risky|dangerous|limitation|leakage|not a theorem|not automatically meaningful/i;

  for (const [index, question] of quiz.entries()) {
    const text = `${question.prompt} ${question.choices.join(' ')} ${question.explanation}`;
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('k-means assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('k-means');
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

test('k-means assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('k-means');
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
