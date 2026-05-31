import assert from 'node:assert/strict';
import test from 'node:test';
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

test('data leakage deep dive has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('data-leakage-deep-dive');
  const ids = new Set(quiz.map((question) => question.id));
  const globalCounts = [0, 0, 0];

  assert.equal(quiz.length, 100);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'trace-information-boundary',
    'leakage-mode-audit',
    'choose-boundary-safe-fix',
  ]);
  assert.equal(ids.size, 100);
  assert.ok(quiz.every((question) => !question.id.startsWith('generated-')));

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^leak-\d{3}-[a-z0-9-]+$/, `${question.id} should use the curated id format`);
    assert.equal(Number(question.id.slice(5, 8)), index + 1, `${question.id} should stay in numeric order`);
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

test('data leakage deep dive assessment avoids duplicate prompts and exact correct answers', () => {
  const { quiz } = getLessonAssessment('data-leakage-deep-dive');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length, 'prompts should be unique');
  assert.equal(new Set(answers).size, answers.length, 'exact correct answers should be unique');
});

test('data leakage deep dive assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('data-leakage-deep-dive');
  const expectedBands = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of expectedBands) {
    assert.deepEqual(
      [...new Set(quiz.slice(start, end).map((question) => question.level))],
      [level],
      `${level} band should occupy questions ${start + 1}-${end}`,
    );
  }

  for (let index = 1; index < quiz.length; index += 1) {
    assert.ok(
      LEVEL_ORDER[quiz[index].level] >= LEVEL_ORDER[quiz[index - 1].level],
      `question ${index + 1} should not move backward in difficulty`,
    );
  }
});

test('data leakage deep dive assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('data-leakage-deep-dive');
  const milestones = [
    [/what is data leakage/, 0, 8],
    [/target derived feature/, 0, 10],
    [/duplicate entities leak/, 4, 12],
    [/time order create leakage/, 5, 14],
    [/preprocessing leak/, 6, 14],
    [/feature selection leak/, 8, 16],
    [/target encoding leak prone/, 10, 18],
    [/repeated test peeking/, 8, 18],
    [/split unit/, 12, 20],
    [/safe pipeline rule/, 12, 22],
    [/leakage mechanism/, 20, 30],
    [/post event field/, 20, 32],
    [/user history features leak/, 20, 35],
    [/fold safe target encoding/, 30, 42],
    [/cross validation/, 35, 45],
    [/serving parity/, 45, 55],
    [/hospital readmission model/, 50, 60],
    [/standardscaler is fit before train test split/, 50, 60],
    [/test score is disappointing/, 60, 70],
    [/leakage claim is false/, 75, 90],
    [/define data leakage in an interview/, 90, 100],
  ];

  for (const [pattern, minIndex, maxIndex] of milestones) {
    const matchIndex = quiz.findIndex((question) => pattern.test(normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`)));
    assert.notEqual(matchIndex, -1, `missing learning point ${pattern}`);
    assert.ok(
      matchIndex >= minIndex && matchIndex < maxIndex,
      `${pattern} appears at question ${matchIndex + 1}, outside expected range ${minIndex + 1}-${maxIndex}`,
    );
  }
});

test('data leakage deep dive assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('data-leakage-deep-dive');
  const unsafePatterns = [
    /a feature is safe whenever it is highly correlated/,
    /unsupervised transforms cannot leak/,
    /random row split is always honest/,
    /shuffling is safe/,
    /full data target means are fine/,
    /creating augmented variants before splitting originals/,
    /choosing row removal rules/,
    /proves production performance/,
    /a feature is safe if its name/,
    /queried indefinitely without overfitting risk/,
    /no leakage exists anywhere/,
    /current warehouse snapshot always represents/,
    /serving checks are unnecessary/,
    /cv automatically prevents all leakage/,
    /validation score is high so leakage is impossible/,
  ];

  for (const [index, question] of quiz.entries()) {
    const prompt = normalized(question.prompt);
    const answer = normalized(correctAnswer(question));
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /trap|false|misleading|unsafe|too strong|wrong|challenge|suspicious|claim|behavior|interpretation|practice/.test(prompt);

    assert.ok(
      !unsafeAnswer || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('data leakage deep dive assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('data-leakage-deep-dive');
  const trapIds = [
    'leak-076-trap-correlation',
    'leak-077-trap-unsupervised',
    'leak-078-trap-random',
    'leak-079-trap-time',
    'leak-080-trap-test',
    'leak-081-trap-target-encoding',
    'leak-082-trap-duplicates',
    'leak-083-trap-aggregates',
    'leak-084-trap-cleaning',
    'leak-085-trap-baseline',
    'leak-086-trap-proxy',
    'leak-087-trap-leaderboard',
    'leak-088-trap-ablation',
    'leak-089-trap-snapshot',
    'leak-090-trap-monitoring',
  ];
  const misconceptionPatterns = [
    /feature is safe whenever it is highly correlated/i,
    /unsupervised transforms cannot leak/i,
    /random row split is always honest/i,
    /shuffling is safe when future rows share/i,
    /changing features after each final test result/i,
    /full-data target means are fine/i,
    /creating augmented variants before splitting originals/i,
    /lifetime spend computed using events after/i,
    /row-removal rules after inspecting final test/i,
    /extremely strong simple baseline/i,
    /name does not mention the target/i,
    /queried indefinitely without overfitting risk/i,
    /no leakage exists anywhere/i,
    /current warehouse snapshot always represents/i,
    /serving checks are unnecessary/i,
  ];
  const trapPrompt = /trap|false|unsafe|misleading|suspicious|too strong|claim|behavior|interpretation|practice|conclusion/i;

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

  for (const [index, question] of quiz.entries()) {
    const text = `${question.prompt} ${question.choices.join(' ')} ${question.explanation}`;
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('data leakage deep dive assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('data-leakage-deep-dive');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [answerIndex, answer] of answers.entries()) {
      for (const [promptIndex, question] of page.entries()) {
        if (answerIndex === promptIndex) continue;
        assert.ok(
          !normalized(question.prompt).includes(answer),
          `question ${pageStart + promptIndex + 1} prompt should not reveal answer from question ${pageStart + answerIndex + 1}`,
        );
      }
    }
  }
});

test('data leakage deep dive assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('data-leakage-deep-dive');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const counts = [0, 1, 2].map((slot) => page.filter((question) => question.answerIndex === slot).length);
    const maxSameSlot = Math.max(...counts);
    const minSameSlot = Math.min(...counts);

    assert.ok(
      maxSameSlot - minSameSlot <= 1,
      `page starting at question ${pageStart + 1} should balance correct option slots, got ${counts.join('/')}`,
    );
  }
});
