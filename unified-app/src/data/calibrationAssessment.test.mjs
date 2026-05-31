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

function normalized(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function correctAnswer(question) {
  return question.choices[question.answerIndex];
}

test('calibration has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('calibration');

  assert.equal(quiz.length, 100);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'reliability-gap',
    'binning-and-ece-audit',
    'held-out-calibration-plan',
  ]);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^cal-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use an ordered cal id`);
    assert.equal(question.id.slice(4, 7), String(index + 1).padStart(3, '0'), `question ${index + 1} id should match its position`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, 3, `question ${index + 1} choices should be distinct`);
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('calibration assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('calibration');
  const prompts = new Map();
  const correctAnswers = new Map();

  for (const [index, question] of quiz.entries()) {
    const questionNumber = index + 1;
    const prompt = normalized(question.prompt);
    const answer = normalized(correctAnswer(question));

    assert.ok(!prompts.has(prompt), `questions ${prompts.get(prompt)} and ${questionNumber} repeat the same prompt`);
    assert.ok(!correctAnswers.has(answer), `questions ${correctAnswers.get(answer)} and ${questionNumber} repeat the same correct answer`);

    prompts.set(prompt, questionNumber);
    correctAnswers.set(answer, questionNumber);
  }
});

test('calibration assessment progresses from reliability meaning to interview readiness', () => {
  const { quiz } = getLessonAssessment('calibration');
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
      `${level} questions should occupy positions ${start + 1}-${end}`,
    );
  }

  for (let index = 1; index < quiz.length; index += 1) {
    assert.ok(
      LEVEL_ORDER[quiz[index].level] >= LEVEL_ORDER[quiz[index - 1].level],
      `question ${index + 1} should not regress in difficulty`,
    );
  }
});

test('calibration assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('calibration');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const orderedMilestones = [
    ['reliability meaning', ['observed outcome frequency', 'predicted probability']],
    ['confidence distinction', ['confident', 'stated probabilities are wrong']],
    ['diagram', ['reliability diagram']],
    ['gap', ['difference between predicted probability and observed outcome frequency']],
    ['ECE', ['weighted average of bucket reliability gaps']],
    ['Platt scaling', ['sigmoid mapping']],
    ['isotonic regression', ['flexible monotonic mapping']],
    ['final evaluation discipline', ['after model and calibrator choices are frozen']],
    ['subgroup checks', ['overall reliability can hide cohort specific']],
    ['deployment monitoring', ['fresh labeled outcomes']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['production ready calibration workflow']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of orderedMilestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('calibration assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('calibration');
  const unsafePatterns = [
    /guarantees calibrated probabilities/i,
    /automatically calibrated/i,
    /fitting the calibrator on the final test labels/i,
    /more bins always give a more trustworthy calibration estimate/i,
    /independent of binning choices/i,
    /cannot overfit when calibration data are small/i,
    /fixes every class-specific calibration problem automatically/i,
    /always changes the ranking order/i,
    /guaranteed forever/i,
    /calibration alone chooses the best threshold/i,
    /guarantees this individual outcome/i,
    /trust any softmax score/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|claim/i.test(question.prompt);

    assert.ok(
      !unsafeAnswer || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('calibration assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('calibration');
  const expectedTrapIds = [
    'cal-076-trap-accuracy-check',
    'cal-077-trap-auc-a',
    'cal-078-trap-sigmoid-b',
    'cal-079-trap-test-set-claim',
    'cal-080-trap-bins',
    'cal-081-trap-ece',
    'cal-082-trap-isotonic',
    'cal-083-trap-temperature',
    'cal-084-trap-ranking-change',
    'cal-085-trap-constant',
    'cal-086-trap-subgroups',
    'cal-087-trap-shift',
    'cal-088-trap-threshold',
    'cal-089-trap-individual',
    'cal-090-tricky-summary',
  ];
  const misconceptionPatterns = [
    /guarantees calibrated probabilities/i,
    /proves the numerical probabilities are reliable/i,
    /automatically calibrated/i,
    /fitting the calibrator on the final test labels/i,
    /more bins always give a more trustworthy calibration estimate/i,
    /independent of binning choices/i,
    /cannot overfit when calibration data are small/i,
    /fixes every class-specific calibration problem automatically/i,
    /always changes the ranking order/i,
    /useful for ranking individuals/i,
    /guarantees every subgroup is calibrated/i,
    /guaranteed forever/i,
    /calibration alone chooses the best threshold/i,
    /guarantees this individual outcome/i,
  ];
  const trapPrompt = /false|unsafe|wrong|trap|claim/i;

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), expectedTrapIds);

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(answer));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('calibration assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('calibration');
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

test('calibration assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('calibration');
  const totals = [0, 0, 0];

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);
    positions.forEach((position) => {
      totals[position] += 1;
    });

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }

  assert.ok(
    Math.max(...totals) - Math.min(...totals) <= 1,
    `correct answers should be balanced across positions, got ${totals.join(', ')}`,
  );
});
