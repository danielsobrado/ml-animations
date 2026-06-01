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

const EXPECTED_LEVEL_COUNTS = {
  Foundation: 20,
  Mechanism: 30,
  Application: 25,
  Tricky: 15,
  Interview: 10,
};

function normalized(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim()
    .replace(/\s+/g, ' ');
}

function correctAnswer(question) {
  return question.choices[question.answerIndex];
}

test('cross entropy has a complete curated 100-question assessment', () => {
  const { labs, quiz } = getLessonAssessment('cross-entropy');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);
  assert.deepEqual(labs.map((lab) => lab.id), ['match-true-class-probability']);
  assert.deepEqual(
    quiz.reduce((counts, question) => {
      counts[question.level] = (counts[question.level] || 0) + 1;
      return counts;
    }, {}),
    EXPECTED_LEVEL_COUNTS,
  );

  for (const [index, question] of quiz.entries()) {
    const questionNumber = String(index + 1).padStart(3, '0');
    assert.match(question.id, /^ce-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use a stable ordered id`);
    assert.ok(question.id.startsWith(`ce-${questionNumber}-`), `question ${index + 1} id should match its position`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.equal(new Set(question.choices.map((choice) => normalized(choice))).size, 3, `question ${index + 1} choices should be distinct`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('cross entropy assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('cross-entropy');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('cross entropy assessment progresses from loss basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('cross-entropy');
  const expectedBands = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of expectedBands) {
    assert.deepEqual([...new Set(quiz.slice(start, end).map((question) => question.level))], [level]);
  }

  for (let index = 1; index < quiz.length; index += 1) {
    assert.ok(LEVEL_ORDER[quiz[index].level] >= LEVEL_ORDER[quiz[index - 1].level]);
  }
});

test('cross entropy assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('cross-entropy');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['classification loss', ['classification loss']],
    ['predicted distribution', ['predicted distribution']],
    ['one hot target', ['one hot target']],
    ['true class probability', ['true class probability']],
    ['negative log', ['negative log']],
    ['softmax prerequisite', ['softmax']],
    ['bits unit', ['bits']],
    ['formula', ['h p q']],
    ['one hot reduction', ['one hot reduction']],
    ['wrong classes', ['wrong class']],
    ['log base', ['preserving the same optimum']],
    ['kl divergence', ['measures extra coding cost']],
    ['label smoothing', ['label smoothing']],
    ['calibration', ['calibration']],
    ['tricky false claims', ['tricky false claims']],
    ['interview readiness', ['interview readiness']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('cross entropy assessment uses misconception traps only after setup', () => {
  const { quiz } = getLessonAssessment('cross-entropy');
  const misconceptionPatterns = [
    /same as accuracy/i,
    /only checks the top predicted label/i,
    /wrong classes never matter/i,
    /kl divergence is symmetric/i,
    /log base changes the best model/i,
    /clipping makes the model correct/i,
    /low cross-entropy guarantees calibration/i,
    /tuning hyperparameters on test/i,
    /one low-loss batch.*prove generalization/i,
    /uniform prediction.*automatically good/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const misconceptionAnswer = misconceptionPatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|shortcut/i.test(question.prompt);
    assert.ok(!misconceptionAnswer || index >= 75, `question ${index + 1} keys a misconception before the tricky band`);
    assert.ok(!misconceptionAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('cross entropy assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('cross-entropy');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [answerIndex, answer] of answers.entries()) {
      for (const [promptIndex, question] of page.entries()) {
        if (answerIndex === promptIndex || answer.length < 8) continue;
        assert.ok(!normalized([question.prompt, ...question.choices].join(' ')).includes(answer));
      }
    }
  }
});

test('cross entropy assessment avoids visible lesson scope leakage', () => {
  const { quiz } = getLessonAssessment('cross-entropy');
  const outOfScopePatterns = [
    /sliders are visible/i,
    /visual distance/i,
    /alphabetical position/i,
    /squared distance between class names/i,
    /training images/i,
    /computer memory bytes/i,
    /measured in pixels/i,
    /display color/i,
    /sorted alphabetically/i,
    /longest name/i,
    /chart color/i,
    /hidden by the browser/i,
    /chart decoration/i,
    /only for cats/i,
    /only works with images/i,
    /chart changes/i,
    /\bui\b/i,
    /hidden layers/i,
    /class name string/i,
    /label color/i,
    /route names/i,
    /batch size can become/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const visibleText = [question.prompt, ...question.choices].join(' ');
    for (const pattern of outOfScopePatterns) {
      assert.ok(!pattern.test(visibleText), `question ${index + 1} has out-of-scope visible text: ${pattern}`);
    }
  }
});

test('cross entropy assessment distributes correct-answer positions globally and across every page', () => {
  const { quiz } = getLessonAssessment('cross-entropy');
  const globalCounts = [0, 1, 2].map((slot) => quiz.filter((question) => question.answerIndex === slot).length);

  assert.ok(Math.max(...globalCounts) - Math.min(...globalCounts) <= 1, `global answer positions should be balanced: ${globalCounts.join(', ')}`);

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const pageCounts = [0, 1, 2].map((slot) => page.filter((question) => question.answerIndex === slot).length);

    assert.ok(Math.max(...pageCounts) - Math.min(...pageCounts) <= 1, `page starting at question ${pageStart + 1} should balance answer positions: ${pageCounts.join(', ')}`);
  }
});
