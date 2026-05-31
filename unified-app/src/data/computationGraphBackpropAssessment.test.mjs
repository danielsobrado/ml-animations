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

test('computation graph backprop has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('computation-graph-backprop');

  assert.equal(quiz.length, 100);
  assert.deepEqual(
    labs.map((lab) => lab.id),
    ['trace-forward-backward-chain', 'debug-relu-blocked-gradient', 'predict-update'],
  );
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^cgb-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the cgb id format`);
    assert.equal(Number(question.id.slice(4, 7)), index + 1, `question ${index + 1} id number should match its position`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map((choice) => normalized(choice))).size, 3, `question ${index + 1} should have distinct choices`);
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('computation graph backprop assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('computation-graph-backprop');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('computation graph backprop assessment progresses from graph basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('computation-graph-backprop');
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

test('computation graph backprop assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('computation-graph-backprop');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['assigns loss blame']],
    ['graph', ['network of operations']],
    ['local derivative', ['sensitivity of one operation']],
    ['chain rule', ['chain rule applied']],
    ['relu prerequisite', ['one above zero and zero below zero']],
    ['seed gradient', ['dl dl equals 1']],
    ['gradient route', ['multiply dl da by da dz']],
    ['debugging application', ['gradient sign and update direction']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['production ready backprop takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('computation graph backprop assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('computation-graph-backprop');
  const unsafePatterns = [
    /magic unrelated/i,
    /without forward values/i,
    /pass gradients unchanged/i,
    /only one branch can contribute/i,
    /positive gradient direction/i,
    /always proves the model is optimal/i,
    /guarantees the next loss decreases/i,
    /updates the target y/i,
    /changes the forward loss before any update/i,
    /parameters do not/i,
    /random gradient at the loss/i,
    /d\(wx\)\/dw equals w/i,
    /dz\/db equals b/i,
    /only the final loss/i,
    /can never learn on any example/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('computation graph backprop assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('computation-graph-backprop');
  const expectedTrapIds = [
    'cgb-076-false-magic-trap',
    'cgb-077-false-forward-trap',
    'cgb-078-false-relu-trap',
    'cgb-079-false-add-trap',
    'cgb-080-false-update-trap',
    'cgb-081-false-zero-trap',
    'cgb-082-false-lr-trap',
    'cgb-083-false-target-trap',
    'cgb-084-false-slider-trap',
    'cgb-085-false-param-trap',
    'cgb-086-false-loss-trap',
    'cgb-087-false-multiply-trap',
    'cgb-088-false-bias-trap',
    'cgb-089-false-debug-trap',
    'cgb-090-tricky-summary',
  ];
  const misconceptionPatterns = [
    /magic unrelated/i,
    /without forward values/i,
    /pass gradients unchanged/i,
    /only one branch can contribute/i,
    /positive gradient direction/i,
    /always proves the model is optimal/i,
    /guarantees the next loss decreases/i,
    /updates the target y/i,
    /changes the forward loss before any update/i,
    /parameters do not/i,
    /random gradient at the loss/i,
    /d\(wx\)\/dw equals w/i,
    /dz\/db equals b/i,
    /only the final loss/i,
    /can never learn on any example/i,
  ];
  const trapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i;
  const trapQuestions = quiz.slice(75, 90);

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), expectedTrapIds);
  assert.ok(trapQuestions.every((question) => question.level === 'Tricky'), 'misconception traps should stay in the Tricky band');

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(answer));
    if (!containsMisconception) continue;
    if (index < 75) {
      assert.match(question.prompt, /misconception.*avoid/i, `${question.id} should scaffold any early misconception`);
      continue;
    }
    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('computation graph backprop assessment stays within visible lesson scope', () => {
  const { quiz } = getLessonAssessment('computation-graph-backprop');
  const lessonScopeLeaks = [
    /vector jacobian/i,
    /full jacobian/i,
    /\bleaf\b/i,
    /\.grad/i,
    /zero_grad/i,
    /detach/i,
    /requires_grad/i,
    /autodiff/i,
    /finite difference/i,
    /custom backward/i,
    /retain.*graph/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const visibleText = `${question.prompt} ${question.choices.join(' ')} ${question.explanation}`;
    assert.ok(!lessonScopeLeaks.some((pattern) => pattern.test(visibleText)), `question ${index + 1} leaks framework or future debugging scope`);
  }
});

test('computation graph backprop assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('computation-graph-backprop');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [answerIndex, answer] of answers.entries()) {
      for (const [promptIndex, question] of page.entries()) {
        if (answerIndex === promptIndex || answer.length < 8) continue;
        const visibleQuestionText = normalized(`${question.prompt} ${question.choices.join(' ')}`);
        assert.ok(!visibleQuestionText.includes(answer));
      }
    }
  }
});

test('computation graph backprop assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('computation-graph-backprop');
  const globalPositionCounts = [0, 1, 2].map((slot) => quiz.filter((question) => question.answerIndex === slot).length);

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }

  assert.ok(
    Math.max(...globalPositionCounts) - Math.min(...globalPositionCounts) <= 1,
    `global answer positions should be balanced, got ${globalPositionCounts.join(', ')}`,
  );
});
