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
    .replace(/\s+/g, ' ')
    .trim();
}

function normalizedKey(value) {
  return normalized(value).replace(/[^a-z0-9]+/g, ' ').trim();
}

function correctAnswer(question) {
  return question.choices[question.answerIndex];
}

test('softmax has a complete curated 100-question assessment', () => {
  const { labs, quiz } = getLessonAssessment('softmax');

  assert.equal(quiz.length, 100);
  assert.deepEqual(labs.map((lab) => lab.id), ['temperature-logit-lab']);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);
  assert.deepEqual(
    quiz.reduce((counts, question) => {
      counts[question.level] = (counts[question.level] || 0) + 1;
      return counts;
    }, {}),
    EXPECTED_LEVEL_COUNTS,
  );

  for (const [index, question] of quiz.entries()) {
    const expectedId = `softmax-${String(index + 1).padStart(3, '0')}`;

    assert.equal(question.id, expectedId, `question ${index + 1} should use a stable ordered id`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map(normalizedKey)).size, 3, `question ${index + 1} choices should be distinct`);
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('softmax assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('softmax');
  const prompts = quiz.map((question) => normalizedKey(question.prompt));
  const correctAnswers = quiz.map((question) => normalizedKey(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(correctAnswers).size, correctAnswers.length);
});

test('softmax assessment progresses from logits to production judgment', () => {
  const { quiz } = getLessonAssessment('softmax');
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

test('softmax assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('softmax');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['logits to distribution', ['logits', 'probability distribution']],
    ['formula', ['p_i = exp', 'sum']],
    ['shared denominator', ['what is the denominator in softmax']],
    ['temperature', ['temperature', 'sharp']],
    ['translation invariance', ['adding the same constant', 'unchanged']],
    ['max subtraction stability', ['subtract', 'largest logit', 'overflow']],
    ['attention and masking', ['attention', 'mask']],
    ['application diagnostics', ['axis', 'temperature', 'stability']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['production-ready takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('softmax assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('softmax');
  const misconceptionPatterns = [
    /rescales each score independently/i,
    /sum to more than one/i,
    /zero logit always gives zero probability/i,
    /proves the model is calibrated/i,
    /changes the softmax distribution/i,
    /same invariance as adding a constant/i,
    /retrains the model weights/i,
    /changes the correct probabilities/i,
    /ideal when every label is independently true/i,
    /axis never matters/i,
    /negative probabilities/i,
    /cannot cause production bugs/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const misconceptionAnswer = misconceptionPatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|claim/i.test(question.prompt);
    assert.ok(!misconceptionAnswer || index >= 75, `question ${index + 1} keys a misconception before the tricky band`);
    assert.ok(!misconceptionAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('softmax assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('softmax');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [answerIndex, answer] of answers.entries()) {
      for (const [promptIndex, question] of page.entries()) {
        if (answerIndex === promptIndex) continue;
        assert.ok(!normalized([question.prompt, ...question.choices].join(' ')).includes(answer), `question ${pageStart + promptIndex + 1} leaks answer ${pageStart + answerIndex + 1}`);
      }
    }
  }
});

test('softmax assessment stays within visible lesson scope', () => {
  const { quiz } = getLessonAssessment('softmax');
  const outOfScopePatterns = [
    /\bui\b/i,
    /class labels are valid/i,
    /order of answer choices/i,
    /class names/i,
    /image file/i,
    /data splitting/i,
    /hidden layer/i,
    /dashboard/i,
    /route id/i,
    /exact color/i,
    /data cleaning/i,
    /manually edits/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const visibleText = [question.prompt, ...question.choices].join(' ');
    for (const pattern of outOfScopePatterns) {
      assert.ok(!pattern.test(visibleText), `question ${index + 1} has out-of-scope visible text: ${pattern}`);
    }
  }
});

test('softmax assessment distributes correct-answer positions globally and across every page', () => {
  const { quiz } = getLessonAssessment('softmax');
  const globalCounts = [0, 1, 2].map((slot) => quiz.filter((question) => question.answerIndex === slot).length);

  assert.ok(Math.max(...globalCounts) - Math.min(...globalCounts) <= 1, `global answer positions should be balanced: ${globalCounts.join(', ')}`);

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const pageCounts = [0, 1, 2].map((slot) => page.filter((question) => question.answerIndex === slot).length);

    assert.ok(Math.max(...pageCounts) - Math.min(...pageCounts) <= 1, `page starting at question ${pageStart + 1} should balance answer positions: ${pageCounts.join(', ')}`);
  }
});
