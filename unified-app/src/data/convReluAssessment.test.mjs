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
    .replace(/\s+/g, ' ')
    .trim();
}

function correctAnswer(question) {
  return question.choices[question.answerIndex];
}

test('conv relu has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('conv-relu');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.ok(question.id.startsWith('cr-'), `question ${index + 1} should use the cr id prefix`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('conv relu assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('conv-relu');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('conv relu assessment progresses from definitions to interview readiness', () => {
  const { quiz } = getLessonAssessment('conv-relu');
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

test('conv relu assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('conv-relu');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['signed local convolution responses']],
    ['conv prerequisite', ['signed pre-activation']],
    ['relu prerequisite', ['max(0, z)']],
    ['order', ['convolve with kernel and bias']],
    ['negative response', ['becomes zero for the next layer']],
    ['bias', ['shifts which convolution responses cross zero']],
    ['backward mask', ['locations where pre-activations were positive']],
    ['dead filter', ['negative pre-activations almost everywhere']],
    ['application debugging', ['signed pre-activation map and bias shift']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['production-ready conv + relu takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('conv relu assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('conv-relu');
  const unsafePatterns = [
    /removes negative values before the convolution/i,
    /relu creates the learned local pattern detector/i,
    /flips negative convolution responses/i,
    /changes the spatial size/i,
    /clipped negative locations pass the same gradient/i,
    /bias cannot affect which relu gates open/i,
    /any zero after relu proves the filter is useless/i,
    /outputs calibrated probabilities/i,
    /preserves both positive and negative evidence signs/i,
    /reconstruct all negative pre-activation values/i,
    /no possible effect on relu activations/i,
    /stride changes which relu threshold is used/i,
    /making every response positive is always ideal/i,
    /only post-relu maps are needed/i,
    /order, sign, and masks cannot cause bugs/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|claim/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('conv relu assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('conv-relu');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [answerIndex, answer] of answers.entries()) {
      for (const [promptIndex, question] of page.entries()) {
        if (answerIndex === promptIndex || answer.length < 8) continue;
        assert.ok(!normalized(question.prompt).includes(answer));
      }
    }
  }
});

test('conv relu assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('conv-relu');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }
});
