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

test('lstm has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('lstm');

  assert.equal(quiz.length, 100);
  assert.deepEqual(
    labs.map((lab) => lab.id),
    ['trace-gated-memory'],
  );
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^lstm-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the ordered lstm id format`);
    assert.equal(Number(question.id.slice(5, 8)), index + 1, `question ${index + 1} should keep ordered ids`);
    assert.ok(!question.id.startsWith('generated-'), `question ${index + 1} should not use a generated id`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, 3, `question ${index + 1} choices should be distinct`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }

  const positionCounts = [0, 1, 2].map((answerIndex) => (
    quiz.filter((question) => question.answerIndex === answerIndex).length
  ));
  assert.ok(Math.max(...positionCounts) - Math.min(...positionCounts) <= 3, `answer positions should be balanced: ${positionCounts.join(', ')}`);
});

test('lstm assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('lstm');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('lstm assessment progresses from memory basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('lstm');
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

test('lstm assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('lstm');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['sequence patterns']],
    ['cell state', ['memory vector']],
    ['hidden state', ['current output']],
    ['gates', ['forgotten written and exposed']],
    ['forget gate', ['scales old memory']],
    ['input gate', ['new content entering']],
    ['output gate', ['visible recurrent signal']],
    ['unfolding', ['states move forward']],
    ['gate equations', ['current input and previous hidden state']],
    ['cell update', ['gated old memory']],
    ['bptt', ['backpropagation through time']],
    ['masking', ['padded positions']],
    ['state reset', ['unrelated sequences']],
    ['application review', ['sequence lengths masks state resets']],
    ['unsafe trap', ['claim is unsafe']],
    ['interview readiness', ['masking reset and causality risks']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('lstm assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('lstm');
  const unsafePatterns = [
    /remembers all earlier inputs perfectly/i,
    /cell state and hidden state are always identical/i,
    /chooses which training examples to delete/i,
    /writes every candidate value completely/i,
    /changes the old cell state before the forget gate/i,
    /produce unrestricted values/i,
    /eliminate vanishing and exploding gradients completely/i,
    /do not need backpropagation through time/i,
    /padding tokens can always be treated as real/i,
    /always carry across unrelated sequences/i,
    /always valid for causal generation/i,
    /always the true last token/i,
    /make lstms useless/i,
    /alone prove the exact human reason/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|risky|absolute|too absolute/i.test(question.prompt);

    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('lstm assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('lstm');
  const expectedTrapIds = [
    'lstm-076-trap-perfect-memory',
    'lstm-077-trap-cell-hidden',
    'lstm-078-trap-forget',
    'lstm-079-trap-input',
    'lstm-080-trap-output',
    'lstm-081-trap-sigmoid',
    'lstm-082-trap-gradient',
    'lstm-083-trap-bptt',
    'lstm-084-trap-padding',
    'lstm-085-trap-stateful',
    'lstm-086-trap-bidirectional',
    'lstm-087-trap-final-state',
    'lstm-088-trap-transformer',
    'lstm-089-trap-interpretability',
    'lstm-090-tricky-summary',
  ];
  const trapQuestions = quiz.slice(75, 90);

  assert.deepEqual(
    trapQuestions.map((question) => question.id),
    expectedTrapIds,
  );
  assert.ok(trapQuestions.every((question) => question.level === 'Tricky'));
  assert.ok(trapQuestions.every((question) => /trap|tricky/i.test(question.id)));
});

test('lstm assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('lstm');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
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

test('lstm assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('lstm');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }
});
