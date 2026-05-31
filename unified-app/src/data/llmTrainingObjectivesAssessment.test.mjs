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

test('llm training objectives has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('llm-training-objectives');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'match-objective-stage',
    'inspect-loss-scope',
    'debug-objective-mismatch',
  ]);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^ltobj-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use a strict ltobj id`);
    assert.equal(Number(question.id.slice(6, 9)), index + 1, `question ${index + 1} id should be sequential`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, 3, `question ${index + 1} should have distinct choices`);
    assert.equal(Number.isInteger(question.answerIndex), true, `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('llm training objectives assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('llm-training-objectives');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('llm training objectives assessment progresses from basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('llm-training-objectives');
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

test('llm training objectives assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('llm-training-objectives');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['turning raw text demonstrations and preferences into different learning signals']],
    ['stage map', ['pretraining teaches prediction supervised fine tuning teaches response format']],
    ['next token', ['predict the next token from preceding context']],
    ['preference signal', ['chosen responses against rejected responses']],
    ['alignment caveat', ['rather than simply adding factual knowledge']],
    ['loss mechanics', ['which tokens or comparisons contribute to the loss']],
    ['objective design', ['define context target loss scope data quality and evaluation']],
    ['application choice', ['match the objective to the missing behavior']],
    ['tricky false claims', ['objective claim is false']],
    ['interview readiness', ['production ready llm objective takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('llm training objectives assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('llm-training-objectives');
  const unsafePatterns = [
    /automatically teaches prediction, instruction following, preferences, and all facts/i,
    /simply injects missing factual knowledge/i,
    /read future target tokens/i,
    /even when reference answers show the wrong behavior/i,
    /compared fairly across unrelated prompts/i,
    /padding tokens should usually be trained/i,
    /guarantees perfect factuality/i,
    /perfectly represents the complete user goal/i,
    /lower training loss always proves/i,
    /requires no preference pairs/i,
    /removes the need for a capable base model/i,
    /cannot affect the trained model/i,
    /prove the model is safe in every possible/i,
    /training on benchmark answers is harmless/i,
    /fashionable objective without checking target/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('llm training objectives misconception traps are placed after concept scaffolding', () => {
  const { quiz } = getLessonAssessment('llm-training-objectives');
  const trapIds = [
    'ltobj-076-false-one-objective',
    'ltobj-077-false-alignment-facts',
    'ltobj-078-false-future',
    'ltobj-079-false-sft',
    'ltobj-080-false-preference',
    'ltobj-081-false-padding',
    'ltobj-082-false-kl',
    'ltobj-083-false-reward',
    'ltobj-084-false-eval',
    'ltobj-085-false-dpo',
    'ltobj-086-false-instructions',
    'ltobj-087-false-data-quality',
    'ltobj-088-false-safety',
    'ltobj-089-false-contamination',
    'ltobj-090-tricky-summary',
  ];

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

  for (const question of quiz.slice(0, 75)) {
    assert.doesNotMatch(question.prompt, /^Which .* claim is false\?/);
  }
});

test('llm training objectives assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('llm-training-objectives');

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

test('llm training objectives assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('llm-training-objectives');
  const allPositions = quiz.map((question) => question.answerIndex);
  const globalCounts = [0, 1, 2].map((slot) => allPositions.filter((position) => position === slot).length);

  assert.ok(
    Math.max(...globalCounts) - Math.min(...globalCounts) <= 1,
    `global answer positions are imbalanced: ${globalCounts.join(', ')}`,
  );

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }
});
