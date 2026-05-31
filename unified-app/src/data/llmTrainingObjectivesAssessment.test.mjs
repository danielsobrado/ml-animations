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

test('llm training objectives has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('llm-training-objectives');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.ok(question.id.startsWith('ltobj-'), `question ${index + 1} should use the ltobj id prefix`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
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
    ['purpose', ['turning raw text, demonstrations, and preferences into different learning signals']],
    ['stage map', ['pretraining teaches prediction, supervised fine-tuning teaches response format']],
    ['next token', ['predict the next token from preceding context']],
    ['preference signal', ['chosen responses against rejected responses']],
    ['alignment caveat', ['rather than simply adding factual knowledge']],
    ['loss mechanics', ['which tokens or comparisons contribute to the loss']],
    ['objective design', ['define context, target, loss scope, data quality, and evaluation']],
    ['application choice', ['match the objective to the missing behavior']],
    ['tricky false claims', ['objective claim is false']],
    ['interview readiness', ['production-ready llm-objective takeaway']],
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

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }
});
