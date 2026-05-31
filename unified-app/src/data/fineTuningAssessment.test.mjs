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

test('fine tuning has a complete curated 100-question assessment with focused labs', () => {
  const { quiz, labs } = getLessonAssessment('fine-tuning');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'choose-finetune-method',
    'inspect-trainable-scope',
    'debug-tuning-failure',
  ]);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^ftune-\d{3}-[a-z0-9-]+$/);
    assert.equal(Number(question.id.slice(6, 9)), index + 1);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, 3);
    assert.ok(Number.isInteger(question.answerIndex));
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }

  const allPositions = quiz.map((question) => question.answerIndex);
  const globalCounts = [0, 1, 2].map((slot) => allPositions.filter((position) => position === slot).length);
  assert.ok(
    Math.max(...globalCounts) - Math.min(...globalCounts) <= 1,
    `answer positions should be globally balanced, got ${globalCounts.join(', ')}`,
  );
});

test('fine tuning assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('fine-tuning');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('fine tuning assessment progresses from method basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('fine-tuning');
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

test('fine tuning assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('fine-tuning');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['changing pretrained model behavior using a smaller task specific signal']],
    ['method comparison', ['full fine tuning lora qlora sft']],
    ['lora', ['small low rank adapter matrices']],
    ['preference data', ['prompt with a chosen answer and a rejected answer']],
    ['not retrieval', ['fine tuning is not retrieval']],
    ['mechanism summary', ['choose trainable parameters data format loss scope']],
    ['application choice', ['limited gpu memory', 'style adaptation']],
    ['production method', ['memory budget data shape desired behavior']],
    ['tricky false claims', ['fine tuning claim is false']],
    ['interview readiness', ['production ready fine tuning takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('fine tuning assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('fine-tuning');
  const unsafePatterns = [
    /one fixed method with one data format/i,
    /same thing as retrieval/i,
    /only the tokenizer vocabulary/i,
    /requires updating every base weight/i,
    /no reference responses or demonstrations/i,
    /unrelated to the same prompt/i,
    /guarantees better production behavior/i,
    /removes the need to evaluate/i,
    /cannot affect a fine-tuned model/i,
    /always cheaper than adapter tuning/i,
    /automatically gives the model every missing private fact/i,
    /always proves production behavior improved/i,
    /proves the model is safe in every deployment/i,
    /cannot overfit because the base model is pretrained/i,
    /acronym without checking data signal/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('fine tuning misconception traps are placed after concept scaffolding', () => {
  const { quiz } = getLessonAssessment('fine-tuning');
  const trapIds = [
    'ftune-076-false-single',
    'ftune-077-false-retrieval',
    'ftune-078-false-lora',
    'ftune-079-false-qlora',
    'ftune-080-false-sft',
    'ftune-081-false-dpo',
    'ftune-082-false-rank',
    'ftune-083-false-quant',
    'ftune-084-false-data',
    'ftune-085-false-memory',
    'ftune-086-false-facts',
    'ftune-087-false-eval',
    'ftune-088-false-safety',
    'ftune-089-false-overfit',
    'ftune-090-tricky-summary',
  ];

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

  for (const question of quiz.slice(0, 75)) {
    assert.doesNotMatch(question.prompt, /^Which .* (false|wrong|unsafe|reject|trap|misconception)/i);
  }
});

test('fine tuning assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('fine-tuning');

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

test('fine tuning assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('fine-tuning');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }
});
