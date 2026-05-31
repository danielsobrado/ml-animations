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

test('rope has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('rope');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'relative-shift-check',
    'trace-frequency-schedule',
    'debug-implementation-alignment',
  ]);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^rope-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the ordered rope id format`);
    assert.equal(Number(question.id.slice(5, 8)), index + 1, `question ${index + 1} should have a sequential id`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map((choice) => normalized(choice))).size, 3, `question ${index + 1} should have distinct choices`);
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('rope assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('rope');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('rope assessment progresses from rotary basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('rope');
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

test('rope assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('rope');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['relative position information through rotations']],
    ['qk rotation', ['query and key vectors']],
    ['not values or masks', ['thinking rope rotates values or replaces masks']],
    ['formula', ['q m t k n']],
    ['dimension pairs', ['2d rotations']],
    ['cache alignment', ['cached keys', 'original token slots']],
    ['application cache bug', ['cached rotary positions no longer match']],
    ['production tests', ['reference score tests']],
    ['tricky false claims', ['rope value vector claim is false']],
    ['interview readiness', ['production ready rope takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('rope assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('rope');
  const unsafePatterns = [
    /rotates value vectors instead of query and key vectors/i,
    /replaces the causal mask/i,
    /deletes token embeddings and uses only positions/i,
    /every position use the same rotation angle/i,
    /relative distance alone determines every attention score/i,
    /applied after softmax/i,
    /cache position offsets unnecessary/i,
    /base can be changed freely with no quality risk/i,
    /guarantees perfect extrapolation to any sequence length/i,
    /broadcast axes do not matter/i,
    /convention can be ignored/i,
    /always works the same with full-dimension rope/i,
    /padding tokens become safe automatically/i,
    /changes vector length to encode position/i,
    /value rotation, mask replacement, and guaranteed unlimited context/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('rope misconception traps are placed after concept scaffolding', () => {
  const { quiz } = getLessonAssessment('rope');
  const trapIds = [
    'rope-076-false-values',
    'rope-077-false-mask',
    'rope-078-false-delete-embeddings',
    'rope-079-false-same-angle',
    'rope-080-false-distance-only',
    'rope-081-false-softmax',
    'rope-082-false-cache',
    'rope-083-false-base',
    'rope-084-false-long-context',
    'rope-085-false-shapes',
    'rope-086-false-sign',
    'rope-087-false-partial',
    'rope-088-false-padding',
    'rope-089-false-orthogonal',
    'rope-090-tricky-summary',
  ];

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

  for (const question of quiz.slice(0, 75)) {
    assert.doesNotMatch(question.prompt, /^Which .* claim is false\?/);
  }
});

test('rope assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('rope');

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

test('rope assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('rope');
  const allPositions = quiz.map((question) => question.answerIndex);
  const globalCounts = [0, 1, 2].map((slot) => allPositions.filter((position) => position === slot).length);

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }

  assert.ok(Math.max(...globalCounts) - Math.min(...globalCounts) <= 1, `global answer positions should stay balanced: ${globalCounts.join(', ')}`);
});
