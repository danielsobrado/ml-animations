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

test('conv2d has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('conv2d');

  assert.equal(quiz.length, 100);
  assert.deepEqual(
    labs.map((lab) => lab.id),
    ['trace-output-size', 'trace-selected-cell', 'compare-kernel-responses'],
  );
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^c2d-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the ordered c2d id format`);
    assert.equal(Number(question.id.slice(4, 7)), index + 1, `question ${index + 1} should keep ordered ids`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, 3, `question ${index + 1} choices should be distinct`);
    assert.equal(Number.isInteger(question.answerIndex), true, `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }

  const positionCounts = [0, 1, 2].map((answerIndex) => (
    quiz.filter((question) => question.answerIndex === answerIndex).length
  ));
  assert.ok(Math.max(...positionCounts) - Math.min(...positionCounts) <= 1, `answer positions should be balanced: ${positionCounts.join(', ')}`);
});

test('conv2d assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('conv2d');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('conv2d assessment progresses from definitions to interview readiness', () => {
  const { quiz } = getLessonAssessment('conv2d');
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

test('conv2d assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('conv2d');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['detecting local spatial patterns']],
    ['kernel', ['small grid of learned weights']],
    ['output cell', ['dot product between the kernel']],
    ['stride', ['how far the kernel moves']],
    ['padding', ['extra border values']],
    ['channels', ['separate value planes']],
    ['output shape formula', ['floor input 2 padding kernel stride 1']],
    ['multi-channel sum', ['sum the aligned products across height width and input channels']],
    ['parameter count', ['f times c times k times k']],
    ['application shape debug', ['padding was omitted']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['production ready conv2d takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('conv2d assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('conv2d');
  const unsafePatterns = [
    /separate learned kernel for each output coordinate/i,
    /averages every pixel in the whole image/i,
    /increasing stride always increases output spatial size/i,
    /padding removes the need for learned kernel weights/i,
    /ignores all but the first channel/i,
    /more filters increase spatial height/i,
    /always probabilities between zero and one/i,
    /known without kernel, stride, padding, or dilation/i,
    /separate for every spatial coordinate/i,
    /nchw and nhwc can be swapped without changing interpretation/i,
    /labels are the only setting worth checking/i,
    /no structural assumptions/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|claim/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('conv2d assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('conv2d');
  const expectedTrapIds = [
    'c2d-076-false-location-trap',
    'c2d-077-false-global-trap',
    'c2d-078-false-stride-trap',
    'c2d-079-false-padding-trap',
    'c2d-080-false-channel-trap',
    'c2d-081-false-filter-trap',
    'c2d-082-false-probability-trap',
    'c2d-083-false-shape-trap',
    'c2d-084-false-border-trap',
    'c2d-085-false-bias-trap',
    'c2d-086-false-layout-trap',
    'c2d-087-false-flip-trap',
    'c2d-088-false-dilation-trap',
    'c2d-089-false-debug-trap',
    'c2d-090-tricky-summary',
  ];
  const misconceptionPatterns = [
    /separate learned kernel for each output coordinate/i,
    /averages every pixel in the whole image/i,
    /increasing stride always increases output spatial size/i,
    /padding removes the need for learned kernel weights/i,
    /ignores all but the first channel/i,
    /more filters increase spatial height/i,
    /always probabilities between zero and one/i,
    /known without kernel, stride, padding, or dilation/i,
    /separate for every spatial coordinate/i,
    /nchw and nhwc can be swapped without changing interpretation/i,
    /every deep-learning conv2d implementation flips kernels/i,
    /dilation has no effect on the effective kernel footprint/i,
    /labels are the only setting worth checking/i,
    /no structural assumptions/i,
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

test('conv2d assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('conv2d');

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

test('conv2d assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('conv2d');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }
});
