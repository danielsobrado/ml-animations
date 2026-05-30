import assert from 'node:assert/strict';
import test from 'node:test';
import { TEST_TIME_COMPUTE_THINKING_BUDGETS_QUIZ } from './testTimeComputeThinkingBudgetsAssessment.js';

const LEVELS = new Set(['Foundation', 'Mechanism', 'Application', 'Tricky', 'Interview']);

function normalize(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

test('test-time compute thinking budgets assessment has 100 production-ready questions', () => {
  assert.equal(TEST_TIME_COMPUTE_THINKING_BUDGETS_QUIZ.length, 100);
  const ids = new Set();
  const prompts = new Set();
  const correctAnswers = new Set();

  for (const [index, item] of TEST_TIME_COMPUTE_THINKING_BUDGETS_QUIZ.entries()) {
    assert.match(item.id, /^ttc-\d{3}$/);
    assert.equal(ids.has(item.id), false, `duplicate id ${item.id}`);
    ids.add(item.id);
    assert.equal(item.id, `ttc-${String(index + 1).padStart(3, '0')}`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
    assert.equal(new Set(item.choices.map(normalize)).size, 3, `${item.id} has duplicate choices`);

    const normalizedPrompt = normalize(item.prompt);
    const normalizedCorrect = normalize(item.choices[item.answerIndex]);
    assert.equal(prompts.has(normalizedPrompt), false, `${item.id} repeats a prompt`);
    assert.equal(correctAnswers.has(normalizedCorrect), false, `${item.id} repeats a correct answer`);
    prompts.add(normalizedPrompt);
    correctAnswers.add(normalizedCorrect);
  }
});

test('test-time compute assessment progresses from scaling basics to production audit', () => {
  const ranges = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of ranges) {
    assert.equal(TEST_TIME_COMPUTE_THINKING_BUDGETS_QUIZ.slice(start, end).every((item) => item.level === level), true, `${level} range mismatch`);
  }

  const milestones = [
    ['scaling contrast', ['training-time scaling', 'inference-time scaling']],
    ['Best-of-N oracle', ['oracle bound', 'best sampled candidate']],
    ['beam PRM', ['PRM', 'pruning']],
    ['adaptive routing', ['difficulty classifier', 'token caps']],
    ['tool tradeoff', ['tool-call latency', 'permissions']],
    ['production audit', ['per-class accuracy', 'latency percentiles']],
  ];

  let previous = -1;
  for (const [name, terms] of milestones) {
    const index = TEST_TIME_COMPUTE_THINKING_BUDGETS_QUIZ.findIndex((item) => (
      terms.every((term) => normalize(`${item.prompt} ${item.choices.join(' ')} ${item.explanation}`).includes(normalize(term)))
    ));
    assert.notEqual(index, -1, `missing milestone: ${name}`);
    assert.ok(index > previous, `${name} appears out of order`);
    previous = index;
  }
});

test('test-time compute assessment marks unsafe misconceptions as traps after setup', () => {
  const misconceptionTerms = [
    /More thinking tokens always help/i,
    /Best-of-N is free/i,
    /oracle bound is always achieved/i,
    /can never prune the correct branch/i,
    /Adaptive routing always beats/i,
    /Tool calls make answers automatically true/i,
    /Stronger length penalties always improve/i,
    /High verifier score proves/i,
    /Average accuracy is enough/i,
    /One fixed cap is optimal/i,
    /Speculative decoding guarantees better/i,
    /Self-critique always converges/i,
    /inference cost is negligible/i,
    /removes context-window constraints/i,
    /single budget sweep on one benchmark proves/i,
  ];
  const trapPrompt = /false|wrong|unsafe|trap|reject|dangerous/i;

  for (const [index, item] of TEST_TIME_COMPUTE_THINKING_BUDGETS_QUIZ.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('test-time compute assessment avoids visible-page answer leakage', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < TEST_TIME_COMPUTE_THINKING_BUDGETS_QUIZ.length; pageStart += pageSize) {
    const page = TEST_TIME_COMPUTE_THINKING_BUDGETS_QUIZ.slice(pageStart, pageStart + pageSize);
    const correctAnswers = page.map((item) => normalize(item.choices[item.answerIndex]));

    for (const [offset, item] of page.entries()) {
      const surroundingPrompts = page
        .filter((_, otherOffset) => otherOffset !== offset)
        .map((other) => normalize(other.prompt));
      const leaked = surroundingPrompts.some((prompt) => prompt.includes(correctAnswers[offset]));
      assert.equal(leaked, false, `${item.id} answer appears in another prompt on same page`);
    }
  }
});

test('test-time compute assessment distributes correct answer positions per page', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < TEST_TIME_COMPUTE_THINKING_BUDGETS_QUIZ.length; pageStart += pageSize) {
    const page = TEST_TIME_COMPUTE_THINKING_BUDGETS_QUIZ.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
