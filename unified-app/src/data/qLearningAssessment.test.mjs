import assert from 'node:assert/strict';
import test from 'node:test';
import { Q_LEARNING_QUIZ } from './qLearningAssessment.js';

const LEVELS = new Set(['Foundation', 'Mechanism', 'Application', 'Tricky', 'Interview']);

function normalize(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

test('q-learning assessment has 100 production-ready questions', () => {
  assert.equal(Q_LEARNING_QUIZ.length, 100);

  const ids = new Set();
  for (const [index, item] of Q_LEARNING_QUIZ.entries()) {
    assert.match(item.id, /^qlearn-\d{3}$/);
    assert.equal(ids.has(item.id), false, `duplicate id ${item.id}`);
    ids.add(item.id);
    assert.equal(item.id, `qlearn-${String(index + 1).padStart(3, '0')}`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.equal(Number.isInteger(item.answerIndex), true, `${item.id} answerIndex must be integer`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
  }
});

test('q-learning assessment progresses from definitions to production reasoning', () => {
  const ranges = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of ranges) {
    const slice = Q_LEARNING_QUIZ.slice(start, end);
    assert.equal(slice.every((item) => item.level === level), true, `${level} range mismatch`);
  }

  const milestones = [
    ['q-value meaning', ['q-value', 'expected discounted return']],
    ['bellman arithmetic', ['target', 'TD error', 'new Q-value']],
    ['off-policy backup', ['off-policy', 'greedy next action']],
    ['training loop', ['reward chart', 'negative spikes']],
    ['applied tuning', ['alpha versus gamma', 'tuning']],
    ['production review', ['validate incentives', 'coverage', 'deployed behavior']],
  ];

  let previous = -1;
  for (const [name, terms] of milestones) {
    const index = Q_LEARNING_QUIZ.findIndex((item) => (
      terms.every((term) => normalize(`${item.prompt} ${item.choices.join(' ')} ${item.explanation}`).includes(normalize(term)))
    ));
    assert.notEqual(index, -1, `missing milestone: ${name}`);
    assert.ok(index > previous, `${name} appears out of order`);
    previous = index;
  }
});

test('q-learning assessment traps common misconceptions only after setup', () => {
  const misconceptionTerms = [
    /It is only the immediate reward/i,
    /must know the full transition model/i,
    /action actually sampled/i,
    /probability that a random action/i,
    /terminal transition always/i,
    /one successful episode guarantees/i,
    /continuous real-world state space/i,
  ];

  const trapPrompt = /false|wrong|trap|unsafe|misconception|reject|dangerous|risky/i;

  for (const [index, item] of Q_LEARNING_QUIZ.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('q-learning assessment avoids visible-page answer leakage', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < Q_LEARNING_QUIZ.length; pageStart += pageSize) {
    const page = Q_LEARNING_QUIZ.slice(pageStart, pageStart + pageSize);
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

test('q-learning assessment distributes correct answer positions per page', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < Q_LEARNING_QUIZ.length; pageStart += pageSize) {
    const page = Q_LEARNING_QUIZ.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) {
      counts[item.answerIndex] += 1;
    }
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 2, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
