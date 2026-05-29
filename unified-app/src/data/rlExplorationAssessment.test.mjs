import assert from 'node:assert/strict';
import test from 'node:test';
import { RL_EXPLORATION_QUIZ } from './rlExplorationAssessment.js';

const LEVELS = new Set(['Foundation', 'Mechanism', 'Application', 'Tricky', 'Interview']);

function normalize(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

test('rl exploration assessment has 100 production-ready questions', () => {
  assert.equal(RL_EXPLORATION_QUIZ.length, 100);
  const ids = new Set();

  for (const [index, item] of RL_EXPLORATION_QUIZ.entries()) {
    assert.match(item.id, /^rlexplore-\d{3}$/);
    assert.equal(ids.has(item.id), false, `duplicate id ${item.id}`);
    ids.add(item.id);
    assert.equal(item.id, `rlexplore-${String(index + 1).padStart(3, '0')}`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
  }
});

test('rl exploration assessment progresses from epsilon basics to readiness reasoning', () => {
  const ranges = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of ranges) {
    assert.equal(RL_EXPLORATION_QUIZ.slice(start, end).every((item) => item.level === level), true, `${level} range mismatch`);
  }

  const milestones = [
    ['epsilon meaning', ['epsilon', 'random exploratory action']],
    ['counts', ['explore and exploit counts']],
    ['cliff risk', ['cliff', 'safety risk']],
    ['alpha gamma tuning', ['alpha', 'gamma', 'learning curves']],
    ['deployment safety', ['deployment', 'noise', 'unsafe']],
    ['production readiness', ['learning value', 'operational risk']],
  ];

  let previous = -1;
  for (const [name, terms] of milestones) {
    const index = RL_EXPLORATION_QUIZ.findIndex((item) => (
      terms.every((term) => normalize(`${item.prompt} ${item.choices.join(' ')} ${item.explanation}`).includes(normalize(term)))
    ));
    assert.notEqual(index, -1, `missing milestone: ${name}`);
    assert.ok(index > previous, `${name} appears out of order`);
    previous = index;
  }
});

test('rl exploration assessment marks unsafe misconceptions as traps after setup', () => {
  const misconceptionTerms = [
    /always finds the best action/i,
    /always the best deployed policy/i,
    /shortest path is always best/i,
    /guarantees no random failures/i,
    /falls can be ignored/i,
    /automatically proves/i,
    /good-looking reward curve alone proves/i,
  ];
  const trapPrompt = /false|wrong|unsafe|trap|reject|dangerous|misconception/i;

  for (const [index, item] of RL_EXPLORATION_QUIZ.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('rl exploration assessment avoids visible-page answer leakage', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < RL_EXPLORATION_QUIZ.length; pageStart += pageSize) {
    const page = RL_EXPLORATION_QUIZ.slice(pageStart, pageStart + pageSize);
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

test('rl exploration assessment distributes correct answer positions per page', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < RL_EXPLORATION_QUIZ.length; pageStart += pageSize) {
    const page = RL_EXPLORATION_QUIZ.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 2, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
