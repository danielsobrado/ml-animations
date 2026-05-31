import assert from 'node:assert/strict';
import test from 'node:test';
import { BLOOM_FILTER_QUIZ } from './bloomFilterAssessment.js';

const LEVELS = new Set(['Foundation', 'Mechanism', 'Application', 'Tricky', 'Interview']);

function normalize(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

test('bloom filter assessment has 100 production-ready questions', () => {
  assert.equal(BLOOM_FILTER_QUIZ.length, 100);
  const ids = new Set();
  const prompts = new Set();
  const correctAnswers = new Set();

  for (const [index, item] of BLOOM_FILTER_QUIZ.entries()) {
    assert.match(item.id, /^bf-\d{3}$/);
    assert.equal(ids.has(item.id), false, `duplicate id ${item.id}`);
    ids.add(item.id);
    assert.equal(item.id, `bf-${String(index + 1).padStart(3, '0')}`);
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

test('bloom filter assessment progresses from membership basics to deployment judgment', () => {
  const ranges = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of ranges) {
    assert.equal(BLOOM_FILTER_QUIZ.slice(start, end).every((item) => item.level === level), true, `${level} range mismatch`);
  }

  const milestones = [
    ['compact membership premise', ['approximate set membership', 'less memory']],
    ['tuning formula', ['p is approximately', 'exp(-kn/m)']],
    ['insert and query mechanism', ['Insert sets k bits', 'query checks k bits']],
    ['safe application pattern', ['verify maybe-present results', 'monitor saturation']],
    ['trap setup', ['probably present proves membership', 'must not be treated as proof']],
    ['interview synthesis', ['one-sided prefilter', 'never standalone proof']],
  ];

  let previous = -1;
  for (const [name, terms] of milestones) {
    const index = BLOOM_FILTER_QUIZ.findIndex((item) => (
      terms.every((term) => normalize(`${item.prompt} ${item.choices.join(' ')} ${item.explanation}`).includes(normalize(term)))
    ));
    assert.notEqual(index, -1, `missing milestone: ${name}`);
    assert.ok(index > previous, `${name} appears out of order`);
    previous = index;
  }
});

test('bloom filter assessment marks misconceptions as traps after setup', () => {
  const misconceptionTerms = [
    /probably present proves membership/i,
    /zero bit is only a weak hint/i,
    /Why is deleting by clearing bits unsafe/i,
    /more hash functions are always better/i,
    /Which misconception about m/i,
    /key-value store/i,
    /exact hash table/i,
    /one repeated hash position/i,
    /granting privileges/i,
    /ignoring capacity drift/i,
    /false belief about the formula/i,
    /reusing old bits after changing hash seeds/i,
    /UI interpretation trap/i,
    /Why is random bit clearing/i,
    /final trap.*deployment/i,
  ];
  const trapPrompt = /false|wrong|unsafe|trap|reject|dangerous|misconception/i;

  for (const [index, item] of BLOOM_FILTER_QUIZ.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('bloom filter assessment avoids visible-page answer leakage', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < BLOOM_FILTER_QUIZ.length; pageStart += pageSize) {
    const page = BLOOM_FILTER_QUIZ.slice(pageStart, pageStart + pageSize);
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

test('bloom filter assessment distributes correct answer positions per page', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < BLOOM_FILTER_QUIZ.length; pageStart += pageSize) {
    const page = BLOOM_FILTER_QUIZ.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
