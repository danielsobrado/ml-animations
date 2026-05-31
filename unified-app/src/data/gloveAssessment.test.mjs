import assert from 'node:assert/strict';
import test from 'node:test';
import { GLOVE_QUIZ } from './gloveAssessment.js';

const LEVELS = new Set(['Foundation', 'Mechanism', 'Application', 'Tricky', 'Interview']);

function normalize(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

test('glove assessment has 100 production-ready questions', () => {
  assert.equal(GLOVE_QUIZ.length, 100);
  const ids = new Set();
  const prompts = new Set();
  const correctAnswers = new Set();

  for (const [index, item] of GLOVE_QUIZ.entries()) {
    assert.match(item.id, /^glv-\d{3}$/);
    assert.equal(ids.has(item.id), false, `duplicate id ${item.id}`);
    ids.add(item.id);
    assert.equal(item.id, `glv-${String(index + 1).padStart(3, '0')}`);
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

test('glove assessment progresses from co-occurrence intuition to deployment judgment', () => {
  const ranges = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of ranges) {
    assert.equal(GLOVE_QUIZ.slice(start, end).every((item) => item.level === level), true, `${level} range mismatch`);
  }

  const milestones = [
    ['cooccurrence matrix', ['co-occurrence matrix X', 'X_ij counts']],
    ['ratio intuition', ['probability-ratio intuition', 'semantic relationships']],
    ['objective mechanics', ['w_i dot w_tilde_j', 'log X_ij']],
    ['final embedding', ['W plus W_tilde', 'final embedding']],
    ['application limitation', ['static vector per word type', 'contextual vector per sentence']],
    ['interview synthesis', ['interview-ready GloVe mastery', 'weighted objective']],
  ];

  let previous = -1;
  for (const [name, terms] of milestones) {
    const index = GLOVE_QUIZ.findIndex((item) => (
      terms.every((term) => normalize(`${item.prompt} ${item.choices.join(' ')} ${item.explanation}`).includes(normalize(term)))
    ));
    assert.notEqual(index, -1, `missing milestone: ${name}`);
    assert.ok(index > previous, `${name} appears out of order`);
    previous = index;
  }
});

test('glove assessment marks misconceptions as traps after setup', () => {
  const misconceptionTerms = [
    /local only/i,
    /negative sampling like Word2Vec/i,
    /use only W|does not use only W/i,
    /every zero matrix entry/i,
    /co-occurrence matrix is dense/i,
    /same as raw co-occurrence counts/i,
    /x_max deletes frequent pairs/i,
    /pretrained GloVe is always best/i,
    /analogy success proves understanding/i,
    /fresh contextual vector per sentence/i,
    /updates online without rebuilding statistics/i,
    /no evaluation/i,
    /original high-dimensional embedding space/i,
    /removes all bias/i,
    /identical just because both output dense vectors/i,
  ];
  const trapPrompt = /false|wrong|unsafe|trap|reject|dangerous/i;

  for (const [index, item] of GLOVE_QUIZ.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('glove assessment avoids visible-page answer leakage', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < GLOVE_QUIZ.length; pageStart += pageSize) {
    const page = GLOVE_QUIZ.slice(pageStart, pageStart + pageSize);
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

test('glove assessment distributes correct answer positions per page', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < GLOVE_QUIZ.length; pageStart += pageSize) {
    const page = GLOVE_QUIZ.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
