import assert from 'node:assert/strict';
import test from 'node:test';
import { BAG_OF_WORDS_QUIZ } from './bagOfWordsAssessment.js';

const LEVELS = new Set(['Foundation', 'Mechanism', 'Application', 'Tricky', 'Interview']);

function normalize(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

test('bag of words assessment has 100 production-ready questions', () => {
  assert.equal(BAG_OF_WORDS_QUIZ.length, 100);
  const ids = new Set();
  const prompts = new Set();
  const correctAnswers = new Set();

  for (const [index, item] of BAG_OF_WORDS_QUIZ.entries()) {
    assert.match(item.id, /^bow-\d{3}$/);
    assert.equal(ids.has(item.id), false, `duplicate id ${item.id}`);
    ids.add(item.id);
    assert.equal(item.id, `bow-${String(index + 1).padStart(3, '0')}`);
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

test('bag of words assessment progresses from counts to applied text modeling', () => {
  const ranges = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of ranges) {
    assert.equal(BAG_OF_WORDS_QUIZ.slice(start, end).every((item) => item.level === level), true, `${level} range mismatch`);
  }

  const milestones = [
    ['text to vectors', ['numeric feature vectors', 'machine learning']],
    ['manual vector', ['vocabulary [cat, dog, mat]', '[2, 1, 0]']],
    ['tf-idf weighting', ['IDF down-weight', 'smaller weights']],
    ['application caveat', ['negation', 'composition']],
    ['trap setup', ['word-order claim is false', 'preserves sentence order']],
    ['interview synthesis', ['interview-ready mastery', 'TF-IDF tradeoffs']],
  ];

  let previous = -1;
  for (const [name, terms] of milestones) {
    const index = BAG_OF_WORDS_QUIZ.findIndex((item) => (
      terms.every((term) => normalize(`${item.prompt} ${item.choices.join(' ')} ${item.explanation}`).includes(normalize(term)))
    ));
    assert.notEqual(index, -1, `missing milestone: ${name}`);
    assert.ok(index > previous, `${name} appears out of order`);
    previous = index;
  }
});

test('bag of words assessment marks misconceptions as traps after setup', () => {
  const misconceptionTerms = [
    /preserves sentence order/i,
    /automatically knows that/i,
    /create new coordinates/i,
    /Removing stopwords is always better/i,
    /fully solves word order/i,
    /always means stronger relevance/i,
    /independent of the corpus/i,
    /negative sentiment/i,
    /remove the need for bias/i,
    /cannot change the BoW matrix/i,
    /understand language like a human/i,
    /all possible n-grams has no cost/i,
    /obsolete for every query/i,
    /different things for each document/i,
    /never worth testing/i,
  ];
  const trapPrompt = /false|wrong|unsafe|trap|reject|dangerous/i;

  for (const [index, item] of BAG_OF_WORDS_QUIZ.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('bag of words assessment avoids visible-page answer leakage', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < BAG_OF_WORDS_QUIZ.length; pageStart += pageSize) {
    const page = BAG_OF_WORDS_QUIZ.slice(pageStart, pageStart + pageSize);
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

test('bag of words assessment distributes correct answer positions per page', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < BAG_OF_WORDS_QUIZ.length; pageStart += pageSize) {
    const page = BAG_OF_WORDS_QUIZ.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
