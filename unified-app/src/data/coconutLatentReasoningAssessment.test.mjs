import assert from 'node:assert/strict';
import test from 'node:test';
import { COCONUT_LATENT_REASONING_QUIZ } from './coconutLatentReasoningAssessment.js';

const LEVELS = new Set(['Foundation', 'Mechanism', 'Application', 'Tricky', 'Interview']);

function normalize(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

test('coconut latent reasoning assessment has 100 production-ready questions', () => {
  assert.equal(COCONUT_LATENT_REASONING_QUIZ.length, 100);
  const ids = new Set();

  for (const [index, item] of COCONUT_LATENT_REASONING_QUIZ.entries()) {
    assert.match(item.id, /^coconut-\d{3}$/);
    assert.equal(ids.has(item.id), false, `duplicate id ${item.id}`);
    ids.add(item.id);
    assert.equal(item.id, `coconut-${String(index + 1).padStart(3, '0')}`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
    assert.equal(new Set(item.choices.map(normalize)).size, 3, `${item.id} has duplicate choices`);
  }
});

test('coconut latent reasoning assessment progresses from feedback mechanics to production audit', () => {
  const ranges = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of ranges) {
    assert.equal(COCONUT_LATENT_REASONING_QUIZ.slice(start, end).every((item) => item.level === level), true, `${level} range mismatch`);
  }

  const milestones = [
    ['hidden-state feedback', ['hidden state', 'intermediate reasoning']],
    ['LM-head bypass', ['lm head', 'word token']],
    ['curriculum', ['curriculum', 'early text reasoning']],
    ['delayed commitment', ['branch entropy']],
    ['faithfulness intervention', ['perturb or remove']],
    ['production audit', ['progressive coverage', 'registry integration']],
  ];

  let previous = -1;
  for (const [name, terms] of milestones) {
    const index = COCONUT_LATENT_REASONING_QUIZ.findIndex((item) => (
      terms.every((term) => normalize(`${item.prompt} ${item.choices.join(' ')} ${item.explanation}`).includes(normalize(term)))
    ));
    assert.notEqual(index, -1, `missing milestone: ${name}`);
    assert.ok(index > previous, `${name} appears out of order`);
    previous = index;
  }
});

test('coconut latent reasoning assessment marks unsafe misconceptions as traps after setup', () => {
  const misconceptionTerms = [
    /automatically faithful/i,
    /reasoning free/i,
    /same thing as a continuous thought/i,
    /probes prove/i,
    /accuracy proves/i,
    /lm head is used inside every/i,
    /receive direct word-label loss/i,
    /fewer visible tokens always/i,
    /CoT and Coconut differ only/i,
    /entropy drop guarantees/i,
    /must correspond to one hidden English sentence/i,
  ];
  const trapPrompt = /false|trap|reject|misconception|wrong|unsafe/i;

  for (const [index, item] of COCONUT_LATENT_REASONING_QUIZ.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('coconut latent reasoning assessment avoids visible-page answer leakage', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < COCONUT_LATENT_REASONING_QUIZ.length; pageStart += pageSize) {
    const page = COCONUT_LATENT_REASONING_QUIZ.slice(pageStart, pageStart + pageSize);
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

test('coconut latent reasoning assessment distributes correct answer positions per page', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < COCONUT_LATENT_REASONING_QUIZ.length; pageStart += pageSize) {
    const page = COCONUT_LATENT_REASONING_QUIZ.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 2, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
