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

test('cosine similarity has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('cosine-similarity');

  assert.equal(quiz.length, 100);
  assert.deepEqual(
    labs.map((lab) => lab.id),
    ['trace-dot-product-normalization', 'compare-movie-matches', 'predict-nearest'],
  );
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^cos-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the cos id format`);
    assert.equal(Number(question.id.slice(4, 7)), index + 1, `question ${index + 1} id number should match its position`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map((choice) => normalized(choice))).size, 3, `question ${index + 1} should have distinct choices`);
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('cosine similarity assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('cosine-similarity');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('cosine similarity assessment progresses from direction basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('cosine-similarity');
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

test('cosine similarity assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('cosine-similarity');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['compares vector direction']],
    ['same direction', ['positive one']],
    ['not proof', ['ranking signals not proof']],
    ['formula', ['dot product divided by the product']],
    ['normalized dot product', ['cosine equals their dot product']],
    ['zero vector', ['norm is zero so cosine is undefined']],
    ['retrieval validation', ['precision and recall around that cutoff']],
    ['production monitoring', ['score distribution or top neighbor quality']],
    ['tricky false claims', ['interpretation is false']],
    ['interview readiness', ['production ready cosine takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('cosine similarity assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('cosine-similarity');
  const unsafePatterns = [
    /calibrated probability of relevance/i,
    /always changes cosine/i,
    /cosine similarity 1 with every vector/i,
    /always rank candidates identically/i,
    /universal cosine cutoff/i,
    /guaranteed factually correct/i,
    /removes training-data bias automatically/i,
    /different coordinate counts directly/i,
    /unrelated to dot product/i,
    /correct cosine rankings automatically/i,
    /always final truth/i,
    /ignores all term overlap/i,
    /needs no monitoring/i,
    /without alignment/i,
    /proves meaning, fairness, and correctness/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|interpretation/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('cosine similarity assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('cosine-similarity');
  const misconceptionPatterns = [
    /equating high cosine with guaranteed relevance/i,
    /calibrated probability of relevance/i,
    /always changes cosine/i,
    /cosine similarity 1 with every vector/i,
    /always rank candidates identically/i,
    /universal cosine cutoff/i,
    /guaranteed factually correct/i,
    /removes training-data bias automatically/i,
    /different coordinate counts directly/i,
    /unrelated to dot product/i,
    /correct cosine rankings automatically/i,
    /always final truth/i,
    /ignores all term overlap/i,
    /needs no monitoring/i,
    /without alignment/i,
    /proves meaning, fairness, and correctness/i,
  ];
  const trapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|interpretation|mistake/i;

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(answer));
    if (!containsMisconception) continue;
    if (index < 75) {
      assert.match(question.prompt, /mistake.*avoid/i, `${question.id} should scaffold any early misconception`);
      continue;
    }
    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('cosine similarity assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('cosine-similarity');

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

test('cosine similarity assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('cosine-similarity');
  const globalPositionCounts = [0, 1, 2].map((slot) => quiz.filter((question) => question.answerIndex === slot).length);

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }

  assert.ok(
    Math.max(...globalPositionCounts) - Math.min(...globalPositionCounts) <= 1,
    `global answer positions should be balanced, got ${globalPositionCounts.join(', ')}`,
  );
});
