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

test('attention mechanism has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('attention-mechanism');

  assert.equal(quiz.length, 100);
  assert.deepEqual(
    labs.map((lab) => lab.id),
    ['trace-library-lookup', 'compare-scenarios', 'shift-query'],
  );
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^attn-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the attn id format`);
    assert.equal(Number(question.id.slice(5, 8)), index + 1, `question ${index + 1} id number should match its position`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map((choice) => normalized(choice))).size, 3, `question ${index + 1} should have distinct choices`);
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('attention mechanism assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('attention-mechanism');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('attention mechanism assessment progresses from QKV basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('attention-mechanism');
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

test('attention mechanism assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('attention-mechanism');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['selects useful context']],
    ['query role', ['what you are looking for']],
    ['value mixing', ['weighted combination of available values']],
    ['not importance', ['permanent word importance']],
    ['preview step one', ['compute q dot k similarity scores']],
    ['library mechanism', ['book titles such as neural networks']],
    ['conversation mechanism', ['terrible and but']],
    ['application prediction', ['which item will receive the largest weight']],
    ['tricky false claims', ['query claim is false']],
    ['interview readiness', ['production ready attention takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('attention mechanism assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('attention-mechanism');
  const unsafePatterns = [
    /already the final mixed output/i,
    /content that gets retrieved directly/i,
    /without query-key scores/i,
    /only a hard lookup/i,
    /deletes values/i,
    /permanent global word importance/i,
    /always averages every item equally/i,
    /cooking recipes should be the strongest/i,
    /mainly attend to cooking recipes/i,
    /terrible has no relevance/i,
    /permanent forever/i,
    /unlimited direct access/i,
    /cannot be parallelized/i,
    /never on values/i,
    /complete explanation by itself/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|interpretation/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('attention mechanism assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('attention-mechanism');
  const misconceptionPatterns = [
    /treating attention weights as permanent global importance/i,
    /already the final mixed output/i,
    /content that gets retrieved directly/i,
    /without query-key scores/i,
    /only a hard lookup/i,
    /deletes values/i,
    /permanent global word importance/i,
    /always averages every item equally/i,
    /cooking recipes should be the strongest/i,
    /mainly attend to cooking recipes/i,
    /terrible has no relevance/i,
    /permanent forever/i,
    /unlimited direct access/i,
    /cannot be parallelized/i,
    /never on values/i,
    /complete explanation by itself/i,
  ];
  const trapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|interpretation|misconception/i;
  const trapQuestions = quiz.slice(75, 90);

  assert.ok(trapQuestions.every((question) => question.level === 'Tricky'), 'misconception traps should stay in the Tricky band');

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(answer));
    if (!containsMisconception) continue;
    if (index < 75) {
      assert.match(question.prompt, /misconception.*avoid/i, `${question.id} should scaffold any early misconception`);
      continue;
    }
    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('attention mechanism assessment stays within visible lesson scope', () => {
  const { quiz } = getLessonAssessment('attention-mechanism');
  const lessonScopeLeaks = [
    /\bmasks?\b/i,
    /\bcausal\b/i,
    /\bcross[- ]?attention\b/i,
    /\bself[- ]?attention\b/i,
    /\bmulti[- ]?head\b/i,
    /\btransformers?\b/i,
    /\bRAG\b/i,
    /\bmultimodal\b/i,
    /\bpadding\b/i,
    /\bsqrt\b/i,
    /\bd_k\b/i,
    /\bQK\^T\b/i,
    /\bresidual\b/i,
    /\bMLP\b/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const visibleText = `${question.prompt} ${question.choices.join(' ')} ${question.explanation}`;
    assert.ok(!lessonScopeLeaks.some((pattern) => pattern.test(visibleText)), `question ${index + 1} leaks later or placeholder attention scope`);
  }
});

test('attention mechanism assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('attention-mechanism');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [answerIndex, answer] of answers.entries()) {
      for (const [promptIndex, question] of page.entries()) {
        if (answerIndex === promptIndex || answer.length < 8) continue;
        const visibleQuestionText = normalized(`${question.prompt} ${question.choices.join(' ')}`);
        assert.ok(!visibleQuestionText.includes(answer));
      }
    }
  }
});

test('attention mechanism assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('attention-mechanism');
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
