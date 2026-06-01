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

test('embeddings has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('embeddings');

  assert.equal(quiz.length, 100);
  assert.deepEqual(
    labs.map((lab) => lab.id),
    ['trace-vector-arithmetic', 'compare-cosine-angle', 'nearest-neighbor'],
  );
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^emb-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the emb id format`);
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

test('embeddings assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('embeddings');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('embeddings assessment progresses from vector basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('embeddings');
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

test('embeddings assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('embeddings');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['dense numerical vector']],
    ['two dimensional display', ['easy to draw and inspect']],
    ['not truth', ['guaranteed semantic truth']],
    ['cosine basics', ['vectors point in almost the same direction']],
    ['word algebra steps', ['start from the king vector']],
    ['3d inspection', ['more than one viewpoint']],
    ['angle application', ['cosine readout should be close to 1']],
    ['geometry caution', ['full space is perfectly separated']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['production ready embeddings takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('embeddings assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('embeddings');
  const unsafePatterns = [
    /only the final loss value/i,
    /exactly one nonzero coordinate/i,
    /only the two plotted coordinates/i,
    /guarantee identical human meaning/i,
    /proves Queen by formal logic/i,
    /can never carry useful patterns/i,
    /vectors are perpendicular/i,
    /maximum cosine score/i,
    /strongly positive cosine/i,
    /exactly the same meaning/i,
    /clear dictionary label/i,
    /comparison rule is irrelevant/i,
    /changes the learned vector meanings/i,
    /color labels alone create/i,
    /objective truth with no need for inspection/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('embeddings assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('embeddings');
  const misconceptionPatterns = [
    /treating embedding distance as guaranteed semantic truth/i,
    /only the final loss value/i,
    /exactly one nonzero coordinate/i,
    /only the two plotted coordinates/i,
    /guarantee identical human meaning/i,
    /proves Queen by formal logic/i,
    /can never carry useful patterns/i,
    /vectors are perpendicular/i,
    /maximum cosine score/i,
    /strongly positive cosine/i,
    /exactly the same meaning/i,
    /clear dictionary label/i,
    /comparison rule is irrelevant/i,
    /changes the learned vector meanings/i,
    /color labels alone create/i,
    /objective truth with no need for inspection/i,
  ];
  const trapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i;
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

test('embeddings assessment stays within visible lesson scope', () => {
  const { quiz } = getLessonAssessment('embeddings');
  const lessonScopeLeaks = [
    /\btokenizers?\b/i,
    /\btokenization\b/i,
    /\bRAG\b/i,
    /\bretrieval\b/i,
    /\brecommenders?\b/i,
    /\busers?\b/i,
    /\bproducts?\b/i,
    /\bcontextual\b/i,
    /\btransformers?\b/i,
    /\battention\b/i,
    /\blogits?\b/i,
    /\bclassifiers?\b/i,
    /\bcontrastive\b/i,
    /\bmultimodal\b/i,
    /\bbias\b/i,
    /\bprivacy\b/i,
    /\bfine[- ]?tuning\b/i,
    /\bfrozen\b/i,
    /\bindex(?:es)?\b/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const visibleText = `${question.prompt} ${question.choices.join(' ')} ${question.explanation}`;
    assert.ok(!lessonScopeLeaks.some((pattern) => pattern.test(visibleText)), `question ${index + 1} leaks later or hidden embedding-system scope`);
  }
});

test('embeddings assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('embeddings');

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

test('embeddings assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('embeddings');
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
