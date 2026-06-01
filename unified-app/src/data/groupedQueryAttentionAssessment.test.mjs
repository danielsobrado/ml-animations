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

const EXPECTED_LEVEL_COUNTS = {
  Foundation: 20,
  Mechanism: 30,
  Application: 25,
  Tricky: 15,
  Interview: 10,
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

test('grouped-query attention has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('grouped-query-attention');

  assert.equal(quiz.length, 100);
  assert.deepEqual(
    labs.map((lab) => lab.id),
    ['map-query-to-kv-groups', 'compare-mha-mqa-gqa', 'estimate-cache-bandwidth'],
  );
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);
  assert.deepEqual(
    Object.fromEntries(Object.keys(EXPECTED_LEVEL_COUNTS).map((level) => [
      level,
      quiz.filter((question) => question.level === level).length,
    ])),
    EXPECTED_LEVEL_COUNTS,
  );

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^gqa-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the gqa id format`);
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

test('grouped-query attention assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('grouped-query-attention');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('grouped-query attention assessment progresses from sharing basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('grouped-query-attention');
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

test('grouped-query attention assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('grouped-query-attention');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['reduces kv cache memory']],
    ['shared content', ['key and value heads']],
    ['mha endpoint', ['each query head has its own kv head']],
    ['mqa endpoint', ['many query heads share one kv head']],
    ['group size', ['query heads divided by kv heads']],
    ['head mapping', ['floor h divided by group size']],
    ['quality tradeoff', ['too few kv heads', 'hurt quality']],
    ['implementation risk', ['assumes mha shapes']],
    ['tricky false claims', ['gqa claim is false']],
    ['interview readiness', ['production ready gqa takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('grouped-query attention assessment avoids unsafe misconception keying before explicit traps', () => {
  const { quiz } = getLessonAssessment('grouped-query-attention');
  const unsafePatterns = [
    /removes all query heads/i,
    /more kv heads than query heads/i,
    /equal query and kv head counts are the gqa/i,
    /cache memory disappear/i,
    /eliminates attention softmax/i,
    /attend to future tokens/i,
    /deleting model weights/i,
    /can never affect quality/i,
    /separate kv head for every query head/i,
    /arbitrary prompt or weight changes/i,
    /must reduce head dimension/i,
    /automatically supports gqa shapes/i,
    /attention reads constant/i,
    /replacement loss function/i,
    /free quality, zero memory/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    const scaffoldPrompt = /misconception.*avoid/i.test(question.prompt);
    assert.ok(
      !unsafeAnswer || scaffoldPrompt || (index >= 75 && explicitTrapPrompt),
      `question ${index + 1} keys a false claim outside a trap prompt`,
    );
  }
});

test('grouped-query attention assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('grouped-query-attention');
  const trapIds = [
    'gqa-076-false-no-query',
    'gqa-077-false-more-kv',
    'gqa-078-false-mha',
    'gqa-079-false-cache-free',
    'gqa-080-false-attention',
    'gqa-081-false-causal',
    'gqa-082-false-bandwidth',
    'gqa-083-false-quality',
    'gqa-084-false-mqa',
    'gqa-085-false-cache-valid',
    'gqa-086-false-dimension',
    'gqa-087-false-kernel',
    'gqa-088-false-context',
    'gqa-089-false-training',
    'gqa-090-tricky-summary',
  ];
  const misconceptionPatterns = [
    /removes all query heads/i,
    /more kv heads than query heads/i,
    /equal query and kv head counts are the gqa/i,
    /cache memory disappear/i,
    /eliminates attention softmax/i,
    /attend to future tokens/i,
    /deleting model weights/i,
    /can never affect quality/i,
    /separate kv head for every query head/i,
    /arbitrary prompt or weight changes/i,
    /must reduce head dimension/i,
    /automatically supports gqa shapes/i,
    /attention reads constant/i,
    /replacement loss function/i,
    /free quality, zero memory/i,
  ];
  const trapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i;

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

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

test('grouped-query attention assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('grouped-query-attention');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [answerIndex, answer] of answers.entries()) {
      for (const [promptIndex, question] of page.entries()) {
        if (answerIndex === promptIndex || answer.length < 8) continue;
        const visibleText = normalized([question.prompt, ...question.choices].join(' '));
        assert.ok(
          !visibleText.includes(answer),
          `question ${pageStart + promptIndex + 1} visible text should not reveal answer from question ${pageStart + answerIndex + 1}`,
        );
      }
    }
  }
});

test('grouped-query attention assessment keeps visible wording scoped to GQA mechanics', () => {
  const { quiz } = getLessonAssessment('grouped-query-attention');
  const genericLeakagePatterns = [
    /full vocabulary table/i,
    /hidden layer per token/i,
    /vocabulary classes/i,
    /raw prompt tokens/i,
    /source maps/i,
    /route icons/i,
    /vocabulary size/i,
    /labels to optimizer states/i,
    /button radius/i,
    /page title/i,
    /validation labels/i,
    /dark mode/i,
    /final answer token/i,
    /generated answers in a dataset/i,
    /browser viewport/i,
    /softmax\(h\) over the vocabulary/i,
    /token id/i,
    /vocabulary matrix/i,
    /optimizer momentum/i,
    /future labels/i,
    /loss function becomes undefined/i,
    /generated users/i,
    /web browser/i,
    /css/i,
    /validation set/i,
    /final text color/i,
    /answer font size/i,
    /route metadata/i,
    /svg icons/i,
    /class labels/i,
    /final css class/i,
    /app navigation items/i,
    /final validation examples/i,
    /package lock/i,
    /route aliases/i,
    /favicon/i,
    /labels are alphabetical/i,
    /page background color/i,
    /ids sort lexically/i,
    /tokenizer algorithm/i,
    /ranking labels/i,
    /optimizer choice/i,
    /final answer probability/i,
    /image dimensions/i,
    /optimizer spelling/i,
    /package count/i,
    /file names/i,
    /homepage/i,
    /route id/i,
    /reload the page title/i,
  ];

  for (const question of quiz) {
    const visibleText = [question.prompt, ...question.choices, question.explanation].join(' ');
    for (const pattern of genericLeakagePatterns) {
      assert.doesNotMatch(visibleText, pattern, `${question.id} contains generic or off-scope visible wording`);
    }
  }
});

test('grouped-query attention assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('grouped-query-attention');
  const globalPositionCounts = [0, 1, 2].map((slot) => quiz.filter((question) => question.answerIndex === slot).length);

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const pagePositionCounts = [0, 1, 2].map((slot) => positions.filter((position) => position === slot).length);

    assert.ok(
      Math.max(...pagePositionCounts) - Math.min(...pagePositionCounts) <= 1,
      `page starting at question ${pageStart + 1} should balance answer positions, saw ${pagePositionCounts.join('/')}`,
    );
  }

  assert.ok(
    Math.max(...globalPositionCounts) - Math.min(...globalPositionCounts) <= 1,
    `global answer positions should be balanced, got ${globalPositionCounts.join(', ')}`,
  );
});
