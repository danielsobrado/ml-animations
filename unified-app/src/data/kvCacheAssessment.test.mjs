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

test('kv-cache has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('kv-cache');

  assert.equal(quiz.length, 100);
  assert.deepEqual(
    labs.map((lab) => lab.id),
    ['trace-cache-table', 'decode-step-savings', 'bound-window-memory'],
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
    assert.match(question.id, /^kvc-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the kvc id format`);
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

test('kv-cache assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('kv-cache');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('kv-cache assessment progresses from inference reuse basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('kv-cache');
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

test('kv-cache assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('kv-cache');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['old key and value projections']],
    ['prefill', ['processing the prompt and filling']],
    ['memory growth', ['visible tokens layers']],
    ['still attends', ['still attends over cached history']],
    ['cache update', ['concatenate k new and v new']],
    ['memory formula', ['2 times layers']],
    ['invalidation', ['prefix tokens or model weights']],
    ['serving oom', ['kv cache footprint']],
    ['tricky false claims', ['kv cache claim is false']],
    ['interview readiness', ['production ready kv cache takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('kv-cache assessment avoids unsafe misconception keying before explicit traps', () => {
  const { quiz } = getLessonAssessment('kv-cache');
  const unsafePatterns = [
    /skips attention entirely/i,
    /final logits for every old token/i,
    /constant no matter how long context gets/i,
    /after all output tokens are generated/i,
    /recomputes all old k\/v projections every step/i,
    /remains valid after changing prefix tokens/i,
    /infinite old context at no cost/i,
    /increases kv heads/i,
    /can never affect output quality/i,
    /no memory-read bottleneck/i,
    /always have identical cache lengths/i,
    /never sensitive/i,
    /changes model weights/i,
    /remove kv cache storage/i,
    /infinite context, zero memory cost/i,
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

test('kv-cache assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('kv-cache');
  const trapIds = [
    'kvc-076-false-skip-trap',
    'kvc-077-false-logits-trap',
    'kvc-078-false-memory-trap',
    'kvc-079-false-prefill-trap',
    'kvc-080-false-decode-trap',
    'kvc-081-false-validity-trap',
    'kvc-082-false-window-trap',
    'kvc-083-false-gqa-trap',
    'kvc-084-false-quant-trap',
    'kvc-085-false-bandwidth-trap',
    'kvc-086-false-batch-trap',
    'kvc-087-false-security-trap',
    'kvc-088-false-paged-trap',
    'kvc-089-false-flash-trap',
    'kvc-090-tricky-summary',
  ];
  const misconceptionPatterns = [
    /skips attention entirely/i,
    /final logits for every old token/i,
    /constant no matter how long context gets/i,
    /after all output tokens are generated/i,
    /recomputes all old k\/v projections every step/i,
    /remains valid after changing prefix tokens/i,
    /infinite old context at no cost/i,
    /increases kv heads/i,
    /can never affect output quality/i,
    /no memory-read bottleneck/i,
    /always have identical cache lengths/i,
    /never sensitive/i,
    /changes model weights/i,
    /remove kv cache storage/i,
    /infinite context, zero memory cost/i,
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

test('kv-cache assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('kv-cache');

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

test('kv-cache assessment keeps visible wording scoped to inference caching', () => {
  const { quiz } = getLessonAssessment('kv-cache');
  const genericLeakagePatterns = [
    /raw prompt strings/i,
    /future answer/i,
    /full dataset/i,
    /vocabulary alphabetically/i,
    /all webpages/i,
    /optimizer state/i,
    /route names/i,
    /class labels/i,
    /css/i,
    /final probabilities/i,
    /class dimension/i,
    /one label/i,
    /accuracy/i,
    /loss value/i,
    /train and test sets/i,
    /tokenizer vocabulary/i,
    /visual labels/i,
    /training with larger labels/i,
    /route icons/i,
    /answer letters/i,
    /homepage/i,
    /route aliases/i,
    /training set size/i,
    /answer option order/i,
    /model card title/i,
    /public labels/i,
    /number of icons/i,
    /validation file names/i,
    /static dictionary/i,
    /final probability count/i,
    /css updates/i,
    /training labels/i,
    /button layout/i,
  ];

  for (const question of quiz) {
    const visibleText = [question.prompt, ...question.choices, question.explanation].join(' ');
    for (const pattern of genericLeakagePatterns) {
      assert.doesNotMatch(visibleText, pattern, `${question.id} contains generic or off-scope visible wording`);
    }
  }
});

test('kv-cache assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('kv-cache');
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
