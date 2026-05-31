import test from 'node:test';
import assert from 'node:assert/strict';

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
    .replace(/\s+/g, ' ')
    .trim();
}

function correctAnswer(question) {
  return question.choices[question.answerIndex];
}

test('matrix multiplication has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('matrix-multiplication');
  const ids = new Set(quiz.map((question) => question.id));
  const globalCounts = [0, 0, 0];

  assert.deepEqual(labs.map((lab) => lab.id), ['compute-one-cell']);

  assert.equal(quiz.length, 100);
  assert.equal(ids.size, 100);
  assert.ok(quiz.every((question) => !question.id.startsWith('generated-')));

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^mm-\d{3}-[a-z0-9-]+$/);
    assert.ok(question.id.startsWith(`mm-${String(index + 1).padStart(3, '0')}-`), `${question.id} should match question order`);
    assert.ok(question.prompt && /\S/.test(question.prompt), `${question.id} should have a prompt`);
    assert.equal(question.choices.length, 3, `${question.id} should have three choices`);
    assert.ok(Number.isInteger(question.answerIndex), `${question.id} should have an integer answer index`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `${question.id} has invalid answer index`);
    assert.ok(question.explanation && /\S/.test(question.explanation), `${question.id} should explain the answer`);
    assert.equal(new Set(question.choices.map(normalized)).size, 3, `${question.id} should not repeat choices`);
    globalCounts[question.answerIndex] += 1;
  }

  assert.ok(
    Math.max(...globalCounts) - Math.min(...globalCounts) <= 1,
    `answer positions should be globally balanced, got ${globalCounts.join(', ')}`,
  );
});

test('matrix multiplication assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('matrix-multiplication');

  const expectedBands = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of expectedBands) {
    const band = quiz.slice(start, end);
    assert.ok(band.every((question) => question.level === level), `${level} band should occupy questions ${start + 1}-${end}`);
  }

  for (let index = 1; index < quiz.length; index += 1) {
    assert.ok(
      LEVEL_ORDER[quiz[index].level] >= LEVEL_ORDER[quiz[index - 1].level],
      `question ${index + 1} should not regress from ${quiz[index - 1].level} to ${quiz[index].level}`,
    );
  }

  const milestones = [
    ['row-column dot product', ['row-column dot product']],
    ['shape compatibility', ['inner dimensions do not match']],
    ['full 2 by 2 product', ['[[19, 22], [43, 50]]']],
    ['transformation order', ['B acts first']],
    ['ML layer shape', ['4 by 10']],
    ['implementation debugging', ['row, column, and inner-index terms']],
    ['misconception traps', ['Square matrices always commute with each other']],
    ['interview readiness', ['rule, a small example, shape constraints, and a failure mode']],
  ];

  let previous = -1;
  for (const [name, terms] of milestones) {
    const milestoneIndex = quiz.findIndex((question) => (
      terms.every((term) => normalized(`${question.prompt} ${question.choices.join(' ')} ${question.explanation}`).includes(normalized(term)))
    ));
    assert.notEqual(milestoneIndex, -1, `missing milestone: ${name}`);
    assert.ok(milestoneIndex > previous, `${name} appears out of order`);
    previous = milestoneIndex;
  }
});

test('matrix multiplication assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('matrix-multiplication');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const correctAnswers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(correctAnswers).size, correctAnswers.length);
});

test('matrix multiplication assessment avoids prior generated-question defects', () => {
  const { quiz } = getLessonAssessment('matrix-multiplication');
  const joined = normalized(quiz.flatMap((question) => [
    question.prompt,
    question.explanation,
    ...question.choices,
  ]).join(' '));
  const correctAnswers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.ok(!joined.includes('core linear algebra operation'), 'assessment should not use generic metadata filler');
  assert.ok(!joined.includes('when explain why'), 'assessment should not contain malformed generated prompt text');
  assert.ok(!joined.includes('answer letter appeared'), 'assessment should not include generic strategy-review distractors');
  assert.ok(!correctAnswers.includes('matrix multiplication is not elementwise multiplication; each output entry combines a row with a column.'), 'safe misconception wording should not be keyed as an unsafe conclusion');
});

test('matrix multiplication assessment marks misconceptions as traps after setup', () => {
  const { quiz } = getLessonAssessment('matrix-multiplication');
  const trapIds = [
    'mm-076-square-trap',
    'mm-077-validity-vs-equality',
    'mm-078-compatible-not-same-shape',
    'mm-079-elementwise-same-shape-trap',
    'mm-080-transpose-trap',
    'mm-081-association-cost',
    'mm-082-non-square-transform',
    'mm-083-invertibility-trap',
    'mm-084-broadcasting-trap',
    'mm-085-batch-axis-trap',
    'mm-086-silent-transpose',
    'mm-087-diagonal-trap',
    'mm-088-zero-product-trap',
    'mm-089-true-fact-wrong-scope',
    'mm-090-tricky-check',
  ];
  const misconceptionTerms = [
    /Square matrices always commute with each other/i,
    /That the two products are equal/i,
    /Both must produce identical 2 by 2 matrices/i,
    /The product must copy the diagonal only/i,
    /Keeping the factor order unchanged/i,
    /Different groupings change the mathematical result/i,
    /They cannot represent useful linear maps/i,
    /Every product has an inverse/i,
    /Whether the operator performed elementwise multiplication/i,
    /Batch size always replaces feature size/i,
    /blindly transposing/i,
    /Every diagonal matrix commutes with every matrix of any shape/i,
    /A zero product is impossible/i,
    /true rule but applies it to incompatible shapes/i,
  ];
  const trapPrompt = /avoid|tricky|unsafe|mistake|verify|risky|surprising|wrong|not an elementwise|goes beyond|confusing|why might/i;

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

  for (const [index, question] of quiz.entries()) {
    const text = `${question.prompt} ${question.choices.join(' ')} ${question.explanation}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('matrix multiplication assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('matrix-multiplication');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact correct answers`);

    for (const [questionIndex, question] of page.entries()) {
      const prompt = normalized(question.prompt);

      for (const [answerIndex, answer] of answers.entries()) {
        if (questionIndex === answerIndex || answer.length < 8) continue;

        assert.ok(
          !prompt.includes(answer),
          `question ${pageStart + questionIndex + 1} prompt should not reveal answer from question ${pageStart + answerIndex + 1}`,
        );
      }
    }
  }
});

test('matrix multiplication assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('matrix-multiplication');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const question of page) counts[question.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
