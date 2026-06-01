import assert from 'node:assert/strict';
import test from 'node:test';
import { getLessonAssessment } from './lessonAssessments.js';

const LEVELS = new Set(['Foundation', 'Mechanism', 'Application', 'Tricky', 'Interview']);

function normalize(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

function correctAnswer(item) {
  return item.choices[item.answerIndex];
}

test('gpt2 comprehensive assessment has 100 lesson-ready questions with focused labs', () => {
  const { quiz, labs } = getLessonAssessment('gpt2-comprehensive');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 1);
  assert.deepEqual(labs.map((lab) => lab.id), ['trace-gpt2-token']);
  assert.equal(new Set(quiz.map((item) => item.id)).size, 100);

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^gpt2-\d{3}-[a-z0-9-]+$/);
    assert.equal(item.id.startsWith(`gpt2-${String(index + 1).padStart(3, '0')}`), true, `${item.id} is out of order`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
    assert.equal(new Set(item.choices.map(normalize)).size, 3, `${item.id} has duplicate choices`);
  }
});

test('gpt2 comprehensive assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('gpt2-comprehensive');
  const prompts = quiz.map((item) => normalize(item.prompt));
  const answers = quiz.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('gpt2 comprehensive assessment follows the lesson step order', () => {
  const { quiz } = getLessonAssessment('gpt2-comprehensive');
  const ranges = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of ranges) {
    assert.equal(quiz.slice(start, end).every((item) => item.level === level), true, `${level} range mismatch`);
  }

  const milestones = [
    ['decoder objective', ['decoder only']],
    ['BPE tokens', ['BPE', 'rare words']],
    ['position embeddings', ['learned absolute positional embeddings']],
    ['causal attention', ['future columns above']],
    ['feed forward', ['FFN', 'GELU']],
    ['pre layer norm', ['pre layer normalization']],
    ['architecture scale', ['GPT 2 Medium']],
    ['weight tying', ['output projection reuses']],
    ['training optimization', ['Several small batch gradients']],
    ['inference caching', ['previous keys and values']],
    ['sampling', ['fixed number of likely tokens']],
    ['trap review', ['architecture claim is false']],
    ['interview synthesis', ['production ready GPT 2 takeaway']],
  ];

  let previous = -1;
  for (const [name, terms] of milestones) {
    const index = quiz.findIndex((item) => (
      terms.every((term) => normalize(`${item.prompt} ${item.choices.join(' ')} ${item.explanation}`).includes(normalize(term)))
    ));
    assert.notEqual(index, -1, `missing milestone: ${name}`);
    assert.ok(index > previous, `${name} appears out of order`);
    previous = index;
  }
});

test('gpt2 comprehensive assessment marks misconceptions as traps after setup', () => {
  const { quiz } = getLessonAssessment('gpt2-comprehensive');
  const misconceptionTerms = [
    /bidirectional encoder attention/i,
    /word-level tokenization/i,
    /ignore positions/i,
    /attend to future target tokens/i,
    /heads must learn identical/i,
    /FFN is the sublayer that mixes/i,
    /decorative diagram arrows/i,
    /Post-LN style/i,
    /largest GPT-2 variant is always/i,
    /Weight tying doubles/i,
    /reduces the number of examples/i,
    /FP16 always improves stability/i,
    /training optimizer for larger effective batch/i,
    /Greedy decoding is always/i,
    /Top-k and top-p define candidate sets in exactly the same way/i,
  ];
  const trapPrompt = /false|rejected|unsafe|wrong|misleading|corrected|contradicts|simplistic|too strong/i;

  for (const [index, item] of quiz.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('gpt2 comprehensive assessment stays within visible lesson scope', () => {
  const { quiz } = getLessonAssessment('gpt2-comprehensive');
  const lessonScopeLeaks = [
    /\bMoE\b/i,
    /\bMLA\b/i,
    /\bGRPO\b/i,
    /\bRLHF\b/i,
    /\bDPO\b/i,
    /\bagentic\b/i,
    /\bdiffusion\b/i,
    /\btool[- ]?call\b/i,
    /\bRAG\b/i,
    /\bfrontier\b/i,
  ];

  for (const [index, item] of quiz.entries()) {
    const visibleText = `${item.prompt} ${item.choices.join(' ')} ${item.explanation}`;
    assert.ok(!lessonScopeLeaks.some((pattern) => pattern.test(visibleText)), `question ${index + 1} leaks later or non-visible GPT-2 scope`);
  }
});

test('gpt2 comprehensive assessment avoids visible-page answer leakage', () => {
  const { quiz } = getLessonAssessment('gpt2-comprehensive');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const answers = page.map((item) => normalize(correctAnswer(item)));

    for (const [offset, item] of page.entries()) {
      const surroundingQuestions = page
        .filter((_, otherOffset) => otherOffset !== offset)
        .map((other) => normalize(`${other.prompt} ${other.choices.join(' ')}`));
      const leaked = surroundingQuestions.some((questionText) => questionText.includes(answers[offset]));
      assert.equal(leaked, false, `${item.id} answer appears in another visible question on same page`);
    }
  }
});

test('gpt2 comprehensive assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('gpt2-comprehensive');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
