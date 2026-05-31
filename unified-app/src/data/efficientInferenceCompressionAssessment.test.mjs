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
    .replace(/\s+/g, ' ')
    .trim();
}

function correctAnswer(question) {
  return question.choices[question.answerIndex];
}

test('efficient inference compression has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('efficient-inference-compression-track');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.ok(question.id.startsWith('effinf-'), `question ${index + 1} should use the effinf id prefix`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('efficient inference compression assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('efficient-inference-compression-track');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('efficient inference compression assessment progresses from serving basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('efficient-inference-compression-track');
  const expectedBands = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of expectedBands) {
    assert.deepEqual(
      [...new Set(quiz.slice(start, end).map((question) => question.level))],
      [level],
      `${level} questions should occupy positions ${start + 1}-${end}`,
    );
  }

  for (let index = 1; index < quiz.length; index += 1) {
    assert.ok(
      LEVEL_ORDER[quiz[index].level] >= LEVEL_ORDER[quiz[index - 1].level],
      `question ${index + 1} should not regress in difficulty`,
    );
  }
});

test('efficient inference compression assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('efficient-inference-compression-track');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const orderedMilestones = [
    ['purpose', ['throughput, latency, memory, cost, and quality budgets']],
    ['throughput', ['work the system completes per unit time']],
    ['time-to-first-token', ['first output token']],
    ['KV cache', ['past key and value states']],
    ['quantization', ['fewer bits']],
    ['distillation', ['smaller student']],
    ['speculative decoding', ['draft future tokens cheaply']],
    ['bottleneck diagnosis', ['compute, memory bandwidth, cache capacity, queueing']],
    ['decode bandwidth', ['reads cached key and value states']],
    ['KV cache formula', ['batch size, sequence length, layers, kv heads']],
    ['continuous batching', ['admits and removes requests across decode steps']],
    ['speculation tradeoff', ['draft acceptance rate', 'draft cost']],
    ['application workflow', ['target bottleneck', 'latency, memory, cost, quality']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['profile prefill and decode']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of orderedMilestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('efficient inference compression assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('efficient-inference-compression-track');
  const unsafePatterns = [
    /raw tokens per second always optimizes user experience/i,
    /lower precision is always quality-neutral/i,
    /kv cache memory is constant/i,
    /larger batches always reduce ttft/i,
    /any sparse pattern automatically creates proportional/i,
    /distilled student always preserves teacher behavior/i,
    /draft tokens can be accepted without target verification/i,
    /paged attention eliminates every memory and latency bottleneck/i,
    /average latency alone proves p99/i,
    /cpu or disk offload is always faster/i,
    /late, failed, or low-quality responses count the same/i,
    /smallest model artifact is automatically the best/i,
    /extending context length has no serving cost/i,
    /no monitoring is needed for serving optimizations/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|claim/i.test(question.prompt);

    assert.ok(
      !unsafeAnswer || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('efficient inference compression assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('efficient-inference-compression-track');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [answerIndex, answer] of answers.entries()) {
      for (const [promptIndex, question] of page.entries()) {
        if (answerIndex === promptIndex) continue;
        assert.ok(
          !normalized(question.prompt).includes(answer),
          `question ${pageStart + promptIndex + 1} prompt should not reveal answer from question ${pageStart + answerIndex + 1}`,
        );
      }
    }
  }
});

test('efficient inference compression assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('efficient-inference-compression-track');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }
});
