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

test('vae assessment has 100 curated questions', () => {
  const { quiz, labs } = getLessonAssessment('vae');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 1);
  assert.equal(new Set(quiz.map((item) => item.id)).size, 100);

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^vae-\d{3}-[a-z0-9-]+$/);
    assert.equal(item.id.startsWith(`vae-${String(index + 1).padStart(3, '0')}`), true, `${item.id} is out of order`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
    assert.equal(new Set(item.choices.map(normalize)).size, 3, `${item.id} has duplicate choices`);
  }
});

test('vae assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('vae');
  const prompts = quiz.map((item) => normalize(item.prompt));
  const answers = quiz.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('vae assessment progresses through lesson learning points', () => {
  const { quiz } = getLessonAssessment('vae');
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
    ['architecture', ['VAE architecture overview']],
    ['encoder distribution', ['q(z|x)']],
    ['reparameterization', ['z = mu + sigma * epsilon']],
    ['prior generation', ['trained VAE generate']],
    ['ELBO', ['Evidence Lower Bound']],
    ['posterior collapse', ['Posterior collapse']],
    ['decoder expansion', ['20 to 200 to 400 to 784']],
    ['latent interpolation', ['z_interp']],
    ['application diagnostics', ['prior samples look invalid']],
    ['misconception traps', ['statement is false']],
    ['interview synthesis', ['strongest VAE takeaway']],
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

test('vae assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('vae');
  const misconceptionTerms = [
    /deterministic autoencoder with a different name/i,
    /encoder must process a real input every time/i,
    /only a reconstruction sharpness booster/i,
    /Larger beta is always better/i,
    /actual variance is allowed to be negative/i,
    /removes all randomness/i,
    /can never generate new data/i,
    /every possible VAE dataset/i,
    /definitely using rich input information/i,
    /no connection to the reconstruction and KL/i,
    /prior can be ignored during training/i,
    /Always set latent dimension equal to the input dimension/i,
    /guaranteed even without KL/i,
    /lowest reconstruction loss alone proves/i,
    /mathematically unable to use neural networks/i,
  ];
  const trapPrompt = /false|misleading|wrong|dangerous/i;

  for (const [index, item] of quiz.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('vae assessment avoids visible-page answer leakage', () => {
  const { quiz } = getLessonAssessment('vae');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const answers = page.map((item) => normalize(correctAnswer(item)));

    for (const [offset, item] of page.entries()) {
      const surroundingPrompts = page
        .filter((_, otherOffset) => otherOffset !== offset)
        .map((other) => normalize(other.prompt));
      const leaked = surroundingPrompts.some((prompt) => prompt.includes(answers[offset]));
      assert.equal(leaked, false, `${item.id} answer appears in another prompt on same page`);
    }
  }
});

test('vae assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('vae');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
