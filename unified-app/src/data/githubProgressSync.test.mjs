import assert from 'node:assert/strict';
import test from 'node:test';

import { CODE_LAB_PROGRESS_KEY } from './codeLabProgress.js';
import {
  DEFAULT_GITHUB_SYNC_SETTINGS,
  GITHUB_SYNC_SETTINGS_KEY,
  buildProgressEnvelope,
  getGitHubSyncAuthUrl,
  mergeCodeLabProgress,
  normalizeGitHubSyncSettings,
  parseProgressEnvelopeJson,
  readGitHubSyncSettings,
  summarizeProgressDocument,
  syncCodeLabProgressToGitHub,
  writeGitHubSyncSettings,
} from './githubProgressSync.js';

function createStorage(seed = {}) {
  const values = new Map(Object.entries(seed));
  return {
    getItem(key) {
      return values.has(key) ? values.get(key) : null;
    },
    setItem(key, value) {
      values.set(key, String(value));
    },
    dump() {
      return Object.fromEntries(values.entries());
    },
  };
}

function jsonResponse(payload, init = {}) {
  return {
    ok: init.ok ?? true,
    status: init.status ?? 200,
    headers: new Map(Object.entries(init.headers || {})),
    async text() {
      return JSON.stringify(payload);
    },
  };
}

test('GitHub sync settings normalize safely and never store tokens', () => {
  const storage = createStorage();
  const settings = writeGitHubSyncSettings({
    enabled: true,
    autoSync: true,
    target: 'gist',
    brokerUrl: 'https://sync.example.com/',
    owner: '  octo  ',
    repo: ' progress ',
    branch: '',
    path: '/custom/progress.json',
    gistId: 'abc',
    token: 'should-not-persist',
  }, storage);

  assert.equal(settings.brokerUrl, 'https://sync.example.com');
  assert.equal(settings.owner, 'octo');
  assert.equal(settings.repo, 'progress');
  assert.equal(settings.branch, DEFAULT_GITHUB_SYNC_SETTINGS.branch);
  assert.equal(settings.path, 'custom/progress.json');
  assert.equal(settings.target, 'gist');
  assert.equal(settings.enabled, true);
  assert.equal(storage.dump()[GITHUB_SYNC_SETTINGS_KEY].includes('token'), false);
  assert.deepEqual(readGitHubSyncSettings(storage), settings);
});

test('GitHub auth URL routes through the configured broker', () => {
  const url = getGitHubSyncAuthUrl({
    brokerUrl: 'https://sync.example.com/',
    target: 'repo',
  }, 'https://danielsobrado.github.io/ml-animations/settings');

  assert.equal(
    url,
    'https://sync.example.com/auth/github/start?returnTo=https%3A%2F%2Fdanielsobrado.github.io%2Fml-animations%2Fsettings&target=repo',
  );
});

test('progress envelopes preserve only normalized pass metadata', () => {
  const envelope = buildProgressEnvelope({
    lesson: {
      passed: {
        passed: true,
        lastPassedAt: '2026-05-25T10:00:00.000Z',
        checkCount: 3,
        sourceCode: 'drop this',
      },
      failed: {
        passed: false,
        lastPassedAt: '2026-05-25T10:01:00.000Z',
        checkCount: 3,
      },
    },
  }, new Date('2026-05-25T11:00:00.000Z'));

  assert.deepEqual(envelope, {
    version: 1,
    updatedAt: '2026-05-25T11:00:00.000Z',
    source: 'ml-animations',
    progress: {
      lesson: {
        passed: {
          passed: true,
          lastPassedAt: '2026-05-25T10:00:00.000Z',
          checkCount: 3,
        },
      },
    },
  });
  assert.equal(JSON.stringify(envelope).includes('sourceCode'), false);
  assert.deepEqual(parseProgressEnvelopeJson(JSON.stringify(envelope)), envelope);
});

test('progress merge keeps newest entry per lesson exercise', () => {
  const merged = mergeCodeLabProgress({
    lesson: {
      a: { passed: true, lastPassedAt: '2026-05-25T10:00:00.000Z', checkCount: 2 },
      b: { passed: true, lastPassedAt: '2026-05-25T10:05:00.000Z', checkCount: 4 },
    },
  }, {
    lesson: {
      a: { passed: true, lastPassedAt: '2026-05-25T10:10:00.000Z', checkCount: 5 },
      c: { passed: true, lastPassedAt: '2026-05-25T10:15:00.000Z', checkCount: 1 },
    },
  });

  assert.equal(merged.lesson.a.checkCount, 5);
  assert.equal(merged.lesson.b.checkCount, 4);
  assert.equal(merged.lesson.c.checkCount, 1);
  assert.deepEqual(summarizeProgressDocument(merged), { scopeCount: 1, passedCount: 3 });
});

test('GitHub sync pulls, merges, writes local progress, then pushes one progress document', async () => {
  const storage = createStorage({
    [CODE_LAB_PROGRESS_KEY]: JSON.stringify({
      local: {
        a: { passed: true, lastPassedAt: '2026-05-25T10:00:00.000Z', checkCount: 2 },
      },
    }),
  });
  const settings = normalizeGitHubSyncSettings({
    enabled: true,
    brokerUrl: 'https://sync.example.com',
    owner: 'octo',
    repo: 'progress',
    branch: 'main',
    path: '.ml-animations/progress.v1.json',
  });
  const calls = [];
  const fetchFn = async (url, options = {}) => {
    calls.push({ url, options });
    if (options.method === 'PUT') {
      const body = JSON.parse(options.body);
      assert.equal(body.target, 'repo');
      assert.equal(body.owner, 'octo');
      assert.equal(body.sha, 'abc123');
      assert.ok(body.envelope.progress.local.a);
      assert.ok(body.envelope.progress.remote.b);
      return jsonResponse({ sha: 'def456' });
    }

    return jsonResponse({
      exists: true,
      sha: 'abc123',
      envelope: buildProgressEnvelope({
        remote: {
          b: { passed: true, lastPassedAt: '2026-05-25T10:03:00.000Z', checkCount: 3 },
        },
      }, new Date('2026-05-25T10:04:00.000Z')),
    });
  };

  const result = await syncCodeLabProgressToGitHub({
    settings,
    storage,
    fetchFn,
    now: new Date('2026-05-25T10:20:00.000Z'),
  });

  assert.equal(calls.length, 2);
  assert.equal(calls[0].options.credentials, 'include');
  assert.equal(calls[1].options.method, 'PUT');
  assert.deepEqual(result.summary, { scopeCount: 2, passedCount: 2 });
  assert.equal(result.settings.lastStatus, 'Synced');
});

test('GitHub sync retries once after a remote conflict', async () => {
  const storage = createStorage({
    [CODE_LAB_PROGRESS_KEY]: JSON.stringify({
      lesson: {
        local: { passed: true, lastPassedAt: '2026-05-25T10:00:00.000Z', checkCount: 2 },
      },
    }),
  });
  const settings = normalizeGitHubSyncSettings({
    enabled: true,
    brokerUrl: 'https://sync.example.com',
    owner: 'octo',
    repo: 'progress',
  });
  let pushCount = 0;
  const fetchFn = async (url, options = {}) => {
    if (options.method === 'PUT') {
      pushCount += 1;
      if (pushCount === 1) return jsonResponse({ error: 'sha changed' }, { ok: false, status: 409 });
      const body = JSON.parse(options.body);
      assert.ok(body.envelope.progress.lesson.local);
      assert.ok(body.envelope.progress.lesson.remote);
      return jsonResponse({ sha: 'final' });
    }

    const remoteId = pushCount === 0 ? 'first' : 'remote';
    return jsonResponse({
      exists: true,
      sha: remoteId,
      envelope: buildProgressEnvelope({
        lesson: {
          [remoteId]: { passed: true, lastPassedAt: '2026-05-25T10:02:00.000Z', checkCount: 1 },
        },
      }, new Date('2026-05-25T10:05:00.000Z')),
    });
  };

  const result = await syncCodeLabProgressToGitHub({
    settings,
    storage,
    fetchFn,
    now: new Date('2026-05-25T10:30:00.000Z'),
  });

  assert.equal(pushCount, 2);
  assert.equal(result.remote.sha, 'final');
});
