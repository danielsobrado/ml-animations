const DEFAULT_COOKIE_NAME = 'mlap_github_session';
const DEFAULT_REPO_SCOPES = 'repo read:user';
const DEFAULT_GIST_SCOPES = 'gist read:user';
const PROGRESS_FILENAME = 'ml-animations-progress.v1.json';

function json(data, init = {}, corsHeaders = {}) {
  return new Response(JSON.stringify(data), {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...corsHeaders,
      ...(init.headers || {}),
    },
  });
}

function redirect(location, init = {}) {
  return new Response(null, {
    status: 302,
    ...init,
    headers: {
      Location: location,
      ...(init.headers || {}),
    },
  });
}

function envValue(env, name, fallback = '') {
  return String(env[name] || fallback);
}

function textEncoder() {
  return new TextEncoder();
}

function toBase64Url(bytes) {
  const binary = String.fromCharCode(...new Uint8Array(bytes));
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function fromBase64Url(value) {
  const padded = value.replace(/-/g, '+').replace(/_/g, '/').padEnd(Math.ceil(value.length / 4) * 4, '=');
  return Uint8Array.from(atob(padded), (char) => char.charCodeAt(0));
}

async function hmac(secret, value) {
  const key = await crypto.subtle.importKey(
    'raw',
    textEncoder().encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  return crypto.subtle.sign('HMAC', key, textEncoder().encode(value));
}

async function signState(payload, secret) {
  const body = toBase64Url(textEncoder().encode(JSON.stringify(payload)));
  const signature = toBase64Url(await hmac(secret, body));
  return `${body}.${signature}`;
}

async function verifyState(state, secret) {
  const [body, signature] = String(state || '').split('.');
  if (!body || !signature) throw new Error('Invalid OAuth state.');
  const expected = toBase64Url(await hmac(secret, body));
  if (signature !== expected) throw new Error('Invalid OAuth state signature.');
  const payload = JSON.parse(new TextDecoder().decode(fromBase64Url(body)));
  if (Date.now() - Number(payload.createdAt || 0) > 10 * 60 * 1000) {
    throw new Error('OAuth state expired.');
  }
  return payload;
}

async function sessionKey(secret) {
  const digest = await crypto.subtle.digest('SHA-256', textEncoder().encode(secret));
  return crypto.subtle.importKey('raw', digest, { name: 'AES-GCM' }, false, ['encrypt', 'decrypt']);
}

async function sealSession(data, secret) {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const key = await sessionKey(secret);
  const ciphertext = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv },
    key,
    textEncoder().encode(JSON.stringify(data)),
  );
  return `${toBase64Url(iv)}.${toBase64Url(ciphertext)}`;
}

async function openSession(value, secret) {
  const [ivText, cipherText] = String(value || '').split('.');
  if (!ivText || !cipherText) return null;
  const key = await sessionKey(secret);
  const plaintext = await crypto.subtle.decrypt(
    { name: 'AES-GCM', iv: fromBase64Url(ivText) },
    key,
    fromBase64Url(cipherText),
  );
  return JSON.parse(new TextDecoder().decode(plaintext));
}

function parseCookies(request) {
  return Object.fromEntries(
    String(request.headers.get('Cookie') || '')
      .split(';')
      .map((part) => part.trim())
      .filter(Boolean)
      .map((part) => {
        const index = part.indexOf('=');
        return [part.slice(0, index), decodeURIComponent(part.slice(index + 1))];
      }),
  );
}

function corsHeaders(request, env) {
  const origin = request.headers.get('Origin') || '';
  const allowed = allowedOrigins(env);
  const allowOrigin = allowed.includes(origin) ? origin : allowed[0] || '';
  return {
    ...(allowOrigin ? { 'Access-Control-Allow-Origin': allowOrigin } : {}),
    'Access-Control-Allow-Credentials': 'true',
    'Access-Control-Allow-Methods': 'GET,PUT,POST,OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type,Accept',
    Vary: 'Origin',
  };
}

function allowedOrigins(env) {
  return envValue(env, 'ALLOWED_ORIGINS')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
}

function safeReturnTo(value, env) {
  const allowed = allowedOrigins(env);
  const fallback = allowed[0]
    ? `${allowed[0].replace(/\/+$/, '')}/Machine-Learning-Visualized/settings`
    : 'https://danielsobrado.github.io/Machine-Learning-Visualized/settings';
  try {
    const url = new URL(value || fallback);
    if (allowed.includes(url.origin)) return url.toString();
  } catch {
    return fallback;
  }
  return fallback;
}

function cookieHeader(name, value, maxAge) {
  return `${name}=${encodeURIComponent(value)}; HttpOnly; Secure; SameSite=None; Path=/; Max-Age=${maxAge}`;
}

function clearCookieHeader(name) {
  return `${name}=; HttpOnly; Secure; SameSite=None; Path=/; Max-Age=0`;
}

function requireEnv(env) {
  for (const name of ['GITHUB_CLIENT_ID', 'GITHUB_CLIENT_SECRET', 'SESSION_SECRET', 'PUBLIC_BASE_URL']) {
    if (!envValue(env, name)) throw new Error(`Missing ${name}.`);
  }
}

async function exchangeCodeForToken(code, env) {
  const response = await fetch('https://github.com/login/oauth/access_token', {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      client_id: envValue(env, 'GITHUB_CLIENT_ID'),
      client_secret: envValue(env, 'GITHUB_CLIENT_SECRET'),
      code,
      redirect_uri: `${envValue(env, 'PUBLIC_BASE_URL').replace(/\/+$/, '')}/auth/github/callback`,
    }),
  });
  const payload = await response.json();
  if (!response.ok || payload.error || !payload.access_token) {
    throw new Error(payload.error_description || payload.error || 'GitHub token exchange failed.');
  }
  return payload.access_token;
}

async function githubFetch(token, path, options = {}) {
  const response = await fetch(`https://api.github.com${path}`, {
    ...options,
    headers: {
      Accept: 'application/vnd.github+json',
      Authorization: `Bearer ${token}`,
      'User-Agent': 'ml-animations-progress-sync',
      'X-GitHub-Api-Version': '2022-11-28',
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...(options.headers || {}),
    },
  });
  const text = await response.text();
  const payload = text ? JSON.parse(text) : {};
  if (!response.ok) {
    return {
      ok: false,
      status: response.status,
      payload,
      headers: response.headers,
    };
  }
  return {
    ok: true,
    status: response.status,
    payload,
    headers: response.headers,
  };
}

async function requireSession(request, env) {
  const cookieName = envValue(env, 'COOKIE_NAME', DEFAULT_COOKIE_NAME);
  const cookie = parseCookies(request)[cookieName];
  const session = await openSession(cookie, envValue(env, 'SESSION_SECRET'));
  if (!session?.token) throw new Error('Not signed in with GitHub.');
  return session;
}

function encodeContent(value) {
  return btoa(unescape(encodeURIComponent(JSON.stringify(value, null, 2))));
}

function decodeContent(value) {
  return JSON.parse(decodeURIComponent(escape(atob(String(value || '').replace(/\n/g, '')))));
}

function repoProgressPath(params) {
  const owner = params.get('owner');
  const repo = params.get('repo');
  const path = params.get('path') || '.ml-animations/progress.v1.json';
  const ref = params.get('branch') || 'main';
  return `/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/contents/${encodeURIComponent(path).replace(/%2F/g, '/')}?ref=${encodeURIComponent(ref)}`;
}

async function readRepoProgress(token, params) {
  const response = await githubFetch(token, repoProgressPath(params));
  if (!response.ok && response.status === 404) return { exists: false, envelope: {}, sha: '' };
  if (!response.ok) throw new Error(response.payload.message || 'Failed to read repository progress.');
  return {
    exists: true,
    sha: response.payload.sha || '',
    envelope: decodeContent(response.payload.content),
  };
}

async function writeRepoProgress(token, body) {
  const owner = body.owner;
  const repo = body.repo;
  const path = body.path || '.ml-animations/progress.v1.json';
  const apiPath = `/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/contents/${encodeURIComponent(path).replace(/%2F/g, '/')}`;
  const response = await githubFetch(token, apiPath, {
    method: 'PUT',
    body: JSON.stringify({
      message: 'Sync Machine Learning Visualized progress',
      content: encodeContent(body.envelope),
      branch: body.branch || 'main',
      ...(body.sha ? { sha: body.sha } : {}),
    }),
  });
  if (!response.ok) {
    return json({ error: response.payload.message || 'Failed to write repository progress.' }, { status: response.status });
  }
  return json({ sha: response.payload.content?.sha || '' });
}

async function readGistProgress(token, params) {
  const gistId = params.get('gistId');
  const filename = params.get('gistFilename') || PROGRESS_FILENAME;
  if (!gistId) return { exists: false, envelope: {}, gistId: '' };
  const response = await githubFetch(token, `/gists/${encodeURIComponent(gistId)}`);
  if (!response.ok && response.status === 404) return { exists: false, envelope: {}, gistId };
  if (!response.ok) throw new Error(response.payload.message || 'Failed to read Gist progress.');
  const file = response.payload.files?.[filename];
  return {
    exists: Boolean(file),
    gistId,
    envelope: file?.content ? JSON.parse(file.content) : {},
  };
}

async function writeGistProgress(token, body) {
  const filename = body.gistFilename || PROGRESS_FILENAME;
  const payload = {
    files: {
      [filename]: {
        content: JSON.stringify(body.envelope, null, 2),
      },
    },
  };
  const response = await githubFetch(token, body.gistId ? `/gists/${encodeURIComponent(body.gistId)}` : '/gists', {
    method: body.gistId ? 'PATCH' : 'POST',
    body: JSON.stringify(body.gistId ? payload : {
      description: 'Machine Learning Visualized progress',
      public: false,
      ...payload,
    }),
  });
  if (!response.ok) {
    return json({ error: response.payload.message || 'Failed to write Gist progress.' }, { status: response.status });
  }
  return json({ gistId: response.payload.id || body.gistId || '' });
}

async function route(request, env) {
  requireEnv(env);
  const url = new URL(request.url);
  const cors = corsHeaders(request, env);
  const cookieName = envValue(env, 'COOKIE_NAME', DEFAULT_COOKIE_NAME);

  if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: cors });
  if (url.pathname === '/health') return json({ ok: true }, {}, cors);

  if (url.pathname === '/auth/github/start') {
    const target = url.searchParams.get('target') === 'gist' ? 'gist' : 'repo';
    const state = await signState({
      returnTo: safeReturnTo(url.searchParams.get('returnTo'), env),
      createdAt: Date.now(),
    }, envValue(env, 'SESSION_SECRET'));
    const params = new URLSearchParams({
      client_id: envValue(env, 'GITHUB_CLIENT_ID'),
      redirect_uri: `${envValue(env, 'PUBLIC_BASE_URL').replace(/\/+$/, '')}/auth/github/callback`,
      scope: target === 'gist'
        ? envValue(env, 'GITHUB_GIST_SCOPES', DEFAULT_GIST_SCOPES)
        : envValue(env, 'GITHUB_REPO_SCOPES', DEFAULT_REPO_SCOPES),
      state,
    });
    return redirect(`https://github.com/login/oauth/authorize?${params.toString()}`);
  }

  if (url.pathname === '/auth/github/callback') {
    const state = await verifyState(url.searchParams.get('state'), envValue(env, 'SESSION_SECRET'));
    const token = await exchangeCodeForToken(url.searchParams.get('code'), env);
    const sealed = await sealSession({ token, createdAt: Date.now() }, envValue(env, 'SESSION_SECRET'));
    const returnTo = new URL(safeReturnTo(state.returnTo, env));
    returnTo.searchParams.set('githubSync', 'connected');
    return redirect(returnTo.toString(), {
      headers: {
        'Set-Cookie': cookieHeader(cookieName, sealed, 60 * 60 * 24 * 30),
      },
    });
  }

  if (url.pathname === '/auth/github/logout') {
    return json({ ok: true }, {
      headers: {
        ...cors,
        'Set-Cookie': clearCookieHeader(cookieName),
      },
    });
  }

  const session = await requireSession(request, env);

  if (url.pathname === '/api/github/session') {
    const user = await githubFetch(session.token, '/user');
    if (!user.ok) return json({ error: user.payload.message || 'Unable to read GitHub user.' }, { status: user.status }, cors);
    return json({
      user: {
        login: user.payload.login,
        avatarUrl: user.payload.avatar_url,
        htmlUrl: user.payload.html_url,
      },
    }, {}, cors);
  }

  if (url.pathname === '/api/github/repos') {
    const repos = await githubFetch(session.token, '/user/repos?per_page=100&sort=updated&type=all');
    if (!repos.ok) return json({ error: repos.payload.message || 'Unable to list repositories.' }, { status: repos.status }, cors);
    return json({
      repos: repos.payload.map((repo) => ({
        fullName: repo.full_name,
        owner: repo.owner?.login,
        name: repo.name,
        private: repo.private === true,
        defaultBranch: repo.default_branch,
        permissions: repo.permissions || {},
      })),
    }, {}, cors);
  }

  if (url.pathname === '/api/github/progress' && request.method === 'GET') {
    const result = url.searchParams.get('target') === 'gist'
      ? await readGistProgress(session.token, url.searchParams)
      : await readRepoProgress(session.token, url.searchParams);
    return json(result, {}, cors);
  }

  if (url.pathname === '/api/github/progress' && request.method === 'PUT') {
    const body = await request.json();
    const response = body.target === 'gist'
      ? await writeGistProgress(session.token, body)
      : await writeRepoProgress(session.token, body);
    response.headers.set('Access-Control-Allow-Credentials', cors['Access-Control-Allow-Credentials']);
    if (cors['Access-Control-Allow-Origin']) response.headers.set('Access-Control-Allow-Origin', cors['Access-Control-Allow-Origin']);
    response.headers.set('Vary', 'Origin');
    return response;
  }

  return json({ error: 'Not found.' }, { status: 404 }, cors);
}

export default {
  async fetch(request, env) {
    try {
      return await route(request, env);
    } catch (error) {
      return json({ error: error.message || 'Progress sync broker failed.' }, { status: 400 }, corsHeaders(request, env));
    }
  },
};
