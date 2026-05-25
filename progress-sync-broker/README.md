# GitHub Progress Sync Broker

This optional worker keeps GitHub OAuth secrets and tokens out of the GitHub Pages app.

Required environment variables:

- `GITHUB_CLIENT_ID`
- `GITHUB_CLIENT_SECRET`
- `SESSION_SECRET` with at least 32 random characters
- `PUBLIC_BASE_URL`, for example `https://ml-progress-sync.example.workers.dev`
- `ALLOWED_ORIGINS`, comma-separated, for example `https://danielsobrado.github.io,http://localhost:5173`

Optional:

- `GITHUB_REPO_SCOPES`, default `repo read:user`
- `GITHUB_GIST_SCOPES`, default `gist read:user`
- `COOKIE_NAME`, default `mlap_github_session`

Configure the GitHub OAuth callback URL as:

```text
https://your-broker.example.com/auth/github/callback
```

The SPA stores only target settings and pass metadata. The broker stores the GitHub token in an encrypted, HttpOnly, Secure cookie.
