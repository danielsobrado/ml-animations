---
name: publish-github-pages
description: Publish this repository's Vite unified app to GitHub Pages through the local manual deploy script. Use when the user asks to publish, deploy, update, refresh, or verify GitHub Pages for ml-animations without GitHub workflows.
---

# Publish GitHub Pages

## Scope

Use this skill only inside `F:\Development\workspace\GitHub\ml-animations`.

This repo intentionally uses a manual GitHub Pages deploy flow instead of a GitHub Actions Pages workflow.

## Quick Start

Run the manual deploy script from the repository root:

```bash
rtk node scripts/deploy-github-pages.mjs
```

The script:

- Uses existing `unified-app` dependencies when available.
- Installs dependencies with `npm install --legacy-peer-deps` when needed.
- Runs `npm run build` in `unified-app`.
- Copies `unified-app/dist` into a temporary `.deploy/gh-pages` worktree.
- Adds `.nojekyll` and `404.html`.
- Commits and pushes the built site to `origin/gh-pages`.

## Expected GitHub Settings

GitHub Pages must be configured in the repository UI as:

- Source: `Deploy from a branch`
- Branch: `gh-pages`
- Folder: `/ (root)`

If the live site keeps serving an old build after a successful deploy, check this setting first.

## Verification

After deploy, verify the pushed Pages branch:

```bash
rtk git log -1 --oneline origin/gh-pages
rtk curl.exe -I https://danielsobrado.github.io/ml-animations/
```

The live site may be cached by GitHub Pages for several minutes. If the `Last-Modified` header is still old, wait briefly and retry before changing code.

## Guardrails

- Do not restore `.github/workflows/deploy.yml` unless the user explicitly asks to return to workflow-based publishing.
- Do not commit `reference/`.
- Do not commit `.deploy/`.
- If `unified-app/dist` changes in the feature branch after publishing, treat it as build output from the deploy run unless the user explicitly wants generated dist committed there.
