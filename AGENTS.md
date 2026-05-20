@C:\Users\User\.codex\RTK.md

# Repository Notes

## GitHub Pages Publishing

This repository publishes the unified app manually from the local machine. Do not add or rely on a GitHub Actions Pages workflow for this repo.

Use the repo skill at `.agents/skills/publish-github-pages/SKILL.md` when the user asks to publish, deploy, update, or refresh GitHub Pages.

The manual deploy command is:

```bash
rtk node scripts/deploy-github-pages.mjs
```

The script builds `unified-app` and publishes `unified-app/dist` to the `gh-pages` branch. GitHub Pages should be configured once in repository settings as:

- Source: `Deploy from a branch`
- Branch: `gh-pages`
- Folder: `/ (root)`

## Visual Theme Audit

Use the repo skill at `.agents/skills/visual-theme-audit/SKILL.md` when the user asks to check every screen, capture Playwright screenshots, inspect visual theme consistency, or fix contrast/alignment/button/tab discrepancies.

The audit command is:

```bash
rtk node .agents/skills/visual-theme-audit/scripts/audit-unified-app.mjs
```
