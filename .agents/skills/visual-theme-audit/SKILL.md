---
name: visual-theme-audit
description: Audit ml-animations unified-app visual surfaces with Playwright screenshots. Use when the user asks to inspect every screen, check theme consistency, review contrast/alignment/buttons/tabs, capture screenshots, or fix Distill styling discrepancies across the catalog.
---

# Visual Theme Audit

Use this skill for repo-local visual QA of the `unified-app` Distill restyle.

## Workflow

1. Run the screenshot audit script from the repo root:

   ```bash
   rtk node .agents/skills/visual-theme-audit/scripts/audit-unified-app.mjs
   ```

2. Inspect the generated output under `screenshots/theme-audit/<timestamp>/`:
   - `manifest.json` lists every captured screen and any automated theme findings.
   - `home/` contains home page desktop/mobile captures.
   - `animations/<animation-id>/` contains one capture for the default screen and one per visible tab.

3. Review screenshots in batches. Prioritize:
   - buttons using bright Tailwind colors instead of `--ds-ink`, `--ds-accent`, `--ds-paper`, or hairline ghost styles
   - gradient tabs, rounded pill chrome, heavy shadows, dark panels, white-on-saturated cards
   - misaligned two-column labs, charts clipped out of panels, labels outside SVG/canvas bounds
   - text contrast that is too faint on the paper background
   - mobile screenshots where header/sidebar/page content overlap

4. Patch the source, not generated `dist`. Prefer:
   - shared CSS bridge fixes in `unified-app/src/index.css` for repeated legacy patterns
   - focused component CSS/classes for one-off visualization layouts
   - `Tabs`, `Btn`, `ParamSlider`, `Figure`, `Aside`, and Distill token variables when touching components

5. Re-run:

   ```bash
   rtk npm run build --prefix unified-app
   rtk node .agents/skills/visual-theme-audit/scripts/audit-unified-app.mjs
   ```

6. Commit source and skill changes. If the user expects GitHub Pages to update, run:

   ```bash
   rtk node scripts/deploy-github-pages.mjs
   ```

## Notes

- The script requires `playwright` in `unified-app` dev dependencies. It uses the local Chrome channel by default; set `THEME_AUDIT_CHANNEL=msedge` or `THEME_AUDIT_CHANNEL=chromium` if needed.
- Keep generated screenshots uncommitted unless the user asks for snapshot artifacts.
- Treat automated findings as triage hints; visual inspection decides whether a screen is acceptable.
