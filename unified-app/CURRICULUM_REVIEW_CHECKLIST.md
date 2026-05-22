# Curriculum Review Checklist

Use this checklist before marking a lesson release-safe.

## Required checks (every sprint)

- Run integrity: `npm test`
- Run route smoke: `npm run test:smoke`
- Regenerate audit: `npm run audit:quality`
- Confirm `unified-app/curriculum-module-audit.md` has no unexpected `D`-tier items in
  `start-here` and `model-reliability-path`.
- Confirm `appFeatures`, `learningPaths`, and `animations` catalogs still satisfy all route/prerequisite contracts.

## Module review for each lesson

For each lesson you touch, complete:

1. Formula/code alignment  
   - Does each displayed formula match implementation?  
   - Are approximations labeled as approximations?  
   - Are assumptions and edge conditions called out?

2. Data realism  
   - Are toy values visibly labeled as toy examples?  
   - Are toy values used only to support intuition, not claim generality?

3. Interaction quality  
   - Is there an observable mechanism for learner action and feedback (not just text)?  
   - Are controls mapped to expected learning outcomes?  
   - Is one clear failure mode demonstrated and explained?

4. Assessment quality  
   - One “predict-before-running” item present.  
   - One “explain the failure mode” item present.  
   - Quiz distractors cover likely misconceptions.

5. Learning sequence safety  
   - Lesson stays on its path prerequisite order.  
   - No prerequisite bypass or broken dependency references.

## Next-action lanes

- `Tier A`: stable, recheck quarterly.
- `Tier B`: stable enough for release, add one misconception-focused micro-case per revision cycle.
- `Tier C`: schedule upgrade in next audit cycle before advancing to release track.
- `Tier D`: block release until upgraded; scaffold or placeholder not acceptable.

## Fast start list for this sprint

1. Finish `Start Here` D/C lessons to `B+` quality where feasible.
2. Expand reliability track placeholders into real modules:
   - `model-debugging`
   - `model-interpretability`
   - `uncertainty-estimation`
   - `model-monitoring`
   - `model-fairness`
