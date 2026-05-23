export function buildGrpoCheckScript(userCode, candidates, weights) {
  return `
import json, math

candidates = ${JSON.stringify(candidates)}
weights = ${JSON.stringify(weights)}

${userCode}

rewards = [total_reward(c, weights) for c in candidates]
advantages = group_advantages(rewards)
directions = [update_direction(a) for a in advantages]

def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol

def ref_total_reward(candidate, weights):
    length_over = max(0, candidate["tokenCount"] - weights["idealLength"])
    return (
        weights["correctnessWeight"] * candidate["isCorrect"]
        + weights["formatWeight"] * candidate["hasValidFormat"]
        - weights["lengthPenalty"] * length_over
        - weights["languagePenalty"] * candidate["languageMixed"]
    )

ref_rewards = [ref_total_reward(c, weights) for c in candidates]
ref_mean = sum(ref_rewards) / len(ref_rewards)
ref_var = sum((r - ref_mean) ** 2 for r in ref_rewards) / len(ref_rewards)
ref_std = math.sqrt(ref_var)
ref_advantages = [(r - ref_mean) / (ref_std + 1e-8) for r in ref_rewards]

def ref_direction(a):
    if a > 0.25:
        return "up"
    if a < -0.25:
        return "down"
    return "neutral"

ref_directions = [ref_direction(a) for a in ref_advantages]

checks = []
checks.append({
    "name": "rewards match reference",
    "passed": all(approx(a, b) for a, b in zip(rewards, ref_rewards)),
})
checks.append({
    "name": "advantages are group-normalized",
    "passed": all(approx(a, b) for a, b in zip(advantages, ref_advantages)),
})
checks.append({
    "name": "directions match advantage thresholds",
    "passed": directions == ref_directions,
})
checks.append({
    "name": "advantages sum close to zero",
    "passed": approx(sum(advantages), 0.0, 1e-5),
})

print(json.dumps({
    "rewards": rewards,
    "advantages": advantages,
    "directions": directions,
    "checks": checks,
    "passed": all(c["passed"] for c in checks)
}))
`;
}
