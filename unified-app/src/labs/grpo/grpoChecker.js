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

def list_matches(actual, expected):
    if not isinstance(actual, list):
        return False
    if len(actual) != len(expected):
        return False
    return all(approx(a, b) for a, b in zip(actual, expected))

def direction_feedback(actual, expected):
    if not isinstance(actual, list):
        return "update_direction did not produce a list of directions when mapped over advantages."
    if actual == expected:
        return "Your update_direction thresholds match the expected policy-pressure labels."
    if all(d == "neutral" for d in actual):
        return "Your update directions are all neutral. Did you implement the > 0.25 and < -0.25 branches?"
    return "Check the thresholds: advantage > 0.25 should be up, advantage < -0.25 should be down, otherwise neutral."

def reward_feedback(actual, expected):
    if not isinstance(actual, list):
        return "total_reward did not return a numeric reward for each candidate."
    if all(approx(r, 0.0) for r in actual):
        return "Your rewards are all zero. Did total_reward still return 0.0?"
    if list_matches(actual, expected):
        return "Your reward formula matches the expected calculation."
    return "Compare each term: correctness and format add reward; extra length and language mixing subtract reward."

def advantage_feedback(actual, expected):
    if not isinstance(actual, list):
        return "group_advantages should return one normalized advantage per reward."
    if len(actual) != len(expected):
        return "group_advantages should return the same number of values as the rewards list."
    if list_matches(actual, expected):
        return "Your advantages match group-relative normalization."
    if all(approx(a, 0.0) for a in actual):
        return "Your advantages are all zero. Did group_advantages still return the starter list?"
    if not approx(sum(actual), 0.0, 1e-5):
        return "Your advantages do not sum close to zero. Did you subtract the group mean?"
    return "Check the standard deviation term: divide by std + eps after subtracting the mean."

checks = []
checks.append({
    "id": "total_reward",
    "label": "Rewards match reference",
    "passed": list_matches(rewards, ref_rewards),
    "feedback": reward_feedback(rewards, ref_rewards),
})
checks.append({
    "id": "group_advantages",
    "label": "Advantages are group-normalized",
    "passed": list_matches(advantages, ref_advantages),
    "feedback": advantage_feedback(advantages, ref_advantages),
})
checks.append({
    "id": "update_direction",
    "label": "Directions match advantage thresholds",
    "passed": directions == ref_directions,
    "feedback": direction_feedback(directions, ref_directions),
})
checks.append({
    "id": "sanity",
    "label": "Advantages sum close to zero",
    "passed": approx(sum(advantages), 0.0, 1e-5),
    "feedback": "Normalized advantages should center around zero because every reward is compared with the group mean.",
})

best_index = max(range(len(ref_advantages)), key=lambda i: (ref_advantages[i], -candidates[i]["tokenCount"]))
worst_index = min(range(len(ref_advantages)), key=lambda i: ref_advantages[i])
best = candidates[best_index]
worst = candidates[worst_index]

insights = [
    f"Candidate {best['id']} receives the strongest positive update because it beats this group's reward baseline.",
    "Candidate H is correct but loses reward to the length penalty when overthinking is expensive.",
    "Candidate G is correct but drops when the language-mixing penalty rises.",
]

if weights["formatWeight"] >= 0.5 and weights["correctnessWeight"] <= 0.2:
    insights.append("Candidate B still earns format reward even though it is wrong, which is the reward-hacking risk in this preset.")

if weights["lengthPenalty"] >= 0.03:
    insights.append("Candidate H falls sharply when length penalty is high, demonstrating the overthinking penalty.")

if weights["languagePenalty"] >= 0.5:
    insights.append("Candidate G drops when language mixing is expensive, even though the answer is mathematically correct.")

print(json.dumps({
    "rewards": rewards,
    "advantages": advantages,
    "directions": directions,
    "groupMean": ref_mean,
    "groupStd": ref_std,
    "checks": checks,
    "insights": insights,
    "strongestPositiveId": best["id"],
    "strongestNegativeId": worst["id"],
    "passed": all(c["passed"] for c in checks)
}))
`;
}
