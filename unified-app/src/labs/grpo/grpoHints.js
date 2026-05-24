export const GRPO_HINTS = {
  total_reward: [
    {
      title: 'Concept',
      body: 'Start from zero. Add positive terms for desired behavior and subtract penalties for undesired behavior.',
    },
    {
      title: 'Formula',
      body: 'Use correctnessWeight x isCorrect, formatWeight x hasValidFormat, subtract lengthPenalty x max(0, tokenCount - idealLength), and subtract languagePenalty x languageMixed.',
    },
    {
      title: 'Near-code',
      body: `length_over = max(0, candidate["tokenCount"] - weights["idealLength"])
reward = (
    weights["correctnessWeight"] * candidate["isCorrect"]
    + weights["formatWeight"] * candidate["hasValidFormat"]
    - weights["lengthPenalty"] * length_over
    - weights["languagePenalty"] * candidate["languageMixed"]
)`,
      code: true,
    },
  ],
  group_advantages: [
    {
      title: 'Concept',
      body: 'GRPO compares each candidate against the group baseline for the same prompt.',
    },
    {
      title: 'Formula',
      body: 'Compute mean reward, compute standard deviation, then normalize each reward by subtracting the mean and dividing by std + eps.',
    },
    {
      title: 'Near-code',
      body: `mean = sum(rewards) / len(rewards)
var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
std = math.sqrt(var)
return [(r - mean) / (std + eps) for r in rewards]`,
      code: true,
    },
  ],
  update_direction: [
    {
      title: 'Concept',
      body: 'Positive advantage means the trace is better than the group baseline. Negative advantage means worse than the group baseline.',
    },
    {
      title: 'Formula',
      body: 'Use the thresholds: > 0.25 means up, < -0.25 means down, otherwise neutral.',
    },
    {
      title: 'Near-code',
      body: `if advantage > 0.25:
    return "up"
if advantage < -0.25:
    return "down"
return "neutral"`,
      code: true,
    },
  ],
};
