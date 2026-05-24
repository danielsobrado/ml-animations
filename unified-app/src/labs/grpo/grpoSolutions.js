export const GRPO_SOLUTIONS = {
  total_reward: `def total_reward(candidate, weights):
    """
    candidate fields:
      isCorrect: 0 or 1
      hasValidFormat: 0 or 1
      tokenCount: integer
      languageMixed: 0 or 1

    weights fields:
      correctnessWeight
      formatWeight
      languagePenalty
      lengthPenalty
      idealLength
    """
    length_over = max(0, candidate["tokenCount"] - weights["idealLength"])
    return (
        weights["correctnessWeight"] * candidate["isCorrect"]
        + weights["formatWeight"] * candidate["hasValidFormat"]
        - weights["lengthPenalty"] * length_over
        - weights["languagePenalty"] * candidate["languageMixed"]
    )`,

  group_advantages: `def group_advantages(rewards, eps=1e-8):
    """
    Return normalized advantages:
      A_i = (r_i - mean_reward) / (std_reward + eps)
    """
    mean_reward = sum(rewards) / len(rewards)
    variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
    std_reward = math.sqrt(variance)
    return [(r - mean_reward) / (std_reward + eps) for r in rewards]`,

  update_direction: `def update_direction(advantage):
    """
    Return:
      "up" if advantage > 0.25
      "down" if advantage < -0.25
      "neutral" otherwise
    """
    if advantage > 0.25:
        return "up"
    if advantage < -0.25:
        return "down"
    return "neutral"`,
};

export const GRPO_FULL_SOLUTION = `import math

${GRPO_SOLUTIONS.total_reward}


${GRPO_SOLUTIONS.group_advantages}


${GRPO_SOLUTIONS.update_direction}
`;
