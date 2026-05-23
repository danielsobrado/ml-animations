export const GRPO_STARTER_CODE = `import math

def total_reward(candidate, weights):
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
    # TODO:
    # 1. reward correctness
    # 2. reward valid format
    # 3. penalize tokens beyond idealLength
    # 4. penalize language mixing
    return 0.0


def group_advantages(rewards, eps=1e-8):
    """
    Return normalized advantages:
      A_i = (r_i - mean_reward) / (std_reward + eps)
    """
    # TODO:
    return [0.0 for _ in rewards]


def update_direction(advantage):
    """
    Return:
      "up" if advantage > 0.25
      "down" if advantage < -0.25
      "neutral" otherwise
    """
    # TODO:
    return "neutral"
`;
