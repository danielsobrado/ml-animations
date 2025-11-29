"""Music chord progression demo."""

import random
from mini_markov import StateChain


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║       Mini Markov Python - Music Chord Progression Demo          ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    # Fixed seed for reproducibility
    rng = random.Random(42)

    # Pop/Rock Progressions
    print("═══════════════════════════════════════════════════════════════")
    print("1. POP/ROCK CHORD PROGRESSIONS")
    print("═══════════════════════════════════════════════════════════════")
    print()
    print("Training on common pop progressions:")
    print("  I-V-vi-IV (Let It Be, No Woman No Cry)")
    print("  I-IV-V-IV (La Bamba, Twist and Shout)")
    print("  vi-IV-I-V (Despacito, Someone Like You)")
    print()

    pop = StateChain()

    # I-V-vi-IV repeated
    for _ in range(10):
        pop.train(["I", "V", "vi", "IV"])

    # I-IV-V-IV
    for _ in range(8):
        pop.train(["I", "IV", "V", "IV"])

    # vi-IV-I-V
    for _ in range(7):
        pop.train(["vi", "IV", "I", "V"])

    print("Learned transitions:")
    for chord in ["I", "IV", "V", "vi"]:
        probs = pop.probabilities_from(chord)
        if probs:
            line = f"  {chord} →"
            for next_chord in sorted(probs.keys()):
                line += f"  {next_chord}: {probs[next_chord]:.0%}"
            print(line)
    print()

    print("Generated progressions (8 bars each):")
    print("────────────────────────────────────────────────────────────────")
    for i in range(3):
        progression = pop.simulate("I", 7, rng)
        print(f"  {i+1}. {' - '.join(progression)}")
    print()

    # Stationary distribution
    stationary = pop.stationary_distribution(1000)
    print("Chord frequency in generated music (stationary distribution):")
    for chord in sorted(stationary.keys(), key=lambda x: stationary[x], reverse=True):
        prob = stationary[chord]
        bar = "█" * int(prob * 40)
        print(f"  {chord:3} {prob:.0%} {bar}")
    print()

    # Jazz Progressions
    print("═══════════════════════════════════════════════════════════════")
    print("2. JAZZ CHORD PROGRESSIONS")
    print("═══════════════════════════════════════════════════════════════")
    print()
    print("Training on jazz standards:")
    print("  ii-V-I (most common jazz turnaround)")
    print("  I-vi-ii-V (rhythm changes)")
    print("  iii-vi-ii-V-I")
    print()

    jazz = StateChain()

    # ii-V-I (very common)
    for _ in range(15):
        jazz.train(["ii", "V", "I"])

    # I-vi-ii-V
    for _ in range(8):
        jazz.train(["I", "vi", "ii", "V"])

    # iii-vi-ii-V-I
    for _ in range(5):
        jazz.train(["iii", "vi", "ii", "V", "I"])

    print("Learned transitions:")
    for chord in ["I", "ii", "iii", "V", "vi"]:
        probs = jazz.probabilities_from(chord)
        if probs:
            line = f"  {chord:4} →"
            for next_chord in sorted(probs.keys()):
                line += f"  {next_chord}: {probs[next_chord]:.0%}"
            print(line)
    print()

    print("Generated jazz progressions (8 bars each):")
    print("────────────────────────────────────────────────────────────────")
    for i in range(3):
        progression = jazz.simulate("ii", 7, rng)
        print(f"  {i+1}. {' - '.join(progression)}")
    print()

    # Blues Progressions
    print("═══════════════════════════════════════════════════════════════")
    print("3. BLUES PROGRESSIONS")
    print("═══════════════════════════════════════════════════════════════")
    print()

    blues = StateChain()

    # 12-bar blues: I-I-I-I-IV-IV-I-I-V-IV-I-V
    for _ in range(10):
        blues.train(["I", "I", "I", "I", "IV", "IV", "I", "I", "V", "IV", "I", "V"])

    # Quick change blues: I-IV-I-I-IV-IV-I-I-V-IV-I-V
    for _ in range(5):
        blues.train(["I", "IV", "I", "I", "IV", "IV", "I", "I", "V", "IV", "I", "V"])

    print("12-bar blues learned transitions:")
    for chord in ["I", "IV", "V"]:
        probs = blues.probabilities_from(chord)
        if probs:
            line = f"  {chord} →"
            for next_chord in sorted(probs.keys()):
                line += f"  {next_chord}: {probs[next_chord]:.0%}"
            print(line)
    print()

    print("Generated blues progressions:")
    print("────────────────────────────────────────────────────────────────")
    for i in range(2):
        progression = blues.simulate("I", 11, rng)
        # Format as 3 lines of 4 bars
        print(f"  Progression {i+1}:")
        for row in range(3):
            start = row * 4
            end = min(start + 4, len(progression))
            print(f"    Bars {row*4+1:2}-{row*4+4:2}: {' - '.join(progression[start:end])}")
        print()

    # Summary
    print("════════════════════════════════════════════════════════════════")
    print("Key Insights:")
    print("  • Pop music favors predictable, cyclic progressions")
    print("  • Jazz emphasizes ii-V-I movement (circle of fifths)")
    print("  • Blues has strong I-IV-V structure with specific patterns")
    print("  • Markov chains capture style-specific tendencies!")
    print("════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
