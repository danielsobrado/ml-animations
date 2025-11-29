"""Text generation demo."""

import random
from mini_markov import TextGenerator


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║       Mini Markov Python - Text Generation Demo                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    sherlock = """
    Sherlock Holmes took his bottle from the corner of the mantelpiece and his hypodermic syringe 
    from its neat morocco case. With his long, white, nervous fingers he adjusted the delicate 
    needle, and rolled back his left shirt-cuff. For some little time his eyes rested thoughtfully 
    upon the sinewy forearm and wrist all dotted and scarred with innumerable puncture-marks. 
    Finally he thrust the sharp point home, pressed down the tiny piston, and sank back into the 
    velvet-lined armchair with a long sigh of satisfaction.
    """

    dickens = """
    It was the best of times, it was the worst of times, it was the age of wisdom, it was the 
    age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was 
    the season of Light, it was the season of Darkness, it was the spring of hope, it was the 
    winter of despair. We had everything before us, we had nothing before us, we were all 
    going direct to Heaven, we were all going direct the other way.
    """

    shakespeare = """
    To be, or not to be, that is the question. Whether tis nobler in the mind to suffer the 
    slings and arrows of outrageous fortune, or to take arms against a sea of troubles, and 
    by opposing end them. To die, to sleep, no more, and by a sleep to say we end the heartache, 
    and the thousand natural shocks that flesh is heir to.
    """

    # Fixed seed for reproducibility
    rng = random.Random(42)

    # Order 1
    print("═══════════════════════════════════════════════════════════════")
    print("1. UNIGRAM MODEL (Order 1)")
    print("═══════════════════════════════════════════════════════════════")
    print("Training on Sherlock Holmes excerpt...")
    print()

    gen1 = TextGenerator(1)
    gen1.train(sherlock)

    stats = gen1.stats
    print("Statistics:")
    print(f"  Unique word contexts: {stats['num_states']}")
    print(f"  Total transitions: {stats['num_transitions']}")
    print(f"  Entropy: {stats['entropy']:.3f} bits")

    print("\nGenerated text (30 words):")
    print("────────────────────────────────────────────────────────────────")
    print(gen1.generate(30, rng))
    print()

    # Order 2
    print("═══════════════════════════════════════════════════════════════")
    print("2. BIGRAM MODEL (Order 2)")
    print("═══════════════════════════════════════════════════════════════")
    print("Training on Dickens excerpt...")
    print()

    gen2 = TextGenerator(2)
    gen2.train(dickens)

    stats = gen2.stats
    print("Statistics:")
    print(f"  Unique word pair contexts: {stats['num_states']}")
    print(f"  Total transitions: {stats['num_transitions']}")
    print(f"  Entropy: {stats['entropy']:.3f} bits")

    print("\nGenerated text (30 words):")
    print("────────────────────────────────────────────────────────────────")
    print(gen2.generate(30, rng))
    print()

    # Order 3
    print("═══════════════════════════════════════════════════════════════")
    print("3. TRIGRAM MODEL (Order 3)")
    print("═══════════════════════════════════════════════════════════════")
    print("Training on Shakespeare excerpt...")
    print()

    gen3 = TextGenerator(3)
    gen3.train(shakespeare)

    stats = gen3.stats
    print("Statistics:")
    print(f"  Unique word triplet contexts: {stats['num_states']}")
    print(f"  Total transitions: {stats['num_transitions']}")
    print(f"  Entropy: {stats['entropy']:.3f} bits")

    print("\nGenerated text (30 words):")
    print("────────────────────────────────────────────────────────────────")
    print(gen3.generate(30, rng))
    print()

    # Mixed
    print("═══════════════════════════════════════════════════════════════")
    print("4. MIXED MODEL (Order 2, Multiple Sources)")
    print("═══════════════════════════════════════════════════════════════")
    print("Training on all three texts...")
    print()

    gen_mixed = TextGenerator(2)
    gen_mixed.train_many(sherlock, dickens, shakespeare)

    stats = gen_mixed.stats
    print("Statistics:")
    print(f"  Unique word pair contexts: {stats['num_states']}")
    print(f"  Total transitions: {stats['num_transitions']}")
    print(f"  Entropy: {stats['entropy']:.3f} bits")
    print(f"  Training sequences: {stats['num_sequences']}")

    print("\nGenerated text (40 words):")
    print("────────────────────────────────────────────────────────────────")
    print(gen_mixed.generate(40, rng))
    print()

    # Comparing orders
    print("═══════════════════════════════════════════════════════════════")
    print("5. COMPARING DIFFERENT N-GRAM ORDERS")
    print("═══════════════════════════════════════════════════════════════")

    combined_text = sherlock + " " + dickens + " " + shakespeare

    for order in range(1, 5):
        gen = TextGenerator(order)
        gen.train(combined_text)

        s = gen.stats
        sample = gen.generate(15, rng)

        print(f"\nOrder {order}: {s['num_states']} states, entropy={s['entropy']:.2f} bits")
        print(f"  Sample: {sample}")

    print()
    print("════════════════════════════════════════════════════════════════")
    print("Observation: Higher order = more coherent but less creative")
    print("             Lower order = more random but potentially novel")
    print("════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
