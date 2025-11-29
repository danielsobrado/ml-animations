package main

import (
	"fmt"
	"math/rand"
	"mini-markov-go/markov"
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║       Mini Markov Go - Text Generation Demo                      ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	sherlock := `
		Sherlock Holmes took his bottle from the corner of the mantelpiece and his hypodermic syringe 
		from its neat morocco case. With his long, white, nervous fingers he adjusted the delicate 
		needle, and rolled back his left shirt-cuff. For some little time his eyes rested thoughtfully 
		upon the sinewy forearm and wrist all dotted and scarred with innumerable puncture-marks. 
		Finally he thrust the sharp point home, pressed down the tiny piston, and sank back into the 
		velvet-lined armchair with a long sigh of satisfaction.
	`

	dickens := `
		It was the best of times, it was the worst of times, it was the age of wisdom, it was the 
		age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was 
		the season of Light, it was the season of Darkness, it was the spring of hope, it was the 
		winter of despair. We had everything before us, we had nothing before us, we were all 
		going direct to Heaven, we were all going direct the other way.
	`

	shakespeare := `
		To be, or not to be, that is the question. Whether tis nobler in the mind to suffer the 
		slings and arrows of outrageous fortune, or to take arms against a sea of troubles, and 
		by opposing end them. To die, to sleep, no more, and by a sleep to say we end the heartache, 
		and the thousand natural shocks that flesh is heir to.
	`

	// Fixed seed for reproducibility
	rng := rand.New(rand.NewSource(42))

	// Order 1
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("1. UNIGRAM MODEL (Order 1)")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("Training on Sherlock Holmes excerpt...")
	fmt.Println()

	gen1 := markov.NewTextGenerator(1)
	gen1.Train(sherlock)

	stats := gen1.Stats()
	fmt.Printf("Statistics:\n")
	fmt.Printf("  Unique word contexts: %d\n", stats.NumStates)
	fmt.Printf("  Total transitions: %d\n", stats.NumTransitions)
	fmt.Printf("  Entropy: %.3f bits\n", stats.Entropy)

	fmt.Println("\nGenerated text (30 words):")
	fmt.Println("────────────────────────────────────────────────────────────────")
	fmt.Println(gen1.Generate(30, rng))
	fmt.Println()

	// Order 2
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("2. BIGRAM MODEL (Order 2)")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("Training on Dickens excerpt...")
	fmt.Println()

	gen2 := markov.NewTextGenerator(2)
	gen2.Train(dickens)

	stats = gen2.Stats()
	fmt.Printf("Statistics:\n")
	fmt.Printf("  Unique word pair contexts: %d\n", stats.NumStates)
	fmt.Printf("  Total transitions: %d\n", stats.NumTransitions)
	fmt.Printf("  Entropy: %.3f bits\n", stats.Entropy)

	fmt.Println("\nGenerated text (30 words):")
	fmt.Println("────────────────────────────────────────────────────────────────")
	fmt.Println(gen2.Generate(30, rng))
	fmt.Println()

	// Order 3
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("3. TRIGRAM MODEL (Order 3)")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("Training on Shakespeare excerpt...")
	fmt.Println()

	gen3 := markov.NewTextGenerator(3)
	gen3.Train(shakespeare)

	stats = gen3.Stats()
	fmt.Printf("Statistics:\n")
	fmt.Printf("  Unique word triplet contexts: %d\n", stats.NumStates)
	fmt.Printf("  Total transitions: %d\n", stats.NumTransitions)
	fmt.Printf("  Entropy: %.3f bits\n", stats.Entropy)

	fmt.Println("\nGenerated text (30 words):")
	fmt.Println("────────────────────────────────────────────────────────────────")
	fmt.Println(gen3.Generate(30, rng))
	fmt.Println()

	// Mixed
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("4. MIXED MODEL (Order 2, Multiple Sources)")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("Training on all three texts...")
	fmt.Println()

	genMixed := markov.NewTextGenerator(2)
	genMixed.TrainMany([]string{sherlock, dickens, shakespeare})

	stats = genMixed.Stats()
	fmt.Printf("Statistics:\n")
	fmt.Printf("  Unique word pair contexts: %d\n", stats.NumStates)
	fmt.Printf("  Total transitions: %d\n", stats.NumTransitions)
	fmt.Printf("  Entropy: %.3f bits\n", stats.Entropy)
	fmt.Printf("  Training sequences: %d\n", stats.NumSequences)

	fmt.Println("\nGenerated text (40 words):")
	fmt.Println("────────────────────────────────────────────────────────────────")
	fmt.Println(genMixed.Generate(40, rng))
	fmt.Println()

	// Comparing orders
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("5. COMPARING DIFFERENT N-GRAM ORDERS")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	combinedText := sherlock + " " + dickens + " " + shakespeare

	for order := 1; order <= 4; order++ {
		gen := markov.NewTextGenerator(order)
		gen.Train(combinedText)

		stats := gen.Stats()
		sample := gen.Generate(15, rng)

		fmt.Printf("\nOrder %d: %d states, entropy=%.2f bits\n", order, stats.NumStates, stats.Entropy)
		fmt.Printf("  Sample: %s\n", sample)
	}

	fmt.Println()
	fmt.Println("════════════════════════════════════════════════════════════════")
	fmt.Println("Observation: Higher order = more coherent but less creative")
	fmt.Println("             Lower order = more random but potentially novel")
	fmt.Println("════════════════════════════════════════════════════════════════")
}
