package main

import (
	"fmt"
	"math/rand"
	"mini-markov-go/markov"
	"sort"
	"strings"
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║       Mini Markov Go - Musical Chord Progression Demo            ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Fixed seed for reproducibility
	rng := rand.New(rand.NewSource(42))

	// Pop/Rock Model
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("1. POP/ROCK CHORD PROGRESSION MODEL")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()
	fmt.Println("Common chord transitions (in key of C major):")
	fmt.Println("  I=C, ii=Dm, iii=Em, IV=F, V=G, vi=Am, vii°=Bdim")
	fmt.Println()

	pop := markov.NewStateChain()

	// I (Tonic)
	pop.AddTransitionCount("I", "I", 5)
	pop.AddTransitionCount("I", "IV", 35)
	pop.AddTransitionCount("I", "V", 30)
	pop.AddTransitionCount("I", "vi", 25)
	pop.AddTransitionCount("I", "ii", 5)

	// IV (Subdominant)
	pop.AddTransitionCount("IV", "I", 30)
	pop.AddTransitionCount("IV", "V", 40)
	pop.AddTransitionCount("IV", "vi", 15)
	pop.AddTransitionCount("IV", "IV", 10)
	pop.AddTransitionCount("IV", "ii", 5)

	// V (Dominant)
	pop.AddTransitionCount("V", "I", 60)
	pop.AddTransitionCount("V", "vi", 20)
	pop.AddTransitionCount("V", "IV", 15)
	pop.AddTransitionCount("V", "V", 5)

	// vi (Relative minor)
	pop.AddTransitionCount("vi", "IV", 40)
	pop.AddTransitionCount("vi", "V", 25)
	pop.AddTransitionCount("vi", "I", 15)
	pop.AddTransitionCount("vi", "ii", 15)
	pop.AddTransitionCount("vi", "vi", 5)

	// ii (Supertonic)
	pop.AddTransitionCount("ii", "V", 50)
	pop.AddTransitionCount("ii", "IV", 25)
	pop.AddTransitionCount("ii", "I", 15)
	pop.AddTransitionCount("ii", "vi", 10)

	chordMap := map[string]string{
		"I": "C", "ii": "Dm", "iii": "Em", "IV": "F",
		"V": "G", "vi": "Am", "vii°": "Bdim",
	}

	fmt.Println("Generated 8-bar progressions:")
	fmt.Println()
	for i := 1; i <= 5; i++ {
		progression := pop.Simulate("I", 7, rng)
		chords := make([]string, len(progression))
		for j, p := range progression {
			if c, ok := chordMap[p]; ok {
				chords[j] = c
			} else {
				chords[j] = p
			}
		}
		fmt.Printf("  #%d: │ %s │\n", i, strings.Join(chords, " │ "))
	}
	fmt.Println()

	// Learn from famous songs
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("2. LEARNING FROM FAMOUS SONGS")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()

	learned := markov.NewStateChain()

	// Canon in D, Let It Be, With or Without You, etc.
	learned.Train([]string{"I", "V", "vi", "iii", "IV", "I", "IV", "V"})
	learned.Train([]string{"I", "V", "vi", "IV", "I", "V", "vi", "IV"})
	learned.Train([]string{"I", "V", "vi", "IV", "I", "V", "vi", "IV"})
	learned.Train([]string{"I", "V", "vi", "IV", "I", "V", "vi", "IV"})
	learned.Train([]string{"I", "V", "vi", "IV", "I", "V", "vi", "IV"})
	learned.Train([]string{"I", "vi", "IV", "V", "I", "vi", "IV", "V"})
	learned.Train([]string{"I", "IV", "V", "I", "IV", "V", "I", "IV", "V"})
	learned.Train([]string{"I", "vi", "IV", "V", "I", "vi", "IV", "V"})

	fmt.Println("Trained on famous progressions:")
	fmt.Println("  • Canon in D (Pachelbel)")
	fmt.Println("  • Let It Be (Beatles)")
	fmt.Println("  • With or Without You (U2)")
	fmt.Println("  • Someone Like You (Adele)")
	fmt.Println("  • Stand By Me (Ben E. King)")
	fmt.Println("  • Twist and Shout")
	fmt.Println("  • Blue Moon")
	fmt.Println()

	fmt.Println("Learned transition probabilities:")
	for _, state := range []string{"I", "IV", "V", "vi"} {
		probs := learned.ProbabilitiesFrom(state)
		fmt.Printf("  %s →", state)
		for next, prob := range probs {
			if prob > 0.05 {
				fmt.Printf("  %s: %.0f%%", next, prob*100)
			}
		}
		fmt.Println()
	}

	fmt.Println()
	fmt.Println("Generated progressions from learned model:")
	for i := 1; i <= 3; i++ {
		prog := learned.Simulate("I", 7, rng)
		fmt.Printf("  #%d: %s\n", i, strings.Join(prog, " → "))
	}
	fmt.Println()

	// Jazz
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("3. JAZZ CHORD PROGRESSION MODEL")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()

	jazz := markov.NewStateChain()

	// ii-V-I jazz movement
	jazz.AddTransitionCount("iim7", "V7", 70)
	jazz.AddTransitionCount("iim7", "bII7", 15)
	jazz.AddTransitionCount("iim7", "Imaj7", 10)
	jazz.AddTransitionCount("iim7", "vim7", 5)

	jazz.AddTransitionCount("V7", "Imaj7", 70)
	jazz.AddTransitionCount("V7", "vim7", 15)
	jazz.AddTransitionCount("V7", "iim7", 10)
	jazz.AddTransitionCount("V7", "IVmaj7", 5)

	jazz.AddTransitionCount("Imaj7", "iim7", 35)
	jazz.AddTransitionCount("Imaj7", "IVmaj7", 25)
	jazz.AddTransitionCount("Imaj7", "vim7", 20)
	jazz.AddTransitionCount("Imaj7", "iiim7", 10)
	jazz.AddTransitionCount("Imaj7", "Imaj7", 10)

	jazz.AddTransitionCount("vim7", "iim7", 50)
	jazz.AddTransitionCount("vim7", "V7", 25)
	jazz.AddTransitionCount("vim7", "IVmaj7", 15)
	jazz.AddTransitionCount("vim7", "vim7", 10)

	jazz.AddTransitionCount("IVmaj7", "iiim7", 30)
	jazz.AddTransitionCount("IVmaj7", "iim7", 25)
	jazz.AddTransitionCount("IVmaj7", "V7", 25)
	jazz.AddTransitionCount("IVmaj7", "Imaj7", 20)

	jazz.AddTransitionCount("iiim7", "vim7", 50)
	jazz.AddTransitionCount("iiim7", "iim7", 30)
	jazz.AddTransitionCount("iiim7", "IVmaj7", 20)

	jazz.AddTransitionCount("bII7", "Imaj7", 80)
	jazz.AddTransitionCount("bII7", "iim7", 20)

	jazzChordMap := map[string]string{
		"Imaj7": "Cmaj7", "iim7": "Dm7", "iiim7": "Em7", "IVmaj7": "Fmaj7",
		"V7": "G7", "vim7": "Am7", "bII7": "Db7",
	}

	fmt.Println("Jazz chord vocabulary:")
	fmt.Println("  Imaj7=Cmaj7, iim7=Dm7, iiim7=Em7, IVmaj7=Fmaj7")
	fmt.Println("  V7=G7, vim7=Am7, bII7=Db7 (tritone sub)")
	fmt.Println()

	fmt.Println("Generated jazz progressions (8 bars):")
	fmt.Println()
	for i := 1; i <= 4; i++ {
		prog := jazz.Simulate("Imaj7", 7, rng)
		chords := make([]string, len(prog))
		for j, p := range prog {
			if c, ok := jazzChordMap[p]; ok {
				chords[j] = c
			} else {
				chords[j] = p
			}
		}
		fmt.Printf("  #%d: %s\n", i, strings.Join(chords, " → "))
	}
	fmt.Println()

	// Long-term distribution
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("4. LONG-TERM CHORD DISTRIBUTION")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()

	fmt.Println("Pop model stationary distribution:")
	popStationary := pop.StationaryDistribution(1000)

	type kv struct {
		Key   string
		Value float64
	}
	var popSorted []kv
	for k, v := range popStationary {
		popSorted = append(popSorted, kv{k, v})
	}
	sort.Slice(popSorted, func(i, j int) bool {
		return popSorted[i].Value > popSorted[j].Value
	})

	for _, item := range popSorted {
		barLen := int(item.Value * 50)
		bar := strings.Repeat("█", barLen)
		fmt.Printf("  %5s │%s %.1f%%\n", item.Key, bar, item.Value*100)
	}

	fmt.Println()
	fmt.Println("Jazz model stationary distribution:")
	jazzStationary := jazz.StationaryDistribution(1000)

	var jazzSorted []kv
	for k, v := range jazzStationary {
		jazzSorted = append(jazzSorted, kv{k, v})
	}
	sort.Slice(jazzSorted, func(i, j int) bool {
		return jazzSorted[i].Value > jazzSorted[j].Value
	})

	for _, item := range jazzSorted {
		barLen := int(item.Value * 50)
		bar := strings.Repeat("█", barLen)
		fmt.Printf("  %7s │%s %.1f%%\n", item.Key, bar, item.Value*100)
	}

	fmt.Println()
	fmt.Println("════════════════════════════════════════════════════════════════")
	fmt.Println("Key Insights:")
	fmt.Println("  • Pop music heavily favors I, IV, V, vi (the 'four chord' song)")
	fmt.Println("  • Jazz emphasizes ii-V-I motion with extended harmonies")
	fmt.Println("  • Markov chains capture harmonic 'expectations' in music")
	fmt.Println("════════════════════════════════════════════════════════════════")
}
