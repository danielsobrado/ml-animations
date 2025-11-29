package main

import (
	"fmt"
	"math/rand"
	"mini-markov-go/markov"
	"strings"
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║       Mini Markov Go - Weather State Machine Demo                ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Fixed seed for reproducibility
	rng := rand.New(rand.NewSource(42))

	// Simple Weather Model
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("1. SIMPLE WEATHER MODEL (3 States)")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()
	fmt.Println("States: Sunny ☀️, Cloudy ☁️, Rainy 🌧️")
	fmt.Println()
	fmt.Println("Transition probabilities (from historical data):")
	fmt.Println("  Sunny  → Sunny: 70%  Cloudy: 20%  Rainy: 10%")
	fmt.Println("  Cloudy → Sunny: 30%  Cloudy: 40%  Rainy: 30%")
	fmt.Println("  Rainy  → Sunny: 20%  Cloudy: 40%  Rainy: 40%")
	fmt.Println()

	weather := markov.NewStateChain()
	weather.WithStates([]string{"sunny", "cloudy", "rainy"})

	weather.AddTransitionCount("sunny", "sunny", 70)
	weather.AddTransitionCount("sunny", "cloudy", 20)
	weather.AddTransitionCount("sunny", "rainy", 10)

	weather.AddTransitionCount("cloudy", "sunny", 30)
	weather.AddTransitionCount("cloudy", "cloudy", 40)
	weather.AddTransitionCount("cloudy", "rainy", 30)

	weather.AddTransitionCount("rainy", "sunny", 20)
	weather.AddTransitionCount("rainy", "cloudy", 40)
	weather.AddTransitionCount("rainy", "rainy", 40)

	fmt.Println("Verification - Computed probabilities:")
	for _, state := range []string{"sunny", "cloudy", "rainy"} {
		probs := weather.ProbabilitiesFrom(state)
		fmt.Printf("  %s →", state)
		for _, next := range []string{"sunny", "cloudy", "rainy"} {
			fmt.Printf("  %s: %.0f%%", next, probs[next]*100)
		}
		fmt.Println()
	}
	fmt.Println()

	// Weekly simulation
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("2. WEEKLY WEATHER SIMULATION")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()

	days := []string{"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}

	for _, startWeather := range []string{"sunny", "cloudy", "rainy"} {
		fmt.Printf("Starting: %s on Monday\n", startWeather)
		forecast := weather.Simulate(startWeather, 6, rng)

		fmt.Print("  ")
		for i, state := range forecast {
			emoji := "❓"
			switch state {
			case "sunny":
				emoji = "☀️"
			case "cloudy":
				emoji = "☁️"
			case "rainy":
				emoji = "🌧️"
			}
			fmt.Printf("%s: %s ", days[i], emoji)
		}
		fmt.Println()
		fmt.Println()
	}

	// Stationary Distribution
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("3. STATIONARY DISTRIBUTION (Long-term Probabilities)")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()

	stationary := weather.StationaryDistribution(1000)

	fmt.Println("After infinite time, regardless of starting state:")
	for _, state := range []string{"sunny", "cloudy", "rainy"} {
		prob := stationary[state]
		emoji := "❓"
		switch state {
		case "sunny":
			emoji = "☀️"
		case "cloudy":
			emoji = "☁️"
		case "rainy":
			emoji = "🌧️"
		}
		barLen := int(prob * 50)
		bar := strings.Repeat("█", barLen)
		fmt.Printf("  %s %s %.1f%% %s\n", emoji, strings.Title(state), prob*100, bar)
	}

	// Verify by simulation
	fmt.Println()
	fmt.Println("Verification by simulation (10,000 steps from 'sunny'):")
	longSim := weather.Simulate("sunny", 10000, rng)
	counts := make(map[string]int)
	for _, state := range longSim {
		counts[state]++
	}
	total := len(longSim)
	for _, state := range []string{"sunny", "cloudy", "rainy"} {
		fmt.Printf("  %s: %.1f%%\n", state, float64(counts[state])/float64(total)*100)
	}
	fmt.Println()

	// Expected hitting times
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("4. EXPECTED HITTING TIMES")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()

	fmt.Println("Average days until sunny (from 1000 simulations):")
	for _, start := range []string{"cloudy", "rainy"} {
		expected := weather.ExpectedStepsTo(start, "sunny", 365, 1000, rng)
		if expected < 0 {
			fmt.Printf("  From %s: unreachable\n", start)
		} else {
			fmt.Printf("  From %s: %.1f days\n", start, expected)
		}
	}

	fmt.Println()
	fmt.Println("Average days until rainy (from 1000 simulations):")
	for _, start := range []string{"sunny", "cloudy"} {
		expected := weather.ExpectedStepsTo(start, "rainy", 365, 1000, rng)
		if expected < 0 {
			fmt.Printf("  From %s: unreachable\n", start)
		} else {
			fmt.Printf("  From %s: %.1f days\n", start, expected)
		}
	}
	fmt.Println()

	// Learning from data
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("5. LEARNING FROM OBSERVED DATA")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()

	historicalData := [][]string{
		{"sunny", "sunny", "cloudy", "rainy", "rainy", "cloudy", "sunny"},
		{"rainy", "rainy", "cloudy", "cloudy", "sunny", "sunny", "sunny"},
		{"cloudy", "sunny", "sunny", "sunny", "cloudy", "rainy", "rainy"},
		{"sunny", "cloudy", "cloudy", "rainy", "cloudy", "sunny", "sunny"},
	}

	learnedWeather := markov.NewStateChain()
	learnedWeather.TrainMany(historicalData)

	fmt.Println("Learned transition probabilities from 4 weeks of data:")
	for _, state := range []string{"sunny", "cloudy", "rainy"} {
		probs := learnedWeather.ProbabilitiesFrom(state)
		fmt.Printf("  %s →", state)
		for _, next := range []string{"sunny", "cloudy", "rainy"} {
			fmt.Printf("  %s: %.0f%%", next, probs[next]*100)
		}
		fmt.Println()
	}

	fmt.Println()
	fmt.Println("════════════════════════════════════════════════════════════════")
	fmt.Println("Key Insights:")
	fmt.Println("  • Markov chains can model any system with discrete states")
	fmt.Println("  • The stationary distribution shows long-term behavior")
	fmt.Println("  • Can be learned from observed data or specified directly")
	fmt.Println("════════════════════════════════════════════════════════════════")
}
