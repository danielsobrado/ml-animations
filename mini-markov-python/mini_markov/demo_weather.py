"""Weather state machine demo."""

import random
from mini_markov import StateChain


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║       Mini Markov Python - Weather State Machine Demo            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    # Fixed seed for reproducibility
    rng = random.Random(42)

    # Simple Weather Model
    print("═══════════════════════════════════════════════════════════════")
    print("1. SIMPLE WEATHER MODEL (3 States)")
    print("═══════════════════════════════════════════════════════════════")
    print()
    print("States: Sunny ☀️, Cloudy ☁️, Rainy 🌧️")
    print()
    print("Transition probabilities (from historical data):")
    print("  Sunny  → Sunny: 70%  Cloudy: 20%  Rainy: 10%")
    print("  Cloudy → Sunny: 30%  Cloudy: 40%  Rainy: 30%")
    print("  Rainy  → Sunny: 20%  Cloudy: 40%  Rainy: 40%")
    print()

    weather = StateChain()
    weather.with_states("sunny", "cloudy", "rainy")

    weather.add_transition_count("sunny", "sunny", 70)
    weather.add_transition_count("sunny", "cloudy", 20)
    weather.add_transition_count("sunny", "rainy", 10)

    weather.add_transition_count("cloudy", "sunny", 30)
    weather.add_transition_count("cloudy", "cloudy", 40)
    weather.add_transition_count("cloudy", "rainy", 30)

    weather.add_transition_count("rainy", "sunny", 20)
    weather.add_transition_count("rainy", "cloudy", 40)
    weather.add_transition_count("rainy", "rainy", 40)

    print("Verification - Computed probabilities:")
    for state in ["sunny", "cloudy", "rainy"]:
        probs = weather.probabilities_from(state)
        line = f"  {state} →"
        for next_state in ["sunny", "cloudy", "rainy"]:
            line += f"  {next_state}: {probs.get(next_state, 0):.0%}"
        print(line)
    print()

    # Weekly simulation
    print("═══════════════════════════════════════════════════════════════")
    print("2. WEEKLY WEATHER SIMULATION")
    print("═══════════════════════════════════════════════════════════════")
    print()

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    emojis = {"sunny": "☀️", "cloudy": "☁️", "rainy": "🌧️"}

    for start_weather in ["sunny", "cloudy", "rainy"]:
        print(f"Starting: {start_weather} on Monday")
        forecast = weather.simulate(start_weather, 6, rng)

        line = "  "
        for i, w in enumerate(forecast):
            line += f"{days[i]}: {emojis.get(w, '❓')} "
        print(line)
        print()

    # Stationary Distribution
    print("═══════════════════════════════════════════════════════════════")
    print("3. STATIONARY DISTRIBUTION (Long-term Probabilities)")
    print("═══════════════════════════════════════════════════════════════")
    print()

    stationary = weather.stationary_distribution(1000)

    print("After infinite time, regardless of starting state:")
    for state in ["sunny", "cloudy", "rainy"]:
        prob = stationary.get(state, 0.0)
        emoji = emojis.get(state, "❓")
        bar = "█" * int(prob * 50)
        cap_state = state.capitalize()
        print(f"  {emoji} {cap_state} {prob:.1%} {bar}")

    # Verify by simulation
    print()
    print("Verification by simulation (10,000 steps from 'sunny'):")
    long_sim = weather.simulate("sunny", 10000, rng)
    counts = {}
    for state in long_sim:
        counts[state] = counts.get(state, 0) + 1
    total = len(long_sim)
    for state in ["sunny", "cloudy", "rainy"]:
        print(f"  {state}: {counts.get(state, 0) / total:.1%}")
    print()

    # Expected hitting times
    print("═══════════════════════════════════════════════════════════════")
    print("4. EXPECTED HITTING TIMES")
    print("═══════════════════════════════════════════════════════════════")
    print()

    print("Average days until sunny (from 1000 simulations):")
    for start in ["cloudy", "rainy"]:
        expected = weather.expected_steps_to(start, "sunny", 365, 1000, rng)
        if expected < 0:
            print(f"  From {start}: unreachable")
        else:
            print(f"  From {start}: {expected:.1f} days")

    print()
    print("Average days until rainy (from 1000 simulations):")
    for start in ["sunny", "cloudy"]:
        expected = weather.expected_steps_to(start, "rainy", 365, 1000, rng)
        if expected < 0:
            print(f"  From {start}: unreachable")
        else:
            print(f"  From {start}: {expected:.1f} days")
    print()

    # Learning from data
    print("═══════════════════════════════════════════════════════════════")
    print("5. LEARNING FROM OBSERVED DATA")
    print("═══════════════════════════════════════════════════════════════")
    print()

    historical_data = [
        ["sunny", "sunny", "cloudy", "rainy", "rainy", "cloudy", "sunny"],
        ["rainy", "rainy", "cloudy", "cloudy", "sunny", "sunny", "sunny"],
        ["cloudy", "sunny", "sunny", "sunny", "cloudy", "rainy", "rainy"],
        ["sunny", "cloudy", "cloudy", "rainy", "cloudy", "sunny", "sunny"],
    ]

    learned_weather = StateChain()
    learned_weather.train_many(historical_data)

    print("Learned transition probabilities from 4 weeks of data:")
    for state in ["sunny", "cloudy", "rainy"]:
        probs = learned_weather.probabilities_from(state)
        line = f"  {state} →"
        for next_state in ["sunny", "cloudy", "rainy"]:
            line += f"  {next_state}: {probs.get(next_state, 0):.0%}"
        print(line)

    print()
    print("════════════════════════════════════════════════════════════════")
    print("Key Insights:")
    print("  • Markov chains can model any system with discrete states")
    print("  • The stationary distribution shows long-term behavior")
    print("  • Can be learned from observed data or specified directly")
    print("════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
