package com.minimarkov.demo;

import com.minimarkov.StateChain;
import java.util.*;

public class DemoWeather {

    private static String getEmoji(String state) {
        if ("sunny".equals(state)) return "[sun]";
        if ("cloudy".equals(state)) return "[cloud]";
        if ("rainy".equals(state)) return "[rain]";
        return "?";
    }

    public static void main(String[] args) {
        System.out.println("+==================================================================+");
        System.out.println("|       Mini Markov Java - Weather State Machine Demo             |");
        System.out.println("+==================================================================+");
        System.out.println();

        // Fixed seed for reproducibility
        Random rng = new Random(42);

        // Simple Weather Model
        System.out.println("===================================================================");
        System.out.println("1. SIMPLE WEATHER MODEL (3 States)");
        System.out.println("===================================================================");
        System.out.println();
        System.out.println("States: Sunny, Cloudy, Rainy");
        System.out.println();
        System.out.println("Transition probabilities (from historical data):");
        System.out.println("  Sunny  -> Sunny: 70%  Cloudy: 20%  Rainy: 10%");
        System.out.println("  Cloudy -> Sunny: 30%  Cloudy: 40%  Rainy: 30%");
        System.out.println("  Rainy  -> Sunny: 20%  Cloudy: 40%  Rainy: 40%");
        System.out.println();

        StateChain weather = new StateChain();
        weather.withStates("sunny", "cloudy", "rainy");

        weather.addTransitionCount("sunny", "sunny", 70);
        weather.addTransitionCount("sunny", "cloudy", 20);
        weather.addTransitionCount("sunny", "rainy", 10);

        weather.addTransitionCount("cloudy", "sunny", 30);
        weather.addTransitionCount("cloudy", "cloudy", 40);
        weather.addTransitionCount("cloudy", "rainy", 30);

        weather.addTransitionCount("rainy", "sunny", 20);
        weather.addTransitionCount("rainy", "cloudy", 40);
        weather.addTransitionCount("rainy", "rainy", 40);

        System.out.println("Verification - Computed probabilities:");
        for (String state : Arrays.asList("sunny", "cloudy", "rainy")) {
            Map<String, Double> probs = weather.probabilitiesFrom(state);
            System.out.printf("  %s ->", state);
            for (String next : Arrays.asList("sunny", "cloudy", "rainy")) {
                System.out.printf("  %s: %.0f%%", next, probs.getOrDefault(next, 0.0) * 100);
            }
            System.out.println();
        }
        System.out.println();

        // Weekly simulation
        System.out.println("===================================================================");
        System.out.println("2. WEEKLY WEATHER SIMULATION");
        System.out.println("===================================================================");
        System.out.println();

        String[] days = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};

        for (String startWeather : Arrays.asList("sunny", "cloudy", "rainy")) {
            System.out.printf("Starting: %s on Monday%n", startWeather);
            List<String> forecast = weather.simulate(startWeather, 6, rng);

            System.out.print("  ");
            for (int i = 0; i < forecast.size(); i++) {
                System.out.printf("%s: %s ", days[i], getEmoji(forecast.get(i)));
            }
            System.out.println();
            System.out.println();
        }

        // Stationary Distribution
        System.out.println("===================================================================");
        System.out.println("3. STATIONARY DISTRIBUTION (Long-term Probabilities)");
        System.out.println("===================================================================");
        System.out.println();

        Map<String, Double> stationary = weather.stationaryDistribution(1000);

        System.out.println("After infinite time, regardless of starting state:");
        for (String state : Arrays.asList("sunny", "cloudy", "rainy")) {
            double prob = stationary.getOrDefault(state, 0.0);
            StringBuilder bar = new StringBuilder();
            for (int i = 0; i < (int)(prob * 50); i++) {
                bar.append("#");
            }
            String capState = state.substring(0, 1).toUpperCase() + state.substring(1);
            System.out.printf("  %s %s %.1f%% %s%n", getEmoji(state), capState, prob * 100, bar.toString());
        }

        // Verify by simulation
        System.out.println();
        System.out.println("Verification by simulation (10,000 steps from 'sunny'):");
        List<String> longSim = weather.simulate("sunny", 10000, rng);
        Map<String, Integer> counts = new HashMap<>();
        for (String state : longSim) {
            counts.merge(state, 1, Integer::sum);
        }
        int total = longSim.size();
        for (String state : Arrays.asList("sunny", "cloudy", "rainy")) {
            System.out.printf("  %s: %.1f%%%n", state, counts.getOrDefault(state, 0) * 100.0 / total);
        }
        System.out.println();

        // Expected hitting times
        System.out.println("===================================================================");
        System.out.println("4. EXPECTED HITTING TIMES");
        System.out.println("===================================================================");
        System.out.println();

        System.out.println("Average days until sunny (from 1000 simulations):");
        for (String start : Arrays.asList("cloudy", "rainy")) {
            double expected = weather.expectedStepsTo(start, "sunny", 365, 1000, rng);
            if (expected < 0) {
                System.out.printf("  From %s: unreachable%n", start);
            } else {
                System.out.printf("  From %s: %.1f days%n", start, expected);
            }
        }

        System.out.println();
        System.out.println("Average days until rainy (from 1000 simulations):");
        for (String start : Arrays.asList("sunny", "cloudy")) {
            double expected = weather.expectedStepsTo(start, "rainy", 365, 1000, rng);
            if (expected < 0) {
                System.out.printf("  From %s: unreachable%n", start);
            } else {
                System.out.printf("  From %s: %.1f days%n", start, expected);
            }
        }
        System.out.println();

        // Learning from data
        System.out.println("===================================================================");
        System.out.println("5. LEARNING FROM OBSERVED DATA");
        System.out.println("===================================================================");
        System.out.println();

        List<List<String>> historicalData = Arrays.asList(
            Arrays.asList("sunny", "sunny", "cloudy", "rainy", "rainy", "cloudy", "sunny"),
            Arrays.asList("rainy", "rainy", "cloudy", "cloudy", "sunny", "sunny", "sunny"),
            Arrays.asList("cloudy", "sunny", "sunny", "sunny", "cloudy", "rainy", "rainy"),
            Arrays.asList("sunny", "cloudy", "cloudy", "rainy", "cloudy", "sunny", "sunny")
        );

        StateChain learnedWeather = new StateChain();
        learnedWeather.trainMany(historicalData);

        System.out.println("Learned transition probabilities from 4 weeks of data:");
        for (String state : Arrays.asList("sunny", "cloudy", "rainy")) {
            Map<String, Double> probs = learnedWeather.probabilitiesFrom(state);
            System.out.printf("  %s ->", state);
            for (String next : Arrays.asList("sunny", "cloudy", "rainy")) {
                System.out.printf("  %s: %.0f%%", next, probs.getOrDefault(next, 0.0) * 100);
            }
            System.out.println();
        }

        System.out.println();
        System.out.println("====================================================================");
        System.out.println("Key Insights:");
        System.out.println("  * Markov chains can model any system with discrete states");
        System.out.println("  * The stationary distribution shows long-term behavior");
        System.out.println("  * Can be learned from observed data or specified directly");
        System.out.println("====================================================================");
    }
}
