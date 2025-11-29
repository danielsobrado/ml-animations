package com.minimarkov;

import java.util.*;
import java.util.stream.Collectors;

/**
 * State chain for modeling discrete state systems.
 */
public class StateChain {
    private final MarkovChain<String> chain;
    private List<String> states;

    /**
     * Creates a new first-order state chain.
     */
    public StateChain() {
        this(1);
    }

    /**
     * Creates a state chain with specified order.
     */
    public StateChain(int order) {
        this.chain = new MarkovChain<>(order);
        this.states = new ArrayList<>();
    }

    /**
     * Sets the possible states for validation.
     */
    public StateChain withStates(String... states) {
        this.states = Arrays.asList(states);
        return this;
    }

    /**
     * Adds a single transition observation.
     */
    public void addTransition(String from, String to) {
        chain.train(Arrays.asList(from, to));
    }

    /**
     * Adds a transition with a specific count.
     */
    public void addTransitionCount(String from, String to, int count) {
        for (int i = 0; i < count; i++) {
            addTransition(from, to);
        }
    }

    /**
     * Trains on a sequence of states.
     */
    public void train(List<String> sequence) {
        chain.train(sequence);
    }

    /**
     * Trains on multiple sequences.
     */
    public void trainMany(List<List<String>> sequences) {
        for (List<String> seq : sequences) {
            train(seq);
        }
    }

    /**
     * Gets the probability of transitioning from one state to another.
     */
    public double probability(String from, String to) {
        Map<String, Double> probs = chain.getProbabilities(Collections.singletonList(from));
        return probs != null ? probs.getOrDefault(to, 0.0) : 0.0;
    }

    /**
     * Gets all transition probabilities from a state.
     */
    public Map<String, Double> probabilitiesFrom(String state) {
        Map<String, Double> probs = chain.getProbabilities(Collections.singletonList(state));
        return probs != null ? probs : new HashMap<>();
    }

    /**
     * Samples the next state.
     */
    public String nextState(String current, Random random) {
        return chain.sampleNext(Collections.singletonList(current), random);
    }

    /**
     * Simulates the chain for a number of steps.
     */
    public List<String> simulate(String start, int steps, Random random) {
        List<String> result = new ArrayList<>();
        result.add(start);

        for (int i = 0; i < steps; i++) {
            String current = result.get(result.size() - 1);
            String next = nextState(current, random);
            if (next == null) {
                break;
            }
            result.add(next);
        }

        return result;
    }

    /**
     * Calculates the stationary distribution using power iteration.
     */
    public Map<String, Double> stationaryDistribution(int iterations) {
        // Get all unique states
        Set<String> stateSet = new HashSet<>();
        for (List<String> key : chain.getTransitions().keySet()) {
            stateSet.addAll(key);
        }
        for (Map<String, Integer> counts : chain.getTransitions().values()) {
            stateSet.addAll(counts.keySet());
        }

        if (stateSet.isEmpty()) {
            return new HashMap<>();
        }

        List<String> stateList = new ArrayList<>(stateSet);
        int n = stateList.size();

        Map<String, Integer> stateToIdx = new HashMap<>();
        for (int i = 0; i < n; i++) {
            stateToIdx.put(stateList.get(i), i);
        }

        // Build transition matrix
        double[][] matrix = new double[n][n];

        for (Map.Entry<List<String>, Map<String, Integer>> entry : chain.getTransitions().entrySet()) {
            if (entry.getKey().size() != 1) continue;

            String fromState = entry.getKey().get(0);
            Integer fromIdx = stateToIdx.get(fromState);
            if (fromIdx == null) continue;

            int total = entry.getValue().values().stream().mapToInt(Integer::intValue).sum();

            for (Map.Entry<String, Integer> toEntry : entry.getValue().entrySet()) {
                Integer toIdx = stateToIdx.get(toEntry.getKey());
                if (toIdx != null) {
                    matrix[fromIdx][toIdx] = (double) toEntry.getValue() / total;
                }
            }
        }

        // Power iteration
        double[] dist = new double[n];
        Arrays.fill(dist, 1.0 / n);

        for (int iter = 0; iter < iterations; iter++) {
            double[] newDist = new double[n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    newDist[j] += dist[i] * matrix[i][j];
                }
            }
            dist = newDist;
        }

        // Normalize
        double sum = Arrays.stream(dist).sum();
        if (sum > 0) {
            for (int i = 0; i < n; i++) {
                dist[i] /= sum;
            }
        }

        // Convert to map
        Map<String, Double> result = new HashMap<>();
        for (int i = 0; i < n; i++) {
            result.put(stateList.get(i), dist[i]);
        }
        return result;
    }

    /**
     * Estimates the expected number of steps to reach a target state.
     */
    public double expectedStepsTo(String from, String to, int maxSteps, int simulations, Random random) {
        int totalSteps = 0;
        int successCount = 0;

        for (int sim = 0; sim < simulations; sim++) {
            String current = from;
            for (int step = 1; step <= maxSteps; step++) {
                String next = nextState(current, random);
                if (next == null) {
                    break;
                }
                if (next.equals(to)) {
                    totalSteps += step;
                    successCount++;
                    break;
                }
                current = next;
            }
        }

        return successCount == 0 ? -1 : (double) totalSteps / successCount;
    }
}
