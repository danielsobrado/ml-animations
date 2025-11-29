package com.minimarkov;

import java.util.*;

/**
 * A generic Markov Chain implementation.
 * 
 * @param <T> The type of states in the chain
 */
public class MarkovChain<T> {
    private final int order;
    private final Map<List<T>, Map<T, Integer>> transitions;
    private final Map<List<T>, Integer> startStates;
    private int numSequences;

    /**
     * Creates a new Markov chain with the specified order.
     * 
     * @param order Number of previous states to consider (must be >= 1)
     */
    public MarkovChain(int order) {
        if (order < 1) {
            throw new IllegalArgumentException("Order must be at least 1");
        }
        this.order = order;
        this.transitions = new HashMap<>();
        this.startStates = new HashMap<>();
        this.numSequences = 0;
    }

    /**
     * Returns the order of the chain.
     */
    public int getOrder() {
        return order;
    }

    /**
     * Trains the chain on a sequence of states.
     * 
     * @param sequence The sequence to train on
     */
    public void train(List<T> sequence) {
        if (sequence.size() <= order) {
            return;
        }

        numSequences++;

        // Record start state
        List<T> start = new ArrayList<>(sequence.subList(0, order));
        startStates.merge(start, 1, Integer::sum);

        // Record transitions
        for (int i = 0; i <= sequence.size() - order - 1; i++) {
            List<T> current = new ArrayList<>(sequence.subList(i, i + order));
            T next = sequence.get(i + order);

            transitions.computeIfAbsent(current, k -> new HashMap<>())
                      .merge(next, 1, Integer::sum);
        }
    }

    /**
     * Trains the chain on multiple sequences.
     */
    public void trainMany(List<List<T>> sequences) {
        for (List<T> seq : sequences) {
            train(seq);
        }
    }

    /**
     * Gets the probability distribution for next states.
     * 
     * @param current The current state sequence
     * @return Map of state to probability, or null if unknown state
     */
    public Map<T, Double> getProbabilities(List<T> current) {
        if (current.size() != order) {
            return null;
        }

        Map<T, Integer> counts = transitions.get(current);
        if (counts == null) {
            return null;
        }

        int total = counts.values().stream().mapToInt(Integer::intValue).sum();
        Map<T, Double> probs = new HashMap<>();
        for (Map.Entry<T, Integer> entry : counts.entrySet()) {
            probs.put(entry.getKey(), (double) entry.getValue() / total);
        }
        return probs;
    }

    /**
     * Samples the next state given the current state.
     * 
     * @param current The current state sequence
     * @param random Random number generator
     * @return The sampled next state, or null if unknown
     */
    public T sampleNext(List<T> current, Random random) {
        if (current.size() != order) {
            return null;
        }

        Map<T, Integer> counts = transitions.get(current);
        if (counts == null || counts.isEmpty()) {
            return null;
        }

        int total = counts.values().stream().mapToInt(Integer::intValue).sum();
        int threshold = random.nextInt(total);

        for (Map.Entry<T, Integer> entry : counts.entrySet()) {
            threshold -= entry.getValue();
            if (threshold < 0) {
                return entry.getKey();
            }
        }

        return counts.keySet().iterator().next();
    }

    /**
     * Samples a random starting state.
     */
    public List<T> sampleStart(Random random) {
        if (startStates.isEmpty()) {
            return null;
        }

        int total = startStates.values().stream().mapToInt(Integer::intValue).sum();
        int threshold = random.nextInt(total);

        for (Map.Entry<List<T>, Integer> entry : startStates.entrySet()) {
            threshold -= entry.getValue();
            if (threshold < 0) {
                return new ArrayList<>(entry.getKey());
            }
        }

        return new ArrayList<>(startStates.keySet().iterator().next());
    }

    /**
     * Generates a sequence of states.
     * 
     * @param length Maximum length of the generated sequence
     * @param random Random number generator
     * @return Generated sequence
     */
    public List<T> generate(int length, Random random) {
        List<T> start = sampleStart(random);
        if (start == null) {
            return new ArrayList<>();
        }

        List<T> result = new ArrayList<>(start);

        while (result.size() < length) {
            List<T> current = result.subList(result.size() - order, result.size());
            T next = sampleNext(current, random);
            if (next == null) {
                break;
            }
            result.add(next);
        }

        return result;
    }

    /**
     * Returns the number of unique state sequences.
     */
    public int getNumStates() {
        return transitions.size();
    }

    /**
     * Returns the total number of transitions recorded.
     */
    public int getNumTransitions() {
        return transitions.values().stream()
                .mapToInt(m -> m.values().stream().mapToInt(Integer::intValue).sum())
                .sum();
    }

    /**
     * Returns the number of training sequences.
     */
    public int getNumSequences() {
        return numSequences;
    }

    /**
     * Calculates the entropy of the chain.
     */
    public double getEntropy() {
        double totalEntropy = 0.0;
        double totalWeight = 0.0;

        for (Map<T, Integer> counts : transitions.values()) {
            int total = counts.values().stream().mapToInt(Integer::intValue).sum();
            if (total == 0) continue;

            double stateEntropy = 0.0;
            for (int count : counts.values()) {
                if (count > 0) {
                    double p = (double) count / total;
                    stateEntropy -= p * Math.log(p) / Math.log(2);
                }
            }

            totalEntropy += stateEntropy * total;
            totalWeight += total;
        }

        return totalWeight > 0 ? totalEntropy / totalWeight : 0.0;
    }

    /**
     * Gets the transition matrix.
     */
    public Map<List<T>, Map<T, Integer>> getTransitions() {
        return Collections.unmodifiableMap(transitions);
    }

    /**
     * Clears all learned transitions.
     */
    public void clear() {
        transitions.clear();
        startStates.clear();
        numSequences = 0;
    }

    @Override
    public String toString() {
        return String.format("MarkovChain(order=%d, states=%d, transitions=%d)",
                order, getNumStates(), getNumTransitions());
    }
}
