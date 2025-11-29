package com.minimarkov;

import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Text generator using Markov chains.
 */
public class TextGenerator {
    private final MarkovChain<String> chain;
    private final Set<String> endTokens;
    private boolean preserveCase;

    /**
     * Creates a new text generator with the specified n-gram size.
     * 
     * @param n The n-gram size (1 = unigram, 2 = bigram, etc.)
     */
    public TextGenerator(int n) {
        this.chain = new MarkovChain<>(n);
        this.endTokens = new HashSet<>(Arrays.asList(".", "!", "?"));
        this.preserveCase = false;
    }

    /**
     * Sets whether to preserve case.
     */
    public TextGenerator setPreserveCase(boolean preserve) {
        this.preserveCase = preserve;
        return this;
    }

    /**
     * Adds a custom end token.
     */
    public TextGenerator addEndToken(String token) {
        endTokens.add(token);
        return this;
    }

    /**
     * Tokenizes text into words.
     */
    private List<String> tokenize(String text) {
        if (!preserveCase) {
            text = text.toLowerCase();
        }

        List<String> tokens = new ArrayList<>();
        String[] words = text.split("\\s+");

        for (String word : words) {
            tokens.addAll(splitPunctuation(word));
        }

        return tokens.stream()
                .filter(t -> !t.isEmpty())
                .collect(Collectors.toList());
    }

    /**
     * Splits punctuation from words.
     */
    private List<String> splitPunctuation(String word) {
        List<String> result = new ArrayList<>();
        StringBuilder current = new StringBuilder();

        for (char c : word.toCharArray()) {
            if (Character.isLetterOrDigit(c)) {
                current.append(c);
            } else {
                if (current.length() > 0) {
                    result.add(current.toString());
                    current.setLength(0);
                }
                result.add(String.valueOf(c));
            }
        }

        if (current.length() > 0) {
            result.add(current.toString());
        }

        return result;
    }

    /**
     * Trains the generator on a text corpus.
     */
    public void train(String text) {
        List<String> tokens = tokenize(text);
        if (!tokens.isEmpty()) {
            chain.train(tokens);
        }
    }

    /**
     * Trains on multiple texts.
     */
    public void trainMany(String... texts) {
        for (String text : texts) {
            train(text);
        }
    }

    /**
     * Generates text with a maximum number of words.
     */
    public String generate(int maxWords, Random random) {
        List<String> tokens = chain.generate(maxWords, random);
        return joinTokens(tokens);
    }

    /**
     * Generates text starting with a specific prompt.
     */
    public String generateFrom(String prompt, int maxWords, Random random) {
        List<String> promptTokens = tokenize(prompt);

        if (promptTokens.size() < chain.getOrder()) {
            return generate(maxWords, random);
        }

        List<String> result = new ArrayList<>(promptTokens);

        while (result.size() < promptTokens.size() + maxWords) {
            List<String> current = result.subList(result.size() - chain.getOrder(), result.size());
            String next = chain.sampleNext(current, random);
            if (next == null) {
                break;
            }
            result.add(next);
            if (endTokens.contains(next)) {
                break;
            }
        }

        return joinTokens(result);
    }

    /**
     * Generates a complete sentence.
     */
    public String generateSentence(int maxWords, Random random) {
        List<String> start = chain.sampleStart(random);
        if (start == null) {
            return "";
        }

        List<String> result = new ArrayList<>(start);

        while (result.size() < maxWords) {
            List<String> current = result.subList(result.size() - chain.getOrder(), result.size());
            String next = chain.sampleNext(current, random);
            if (next == null) {
                break;
            }
            result.add(next);
            if (endTokens.contains(next)) {
                break;
            }
        }

        return joinTokens(result);
    }

    /**
     * Joins tokens back into text with proper spacing.
     */
    private String joinTokens(List<String> tokens) {
        if (tokens.isEmpty()) {
            return "";
        }

        Pattern punctuation = Pattern.compile("^[.!?,;:'\"\\)\\]]$");
        StringBuilder result = new StringBuilder();

        for (int i = 0; i < tokens.size(); i++) {
            String token = tokens.get(i);
            if (i > 0 && !punctuation.matcher(token).matches()) {
                result.append(" ");
            }
            result.append(token);
        }

        return result.toString();
    }

    /**
     * Returns statistics about the generator.
     */
    public Stats getStats() {
        return new Stats(
                chain.getOrder(),
                chain.getNumStates(),
                chain.getNumTransitions(),
                chain.getNumSequences(),
                chain.getEntropy()
        );
    }

    /**
     * Statistics about a text generator.
     */
    public static class Stats {
        public final int order;
        public final int numStates;
        public final int numTransitions;
        public final int numSequences;
        public final double entropy;

        public Stats(int order, int numStates, int numTransitions, int numSequences, double entropy) {
            this.order = order;
            this.numStates = numStates;
            this.numTransitions = numTransitions;
            this.numSequences = numSequences;
            this.entropy = entropy;
        }

        @Override
        public String toString() {
            return String.format("Stats(order=%d, states=%d, transitions=%d, sequences=%d, entropy=%.3f)",
                    order, numStates, numTransitions, numSequences, entropy);
        }
    }
}
