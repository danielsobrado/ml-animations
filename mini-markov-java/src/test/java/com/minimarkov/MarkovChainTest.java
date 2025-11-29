package com.minimarkov;

import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

class MarkovChainTest {

    @Test
    void testBasicChain() {
        MarkovChain<Character> chain = new MarkovChain<>(1);
        chain.train(Arrays.asList('a', 'b', 'a', 'b', 'a', 'b'));

        assertEquals(1, chain.getOrder());
        assertTrue(chain.getNumStates() > 0);

        // After 'a', should always go to 'b' (probability 1.0)
        Map<Character, Double> probs = chain.getProbabilities(Collections.singletonList('a'));
        assertNotNull(probs);
        assertEquals(1.0, probs.get('b'), 0.001);
    }

    @Test
    void testSecondOrder() {
        MarkovChain<Character> chain = new MarkovChain<>(2);
        chain.train(Arrays.asList('a', 'b', 'c', 'a', 'b', 'd'));

        // After 'a','b', we've seen 'c' once and 'd' once
        Map<Character, Double> probs = chain.getProbabilities(Arrays.asList('a', 'b'));
        assertNotNull(probs);
        assertEquals(0.5, probs.get('c'), 0.001);
        assertEquals(0.5, probs.get('d'), 0.001);
    }

    @Test
    void testGeneration() {
        MarkovChain<Integer> chain = new MarkovChain<>(1);
        chain.train(Arrays.asList(1, 2, 3, 1, 2, 3, 1, 2, 3));

        Random random = new Random(42);
        List<Integer> generated = chain.generate(10, random);

        assertFalse(generated.isEmpty());
        assertTrue(generated.size() <= 10);
    }

    @Test
    void testEntropy() {
        // Deterministic chain (entropy = 0)
        MarkovChain<Character> chain1 = new MarkovChain<>(1);
        chain1.train(Arrays.asList('a', 'b', 'a', 'b', 'a', 'b'));
        assertTrue(chain1.getEntropy() < 0.01);

        // Random chain (higher entropy)
        MarkovChain<Character> chain2 = new MarkovChain<>(1);
        chain2.train(Arrays.asList('a', 'b', 'a', 'c', 'a', 'd', 'a', 'e'));
        assertTrue(chain2.getEntropy() > 1.0);
    }

    @Test
    void testStateChain() {
        StateChain weather = new StateChain();
        weather.addTransitionCount("sunny", "sunny", 70);
        weather.addTransitionCount("sunny", "cloudy", 20);
        weather.addTransitionCount("sunny", "rainy", 10);

        double prob = weather.probability("sunny", "sunny");
        assertEquals(0.70, prob, 0.01);
    }

    @Test
    void testStationaryDistribution() {
        StateChain weather = new StateChain();
        weather.addTransitionCount("sunny", "sunny", 70);
        weather.addTransitionCount("sunny", "cloudy", 20);
        weather.addTransitionCount("sunny", "rainy", 10);
        weather.addTransitionCount("cloudy", "sunny", 30);
        weather.addTransitionCount("cloudy", "cloudy", 40);
        weather.addTransitionCount("cloudy", "rainy", 30);
        weather.addTransitionCount("rainy", "sunny", 20);
        weather.addTransitionCount("rainy", "cloudy", 40);
        weather.addTransitionCount("rainy", "rainy", 40);

        Map<String, Double> stationary = weather.stationaryDistribution(1000);

        // Check that probabilities sum to ~1
        double sum = stationary.values().stream().mapToDouble(Double::doubleValue).sum();
        assertEquals(1.0, sum, 0.01);

        // Sunny should be most likely (~46%)
        assertTrue(stationary.get("sunny") > 0.40 && stationary.get("sunny") < 0.52);
    }

    @Test
    void testTextGenerator() {
        TextGenerator gen = new TextGenerator(2);
        gen.train("the quick brown fox jumps over the lazy dog");

        Random random = new Random(42);
        String text = gen.generate(20, random);

        assertFalse(text.isEmpty());
    }
}
