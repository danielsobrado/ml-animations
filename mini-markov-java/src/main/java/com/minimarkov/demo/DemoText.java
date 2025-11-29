package com.minimarkov.demo;

import com.minimarkov.TextGenerator;
import java.util.Random;

public class DemoText {

    public static void main(String[] args) {
        System.out.println("+==================================================================+");
        System.out.println("|       Mini Markov Java - Text Generation Demo                   |");
        System.out.println("+==================================================================+");
        System.out.println();

        String sherlock = "Sherlock Holmes took his bottle from the corner of the mantelpiece and his hypodermic syringe " +
            "from its neat morocco case. With his long, white, nervous fingers he adjusted the delicate " +
            "needle, and rolled back his left shirt-cuff. For some little time his eyes rested thoughtfully " +
            "upon the sinewy forearm and wrist all dotted and scarred with innumerable puncture-marks. " +
            "Finally he thrust the sharp point home, pressed down the tiny piston, and sank back into the " +
            "velvet-lined armchair with a long sigh of satisfaction.";

        String dickens = "It was the best of times, it was the worst of times, it was the age of wisdom, it was the " +
            "age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was " +
            "the season of Light, it was the season of Darkness, it was the spring of hope, it was the " +
            "winter of despair. We had everything before us, we had nothing before us, we were all " +
            "going direct to Heaven, we were all going direct the other way.";

        String shakespeare = "To be, or not to be, that is the question. Whether tis nobler in the mind to suffer the " +
            "slings and arrows of outrageous fortune, or to take arms against a sea of troubles, and " +
            "by opposing end them. To die, to sleep, no more, and by a sleep to say we end the heartache, " +
            "and the thousand natural shocks that flesh is heir to.";

        // Fixed seed for reproducibility
        Random rng = new Random(42);

        // Order 1
        System.out.println("===================================================================");
        System.out.println("1. UNIGRAM MODEL (Order 1)");
        System.out.println("===================================================================");
        System.out.println("Training on Sherlock Holmes excerpt...");
        System.out.println();

        TextGenerator gen1 = new TextGenerator(1);
        gen1.train(sherlock);

        TextGenerator.Stats stats = gen1.getStats();
        System.out.println("Statistics:");
        System.out.printf("  Unique word contexts: %d%n", stats.numStates);
        System.out.printf("  Total transitions: %d%n", stats.numTransitions);
        System.out.printf("  Entropy: %.3f bits%n", stats.entropy);

        System.out.println("\nGenerated text (30 words):");
        System.out.println("--------------------------------------------------------------------");
        System.out.println(gen1.generate(30, rng));
        System.out.println();

        // Order 2
        System.out.println("===================================================================");
        System.out.println("2. BIGRAM MODEL (Order 2)");
        System.out.println("===================================================================");
        System.out.println("Training on Dickens excerpt...");
        System.out.println();

        TextGenerator gen2 = new TextGenerator(2);
        gen2.train(dickens);

        stats = gen2.getStats();
        System.out.println("Statistics:");
        System.out.printf("  Unique word pair contexts: %d%n", stats.numStates);
        System.out.printf("  Total transitions: %d%n", stats.numTransitions);
        System.out.printf("  Entropy: %.3f bits%n", stats.entropy);

        System.out.println("\nGenerated text (30 words):");
        System.out.println("--------------------------------------------------------------------");
        System.out.println(gen2.generate(30, rng));
        System.out.println();

        // Order 3
        System.out.println("===================================================================");
        System.out.println("3. TRIGRAM MODEL (Order 3)");
        System.out.println("===================================================================");
        System.out.println("Training on Shakespeare excerpt...");
        System.out.println();

        TextGenerator gen3 = new TextGenerator(3);
        gen3.train(shakespeare);

        stats = gen3.getStats();
        System.out.println("Statistics:");
        System.out.printf("  Unique word triplet contexts: %d%n", stats.numStates);
        System.out.printf("  Total transitions: %d%n", stats.numTransitions);
        System.out.printf("  Entropy: %.3f bits%n", stats.entropy);

        System.out.println("\nGenerated text (30 words):");
        System.out.println("--------------------------------------------------------------------");
        System.out.println(gen3.generate(30, rng));
        System.out.println();

        // Mixed
        System.out.println("===================================================================");
        System.out.println("4. MIXED MODEL (Order 2, Multiple Sources)");
        System.out.println("===================================================================");
        System.out.println("Training on all three texts...");
        System.out.println();

        TextGenerator genMixed = new TextGenerator(2);
        genMixed.trainMany(sherlock, dickens, shakespeare);

        stats = genMixed.getStats();
        System.out.println("Statistics:");
        System.out.printf("  Unique word pair contexts: %d%n", stats.numStates);
        System.out.printf("  Total transitions: %d%n", stats.numTransitions);
        System.out.printf("  Entropy: %.3f bits%n", stats.entropy);
        System.out.printf("  Training sequences: %d%n", stats.numSequences);

        System.out.println("\nGenerated text (40 words):");
        System.out.println("--------------------------------------------------------------------");
        System.out.println(genMixed.generate(40, rng));
        System.out.println();

        // Comparing orders
        System.out.println("===================================================================");
        System.out.println("5. COMPARING DIFFERENT N-GRAM ORDERS");
        System.out.println("===================================================================");

        String combinedText = sherlock + " " + dickens + " " + shakespeare;

        for (int order = 1; order <= 4; order++) {
            TextGenerator gen = new TextGenerator(order);
            gen.train(combinedText);

            TextGenerator.Stats s = gen.getStats();
            String sample = gen.generate(15, rng);

            System.out.printf("%nOrder %d: %d states, entropy=%.2f bits%n", order, s.numStates, s.entropy);
            System.out.printf("  Sample: %s%n", sample);
        }

        System.out.println();
        System.out.println("====================================================================");
        System.out.println("Observation: Higher order = more coherent but less creative");
        System.out.println("             Lower order = more random but potentially novel");
        System.out.println("====================================================================");
    }
}
