package com.mininn;

import java.util.Random;

/**
 * XOR classification demo using Mini-NN Java.
 */
public class DemoXOR {
    public static void main(String[] args) {
        System.out.println("=== Mini-NN Java: XOR Demo ===");
        System.out.println();

        Random rng = new Random(42);

        // Generate expanded XOR data for training
        int trainSize = 1000;
        Tensor[] trainData = generateExpandedXORData(trainSize, rng);
        Tensor xTrain = trainData[0];
        Tensor yTrain = trainData[1];

        // Build network: 2 -> 8 -> 8 -> 1
        Network network = new Network(rng)
            .addDense(2, 8, Activation.RELU, DenseLayer.InitType.HE)
            .addDense(8, 8, Activation.RELU, DenseLayer.InitType.HE)
            .addDense(8, 1, Activation.SIGMOID, DenseLayer.InitType.XAVIER);

        network.summary();
        System.out.println();

        // Create optimizer
        Optimizer optimizer = new AdamOptimizer(0.01);

        // Create loss function
        Loss loss = Loss.BINARY_CROSS_ENTROPY;

        // Create trainer
        Trainer trainer = new Trainer(
            100,    // epochs
            32,     // batch size
            0.2,    // validation split
            true,   // shuffle
            true,   // verbose
            15,     // early stop patience
            rng
        );

        // Train
        System.out.println("Training...");
        Trainer.TrainingHistory history = trainer.fit(network, xTrain, yTrain, loss, optimizer);

        // Evaluate on original XOR patterns
        System.out.println();
        System.out.println("=== Final Evaluation on Pure XOR ===");

        Tensor[] testData = generateXORData();
        Tensor xTest = testData[0];
        Tensor yTest = testData[1];

        int correct = 0;
        for (int i = 0; i < 4; i++) {
            Tensor input = xTest.sliceRows(i, i + 1);
            double target = yTest.get(i, 0);
            Tensor pred = network.predict(input);
            double predVal = pred.get(0, 0);
            int predClass = predVal >= 0.5 ? 1 : 0;
            int targetClass = (int) target;

            String status = predClass == targetClass ? "✓" : "✗";
            if (predClass == targetClass) {
                correct++;
            }

            System.out.printf("Input: [%.0f, %.0f] -> Expected: %d, Predicted: %.4f (%d) %s%n",
                input.get(0, 0), input.get(0, 1), targetClass, predVal, predClass, status);
        }

        double accuracy = (double) correct / 4.0 * 100;
        System.out.printf("%nFinal Accuracy: %.1f%% (%d/4)%n", accuracy, correct);

        // Print final training stats
        if (!history.getValAccuracy().isEmpty()) {
            double finalValAcc = history.getValAccuracy().get(history.getValAccuracy().size() - 1) * 100;
            System.out.printf("Final Validation Accuracy: %.1f%%%n", finalValAcc);
        }
    }

    private static Tensor[] generateXORData() {
        Tensor x = Tensor.zeros(4, 2);
        x.set(0, 0, 0); x.set(0, 1, 0);
        x.set(1, 0, 0); x.set(1, 1, 1);
        x.set(2, 0, 1); x.set(2, 1, 0);
        x.set(3, 0, 1); x.set(3, 1, 1);

        Tensor y = Tensor.zeros(4, 1);
        y.set(0, 0, 0);
        y.set(1, 0, 1);
        y.set(2, 0, 1);
        y.set(3, 0, 0);

        return new Tensor[]{x, y};
    }

    private static Tensor[] generateExpandedXORData(int n, Random rng) {
        Tensor x = Tensor.zeros(n, 2);
        Tensor y = Tensor.zeros(n, 1);

        for (int i = 0; i < n; i++) {
            int a = rng.nextInt(2);
            int b = rng.nextInt(2);
            int xor = a ^ b;

            // Add small noise to inputs
            double noise = 0.1;
            x.set(i, 0, a + rng.nextDouble() * noise - noise / 2);
            x.set(i, 1, b + rng.nextDouble() * noise - noise / 2);
            y.set(i, 0, xor);
        }

        return new Tensor[]{x, y};
    }
}
