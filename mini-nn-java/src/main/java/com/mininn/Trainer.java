package com.mininn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Training utilities for neural networks.
 */
public class Trainer {
    private final int epochs;
    private final int batchSize;
    private final double validationSplit;
    private final boolean shuffle;
    private final boolean verbose;
    private final int earlyStopPatience;
    private final Random rng;

    public Trainer(int epochs, int batchSize, double validationSplit, 
                   boolean shuffle, boolean verbose, int earlyStopPatience, Random rng) {
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.validationSplit = validationSplit;
        this.shuffle = shuffle;
        this.verbose = verbose;
        this.earlyStopPatience = earlyStopPatience;
        this.rng = rng;
    }

    public TrainingHistory fit(Network network, Tensor x, Tensor y, Loss loss, Optimizer optimizer) {
        TrainingHistory history = new TrainingHistory();

        int nSamples = x.getRows();
        int nVal = (int) (nSamples * validationSplit);
        int nTrain = nSamples - nVal;

        // Create shuffled indices
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < nSamples; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, rng);

        // Split data
        Tensor xTrain = selectRows(x, indices.subList(0, nTrain));
        Tensor yTrain = selectRows(y, indices.subList(0, nTrain));
        Tensor xVal = selectRows(x, indices.subList(nTrain, nSamples));
        Tensor yVal = selectRows(y, indices.subList(nTrain, nSamples));

        double bestValLoss = Double.POSITIVE_INFINITY;
        int patienceCounter = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            // Shuffle training data
            if (shuffle) {
                shuffleData(xTrain, yTrain);
            }

            // Training
            double[] trainMetrics = trainEpoch(network, xTrain, yTrain, loss, optimizer);
            double trainLoss = trainMetrics[0];
            double trainAcc = trainMetrics[1];

            // Validation
            double[] valMetrics = evaluate(network, xVal, yVal, loss);
            double valLoss = valMetrics[0];
            double valAcc = valMetrics[1];

            history.addEpoch(trainLoss, trainAcc, valLoss, valAcc);

            if (verbose && (epoch + 1) % 10 == 0) {
                System.out.printf("Epoch %3d: train_loss=%.4f, train_acc=%.2f%%, val_loss=%.4f, val_acc=%.2f%%%n",
                    epoch + 1, trainLoss, trainAcc * 100, valLoss, valAcc * 100);
            }

            // Early stopping
            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                patienceCounter = 0;
            } else {
                patienceCounter++;
                if (patienceCounter >= earlyStopPatience) {
                    if (verbose) {
                        System.out.printf("Early stopping at epoch %d%n", epoch + 1);
                    }
                    break;
                }
            }
        }

        return history;
    }

    private double[] trainEpoch(Network network, Tensor x, Tensor y, Loss loss, Optimizer optimizer) {
        int nSamples = x.getRows();
        int nBatches = (nSamples + batchSize - 1) / batchSize;

        double totalLoss = 0;
        int correct = 0;
        int total = 0;

        for (int b = 0; b < nBatches; b++) {
            int start = b * batchSize;
            int end = Math.min(start + batchSize, nSamples);

            Tensor xBatch = x.sliceRows(start, end);
            Tensor yBatch = y.sliceRows(start, end);

            // Forward pass
            Tensor pred = network.forward(xBatch);

            // Compute loss
            double batchLoss = loss.compute(pred, yBatch);
            totalLoss += batchLoss * (end - start);

            // Compute accuracy
            correct += computeAccuracy(pred, yBatch);
            total += end - start;

            // Backward pass
            Tensor grad = loss.gradient(pred, yBatch);
            network.backward(grad);

            // Update weights
            optimizer.step(network.getLayers());
        }

        return new double[]{totalLoss / nSamples, (double) correct / total};
    }

    private double[] evaluate(Network network, Tensor x, Tensor y, Loss loss) {
        Tensor pred = network.forward(x);
        double lossValue = loss.compute(pred, y);
        int correct = computeAccuracy(pred, y);
        return new double[]{lossValue, (double) correct / x.getRows()};
    }

    private int computeAccuracy(Tensor pred, Tensor target) {
        int correct = 0;
        int rows = pred.getRows();
        int cols = pred.getCols();

        for (int i = 0; i < rows; i++) {
            if (cols == 1) {
                // Binary classification
                double p = pred.get(i, 0);
                double t = target.get(i, 0);
                int predClass = p >= 0.5 ? 1 : 0;
                int targetClass = (int) Math.round(t);
                if (predClass == targetClass) {
                    correct++;
                }
            } else {
                // Multi-class classification
                int predArgmax = argmax(pred.getRow(i));
                int targetArgmax = argmax(target.getRow(i));
                if (predArgmax == targetArgmax) {
                    correct++;
                }
            }
        }

        return correct;
    }

    private int argmax(double[] arr) {
        int maxIdx = 0;
        double maxVal = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private Tensor selectRows(Tensor t, List<Integer> indices) {
        int cols = t.getCols();
        Tensor result = new Tensor(indices.size(), cols);
        for (int i = 0; i < indices.size(); i++) {
            for (int j = 0; j < cols; j++) {
                result.set(i, j, t.get(indices.get(i), j));
            }
        }
        return result;
    }

    private void shuffleData(Tensor x, Tensor y) {
        int n = x.getRows();
        int cx = x.getCols();
        int cy = y.getCols();

        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            // Swap rows i and j
            for (int k = 0; k < cx; k++) {
                double tmp = x.get(i, k);
                x.set(i, k, x.get(j, k));
                x.set(j, k, tmp);
            }
            for (int k = 0; k < cy; k++) {
                double tmp = y.get(i, k);
                y.set(i, k, y.get(j, k));
                y.set(j, k, tmp);
            }
        }
    }

    /**
     * Training history container.
     */
    public static class TrainingHistory {
        private final List<Double> trainLoss = new ArrayList<>();
        private final List<Double> trainAccuracy = new ArrayList<>();
        private final List<Double> valLoss = new ArrayList<>();
        private final List<Double> valAccuracy = new ArrayList<>();

        public void addEpoch(double tLoss, double tAcc, double vLoss, double vAcc) {
            trainLoss.add(tLoss);
            trainAccuracy.add(tAcc);
            valLoss.add(vLoss);
            valAccuracy.add(vAcc);
        }

        public List<Double> getTrainLoss() { return trainLoss; }
        public List<Double> getTrainAccuracy() { return trainAccuracy; }
        public List<Double> getValLoss() { return valLoss; }
        public List<Double> getValAccuracy() { return valAccuracy; }
    }
}
