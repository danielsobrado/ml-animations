package com.mininn;

import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/**
 * A simple 2D tensor (matrix) implementation for neural network operations.
 */
public class Tensor {
    private final double[][] data;
    private final int rows;
    private final int cols;

    public Tensor(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public Tensor(double[][] data) {
        this.rows = data.length;
        this.cols = data.length > 0 ? data[0].length : 0;
        this.data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, this.data[i], 0, cols);
        }
    }

    // Static factory methods
    public static Tensor zeros(int rows, int cols) {
        return new Tensor(rows, cols);
    }

    public static Tensor ones(int rows, int cols) {
        Tensor t = new Tensor(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                t.data[i][j] = 1.0;
            }
        }
        return t;
    }

    public static Tensor xavier(int rows, int cols, Random rng) {
        Tensor t = new Tensor(rows, cols);
        double scale = Math.sqrt(2.0 / (rows + cols));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                t.data[i][j] = rng.nextGaussian() * scale;
            }
        }
        return t;
    }

    public static Tensor he(int rows, int cols, Random rng) {
        Tensor t = new Tensor(rows, cols);
        double scale = Math.sqrt(2.0 / rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                t.data[i][j] = rng.nextGaussian() * scale;
            }
        }
        return t;
    }

    public static Tensor fromArray(double[] arr, int rows, int cols) {
        Tensor t = new Tensor(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                t.data[i][j] = arr[i * cols + j];
            }
        }
        return t;
    }

    // Getters
    public int getRows() { return rows; }
    public int getCols() { return cols; }
    public int[] getShape() { return new int[]{rows, cols}; }

    public double get(int i, int j) {
        return data[i][j];
    }

    public void set(int i, int j, double value) {
        data[i][j] = value;
    }

    public double[] getRow(int i) {
        double[] row = new double[cols];
        System.arraycopy(data[i], 0, row, 0, cols);
        return row;
    }

    public void setRow(int i, double[] row) {
        System.arraycopy(row, 0, data[i], 0, Math.min(cols, row.length));
    }

    // Matrix operations
    public Tensor matmul(Tensor other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException(
                String.format("Matrix dimensions don't match for multiplication: (%d,%d) x (%d,%d)",
                    rows, cols, other.rows, other.cols));
        }
        Tensor result = new Tensor(this.rows, other.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0;
                for (int k = 0; k < this.cols; k++) {
                    sum += this.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    public Tensor transpose() {
        Tensor result = new Tensor(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }

    public Tensor add(Tensor other) {
        if (other.rows == 1 && other.cols == this.cols) {
            // Broadcasting: add row vector to each row
            Tensor result = new Tensor(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result.data[i][j] = data[i][j] + other.data[0][j];
                }
            }
            return result;
        }
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Tensor dimensions must match for addition");
        }
        Tensor result = new Tensor(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    public Tensor sub(Tensor other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Tensor dimensions must match for subtraction");
        }
        Tensor result = new Tensor(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    public Tensor mul(Tensor other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Tensor dimensions must match for element-wise multiplication");
        }
        Tensor result = new Tensor(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    public Tensor scale(double scalar) {
        Tensor result = new Tensor(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }

    public Tensor apply(DoubleUnaryOperator op) {
        Tensor result = new Tensor(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = op.applyAsDouble(data[i][j]);
            }
        }
        return result;
    }

    public double sum() {
        double total = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                total += data[i][j];
            }
        }
        return total;
    }

    public Tensor sumAxis(int axis) {
        if (axis == 0) {
            // Sum along rows -> result is 1 x cols
            Tensor result = new Tensor(1, cols);
            for (int j = 0; j < cols; j++) {
                double sum = 0;
                for (int i = 0; i < rows; i++) {
                    sum += data[i][j];
                }
                result.data[0][j] = sum;
            }
            return result;
        } else {
            // Sum along cols -> result is rows x 1
            Tensor result = new Tensor(rows, 1);
            for (int i = 0; i < rows; i++) {
                double sum = 0;
                for (int j = 0; j < cols; j++) {
                    sum += data[i][j];
                }
                result.data[i][0] = sum;
            }
            return result;
        }
    }

    public Tensor sliceRows(int start, int end) {
        Tensor result = new Tensor(end - start, cols);
        for (int i = start; i < end; i++) {
            System.arraycopy(data[i], 0, result.data[i - start], 0, cols);
        }
        return result;
    }

    public Tensor copy() {
        Tensor result = new Tensor(rows, cols);
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, result.data[i], 0, cols);
        }
        return result;
    }

    public void addInPlace(Tensor other) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] += other.data[i][j];
            }
        }
    }

    public void scaleInPlace(double scalar) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] *= scalar;
            }
        }
    }

    public void fill(double value) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = value;
            }
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Tensor(%d, %d)\n", rows, cols));
        for (int i = 0; i < Math.min(rows, 5); i++) {
            sb.append("[");
            for (int j = 0; j < Math.min(cols, 5); j++) {
                sb.append(String.format("%8.4f", data[i][j]));
                if (j < cols - 1) sb.append(", ");
            }
            if (cols > 5) sb.append("...");
            sb.append("]\n");
        }
        if (rows > 5) sb.append("...\n");
        return sb.toString();
    }
}
