# Mini Markov Java

A from-scratch implementation of Markov Chains in Java for educational purposes.

## Overview

This is a Java port of the mini-markov Rust implementation, providing the same features and similar results.

## Features

- ✅ Generic Markov chain implementation
- ✅ N-gram support (configurable order)
- ✅ Text generation (word-level)
- ✅ State machine modeling
- ✅ Stationary distribution calculation
- ✅ Entropy measurement

## Project Structure

```
mini-markov-java/
├── pom.xml
├── README.md
└── src/
    ├── main/java/com/minimarkov/
    │   ├── MarkovChain.java    # Core implementation
    │   ├── TextGenerator.java  # Text generation
    │   ├── StateChain.java     # State modeling
    │   └── demo/
    │       ├── DemoText.java
    │       └── DemoWeather.java
    └── test/java/com/minimarkov/
        └── MarkovChainTest.java
```

## Requirements

- Java 17+
- Maven 3.6+

## Usage

### Building

```bash
mvn compile
```

### Running Tests

```bash
mvn test
```

### Running Demos

```bash
# Text generation demo
mvn exec:java -Dexec.mainClass="com.minimarkov.demo.DemoText"

# Weather simulation demo
mvn exec:java -Dexec.mainClass="com.minimarkov.demo.DemoWeather"
```

## Example

```java
import com.minimarkov.MarkovChain;
import java.util.*;

public class Example {
    public static void main(String[] args) {
        // Create a first-order chain
        MarkovChain<Character> chain = new MarkovChain<>(1);
        
        // Train on a sequence
        chain.train(Arrays.asList('a', 'b', 'a', 'b', 'a', 'c'));
        
        // Get probabilities
        Map<Character, Double> probs = chain.getProbabilities(
            Collections.singletonList('a')
        );
        System.out.println(probs); // {b=0.666..., c=0.333...}
        
        // Generate sequence
        Random random = new Random(42);
        List<Character> generated = chain.generate(10, random);
        System.out.println(generated);
    }
}
```

## License

MIT License
