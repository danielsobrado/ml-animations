import React, { useState, useRef, useEffect } from 'react';
import gsap from 'gsap';
import Prism from 'prismjs';
import 'prismjs/components/prism-rust';
import 'prismjs/themes/prism-tomorrow.css';

// GitHub repo base URL
const GITHUB_BASE = 'https://github.com/danielsobrado/ml-animations/blob/main/mini-nn/src';

// Neural Network Architecture
const NETWORK = {
  layers: [
    { name: 'Input', neurons: 2, color: '#60a5fa' },
    { name: 'Hidden 1', neurons: 4, color: '#a78bfa' },
    { name: 'Hidden 2', neurons: 4, color: '#f472b6' },
    { name: 'Output', neurons: 1, color: '#34d399' },
  ],
};

// Activation functions
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const relu = (x) => Math.max(0, x);
const sigmoidDerivative = (x) => sigmoid(x) * (1 - sigmoid(x));

// Code snippets from mini-nn
const CODE_SNIPPETS = {
  forward: {
    title: 'Forward Propagation',
    file: 'layer.rs',
    lineStart: 89,
    lineEnd: 108,
    code: `/// Forward pass through the dense layer
fn forward(&mut self, input: &Tensor) -> Tensor {
    self.input = Some(input.clone());
    
    // z = W × x + b (Linear transformation)
    let z = input.matmul(&self.weights);
    let z = z.add(&self.biases);
    
    self.z = Some(z.clone());
    z
}`,
    explanation: `This is the core of forward propagation. For each layer:
    
1. **Store input** for use in backward pass
2. **Matrix multiply**: z = W × x (weights × inputs)
3. **Add bias**: z = z + b
4. Return the result for the next layer or activation`,
  },
  backward: {
    title: 'Backward Propagation',
    file: 'layer.rs',
    lineStart: 110,
    lineEnd: 140,
    code: `/// Backward pass - compute gradients
fn backward(&mut self, grad_output: &Tensor) -> Tensor {
    let input = self.input.as_ref().unwrap();
    let (batch_size, _) = input.shape();
    
    // Gradient w.r.t. weights: dL/dW = x^T × δ
    self.grad_weights = Some(
        input.transpose().matmul(grad_output)
    );
    
    // Gradient w.r.t. biases: dL/db = sum(δ)
    self.grad_biases = Some(grad_output.sum_axis(0));
    
    // Gradient w.r.t. input: dL/dx = δ × W^T
    grad_output.matmul(&self.weights.transpose())
}`,
    explanation: `Backpropagation computes gradients using the chain rule:

1. **Weight gradient**: How much each weight contributed to the error
   - dL/dW = input^T × gradient (outer product)
   
2. **Bias gradient**: Sum of gradients (bias affects all equally)
   - dL/db = sum(gradients)
   
3. **Input gradient**: Pass error to previous layer
   - dL/dx = gradient × weights^T`,
  },
  activation: {
    title: 'Activation Functions',
    file: 'activation.rs',
    lineStart: 25,
    lineEnd: 55,
    code: `pub fn forward(&self, x: &Tensor) -> Tensor {
    match self.activation {
        Activation::ReLU => {
            // ReLU: max(0, x) - Simple but powerful!
            Tensor::new(x.data.mapv(|v| v.max(0.0)))
        }
        Activation::Sigmoid => {
            // Sigmoid: 1/(1+e^-x) - Squashes to (0,1)
            Tensor::new(x.data.mapv(|v| {
                1.0 / (1.0 + (-v).exp())
            }))
        }
        Activation::Softmax => {
            // Softmax: e^xi / Σe^xj - Probability dist.
            let max = x.data.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp = x.data.mapv(|v| (v - max).exp());
            let sum = exp.sum();
            Tensor::new(exp / sum)
        }
    }
}`,
    explanation: `Activation functions introduce non-linearity:

🔹 **ReLU** (Rectified Linear Unit)
   - f(x) = max(0, x)
   - Fast, prevents vanishing gradients
   - Used in hidden layers

🔹 **Sigmoid**
   - f(x) = 1/(1+e^(-x))
   - Outputs between 0 and 1
   - Used for binary classification

🔹 **Softmax**
   - Converts logits to probabilities
   - Used for multi-class classification`,
  },
  loss: {
    title: 'Loss Functions',
    file: 'loss.rs',
    lineStart: 30,
    lineEnd: 60,
    code: `pub fn compute(&self, predicted: &Tensor, target: &Tensor) -> f64 {
    match self.loss {
        Loss::MSE => {
            // Mean Squared Error: Σ(y - ŷ)² / n
            let diff = predicted.sub(target);
            let squared = diff.data.mapv(|x| x * x);
            squared.mean().unwrap()
        }
        Loss::BinaryCrossEntropy => {
            // BCE: -[y·log(ŷ) + (1-y)·log(1-ŷ)]
            let eps = 1e-15; // Numerical stability
            let p = predicted.data.mapv(|x| x.clamp(eps, 1.0 - eps));
            let loss = -(&target.data * p.mapv(|x| x.ln())
                + (1.0 - &target.data) * p.mapv(|x| (1.0 - x).ln()));
            loss.mean().unwrap()
        }
    }
}`,
    explanation: `Loss functions measure prediction error:

📉 **MSE** (Mean Squared Error)
   - L = Σ(y - ŷ)² / n
   - Penalizes large errors more
   - Used for regression

📉 **Binary Cross-Entropy**
   - L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
   - Works with probabilities
   - Used for binary classification

The gradient of the loss tells us which direction to adjust weights!`,
  },
  optimizer: {
    title: 'Gradient Descent',
    file: 'optimizer.rs',
    lineStart: 45,
    lineEnd: 80,
    code: `pub fn step(&mut self, params: &mut Tensor, grads: &Tensor, lr: f64) {
    match &mut self.state {
        OptimizerState::SGD => {
            // Simple: w = w - lr × ∇w
            let update = grads.mul_scalar(lr);
            params.data -= &update.data;
        }
        OptimizerState::Adam { m, v, t } => {
            *t += 1;
            let beta1 = 0.9;
            let beta2 = 0.999;
            let eps = 1e-8;
            
            // Update biased first moment estimate
            *m = m.mul_scalar(beta1).add(&grads.mul_scalar(1.0 - beta1));
            
            // Update biased second moment estimate  
            let grad_sq = Tensor::new(grads.data.mapv(|x| x * x));
            *v = v.mul_scalar(beta2).add(&grad_sq.mul_scalar(1.0 - beta2));
            
            // Bias correction and update
            let m_hat = m.mul_scalar(1.0 / (1.0 - beta1.powi(*t)));
            let v_hat = v.mul_scalar(1.0 / (1.0 - beta2.powi(*t)));
            
            // w = w - lr × m̂ / (√v̂ + ε)
            let denom = v_hat.sqrt().add_scalar(eps);
            let update = m_hat.div(&denom).mul_scalar(lr);
            params.data -= &update.data;
        }
    }
}`,
    explanation: `Optimizers update weights to minimize loss:

⚡ **SGD** (Stochastic Gradient Descent)
   - w = w - lr × ∇w
   - Simple but can be slow

⚡ **Adam** (Adaptive Moment Estimation)
   - Maintains running averages of gradients (m) and squared gradients (v)
   - Adapts learning rate per-parameter
   - Usually faster convergence
   
The learning rate (lr) controls step size - too big overshoots, too small is slow!`,
  },
  matmul: {
    title: 'Matrix Multiplication',
    file: 'tensor.rs',
    lineStart: 60,
    lineEnd: 75,
    code: `/// Matrix multiplication: C = A × B
pub fn matmul(&self, other: &Tensor) -> Tensor {
    // Shape check: (m, n) × (n, p) → (m, p)
    assert_eq!(
        self.shape().1, other.shape().0,
        "Matrix dimensions must match for multiplication"
    );
    
    Tensor::new(self.data.dot(&other.data))
}`,
    explanation: `Matrix multiplication is the core operation in neural networks:

🔢 **Shape Rules**
   - (m × n) × (n × p) = (m × p)
   - Inner dimensions must match!

🧮 **What it computes**
   - Each output element is a dot product
   - C[i,j] = Σ A[i,k] × B[k,j]
   
In neural networks:
   - Input: (batch_size × features)
   - Weights: (features × neurons)
   - Output: (batch_size × neurons)`,
  },
};

// Intuition cards content
const INTUITION_CARDS = [
  {
    id: 'neurons',
    icon: '🧠',
    title: 'What is a Neuron?',
    content: `A neuron is like a tiny decision maker. It:
    
1. **Receives inputs** (numbers from previous layer)
2. **Multiplies each by a weight** (importance)
3. **Adds them up** with a bias
4. **Applies activation** (decides how much to "fire")

Think of it like voting: each input gets a vote (weight), 
and the neuron decides based on the total.`,
  },
  {
    id: 'forward',
    icon: '➡️',
    title: 'Forward Pass',
    content: `Information flows from input to output:

**Input → Hidden → Output**

At each layer:
• Multiply inputs by weights
• Add bias
• Apply activation function

It's like a game of telephone, but with math!
Each layer transforms the data to find patterns.`,
  },
  {
    id: 'backward',
    icon: '⬅️',
    title: 'Backward Pass',
    content: `After making a prediction, we ask:
"How wrong were we, and who's responsible?"

**Chain Rule Magic:**
Starting from the output, we trace back asking:
• How much did each weight contribute to the error?
• How should we adjust to do better?

Like finding who broke the window by asking 
each person who they passed the ball to!`,
  },
  {
    id: 'gradient',
    icon: '📉',
    title: 'Gradient Descent',
    content: `Imagine you're blindfolded on a hill, trying to find the lowest point.

**Strategy:** Feel the slope and step downhill!

• **Gradient** = direction of steepest increase
• **Negative gradient** = downhill direction
• **Learning rate** = step size

Too big steps? You might overshoot.
Too small? Takes forever!`,
  },
  {
    id: 'activation',
    icon: '⚡',
    title: 'Why Activation Functions?',
    content: `Without activation functions, neural networks 
would just be fancy linear regression!

**Activation adds non-linearity:**
• ReLU: "If negative, ignore it" → max(0, x)
• Sigmoid: "Squash to 0-1" → probability
• Tanh: "Squash to -1 to 1" → centered

This lets networks learn complex patterns 
like curves, not just straight lines!`,
  },
  {
    id: 'xor',
    icon: '🎯',
    title: 'The XOR Problem',
    content: `XOR is the classic neural network test:

| A | B | A XOR B |
|---|---|---------|
| 0 | 0 |    0    |
| 0 | 1 |    1    |
| 1 | 0 |    1    |
| 1 | 1 |    0    |

**Why it matters:**
XOR can't be solved with a single line!
You need hidden layers to create 
non-linear decision boundaries.

Our mini-nn achieves 100% on XOR! 🎉`,
  },
];

function App() {
  const [activeTab, setActiveTab] = useState('visualize');
  const [mode, setMode] = useState('intro');
  const [step, setStep] = useState(0);
  const [activations, setActivations] = useState([]);
  const [weights, setWeights] = useState([]);
  const [gradients, setGradients] = useState([]);
  const [input, setInput] = useState([1, 0]);
  const [target, setTarget] = useState(1);
  const [loss, setLoss] = useState(null);
  const [epoch, setEpoch] = useState(0);
  const [selectedCode, setSelectedCode] = useState('forward');
  const [selectedIntuition, setSelectedIntuition] = useState(null);
  const svgRef = useRef(null);
  const neuronRefs = useRef({});
  const connectionRefs = useRef({});

  useEffect(() => {
    initializeWeights();
  }, []);

  const initializeWeights = () => {
    const newWeights = [];
    for (let l = 0; l < NETWORK.layers.length - 1; l++) {
      const fromSize = NETWORK.layers[l].neurons;
      const toSize = NETWORK.layers[l + 1].neurons;
      const layerWeights = [];
      for (let i = 0; i < fromSize; i++) {
        const row = [];
        for (let j = 0; j < toSize; j++) {
          row.push((Math.random() - 0.5) * 2);
        }
        layerWeights.push(row);
      }
      newWeights.push(layerWeights);
    }
    setWeights(newWeights);
    setActivations([input.map(x => ({ value: x, preActivation: x }))]);
    setGradients([]);
    setLoss(null);
    setEpoch(0);
    setStep(0);
    setMode('intro');
  };

  const getPosition = (layerIdx, neuronIdx) => {
    const width = 800;
    const height = 400;
    const layerSpacing = width / (NETWORK.layers.length + 1);
    const layer = NETWORK.layers[layerIdx];
    const numNeurons = layer.neurons;
    const neuronGap = 70;
    const totalHeight = (numNeurons - 1) * neuronGap;
    const startY = (height - totalHeight) / 2;

    return {
      x: (layerIdx + 1) * layerSpacing,
      y: startY + neuronIdx * neuronGap,
    };
  };

  const forwardStep = () => {
    if (step >= NETWORK.layers.length - 1) {
      const output = activations[activations.length - 1][0].value;
      const mse = Math.pow(output - target, 2) / 2;
      setLoss(mse);
      setMode('forward-complete');
      return;
    }

    const currentActivations = [...activations];
    const layerActivations = currentActivations[step];
    const nextLayerSize = NETWORK.layers[step + 1].neurons;
    const layerWeights = weights[step];

    const nextActivations = [];
    for (let j = 0; j < nextLayerSize; j++) {
      let sum = 0;
      for (let i = 0; i < layerActivations.length; i++) {
        sum += layerActivations[i].value * layerWeights[i][j];
      }
      sum += 0.1;
      const activated = step < NETWORK.layers.length - 2 ? relu(sum) : sigmoid(sum);
      nextActivations.push({ value: activated, preActivation: sum });
    }

    currentActivations.push(nextActivations);
    setActivations(currentActivations);
    animateForwardPropagation(step);
    setStep(step + 1);
  };

  const backwardStep = () => {
    const backStep = NETWORK.layers.length - 2 - (step - (NETWORK.layers.length - 1));

    if (backStep < 0) {
      setMode('backward-complete');
      return;
    }

    const currentGradients = [...gradients];

    if (backStep === NETWORK.layers.length - 2) {
      const output = activations[activations.length - 1][0];
      const error = output.value - target;
      const delta = error * sigmoidDerivative(output.preActivation);
      currentGradients.push([delta]);
    } else {
      const nextGradients = currentGradients[currentGradients.length - 1];
      const layerWeights = weights[backStep + 1];
      const layerActivations = activations[backStep + 1];

      const layerGradients = [];
      for (let i = 0; i < layerActivations.length; i++) {
        let sum = 0;
        for (let j = 0; j < nextGradients.length; j++) {
          sum += nextGradients[j] * layerWeights[i][j];
        }
        const grad = layerActivations[i].preActivation > 0 ? sum : 0;
        layerGradients.push(grad);
      }
      currentGradients.push(layerGradients);
    }

    setGradients(currentGradients);
    animateBackwardPropagation(backStep);
    setStep(step + 1);
  };

  const animateForwardPropagation = (layerIdx) => {
    const layer = NETWORK.layers[layerIdx];
    const nextLayer = NETWORK.layers[layerIdx + 1];

    for (let i = 0; i < layer.neurons; i++) {
      for (let j = 0; j < nextLayer.neurons; j++) {
        const key = `${layerIdx}-${i}-${j}`;
        const line = connectionRefs.current[key];
        if (line) {
          gsap.fromTo(line,
            { strokeOpacity: 0.2, stroke: '#4ade80' },
            {
              strokeOpacity: 0.8,
              duration: 0.3,
              delay: i * 0.05,
              ease: 'power2.out',
              onComplete: () => {
                gsap.to(line, { strokeOpacity: 0.5, stroke: '#6366f1', duration: 0.2 });
              }
            }
          );
        }
      }
    }
  };

  const animateBackwardPropagation = (layerIdx) => {
    const layer = NETWORK.layers[layerIdx];
    const nextLayer = NETWORK.layers[layerIdx + 1];

    for (let i = 0; i < layer.neurons; i++) {
      for (let j = 0; j < nextLayer.neurons; j++) {
        const key = `${layerIdx}-${i}-${j}`;
        const line = connectionRefs.current[key];
        if (line) {
          gsap.fromTo(line,
            { stroke: '#ef4444', strokeOpacity: 0.8 },
            {
              stroke: '#f97316',
              strokeOpacity: 0.5,
              duration: 0.3,
              delay: j * 0.05,
              ease: 'power2.inOut'
            }
          );
        }
      }
    }
  };

  const updateWeights = () => {
    const learningRate = 0.1;
    const newWeights = weights.map(layer => layer.map(row => [...row]));

    for (let l = 0; l < weights.length; l++) {
      const gradientLayer = gradients[weights.length - 1 - l];
      if (!gradientLayer) continue;

      for (let i = 0; i < weights[l].length; i++) {
        for (let j = 0; j < weights[l][i].length; j++) {
          const activation = activations[l][i].value;
          const gradient = gradientLayer[j] || 0;
          newWeights[l][i][j] -= learningRate * gradient * activation;
        }
      }
    }

    setWeights(newWeights);
    setEpoch(epoch + 1);
  };

  const resetForNewIteration = () => {
    setActivations([input.map(x => ({ value: x, preActivation: x }))]);
    setGradients([]);
    setStep(0);
    setLoss(null);
    setMode('forward');
  };

  const renderConnections = () => {
    const connections = [];
    const neuronRadius = 20;

    for (let l = 0; l < NETWORK.layers.length - 1; l++) {
      const fromLayer = NETWORK.layers[l];
      const toLayer = NETWORK.layers[l + 1];

      for (let i = 0; i < fromLayer.neurons; i++) {
        for (let j = 0; j < toLayer.neurons; j++) {
          const from = getPosition(l, i);
          const to = getPosition(l + 1, j);
          const key = `${l}-${i}-${j}`;
          const weight = weights[l] ? weights[l][i]?.[j] : 0;
          const strokeWidth = Math.min(Math.abs(weight) * 1.5 + 0.5, 3);

          const dx = to.x - from.x;
          const dy = to.y - from.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          const unitX = dx / distance;
          const unitY = dy / distance;

          const x1 = from.x + unitX * neuronRadius;
          const y1 = from.y + unitY * neuronRadius;
          const x2 = to.x - unitX * neuronRadius;
          const y2 = to.y - unitY * neuronRadius;

          connections.push(
            <line
              key={key}
              ref={(el) => (connectionRefs.current[key] = el)}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke={weight >= 0 ? "#6366f1" : "#ef4444"}
              strokeWidth={strokeWidth}
              strokeOpacity={0.5}
            />
          );
        }
      }
    }
    return connections;
  };

  const renderNeurons = () => {
    const neurons = [];
    NETWORK.layers.forEach((layer, layerIdx) => {
      for (let i = 0; i < layer.neurons; i++) {
        const pos = getPosition(layerIdx, i);
        const key = `${layerIdx}-${i}`;
        const activation = activations[layerIdx] ? activations[layerIdx][i]?.value : 0;
        const gradient = gradients[NETWORK.layers.length - 2 - layerIdx]?.[i] || 0;

        neurons.push(
          <g key={key} transform={`translate(${pos.x}, ${pos.y})`}>
            <circle r={24} fill={layer.color} opacity={0.3} filter="url(#glow)" />
            <circle
              ref={(el) => (neuronRefs.current[key] = el)}
              r={20}
              fill={`url(#gradient-${layerIdx})`}
              stroke={layer.color}
              strokeWidth={2}
            />
            <text textAnchor="middle" dy="5" fill="white" fontSize="11" fontWeight="bold">
              {activation?.toFixed(2) || '0.00'}
            </text>
            {mode.includes('backward') && Math.abs(gradient) > 0.001 && (
              <text textAnchor="middle" dy="32" fill="#f97316" fontSize="9">
                δ={gradient.toFixed(3)}
              </text>
            )}
          </g>
        );
      }
    });
    return neurons;
  };

  const renderLayerLabels = () => {
    return NETWORK.layers.map((layer, idx) => {
      const pos = getPosition(idx, -0.8);
      return (
        <text key={idx} x={pos.x} y={25} textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">
          {layer.name}
        </text>
      );
    });
  };

  const renderGradientDefs = () => (
    <defs>
      <filter id="glow">
        <feGaussianBlur stdDeviation="4" result="coloredBlur" />
        <feMerge>
          <feMergeNode in="coloredBlur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
      {NETWORK.layers.map((layer, idx) => (
        <radialGradient key={idx} id={`gradient-${idx}`}>
          <stop offset="0%" stopColor={layer.color} />
          <stop offset="100%" stopColor={`${layer.color}88`} />
        </radialGradient>
      ))}
    </defs>
  );

  // Code viewer component with syntax highlighting
  const CodeViewer = ({ snippet }) => {
    useEffect(() => {
      Prism.highlightAll();
    }, [snippet]);

    // Format explanation with simple markdown-like styling
    const formatExplanation = (text) => {
      return text
        .split('\n')
        .map((line, idx) => {
          // Bold text
          let formatted = line.replace(/\*\*(.+?)\*\*/g, '<strong class="text-white">$1</strong>');
          // Inline code
          formatted = formatted.replace(/`(.+?)`/g, '<code class="bg-gray-700 px-1 rounded text-orange-300">$1</code>');
          // Emoji bullets
          if (line.trim().startsWith('🔹') || line.trim().startsWith('🔢') || line.trim().startsWith('🧮') ||
            line.trim().startsWith('📉') || line.trim().startsWith('⚡') || line.trim().startsWith('•')) {
            return <div key={idx} className="ml-2 mb-1" dangerouslySetInnerHTML={{ __html: formatted }} />;
          }
          return <div key={idx} dangerouslySetInnerHTML={{ __html: formatted }} />;
        });
    };

    return (
      <div className="bg-gray-900 rounded-xl overflow-hidden border border-gray-700 shadow-xl">
        <div className="flex items-center justify-between px-4 py-3 bg-gray-800 border-b border-gray-700">
          <div className="flex items-center gap-3">
            <div className="flex gap-1.5">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
            </div>
            <span className="text-gray-800 dark:text-sm font-mono">{snippet.file}</span>
          </div>
          <a
            href={`${GITHUB_BASE}/${snippet.file}#L${snippet.lineStart}-L${snippet.lineEnd}`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-xs text-blue-600 dark:text-blue-400 hover:text-blue-300 transition-colors"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
            </svg>
            View on GitHub
          </a>
        </div>
        <pre className="p-4 overflow-x-auto text-sm !bg-gray-900 !m-0">
          <code className="language-rust">{snippet.code}</code>
        </pre>
        <div className="px-4 py-4 bg-gradient-to-r from-gray-800/80 to-gray-800/50 border-t border-gray-700">
          <h4 className="text-sm font-semibold text-green-400 mb-3 flex items-center gap-2">
            <span className="text-lg">💡</span>
            Understanding the Code:
          </h4>
          <div className="text-gray-700 dark:text-sm space-y-1">
            {formatExplanation(snippet.explanation)}
          </div>
        </div>
      </div>
    );
  };

  // Intuition card component with markdown-like formatting
  const IntuitionCard = ({ card, isSelected, onClick }) => {
    const formatContent = (text) => {
      return text.split('\n').map((line, idx) => {
        // Bold text
        let formatted = line.replace(/\*\*(.+?)\*\*/g, '<strong class="text-white font-semibold">$1</strong>');
        // Inline code
        formatted = formatted.replace(/`(.+?)`/g, '<code class="bg-gray-700/70 px-1 rounded text-orange-300">$1</code>');
        // Headers with •
        if (line.trim().startsWith('•')) {
          return <div key={idx} className="ml-2 mb-1" dangerouslySetInnerHTML={{ __html: formatted }} />;
        }
        // Numbered lists
        if (/^\d+\./.test(line.trim())) {
          return <div key={idx} className="ml-2 mb-1" dangerouslySetInnerHTML={{ __html: formatted }} />;
        }
        return <div key={idx} dangerouslySetInnerHTML={{ __html: formatted }} />;
      });
    };

    return (
      <div
        onClick={onClick}
        className={`p-4 rounded-xl cursor-pointer transition-all duration-300 border ${isSelected
            ? 'bg-indigo-900/50 border-indigo-500 shadow-lg shadow-indigo-500/20'
            : 'bg-gray-800/50 border-gray-700 hover:border-gray-600 hover:bg-gray-800/70'
          }`}
      >
        <div className="flex items-center gap-3 mb-2">
          <span className="text-2xl">{card.icon}</span>
          <h3 className="font-semibold text-white">{card.title}</h3>
          <span className={`ml-auto text-gray-700 dark:text-gray-500 transition-transform duration-300 ${isSelected ? 'rotate-180' : ''}`}>
            ▼
          </span>
        </div>
        {isSelected && (
          <div className="text-gray-700 dark:text-sm mt-3 space-y-1 animate-fadeIn">
            {formatContent(card.content)}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="text-white">
      <div className="max-w-7xl mx-auto">
        {/* Tab Navigation */}
        <div className="flex justify-center gap-2 mb-6">
          {[
            { id: 'visualize', label: '🎬 Interactive', desc: 'Step through the network' },
            { id: 'intuition', label: '💡 Intuition', desc: 'Understand the concepts' },
            { id: 'code', label: '🦀 Rust Code', desc: 'See the implementation' },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-6 py-3 rounded-xl transition-all duration-300 ${activeTab === tab.id
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/30'
                  : 'bg-gray-800 text-gray-800 dark:text-gray-400 hover:bg-gray-700 hover:text-white'
                }`}
            >
              <div className="font-semibold">{tab.label}</div>
              <div className="text-xs opacity-70">{tab.desc}</div>
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {activeTab === 'visualize' && (
          <div className="space-y-6">
            {/* Control Panel */}
            <div className="bg-gray-800/50 rounded-xl p-4 backdrop-blur-sm border border-gray-700">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-4">
                  <div>
                    <label className="text-gray-800 dark:text-xs block mb-1">Input (XOR)</label>
                    <select
                      className="bg-gray-700 text-white rounded px-3 py-2 text-sm"
                      value={`${input[0]},${input[1]}`}
                      onChange={(e) => {
                        const [x1, x2] = e.target.value.split(',').map(Number);
                        setInput([x1, x2]);
                        setTarget(x1 ^ x2);
                        setActivations([[{ value: x1, preActivation: x1 }, { value: x2, preActivation: x2 }]]);
                        setStep(0);
                        setMode('intro');
                        setGradients([]);
                        setLoss(null);
                      }}
                    >
                      <option value="0,0">0 XOR 0 = 0</option>
                      <option value="0,1">0 XOR 1 = 1</option>
                      <option value="1,0">1 XOR 0 = 1</option>
                      <option value="1,1">1 XOR 1 = 0</option>
                    </select>
                  </div>

                  <div className="text-center">
                    <label className="text-gray-800 dark:text-xs block mb-1">Target</label>
                    <div className="bg-gray-700 text-green-400 rounded px-4 py-2 font-mono text-sm">
                      {target}
                    </div>
                  </div>

                  {loss !== null && (
                    <div className="text-center">
                      <label className="text-gray-800 dark:text-xs block mb-1">Loss</label>
                      <div className="bg-gray-700 text-red-400 rounded px-4 py-2 font-mono text-sm">
                        {loss.toFixed(4)}
                      </div>
                    </div>
                  )}

                  <div className="text-center">
                    <label className="text-gray-800 dark:text-xs block mb-1">Epoch</label>
                    <div className="bg-gray-700 text-yellow-400 rounded px-4 py-2 font-mono text-sm">
                      {epoch}
                    </div>
                  </div>
                </div>

                <div className="flex gap-2">
                  <button onClick={initializeWeights}
                    className="px-4 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded-lg text-sm">
                    Reset
                  </button>

                  {mode === 'intro' && (
                    <button onClick={() => { setMode('forward'); setStep(0); }}
                      className="px-5 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-sm font-medium">
                      Start Forward →
                    </button>
                  )}

                  {mode === 'forward' && (
                    <button onClick={forwardStep}
                      className="px-5 py-2 bg-green-600 hover:bg-green-500 text-white rounded-lg text-sm font-medium">
                      Forward Step →
                    </button>
                  )}

                  {mode === 'forward-complete' && (
                    <button onClick={() => { setMode('backward'); setStep(NETWORK.layers.length - 1); }}
                      className="px-5 py-2 bg-orange-600 hover:bg-orange-500 text-white rounded-lg text-sm font-medium">
                      Start Backward ←
                    </button>
                  )}

                  {mode === 'backward' && (
                    <button onClick={backwardStep}
                      className="px-5 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg text-sm font-medium">
                      ← Backward Step
                    </button>
                  )}

                  {mode === 'backward-complete' && (
                    <button onClick={() => { updateWeights(); resetForNewIteration(); }}
                      className="px-5 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-sm font-medium">
                      Update & Train Again
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* Network Visualization */}
            <div className="bg-gray-800/30 rounded-xl p-4 backdrop-blur-sm border border-gray-700">
              <svg ref={svgRef} viewBox="0 0 800 400" className="w-full h-auto">
                {renderGradientDefs()}
                {renderConnections()}
                {renderNeurons()}
                {renderLayerLabels()}
              </svg>
            </div>

            {/* Quick Info Panels */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
                <h3 className="text-lg font-bold text-green-400 mb-2">➡️ Forward Pass</h3>
                <div className="text-gray-700 dark:text-sm font-mono space-y-1">
                  <p>z<sup>(l)</sup> = W<sup>(l)</sup> · a<sup>(l-1)</sup> + b</p>
                  <p>a<sup>(l)</sup> = σ(z<sup>(l)</sup>)</p>
                </div>
                <p className="text-gray-700 dark:text-xs mt-2">
                  Compute weighted sum, apply activation
                </p>
              </div>

              <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
                <h3 className="text-lg font-bold text-orange-600 dark:text-orange-400 mb-2">⬅️ Backward Pass</h3>
                <div className="text-gray-700 dark:text-sm font-mono space-y-1">
                  <p>δ<sup>(L)</sup> = ∇L · σ'(z<sup>(L)</sup>)</p>
                  <p>δ<sup>(l)</sup> = (W<sup>(l+1)</sup>)<sup>T</sup>δ<sup>(l+1)</sup> ⊙ σ'</p>
                </div>
                <p className="text-gray-700 dark:text-xs mt-2">
                  Compute gradients via chain rule
                </p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'intuition' && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {INTUITION_CARDS.map(card => (
                <IntuitionCard
                  key={card.id}
                  card={card}
                  isSelected={selectedIntuition === card.id}
                  onClick={() => setSelectedIntuition(selectedIntuition === card.id ? null : card.id)}
                />
              ))}
            </div>

            {/* Results highlight */}
            <div className="bg-gradient-to-r from-green-900/30 to-blue-900/30 rounded-xl p-6 border border-green-700/50">
              <h3 className="text-xl font-bold text-white mb-4">🏆 Mini-NN Results</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-gray-800/50 rounded-lg">
                  <div className="text-3xl font-bold text-green-400">100%</div>
                  <div className="text-gray-800 dark:text-sm">XOR Accuracy</div>
                </div>
                <div className="text-center p-4 bg-gray-800/50 rounded-lg">
                  <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">84.3%</div>
                  <div className="text-gray-800 dark:text-sm">Titanic Accuracy</div>
                </div>
                <div className="text-center p-4 bg-gray-800/50 rounded-lg">
                  <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">993</div>
                  <div className="text-gray-800 dark:text-sm">Parameters</div>
                </div>
                <div className="text-center p-4 bg-gray-800/50 rounded-lg">
                  <div className="text-3xl font-bold text-yellow-400">0</div>
                  <div className="text-gray-800 dark:text-sm">ML Frameworks</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'code' && (
          <div className="space-y-6">
            {/* Code topic selector */}
            <div className="flex flex-wrap gap-2">
              {Object.entries(CODE_SNIPPETS).map(([key, snippet]) => (
                <button
                  key={key}
                  onClick={() => setSelectedCode(key)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${selectedCode === key
                      ? 'bg-indigo-600 text-white'
                      : 'bg-gray-800 text-gray-800 dark:text-gray-400 hover:bg-gray-700 hover:text-white'
                    }`}
                >
                  {snippet.title}
                </button>
              ))}
            </div>

            {/* Selected code viewer */}
            <CodeViewer snippet={CODE_SNIPPETS[selectedCode]} />

            {/* Mini-nn project link */}
            <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-bold text-white mb-1">🦀 Full Implementation</h3>
                  <p className="text-gray-800 dark:text-sm">
                    Explore the complete neural network implementation in Rust
                  </p>
                </div>
                <a
                  href="https://github.com/danielsobrado/ml-animations/tree/main/mini-nn"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-6 py-3 bg-orange-600 hover:bg-orange-500 text-white rounded-lg font-medium transition-colors flex items-center gap-2"
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                  </svg>
                  View mini-nn on GitHub
                </a>
              </div>

              {/* File structure */}
              <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                {[
                  { file: 'tensor.rs', desc: 'Matrix operations' },
                  { file: 'activation.rs', desc: 'Activation functions' },
                  { file: 'layer.rs', desc: 'Dense layers' },
                  { file: 'loss.rs', desc: 'Loss functions' },
                  { file: 'optimizer.rs', desc: 'SGD, Adam' },
                  { file: 'network.rs', desc: 'Network builder' },
                  { file: 'training.rs', desc: 'Training loop' },
                  { file: 'data.rs', desc: 'Data loading' },
                ].map(item => (
                  <a
                    key={item.file}
                    href={`${GITHUB_BASE}/${item.file}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 p-2 bg-gray-700/50 rounded hover:bg-gray-700 transition-colors"
                  >
                    <span className="text-orange-600 dark:text-orange-400">📄</span>
                    <div>
                      <div className="text-white font-mono text-xs">{item.file}</div>
                      <div className="text-gray-700 dark:text-xs">{item.desc}</div>
                    </div>
                  </a>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="mt-8 text-center text-gray-700 dark:text-sm">
          Built with React + GSAP | Rust implementation:{' '}
          <a href="https://github.com/danielsobrado/ml-animations/tree/main/mini-nn"
            className="text-blue-600 dark:text-blue-400 hover:text-blue-300">mini-nn</a>
          {' '}| 84.3% Titanic accuracy, 100% XOR accuracy
        </div>
      </div>
    </div>
  );
}

export default App;
