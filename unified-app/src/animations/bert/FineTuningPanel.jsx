import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Settings, ArrowRight, Layers } from 'lucide-react';

export default function FineTuningPanel() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedTask, setSelectedTask] = useState('classification');

  const steps = [
    { title: 'Load Pre-trained BERT', description: 'Start with BERT weights trained on MLM + NSP (110M parameters for Base).' },
    { title: 'Add Task-Specific Head', description: 'Add a small neural network on top for your specific task (e.g., classifier).' },
    { title: 'Prepare Task Data', description: 'Format your labeled dataset: add [CLS], [SEP] tokens, segment IDs.' },
    { title: 'Fine-tune All Layers', description: 'Train the entire model end-to-end on your task with a small learning rate.' },
    { title: 'Evaluate & Deploy', description: 'Test on held-out data. Fine-tuning takes minutes to hours, not days!' },
  ];

  const tasks = {
    classification: {
      name: 'Sequence Classification',
      description: 'Classify entire text (sentiment, topic, intent)',
      inputFormat: '[CLS] This movie was amazing! [SEP]',
      outputFormat: 'Single label from [CLS]',
      examples: ['Sentiment Analysis', 'Spam Detection', 'Topic Classification'],
      head: 'Linear(768, num_classes)',
      color: 'blue'
    },
    ner: {
      name: 'Token Classification (NER)',
      description: 'Label each token (named entities, POS tags)',
      inputFormat: '[CLS] John works at Google [SEP]',
      outputFormat: 'Label per token',
      examples: ['Named Entity Recognition', 'POS Tagging', 'Chunking'],
      head: 'Linear(768, num_tags) per token',
      color: 'green'
    },
    qa: {
      name: 'Question Answering',
      description: 'Find answer span in context passage',
      inputFormat: '[CLS] Question [SEP] Context passage [SEP]',
      outputFormat: 'Start & end positions',
      examples: ['SQuAD', 'Reading Comprehension', 'Extractive QA'],
      head: 'Linear(768, 2) for start/end',
      color: 'purple'
    },
    nli: {
      name: 'Sentence Pair Classification',
      description: 'Classify relationship between two sentences',
      inputFormat: '[CLS] Sentence A [SEP] Sentence B [SEP]',
      outputFormat: 'Entailment/Contradiction/Neutral',
      examples: ['Natural Language Inference', 'Paraphrase Detection', 'Semantic Similarity'],
      head: 'Linear(768, 3) from [CLS]',
      color: 'orange'
    }
  };

  const currentTask = tasks[selectedTask];

  useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= steps.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 2500);
      return () => clearInterval(interval);
    }
  }, [isPlaying]);

  const reset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
  };

  const getColorClasses = (color) => ({
    bg: `bg-${color}-900/30`,
    border: `border-${color}-500/50`,
    text: `text-${color}-400`,
    button: `bg-${color}-600`
  });

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="text-yellow-400">Fine-Tuning</span> BERT for Downstream Tasks
        </h2>
        <p className="text-gray-400">
          Transfer learning: adapt pre-trained BERT to specific tasks with minimal training
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap justify-center gap-3">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="flex items-center gap-2 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded-lg transition-colors"
        >
          {isPlaying ? <Pause size={18} /> : <Play size={18} />}
          {isPlaying ? 'Pause' : 'Play Animation'}
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
        >
          <RotateCcw size={18} />
          Reset
        </button>
      </div>

      {/* Step Progress */}
      <div className="flex flex-wrap justify-center gap-2">
        {steps.map((step, i) => (
          <button
            key={i}
            onClick={() => { setCurrentStep(i); setIsPlaying(false); }}
            className={`px-3 py-1 rounded-full text-sm transition-all ${
              i === currentStep 
                ? 'bg-yellow-500 text-black' 
                : i < currentStep 
                ? 'bg-yellow-900 text-yellow-300' 
                : 'bg-white/10 text-gray-500'
            }`}
          >
            {i + 1}. {step.title}
          </button>
        ))}
      </div>

      {/* Current Step Description */}
      <div className="bg-yellow-900/20 border border-yellow-500/30 rounded-xl p-4">
        <h3 className="font-bold text-yellow-400">Step {currentStep + 1}: {steps[currentStep].title}</h3>
        <p className="text-gray-300 mt-1">{steps[currentStep].description}</p>
      </div>

      {/* Fine-Tuning Pipeline Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10 overflow-x-auto">
        <div className="flex items-center justify-center gap-4 min-w-max">
          {/* Pre-trained BERT */}
          <div className={`text-center transition-all duration-500 ${currentStep >= 0 ? 'opacity-100 scale-100' : 'opacity-30 scale-90'}`}>
            <div className="bg-blue-900/30 border-2 border-blue-500 rounded-xl p-4 w-48">
              <Layers className="mx-auto mb-2 text-blue-400" size={32} />
              <p className="font-bold text-blue-300">Pre-trained BERT</p>
              <p className="text-xs text-gray-400 mt-1">12 Encoder Layers</p>
              <p className="text-xs text-gray-400">110M Parameters</p>
            </div>
          </div>

          <ArrowRight className={`text-gray-500 transition-all duration-500 ${currentStep >= 1 ? 'opacity-100' : 'opacity-30'}`} />

          {/* Task Head */}
          <div className={`text-center transition-all duration-500 ${currentStep >= 1 ? 'opacity-100 scale-100' : 'opacity-30 scale-90'}`}>
            <div className={`bg-${currentTask.color}-900/30 border-2 border-${currentTask.color}-500 rounded-xl p-4 w-48`}>
              <Settings className={`mx-auto mb-2 text-${currentTask.color}-400`} size={32} />
              <p className={`font-bold text-${currentTask.color}-300`}>+ Task Head</p>
              <p className="text-xs text-gray-400 mt-1">{currentTask.head}</p>
              <p className="text-xs text-gray-400">~Few K params</p>
            </div>
          </div>

          <ArrowRight className={`text-gray-500 transition-all duration-500 ${currentStep >= 2 ? 'opacity-100' : 'opacity-30'}`} />

          {/* Task Data */}
          <div className={`text-center transition-all duration-500 ${currentStep >= 2 ? 'opacity-100 scale-100' : 'opacity-30 scale-90'}`}>
            <div className="bg-green-900/30 border-2 border-green-500 rounded-xl p-4 w-48">
              <div className="text-2xl mb-2">📊</div>
              <p className="font-bold text-green-300">Task Dataset</p>
              <p className="text-xs text-gray-400 mt-1">Labeled examples</p>
              <p className="text-xs text-gray-400">~1K - 100K samples</p>
            </div>
          </div>

          <ArrowRight className={`text-gray-500 transition-all duration-500 ${currentStep >= 3 ? 'opacity-100' : 'opacity-30'}`} />

          {/* Fine-tuned Model */}
          <div className={`text-center transition-all duration-500 ${currentStep >= 3 ? 'opacity-100 scale-100' : 'opacity-30 scale-90'}`}>
            <div className="bg-yellow-900/30 border-2 border-yellow-500 rounded-xl p-4 w-48">
              <div className="text-2xl mb-2">🎯</div>
              <p className="font-bold text-yellow-300">Fine-tuned Model</p>
              <p className="text-xs text-gray-400 mt-1">Task-specific</p>
              <p className="text-xs text-gray-400">2-4 epochs</p>
            </div>
          </div>

          <ArrowRight className={`text-gray-500 transition-all duration-500 ${currentStep >= 4 ? 'opacity-100' : 'opacity-30'}`} />

          {/* Deployed */}
          <div className={`text-center transition-all duration-500 ${currentStep >= 4 ? 'opacity-100 scale-100' : 'opacity-30 scale-90'}`}>
            <div className="bg-purple-900/30 border-2 border-purple-500 rounded-xl p-4 w-48">
              <div className="text-2xl mb-2">🚀</div>
              <p className="font-bold text-purple-300">Production</p>
              <p className="text-xs text-gray-400 mt-1">Ready for inference</p>
              <p className="text-xs text-gray-400">State-of-the-art!</p>
            </div>
          </div>
        </div>
      </div>

      {/* Task Selection */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-lg font-bold mb-4 text-center">Select a Downstream Task</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
          {Object.entries(tasks).map(([key, task]) => (
            <button
              key={key}
              onClick={() => setSelectedTask(key)}
              className={`p-3 rounded-xl border-2 transition-all ${
                selectedTask === key 
                  ? `bg-${task.color}-900/30 border-${task.color}-500 scale-105` 
                  : 'bg-white/5 border-white/10 hover:border-white/30'
              }`}
            >
              <p className={`font-bold text-sm ${selectedTask === key ? `text-${task.color}-400` : 'text-gray-300'}`}>
                {task.name}
              </p>
            </button>
          ))}
        </div>

        {/* Selected Task Details */}
        <div className={`bg-${currentTask.color}-900/20 rounded-xl p-4 border border-${currentTask.color}-500/30`}>
          <h4 className={`font-bold text-${currentTask.color}-400 mb-3`}>{currentTask.name}</h4>
          <p className="text-sm text-gray-300 mb-4">{currentTask.description}</p>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-gray-400 mb-1">Input Format:</p>
              <code className="text-xs bg-black/30 px-2 py-1 rounded text-green-300 block">
                {currentTask.inputFormat}
              </code>
            </div>
            <div>
              <p className="text-xs text-gray-400 mb-1">Output:</p>
              <code className="text-xs bg-black/30 px-2 py-1 rounded text-orange-300 block">
                {currentTask.outputFormat}
              </code>
            </div>
          </div>

          <div className="mt-4">
            <p className="text-xs text-gray-400 mb-2">Example Tasks:</p>
            <div className="flex flex-wrap gap-2">
              {currentTask.examples.map((ex, i) => (
                <span key={i} className="text-xs bg-white/10 px-2 py-1 rounded">
                  {ex}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Fine-Tuning Tips */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-green-900/20 rounded-xl p-4 border border-green-500/30">
          <h4 className="font-bold text-green-400 mb-3">✅ Best Practices</h4>
          <ul className="text-sm text-gray-300 space-y-2">
            <li>• <strong>Learning rate:</strong> 2e-5 to 5e-5 (small!)</li>
            <li>• <strong>Batch size:</strong> 16 or 32</li>
            <li>• <strong>Epochs:</strong> 2-4 (avoid overfitting)</li>
            <li>• <strong>Warmup:</strong> 10% of training steps</li>
            <li>• <strong>Max length:</strong> Task-dependent (128-512)</li>
          </ul>
        </div>
        <div className="bg-red-900/20 rounded-xl p-4 border border-red-500/30">
          <h4 className="font-bold text-red-400 mb-3">⚠️ Common Mistakes</h4>
          <ul className="text-sm text-gray-300 space-y-2">
            <li>• Learning rate too high → catastrophic forgetting</li>
            <li>• Too many epochs → overfitting</li>
            <li>• Wrong tokenizer → vocabulary mismatch</li>
            <li>• No [CLS]/[SEP] tokens → poor performance</li>
            <li>• Sequence too long → truncation issues</li>
          </ul>
        </div>
      </div>

      {/* What Gets Updated */}
      <div className="bg-gradient-to-r from-blue-900/20 to-yellow-900/20 rounded-xl p-4 border border-white/20">
        <h4 className="font-bold text-white mb-3">🔄 What Gets Updated During Fine-Tuning?</h4>
        <div className="grid md:grid-cols-3 gap-4 text-sm">
          <div className="bg-black/30 rounded-lg p-3">
            <p className="text-blue-400 font-medium mb-1">BERT Weights (All)</p>
            <p className="text-gray-400 text-xs">All 110M parameters get updated with small gradients</p>
          </div>
          <div className="bg-black/30 rounded-lg p-3">
            <p className="text-yellow-400 font-medium mb-1">Task Head (New)</p>
            <p className="text-gray-400 text-xs">New layers trained from scratch</p>
          </div>
          <div className="bg-black/30 rounded-lg p-3">
            <p className="text-green-400 font-medium mb-1">Option: Freeze BERT</p>
            <p className="text-gray-400 text-xs">Only train head (faster, less accurate)</p>
          </div>
        </div>
      </div>

      {/* GLUE Benchmark Results */}
      <div className="bg-black/30 rounded-xl p-4 border border-white/10">
        <h4 className="font-bold text-yellow-400 mb-3">📈 BERT Performance on GLUE Benchmark</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/20">
                <th className="text-left py-2 text-gray-400">Task</th>
                <th className="text-center py-2 text-gray-400">Dataset</th>
                <th className="text-center py-2 text-gray-400">BERT-Base</th>
                <th className="text-center py-2 text-gray-400">BERT-Large</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-white/10">
                <td className="py-2">Sentiment</td>
                <td className="text-center text-gray-400">SST-2</td>
                <td className="text-center font-mono text-green-400">93.5%</td>
                <td className="text-center font-mono text-green-400">94.9%</td>
              </tr>
              <tr className="border-b border-white/10">
                <td className="py-2">NLI</td>
                <td className="text-center text-gray-400">MNLI</td>
                <td className="text-center font-mono text-green-400">84.6%</td>
                <td className="text-center font-mono text-green-400">86.7%</td>
              </tr>
              <tr className="border-b border-white/10">
                <td className="py-2">Paraphrase</td>
                <td className="text-center text-gray-400">QQP</td>
                <td className="text-center font-mono text-green-400">71.2%</td>
                <td className="text-center font-mono text-green-400">72.1%</td>
              </tr>
              <tr>
                <td className="py-2">Question NLI</td>
                <td className="text-center text-gray-400">QNLI</td>
                <td className="text-center font-mono text-green-400">90.5%</td>
                <td className="text-center font-mono text-green-400">92.7%</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-black/40 rounded-xl p-4 border border-white/10">
        <p className="text-sm text-gray-400 mb-3">🐍 PyTorch Fine-Tuning Example (Classification):</p>
        <pre className="text-sm overflow-x-auto">
          <code className="text-orange-300">{`from transformers import BertForSequenceClassification, BertTokenizer, AdamW

# Load pre-trained BERT with classification head
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare data
texts = ["I love this movie!", "This was terrible."]
labels = [1, 0]  # Positive, Negative

# Tokenize
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
inputs['labels'] = torch.tensor(labels)

# Fine-tuning setup
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
model.train()
for epoch in range(3):
    optimizer.zero_grad()
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Inference
model.eval()
with torch.no_grad():
    test_input = tokenizer("Great performance!", return_tensors='pt')
    output = model(**test_input)
    prediction = torch.argmax(output.logits, dim=1)
    print(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")`}</code>
        </pre>
      </div>
    </div>
  );
}
