import React, { useState } from 'react';
import { MessageSquare, Search, Tag, FileText, CheckCircle, ArrowRight } from 'lucide-react';

export default function TasksPanel() {
  const [selectedTask, setSelectedTask] = useState('sentiment');

  const tasks = {
    sentiment: {
      name: 'Sentiment Analysis',
      icon: '😊',
      description: 'Classify text as positive, negative, or neutral',
      example: {
        input: 'The new restaurant downtown has amazing food and great service!',
        process: ['Tokenize', 'Add [CLS]/[SEP]', 'Run through BERT', 'Classify [CLS] output'],
        output: 'Positive (95.3%)'
      },
      inputFormat: '[CLS] Review text [SEP]',
      outputType: 'Single class from [CLS] embedding',
      applications: ['Product reviews', 'Social media monitoring', 'Customer feedback'],
      metrics: 'Accuracy: 94-95% on SST-2'
    },
    ner: {
      name: 'Named Entity Recognition',
      icon: '🏷️',
      description: 'Identify and classify named entities (person, organization, location)',
      example: {
        input: 'Elon Musk founded SpaceX in Hawthorne, California.',
        entities: [
          { text: 'Elon Musk', type: 'PERSON', color: 'blue' },
          { text: 'SpaceX', type: 'ORG', color: 'green' },
          { text: 'Hawthorne', type: 'LOC', color: 'purple' },
          { text: 'California', type: 'LOC', color: 'purple' }
        ],
        process: ['Tokenize each word', 'Classify each token', 'Handle subwords (BIO tagging)'],
        output: 'Entity labels per token'
      },
      inputFormat: '[CLS] Sentence with entities [SEP]',
      outputType: 'Label per token (B-PER, I-PER, B-ORG, etc.)',
      applications: ['Information extraction', 'Knowledge graphs', 'Search engines'],
      metrics: 'F1: 92-93% on CoNLL-2003'
    },
    qa: {
      name: 'Question Answering',
      icon: '❓',
      description: 'Find answer span within a context passage',
      example: {
        question: 'When was BERT released?',
        context: 'BERT was introduced by Google AI in October 2018. It quickly became one of the most influential NLP models.',
        answer: 'October 2018',
        answerStart: 37,
        answerEnd: 49,
        process: ['Combine Q + Context', 'Predict start/end positions', 'Extract span'],
        output: 'Answer span: "October 2018"'
      },
      inputFormat: '[CLS] Question [SEP] Context passage [SEP]',
      outputType: 'Start & end position indices',
      applications: ['Search engines', 'Customer support', 'Educational tools'],
      metrics: 'F1: 88-90% on SQuAD 2.0'
    },
    nli: {
      name: 'Natural Language Inference',
      icon: '🔗',
      description: 'Determine relationship between two sentences',
      example: {
        premise: 'A man is playing guitar on stage.',
        hypothesis: 'A musician is performing.',
        process: ['Combine premise + hypothesis', 'Classify relationship from [CLS]'],
        output: 'Entailment (89.2%)',
        labels: ['Entailment', 'Contradiction', 'Neutral']
      },
      inputFormat: '[CLS] Premise [SEP] Hypothesis [SEP]',
      outputType: '3-way classification: Entailment/Contradiction/Neutral',
      applications: ['Fact verification', 'Textual reasoning', 'Summarization evaluation'],
      metrics: 'Accuracy: 86-87% on MNLI'
    },
    similarity: {
      name: 'Semantic Similarity',
      icon: '📊',
      description: 'Measure how similar two sentences are in meaning',
      example: {
        sentA: 'How do I reset my password?',
        sentB: 'I forgot my login credentials.',
        process: ['Encode both sentences', 'Compare [CLS] embeddings', 'Compute similarity score'],
        output: 'Similarity: 0.87 (High)',
        scale: '0 = Not similar, 1 = Very similar'
      },
      inputFormat: '[CLS] Sentence A [SEP] Sentence B [SEP]',
      outputType: 'Similarity score (0-1) or regression',
      applications: ['Duplicate detection', 'Search ranking', 'FAQ matching'],
      metrics: 'Pearson: 0.89 on STS-B'
    }
  };

  const currentTask = tasks[selectedTask];

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          BERT for <span className="text-yellow-400">NLP Tasks</span>
        </h2>
        <p className="text-gray-400">
          See how BERT handles different natural language understanding tasks
        </p>
      </div>

      {/* Task Selection */}
      <div className="flex flex-wrap justify-center gap-2">
        {Object.entries(tasks).map(([key, task]) => (
          <button
            key={key}
            onClick={() => setSelectedTask(key)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all ${
              selectedTask === key 
                ? 'bg-yellow-600 text-white scale-105' 
                : 'bg-white/10 hover:bg-white/20'
            }`}
          >
            <span className="text-xl">{task.icon}</span>
            <span className="text-sm font-medium">{task.name}</span>
          </button>
        ))}
      </div>

      {/* Task Description */}
      <div className="bg-yellow-900/20 border border-yellow-500/30 rounded-xl p-4 text-center">
        <h3 className="text-xl font-bold text-yellow-400">{currentTask.icon} {currentTask.name}</h3>
        <p className="text-gray-300 mt-1">{currentTask.description}</p>
      </div>

      {/* Main Task Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        {/* Sentiment Analysis */}
        {selectedTask === 'sentiment' && (
          <div className="space-y-4">
            <div className="bg-blue-900/20 rounded-xl p-4 border border-blue-500/30">
              <p className="text-sm text-gray-400 mb-2">Input Text:</p>
              <p className="text-lg text-blue-300">"{currentTask.example.input}"</p>
            </div>
            
            <div className="flex justify-center">
              <ArrowRight className="text-gray-500 rotate-90" size={24} />
            </div>

            <div className="flex justify-center gap-2 text-sm">
              {currentTask.example.process.map((step, i) => (
                <React.Fragment key={i}>
                  <span className="bg-gray-700 px-3 py-1 rounded">{step}</span>
                  {i < currentTask.example.process.length - 1 && <span className="text-gray-500">→</span>}
                </React.Fragment>
              ))}
            </div>

            <div className="flex justify-center">
              <ArrowRight className="text-gray-500 rotate-90" size={24} />
            </div>

            <div className="bg-green-900/20 rounded-xl p-4 border border-green-500/30 text-center">
              <p className="text-sm text-gray-400 mb-2">Output:</p>
              <p className="text-2xl font-bold text-green-400">{currentTask.example.output}</p>
              <div className="mt-3 flex justify-center gap-4">
                <span className="px-3 py-1 bg-green-600/30 rounded text-green-300 text-sm">Positive</span>
                <span className="px-3 py-1 bg-gray-600/30 rounded text-gray-400 text-sm">Neutral</span>
                <span className="px-3 py-1 bg-red-600/30 rounded text-gray-400 text-sm">Negative</span>
              </div>
            </div>
          </div>
        )}

        {/* NER */}
        {selectedTask === 'ner' && (
          <div className="space-y-4">
            <div className="bg-blue-900/20 rounded-xl p-4 border border-blue-500/30">
              <p className="text-sm text-gray-400 mb-2">Input Text:</p>
              <p className="text-lg">
                <span className="bg-blue-600/40 px-1 rounded text-blue-200">Elon Musk</span>{' '}
                founded{' '}
                <span className="bg-green-600/40 px-1 rounded text-green-200">SpaceX</span>{' '}
                in{' '}
                <span className="bg-purple-600/40 px-1 rounded text-purple-200">Hawthorne</span>,{' '}
                <span className="bg-purple-600/40 px-1 rounded text-purple-200">California</span>.
              </p>
            </div>

            <div className="flex justify-center">
              <ArrowRight className="text-gray-500 rotate-90" size={24} />
            </div>

            <div className="bg-gray-800/50 rounded-xl p-4">
              <p className="text-sm text-gray-400 mb-3">Entity Labels (BIO format):</p>
              <div className="flex flex-wrap gap-2 text-sm font-mono">
                {['[CLS]', 'Elon', 'Musk', 'founded', 'Space', '##X', 'in', 'Haw', '##thorne', ',', 'California', '.', '[SEP]'].map((token, i) => (
                  <div key={i} className="flex flex-col items-center">
                    <span className="bg-gray-700 px-2 py-1 rounded">{token}</span>
                    <span className={`text-xs mt-1 ${
                      ['Elon'].includes(token) ? 'text-blue-400' :
                      ['Musk'].includes(token) ? 'text-blue-300' :
                      ['Space'].includes(token) ? 'text-green-400' :
                      ['##X'].includes(token) ? 'text-green-300' :
                      ['Haw'].includes(token) ? 'text-purple-400' :
                      ['##thorne', 'California'].includes(token) ? 'text-purple-300' :
                      'text-gray-500'
                    }`}>
                      {['Elon'].includes(token) ? 'B-PER' :
                       ['Musk'].includes(token) ? 'I-PER' :
                       ['Space'].includes(token) ? 'B-ORG' :
                       ['##X'].includes(token) ? 'I-ORG' :
                       ['Haw'].includes(token) ? 'B-LOC' :
                       ['##thorne', 'California'].includes(token) ? 'I-LOC' :
                       'O'}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3 text-center">
              <div className="bg-blue-900/30 rounded-lg p-2 border border-blue-500/30">
                <p className="text-blue-400 font-medium">PERSON</p>
                <p className="text-sm text-gray-300">Elon Musk</p>
              </div>
              <div className="bg-green-900/30 rounded-lg p-2 border border-green-500/30">
                <p className="text-green-400 font-medium">ORGANIZATION</p>
                <p className="text-sm text-gray-300">SpaceX</p>
              </div>
              <div className="bg-purple-900/30 rounded-lg p-2 border border-purple-500/30">
                <p className="text-purple-400 font-medium">LOCATION</p>
                <p className="text-sm text-gray-300">Hawthorne, California</p>
              </div>
            </div>
          </div>
        )}

        {/* Question Answering */}
        {selectedTask === 'qa' && (
          <div className="space-y-4">
            <div className="bg-purple-900/20 rounded-xl p-4 border border-purple-500/30">
              <p className="text-sm text-gray-400 mb-1">Question:</p>
              <p className="text-lg text-purple-300 font-medium">{currentTask.example.question}</p>
            </div>

            <div className="bg-blue-900/20 rounded-xl p-4 border border-blue-500/30">
              <p className="text-sm text-gray-400 mb-1">Context:</p>
              <p className="text-lg">
                BERT was introduced by Google AI in{' '}
                <span className="bg-green-500/40 px-1 rounded text-green-200 font-bold">October 2018</span>
                . It quickly became one of the most influential NLP models.
              </p>
            </div>

            <div className="bg-gray-800/50 rounded-xl p-4">
              <p className="text-sm text-gray-400 mb-2">BERT predicts:</p>
              <div className="flex gap-4 justify-center">
                <div className="text-center">
                  <p className="text-xs text-gray-500">Start Position</p>
                  <p className="text-2xl font-mono text-yellow-400">7</p>
                  <p className="text-xs text-gray-400">"October"</p>
                </div>
                <div className="text-center">
                  <p className="text-xs text-gray-500">End Position</p>
                  <p className="text-2xl font-mono text-yellow-400">8</p>
                  <p className="text-xs text-gray-400">"2018"</p>
                </div>
              </div>
            </div>

            <div className="bg-green-900/20 rounded-xl p-4 border border-green-500/30 text-center">
              <p className="text-sm text-gray-400 mb-1">Answer:</p>
              <p className="text-2xl font-bold text-green-400">"{currentTask.example.answer}"</p>
            </div>
          </div>
        )}

        {/* NLI */}
        {selectedTask === 'nli' && (
          <div className="space-y-4">
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-blue-900/20 rounded-xl p-4 border border-blue-500/30">
                <p className="text-sm text-blue-400 mb-1">Premise:</p>
                <p className="text-lg text-gray-200">{currentTask.example.premise}</p>
              </div>
              <div className="bg-purple-900/20 rounded-xl p-4 border border-purple-500/30">
                <p className="text-sm text-purple-400 mb-1">Hypothesis:</p>
                <p className="text-lg text-gray-200">{currentTask.example.hypothesis}</p>
              </div>
            </div>

            <div className="flex justify-center">
              <ArrowRight className="text-gray-500 rotate-90" size={24} />
            </div>

            <div className="bg-gray-800/50 rounded-xl p-4 text-center">
              <p className="text-sm text-gray-400 mb-3">Does the premise entail the hypothesis?</p>
              <div className="flex justify-center gap-4">
                <div className="bg-green-600/30 px-4 py-2 rounded-lg border-2 border-green-500">
                  <p className="text-green-300 font-bold">Entailment ✓</p>
                  <p className="text-xs text-gray-400">89.2%</p>
                </div>
                <div className="bg-gray-600/30 px-4 py-2 rounded-lg border border-gray-500">
                  <p className="text-gray-400">Contradiction</p>
                  <p className="text-xs text-gray-500">3.1%</p>
                </div>
                <div className="bg-gray-600/30 px-4 py-2 rounded-lg border border-gray-500">
                  <p className="text-gray-400">Neutral</p>
                  <p className="text-xs text-gray-500">7.7%</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Semantic Similarity */}
        {selectedTask === 'similarity' && (
          <div className="space-y-4">
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-blue-900/20 rounded-xl p-4 border border-blue-500/30">
                <p className="text-sm text-blue-400 mb-1">Sentence A:</p>
                <p className="text-lg text-gray-200">{currentTask.example.sentA}</p>
              </div>
              <div className="bg-green-900/20 rounded-xl p-4 border border-green-500/30">
                <p className="text-sm text-green-400 mb-1">Sentence B:</p>
                <p className="text-lg text-gray-200">{currentTask.example.sentB}</p>
              </div>
            </div>

            <div className="flex justify-center">
              <ArrowRight className="text-gray-500 rotate-90" size={24} />
            </div>

            <div className="bg-gray-800/50 rounded-xl p-4">
              <p className="text-sm text-gray-400 mb-3 text-center">Similarity Score:</p>
              <div className="relative h-8 bg-gray-700 rounded-full overflow-hidden">
                <div 
                  className="absolute h-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 rounded-full transition-all"
                  style={{ width: '87%' }}
                />
                <div 
                  className="absolute top-0 h-full w-1 bg-white"
                  style={{ left: '87%' }}
                />
              </div>
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>0 (Different)</span>
                <span>0.5</span>
                <span>1 (Same)</span>
              </div>
              <p className="text-center mt-3 text-2xl font-bold text-green-400">0.87 - High Similarity</p>
            </div>
          </div>
        )}
      </div>

      {/* Task Format Summary */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-blue-900/20 rounded-xl p-4 border border-blue-500/30">
          <p className="text-sm text-blue-400 font-medium mb-2">📥 Input Format</p>
          <code className="text-xs bg-black/30 px-2 py-1 rounded text-gray-300 block">
            {currentTask.inputFormat}
          </code>
        </div>
        <div className="bg-green-900/20 rounded-xl p-4 border border-green-500/30">
          <p className="text-sm text-green-400 font-medium mb-2">📤 Output Type</p>
          <p className="text-xs text-gray-300">{currentTask.outputType}</p>
        </div>
        <div className="bg-purple-900/20 rounded-xl p-4 border border-purple-500/30">
          <p className="text-sm text-purple-400 font-medium mb-2">📊 Performance</p>
          <p className="text-xs text-gray-300">{currentTask.metrics}</p>
        </div>
      </div>

      {/* Applications */}
      <div className="bg-black/30 rounded-xl p-4 border border-white/10">
        <h4 className="font-bold text-yellow-400 mb-3">🎯 Real-World Applications</h4>
        <div className="flex flex-wrap gap-2">
          {currentTask.applications.map((app, i) => (
            <span key={i} className="bg-yellow-900/30 text-yellow-300 px-3 py-1 rounded-full text-sm">
              {app}
            </span>
          ))}
        </div>
      </div>

      {/* All Tasks Summary */}
      <div className="bg-black/30 rounded-xl p-4 border border-white/10 overflow-x-auto">
        <h4 className="font-bold text-white mb-4">📋 BERT NLP Tasks Summary</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-white/20">
              <th className="text-left py-2 text-gray-400">Task</th>
              <th className="text-center py-2 text-gray-400">Output</th>
              <th className="text-center py-2 text-gray-400">Head</th>
              <th className="text-center py-2 text-gray-400">From</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-white/10">
              <td className="py-2">Classification</td>
              <td className="text-center text-gray-400">Single label</td>
              <td className="text-center font-mono text-xs">Linear(H, C)</td>
              <td className="text-center text-blue-400">[CLS]</td>
            </tr>
            <tr className="border-b border-white/10">
              <td className="py-2">NER</td>
              <td className="text-center text-gray-400">Label per token</td>
              <td className="text-center font-mono text-xs">Linear(H, T)</td>
              <td className="text-center text-green-400">All tokens</td>
            </tr>
            <tr className="border-b border-white/10">
              <td className="py-2">QA</td>
              <td className="text-center text-gray-400">Start/End pos</td>
              <td className="text-center font-mono text-xs">Linear(H, 2)</td>
              <td className="text-center text-purple-400">Context tokens</td>
            </tr>
            <tr className="border-b border-white/10">
              <td className="py-2">NLI</td>
              <td className="text-center text-gray-400">3 classes</td>
              <td className="text-center font-mono text-xs">Linear(H, 3)</td>
              <td className="text-center text-blue-400">[CLS]</td>
            </tr>
            <tr>
              <td className="py-2">Similarity</td>
              <td className="text-center text-gray-400">Score 0-1</td>
              <td className="text-center font-mono text-xs">Linear(H, 1)</td>
              <td className="text-center text-blue-400">[CLS]</td>
            </tr>
          </tbody>
        </table>
        <p className="text-xs text-gray-500 mt-2">H = hidden size (768), C = num classes, T = num tags</p>
      </div>
    </div>
  );
}
