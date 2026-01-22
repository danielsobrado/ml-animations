import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Code, Copy, Check, Terminal } from 'lucide-react';

function CodePanel() {
  const [copiedIndex, setCopiedIndex] = useState(null);
  const [activeTab, setActiveTab] = useState('python');

  const copyToClipboard = (text, index) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const pythonExamples = [
    {
      title: 'Install FastText',
      code: `# Install via pip
pip install fasttext

# Or install from source for latest features
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install .`,
    },
    {
      title: 'Train Word Embeddings',
      code: `import fasttext

# Train unsupervised word embeddings
# model options: 'skipgram' or 'cbow'
model = fasttext.train_unsupervised(
    'data.txt',
    model='skipgram',
    dim=100,           # embedding dimension
    ws=5,              # context window size
    minCount=5,        # min word frequency
    minn=3,            # min n-gram length
    maxn=6,            # max n-gram length
    epoch=5
)

# Save model
model.save_model('fasttext_model.bin')`,
    },
    {
      title: 'Load and Use Embeddings',
      code: `import fasttext

# Load pre-trained model
model = fasttext.load_model('fasttext_model.bin')

# Get word vector
vec = model.get_word_vector('king')
print(f"Vector shape: {vec.shape}")

# Works for OOV words too!
oov_vec = model.get_word_vector('kingship')
print("OOV word handled successfully!")

# Find similar words
similar = model.get_nearest_neighbors('king', k=5)
for score, word in similar:
    print(f"{word}: {score:.4f}")`,
    },
    {
      title: 'Word Analogies',
      code: `# Solve word analogies: king - man + woman = ?
# FastText uses subword information for better results

def analogy(model, a, b, c, k=5):
    """Find word d such that a:b :: c:d"""
    results = model.get_analogies(a, b, c, k)
    return results

# Example: king - man + woman = queen
results = analogy(model, 'king', 'man', 'woman')
print("king - man + woman =")
for score, word in results:
    print(f"  {word}: {score:.4f}")`,
    },
    {
      title: 'Text Classification',
      code: `import fasttext

# Prepare labeled data (format: __label__<class> <text>)
# data.txt:
# __label__positive This movie was great!
# __label__negative Terrible waste of time

# Train classifier
classifier = fasttext.train_supervised(
    'train.txt',
    dim=100,
    epoch=25,
    lr=0.5,
    wordNgrams=2,
    loss='softmax'
)

# Predict
label, confidence = classifier.predict("I loved this!")
print(f"Predicted: {label[0]} ({confidence[0]:.4f})")

# Evaluate
result = classifier.test('test.txt')
print(f"Precision: {result[1]:.4f}")
print(f"Recall: {result[2]:.4f}")`,
    },
  ];

  const gensimExamples = [
    {
      title: 'Using Gensim FastText',
      code: `from gensim.models import FastText

# Train model
sentences = [
    ['king', 'queen', 'royal', 'palace'],
    ['man', 'woman', 'child', 'family'],
    # more sentences...
]

model = FastText(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    min_n=3,        # min n-gram
    max_n=6,        # max n-gram
    workers=4
)

# Get word vector (works for OOV too!)
vec = model.wv['kingship']  # OOV word

# Most similar words
similar = model.wv.most_similar('king', topn=5)`,
    },
    {
      title: 'Load Pre-trained Facebook Model',
      code: `from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model

# Download from https://fasttext.cc/docs/en/crawl-vectors.html
# Load pre-trained model (e.g., English)
model = load_facebook_model('cc.en.300.bin')

# Use the model
vec = model.wv['python']
similar = model.wv.most_similar('programming')`,
    },
  ];

  const examples = activeTab === 'python' ? pythonExamples : gensimExamples;

  return (
    <div className="space-y-6">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold text-white mb-2">Code Examples</h2>
        <p className="text-purple-200/70">
          Practical FastText implementations in Python
        </p>
      </div>

      {/* Tabs */}
      <div className="flex justify-center gap-2 mb-6">
        {[
          { id: 'python', label: 'FastText (Official)' },
          { id: 'gensim', label: 'Gensim' },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 rounded-lg transition-all ${
              activeTab === tab.id
                ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
                : 'bg-white/10 text-purple-300 hover:bg-white/20'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Code Examples */}
      <div className="space-y-4">
        {examples.map((example, index) => (
          <motion.div
            key={example.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="bg-white/5 rounded-xl overflow-hidden"
          >
            <div className="flex items-center justify-between px-4 py-2 bg-white/5 border-b border-white/10">
              <div className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                <span className="text-purple-200 font-medium">{example.title}</span>
              </div>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => copyToClipboard(example.code, index)}
                className="flex items-center gap-1 px-2 py-1 rounded bg-white/10 hover:bg-white/20 text-sm"
              >
                {copiedIndex === index ? (
                  <>
                    <Check className="w-4 h-4 text-green-400" />
                    <span className="text-green-400">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-4 h-4 text-purple-300" />
                    <span className="text-purple-300">Copy</span>
                  </>
                )}
              </motion.button>
            </div>
            <pre className="p-4 overflow-x-auto">
              <code className="text-sm text-purple-100 font-mono whitespace-pre">
                {example.code}
              </code>
            </pre>
          </motion.div>
        ))}
      </div>

      {/* Pre-trained Models */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl p-6"
      >
        <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
          <Code className="w-5 h-5 text-purple-600 dark:text-purple-400" />
          Pre-trained Models
        </h3>
        <p className="text-purple-200/80 mb-4">
          Facebook provides pre-trained FastText models for 157 languages:
        </p>
        <a
          href="https://fasttext.cc/docs/en/crawl-vectors.html"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 px-4 py-2 bg-purple-500/30 rounded-lg text-purple-200 hover:bg-purple-500/40 transition-colors"
        >
          fasttext.cc/docs/en/crawl-vectors.html
          <span>↗</span>
        </a>
      </motion.div>
    </div>
  );
}

export default CodePanel;
