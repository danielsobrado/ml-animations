import React, { useState } from 'react';
import { BookOpen, ArrowRight, FileText, Hash, BarChart } from 'lucide-react';

export default function IntroPanel() {
  const [showPipeline, setShowPipeline] = useState(false);

  const exampleSentences = [
    "The cat sat on the mat",
    "The dog ran in the park",
    "I love machine learning"
  ];

  return (
    <div className="space-y-6 pb-20">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          Text to <span className="text-green-400">Numbers</span>: The Foundation of NLP
        </h2>
        <p className="text-gray-400">
          How do we convert human language into something machines can understand?
        </p>
      </div>

      {/* The Problem */}
      <div className="bg-red-900/20 rounded-2xl p-6 border border-red-500/30">
        <h3 className="text-xl font-bold text-red-400 mb-4">🤔 The Problem</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <p className="text-gray-300 mb-4">
              Machine learning algorithms work with <strong>numbers</strong>, not text. 
              But human language is made of <strong>words</strong>.
            </p>
            <div className="bg-black/30 rounded-lg p-4">
              <p className="text-sm text-gray-400 mb-2">Raw Text:</p>
              <p className="text-lg text-white font-mono">"I love machine learning"</p>
              <p className="text-4xl text-center my-4">↓ ?</p>
              <p className="text-sm text-gray-400 mb-2">Numbers for ML:</p>
              <p className="text-lg text-green-400 font-mono">[0.2, 0.8, 0.1, 0.5, ...]</p>
            </div>
          </div>
          <div>
            <p className="text-gray-300 mb-4">We need a way to:</p>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-green-400 mt-1">✓</span>
                <span>Represent text as fixed-size vectors</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400 mt-1">✓</span>
                <span>Capture word importance</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400 mt-1">✓</span>
                <span>Enable similarity comparisons</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400 mt-1">✓</span>
                <span>Feed into ML algorithms</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Solution Overview */}
      <div className="bg-green-900/20 rounded-2xl p-6 border border-green-500/30">
        <h3 className="text-xl font-bold text-green-400 mb-4">💡 Classic Solutions</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-black/30 rounded-xl p-4 border border-blue-500/30">
            <div className="flex items-center gap-2 mb-3">
              <Hash className="text-blue-400" size={24} />
              <h4 className="font-bold text-blue-400">Bag of Words (BoW)</h4>
            </div>
            <p className="text-sm text-gray-300 mb-3">
              Count how many times each word appears in a document.
            </p>
            <div className="bg-blue-900/20 rounded p-2 text-xs font-mono">
              "the cat sat" → {`{the: 1, cat: 1, sat: 1}`}
            </div>
            <p className="text-xs text-gray-500 mt-2">Simple, interpretable, but ignores word importance</p>
          </div>
          
          <div className="bg-black/30 rounded-xl p-4 border border-yellow-500/30">
            <div className="flex items-center gap-2 mb-3">
              <BarChart className="text-yellow-400" size={24} />
              <h4 className="font-bold text-yellow-400">TF-IDF</h4>
            </div>
            <p className="text-sm text-gray-300 mb-3">
              Weight words by importance: frequent in doc, rare across docs.
            </p>
            <div className="bg-yellow-900/20 rounded p-2 text-xs font-mono">
              "the" → low score (common everywhere)<br/>
              "cat" → high score (specific to doc)
            </div>
            <p className="text-xs text-gray-500 mt-2">Better for search, document similarity</p>
          </div>
        </div>
      </div>

      {/* Visual Pipeline */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold">📊 Text Processing Pipeline</h3>
          <button
            onClick={() => setShowPipeline(!showPipeline)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm"
          >
            {showPipeline ? 'Hide' : 'Show'} Animation
          </button>
        </div>

        <div className="flex items-center justify-between gap-4 overflow-x-auto pb-4">
          {/* Raw Text */}
          <div className={`flex-shrink-0 transition-all duration-500 ${showPipeline ? 'opacity-100' : 'opacity-50'}`}>
            <div className="bg-gray-700 rounded-xl p-4 text-center w-40">
              <FileText className="mx-auto mb-2 text-gray-400" size={32} />
              <p className="text-sm font-medium">Raw Text</p>
              <p className="text-xs text-gray-400 mt-1">"The cat sat..."</p>
            </div>
          </div>

          <ArrowRight className={`text-gray-500 flex-shrink-0 transition-all duration-500 delay-100 ${showPipeline ? 'opacity-100' : 'opacity-30'}`} />

          {/* Preprocessing */}
          <div className={`flex-shrink-0 transition-all duration-500 delay-200 ${showPipeline ? 'opacity-100' : 'opacity-50'}`}>
            <div className="bg-purple-900/50 rounded-xl p-4 text-center w-40 border border-purple-500/30">
              <p className="text-2xl mb-2">🔧</p>
              <p className="text-sm font-medium text-purple-300">Preprocessing</p>
              <p className="text-xs text-gray-400 mt-1">lowercase, tokenize</p>
            </div>
          </div>

          <ArrowRight className={`text-gray-500 flex-shrink-0 transition-all duration-500 delay-300 ${showPipeline ? 'opacity-100' : 'opacity-30'}`} />

          {/* Vocabulary */}
          <div className={`flex-shrink-0 transition-all duration-500 delay-400 ${showPipeline ? 'opacity-100' : 'opacity-50'}`}>
            <div className="bg-blue-900/50 rounded-xl p-4 text-center w-40 border border-blue-500/30">
              <p className="text-2xl mb-2">📚</p>
              <p className="text-sm font-medium text-blue-300">Build Vocab</p>
              <p className="text-xs text-gray-400 mt-1">unique words</p>
            </div>
          </div>

          <ArrowRight className={`text-gray-500 flex-shrink-0 transition-all duration-500 delay-500 ${showPipeline ? 'opacity-100' : 'opacity-30'}`} />

          {/* Vectorize */}
          <div className={`flex-shrink-0 transition-all duration-500 delay-600 ${showPipeline ? 'opacity-100' : 'opacity-50'}`}>
            <div className="bg-green-900/50 rounded-xl p-4 text-center w-40 border border-green-500/30">
              <p className="text-2xl mb-2">🔢</p>
              <p className="text-sm font-medium text-green-300">Vectorize</p>
              <p className="text-xs text-gray-400 mt-1">BoW or TF-IDF</p>
            </div>
          </div>

          <ArrowRight className={`text-gray-500 flex-shrink-0 transition-all duration-500 delay-700 ${showPipeline ? 'opacity-100' : 'opacity-30'}`} />

          {/* Vector Output */}
          <div className={`flex-shrink-0 transition-all duration-500 delay-800 ${showPipeline ? 'opacity-100' : 'opacity-50'}`}>
            <div className="bg-yellow-900/50 rounded-xl p-4 text-center w-40 border border-yellow-500/30">
              <p className="text-2xl mb-2">📊</p>
              <p className="text-sm font-medium text-yellow-300">Feature Vector</p>
              <p className="text-xs text-gray-400 mt-1">[0.2, 0.8, ...]</p>
            </div>
          </div>
        </div>
      </div>

      {/* Key Concepts */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-black/30 rounded-xl p-4 border border-white/10">
          <h4 className="font-bold text-blue-400 mb-2">📖 Vocabulary</h4>
          <p className="text-sm text-gray-300">
            The set of all unique words across all documents. Each word gets a unique index in the vector.
          </p>
        </div>
        <div className="bg-black/30 rounded-xl p-4 border border-white/10">
          <h4 className="font-bold text-green-400 mb-2">📄 Document</h4>
          <p className="text-sm text-gray-300">
            A piece of text (sentence, paragraph, or entire article) that we want to represent as a vector.
          </p>
        </div>
        <div className="bg-black/30 rounded-xl p-4 border border-white/10">
          <h4 className="font-bold text-purple-400 mb-2">📊 Corpus</h4>
          <p className="text-sm text-gray-300">
            The entire collection of documents. Used to build vocabulary and calculate word frequencies.
          </p>
        </div>
      </div>

      {/* History */}
      <div className="bg-gradient-to-r from-blue-900/20 to-purple-900/20 rounded-xl p-4 border border-white/10">
        <h4 className="font-bold text-white mb-3">📜 Historical Context</h4>
        <div className="flex items-center gap-4 overflow-x-auto pb-2">
          <div className="flex-shrink-0 text-center">
            <p className="text-2xl font-bold text-blue-400">1950s</p>
            <p className="text-xs text-gray-400">BoW introduced</p>
          </div>
          <div className="w-16 h-0.5 bg-gray-600 flex-shrink-0" />
          <div className="flex-shrink-0 text-center">
            <p className="text-2xl font-bold text-green-400">1972</p>
            <p className="text-xs text-gray-400">TF-IDF created</p>
          </div>
          <div className="w-16 h-0.5 bg-gray-600 flex-shrink-0" />
          <div className="flex-shrink-0 text-center">
            <p className="text-2xl font-bold text-yellow-400">2000s</p>
            <p className="text-xs text-gray-400">Search engines</p>
          </div>
          <div className="w-16 h-0.5 bg-gray-600 flex-shrink-0" />
          <div className="flex-shrink-0 text-center">
            <p className="text-2xl font-bold text-purple-400">Today</p>
            <p className="text-xs text-gray-400">Still widely used!</p>
          </div>
        </div>
      </div>

      {/* What's Next */}
      <div className="bg-green-900/20 rounded-xl p-4 border border-green-500/30">
        <h4 className="font-bold text-green-400 mb-2">🎯 What You'll Learn</h4>
        <ul className="grid md:grid-cols-2 gap-2 text-sm text-gray-300">
          <li>✓ How Bag of Words converts text to vectors</li>
          <li>✓ Understanding Term Frequency (TF)</li>
          <li>✓ Understanding Inverse Document Frequency (IDF)</li>
          <li>✓ When to use BoW vs TF-IDF</li>
          <li>✓ Limitations and modern alternatives</li>
          <li>✓ Hands-on practice with examples</li>
        </ul>
      </div>
    </div>
  );
}
