import React, { useState } from 'react';
import { ArrowRight, ArrowLeftRight, Check, Info, BookOpen, Calendar } from 'lucide-react';

export default function OverviewPanel() {
  const [hoveredComponent, setHoveredComponent] = useState(null);

  const timeline = [
    { year: '2017', event: 'Transformer ("Attention is All You Need")', color: 'blue' },
    { year: '2018', event: 'BERT released by Google', color: 'purple' },
    { year: '2019', event: 'RoBERTa, ALBERT, DistilBERT variants', color: 'green' },
    { year: '2020+', event: 'Foundation for GPT, T5, and modern LLMs', color: 'orange' },
  ];

  const keyFeatures = [
    {
      id: 'bidirectional',
      title: 'Bidirectional Context',
      description: 'BERT reads text both left-to-right AND right-to-left simultaneously, unlike GPT which only goes left-to-right.',
      icon: ArrowLeftRight,
      color: 'from-blue-500 to-cyan-500'
    },
    {
      id: 'encoder',
      title: 'Encoder-Only Architecture',
      description: 'Uses only the Transformer encoder stack (no decoder). Perfect for understanding tasks, not generation.',
      icon: BookOpen,
      color: 'from-purple-500 to-pink-500'
    },
    {
      id: 'pretraining',
      title: 'Pre-training + Fine-tuning',
      description: 'Pre-trained on massive unlabeled data (MLM + NSP), then fine-tuned on specific tasks with small labeled datasets.',
      icon: Check,
      color: 'from-green-500 to-emerald-500'
    }
  ];

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="text-center">
        <h2 className="text-4xl font-bold mb-4">
          <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            BERT: The Complete Guide
          </span>
        </h2>
        <p className="text-xl text-gray-300 max-w-3xl mx-auto">
          <strong>B</strong>idirectional <strong>E</strong>ncoder <strong>R</strong>epresentations from <strong>T</strong>ransformers
        </p>
        <p className="text-gray-400 mt-2">
          The model that revolutionized NLP and became the foundation for modern language understanding
        </p>
      </div>

      {/* What is BERT? */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <span className="text-3xl">🤔</span> What is BERT?
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <p className="text-gray-300 leading-relaxed">
              BERT is a <strong className="text-blue-400">pre-trained language model</strong> developed by Google in 2018. 
              It learns deep bidirectional representations by jointly conditioning on both left and right context in all layers.
            </p>
            <p className="text-gray-300 leading-relaxed">
              Unlike previous models that read text sequentially (left-to-right or right-to-left), 
              BERT considers the <strong className="text-purple-400">full context</strong> of a word by looking at 
              all surrounding words simultaneously.
            </p>
          </div>
          <div className="bg-gradient-to-br from-blue-900/30 to-purple-900/30 rounded-xl p-4 border border-blue-500/30">
            <h4 className="font-bold text-blue-300 mb-3">Example: "Bank" disambiguation</h4>
            <div className="space-y-2 text-sm">
              <div className="p-2 bg-black/30 rounded">
                <p className="text-gray-400">Sentence 1:</p>
                <p>"I went to the <span className="text-yellow-400 font-bold">bank</span> to deposit money"</p>
                <p className="text-green-400 text-xs mt-1">→ BERT understands: financial institution 🏦</p>
              </div>
              <div className="p-2 bg-black/30 rounded">
                <p className="text-gray-400">Sentence 2:</p>
                <p>"I sat by the river <span className="text-yellow-400 font-bold">bank</span> and relaxed"</p>
                <p className="text-green-400 text-xs mt-1">→ BERT understands: riverside 🏞️</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Key Features */}
      <div className="grid md:grid-cols-3 gap-4">
        {keyFeatures.map((feature) => {
          const Icon = feature.icon;
          return (
            <div
              key={feature.id}
              className="bg-black/30 rounded-xl p-5 border border-white/10 hover:border-white/30 transition-all cursor-pointer"
              onMouseEnter={() => setHoveredComponent(feature.id)}
              onMouseLeave={() => setHoveredComponent(null)}
            >
              <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${feature.color} flex items-center justify-center mb-3`}>
                <Icon size={24} />
              </div>
              <h4 className="font-bold text-lg mb-2">{feature.title}</h4>
              <p className="text-sm text-gray-400">{feature.description}</p>
            </div>
          );
        })}
      </div>

      {/* Architecture Overview */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-6 text-center">BERT Architecture at a Glance</h3>
        
        <div className="flex flex-col md:flex-row items-center justify-center gap-4">
          {/* Input */}
          <div className="bg-blue-900/30 border border-blue-500/30 rounded-xl p-4 text-center">
            <p className="text-blue-400 font-bold mb-2">Input</p>
            <div className="flex gap-1 justify-center">
              {['[CLS]', 'The', 'cat', 'sat', '[SEP]'].map((token, i) => (
                <span key={i} className="bg-blue-600/30 px-2 py-1 rounded text-xs">{token}</span>
              ))}
            </div>
            <p className="text-xs text-gray-500 mt-2">Token + Segment + Position</p>
          </div>

          <ArrowRight className="text-gray-500 rotate-90 md:rotate-0" />

          {/* Encoder Stack */}
          <div className="bg-purple-900/30 border border-purple-500/30 rounded-xl p-4 text-center">
            <p className="text-purple-400 font-bold mb-2">Transformer Encoders</p>
            <div className="flex flex-col gap-1">
              {[12, 11, 10, '...', 2, 1].map((layer, i) => (
                <div key={i} className="bg-purple-600/30 px-4 py-1 rounded text-xs">
                  {layer === '...' ? '...' : `Layer ${layer}`}
                </div>
              ))}
            </div>
            <p className="text-xs text-gray-500 mt-2">Self-Attention + FFN</p>
          </div>

          <ArrowRight className="text-gray-500 rotate-90 md:rotate-0" />

          {/* Output */}
          <div className="bg-green-900/30 border border-green-500/30 rounded-xl p-4 text-center">
            <p className="text-green-400 font-bold mb-2">Output</p>
            <div className="flex gap-1 justify-center">
              {['T[CLS]', 'T₁', 'T₂', 'T₃', 'T[SEP]'].map((token, i) => (
                <span key={i} className="bg-green-600/30 px-2 py-1 rounded text-xs">{token}</span>
              ))}
            </div>
            <p className="text-xs text-gray-500 mt-2">Contextual embeddings</p>
          </div>
        </div>
      </div>

      {/* Timeline */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
          <Calendar size={24} className="text-purple-400" />
          BERT in History
        </h3>
        <div className="flex flex-wrap gap-4 justify-center">
          {timeline.map((item, i) => (
            <div key={i} className="flex items-center gap-2">
              <div className={`w-16 h-16 rounded-full bg-${item.color}-600/30 border-2 border-${item.color}-500 flex items-center justify-center`}>
                <span className="font-bold text-sm">{item.year}</span>
              </div>
              <div className="max-w-[150px]">
                <p className="text-sm text-gray-300">{item.event}</p>
              </div>
              {i < timeline.length - 1 && <ArrowRight className="text-gray-600 hidden md:block" />}
            </div>
          ))}
        </div>
      </div>

      {/* BERT Variants */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">📊 BERT Model Sizes</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left py-2 px-4">Model</th>
                <th className="text-center py-2 px-4">Layers</th>
                <th className="text-center py-2 px-4">Hidden Size</th>
                <th className="text-center py-2 px-4">Attention Heads</th>
                <th className="text-center py-2 px-4">Parameters</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-white/5">
                <td className="py-2 px-4 text-blue-400 font-medium">BERT-Base</td>
                <td className="text-center py-2 px-4">12</td>
                <td className="text-center py-2 px-4">768</td>
                <td className="text-center py-2 px-4">12</td>
                <td className="text-center py-2 px-4 text-yellow-400">110M</td>
              </tr>
              <tr className="border-b border-white/5">
                <td className="py-2 px-4 text-purple-400 font-medium">BERT-Large</td>
                <td className="text-center py-2 px-4">24</td>
                <td className="text-center py-2 px-4">1024</td>
                <td className="text-center py-2 px-4">16</td>
                <td className="text-center py-2 px-4 text-yellow-400">340M</td>
              </tr>
              <tr>
                <td className="py-2 px-4 text-green-400 font-medium">DistilBERT</td>
                <td className="text-center py-2 px-4">6</td>
                <td className="text-center py-2 px-4">768</td>
                <td className="text-center py-2 px-4">12</td>
                <td className="text-center py-2 px-4 text-yellow-400">66M</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* What You'll Learn */}
      <div className="bg-gradient-to-br from-purple-900/30 to-blue-900/30 rounded-2xl p-6 border border-purple-500/30">
        <h3 className="text-xl font-bold mb-4">🎯 What You'll Learn in This Guide</h3>
        <div className="grid md:grid-cols-2 gap-4">
          {[
            { num: 1, title: 'Overview', desc: 'BERT architecture and key concepts' },
            { num: 2, title: 'Tokenization', desc: 'WordPiece tokenizer and special tokens' },
            { num: 3, title: 'Embeddings', desc: 'Token, Segment, and Position embeddings' },
            { num: 4, title: 'Self-Attention', desc: 'Multi-head attention mechanism' },
            { num: 5, title: 'Encoder Layers', desc: 'Full encoder block walkthrough' },
            { num: 6, title: 'Pre-training', desc: 'MLM and NSP objectives' },
            { num: 7, title: 'Fine-tuning', desc: 'Adapting BERT for downstream tasks' },
            { num: 8, title: 'Live Examples', desc: 'Interactive demos with real text' },
            { num: 9, title: 'Practice Lab', desc: 'Quiz and exercises' },
          ].map(item => (
            <div key={item.num} className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center text-sm font-bold">
                {item.num}
              </div>
              <div>
                <p className="font-medium">{item.title}</p>
                <p className="text-xs text-gray-400">{item.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
