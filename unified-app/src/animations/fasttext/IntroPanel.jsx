import React from 'react';
import { motion } from 'framer-motion';
import { Zap, Puzzle, Target, Languages, Sparkles } from 'lucide-react';

function IntroPanel() {
  const features = [
    {
      icon: Puzzle,
      title: 'Subword Information',
      description: 'Breaks words into character n-grams to capture morphology',
      color: 'from-purple-500 to-pink-500'
    },
    {
      icon: Target,
      title: 'OOV Handling',
      description: 'Can generate embeddings for unseen words',
      color: 'from-blue-500 to-cyan-500'
    },
    {
      icon: Languages,
      title: 'Morphologically Rich',
      description: 'Excellent for languages with complex word forms',
      color: 'from-green-500 to-emerald-500'
    },
    {
      icon: Zap,
      title: 'Fast Training',
      description: 'Efficient training with hierarchical softmax',
      color: 'from-orange-500 to-yellow-500'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <div className="flex items-center justify-center gap-3 mb-4">
          <motion.div
            animate={{ rotate: [0, 360] }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          >
            <Sparkles className="w-8 h-8 text-purple-600 dark:text-purple-400" />
          </motion.div>
          <h2 className="text-3xl font-bold text-white">What is FastText?</h2>
        </div>
        <p className="text-purple-200/80 text-lg max-w-3xl mx-auto">
          FastText extends Word2Vec by representing each word as a bag of character n-grams.
          This allows it to capture morphological information and handle out-of-vocabulary words.
        </p>
      </motion.div>

      {/* Visual Word Breakdown */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.2 }}
        className="bg-white/5 rounded-xl p-6 mb-6"
      >
        <h3 className="text-lg font-semibold text-white mb-4 text-center">Word Representation Example</h3>
        <div className="flex flex-col items-center gap-4">
          <div className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
            "learning"
          </div>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="flex items-center gap-2"
          >
            <span className="text-purple-300">→</span>
          </motion.div>
          <div className="flex flex-wrap justify-center gap-2">
            {['<le', 'lea', 'ear', 'arn', 'rni', 'nin', 'ing', 'ng>'].map((ngram, i) => (
              <motion.span
                key={ngram}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 + i * 0.1 }}
                className="ngram-box text-purple-100 font-mono text-sm"
              >
                {ngram}
              </motion.span>
            ))}
          </div>
          <p className="text-purple-300/70 text-sm text-center mt-2">
            Character 3-grams with boundary markers ({'<'} and {'>'})
          </p>
        </div>
      </motion.div>

      {/* Key Features */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {features.map((feature, index) => (
          <motion.div
            key={feature.title}
            initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 + index * 0.1 }}
            className="bg-white/5 rounded-xl p-4 hover:bg-white/10 transition-colors"
          >
            <div className="flex items-start gap-3">
              <div className={`p-2 rounded-lg bg-gradient-to-br ${feature.color}`}>
                <feature.icon className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-white mb-1">{feature.title}</h3>
                <p className="text-purple-200/70 text-sm">{feature.description}</p>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Timeline */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl p-4 mt-6"
      >
        <div className="flex items-center gap-4 justify-center flex-wrap">
          <div className="text-center">
            <div className="text-purple-600 dark:text-purple-400 font-semibold">2013</div>
            <div className="text-sm text-purple-200/70">Word2Vec</div>
          </div>
          <div className="text-purple-600 dark:text-purple-400">→</div>
          <div className="text-center">
            <div className="text-purple-600 dark:text-purple-400 font-semibold">2014</div>
            <div className="text-sm text-purple-200/70">GloVe</div>
          </div>
          <div className="text-purple-600 dark:text-purple-400">→</div>
          <div className="text-center px-4 py-2 bg-purple-500/30 rounded-lg border border-purple-400/50">
            <div className="text-pink-600 dark:text-pink-400 font-semibold">2016</div>
            <div className="text-sm text-white">FastText</div>
          </div>
          <div className="text-purple-600 dark:text-purple-400">→</div>
          <div className="text-center">
            <div className="text-purple-600 dark:text-purple-400 font-semibold">2018</div>
            <div className="text-sm text-purple-200/70">BERT</div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

export default IntroPanel;
