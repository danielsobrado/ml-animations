import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Sparkles } from 'lucide-react';
import { categories } from '../data/animations';

export default function HomePage() {
  return (
    <div className="p-6 lg:p-8">
      {/* Hero Section */}
      <div className="mb-12">
        <div className="max-w-4xl">
          <h1 className="text-4xl lg:text-5xl font-bold mb-4">
            <span className="text-gradient">Interactive Machine Learning</span>
            <br />
            <span className="text-slate-900 dark:text-white">Visualizations</span>
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-400 mb-6 max-w-2xl">
            Explore machine learning concepts through beautiful, interactive animations. 
            From basic math fundamentals to advanced transformer architectures.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link 
              to="/animation/attention-mechanism" 
              className="btn-primary flex items-center gap-2"
            >
              <Sparkles size={18} />
              Start with Attention
              <ArrowRight size={18} />
            </Link>
            <a 
              href="#categories" 
              className="btn-secondary"
            >
              Browse All Topics
            </a>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
        {[
          { label: 'Categories', value: categories.length },
          { label: 'Animations', value: categories.reduce((acc, c) => acc + c.items.length, 0) },
          { label: 'Interactive', value: '100%' },
          { label: 'Open Source', value: '✓' },
        ].map((stat, i) => (
          <div key={i} className="card p-4 text-center">
            <div className="text-2xl font-bold text-gradient">{stat.value}</div>
            <div className="text-sm text-slate-600 dark:text-slate-400">{stat.label}</div>
          </div>
        ))}
      </div>

      {/* Categories */}
      <div id="categories" className="space-y-10">
        {categories.map((category) => (
          <section key={category.id}>
            {/* Category Header */}
            <div className="flex items-center gap-3 mb-6">
              <div className={`p-2.5 rounded-xl bg-gradient-to-r ${category.color}`}>
                <category.icon size={24} className="text-white" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-slate-900 dark:text-white">
                  {category.name}
                </h2>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  {category.items.length} animations
                </p>
              </div>
            </div>

            {/* Animation Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {category.items.map((item) => (
                <Link
                  key={item.id}
                  to={`/animation/${item.id}`}
                  className="animation-card group"
                >
                  <div className="flex items-start gap-4">
                    <div className={`p-2 rounded-lg bg-gradient-to-r ${category.color} opacity-80 group-hover:opacity-100 transition-opacity`}>
                      <item.icon size={20} className="text-white" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-slate-900 dark:text-white mb-1 truncate">
                        {item.name}
                      </h3>
                      <p className="text-sm text-slate-600 dark:text-slate-400 line-clamp-2">
                        {item.description}
                      </p>
                    </div>
                    <ArrowRight 
                      size={18} 
                      className="text-slate-400 group-hover:text-slate-600 dark:group-hover:text-slate-300 group-hover:translate-x-1 transition-all flex-shrink-0" 
                    />
                  </div>
                </Link>
              ))}
            </div>
          </section>
        ))}
      </div>

      {/* Footer */}
      <footer className="mt-16 pt-8 border-t border-slate-200 dark:border-slate-800">
        <div className="text-center text-sm text-slate-500 dark:text-slate-400">
          <p className="mb-2">
            Built with React, Tailwind CSS, and ❤️
          </p>
          <p>
            <a 
              href="https://github.com/danielsobrado/ml-animations" 
              className="text-blue-500 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              View on GitHub
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
}
