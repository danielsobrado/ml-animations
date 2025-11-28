import React, { Suspense } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, ArrowRight, ExternalLink } from 'lucide-react';
import { getAnimationById, allAnimations, getCategoryById } from '../data/animations';
import { getAnimationComponent, isAnimationAvailable } from '../animations';

export default function AnimationPage() {
  const { id } = useParams();
  const animation = getAnimationById(id);
  
  if (!animation) {
    return (
      <div className="p-6 text-center">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white mb-4">
          Animation Not Found
        </h1>
        <p className="text-slate-600 dark:text-slate-400 mb-6">
          The animation "{id}" could not be found.
        </p>
        <Link to="/" className="btn-primary">
          Back to Home
        </Link>
      </div>
    );
  }

  // Find previous and next animations
  const currentIndex = allAnimations.findIndex(a => a.id === id);
  const prevAnimation = currentIndex > 0 ? allAnimations[currentIndex - 1] : null;
  const nextAnimation = currentIndex < allAnimations.length - 1 ? allAnimations[currentIndex + 1] : null;

  const category = getCategoryById(animation.categoryId);

  return (
    <div className="flex flex-col min-h-full">
      {/* Breadcrumb */}
      <div className="px-6 py-4 border-b border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm">
        <div className="flex items-center gap-2 text-sm">
          <Link to="/" className="text-slate-500 hover:text-slate-700 dark:hover:text-slate-300">
            Home
          </Link>
          <span className="text-slate-400">/</span>
          <span className="text-slate-500">{animation.categoryName}</span>
          <span className="text-slate-400">/</span>
          <span className="font-medium text-slate-900 dark:text-white">{animation.name}</span>
        </div>
      </div>

      {/* Header */}
      <div className="px-6 py-6 border-b border-slate-200 dark:border-slate-800">
        <div className="flex items-start gap-4">
          <div className={`p-3 rounded-xl bg-gradient-to-r ${animation.categoryColor}`}>
            <animation.icon size={28} className="text-white" />
          </div>
          <div className="flex-1">
            <h1 className="text-2xl lg:text-3xl font-bold text-slate-900 dark:text-white mb-2">
              {animation.name}
            </h1>
            <p className="text-slate-600 dark:text-slate-400">
              {animation.description}
            </p>
          </div>
        </div>
      </div>

      {/* Animation Content */}
      <div className="flex-1 p-6">
        <AnimationContent animationId={id} animation={animation} />
      </div>

      {/* Navigation Footer */}
      <div className="px-6 py-4 border-t border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm">
        <div className="flex items-center justify-between">
          {prevAnimation ? (
            <Link 
              to={`/animation/${prevAnimation.id}`}
              className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white transition-colors"
            >
              <ArrowLeft size={16} />
              <span className="hidden sm:inline">{prevAnimation.name}</span>
              <span className="sm:hidden">Previous</span>
            </Link>
          ) : <div />}
          
          <Link to="/" className="text-sm text-blue-500 hover:text-blue-600">
            All Animations
          </Link>
          
          {nextAnimation ? (
            <Link 
              to={`/animation/${nextAnimation.id}`}
              className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white transition-colors"
            >
              <span className="hidden sm:inline">{nextAnimation.name}</span>
              <span className="sm:hidden">Next</span>
              <ArrowRight size={16} />
            </Link>
          ) : <div />}
        </div>
      </div>
    </div>
  );
}

// Loading component for lazy-loaded animations
function LoadingAnimation() {
  return (
    <div className="flex items-center justify-center p-12">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
        <p className="text-slate-600 dark:text-slate-400">Loading animation...</p>
      </div>
    </div>
  );
}

// Animation content component with dynamic loading
function AnimationContent({ animationId, animation }) {
  const AnimationComponent = getAnimationComponent(animationId);
  
  // If animation is available, render it
  if (AnimationComponent) {
    return (
      <Suspense fallback={<LoadingAnimation />}>
        <AnimationComponent />
      </Suspense>
    );
  }
  
  // Otherwise show placeholder with integration instructions
  return (
    <div className="card p-8 text-center">
      <div className={`inline-flex p-4 rounded-2xl bg-gradient-to-r ${animation.categoryColor} mb-6`}>
        <animation.icon size={48} className="text-white" />
      </div>
      
      <h2 className="text-xl font-bold text-slate-900 dark:text-white mb-4">
        {animation.name} Animation
      </h2>
      
      <p className="text-slate-600 dark:text-slate-400 mb-6 max-w-lg mx-auto">
        This animation is ready to be integrated. Copy the components from the 
        <code className="mx-1 px-2 py-0.5 bg-slate-100 dark:bg-slate-800 rounded text-sm">
          {animationId}-animation/src/
        </code> 
        folder into this unified application.
      </p>

      <div className="flex flex-wrap justify-center gap-3">
        <a 
          href={`https://github.com/danielsobrado/ml-animations/tree/main/${animationId}-animation`}
          target="_blank"
          rel="noopener noreferrer"
          className="btn-secondary flex items-center gap-2"
        >
          <ExternalLink size={16} />
          View Source
        </a>
      </div>

      {/* Integration Instructions */}
      <div className="mt-8 p-6 bg-slate-50 dark:bg-slate-800/50 rounded-xl text-left">
        <h3 className="font-semibold text-slate-900 dark:text-white mb-3">
          Integration Steps:
        </h3>
        <ol className="text-sm text-slate-600 dark:text-slate-400 space-y-2 list-decimal list-inside">
          <li>Copy panel components from <code className="px-1 bg-slate-200 dark:bg-slate-700 rounded">{animationId}-animation/src/</code></li>
          <li>Place them in <code className="px-1 bg-slate-200 dark:bg-slate-700 rounded">unified-app/src/animations/{animationId}/</code></li>
          <li>Register the animation in <code className="px-1 bg-slate-200 dark:bg-slate-700 rounded">src/animations/index.js</code></li>
          <li>The animation will automatically inherit the unified theme</li>
        </ol>
      </div>
    </div>
  );
}
