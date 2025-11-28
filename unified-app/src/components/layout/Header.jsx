import React from 'react';
import { Link } from 'react-router-dom';
import { 
  Menu, 
  Sun, 
  Moon, 
  Github, 
  PanelLeftClose, 
  PanelLeft,
  Sparkles,
} from 'lucide-react';
import { useTheme } from '../../context/ThemeContext';

export default function Header({ onMenuClick, onCollapseClick, sidebarCollapsed }) {
  const { isDark, toggleTheme } = useTheme();

  return (
    <header className="sticky top-0 z-50 h-16 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md border-b border-slate-200 dark:border-slate-800">
      <div className="h-full px-4 flex items-center justify-between">
        {/* Left section */}
        <div className="flex items-center gap-3">
          {/* Mobile menu button */}
          <button 
            onClick={onMenuClick}
            className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 md:hidden"
            aria-label="Toggle menu"
          >
            <Menu size={20} />
          </button>

          {/* Collapse button (desktop) */}
          <button 
            onClick={onCollapseClick}
            className="hidden md:flex p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800"
            aria-label="Toggle sidebar"
          >
            {sidebarCollapsed ? <PanelLeft size={20} /> : <PanelLeftClose size={20} />}
          </button>

          {/* Logo */}
          <Link to="/" className="flex items-center gap-3">
            <div className="bg-gradient-to-r from-blue-500 to-purple-500 p-2 rounded-xl">
              <Sparkles className="text-white" size={24} />
            </div>
            <div className="hidden sm:block">
              <h1 className="text-lg font-bold text-slate-900 dark:text-white">
                ML Animations
              </h1>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Interactive Machine Learning
              </p>
            </div>
          </Link>
        </div>

        {/* Right section */}
        <div className="flex items-center gap-2">
          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            className="p-2.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            aria-label="Toggle theme"
          >
            {isDark ? (
              <Sun size={20} className="text-amber-500" />
            ) : (
              <Moon size={20} className="text-slate-600" />
            )}
          </button>

          {/* GitHub link */}
          <a
            href="https://github.com/danielsobrado/ml-animations"
            target="_blank"
            rel="noopener noreferrer"
            className="p-2.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            aria-label="GitHub repository"
          >
            <Github size={20} />
          </a>
        </div>
      </div>
    </header>
  );
}
