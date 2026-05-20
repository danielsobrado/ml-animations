import React from 'react';
import { Link } from 'react-router-dom';
import { BookOpen, Github, Menu, PanelLeft, PanelLeftClose, PencilLine } from 'lucide-react';

export default function Header({
  onMenuClick,
  onSidebarControlClick,
  onOpenCommandPalette,
  onOpenNotes,
  progress,
  sidebarOpen,
  sidebarCollapsed,
}) {
  const progressLabel = `Σ ${progress.visited} / ${progress.total} lessons`;
  const progressPercent = progress.total > 0 ? (progress.visited / progress.total) * 100 : 0;
  const DesktopIcon = !sidebarOpen || sidebarCollapsed ? PanelLeft : PanelLeftClose;
  return (
    <header className="ua-header">
      <div className="ua-header-left">
        <button
          className="ua-icon-btn md:hidden"
          onClick={onMenuClick}
          aria-label="Toggle menu"
          aria-expanded={sidebarOpen}
        >
          <Menu size={20} />
        </button>
        <button
          className="ua-icon-btn hidden md:inline-flex"
          onClick={onSidebarControlClick}
          aria-label="Toggle sidebar"
          aria-expanded={sidebarOpen && !sidebarCollapsed}
        >
          <DesktopIcon size={20} />
        </button>
        <Link to="/" className="ua-brand">
          <span className="ua-brand-mark">ml</span>
          <span className="ua-brand-text">
            <span className="ua-brand-title">ML Animations</span>
            <span className="ua-brand-sub">Interactive machine learning notes</span>
          </span>
        </Link>
      </div>

      <div className="ua-header-right">
        <div className="ua-progress-track" aria-label={progressLabel}>
          <span>{progressLabel}</span>
          <div>
            <i style={{ width: `${progressPercent}%` }} />
          </div>
        </div>
        <button
          type="button"
          className="ua-header-action"
          onClick={onOpenCommandPalette}
          aria-label="Open glossary search"
        >
          <BookOpen size={16} />
          Glossary
        </button>
        <button type="button" className="ua-icon-btn" onClick={onOpenNotes} aria-label="Open notes">
          <PencilLine size={18} />
        </button>
        <a
          className="ua-icon-btn"
          href="https://github.com/danielsobrado/ml-animations"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="GitHub repository"
        >
          <Github size={19} />
        </a>
      </div>
    </header>
  );
}
