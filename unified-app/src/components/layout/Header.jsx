import React from 'react';
import { Link } from 'react-router-dom';
import { Github, Menu, PanelLeft, PanelLeftClose } from 'lucide-react';

export default function Header({ onMenuClick, onCollapseClick, sidebarCollapsed }) {
  return (
    <header className="ua-header">
      <div className="ua-header-left">
        <button className="ua-icon-btn md:hidden" onClick={onMenuClick} aria-label="Toggle menu">
          <Menu size={20} />
        </button>
        <button className="ua-icon-btn hidden md:inline-flex" onClick={onCollapseClick} aria-label="Toggle sidebar">
          {sidebarCollapsed ? <PanelLeft size={20} /> : <PanelLeftClose size={20} />}
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
