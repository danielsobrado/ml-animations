import React from 'react';
import { Link } from 'react-router-dom';
import { Menu, Github, PanelLeftClose, PanelLeft } from 'lucide-react';

export default function Header({ onMenuClick, onCollapseClick, sidebarCollapsed }) {
    return (
        <header className="ua-header">
            <div className="ua-header-left">
                {/* Mobile menu */}
                <button
                    onClick={onMenuClick}
                    className="ua-icon-btn md:hidden"
                    aria-label="Toggle menu"
                >
                    <Menu size={18} />
                </button>

                {/* Desktop collapse */}
                <button
                    onClick={onCollapseClick}
                    className="ua-icon-btn"
                    style={{ display: 'inline-flex' }}
                    aria-label="Toggle sidebar"
                >
                    {sidebarCollapsed ? <PanelLeft size={18} /> : <PanelLeftClose size={18} />}
                </button>

                {/* Brand */}
                <Link to="/" className="ua-brand" aria-label="ML Animations — home">
                    <span className="ua-brand-mark">ƒ</span>
                    <span className="ua-brand-text">
                        <span className="ua-brand-title">ML / Animations</span>
                        <span className="ua-brand-sub">an interactive notebook</span>
                    </span>
                </Link>
            </div>

            <div className="ua-header-right">
                <a
                    href="https://github.com/danielsobrado/ml-animations"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="ua-icon-btn"
                    aria-label="GitHub repository"
                >
                    <Github size={18} />
                </a>
            </div>
        </header>
    );
}
