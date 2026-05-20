import React, { useEffect, useState } from 'react';
import { Routes, Route, useLocation, useNavigate } from 'react-router-dom';
import Header from './components/layout/Header';
import Sidebar from './components/layout/Sidebar';
import CommandPalette from './components/app/CommandPalette';
import KeyboardHintDock from './components/app/KeyboardHintDock';
import NotesDrawer from './components/app/NotesDrawer';
import HomePage from './pages/HomePage';
import AnimationPage from './pages/AnimationPage';
import GlossaryPage from './pages/GlossaryPage';
import { allAnimations, getAnimationById } from './data/animations';

const VISITED_KEY = 'ml-animations:visited-lessons';

function isEditableTarget(target) {
  return target?.closest?.('input, textarea, select, [contenteditable="true"]');
}

function readVisitedLessons() {
  try {
    return JSON.parse(localStorage.getItem(VISITED_KEY) || '[]');
  } catch {
    return [];
  }
}

export default function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth >= 768);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const [notesOpen, setNotesOpen] = useState(false);
  const [visitedLessons, setVisitedLessons] = useState(() => new Set(readVisitedLessons()));

  useEffect(() => {
    if (window.innerWidth < 768) setSidebarOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    const match = location.pathname.match(/^\/animation\/([^/]+)/);
    if (!match) return;

    const lessonId = match[1];
    setVisitedLessons((current) => {
      if (current.has(lessonId)) return current;
      const next = new Set(current);
      next.add(lessonId);
      localStorage.setItem(VISITED_KEY, JSON.stringify([...next]));
      return next;
    });
  }, [location.pathname]);

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (isEditableTarget(event.target)) return;

      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        setCommandPaletteOpen(true);
        return;
      }

      if (event.key === '/') {
        event.preventDefault();
        setCommandPaletteOpen(true);
        return;
      }

      if (event.key.toLowerCase() === 'n') {
        event.preventDefault();
        setNotesOpen((open) => !open);
        return;
      }

      if (event.key.toLowerCase() === 's') {
        event.preventDefault();
        navigate('/animation/softmax');
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [navigate]);

  const currentLessonId = location.pathname.match(/^\/animation\/([^/]+)/)?.[1];
  const currentLesson = currentLessonId ? getAnimationById(currentLessonId) : null;

  const handleSidebarControlClick = () => {
    if (!sidebarOpen) {
      setSidebarOpen(true);
      setSidebarCollapsed(false);
      return;
    }

    setSidebarCollapsed((collapsed) => !collapsed);
  };

  return (
    <div className="ua-app">
      <Header
        onMenuClick={() => setSidebarOpen((open) => !open)}
        onSidebarControlClick={handleSidebarControlClick}
        onOpenCommandPalette={() => setCommandPaletteOpen(true)}
        onOpenNotes={() => setNotesOpen(true)}
        progress={{ visited: visitedLessons.size, total: allAnimations.length }}
        sidebarOpen={sidebarOpen}
        sidebarCollapsed={sidebarCollapsed}
      />
      <Sidebar
        isOpen={sidebarOpen}
        isCollapsed={sidebarCollapsed}
        onClose={() => setSidebarOpen(false)}
        onOpenCommandPalette={() => setCommandPaletteOpen(true)}
      />
      <main
        className={`ua-main ${sidebarCollapsed ? 'sidebar-collapsed' : ''} ${
          sidebarOpen ? '' : 'sidebar-closed'
        }`}
      >
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/animation/:id" element={<AnimationPage />} />
          <Route path="/glossary/:slug" element={<GlossaryPage />} />
        </Routes>
      </main>
      <CommandPalette open={commandPaletteOpen} onClose={() => setCommandPaletteOpen(false)} />
      <NotesDrawer
        open={notesOpen}
        scope={currentLessonId || 'catalog'}
        title={currentLesson?.name || 'Catalog'}
        onClose={() => setNotesOpen(false)}
      />
      <KeyboardHintDock />
    </div>
  );
}
