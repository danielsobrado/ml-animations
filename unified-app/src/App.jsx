import React, { useEffect, useState } from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import Header from './components/layout/Header';
import Sidebar from './components/layout/Sidebar';
import HomePage from './pages/HomePage';
import AnimationPage from './pages/AnimationPage';
import GlossaryPage from './pages/GlossaryPage';

export default function App() {
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth >= 768);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  useEffect(() => {
    if (window.innerWidth < 768) setSidebarOpen(false);
  }, [location.pathname]);

  return (
    <div className="ua-app">
      <Header
        onMenuClick={() => setSidebarOpen((open) => !open)}
        onCollapseClick={() => setSidebarCollapsed((collapsed) => !collapsed)}
        sidebarCollapsed={sidebarCollapsed}
      />
      <Sidebar
        isOpen={sidebarOpen}
        isCollapsed={sidebarCollapsed}
        onClose={() => setSidebarOpen(false)}
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
    </div>
  );
}
