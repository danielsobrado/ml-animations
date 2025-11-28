import React, { useState } from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import Sidebar from './components/layout/Sidebar';
import Header from './components/layout/Header';
import HomePage from './pages/HomePage';
import AnimationPage from './pages/AnimationPage';
import { useTheme } from './context/ThemeContext';

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const location = useLocation();
  const { isDark } = useTheme();

  const isHomePage = location.pathname === '/';

  return (
    <div className="min-h-screen flex flex-col">
      <Header 
        onMenuClick={() => setSidebarOpen(!sidebarOpen)}
        onCollapseClick={() => setSidebarCollapsed(!sidebarCollapsed)}
        sidebarCollapsed={sidebarCollapsed}
      />
      
      <div className="flex flex-1 overflow-hidden">
        <Sidebar 
          isOpen={sidebarOpen} 
          isCollapsed={sidebarCollapsed}
          onClose={() => setSidebarOpen(false)}
        />
        
        <main 
          className={`flex-1 overflow-y-auto transition-all duration-300 ${
            sidebarOpen 
              ? sidebarCollapsed 
                ? 'md:ml-20' 
                : 'md:ml-72' 
              : 'ml-0'
          }`}
        >
          <div className="min-h-full">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/animation/:id" element={<AnimationPage />} />
            </Routes>
          </div>
        </main>
      </div>
    </div>
  );
}
