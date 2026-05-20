import React, { useState } from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import Sidebar from './components/layout/Sidebar';
import Header from './components/layout/Header';
import HomePage from './pages/HomePage';
import AnimationPage from './pages/AnimationPage';

export default function App() {
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
    const location = useLocation();

    const mainModifier = sidebarOpen
        ? sidebarCollapsed ? 'sidebar-collapsed' : ''
        : 'sidebar-closed';

    return (
        <div className="min-h-screen flex flex-col">
            <Header
                onMenuClick={() => setSidebarOpen(!sidebarOpen)}
                onCollapseClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                sidebarCollapsed={sidebarCollapsed}
            />

            <Sidebar
                isOpen={sidebarOpen}
                isCollapsed={sidebarCollapsed}
                onClose={() => setSidebarOpen(false)}
            />

            <main className={`ua-main ${mainModifier}`}>
                <Routes>
                    <Route path="/" element={<HomePage />} />
                    <Route path="/animation/:id" element={<AnimationPage />} />
                </Routes>
            </main>
        </div>
    );
}
