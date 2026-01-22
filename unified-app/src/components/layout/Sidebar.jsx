import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  ChevronDown, 
  ChevronRight, 
  X,
  Home,
} from 'lucide-react';
import { categories } from '../../data/animations';

export default function Sidebar({ isOpen, isCollapsed, onClose }) {
  const location = useLocation();
  const [expandedCategories, setExpandedCategories] = React.useState(() => {
    // Expand all categories by default
    return categories.reduce((acc, cat) => ({ ...acc, [cat.id]: true }), {});
  });

  const toggleCategory = (categoryId) => {
    setExpandedCategories(prev => ({
      ...prev,
      [categoryId]: !prev[categoryId]
    }));
  };

  const isActive = (path) => location.pathname === path;

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <aside 
        className={`
          fixed top-16 bottom-0 left-0 z-40
          bg-white dark:bg-slate-900 
          border-r border-slate-200 dark:border-slate-800
          transition-all duration-300 ease-in-out
          overflow-hidden
          ${isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0 md:w-0'}
          ${isCollapsed ? 'w-20' : 'w-72'}
        `}
      >
        <div className="flex flex-col h-full">
          {/* Mobile close button */}
          <div className="md:hidden flex justify-end p-2">
            <button 
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800"
            >
              <X size={20} />
            </button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 overflow-y-auto custom-scrollbar p-3">
            {/* Home link */}
            <Link
              to="/"
              className={`sidebar-item mb-4 ${isActive('/') ? 'active' : ''}`}
            >
              <Home size={20} />
              {!isCollapsed && <span>Home</span>}
            </Link>

            {/* Divider */}
            <div className="h-px bg-slate-200 dark:bg-slate-700 mb-4" />

            {/* Categories */}
            {categories.map((category) => (
              <div key={category.id} className="mb-2">
                {/* Category header */}
                <button
                  onClick={() => !isCollapsed && toggleCategory(category.id)}
                  className={`
                    w-full flex items-center gap-3 px-3 py-2 rounded-lg
                    text-slate-700 dark:text-slate-300
                    hover:bg-slate-100 dark:hover:bg-slate-800
                    transition-colors duration-200
                    ${isCollapsed ? 'justify-center' : ''}
                  `}
                  title={isCollapsed ? category.name : undefined}
                >
                  <div className={`p-1.5 rounded-lg bg-gradient-to-r ${category.color}`}>
                    <category.icon size={16} className="text-white" />
                  </div>
                  {!isCollapsed && (
                    <>
                      <span className="flex-1 text-left text-sm font-medium truncate">
                        {category.name}
                      </span>
                      {expandedCategories[category.id] ? (
                        <ChevronDown size={16} className="text-slate-700 dark:text-slate-400" />
                      ) : (
                        <ChevronRight size={16} className="text-slate-700 dark:text-slate-400" />
                      )}
                    </>
                  )}
                </button>

                {/* Category items */}
                {!isCollapsed && expandedCategories[category.id] && (
                  <div className="ml-4 mt-1 space-y-1 animate-fade-in">
                    {category.items.map((item) => (
                      <Link
                        key={item.id}
                        to={`/animation/${item.id}`}
                        className={`sidebar-item text-sm ${
                          isActive(`/animation/${item.id}`) ? 'active' : ''
                        }`}
                      >
                        <item.icon size={16} />
                        <span className="truncate">{item.name}</span>
                      </Link>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </nav>

          {/* Footer */}
          {!isCollapsed && (
            <div className="p-4 border-t border-slate-200 dark:border-slate-800">
              <p className="text-xs text-slate-700 dark:text-center">
                ML Animations v1.0
              </p>
            </div>
          )}
        </div>
      </aside>
    </>
  );
}
