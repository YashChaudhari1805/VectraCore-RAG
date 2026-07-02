import React, { useState } from 'react';
import { useAuth } from '@/context/AuthContext';
import { Feed } from '../dashboard/Feed';
import { CombatEngine } from '../dashboard/CombatEngine';
import { Sidebar } from './Sidebar';
import { Button } from '@/components/ui/Button';

export const MainLayout: React.FC = () => {
  const { apiKey, logout } = useAuth();
  // Added 'docs' to the possible state values
  const [activeTab, setActiveTab] = useState<'feed' | 'combat' | 'docs'>('feed');

  return (
    <div className="flex h-screen sarvam-mesh overflow-hidden">
      
      {/* Translucent Glass Sidebar */}
      <div className="w-80 h-full p-4">
        <Sidebar />
      </div>
      
      <div className="flex-1 flex flex-col min-w-0 pr-4 py-4">
        
        {/* Floating Top Nav */}
        <header className="h-16 glass-panel rounded-full px-6 flex items-center justify-between mb-4 shrink-0">
          <nav className="flex gap-2">
            <button 
              onClick={() => setActiveTab('feed')}
              className={`px-5 py-2 text-sm font-medium rounded-full transition-all ${activeTab === 'feed' ? 'bg-accent text-accent-fg' : 'text-foreground/70 hover:bg-white/40'}`}
            >
              Live Feed
            </button>
            <button 
              onClick={() => setActiveTab('combat')}
              className={`px-5 py-2 text-sm font-medium rounded-full transition-all ${activeTab === 'combat' ? 'bg-accent text-accent-fg' : 'text-foreground/70 hover:bg-white/40'}`}
            >
              Combat Engine
            </button>
          </nav>

          <div className="flex items-center gap-4">
            {apiKey && (
              <button 
                onClick={logout}
                className="group flex items-center gap-2 px-4 py-2 bg-white/50 rounded-full text-xs font-medium text-foreground border border-white/60 hover:bg-red-500/10 hover:text-red-600 hover:border-red-500/20 transition-all cursor-pointer"
                title="Click to logout"
              >
                <div className="h-2 w-2 rounded-full bg-[#f2722b] group-hover:bg-red-500 transition-colors" />
                <span className="group-hover:hidden">Connected</span>
                <span className="hidden group-hover:inline">Logout</span>
              </button>
            )}
            
            {/* Changed onClick to set the active tab to 'docs' instead of window.open */}
            <Button 
              variant={activeTab === 'docs' ? 'primary' : 'secondary'} 
              size="sm" 
              className="hidden md:flex"
              onClick={() => setActiveTab('docs')}
            >
              API Docs
            </Button>
          </div>
        </header>

        {/* Content Area */}
        <main className="flex-1 overflow-y-auto glass-panel rounded-3xl p-8">
          <div className="max-w-4xl mx-auto h-full">
             {activeTab === 'feed' && <Feed />}
             {activeTab === 'combat' && <CombatEngine />}
             
             {/* Embedded API Docs using an iframe */}
             {activeTab === 'docs' && (
               <div className="w-full h-full bg-white rounded-xl overflow-hidden border border-border shadow-sm animate-in fade-in duration-500">
                 <iframe 
                   src={import.meta.env.DEV ? "http://127.0.0.1:8000/docs" : "/docs"} 
                   className="w-full h-full border-0"
                   title="API Documentation"
                 />
               </div>
             )}
          </div>
        </main>
      </div>
    </div>
  );
};