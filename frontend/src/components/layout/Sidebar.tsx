import React, { useEffect, useState } from 'react';
import { api } from '@/api/endpoints';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';

export const Sidebar: React.FC = () => {
  const [bots, setBots] = useState<any[]>([]);
  const [activeBot, setActiveBot] = useState<string | null>(null);
  const [memory, setMemory] = useState<any>(null);
  const [routeInput, setRouteInput] = useState('');
  const [routeResult, setRouteResult] = useState<any>(null);
  const [isRouting, setIsRouting] = useState(false);
  const [generatingFor, setGeneratingFor] = useState<string | null>(null);

  useEffect(() => {
    api.bots.list().then(res => setBots(res.bots || []));
  }, []);

  const handleSelectBot = async (botId: string) => {
    setActiveBot(botId);
    setMemory(null);
    try {
      const mem = await api.bots.getMemory(botId);
      setMemory(mem);
    } catch (e) {}
  };

  const handleGenerate = async (e: React.MouseEvent, botId: string) => {
    e.stopPropagation();
    setGeneratingFor(botId);
    try {
      await api.content.generate(botId);
      if (activeBot === botId) handleSelectBot(botId);
      // Trigger a custom event to tell the Feed component to refresh
      window.dispatchEvent(new Event('refreshFeed'));
    } finally {
      setGeneratingFor(null);
    }
  };

  const handleRoute = async () => {
    if (!routeInput.trim()) return;
    setIsRouting(true);
    try {
      const res = await api.content.route(routeInput);
      setRouteResult(res.matched_bots || []);
    } finally {
      setIsRouting(false);
    }
  };

  return (
    <aside className="h-full glass-panel rounded-3xl flex flex-col overflow-hidden">
      <div className="h-20 flex items-center px-8 border-b border-white/30 shrink-0">
        <span className="font-serif text-2xl tracking-tight text-accent">vectracore</span>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        
        <div>
          <div className="text-xs font-semibold text-foreground/50 uppercase tracking-widest mb-4 px-4">Personas</div>
          <div className="space-y-3">
            {bots.map(bot => (
              <div 
                key={bot.id}
                onClick={() => handleSelectBot(bot.id)}
                className={`p-4 rounded-2xl transition-all cursor-pointer ${
                  activeBot === bot.id 
                    ? 'bg-white shadow-sm border border-white' 
                    : 'bg-white/30 border border-transparent hover:bg-white/50'
                }`}
              >
                <div className="font-serif text-lg text-accent mb-1">{bot.display_name}</div>
                <div className="text-xs text-foreground/70 leading-relaxed line-clamp-2 mb-4">
                  {bot.description}
                </div>
                <Button 
                  variant={activeBot === bot.id ? 'primary' : 'secondary'} 
                  size="sm" 
                  className="w-full"
                  isLoading={generatingFor === bot.id}
                  onClick={(e) => handleGenerate(e, bot.id)}
                >
                  Generate Post
                </Button>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white/40 rounded-2xl p-4">
          <div className="text-xs font-semibold text-foreground/50 uppercase tracking-widest mb-3">Semantic Router</div>
          <div className="flex gap-2 flex-col">
            <Input 
              placeholder="Test a post..." 
              value={routeInput}
              onChange={e => setRouteInput(e.target.value)}
              className="text-xs"
            />
            <Button size="sm" onClick={handleRoute} isLoading={isRouting} className="w-full">
              Route
            </Button>
          </div>
        </div>
      </div>
    </aside>
  );
};