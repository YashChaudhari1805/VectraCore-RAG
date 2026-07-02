import React, { useEffect, useState } from 'react';
import { api } from '@/api/endpoints';
import { Button } from '@/components/ui/Button';
import { RefreshCw, MessageSquare } from 'lucide-react';

interface Post {
  bot_id: string;
  display_name: string;
  text: string;
  topic: string;
  timestamp: string;
}

export const Feed: React.FC = () => {
  const [posts, setPosts] = useState<Post[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const fetchFeed = async () => {
    setIsLoading(true);
    try {
      const data = await api.content.getFeed();
      setPosts(data.posts || []);
    } catch (error) {
      console.error('Failed to load feed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchFeed();

    // Listen for the custom event fired by the Sidebar
    const handleRefresh = () => fetchFeed();
    window.addEventListener('refreshFeed', handleRefresh);

    return () => {
      window.removeEventListener('refreshFeed', handleRefresh);
    };
  }, []);

  return (
    <div className="flex flex-col h-full space-y-6 animate-in fade-in duration-500">
      <div className="flex items-center justify-between shrink-0">
        <div>
          <h2 className="text-xl font-semibold tracking-tight">Generated Posts</h2>
          <p className="text-sm text-muted mt-1 font-mono">
            {posts.length} {posts.length === 1 ? 'post' : 'posts'} total
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={fetchFeed} isLoading={isLoading}>
          <RefreshCw size={14} className="mr-2" /> Refresh
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto space-y-4 pb-12">
        {posts.length === 0 && !isLoading ? (
          <div className="flex flex-col items-center justify-center h-64 text-muted border border-dashed border-border rounded-xl">
            <MessageSquare size={32} className="mb-4 opacity-50" />
            <p className="text-sm">No posts yet. Generate one from the sidebar.</p>
          </div>
        ) : (
          posts.map((post, idx) => (
            <div key={idx} className="bg-surface border border-border rounded-xl p-5 shadow-sm transition-colors hover:border-muted/30">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="h-8 w-8 rounded-lg bg-accent/10 text-accent flex items-center justify-center font-mono text-xs font-bold">
                    {post.bot_id.includes('Tech') ? 'TM' : post.bot_id.includes('Doom') ? 'DR' : 'FB'}
                  </div>
                  <div>
                    <h4 className="text-sm font-semibold text-foreground">{post.display_name}</h4>
                    <p className="text-xs text-muted font-mono">{post.bot_id}</p>
                  </div>
                </div>
                <span className="px-2.5 py-1 rounded-full bg-accent/5 text-accent text-xs font-mono font-medium">
                  {post.topic}
                </span>
              </div>
              <p className="text-sm text-foreground/90 leading-relaxed">{post.text}</p>
              <div className="mt-4 text-xs text-muted font-mono">
                {new Date(post.timestamp).toLocaleString(undefined, { 
                  month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' 
                })}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};