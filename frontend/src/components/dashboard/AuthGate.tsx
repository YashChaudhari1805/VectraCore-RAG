import React, { useState } from 'react';
import { useAuth } from '@/context/AuthContext';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';

export const AuthGate: React.FC = () => {
  const { login } = useAuth();
  const [key, setKey] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    try {
      await login(key);
    } catch (err: any) {
      setError('Invalid API Key');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen sarvam-mesh flex flex-col font-sans">
      
      {/* Floating Navbar */}
      <div className="p-6 w-full max-w-6xl mx-auto">
        <header className="glass-panel rounded-full px-8 py-4 flex items-center justify-between">
          <div className="text-xl font-bold tracking-tight text-accent">vectracore</div>
          <div className="hidden md:flex gap-8 text-sm font-medium text-foreground/80">
            <span className="cursor-pointer hover:text-black">Platform</span>
            <span className="cursor-pointer hover:text-black">Architecture</span>
            <span className="cursor-pointer hover:text-black">Company</span>
          </div>
          <Button variant="secondary" size="sm">Contact Us</Button>
        </header>
      </div>

      {/* Hero Section */}
      <main className="flex-1 flex flex-col items-center justify-center px-4 text-center pb-32">
        
        {/* Decorative Element */}
        <div className="mb-6 text-accent/40 opacity-70">
          <svg width="120" height="24" viewBox="0 0 120 24" fill="currentColor">
             <path d="M60 12c-15-10-25-10-40 0 15 10 25 10 40 0zm0 0c15-10 25-10 40 0-15 10-25 10-40 0z" />
          </svg>
        </div>
        
        <p className="text-accent mb-4 font-medium tracking-wide">Intelligent RAG Platform</p>
        
        <h1 className="font-serif text-6xl md:text-7xl lg:text-8xl text-accent mb-6 leading-tight">
          AI memory,<br />perfected.
        </h1>
        
        <p className="text-lg md:text-xl text-foreground/70 max-w-2xl mx-auto mb-12">
          Built on dynamic semantic routing. Powered by frontier-class models.<br/>
          Delivering enterprise-scale impact.
        </p>

        <form onSubmit={handleSubmit} className="w-full max-w-md flex flex-col gap-4">
          <Input 
            type="password" 
            placeholder="Enter your Access Key..." 
            value={key}
            onChange={(e) => setKey(e.target.value)}
            required
            className="text-center text-base h-14"
          />
          {error && <p className="text-sm text-red-500">{error}</p>}
          <div className="flex justify-center gap-4 mt-2">
            <Button type="submit" size="lg" className="w-40" isLoading={isLoading}>
              Enter Platform
            </Button>
          </div>
        </form>
      </main>
    </div>
  );
};