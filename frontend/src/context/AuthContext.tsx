import React, { createContext, useContext, useEffect, useState } from 'react';
import { api } from '@/api/endpoints';
import { apiClient } from '@/api/client';

interface AuthContextType {
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (key: string) => Promise<void>;
  logout: () => void;
  apiKey: string;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [apiKey, setApiKeyValue] = useState('');

  useEffect(() => {
    checkAuthStatus();
  }, []);

  const checkAuthStatus = async () => {
    const storedKey = localStorage.getItem('vectracore_api_key');
    if (storedKey) {
      apiClient.setApiKey(storedKey);
      try {
        // Test the stored key against a known endpoint
        await api.bots.list(); 
        setApiKeyValue(storedKey);
        setIsAuthenticated(true);
      } catch (e) {
        // Key is invalid or expired
        apiClient.setApiKey('');
        localStorage.removeItem('vectracore_api_key');
      }
    }
    setIsLoading(false);
  };

  const login = async (key: string) => {
    apiClient.setApiKey(key);
    
    // Instead of a dedicated verify route, we test the key by asking for the bots list
    await api.bots.list(); 
    
    setApiKeyValue(key);
    localStorage.setItem('vectracore_api_key', key);
    setIsAuthenticated(true);
  };

  const logout = () => {
    apiClient.setApiKey('');
    localStorage.removeItem('vectracore_api_key');
    setApiKeyValue('');
    setIsAuthenticated(false);
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, isLoading, login, logout, apiKey }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within AuthProvider');
  return context;
};