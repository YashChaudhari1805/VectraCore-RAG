import { apiClient } from './client';

export const api = {
  bots: {
    list: () => apiClient.request<{ bots: any[] }>('/api/bots'),
    getMemory: (botId: string) => apiClient.request<any>(`/api/memory/${botId}`),
  },
  content: {
    getFeed: () => apiClient.request<{ total: number; posts: any[] }>('/api/feed'),
    generate: (botId: string) => apiClient.request<any>('/api/generate', {
      method: 'POST',
      body: JSON.stringify({ bot_id: botId }),
    }),
    route: (content: string) => apiClient.request<any>('/api/route', {
      method: 'POST',
      body: JSON.stringify({ post_content: content }),
    }),
    reply: (payload: any) => apiClient.request<any>('/api/reply', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  },
};