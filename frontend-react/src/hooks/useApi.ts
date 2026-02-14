import axios from 'axios';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import apiClient from '@/api/client';
import type {
  PromptRequest,
  PromptResponse,
  ComplexityResult,
} from '@/types/api';

// Helper to format API errors for display
const getApiErrorMessage = (error: any): string => {
  if (axios.isAxiosError(error) && error.response?.data?.detail) {
    const detail = error.response.data.detail;
    if (Array.isArray(detail)) {
      return detail.map(err => `Error in '${err.loc.join('.')}': ${err.msg}`).join('\n');
    }
    if (typeof detail === 'string') {
      return detail;
    }
  }
  if (axios.isAxiosError(error) && error.code === 'ERR_NETWORK') {
    return 'Cannot connect to backend. Is the server running?';
  }
  return error.message || 'An unexpected error occurred.';
};

// Query Keys
export const queryKeys = {
  stats: ['stats'] as const,
  history: (limit?: number) => ['history', limit] as const,
  health: ['health'] as const,
  templates: (domain?: string, query?: string) => ['templates', domain, query] as const,
};

// Caching layer for prompt processing
const CACHE_KEY_PREFIX = 'prompt_cache_';
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

export const getCacheKey = (request: PromptRequest): string => {
  const sortedRequest = Object.keys(request)
    .sort()
    .reduce((acc: { [key: string]: any }, key) => {
      acc[key] = request[key as keyof PromptRequest];
      return acc;
    }, {});
  return `${CACHE_KEY_PREFIX}${JSON.stringify(sortedRequest)}`;
};

// Clear cache for a specific request
export const clearCacheForRequest = (request: PromptRequest): void => {
  const cacheKey = getCacheKey(request);
  localStorage.removeItem(cacheKey);
};

const getCachedResponse = (cacheKey: string): PromptResponse | null => {
  const cachedItem = localStorage.getItem(cacheKey);
  if (cachedItem) {
    try {
      const { data, timestamp } = JSON.parse(cachedItem);
      if (Date.now() - timestamp < CACHE_TTL) {
        return data as PromptResponse;
      }
      localStorage.removeItem(cacheKey);
    } catch (error) {
      console.error('Failed to parse cache item:', error);
      localStorage.removeItem(cacheKey);
    }
  }
  return null;
};

export const setCachedResponse = (cacheKey: string, data: PromptResponse): void => {
  try {
    const cacheItem = {
      data,
      timestamp: Date.now(),
    };
    localStorage.setItem(cacheKey, JSON.stringify(cacheItem));
  } catch (error) {
    console.error('Failed to set cache item:', error);
  }
};

// Core API Hooks - Unified Processing with Optional Memory
export const useProcessPrompt = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async ({ request, signal, skipCache = false }: { request: PromptRequest & { user_id?: string }; signal?: AbortSignal; skipCache?: boolean }) => {
      const cacheKey = getCacheKey(request);
      
      // Clear any existing cache for this request when skipCache is true (for retries)
      if (skipCache) {
        localStorage.removeItem(cacheKey);
      }
      
      // Check cache only if not explicitly skipping
      if (!skipCache) {
        const cachedResponse = getCachedResponse(cacheKey);
        if (cachedResponse) {
          toast.info('Returning cached response.');
          return cachedResponse;
        }
      }

      try {
        // Use memory-enhanced processing if user_id is provided, otherwise standard processing
        const response = request.user_id 
          ? await apiClient.processPromptWithMemory(request as PromptRequest & { user_id: string }, signal)
          : await apiClient.processPrompt(request, signal);
        return response;
      } catch (error: any) {
        if (axios.isCancel(error)) {
          // Don't show success toast for cancelled requests
          return;
        }
        console.error('API Error:', error);
        const errorMessage = error.response?.data?.detail || 
                           error.message || 
                           'Failed to process prompt';
        throw new Error(errorMessage);
      }
    },
    onSuccess: (data) => {
      if (data) { 
        queryClient.invalidateQueries({ queryKey: queryKeys.stats });
        queryClient.invalidateQueries({ queryKey: queryKeys.history() });
        toast.success('Prompt processed successfully!');
      }
    },
    onError: (error: any) => {
      toast.error(getApiErrorMessage(error));
    },
  });
};


export const useStats = () => {
  return useQuery({
    queryKey: queryKeys.stats,
    queryFn: () => apiClient.getStats(),
    staleTime: 60 * 1000,
    refetchInterval: 60 * 1000,
    retry: 1,
    retryDelay: 5000,
  });
};

export const useHistory = (limit?: number) => {
  return useQuery({
    queryKey: queryKeys.history(limit),
    queryFn: () => apiClient.getHistory(limit),
    staleTime: 30 * 1000,
    refetchInterval: 60 * 1000,
    retry: 1,
  });
};

export const useHealth = () => {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: () => apiClient.getHealth(),
    refetchInterval: 30 * 1000,
    staleTime: 20 * 1000,
    refetchOnWindowFocus: true,
    refetchOnMount: true,
    retry: 1,
    retryDelay: 5000,
  });
};



// Analytics Hooks

// Legacy alias for backward compatibility - will be removed
export const useProcessPromptWithMemory = useProcessPrompt;


export const useCancelWorkflow = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (workflowId: string) => apiClient.cancelWorkflow(workflowId),
    onSuccess: (_, workflowId) => {
      toast.success('Workflow cancellation requested.');
      queryClient.invalidateQueries({ queryKey: ['workflows', workflowId] });
    },
    onError: (error: any) => {
      toast.error(getApiErrorMessage(error));
    },
  });
};

export const useWorkflowStatus = (workflowId: string | null) => {
  return useQuery<any, Error>({
    queryKey: ['workflowStatus', workflowId],
    queryFn: () => apiClient.getWorkflowStatus(workflowId!),
    enabled: !!workflowId,
    refetchInterval: (query) => {
        const data: any = query.state.data;
        return data?.status === 'running' ? 2000 : false;
    },
  });
};

// ─── Template & Render Hooks ──────────────────────────────────

export const useTemplates = (domain?: string, query?: string) => {
  return useQuery({
    queryKey: queryKeys.templates(domain, query),
    queryFn: () => apiClient.getTemplates(domain, query),
    staleTime: 5 * 60 * 1000,
    retry: 1,
  });
};

export const useRenderTemplate = () => {
  return useMutation({
    mutationFn: ({ template_id, variables }: { template_id: string; variables?: Record<string, string> }) =>
      apiClient.renderTemplate(template_id, variables),
    onError: (error: any) => toast.error(getApiErrorMessage(error)),
  });
};

export const useCreateTemplate = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: { name: string; domain: string; template_text: string; description?: string; variables?: string[] }) =>
      apiClient.createTemplate(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.templates() });
      toast.success('Template created!');
    },
    onError: (error: any) => toast.error(getApiErrorMessage(error)),
  });
};

export const useComplexity = () => {
  return useMutation<ComplexityResult, Error, string>({
    mutationFn: (text: string) => apiClient.analyzeComplexity(text),
  });
};

export const useDetectLanguage = () => {
  return useMutation<any, Error, string>({
    mutationFn: (text: string) => apiClient.detectLanguage(text),
  });
};


