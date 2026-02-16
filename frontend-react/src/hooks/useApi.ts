import axios from 'axios';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import apiClient from '@/api/client';
import type {
  PromptRequest,
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


// Core API Hooks - Unified Processing with Optional Memory
export const useProcessPrompt = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async ({ request, signal }: { request: PromptRequest & { user_id?: string }; signal?: AbortSignal }) => {
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
        // Don't toast here — completion notification is handled by polling/streaming in PromptProcessor
      }
    },
    onError: (error: any) => {
      toast.error(getApiErrorMessage(error), { id: 'process-error' });
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


export const useCancelWorkflow = () => {
  return useMutation({
    mutationFn: (workflowId: string) => apiClient.cancelWorkflow(workflowId),
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
    mutationFn: (data: { name: string; domain: string; template_text: string; description?: string; variables?: string[]; is_public?: boolean }) =>
      apiClient.createTemplate(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.templates() });
      toast.success('Template created!');
    },
    onError: (error: any) => toast.error(getApiErrorMessage(error)),
  });
};

export const useUpdateTemplate = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ templateId, data }: { templateId: string; data: { name?: string; domain?: string; template_text?: string; description?: string; variables?: string[]; is_public?: boolean } }) =>
      apiClient.updateTemplate(templateId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.templates() });
      toast.success('Template updated!');
    },
    onError: (error: any) => toast.error(getApiErrorMessage(error)),
  });
};

export const useDeleteTemplate = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (templateId: string) => apiClient.deleteTemplate(templateId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.templates() });
      toast.success('Template deleted!');
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

export const useCacheStats = () => {
  return useQuery({
    queryKey: ['cacheStats'] as const,
    queryFn: () => apiClient.getCacheStats(),
    staleTime: 30 * 1000,
    refetchInterval: 60 * 1000,
    retry: 1,
  });
};
