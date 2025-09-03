import axios from 'axios';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import apiClient from '@/api/client';
import type {
  PromptRequest,
  PromptResponse,
  PromptMetadata,
  PromptVersion,
  Template,
  ExperimentResult,
  PromptFilters,
  WorkflowFilters,
} from '@/types/api';

// Query Keys
export const queryKeys = {
  domains: ['domains'] as const,
  stats: ['stats'] as const,
  history: (limit?: number) => ['history', limit] as const,
  health: ['health'] as const,
  prompts: (filters?: PromptFilters) => ['prompts', filters] as const,
  prompt: (id: string) => ['prompts', id] as const,
  promptVersions: (promptId: string) => ['prompts', promptId, 'versions'] as const,
  promptVersion: (promptId: string, version: string) => ['prompts', promptId, 'versions', version] as const,
  templates: ['templates'] as const,
  template: (id: string) => ['templates', id] as const,
  experiments: ['experiments'] as const,
  experiment: (id: string) => ['experiments', id] as const,
  workflowAnalytics: (filters?: WorkflowFilters) => ['analytics', 'workflows', filters] as const,
  performanceMetrics: ['analytics', 'performance'] as const,
  domainAnalytics: ['analytics', 'domains'] as const,
  analytics: (timeRange?: string) => ['analytics', timeRange] as const,
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

// Core API Hooks
export const useProcessPrompt = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async ({ request, signal, skipCache = false }: { request: PromptRequest; signal?: AbortSignal; skipCache?: boolean }) => {
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
        const response = await apiClient.processPrompt(request, signal);
        // Don't cache here - let the workflow completion handler decide when to cache
        // setCachedResponse(cacheKey, response);
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
    onError: (error: Error) => {
      toast.error(error.message);
    },
  });
};

export const useDomains = () => {
  return useQuery({
    queryKey: queryKeys.domains,
    queryFn: () => apiClient.getDomains(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

export const useStats = () => {
  return useQuery({
    queryKey: queryKeys.stats,
    queryFn: () => apiClient.getStats(),
    refetchInterval: 30 * 1000, // 30 seconds
  });
};

export const useHistory = (limit: number = 10) => {
  return useQuery({
    queryKey: queryKeys.history(limit),
    queryFn: () => apiClient.getHistory(limit),
    refetchInterval: 60 * 1000, // 1 minute
  });
};

export const useHealth = () => {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: () => apiClient.getHealth(),
    refetchInterval: 30 * 1000, // 30 seconds
  });
};

// Prompt Management Hooks
export const usePrompts = (filters?: PromptFilters) => {
  return useQuery({
    queryKey: queryKeys.prompts(filters),
    queryFn: () => apiClient.getPrompts(filters),
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
};

export const usePrompt = (id: string) => {
  return useQuery({
    queryKey: queryKeys.prompt(id),
    queryFn: () => apiClient.getPrompt(id),
    enabled: !!id,
  });
};

export const useCreatePrompt = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (prompt: Omit<PromptMetadata, 'id' | 'created_at' | 'updated_at'>) => 
      apiClient.createPrompt(prompt),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.prompts() });
      toast.success('Prompt created successfully!');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to create prompt');
    },
  });
};

export const useUpdatePrompt = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ id, prompt }: { id: string; prompt: Partial<PromptMetadata> }) => 
      apiClient.updatePrompt(id, prompt),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.prompts() });
      queryClient.setQueryData(queryKeys.prompt(data.id), data);
      toast.success('Prompt updated successfully!');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to update prompt');
    },
  });
};

export const useDeletePrompt = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (id: string) => apiClient.deletePrompt(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.prompts() });
      toast.success('Prompt deleted successfully!');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to delete prompt');
    },
  });
};

// Prompt Version Hooks
export const usePromptVersions = (promptId: string) => {
  return useQuery({
    queryKey: queryKeys.promptVersions(promptId),
    queryFn: () => apiClient.getPromptVersions(promptId),
    enabled: !!promptId,
  });
};

export const usePromptVersion = (promptId: string, version: string) => {
  return useQuery({
    queryKey: queryKeys.promptVersion(promptId, version),
    queryFn: () => apiClient.getPromptVersion(promptId, version),
    enabled: !!promptId && !!version,
  });
};

export const useCreatePromptVersion = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ promptId, version }: { promptId: string; version: Omit<PromptVersion, 'id' | 'created_at'> }) => 
      apiClient.createPromptVersion(promptId, version),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.promptVersions(data.prompt_id) });
      queryClient.invalidateQueries({ queryKey: queryKeys.prompt(data.prompt_id) });
      toast.success('New version created successfully!');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to create version');
    },
  });
};

// Template Hooks
export const useTemplates = () => {
  return useQuery({
    queryKey: queryKeys.templates,
    queryFn: () => apiClient.getTemplates(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

export const useTemplate = (id: string) => {
  return useQuery({
    queryKey: queryKeys.template(id),
    queryFn: () => apiClient.getTemplate(id),
    enabled: !!id,
  });
};

export const useCreateTemplate = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (template: Omit<Template, 'id' | 'created_at' | 'updated_at'>) => 
      apiClient.createTemplate(template),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.templates });
      toast.success('Template created successfully!');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to create template');
    },
  });
};

export const useUpdateTemplate = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ id, template }: { id: string; template: Partial<Template> }) => 
      apiClient.updateTemplate(id, template),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.templates });
      queryClient.setQueryData(queryKeys.template(data.id), data);
      toast.success('Template updated successfully!');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to update template');
    },
  });
};

export const useDeleteTemplate = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (id: string) => apiClient.deleteTemplate(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.templates });
      toast.success('Template deleted successfully!');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to delete template');
    },
  });
};

// Experiment Hooks
export const useExperiments = () => {
  return useQuery({
    queryKey: queryKeys.experiments,
    queryFn: () => apiClient.getExperiments(),
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
};

export const useExperiment = (id: string) => {
  return useQuery({
    queryKey: queryKeys.experiment(id),
    queryFn: () => apiClient.getExperiment(id),
    enabled: !!id,
  });
};

export const useCreateExperiment = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (experiment: Omit<ExperimentResult, 'experiment_id' | 'created_at' | 'updated_at'>) => 
      apiClient.createExperiment(experiment),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.experiments });
      toast.success('Experiment created successfully!');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to create experiment');
    },
  });
};

// Analytics Hooks
export const useWorkflowAnalytics = (filters?: WorkflowFilters) => {
  return useQuery({
    queryKey: queryKeys.workflowAnalytics(filters),
    queryFn: () => apiClient.getWorkflowAnalytics(filters),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

export const usePerformanceMetrics = () => {
  return useQuery({
    queryKey: queryKeys.performanceMetrics,
    queryFn: () => apiClient.getPerformanceMetrics(),
    refetchInterval: 60 * 1000, // 1 minute
  });
};

export const useDomainAnalytics = () => {
  return useQuery({
    queryKey: queryKeys.domainAnalytics,
    queryFn: () => apiClient.getDomainAnalytics(),
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
};

// Combined Analytics Hook for Analytics page
export const useAnalytics = (timeRange: '7d' | '30d' | '90d' = '30d') => {
  return useQuery({
    queryKey: ['analytics', timeRange],
    queryFn: () => apiClient.getAnalytics(timeRange),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

// Advanced Processing Hooks
export const useProcessPromptWithMemory = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async ({ request, signal, skipCache = false }: { request: PromptRequest & { user_id: string }; signal?: AbortSignal; skipCache?: boolean }) => {
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
        if (!request.user_id) {
          throw new Error('User ID is required for processing with memory');
        }
        const response = await apiClient.processPromptWithMemory(request, signal);
        // Don't cache here - let the workflow completion handler decide when to cache
        // setCachedResponse(cacheKey, response);
        return response;
      } catch (error: any) {
        if (axios.isCancel(error)) {
          // Don't show success toast for cancelled requests
          return;
        }
        console.error('API Error (with memory):', error);
        const errorMessage = error.response?.data?.detail || 
                           error.message || 
                           'Failed to process prompt with memory';
        throw new Error(errorMessage);
      }
    },
    onSuccess: (data) => {
      if (data) { 
        queryClient.invalidateQueries({ queryKey: queryKeys.stats });
        queryClient.invalidateQueries({ queryKey: queryKeys.history() });
        toast.success('Prompt processed with memory successfully!');
      }
    },
    onError: (error: Error) => {
      toast.error(error.message);
    },
  });
};

export const useGeneratePrompt = () => {
  return useMutation({
    mutationFn: ({ task, domain }: { task: string; domain?: string }) => 
      apiClient.generatePrompt(task, domain),
    onSuccess: () => {
      toast.success('Prompt generated successfully!');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to generate prompt');
    },
  });
};

// Workflow Management Hooks
export const useWorkflows = (filters?: { status?: string; domain?: string; page?: number; limit?: number }) => {
  return useQuery({
    queryKey: ['workflows', filters],
    queryFn: () => apiClient.getWorkflows(filters),
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
};

export const useWorkflowDetails = (workflowId: string) => {
  return useQuery({
    queryKey: ['workflows', workflowId],
    queryFn: () => apiClient.getWorkflowDetails(workflowId),
    enabled: !!workflowId,
  });
};

export const useCancelWorkflow = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (workflowId: string) => apiClient.cancelWorkflow(workflowId),
    onSuccess: (_, workflowId) => {
      toast.success('Workflow cancellation requested.');
      // Optionally, you can invalidate queries related to this workflow
      queryClient.invalidateQueries({ queryKey: ['workflows', workflowId] });
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to cancel workflow');
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
