import axios, { AxiosInstance } from 'axios';
import type {
  PromptRequest,
  PromptResponse,
  DomainInfo,
  SystemStats,
  WorkflowHistory,
  HealthStatus,
  PromptMetadata,
  PromptVersion,
  Template,
  ExperimentResult,
  PromptFilters,
  WorkflowFilters,
  PaginatedResponse,
} from '@/types/api';

class ApiClient {
  private client: AxiosInstance;

  constructor(baseURL: string = 'http://localhost:8000') {
    this.client = axios.create({
      baseURL,
      timeout: 120000, // Increased to 2 minutes for prompt processing
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          // Handle unauthorized access
          localStorage.removeItem('auth_token');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // Core Prompt Processing
    async processPrompt(request: PromptRequest, signal?: AbortSignal): Promise<PromptResponse> {
    const response = await this.client.post('/api/process-prompt', request, { 
      signal,
      timeout: 0 // Remove timeout since we're using AbortController
    });
    return response.data;
  }


  // System Information
  async getDomains(signal?: AbortSignal): Promise<DomainInfo[]> {
    const response = await this.client.get('/api/domains', { signal });
    return response.data;
  }
  async getStats(signal?: AbortSignal): Promise<SystemStats> {
    const response = await this.client.get('/api/stats', { signal });
    return response.data;
  }

  async getHistory(limit: number = 10, signal?: AbortSignal): Promise<WorkflowHistory[]> {
    const response = await this.client.get(`/api/history?limit=${limit}`, { signal });
    return response.data;
  }

  async getHealth(signal?: AbortSignal): Promise<HealthStatus> {
    const response = await this.client.get('/health', { signal });
    return response.data;
  }

  // Prompt Management
  async getPrompts(filters?: PromptFilters, signal?: AbortSignal): Promise<PaginatedResponse<PromptMetadata>> {
    const params = new URLSearchParams();
    if (filters) {
      Object.entries(filters).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          if (Array.isArray(value)) {
            value.forEach(v => params.append(key, v));
          } else {
            params.append(key, String(value));
          }
        }
      });
    }
    
    const response = await this.client.get(`/api/prompts?${params}`, { signal });
    return response.data;
  }

  async getPrompt(id: string, signal?: AbortSignal): Promise<PromptMetadata> {
    const response = await this.client.get(`/api/prompts/${id}`, { signal });
    return response.data;
  }

  async createPrompt(prompt: Omit<PromptMetadata, 'id' | 'created_at' | 'updated_at'>, signal?: AbortSignal): Promise<PromptMetadata> {
    const response = await this.client.post('/api/prompts', prompt, { signal });
    return response.data;
  }

  async updatePrompt(id: string, prompt: Partial<PromptMetadata>, signal?: AbortSignal): Promise<PromptMetadata> {
    const response = await this.client.put(`/api/prompts/${id}`, prompt, { signal });
    return response.data;
  }

  async deletePrompt(id: string, signal?: AbortSignal): Promise<void> {
    await this.client.delete(`/api/prompts/${id}`, { signal });
  }

  // Prompt Versions
  async getPromptVersions(promptId: string, signal?: AbortSignal): Promise<PromptVersion[]> {
    const response = await this.client.get(`/api/prompts/${promptId}/versions`, { signal });
    return response.data;
  }

  async getPromptVersion(promptId: string, version: string, signal?: AbortSignal): Promise<PromptVersion> {
    const response = await this.client.get(`/api/prompts/${promptId}/versions/${version}`, { signal });
    return response.data;
  }

  async createPromptVersion(promptId: string, version: Omit<PromptVersion, 'id' | 'created_at'>, signal?: AbortSignal): Promise<PromptVersion> {
    const response = await this.client.post(`/api/prompts/${promptId}/versions`, version, { signal });
    return response.data;
  }

  // Templates
  async getTemplates(signal?: AbortSignal): Promise<Template[]> {
    const response = await this.client.get('/api/templates', { signal });
    return response.data;
  }

  async getTemplate(id: string, signal?: AbortSignal): Promise<Template> {
    const response = await this.client.get(`/api/templates/${id}`, { signal });
    return response.data;
  }

  async createTemplate(template: Omit<Template, 'id' | 'created_at' | 'updated_at'>, signal?: AbortSignal): Promise<Template> {
    const response = await this.client.post('/api/templates', template, { signal });
    return response.data;
  }

  async updateTemplate(id: string, template: Partial<Template>, signal?: AbortSignal): Promise<Template> {
    const response = await this.client.put(`/api/templates/${id}`, template, { signal });
    return response.data;
  }

  async deleteTemplate(id: string, signal?: AbortSignal): Promise<void> {
    await this.client.delete(`/api/templates/${id}`, { signal });
  }

  // Experiments
  async getExperiments(signal?: AbortSignal): Promise<ExperimentResult[]> {
    const response = await this.client.get('/api/experiments', { signal });
    return response.data;
  }

  async getExperiment(id: string, signal?: AbortSignal): Promise<ExperimentResult> {
    const response = await this.client.get(`/api/experiments/${id}`, { signal });
    return response.data;
  }

  async createExperiment(experiment: Omit<ExperimentResult, 'experiment_id' | 'created_at' | 'updated_at'>, signal?: AbortSignal): Promise<ExperimentResult> {
    const response = await this.client.post('/api/experiments', experiment, { signal });
    return response.data;
  }

  async updateExperiment(id: string, experiment: Partial<ExperimentResult>, signal?: AbortSignal): Promise<ExperimentResult> {
    const response = await this.client.put(`/api/experiments/${id}`, experiment, { signal });
    return response.data;
  }

  async deleteExperiment(id: string): Promise<void> {
    await this.client.delete(`/api/experiments/${id}`);
  }

  // Analytics
  async getWorkflowAnalytics(filters?: WorkflowFilters): Promise<any> {
    const params = new URLSearchParams();
    if (filters) {
      Object.entries(filters).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          params.append(key, String(value));
        }
      });
    }
    
    const response = await this.client.get(`/api/analytics/workflows?${params}`);
    return response.data;
  }

  async getPerformanceMetrics(): Promise<any> {
    const response = await this.client.get('/api/analytics/performance');
    return response.data;
  }

  async getDomainAnalytics(): Promise<any> {
    const response = await this.client.get('/api/analytics/domains');
    return response.data;
  }

  // Memory-enhanced processing
      async processPromptWithMemory(request: PromptRequest & { user_id: string }, signal?: AbortSignal): Promise<PromptResponse> {
    const response = await this.client.post<PromptResponse>(
      '/api/process-prompt-with-memory',
      request,
      { 
        signal,
        timeout: 0 // Remove timeout since we're using AbortController
      }
    );
    return response.data;
  }


  async generatePrompt(task: string, domain?: string): Promise<PromptResponse> {
    const response = await this.client.post<PromptResponse>('/api/generate-prompt', { task, domain });
    return response.data;
  }

  // Combined Analytics for Analytics page
  async getAnalytics(timeRange: '7d' | '30d' | '90d' = '30d'): Promise<any> {
    const response = await this.client.get(`/api/analytics?time_range=${timeRange}`);
    return response.data;
  }

  // Workflow Management
  async getWorkflows(filters?: { status?: string; domain?: string; page?: number; limit?: number }): Promise<any> {
    const params = new URLSearchParams();
    if (filters) {
      Object.entries(filters).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          params.append(key, String(value));
        }
      });
    }
    
    const response = await this.client.get(`/api/workflows?${params}`);
    return response.data;
  }

  async getWorkflowDetails(workflowId: string): Promise<any> {
    const response = await this.client.get(`/api/workflows/${workflowId}`);
    return response.data;
  }

  async cancelWorkflow(workflowId: string): Promise<any> {
    const response = await this.client.post(`/api/cancel-workflow/${workflowId}`);
    return response.data;
  }

  async getWorkflowStatus(workflowId: string): Promise<any> {
    const response = await this.client.get(`/api/workflow-status/${workflowId}`);
    return response.data;
  }
}

// Create singleton instance
const apiClient = new ApiClient();

// Export for use in React Query
export default apiClient;
