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
  async processPrompt(request: PromptRequest): Promise<PromptResponse> {
    const response = await this.client.post<PromptResponse>('/api/process-prompt', request);
    return response.data;
  }


  // System Information
  async getDomains(): Promise<DomainInfo[]> {
    const response = await this.client.get<DomainInfo[]>('/api/domains');
    return response.data;
  }
  async getStats(): Promise<SystemStats> {
    const response = await this.client.get<SystemStats>('/api/stats');
    return response.data;
  }

  async getHistory(limit: number = 10): Promise<WorkflowHistory[]> {
    const response = await this.client.get<WorkflowHistory[]>(`/api/history?limit=${limit}`);
    return response.data;
  }

  async getHealth(): Promise<HealthStatus> {
    const response = await this.client.get<HealthStatus>('/health');
    return response.data;
  }

  // Prompt Management
  async getPrompts(filters?: PromptFilters): Promise<PaginatedResponse<PromptMetadata>> {
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
    
    const response = await this.client.get<PaginatedResponse<PromptMetadata>>(`/api/prompts?${params}`);
    return response.data;
  }

  async getPrompt(id: string): Promise<PromptMetadata> {
    const response = await this.client.get<PromptMetadata>(`/api/prompts/${id}`);
    return response.data;
  }

  async createPrompt(prompt: Omit<PromptMetadata, 'id' | 'created_at' | 'updated_at'>): Promise<PromptMetadata> {
    const response = await this.client.post<PromptMetadata>('/api/prompts', prompt);
    return response.data;
  }

  async updatePrompt(id: string, prompt: Partial<PromptMetadata>): Promise<PromptMetadata> {
    const response = await this.client.put<PromptMetadata>(`/api/prompts/${id}`, prompt);
    return response.data;
  }

  async deletePrompt(id: string): Promise<void> {
    await this.client.delete(`/api/prompts/${id}`);
  }

  // Prompt Versions
  async getPromptVersions(promptId: string): Promise<PromptVersion[]> {
    const response = await this.client.get<PromptVersion[]>(`/api/prompts/${promptId}/versions`);
    return response.data;
  }

  async getPromptVersion(promptId: string, version: string): Promise<PromptVersion> {
    const response = await this.client.get<PromptVersion>(`/api/prompts/${promptId}/versions/${version}`);
    return response.data;
  }

  async createPromptVersion(promptId: string, version: Omit<PromptVersion, 'id' | 'created_at'>): Promise<PromptVersion> {
    const response = await this.client.post<PromptVersion>(`/api/prompts/${promptId}/versions`, version);
    return response.data;
  }

  // Templates
  async getTemplates(): Promise<Template[]> {
    const response = await this.client.get<Template[]>('/api/templates');
    return response.data;
  }

  async getTemplate(id: string): Promise<Template> {
    const response = await this.client.get<Template>(`/api/templates/${id}`);
    return response.data;
  }

  async createTemplate(template: Omit<Template, 'id' | 'created_at' | 'updated_at'>): Promise<Template> {
    const response = await this.client.post<Template>('/api/templates', template);
    return response.data;
  }

  async updateTemplate(id: string, template: Partial<Template>): Promise<Template> {
    const response = await this.client.put<Template>(`/api/templates/${id}`, template);
    return response.data;
  }

  async deleteTemplate(id: string): Promise<void> {
    await this.client.delete(`/api/templates/${id}`);
  }

  // Experiments
  async getExperiments(): Promise<ExperimentResult[]> {
    const response = await this.client.get<ExperimentResult[]>('/api/experiments');
    return response.data;
  }

  async getExperiment(id: string): Promise<ExperimentResult> {
    const response = await this.client.get<ExperimentResult>(`/api/experiments/${id}`);
    return response.data;
  }

  async createExperiment(experiment: Omit<ExperimentResult, 'experiment_id' | 'created_at' | 'updated_at'>): Promise<ExperimentResult> {
    const response = await this.client.post<ExperimentResult>('/api/experiments', experiment);
    return response.data;
  }

  async updateExperiment(id: string, experiment: Partial<ExperimentResult>): Promise<ExperimentResult> {
    const response = await this.client.put<ExperimentResult>(`/api/experiments/${id}`, experiment);
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

  // Memory and Planning (if available)
  async processPromptWithMemory(request: PromptRequest & { user_id: string }): Promise<PromptResponse> {
    const { user_id, ...promptRequest } = request;
    const response = await this.client.post<PromptResponse>(
      `/api/process-prompt-with-memory?user_id=${encodeURIComponent(user_id)}`, 
      promptRequest
    );
    return response.data;
  }

  async processPromptWithPlanning(request: PromptRequest & { user_id?: string }): Promise<PromptResponse> {
    const response = await this.client.post<PromptResponse>('/api/process-prompt-planning', request);
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
}

// Create singleton instance
export const apiClient = new ApiClient();

// Export for use in React Query
export default apiClient;
