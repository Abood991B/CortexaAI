import axios, { AxiosInstance } from 'axios';
import type {
  PromptRequest,
  PromptResponse,
  SystemStats,
  WorkflowHistory,
  HealthStatus,
  ComplexityResult,
  PromptTemplate,
} from '@/types/api';

class ApiClient {
  private client: AxiosInstance;

  constructor(baseURL: string = '') {
    this.client = axios.create({
      baseURL,
      timeout: 120000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add API key if available.
        // sessionStorage is preferred over localStorage to reduce XSS exposure:
        // the token is cleared when the browser tab closes.
        const token = sessionStorage.getItem('api_key') || localStorage.getItem('api_key');
        if (token) {
          config.headers['X-API-Key'] = token;
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
          sessionStorage.removeItem('api_key');
          localStorage.removeItem('api_key');
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



  async cancelWorkflow(workflowId: string): Promise<any> {
    const response = await this.client.post(`/api/cancel-workflow/${workflowId}`);
    return response.data;
  }

  async getWorkflowStatus(workflowId: string): Promise<any> {
    const response = await this.client.get(`/api/workflow-status/${workflowId}`);
    return response.data;
  }

  // ─── Templates ──────────────────────────────────────────────
  async getTemplates(domain?: string, query?: string): Promise<any> {
    const params = new URLSearchParams();
    if (domain) params.set('domain', domain);
    if (query) params.set('query', query);
    const qs = params.toString();
    const response = await this.client.get(`/api/templates${qs ? '?' + qs : ''}`);
    return response.data;
  }

  async renderTemplate(template_id: string, variables: Record<string, string> = {}): Promise<any> {
    const response = await this.client.post('/api/templates/render', { template_id, variables });
    return response.data;
  }

  // ─── Template Creation ──────────────────────────────────────
  async createTemplate(data: {
    name: string;
    domain: string;
    template_text: string;
    description?: string;
    variables?: string[];
    is_public?: boolean;
  }): Promise<PromptTemplate> {
    const response = await this.client.post('/api/templates', data);
    return response.data;
  }

  // ─── Template Update ────────────────────────────────────────
  async updateTemplate(templateId: string, data: {
    name?: string;
    domain?: string;
    template_text?: string;
    description?: string;
    variables?: string[];
    is_public?: boolean;
  }): Promise<PromptTemplate> {
    const response = await this.client.put(`/api/templates/${templateId}`, data);
    return response.data;
  }

  // ─── Template Delete ────────────────────────────────────────
  async deleteTemplate(templateId: string): Promise<void> {
    await this.client.delete(`/api/templates/${templateId}`);
  }

  // ─── Complexity Analysis ────────────────────────────────────
  async analyzeComplexity(text: string): Promise<ComplexityResult> {
    const response = await this.client.post('/api/complexity', { text });
    return response.data;
  }

  // ─── Cache Stats ────────────────────────────────────────────
  async getCacheStats(): Promise<any> {
    const response = await this.client.get('/api/cache/stats');
    return response.data;
  }

  // ─── Language Detection ─────────────────────────────────────
  async detectLanguage(text: string): Promise<any> {
    const response = await this.client.post('/api/language/detect', { text });
    return response.data;
  }

  // ─── SSE Streaming ──────────────────────────────────────────
  getStreamUrl(): string {
    return `${this.client.defaults.baseURL || ''}/api/process-prompt/stream`;
  }
}

// Create singleton instance
const apiClient = new ApiClient();
export default apiClient;
