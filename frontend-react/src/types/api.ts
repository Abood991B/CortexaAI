// API Types for Multi-Agent Prompt Engineering System

export interface PromptRequest {
  prompt: string;
  prompt_type: 'auto' | 'raw' | 'structured';
  return_comparison: boolean;
  use_langgraph: boolean;
  chat_history?: Array<{ role: 'user' | 'assistant'; content: string }>;
}

export interface PromptResponse {
  workflow_id: string;
  status: string;
  timestamp: string;
  processing_time_seconds?: number;
  input: Record<string, any>;
  output: {
    domain: string;
    quality_score: number;
    iterations_used: number;
    optimized_prompt: string;
  };
  analysis?: {
    classification: {
      reasoning: string;
      key_topics: string[];
    };
  };
  comparison?: {
    side_by_side: {
      original: string;
      optimized: string;
    };
    improvement_ratio: number;
  };
  metadata: Record<string, any>;
}

export interface SystemStats {
  total_workflows: number;
  completed_workflows: number;
  error_workflows: number;
  success_rate: number;
  average_quality_score: number;
  average_processing_time: number;
  domain_distribution: Record<string, number>;
}

export interface WorkflowHistory {
  workflow_id: string;
  timestamp: string;
  status: string;
  domain: string;
  quality_score: number;
  processing_time: number;
  prompt_preview: string;
}

export interface HealthStatus {
  status: 'healthy' | 'unhealthy';
  timestamp: number;
  version: string;
  uptime_seconds: number;
  components: {
    llm_providers: Record<string, {
      configured: boolean;
      status: string;
    }>;
    langsmith: {
      enabled: boolean;
      status: string;
    };
    coordinator: {
      status: string;
      available_domains?: number;
      error?: string;
    };
  };
  metrics: {
    total_workflows: number;
    successful_workflows: number;
    failed_workflows: number;
    llm_calls_total: number;
    retry_attempts: number;
  };
  system: {
    memory_percent: number;
    cpu_percent: number;
    active_connections: number;
  };
  readiness: boolean;
  liveness: boolean;
}

export interface ExperimentResult {
  id: string;
  name: string;
  description: string;
  status: 'running' | 'completed' | 'failed';
  variants: Array<{
    name: string;
    conversion_rate?: number;
    traffic_percentage?: number;
  }>;
  metrics: {
    total_samples?: number;
    confidence_level?: number;
    winner?: string;
  };
  created_at: string;
  completed_at?: string;
}

// Error Types
export interface ApiError {
  detail: string;
  error_code?: string;
  timestamp?: string;
  workflow_id?: string;
}

// Request/Response wrapper types
export interface ApiResponse<T> {
  data: T;
  success: boolean;
  message?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  per_page: number;
  pages: number;
}

// Filter and search types
export interface WorkflowFilters {
  status?: string;
  domain?: string;
  date_from?: string;
  date_to?: string;
  min_quality_score?: number;
  max_processing_time?: number;
}

// Additional types
export interface WorkflowSummary {
  workflow_id: string;
  timestamp: string;
  status: string;
  domain: string;
  quality_score: number;
  processing_time: number;
  prompt_preview: string;
}

export interface WorkflowDetails {
  workflow_id: string;
  status: string;
  original_prompt: string;
  optimized_prompt?: string;
  quality_score: number;
  iterations_used: number;
  processing_time: number;
  agent_steps?: Array<{
    agent_type: string;
    output: string;
    processing_time: number;
  }>;
}

