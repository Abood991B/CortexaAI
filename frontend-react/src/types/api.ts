// API Types for Cortexa Prompt Optimization Tool

export interface PromptRequest {
  prompt: string;
  prompt_type: 'auto' | 'raw' | 'structured';
  return_comparison: boolean;
  use_langgraph: boolean;
  chat_history?: Array<{ role: 'user' | 'assistant'; content: string }>;
  advanced_mode?: boolean;
  synchronous?: boolean;
  user_id?: string;
}

export interface CriteriaScores {
  clarity: number;
  specificity: number;
  structure: number;
  completeness: number;
  actionability: number;
  domain_alignment: number;
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
    passes_threshold?: boolean;
  };
  analysis?: {
    classification: {
      domain?: string;
      confidence?: number;
      reasoning: string;
      key_topics: string[];
    };
    improvements?: {
      improvements_made: string[];
      key_additions: string[];
      effectiveness_score: number;
    };
    evaluation?: {
      overall_score: number;
      criteria_scores: CriteriaScores;
      strengths: string[];
      weaknesses: string[];
      reasoning: string;
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

export interface ComplexityResult {
  score: number;
  level: 'simple' | 'medium' | 'complex';
  signals: Record<string, number>;
  recommended_iterations: number;
  skip_evaluation: boolean;
  token_count: number;
}

export interface SSEEvent {
  event: string;
  data: any;
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

// Template Types
export interface PromptTemplate {
  id?: string;
  name: string;
  description?: string;
  template: string;
  variables?: string[];
  domain?: string;
  category?: string;
  tags?: string[];
  author?: string;
  is_public?: boolean;
  usage_count?: number;
  rating?: number;
  created_at?: string;
}

// Error Types
export interface ApiError {
  detail: string;
  error_code?: string;
  timestamp?: string;
  workflow_id?: string;
}
