import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Send, Copy, Download, Trash2, Edit, Sparkles, Bot, User, Plus, Bell, ChevronDown, ChevronUp, Check, HelpCircle, Gauge, BarChart3, ArrowLeftRight, FileJson, FileText, ClipboardCopy, Keyboard, Clock, RefreshCw } from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, Tooltip } from 'recharts';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { LoadingSpinner } from '@/components/ui/loading';
import { ConfirmationDialog } from '@/components/ui/confirmation-dialog';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';
import { 
  useProcessPrompt,
  useCancelWorkflow,
  useWorkflowStatus,
  useComplexity
} from '@/hooks/useApi';
import { formatDuration } from '@/utils';
import { toast } from 'sonner';
import { useNotifications } from '@/hooks/useNotifications';
import { NotificationsDropdown } from '@/components/ui/notifications';
import { Sidebar } from '@/components/layout';
import type { PromptRequest, PromptResponse, ComplexityResult } from '@/types/api';

// Define message types for chat interface
interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  model?: 'standard' | 'langgraph';
  response?: PromptResponse;
  isLoading?: boolean;
  error?: string;
}

interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  lastUpdated: Date;
  model: 'standard' | 'langgraph';
}

/* ── Evaluation Criteria Breakdown ────────────────────────────── */
const CRITERIA_LABELS: Record<string, string> = {
  clarity: 'Clarity',
  specificity: 'Specificity',
  structure: 'Structure',
  completeness: 'Completeness',
  actionability: 'Actionability',
  domain_alignment: 'Domain Fit',
};

const CRITERIA_DESCRIPTIONS: Record<string, string> = {
  clarity: 'How unambiguous and easy to understand the prompt is',
  specificity: 'How precisely inputs, outputs, and constraints are defined',
  structure: 'How well-organized with headings, lists, and logical flow',
  completeness: 'Whether all information needed to fulfill the task is present',
  actionability: 'Whether someone can execute the prompt immediately',
  domain_alignment: 'How well the prompt fits the detected domain',
};

function EvaluationBreakdown({ response }: { response: PromptResponse }) {
  const [expanded, setExpanded] = useState(false);
  const criteria = response.analysis?.evaluation?.criteria_scores;
  const strengths = response.analysis?.evaluation?.strengths || [];
  const weaknesses = response.analysis?.evaluation?.weaknesses || [];

  if (!criteria || Object.keys(criteria).length === 0) return null;

  const radarData = Object.entries(criteria).map(([key, value]) => ({
    criterion: CRITERIA_LABELS[key] || key,
    score: Math.round((value as number) * 100),
    fullMark: 100,
  }));

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 dark:text-green-400';
    if (score >= 0.6) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getBarWidth = (score: number) => `${Math.round(score * 100)}%`;
  const getBarColor = (score: number) => {
    if (score >= 0.8) return 'bg-green-500';
    if (score >= 0.6) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="mt-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs font-medium text-primary hover:text-primary/80 transition-colors"
      >
        <BarChart3 className="h-3.5 w-3.5" />
        {expanded ? 'Hide' : 'Show'} Quality Breakdown
        {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
      </button>

      {expanded && (
        <div className="mt-3 p-4 rounded-xl bg-muted/50 border border-border/40 space-y-4 animate-fade-in">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Radar Chart */}
            <div className="flex items-center justify-center">
              <ResponsiveContainer width="100%" height={220}>
                <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
                  <PolarGrid stroke="hsl(var(--border))" />
                  <PolarAngleAxis dataKey="criterion" tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 10 }} tickCount={5} />
                  <Radar name="Score" dataKey="score" stroke="hsl(var(--primary))" fill="hsl(var(--primary))" fillOpacity={0.25} strokeWidth={2} />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '0.5rem', fontSize: '0.75rem' }}
                    formatter={(value: number) => [`${value}%`, 'Score']}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* Score bars */}
            <div className="space-y-2.5">
              {Object.entries(criteria).map(([key, value]) => (
                <div key={key} className="group" title={CRITERIA_DESCRIPTIONS[key]}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-medium">{CRITERIA_LABELS[key] || key}</span>
                    <span className={`text-xs font-bold ${getScoreColor(value as number)}`}>{Math.round((value as number) * 100)}%</span>
                  </div>
                  <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${getBarColor(value as number)}`}
                      style={{ width: getBarWidth(value as number) }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Strengths & Weaknesses */}
          {(strengths.length > 0 || weaknesses.length > 0) && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2 border-t border-border/30">
              {strengths.length > 0 && (
                <div>
                  <h5 className="text-xs font-semibold text-green-600 dark:text-green-400 mb-1.5">Strengths</h5>
                  <ul className="space-y-1">
                    {strengths.slice(0, 3).map((s, i) => (
                      <li key={i} className="text-xs text-muted-foreground flex gap-1.5">
                        <span className="text-green-500 shrink-0">+</span> {s}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {weaknesses.length > 0 && (
                <div>
                  <h5 className="text-xs font-semibold text-amber-600 dark:text-amber-400 mb-1.5">Areas to Improve</h5>
                  <ul className="space-y-1">
                    {weaknesses.slice(0, 3).map((w, i) => (
                      <li key={i} className="text-xs text-muted-foreground flex gap-1.5">
                        <span className="text-amber-500 shrink-0">-</span> {w}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Complexity Badge ─────────────────────────────────────────── */
function ComplexityBadge({ complexity }: { complexity: ComplexityResult | null }) {
  if (!complexity) return null;

  const config = {
    simple: { color: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 border-green-200 dark:border-green-800', label: 'Simple' },
    medium: { color: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400 border-yellow-200 dark:border-yellow-800', label: 'Medium' },
    complex: { color: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400 border-red-200 dark:border-red-800', label: 'Complex' },
  };

  const c = config[complexity.level] || config.simple;

  return (
    <div className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] font-medium border ${c.color} transition-all`} title={`Complexity: ${(complexity.score * 100).toFixed(0)}% · ~${complexity.token_count} tokens · ${complexity.recommended_iterations} iteration${complexity.recommended_iterations > 1 ? 's' : ''} recommended`}>
      <Gauge className="h-3 w-3" />
      {c.label}
      <span className="opacity-70">·</span>
      <span className="opacity-70">{complexity.token_count} tokens</span>
    </div>
  );
}

/* ── SSE Streaming Hook ───────────────────────────────────────── */

/* ── Diff View ────────────────────────────────────────────────── */
function DiffView({ comparison }: { comparison: PromptResponse['comparison'] }) {
  const [showDiff, setShowDiff] = useState(false);

  if (!comparison?.side_by_side) return null;

  const { original, optimized } = comparison.side_by_side;
  const improvement = comparison.improvement_ratio;

  return (
    <div className="mt-2">
      <button
        onClick={() => setShowDiff(!showDiff)}
        className="flex items-center gap-1.5 text-xs font-medium text-primary hover:text-primary/80 transition-colors"
      >
        <ArrowLeftRight className="h-3.5 w-3.5" />
        {showDiff ? 'Hide' : 'Show'} Before / After
        {showDiff ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
      </button>

      {showDiff && (
        <div className="mt-3 rounded-xl bg-muted/50 border border-border/40 overflow-hidden animate-fade-in">
          {improvement !== undefined && (
            <div className="px-4 py-2 border-b border-border/40 bg-muted/30">
              <span className="text-xs text-muted-foreground">
                Improvement: <strong className="text-green-600 dark:text-green-400">+{Math.round(improvement * 100)}%</strong>
              </span>
            </div>
          )}
          <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-border/40">
            <div className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-semibold text-red-600 dark:text-red-400 uppercase tracking-wide">Original</span>
              </div>
              <pre className="text-xs whitespace-pre-wrap text-muted-foreground leading-relaxed max-h-64 overflow-auto">{original}</pre>
            </div>
            <div className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-semibold text-green-600 dark:text-green-400 uppercase tracking-wide">Optimized</span>
              </div>
              <pre className="text-xs whitespace-pre-wrap text-foreground/90 leading-relaxed max-h-64 overflow-auto">{optimized}</pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Export Menu ───────────────────────────────────────────────── */
function ExportMenu({ message }: { message: ChatMessage }) {
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setOpen(false);
    };
    if (open) document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  const copyAsText = () => {
    navigator.clipboard.writeText(message.content);
    toast.success('Copied as plain text');
    setOpen(false);
  };

  const copyAsMarkdown = () => {
    const md = buildMarkdown(message);
    navigator.clipboard.writeText(md);
    toast.success('Copied as Markdown');
    setOpen(false);
  };

  const downloadAsJson = () => {
    const json = {
      role: message.role,
      content: message.content,
      timestamp: message.timestamp,
      model: message.model,
      ...(message.response && {
        quality_score: message.response.output?.quality_score,
        domain: message.response.output?.domain,
        iterations: message.response.output?.iterations_used,
        analysis: message.response.analysis,
        comparison: message.response.comparison,
      }),
    };
    const blob = new Blob([JSON.stringify(json, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `prompt_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success('Downloaded as JSON');
    setOpen(false);
  };

  return (
    <div ref={menuRef} className="relative inline-block">
      <Button variant="ghost" size="sm" onClick={() => setOpen(!open)} className="h-5 w-5 p-0">
        <Copy className="h-3 w-3" />
      </Button>
      {open && (
        <div className="absolute bottom-full mb-1 right-0 w-44 bg-popover border rounded-xl shadow-lg z-20 overflow-hidden animate-fade-in">
          <button onClick={copyAsText} className="flex items-center gap-2 w-full px-3 py-2 text-xs hover:bg-accent transition-colors text-left">
            <ClipboardCopy className="h-3.5 w-3.5" /> Copy as Text
          </button>
          <button onClick={copyAsMarkdown} className="flex items-center gap-2 w-full px-3 py-2 text-xs hover:bg-accent transition-colors text-left">
            <FileText className="h-3.5 w-3.5" /> Copy as Markdown
          </button>
          <button onClick={downloadAsJson} className="flex items-center gap-2 w-full px-3 py-2 text-xs hover:bg-accent transition-colors text-left border-t border-border/40">
            <FileJson className="h-3.5 w-3.5" /> Download JSON
          </button>
        </div>
      )}
    </div>
  );
}

function buildMarkdown(msg: ChatMessage): string {
  const lines: string[] = [];
  lines.push(`## Optimized Prompt\n`);
  lines.push(msg.content);
  if (msg.response?.output) {
    const o = msg.response.output;
    lines.push(`\n---\n`);
    lines.push(`| Metric | Value |`);
    lines.push(`|--------|-------|`);
    lines.push(`| Quality | ${o.quality_score?.toFixed(2)} |`);
    lines.push(`| Domain | ${o.domain} |`);
    lines.push(`| Iterations | ${o.iterations_used} |`);
  }
  if (msg.response?.comparison?.side_by_side) {
    const c = msg.response.comparison.side_by_side;
    lines.push(`\n### Original\n`);
    lines.push(c.original);
    lines.push(`\n### Optimized\n`);
    lines.push(c.optimized);
  }
  return lines.join('\n');
}

/* ── Keyboard Shortcuts Dialog ────────────────────────────────── */
const SHORTCUTS = [
  { keys: ['Enter'], description: 'Send message' },
  { keys: ['Shift', 'Enter'], description: 'New line' },
  { keys: ['Ctrl', 'N'], description: 'New chat' },
  { keys: ['Ctrl', '/'], description: 'Toggle Advanced mode' },
  { keys: ['Ctrl', 'Shift', 'E'], description: 'Download conversation' },
  { keys: ['Alt', '1-4'], description: 'Navigate pages' },
  { keys: ['?'], description: 'Show this dialog' },
];

function KeyboardShortcutsDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md rounded-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Keyboard className="h-5 w-5 text-primary" />
            Keyboard Shortcuts
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-2 py-2">
          {SHORTCUTS.map((s, i) => (
            <div key={i} className="flex items-center justify-between px-2 py-1.5 rounded-lg hover:bg-muted/50">
              <span className="text-sm text-muted-foreground">{s.description}</span>
              <div className="flex items-center gap-1">
                {s.keys.map((k, j) => (
                  <span key={j}>
                    <kbd className="px-2 py-0.5 text-xs font-mono bg-muted border border-border/60 rounded-md shadow-sm">{k}</kbd>
                    {j < s.keys.length - 1 && <span className="text-muted-foreground mx-0.5">+</span>}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
        <DialogFooter>
          <Button onClick={() => onOpenChange(false)} className="rounded-xl">Close</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

interface StreamingState {
  stage: string;
  message: string;
  domain?: string;
  preview?: string;
  score?: number;
  iterations?: number;
  result?: PromptResponse;
  error?: string;
}

function useSSEStream() {
  const [streamState, setStreamState] = useState<StreamingState | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const startStream = async (request: PromptRequest): Promise<PromptResponse | null> => {
    abortRef.current = new AbortController();
    setIsStreaming(true);
    setStreamState({ stage: 'started', message: 'Starting workflow...' });

    return new Promise((resolve, reject) => {
      const url = '/api/process-prompt/stream';
      
      fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
        signal: abortRef.current!.signal,
      }).then(async (response) => {
        if (!response.ok) {
          const err = await response.text();
          throw new Error(err || 'Stream request failed');
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error('No response body');

        const decoder = new TextDecoder();
        let buffer = '';
        let finalResult: PromptResponse | null = null;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          let currentEvent = '';
          for (const line of lines) {
            if (line.startsWith('event: ')) {
              currentEvent = line.slice(7).trim();
            } else if (line.startsWith('data: ') && currentEvent) {
              try {
                const data = JSON.parse(line.slice(6));
                switch (currentEvent) {
                  case 'started':
                    setStreamState({ stage: 'started', message: 'Workflow started...', preview: data.prompt_preview });
                    break;
                  case 'classifying':
                    setStreamState({ stage: 'classifying', message: 'Classifying domain...' });
                    break;
                  case 'classified':
                    setStreamState({ stage: 'classified', message: `Domain: ${data.domain}`, domain: data.domain });
                    break;
                  case 'improving':
                    setStreamState(prev => ({ ...prev!, stage: 'improving', message: data.message || 'Generating improvement...' }));
                    break;
                  case 'improved':
                    setStreamState(prev => ({ ...prev!, stage: 'improved', message: 'Improvement ready', preview: data.preview }));
                    break;
                  case 'evaluating':
                    setStreamState(prev => ({ ...prev!, stage: 'evaluating', message: 'Evaluating quality...' }));
                    break;
                  case 'evaluated':
                    setStreamState(prev => ({ ...prev!, stage: 'evaluated', message: `Score: ${data.score?.toFixed(2)}`, score: data.score, iterations: data.iterations }));
                    break;
                  case 'iterating':
                    setStreamState(prev => ({ ...prev!, stage: 'iterating', message: `Iteration ${data.iteration || ''}...` }));
                    break;
                  case 'completed':
                    finalResult = data.result;
                    setStreamState(prev => ({ ...prev!, stage: 'completed', message: 'Complete', result: data.result }));
                    break;
                  case 'error':
                    setStreamState(prev => ({ ...prev!, stage: 'error', message: data.message, error: data.message }));
                    break;
                }
              } catch { /* skip malformed json */ }
              currentEvent = '';
            }
          }
        }
        setIsStreaming(false);
        resolve(finalResult);
      }).catch((err) => {
        if (err.name === 'AbortError') {
          setIsStreaming(false);
          resolve(null);
        } else {
          setIsStreaming(false);
          setStreamState({ stage: 'error', message: err.message, error: err.message });
          reject(err);
        }
      });
    });
  };

  const cancelStream = () => {
    abortRef.current?.abort();
    setIsStreaming(false);
    setStreamState(null);
  };

  return { streamState, isStreaming, startStream, cancelStream };
}

/* ── Streaming Progress Indicator ─────────────────────────────── */
const STAGE_ORDER = ['started', 'classifying', 'classified', 'improving', 'improved', 'evaluating', 'evaluated', 'completed'];
const STAGE_LABELS: Record<string, string> = {
  started: 'Starting',
  classifying: 'Analyzing Domain',
  classified: 'Domain Identified',
  improving: 'Generating Improvement',
  improved: 'Improvement Ready',
  evaluating: 'Evaluating Quality',
  evaluated: 'Quality Assessed',
  iterating: 'Refining Further',
  completed: 'Complete',
};

/* Pipeline step icons for visual timeline */
const PIPELINE_STEPS = [
  { stage: 'classifying', label: 'Classify', icon: '1' },
  { stage: 'improving', label: 'Improve', icon: '2' },
  { stage: 'evaluating', label: 'Evaluate', icon: '3' },
  { stage: 'completed', label: 'Done', icon: '4' },
];

/* Example prompts for the welcome screen */
const EXAMPLE_PROMPTS = [
  { text: 'Build a REST API with authentication and rate limiting', domain: 'Software Engineering' },
  { text: 'Create a lesson plan for teaching photosynthesis to 8th graders', domain: 'Education' },
  { text: 'Write a quarterly financial report summarizing Q3 revenue', domain: 'Report Writing' },
  { text: 'Analyze customer churn using logistic regression on this dataset', domain: 'Data Science' },
];

/** Generate a descriptive session title from the domain and prompt text. */
function generateSessionTitle(prompt: string, domain?: string): string {
  // Clean up the prompt text: collapse whitespace, strip leading filler words
  const cleaned = prompt.replace(/\s+/g, ' ').trim();
  const short = cleaned.length > 48 ? cleaned.substring(0, 48).replace(/\s\S*$/, '') + '…' : cleaned;
  if (domain && domain !== 'general' && domain !== 'unknown') {
    // Capitalise first letter of domain
    const label = domain.charAt(0).toUpperCase() + domain.slice(1);
    return `${label}: ${short}`;
  }
  return short || 'New Conversation';
}

function StreamingProgress({ state }: { state: StreamingState }) {
  const currentIdx = STAGE_ORDER.indexOf(state.stage);
  const progress = state.stage === 'completed' ? 100 : Math.max(5, ((currentIdx + 1) / STAGE_ORDER.length) * 100);

  /* Map stage to pipeline step index */
  const getStepStatus = (stepStage: string) => {
    const stepIdx = STAGE_ORDER.indexOf(stepStage);
    if (stepIdx < 0) return 'pending';
    if (currentIdx > stepIdx) return 'done';
    if (currentIdx === stepIdx || currentIdx === stepIdx + 1) return 'active';
    return 'pending';
  };

  return (
    <div className="space-y-3">
      {/* Step timeline */}
      <div className="flex items-center gap-1">
        {PIPELINE_STEPS.map((step, i) => {
          const status = getStepStatus(step.stage);
          return (
            <React.Fragment key={step.stage}>
              <div className="flex flex-col items-center gap-1">
                <div className={`h-7 w-7 rounded-full flex items-center justify-center text-xs font-bold transition-all duration-500 ${
                  status === 'done' ? 'bg-green-500 text-white' :
                  status === 'active' ? 'bg-primary text-primary-foreground ring-2 ring-primary/30 animate-pulse' :
                  'bg-muted text-muted-foreground'
                }`}>
                  {status === 'done' ? '\u2713' : step.icon}
                </div>
                <span className={`text-[10px] font-medium ${
                  status === 'active' ? 'text-primary' : status === 'done' ? 'text-green-600' : 'text-muted-foreground'
                }`}>{step.label}</span>
              </div>
              {i < PIPELINE_STEPS.length - 1 && (
                <div className={`flex-1 h-0.5 rounded-full mb-4 transition-colors duration-500 ${
                  getStepStatus(PIPELINE_STEPS[i + 1].stage) !== 'pending' ? 'bg-green-500' : 'bg-muted'
                }`} />
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* Status text + progress bar */}
      <div className="flex items-center gap-2.5">
        <LoadingSpinner size="sm" />
        <div className="flex-1 space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">{STAGE_LABELS[state.stage] || state.stage}...</span>
            {state.domain && <Badge variant="outline" className="text-[10px] h-4">{state.domain}</Badge>}
          </div>
          <div className="h-1.5 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-all duration-700 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      </div>
      {state.preview && state.stage === 'improved' && (
        <p className="text-xs text-muted-foreground italic pl-7 line-clamp-2">{state.preview}</p>
      )}
      {state.score !== undefined && state.stage === 'evaluated' && (
        <p className="text-xs text-muted-foreground pl-7">Quality score: <strong>{state.score.toFixed(2)}</strong></p>
      )}
    </div>
  );
}

export function PromptProcessor() {
  const location = useLocation();
  const navigate = useNavigate();
  const [selectedModel, setSelectedModel] = useState<'standard' | 'langgraph'>('standard');
  const [currentInput, setCurrentInput] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const currentSessionIdRef = useRef<string | null>(null);
  const [showSidebar, setShowSidebar] = useState(true);
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');
  const [userId, setUserId] = useState('');
  const [workflowId, setWorkflowId] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
  const [isAdvancedMode, setIsAdvancedMode] = useState(false);
  const [isUserGuideOpen, setIsUserGuideOpen] = useState(false);
  const [reiteratingId, setReiteratingId] = useState<string | null>(null);
  const [reiterateFeedback, setReiterateFeedback] = useState('');
  const [reiterateDialogMsg, setReiterateDialogMsg] = useState<ChatMessage | null>(null);
  const [isShortcutsOpen, setIsShortcutsOpen] = useState(false);
  const [complexity, setComplexity] = useState<ComplexityResult | null>(null);
  const complexityTimerRef = useRef<NodeJS.Timeout | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const processPromptWithMemoryMutation = useProcessPrompt();
  const cancelWorkflowMutation = useCancelWorkflow();
  const complexityMutation = useComplexity();
  const { streamState, isStreaming, startStream, cancelStream } = useSSEStream();
  const {
    notifications,
    unreadCount,
    isOpen: isDropdownOpen,
    toggle: toggleDropdown,
    markAsRead,
    markAllAsRead,
    removeNotification,
    clearAll,
    addNotification,
  } = useNotifications();

  // Handle initial prompt from navigation state
  useEffect(() => {
    if (location.state?.initialPrompt) {
      setCurrentInput(location.state.initialPrompt);
      // Clear the state to prevent re-setting on re-renders
      window.history.replaceState({}, document.title);
    }
  }, [location.state]);

  // Debounced complexity analysis
  useEffect(() => {
    if (complexityTimerRef.current) clearTimeout(complexityTimerRef.current);
    if (!currentInput.trim() || currentInput.trim().length < 15) {
      setComplexity(null);
      return;
    }
    complexityTimerRef.current = setTimeout(() => {
      complexityMutation.mutate(currentInput.trim(), {
        onSuccess: (data) => setComplexity(data),
        onError: () => setComplexity(null),
      });
    }, 800);
    return () => { if (complexityTimerRef.current) clearTimeout(complexityTimerRef.current); };
  }, [currentInput]);

  // Stable refs for keyboard shortcut callbacks
  const messagesRef = useRef(messages);
  messagesRef.current = messages;
  const isAdvancedModeRef = useRef(isAdvancedMode);
  isAdvancedModeRef.current = isAdvancedMode;

  // Keyboard shortcuts — includes global navigation (since PromptProcessor has its own layout)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      const inInput = tag === 'INPUT' || tag === 'TEXTAREA';

      // ? key — show shortcuts dialog (only when not typing in an input)
      if (e.key === '?' && !inInput && !e.ctrlKey && !e.metaKey && !e.altKey) {
        e.preventDefault();
        setIsShortcutsOpen(true);
        return;
      }

      // Alt+1-4 — navigate pages
      if (e.altKey && !e.ctrlKey && !e.metaKey && ['1', '2', '3', '4'].includes(e.key)) {
        e.preventDefault();
        const routes = ['/', '/dashboard', '/templates', '/system-health'];
        navigate(routes[parseInt(e.key) - 1]);
        return;
      }

      // Ctrl+N / Cmd+N — new chat
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'n') {
        e.preventDefault();
        createNewSession();
        return;
      }

      // Ctrl+/ / Cmd+/ — toggle advanced mode
      if ((e.ctrlKey || e.metaKey) && e.key === '/') {
        e.preventDefault();
        setIsAdvancedMode(prev => {
          const next = !prev;
          toast.info(next ? 'Advanced mode on' : 'Advanced mode off');
          return next;
        });
        return;
      }

      // Ctrl+Shift+E / Cmd+Shift+E — download conversation
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'e') {
        e.preventDefault();
        if (messagesRef.current.length > 0) handleDownloadConversation();
        return;
      }
    };
    window.addEventListener('keydown', handler, true);
    return () => window.removeEventListener('keydown', handler, true);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [navigate]);

  // Workflow status polling
  const workflowStatusQuery = useWorkflowStatus(workflowId);

  // Auto-scroll to bottom when new messages are added
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = `${inputRef.current.scrollHeight}px`;
    }
  }, [currentInput]);

  // Initialize user ID, load saved sessions, and show user guide for first-time visitors
  useEffect(() => {
    // Show user guide on first visit
    const hasVisited = localStorage.getItem('cortexa_has_visited');
    if (!hasVisited) {
      setIsUserGuideOpen(true);
      localStorage.setItem('cortexa_has_visited', 'true');
    }

    // Get or create user ID
    const getOrCreateUserId = () => {
      let storedUserId = localStorage.getItem('cortexa_userId');
      if (!storedUserId) {
        storedUserId = sessionStorage.getItem('cortexa_userId');
      }
      if (!storedUserId) {
        const newUserId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        localStorage.setItem('cortexa_userId', newUserId);
        sessionStorage.setItem('cortexa_userId', newUserId);
        return newUserId;
      }
      return storedUserId;
    };

    const userId = getOrCreateUserId();
    setUserId(userId);

    // Load saved sessions
    const loadSessions = () => {
      const storedSessions = localStorage.getItem(`cortexa_sessions_${userId}`);
      if (storedSessions) {
        try {
          const parsedSessions = JSON.parse(storedSessions);
          const sessionsWithDates = parsedSessions.map((session: any) => ({
            ...session,
            lastUpdated: new Date(session.lastUpdated),
            messages: session.messages.map((msg: any) => ({
              ...msg,
              timestamp: new Date(msg.timestamp)
            }))
          }));
          setSessions(sessionsWithDates);
        } catch (error) {
          console.error('Failed to parse saved sessions', error);
          localStorage.removeItem(`cortexa_sessions_${userId}`);
        }
      }
    };

    if (userId) {
      loadSessions();
    }
  }, []);

  // Save sessions to localStorage
  useEffect(() => {
    if (userId && sessions.length >= 0) {
      localStorage.setItem(`cortexa_sessions_${userId}`, JSON.stringify(sessions));
    }
  }, [sessions, userId]);

  // Handle workflow completion
  useEffect(() => {
    if (workflowStatusQuery.data) {
      const { status, result: workflowResult, error } = workflowStatusQuery.data;

      if (status === 'completed' && workflowResult) {
        // Update the loading message with the actual response
        const pollingDomain = workflowResult.output?.domain
          || workflowResult.analysis?.classification?.domain;
        setMessages(prev => prev.map(msg => 
          msg.isLoading && msg.id === `loading_${workflowId}` 
            ? {
                ...msg,
                isLoading: false,
                content: workflowResult.output.optimized_prompt,
                response: workflowResult
              }
            : msg
        ));

        // Update session title with domain context (use ref to avoid stale closure)
        const sid = currentSessionIdRef.current;
        if (sid) {
          const originalPrompt = messages.find(m => m.role === 'user')?.content || '';
          setSessions(prev => prev.map(s =>
            s.id === sid
              ? { ...s, title: generateSessionTitle(originalPrompt, pollingDomain) }
              : s
          ));
        }

        setIsPolling(false);
        setWorkflowId(null);

        // Add notification
        addNotification({
          type: 'success',
          title: `${selectedModel === 'langgraph' ? 'LangGraph' : 'Standard'} Model Complete`,
          message: `Your prompt has been optimized successfully. Quality score: ${workflowResult?.output?.quality_score?.toFixed(2) || 'N/A'}`,
          action: {
            label: 'View Results',
            onClick: () => {
              messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
            }
          }
        });
      } else if (status === 'cancelled') {
        // Remove loading message and reset all states
        setMessages(prev => prev.filter(msg => msg.id !== `loading_${workflowId}`));
        setIsPolling(false);
        setWorkflowId(null);
        toast.info('Processing cancelled.', { id: 'cancel-toast' });
      } else if (status === 'failed') {
        // Update loading message with error
        setMessages(prev => prev.map(msg => 
          msg.isLoading && msg.id === `loading_${workflowId}` 
            ? {
                ...msg,
                isLoading: false,
                error: error || 'Processing failed',
                content: 'Sorry, I encountered an error while processing your request.'
              }
            : msg
        ));
        setIsPolling(false);
        setWorkflowId(null);
        toast.error(error || 'Workflow failed', { id: 'workflow-error' });
      }
    }
  }, [workflowStatusQuery.data, isPolling, workflowId, selectedModel, messages, currentSessionId]);

  // Handle form submission
  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    
    if (!currentInput.trim()) {
      toast.error('Please enter a prompt', { id: 'empty-prompt' });
      return;
    }

    // Session will be auto-created by useEffect if needed

    // Create user message
    const userMessage: ChatMessage = {
      id: `user_${Date.now()}`,
      role: 'user',
      content: currentInput.trim(),
      timestamp: new Date(),
      model: selectedModel
    };

    // Create loading assistant message
    const loadingMessage: ChatMessage = {
      id: `loading_${Date.now()}`,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      model: selectedModel,
      isLoading: true
    };

    // Add messages to chat
    const newMessages = [...messages, userMessage, loadingMessage];
    setMessages(newMessages);
    
    // Clear input
    const inputText = currentInput;
    setCurrentInput('');

    // Build chat history for context
    const chatHistory: Array<{role: 'user' | 'assistant', content: string}> = [];
    messages.forEach(msg => {
      if (!msg.isLoading && !msg.error) {
        chatHistory.push({
          role: msg.role,
          content: msg.content
        });
      }
    });

    // Prepare request
    const request: PromptRequest = {
      prompt: inputText,
      prompt_type: 'auto',
      return_comparison: true,
      use_langgraph: selectedModel === 'langgraph',
      chat_history: chatHistory,
      advanced_mode: isAdvancedMode
    };

    try {
      // Try SSE streaming first for real-time progress, fall back to polling
      try {
        const streamResult = await startStream(request);
        if (streamResult) {
          // SSE streaming completed successfully
          const resultDomain = streamResult.output?.domain
            || streamResult.analysis?.classification?.domain;
          setMessages(prev => prev.map(msg => 
            msg.id === loadingMessage.id 
              ? {
                  ...msg,
                  id: `result_${Date.now()}`,
                  isLoading: false,
                  content: streamResult.output?.optimized_prompt || '',
                  response: streamResult
                }
              : msg
          ));
          // Update session title with domain context (use ref to avoid stale closure)
          const sid = currentSessionIdRef.current;
          if (sid) {
            setSessions(prev => prev.map(s =>
              s.id === sid
                ? { ...s, title: generateSessionTitle(inputText, resultDomain) }
                : s
            ));
          }
          addNotification({
            type: 'success',
            title: `${selectedModel === 'langgraph' ? 'LangGraph' : 'Standard'} Model Complete`,
            message: `Quality score: ${streamResult.output?.quality_score?.toFixed(2) || 'N/A'}`,
            action: { label: 'View Results', onClick: () => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }
          });
          return;
        }
      } catch {
        // SSE failed — fall back to standard polling approach
        console.log('SSE streaming unavailable, falling back to polling');
      }

      let response: PromptResponse | undefined;
      
      // Always use memory-enhanced processing
      response = await processPromptWithMemoryMutation.mutateAsync({
        request: { ...request, user_id: userId },
      });
      
      if (response) {
        // Update loading message ID to match workflow
        setMessages(prev => prev.map(msg => 
          msg.id === loadingMessage.id 
            ? { ...msg, id: `loading_${response.workflow_id}` }
            : msg
        ));
        
        // Start polling for the workflow result
        setWorkflowId(response.workflow_id);
        setIsPolling(true);
      }
    } catch (error: any) {
      console.error('Processing failed:', error);
      // Remove loading message on error
      setMessages(prev => prev.filter(msg => msg.id !== loadingMessage.id));
      toast.error('Failed to process prompt', { id: 'process-error' });
    }
  };

  // Handle cancellation - comprehensive state reset
  const handleCancel = () => {
    // Cancel streaming if active
    if (isStreaming) {
      cancelStream();
    }

    // Cancel workflow API call if in polling mode
    if (workflowId) {
      cancelWorkflowMutation.mutate(workflowId, {
        onError: (error) => {
          console.error('Failed to cancel workflow:', error);
        }
      });
    }

    // Immediately reset all processing states
    setIsPolling(false);
    setWorkflowId(null);
    
    // Remove any loading messages
    setMessages(prev => prev.filter(msg => !msg.isLoading));
    
    // Show cancellation toast (stable ID prevents duplicates with polling handler)
    toast.info('Processing cancelled.', { id: 'cancel-toast' });
  };

  // ── Re-iterate: refine an already-optimized prompt ─────────────────────
  const handleReiterate = async (message: ChatMessage, feedback?: string) => {
    if (!message.response?.output) return;

    const optimizedPrompt = message.content;
    const domain = message.response.output.domain || 'general';

    // Find the user message that preceded this assistant message
    const msgIndex = messages.findIndex(m => m.id === message.id);
    const userMsg = messages.slice(0, msgIndex).reverse().find(m => m.role === 'user');
    const originalPrompt = userMsg?.content || message.response.input?.original_prompt || optimizedPrompt;

    setReiteratingId(message.id);

    // Add a loading assistant message for the refined version
    const loadingMsg: ChatMessage = {
      id: `reiterate_loading_${Date.now()}`,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      model: selectedModel,
      isLoading: true,
    };

    setMessages(prev => [...prev, loadingMsg]);
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });

    try {
      const abortCtrl = new AbortController();
      const resp = await fetch('/api/process-prompt/reiterate/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          original_prompt: originalPrompt,
          optimized_prompt: optimizedPrompt,
          domain,
          use_langgraph: selectedModel === 'langgraph',
          user_feedback: feedback || null,
        }),
        signal: abortCtrl.signal,
      });

      if (!resp.ok) throw new Error(await resp.text() || 'Re-iterate request failed');

      const reader = resp.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let buffer = '';
      let finalResult: PromptResponse | null = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let currentEvent = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith('data: ') && currentEvent) {
            try {
              const data = JSON.parse(line.slice(6));
              if (currentEvent === 'completed' && data.result) {
                finalResult = data.result;
              }
            } catch { /* skip */ }
            currentEvent = '';
          }
        }
      }

      if (finalResult) {
        setMessages(prev => prev.map(msg =>
          msg.id === loadingMsg.id
            ? {
                ...msg,
                id: `reiterate_${Date.now()}`,
                isLoading: false,
                content: finalResult!.output?.optimized_prompt || '',
                response: finalResult!,
              }
            : msg
        ));
        toast.success('Prompt refined successfully!', {
          description: `New quality score: ${(finalResult.output?.quality_score * 100).toFixed(0)}%`,
        });
      } else {
        setMessages(prev => prev.filter(msg => msg.id !== loadingMsg.id));
        toast.error('Re-iterate completed but no result was returned.');
      }
    } catch (err: any) {
      setMessages(prev => prev.filter(msg => msg.id !== loadingMsg.id));
      toast.error(`Re-iterate failed: ${err.message}`);
    } finally {
      setReiteratingId(null);
    }
  };

  // Create new session
  const createNewSession = () => {
    setCurrentSessionId(null);
    currentSessionIdRef.current = null;
    setMessages([]);
  };

  // Load session
  const loadSession = (sessionId: string) => {
    const session = sessions.find(s => s.id === sessionId);
    if (session) {
      setCurrentSessionId(sessionId);
      currentSessionIdRef.current = sessionId;
      setMessages(session.messages);
      setSelectedModel(session.model);
    }
  };

  // Delete session
  const deleteSession = (sessionId: string) => {
    setSessions(prev => prev.filter(s => s.id !== sessionId));
    if (currentSessionId === sessionId) {
      setCurrentSessionId(null);
      currentSessionIdRef.current = null;
      setMessages([]);
    }
    setIsDeleteDialogOpen(false);
    setSessionToDelete(null);
  };

  // Update session title
  const updateSessionTitle = (sessionId: string, newTitle: string) => {
    setSessions(prev => prev.map(session =>
      session.id === sessionId
        ? { ...session, title: newTitle, lastUpdated: new Date() }
        : session
    ));
    setEditingSessionId(null);
    setEditingTitle('');
  };

  // Auto-save current session
  useEffect(() => {
    if (currentSessionId && messages.length > 0) {
      setSessions(prev => prev.map(session =>
        session.id === currentSessionId
          ? { ...session, messages, lastUpdated: new Date() }
          : session
      ));
    }
  }, [messages, currentSessionId]);

  // Auto-create session when first message is sent
  useEffect(() => {
    if (messages.length > 0 && !currentSessionId) {
      const firstUserMessage = messages.find(msg => msg.role === 'user');
      if (firstUserMessage) {
        const newSession: ChatSession = {
          id: `session_${Date.now()}`,
          title: generateSessionTitle(firstUserMessage.content),
          messages: messages,
          lastUpdated: new Date(),
          model: selectedModel
        };
        setSessions(prev => [newSession, ...prev]);
        setCurrentSessionId(newSession.id);
        currentSessionIdRef.current = newSession.id;
      }
    }
  }, [messages, currentSessionId, selectedModel]);

  // Download conversation
  const handleDownloadConversation = () => {
    const conversation = {
      sessionId: currentSessionId,
      model: selectedModel,
      messages: messages.map(msg => ({
        role: msg.role,
        content: msg.content,
        timestamp: msg.timestamp,
        model: msg.model
      }))
    };
    const blob = new Blob([JSON.stringify(conversation, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `conversation_${currentSessionId || 'new'}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    toast.success('Conversation downloaded');
  };

  const isLoading = processPromptWithMemoryMutation.isPending || isPolling || isStreaming;
  const canCancel = isStreaming || isPolling;

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Sidebar with chat sessions */}
      <Sidebar collapsed={!showSidebar} onToggle={() => setShowSidebar(!showSidebar)}>
        {/* New chat button */}
        <div className="px-3 pt-2 pb-1">
          <Button
            onClick={createNewSession}
            variant="outline"
            className="w-full justify-start gap-2 border-dashed"
          >
            <Plus className="h-4 w-4" />
            {showSidebar && 'New chat'}
          </Button>
        </div>

        {/* Session list */}
        {showSidebar && (
          <div className="flex-1 overflow-y-auto px-3 py-2 space-y-0.5">
            {sessions.map(session => (
              <div
                key={session.id}
                className={`group relative px-3 py-2.5 rounded-lg cursor-pointer transition-colors ${
                  currentSessionId === session.id
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-accent/60 hover:text-foreground'
                }`}
                onClick={() => loadSession(session.id)}
              >
                {editingSessionId === session.id ? (
                  <input
                    type="text"
                    value={editingTitle}
                    onChange={(e) => setEditingTitle(e.target.value)}
                    onBlur={() => updateSessionTitle(session.id, editingTitle)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') updateSessionTitle(session.id, editingTitle);
                      else if (e.key === 'Escape') { setEditingSessionId(null); setEditingTitle(''); }
                    }}
                    className="w-full px-2 py-1 text-sm bg-background border rounded-md"
                    autoFocus
                    onClick={(e) => e.stopPropagation()}
                  />
                ) : (
                  <>
                    <p className="text-sm truncate pr-12">{session.title}</p>
                    <p className="text-xs opacity-60 mt-0.5">{session.messages.length} messages</p>
                    <div className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity flex gap-0.5">
                      <Button
                        variant="ghost" size="icon"
                        className="h-6 w-6"
                        onClick={(e) => { e.stopPropagation(); setEditingSessionId(session.id); setEditingTitle(session.title); }}
                      ><Edit className="h-3 w-3" /></Button>
                      <Button
                        variant="ghost" size="icon"
                        className="h-6 w-6 hover:text-destructive"
                        onClick={(e) => { e.stopPropagation(); setSessionToDelete(session.id); setIsDeleteDialogOpen(true); }}
                      ><Trash2 className="h-3 w-3" /></Button>
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        )}
      </Sidebar>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="sticky top-0 z-30 flex items-center justify-end h-14 px-6 border-b border-border/60 bg-background/80 backdrop-blur-md">
          <div className="flex items-center gap-1.5">
            <div className="relative">
              <Button variant="ghost" size="icon" onClick={toggleDropdown} className="relative h-9 w-9">
                <Bell className="h-4 w-4" />
                {unreadCount > 0 && (
                  <span className="absolute top-1.5 right-1.5 h-2 w-2 bg-destructive rounded-full ring-2 ring-background" />
                )}
              </Button>
              <NotificationsDropdown
                notifications={notifications}
                isOpen={isDropdownOpen}
                unreadCount={unreadCount}
                onClose={toggleDropdown}
                onMarkAsRead={markAsRead}
                onMarkAllAsRead={markAllAsRead}
                onRemove={removeNotification}
                onClearAll={clearAll}
              />
            </div>
            {messages.length > 0 && (
              <Button variant="ghost" size="icon" onClick={handleDownloadConversation} className="h-9 w-9">
                <Download className="h-4 w-4" />
              </Button>
            )}
            <Button variant="ghost" size="icon" onClick={() => setIsUserGuideOpen(true)} className="h-9 w-9">
              <HelpCircle className="h-4 w-4" />
            </Button>
          </div>
        </header>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto px-6 py-6">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="max-w-4xl mx-auto p-8 space-y-8">
                {/* Hero */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-10 items-center">
                  <div className="flex justify-center">
                    <img src="/Cortexa Logo.png" alt="Cortexa Logo" className="w-40 h-40 drop-shadow-lg" />
                  </div>
                  <div className="text-center md:text-left space-y-4">
                    <h2 className="text-3xl font-bold tracking-tight text-foreground">Welcome to Cortexa</h2>
                    <p className="text-muted-foreground leading-relaxed">
                      Unlock the power of AI with our advanced multi-agent system. Craft, refine, and optimize your prompts for exceptional results.
                    </p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
                      <button
                        className={`p-4 border rounded-xl text-left transition-all hover:shadow-md ${
                          selectedModel === 'standard' ? 'border-primary/50 bg-primary/5 shadow-sm' : 'hover:border-border/80'
                        }`}
                        onClick={() => setSelectedModel('standard')}
                      >
                        <div className="flex items-center gap-2.5 mb-1.5">
                          <Sparkles className="h-5 w-5 text-primary shrink-0" />
                          <h3 className="font-semibold text-sm">Standard Model</h3>
                        </div>
                        <p className="text-xs text-muted-foreground leading-relaxed">Memory-enhanced optimization for quick, reliable results.</p>
                      </button>
                      <button
                        className={`p-4 border rounded-xl text-left transition-all hover:shadow-md ${
                          selectedModel === 'langgraph' ? 'border-primary/50 bg-primary/5 shadow-sm' : 'hover:border-border/80'
                        }`}
                        onClick={() => setSelectedModel('langgraph')}
                      >
                        <div className="flex items-center gap-2.5 mb-1.5">
                          <Bot className="h-5 w-5 text-primary shrink-0" />
                          <h3 className="font-semibold text-sm">LangGraph Model</h3>
                        </div>
                        <p className="text-xs text-muted-foreground leading-relaxed">Complex multi-agent workflow for in-depth analysis.</p>
                      </button>
                    </div>
                  </div>
                </div>

                {/* Example prompts */}
                <div className="space-y-3">
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider text-center">Try an example</p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                    {EXAMPLE_PROMPTS.map((ex, i) => (
                      <button
                        key={i}
                        className="group p-3.5 border rounded-xl text-left hover:border-primary/40 hover:bg-primary/5 transition-all"
                        onClick={() => setCurrentInput(ex.text)}
                      >
                        <Badge variant="outline" className="text-[10px] h-4 px-1.5 mb-1.5">{ex.domain}</Badge>
                        <p className="text-sm text-muted-foreground group-hover:text-foreground transition-colors line-clamp-1">{ex.text}</p>
                      </button>
                    ))}
                  </div>
                  <p className="text-center text-[11px] text-muted-foreground/60 flex items-center justify-center gap-1.5">
                    <Keyboard className="h-3 w-3" /> Press <kbd className="px-1 py-0.5 rounded bg-muted text-[10px] font-mono">?</kbd> for shortcuts
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-5 max-w-4xl mx-auto">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}
                >
                  <div className={`flex gap-3 ${message.role === 'user' ? 'max-w-[80%] flex-row-reverse' : 'max-w-[90%]'}`}>
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
                      message.role === 'user' ? 'bg-primary' : 'bg-muted'
                    }`}>
                      {message.role === 'user' ? (
                        <User className="h-4 w-4 text-primary-foreground" />
                      ) : (
                        <Bot className="h-4 w-4 text-foreground" />
                      )}
                    </div>
                    
                    <div className="flex-1 space-y-1.5 min-w-0">
                      <div className={`rounded-2xl px-4 py-3 overflow-hidden ${
                        message.role === 'user' 
                          ? 'bg-primary text-primary-foreground rounded-tr-md' 
                          : message.error 
                            ? 'bg-destructive/10 border border-destructive/20 rounded-tl-md'
                            : 'bg-card border border-border/60 shadow-sm rounded-tl-md'
                      }`}>
                        {message.isLoading ? (
                          streamState && isStreaming ? (
                            <StreamingProgress state={streamState} />
                          ) : (
                            <div className="flex items-center space-x-2">
                              <LoadingSpinner size="sm" />
                              <span className="text-sm">
                                Processing with {selectedModel === 'langgraph' ? 'LangGraph' : 'Standard'} model...
                              </span>
                            </div>
                          )
                        ) : message.error ? (
                          <div className="text-sm text-destructive">{message.content}</div>
                        ) : message.role === 'assistant' && message.response ? (
                          <div className="space-y-3">
                            <SyntaxHighlighter
                              language="markdown"
                              style={vscDarkPlus}
                              className="code-block-wrapper"
                              customStyle={{
                                margin: '0',
                                fontSize: '0.875rem',
                                maxWidth: '100%',
                                overflowWrap: 'break-word',
                                whiteSpace: 'pre-wrap',
                                borderRadius: '0.5rem'
                              }}
                              wrapLines={true}
                            >
                              {message.content}
                            </SyntaxHighlighter>
                            
                            {message.response.output && (
                              <div className="pt-3 mt-3 border-t border-border/40 space-y-3">
                                {/* Action buttons */}
                                <div className="flex items-center gap-2 flex-wrap">
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    className="gap-1.5 h-7 text-xs rounded-lg"
                                    onClick={() => {
                                      navigator.clipboard.writeText(message.content);
                                      toast.success('Optimized prompt copied!');
                                    }}
                                  >
                                    <ClipboardCopy className="h-3 w-3" />
                                    Copy Optimized Prompt
                                  </Button>
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    className="gap-1.5 h-7 text-xs rounded-lg border-primary/40 text-primary hover:bg-primary/10"
                                    disabled={reiteratingId !== null || isLoading}
                                    onClick={() => {
                                      setReiterateFeedback('');
                                      setReiterateDialogMsg(message);
                                    }}
                                  >
                                    {reiteratingId === message.id ? (
                                      <><LoadingSpinner size="sm" /> Refining…</>
                                    ) : (
                                      <><RefreshCw className="h-3 w-3" /> Refine Further</>
                                    )}
                                  </Button>
                                </div>

                                {/* Quality score ring + metadata */}
                                <div className="flex items-center gap-5 flex-wrap">
                                  {message.response.output.quality_score != null && (
                                    <div className="flex items-center gap-2">
                                      <div className={`h-8 w-8 rounded-full flex items-center justify-center text-xs font-bold ring-2 ${
                                        message.response.output.quality_score >= 0.8
                                          ? 'bg-green-100 text-green-700 ring-green-300 dark:bg-green-900/30 dark:text-green-400 dark:ring-green-600'
                                          : message.response.output.quality_score >= 0.6
                                            ? 'bg-yellow-100 text-yellow-700 ring-yellow-300 dark:bg-yellow-900/30 dark:text-yellow-400 dark:ring-yellow-600'
                                            : 'bg-red-100 text-red-700 ring-red-300 dark:bg-red-900/30 dark:text-red-400 dark:ring-red-600'
                                      }`}>
                                        {(message.response.output.quality_score * 100).toFixed(0)}
                                      </div>
                                      <span className="text-xs text-muted-foreground">Quality</span>
                                    </div>
                                  )}
                                  <div className="flex items-center gap-3 text-xs text-muted-foreground flex-wrap">
                                    {message.response.output.domain && (
                                      <Badge variant="outline" className="text-[10px] h-5 capitalize">{message.response.output.domain}</Badge>
                                    )}
                                    <span className="flex items-center gap-1">Iterations: <strong>{message.response.output.iterations_used}</strong></span>
                                    {message.response.processing_time_seconds != null && (
                                      <span className="flex items-center gap-1">
                                        <Clock className="h-3 w-3" />
                                        {formatDuration(message.response.processing_time_seconds)}
                                      </span>
                                    )}
                                    {(message.response as any).metadata?.cache_hit && (
                                      <Badge className="text-[10px] h-4 bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400">Cached</Badge>
                                    )}
                                  </div>
                                </div>
                                <EvaluationBreakdown response={message.response} />
                                <DiffView comparison={message.response.comparison} />
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="text-sm whitespace-pre-wrap">{message.content}</div>
                        )}
                      </div>
                      
                      {!message.isLoading && !message.error && (
                        <div className="flex items-center gap-2 text-[11px] text-muted-foreground/70 px-1 pt-0.5">
                          <span>{message.timestamp.toLocaleTimeString()}</span>
                          {message.model && (
                            <Badge variant="outline" className="text-[10px] h-4 px-1.5">
                              {message.model === 'langgraph' ? 'LangGraph' : 'Standard'}
                            </Badge>
                          )}
                          {message.role === 'assistant' && (
                            <ExportMenu message={message} />
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t border-border/60 px-6 py-4 bg-background/80 backdrop-blur-md">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
            <div className="relative rounded-2xl border border-border/80 bg-card shadow-sm focus-within:ring-2 focus-within:ring-ring/40 focus-within:border-primary/40 transition-all">
              <Textarea
                ref={inputRef}
                value={currentInput}
                onChange={(e) => setCurrentInput(e.target.value)}
                placeholder="Message Cortexa..."
                className="resize-none w-full border-0 bg-transparent pt-3 pb-14 pl-4 pr-14 min-h-[56px] max-h-[300px] focus-visible:ring-0 overflow-y-auto rounded-2xl"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit();
                  }
                }}
                disabled={isLoading}
                rows={1}
              />
              <div className="absolute bottom-3 left-4 flex items-center gap-3">
                <div className="relative">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => setIsModelDropdownOpen(!isModelDropdownOpen)}
                    className="flex items-center gap-1 text-xs font-medium h-7 px-2.5 rounded-lg"
                  >
                    <span>{selectedModel === 'standard' ? 'Standard' : 'LangGraph'}</span>
                    <ChevronDown className="h-3.5 w-3.5" />
                  </Button>
                  {isModelDropdownOpen && (
                    <div className="absolute bottom-full mb-2 w-64 bg-popover border rounded-xl shadow-lg z-10 overflow-hidden">
                      <div
                        className="flex items-center justify-between p-3 hover:bg-accent cursor-pointer transition-colors"
                        onClick={() => { setSelectedModel('standard'); setIsModelDropdownOpen(false); }}
                      >
                        <div>
                          <h4 className="text-sm font-semibold">Standard</h4>
                          <p className="text-xs text-muted-foreground">Memory-enhanced optimization.</p>
                        </div>
                        {selectedModel === 'standard' && <Check className="h-4 w-4 text-primary shrink-0" />}
                      </div>
                      <div
                        className="flex items-center justify-between p-3 hover:bg-accent cursor-pointer transition-colors border-t border-border/40"
                        onClick={() => { setSelectedModel('langgraph'); setIsModelDropdownOpen(false); }}
                      >
                        <div>
                          <h4 className="text-sm font-semibold">LangGraph</h4>
                          <p className="text-xs text-muted-foreground">Multi-agent workflow.</p>
                        </div>
                        {selectedModel === 'langgraph' && <Check className="h-4 w-4 text-primary shrink-0" />}
                      </div>
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-1.5">
                  <Switch
                    id="advanced-mode"
                    checked={isAdvancedMode}
                    onCheckedChange={setIsAdvancedMode}
                  />
                  <Label htmlFor="advanced-mode" className="text-xs">Advanced</Label>
                </div>
              </div>
              <div className="absolute bottom-3 right-3">
                {canCancel ? (
                  <Button
                    type="button"
                    size="sm"
                    variant="destructive"
                    onClick={handleCancel}
                    className="rounded-xl h-8"
                  >
                    Cancel
                  </Button>
                ) : (
                  <Button
                    type="submit"
                    size="icon"
                    disabled={isLoading || !currentInput.trim()}
                    className="rounded-xl h-8 w-8"
                  >
                    {isLoading ? <LoadingSpinner size="sm" /> : <Send className="h-4 w-4" />}
                  </Button>
                )}
              </div>
            </div>
            <div className="flex items-center justify-between mt-1.5 px-1 text-[11px] text-muted-foreground/70">
              <div className="flex items-center gap-2">
                <span>Enter to send &middot; Shift+Enter for new line</span>
                <ComplexityBadge complexity={complexity} />
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  className="hover:text-foreground transition-colors cursor-pointer"
                  onClick={() => setIsShortcutsOpen(true)}
                >
                  <Keyboard className="h-3 w-3 inline mr-0.5" />Shortcuts
                </button>
                <span>&middot;</span>
                <span>{currentInput.length} chars</span>
              </div>
            </div>
          </form>
        </div>
      </div>

      {/* Delete Confirmation Dialog */}
      <ConfirmationDialog
        open={isDeleteDialogOpen}
        onOpenChange={(open) => {
          setIsDeleteDialogOpen(open);
          if (!open) setSessionToDelete(null);
        }}
        onConfirm={() => sessionToDelete && deleteSession(sessionToDelete)}
        title="Delete Conversation"
        description="Are you sure you want to delete this conversation? This action cannot be undone."
        confirmText="Delete"
        cancelText="Cancel"
      />

      {/* User Guide Dialog */}
      <Dialog open={isUserGuideOpen} onOpenChange={setIsUserGuideOpen}>
        <DialogContent className="sm:max-w-[560px] rounded-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <HelpCircle className="h-5 w-5 text-primary" />
              User Guide
            </DialogTitle>
            <DialogDescription>
              Welcome to Cortexa! Here's a quick guide to get you started.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-5 py-4 text-sm">
            <div className="space-y-1.5">
              <h4 className="font-semibold text-foreground">What is Cortexa?</h4>
              <p className="text-muted-foreground leading-relaxed">
                Cortexa is an advanced multi-agent system designed to help you craft, refine, and optimize your prompts for exceptional results from AI models.
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="font-semibold text-foreground">Choosing a Model</h4>
              <ul className="space-y-1.5 text-muted-foreground">
                <li className="flex gap-2"><span className="text-primary font-bold">&bull;</span><span><b className="text-foreground">Standard</b> &mdash; Quick, reliable prompt optimization enhanced with memory.</span></li>
                <li className="flex gap-2"><span className="text-primary font-bold">&bull;</span><span><b className="text-foreground">LangGraph</b> &mdash; Multi-agent workflow for in-depth analysis.</span></li>
              </ul>
            </div>
            <div className="space-y-1.5">
              <h4 className="font-semibold text-foreground">Advanced Mode</h4>
              <p className="text-muted-foreground leading-relaxed">
                Enable Advanced mode for a conversational prompt-engineering session. The AI will ask clarifying questions to deeply understand your needs before generating an improved prompt.
              </p>
            </div>
            <div className="space-y-1.5">
              <h4 className="font-semibold text-foreground">Refine Further</h4>
              <p className="text-muted-foreground leading-relaxed">
                After receiving an optimized prompt, click <b className="text-foreground">Refine Further</b> to re-iterate. You can add optional feedback to guide the refinement, or let the system automatically fix detected weaknesses.
              </p>
            </div>
          </div>
          <DialogFooter>
            <Button onClick={() => setIsUserGuideOpen(false)} className="rounded-xl">Got it</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Re-iterate Feedback Dialog */}
      <Dialog open={reiterateDialogMsg !== null} onOpenChange={(open) => { if (!open) setReiterateDialogMsg(null); }}>
        <DialogContent className="sm:max-w-[480px] rounded-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <RefreshCw className="h-5 w-5 text-primary" />
              Refine Prompt Further
            </DialogTitle>
            <DialogDescription>
              The system will automatically target detected weaknesses. You can optionally add guidance below to steer the refinement.
            </DialogDescription>
          </DialogHeader>
          <div className="py-3">
            <Label htmlFor="reiterate-feedback" className="text-sm font-medium mb-2 block">
              Optional feedback
            </Label>
            <Textarea
              id="reiterate-feedback"
              placeholder="e.g. Make it more concise, add error handling examples, focus on security…"
              value={reiterateFeedback}
              onChange={(e) => setReiterateFeedback(e.target.value)}
              className="rounded-xl resize-none"
              rows={3}
            />
          </div>
          <DialogFooter className="gap-2 sm:gap-0">
            <Button variant="outline" className="rounded-xl" onClick={() => setReiterateDialogMsg(null)}>
              Cancel
            </Button>
            <Button
              className="rounded-xl gap-1.5"
              onClick={() => {
                if (reiterateDialogMsg) {
                  handleReiterate(reiterateDialogMsg, reiterateFeedback.trim() || undefined);
                  setReiterateDialogMsg(null);
                }
              }}
            >
              <RefreshCw className="h-3.5 w-3.5" />
              Refine
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Keyboard Shortcuts Dialog */}
      <KeyboardShortcutsDialog open={isShortcutsOpen} onOpenChange={setIsShortcutsOpen} />
    </div>
  );
}
