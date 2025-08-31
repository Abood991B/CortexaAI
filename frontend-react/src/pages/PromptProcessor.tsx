import React, { useState } from 'react';
import { Brain, Zap, Play, Copy, Download } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { LoadingSpinner } from '@/components/ui/loading';
import { useProcessPrompt, useProcessPromptWithMemory } from '@/hooks/useApi';
import { formatDuration } from '@/lib/utils';
import { toast } from 'sonner';
import type { PromptRequest, PromptResponse } from '@/types/api';

export function PromptProcessor() {
  const [prompt, setPrompt] = useState('');
  const [promptType, setPromptType] = useState<'auto' | 'raw' | 'structured'>('auto');
  const [returnComparison, setReturnComparison] = useState(true);
  const [useLangGraph, setUseLangGraph] = useState(false);
  const [useMemory, setUseMemory] = useState(false);
  const [usePlanning, setUsePlanning] = useState(false);
  const [userId, setUserId] = useState('user_001');
  const [result, setResult] = useState<PromptResponse | null>(null);

  const processPromptMutation = useProcessPrompt();
  const processPromptWithMemoryMutation = useProcessPromptWithMemory();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!prompt.trim()) {
      toast.error('Please enter a prompt');
      return;
    }

    // Reset previous result
    setResult(null);

    const request: PromptRequest = {
      prompt: prompt.trim(),
      prompt_type: promptType,
      return_comparison: returnComparison,
      use_langgraph: useLangGraph,
    };

    try {
      let response: PromptResponse;
      
      if (useMemory) {
        if (!userId.trim()) {
          toast.error('User ID is required when using memory');
          return;
        }
        response = await processPromptWithMemoryMutation.mutateAsync({
          ...request,
          user_id: userId,
        });
      } else if (usePlanning) {
        toast.error('Planning feature coming soon');
        return;
      } else {
        response = await processPromptMutation.mutateAsync(request);
      }
      
      setResult(response);
    } catch (error: any) {
      console.error('Processing failed:', error);
      // Error is already handled by the mutation's onError
    }
  };

  const isLoading = processPromptMutation.isPending || 
                   processPromptWithMemoryMutation.isPending;
  
  // Reset form after successful submission
  React.useEffect(() => {
    if (processPromptMutation.isSuccess || processPromptWithMemoryMutation.isSuccess) {
      // Scroll to results
      window.scrollTo({
        top: document.body.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [processPromptMutation.isSuccess, processPromptWithMemoryMutation.isSuccess]);

  const handleCopyResult = () => {
    if (result?.output.optimized_prompt) {
      navigator.clipboard.writeText(result.output.optimized_prompt);
      toast.success('Optimized prompt copied to clipboard');
    }
  };

  const handleDownloadResult = () => {
    if (result) {
      const content = JSON.stringify(result, null, 2);
      const blob = new Blob([content], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `prompt_result_${result.workflow_id}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      toast.success('Result downloaded');
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold flex items-center">
          <Brain className="mr-3 h-8 w-8 text-primary" />
          Prompt Processor
        </h1>
        <p className="text-muted-foreground">
          Optimize your prompts using our multi-agent system
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Section */}
        <Card>
          <CardHeader>
            <CardTitle>Input Prompt</CardTitle>
            <CardDescription>
              Enter your prompt and configure processing options
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-2 block">
                  Prompt Content
                </label>
                <Textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Enter your prompt here...&#10;&#10;Examples:&#10;- Write a function to sort a list&#10;- Create a data analysis report&#10;- Draft a business strategy document"
                  className="min-h-[200px]"
                  required
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">
                    Prompt Type
                  </label>
                  <select
                    value={promptType}
                    onChange={(e) => setPromptType(e.target.value as any)}
                    className="w-full p-2 border border-input rounded-md bg-background"
                  >
                    <option value="auto">Auto-detect</option>
                    <option value="raw">Raw Prompt</option>
                    <option value="structured">Structured Prompt</option>
                  </select>
                </div>

                {(useMemory || usePlanning) && (
                  <div>
                    <label className="text-sm font-medium mb-2 block">
                      User ID
                    </label>
                    <input
                      type="text"
                      value={userId}
                      onChange={(e) => setUserId(e.target.value)}
                      className="w-full p-2 border border-input rounded-md bg-background"
                      placeholder="user_001"
                    />
                  </div>
                )}
              </div>

              {/* Processing Options */}
              <div className="space-y-3">
                <h4 className="text-sm font-medium">Processing Options</h4>
                
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="returnComparison"
                    checked={returnComparison}
                    onChange={(e) => setReturnComparison(e.target.checked)}
                    className="rounded"
                  />
                  <label htmlFor="returnComparison" className="text-sm">
                    Return before/after comparison
                  </label>
                </div>

                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="useLangGraph"
                    checked={useLangGraph}
                    onChange={(e) => setUseLangGraph(e.target.checked)}
                    className="rounded"
                  />
                  <label htmlFor="useLangGraph" className="text-sm">
                    Use LangGraph workflow
                  </label>
                </div>

                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="useMemory"
                    checked={useMemory}
                    onChange={(e) => {
                      setUseMemory(e.target.checked);
                      if (e.target.checked) setUsePlanning(false);
                    }}
                    className="rounded"
                  />
                  <label htmlFor="useMemory" className="text-sm">
                    Use memory-enhanced processing
                  </label>
                </div>

                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="usePlanning"
                    checked={usePlanning}
                    onChange={(e) => {
                      setUsePlanning(e.target.checked);
                      if (e.target.checked) setUseMemory(false);
                    }}
                    className="rounded"
                  />
                  <label htmlFor="usePlanning" className="text-sm">
                    Use planning-enhanced processing
                  </label>
                </div>
              </div>

              <Button 
                type="submit" 
                className="w-full" 
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <LoadingSpinner size="sm" className="mr-2" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Optimize Prompt
                  </>
                )}
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* Results Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              Results
              {result && (
                <div className="flex space-x-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleCopyResult}
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleDownloadResult}
                  >
                    <Download className="h-4 w-4" />
                  </Button>
                </div>
              )}
            </CardTitle>
            <CardDescription>
              Optimized prompt and analysis results
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="flex flex-col items-center justify-center py-12 space-y-4">
                <LoadingSpinner size="lg" />
                <div className="text-center">
                  <h3 className="font-medium">Processing your prompt...</h3>
                  <p className="text-sm text-muted-foreground">
                    Our AI agents are analyzing, classifying, improving, and evaluating your prompt.
                  </p>
                </div>
              </div>
            ) : result ? (
              <div className="space-y-6">
                {/* Workflow Info */}
                <div className="flex items-center justify-between p-3 bg-muted rounded-lg">
                  <div>
                    <p className="font-medium">Workflow ID: {result.workflow_id}</p>
                    <p className="text-sm text-muted-foreground">
                      Domain: {result.output.domain} | 
                      Quality: {result.output.quality_score.toFixed(2)} | 
                      Iterations: {result.output.iterations_used}
                    </p>
                  </div>
                  <div className="text-right">
                    <Badge variant="success">{result.status}</Badge>
                    {result.processing_time_seconds && (
                      <p className="text-sm text-muted-foreground">
                        {formatDuration(result.processing_time_seconds)}
                      </p>
                    )}
                  </div>
                </div>

                {/* Comparison or Optimized Prompt */}
                {result.comparison ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-medium mb-2">📝 Original Prompt</h4>
                      <div className="p-3 bg-muted rounded-lg text-sm font-mono whitespace-pre-wrap">
                        {result.comparison.side_by_side.original}
                      </div>
                    </div>
                    <div>
                      <h4 className="font-medium mb-2">✨ Optimized Prompt</h4>
                      <div className="p-3 bg-primary/10 rounded-lg text-sm font-mono whitespace-pre-wrap">
                        {result.comparison.side_by_side.optimized}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div>
                    <h4 className="font-medium mb-2">✨ Optimized Prompt</h4>
                    <div className="p-3 bg-primary/10 rounded-lg text-sm font-mono whitespace-pre-wrap">
                      {result.output.optimized_prompt}
                    </div>
                  </div>
                )}

                {/* Improvement Ratio */}
                {result.comparison && (
                  <div className="text-center">
                    <p className="text-sm text-muted-foreground">Improvement Ratio</p>
                    <p className="text-2xl font-bold text-green-600">
                      {(result.comparison.improvement_ratio * 100).toFixed(1)}%
                    </p>
                  </div>
                )}

                {/* Analysis */}
                {result.analysis && (
                  <div className="space-y-3">
                    <h4 className="font-medium">🔍 Analysis</h4>
                    <div className="p-3 bg-muted rounded-lg">
                      <p className="text-sm">
                        <strong>Classification:</strong> {result.analysis.classification.reasoning}
                      </p>
                      <p className="text-sm mt-2">
                        <strong>Key Topics:</strong> {result.analysis.classification.key_topics.join(', ')}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                <Zap className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Enter a prompt and click "Optimize Prompt" to see results</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
