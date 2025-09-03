import React, { useState, useEffect } from 'react';
import { Brain, Zap, Play, Copy, Download, ChevronDown, ChevronUp, History, ArrowLeft, MessageSquare, Trash2, RefreshCw, Edit, Check, Settings } from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { LoadingSpinner } from '@/components/ui/loading';
import { ConfirmationDialog } from '@/components/ui/confirmation-dialog';
import { 
  useProcessPrompt, 
  useProcessPromptWithMemory,
  useCancelWorkflow,
  useWorkflowStatus,
  clearCacheForRequest,
  getCacheKey,
  setCachedResponse
} from '@/hooks/useApi';
import { formatDuration } from '@/lib/utils';
import { toast } from 'sonner';
import { useNotificationSender } from '@/hooks/useNotifications';
import type { PromptRequest, PromptResponse } from '@/types/api';

export function PromptProcessor() {
  const [prompt, setPrompt] = useState('');
  const [promptType, setPromptType] = useState<'auto' | 'raw' | 'structured'>('auto');
  const [returnComparison, setReturnComparison] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [processingMethod, setProcessingMethod] = useState<'standard' | 'memory' | 'langgraph'>('standard');
  const [userId, setUserId] = useState('');
  const [chatHistory, setChatHistory] = useState<Array<{id: string, prompt: string, response: PromptResponse, timestamp: Date}>>([]);
  const [savedChats, setSavedChats] = useState<Array<{id: string, title: string, messages: Array<{id: string, prompt: string, response: PromptResponse, timestamp: Date}>, lastUpdated: Date}>>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [showChatList, setShowChatList] = useState(false);
    const [currentChatPrompt, setCurrentChatPrompt] = useState('');
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
    const [editingTitle, setEditingTitle] = useState('');
  const [isComparisonOpen, setIsComparisonOpen] = useState(true);
  const [isAnalysisOpen, setIsAnalysisOpen] = useState(true);
    const [result, setResult] = useState<PromptResponse | null>(null);
  const [workflowId, setWorkflowId] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [gracePeriodCountdown, setGracePeriodCountdown] = useState(3);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [chatToDelete, setChatToDelete] = useState<string | null>(null);

  const processPromptMutation = useProcessPrompt();
  const processPromptWithMemoryMutation = useProcessPromptWithMemory();
  const cancelWorkflowMutation = useCancelWorkflow();
  const { addNotification } = useNotificationSender();
  
  // Workflow status polling
  const workflowStatusQuery = useWorkflowStatus(workflowId);
  
  // Handle workflow completion and countdown timer
  useEffect(() => {
    let timer: NodeJS.Timeout;

    if (isPolling && workflowId && workflowStatusQuery.data?.grace_period_active) {
      setGracePeriodCountdown(3);
      timer = setInterval(() => {
        setGracePeriodCountdown(prev => {
          if (prev <= 1) {
            clearInterval(timer);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    } else {
      setGracePeriodCountdown(0);
    }

    if (workflowStatusQuery.data) {
      const { status, result: workflowResult, error, grace_period_active } = workflowStatusQuery.data;

      const handleTerminalState = (toastMessage: string, toastType: 'success' | 'info' | 'error') => {
        setIsPolling(false);
        setWorkflowId(null);
        if (toastType === 'success') toast.success(toastMessage);
        else if (toastType === 'info') toast.info(toastMessage);
        else toast.error(toastMessage);
      };

      if (status === 'completed') {
        setResult(workflowResult);
        
        // Cache the successful response
        const currentRequest: PromptRequest = {
          prompt: processingMethod === 'memory' ? currentChatPrompt : prompt,
          prompt_type: promptType,
          return_comparison: returnComparison,
          use_langgraph: processingMethod === 'langgraph'
        };
        
        // Add user_id for memory requests
        if (processingMethod === 'memory' && userId) {
          (currentRequest as any).user_id = userId;
        }
        
        // Cache the successful response
        const cacheKey = getCacheKey(currentRequest);
        if (workflowResult) {
          setCachedResponse(cacheKey, workflowResult);
        }
        
        // Add completion notification
        addNotification({
          type: 'success',
          title: `${processingMethod === 'langgraph' ? 'LangGraph Workflow' : processingMethod === 'memory' ? 'Memory Processing' : 'Standard Processing'} Complete`,
          message: `Your prompt has been optimized successfully. Quality score: ${workflowResult?.output?.quality_score?.toFixed(2) || 'N/A'}`,
          action: {
            label: 'View Results',
            onClick: () => {
              window.scrollTo({
                top: document.body.scrollHeight,
                behavior: 'smooth'
              });
            }
          }
        });
        
        // Don't show toast here since we already show a notification
        setIsPolling(false);
        setWorkflowId(null);
        if (processingMethod === 'memory') {
            // Add to chat history for memory mode
            const chatEntry = {
              id: `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
              prompt: currentChatPrompt,
              response: workflowResult,
              timestamp: new Date()
            };
            
            setChatHistory(prev => [...prev, chatEntry]);
            setCurrentChatPrompt('');
            
            // Create or update saved chat
            if (!currentChatId) {
              const newChatId = `chat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
              const newChat = {
                id: newChatId,
                title: currentChatPrompt.substring(0, 50) + (currentChatPrompt.length > 50 ? '...' : ''),
                messages: [chatEntry],
                lastUpdated: new Date()
              };
              setSavedChats(prev => [newChat, ...prev]);
              setCurrentChatId(newChatId);
            }
        } else {
            setPrompt('');
        }
      } else if (status === 'cancelled') {
        // Clear frontend cache when workflow is cancelled to prevent returning stale results
        const currentRequest: PromptRequest = {
          prompt: processingMethod === 'memory' ? currentChatPrompt : prompt,
          prompt_type: promptType,
          return_comparison: returnComparison,
          use_langgraph: processingMethod === 'langgraph'
        };
        
        // Add user_id for memory requests
        if (processingMethod === 'memory' && userId) {
          (currentRequest as any).user_id = userId;
        }
        
        clearCacheForRequest(currentRequest);
        
        addNotification({
          type: 'warning',
          title: 'Workflow Cancelled',
          message: `${processingMethod === 'langgraph' ? 'LangGraph workflow' : processingMethod === 'memory' ? 'Memory processing' : 'Standard processing'} was cancelled by user.`
        });
        handleTerminalState('Workflow was cancelled.', 'info');
      } else if (status === 'failed') {
        setResult(null);
        
        // Clear cache for failed workflows to allow retry
        const currentRequest: PromptRequest = {
          prompt: processingMethod === 'memory' ? currentChatPrompt : prompt,
          prompt_type: promptType,
          return_comparison: returnComparison,
          use_langgraph: processingMethod === 'langgraph'
        };
        
        // Add user_id for memory requests
        if (processingMethod === 'memory' && userId) {
          (currentRequest as any).user_id = userId;
        }
        
        clearCacheForRequest(currentRequest);
        
        addNotification({
          type: 'error',
          title: 'Workflow Failed',
          message: `${processingMethod === 'langgraph' ? 'LangGraph workflow' : processingMethod === 'memory' ? 'Memory processing' : 'Standard processing'} encountered an error: ${error || 'Unknown error'}`
        });
        handleTerminalState(error || 'Workflow failed', 'error');
      } else if (status === 'running' && !grace_period_active) {
        // Remove the toast here as it causes repetitive messages
        // The grace period countdown visual feedback is enough
      }
    }

    return () => {
      if (timer) clearInterval(timer);
    };
  }, [workflowStatusQuery.data, isPolling, workflowId, processingMethod, currentChatPrompt, prompt, currentChatId]);

  // Handle cancellation
  const handleCancel = () => {
    if (workflowId) {
      // Clear frontend cache for the current request to prevent stale cached results
      const currentRequest: PromptRequest = {
        prompt: processingMethod === 'memory' ? currentChatPrompt : prompt,
        prompt_type: promptType,
        return_comparison: returnComparison,
        use_langgraph: processingMethod === 'langgraph'
      };
      
      // Add user_id for memory requests
      if (processingMethod === 'memory' && userId) {
        (currentRequest as any).user_id = userId;
      }
      
      clearCacheForRequest(currentRequest);
      
      cancelWorkflowMutation.mutate(workflowId, {
        onSuccess: () => {
          toast.info('Workflow cancellation requested.');
          // The useWorkflowStatus hook will handle the final state change
        },
        onError: (error) => {
          toast.error(`Failed to cancel workflow: ${error.message}`);
          // If cancellation fails, we might need to reset state here too
          setIsPolling(false);
          setWorkflowId(null);
        }
      });
    }
  };

  // Initialize user ID, chat history, and saved chats from localStorage on component mount
  useEffect(() => {
    // Get or create user ID with better persistence
    const getOrCreateUserId = () => {
      // Try to get from localStorage first
      let storedUserId = localStorage.getItem('promptEngineer_userId');
      
      // If not in localStorage, try sessionStorage
      if (!storedUserId) {
        storedUserId = sessionStorage.getItem('promptEngineer_userId');
      }
      
      // If still no ID, create a new one
      if (!storedUserId) {
        const newUserId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        // Store in both localStorage and sessionStorage for redundancy
        localStorage.setItem('promptEngineer_userId', newUserId);
        sessionStorage.setItem('promptEngineer_userId', newUserId);
        return newUserId;
      }
      
      return storedUserId;
    };

    // Set the user ID
    const userId = getOrCreateUserId();
    setUserId(userId);

    // Load saved chats and current chat history
    const loadSavedChats = () => {
      const storedChats = localStorage.getItem(`promptEngineer_savedChats_${userId}`);
      if (storedChats) {
        try {
          const parsedChats = JSON.parse(storedChats);
          const chatsWithDates = parsedChats.map((chat: any) => ({
            ...chat,
            lastUpdated: new Date(chat.lastUpdated),
            messages: chat.messages.map((msg: any) => ({
              ...msg,
              timestamp: new Date(msg.timestamp)
            }))
          }));
          setSavedChats(chatsWithDates);
        } catch (error) {
          console.error('Failed to parse saved chats', error);
          localStorage.removeItem(`promptEngineer_savedChats_${userId}`);
        }
      }
    };

    const loadCurrentChat = () => {
      const currentChatIdStored = localStorage.getItem(`promptEngineer_currentChatId_${userId}`);
      if (currentChatIdStored) {
        setCurrentChatId(currentChatIdStored);
        const chat = savedChats.find(c => c.id === currentChatIdStored);
        if (chat) {
          setChatHistory(chat.messages);
        }
      }
    };

    if (userId) {
      loadSavedChats();
      loadCurrentChat();
    }

    // Set up storage event listener for cross-tab synchronization
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'promptEngineer_userId' && e.newValue && e.newValue !== userId) {
        setUserId(e.newValue);
      }
      if (e.key === 'promptEngineer_chatHistory' && e.newValue) {
        try {
          const parsedHistory = JSON.parse(e.newValue);
          // Filter history for current user
          const userHistory = parsedHistory.filter((chat: any) => chat.userId === userId);
          const historyWithDates = userHistory.map((chat: any) => ({
            ...chat,
            timestamp: new Date(chat.timestamp)
          }));
          setChatHistory(historyWithDates);
        } catch (error) {
          console.error('Failed to parse updated chat history', error);
        }
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, []);

  // Update storage when saved chats change
  useEffect(() => {
    if (userId && savedChats.length >= 0) {
      localStorage.setItem(`promptEngineer_savedChats_${userId}`, JSON.stringify(savedChats));
    }
  }, [savedChats, userId]);

  // Update current chat when chat history changes
  useEffect(() => {
    if (userId && currentChatId && chatHistory.length > 0) {
      setSavedChats(prev => {
        const updatedChats = prev.map(chat => {
          if (chat.id === currentChatId) {
            return {
              ...chat,
              messages: chatHistory,
              lastUpdated: new Date()
            };
          }
          return chat;
        });
        return updatedChats;
      });
      localStorage.setItem(`promptEngineer_currentChatId_${userId}`, currentChatId);
    }
  }, [chatHistory, currentChatId, userId]);
  
  // Load user-specific data when userId changes
  useEffect(() => {
    if (userId) {
      localStorage.setItem('promptEngineer_userId', userId);
      sessionStorage.setItem('promptEngineer_userId', userId);
      
      // Load saved chats for this user
      const storedChats = localStorage.getItem(`promptEngineer_savedChats_${userId}`);
      if (storedChats) {
        try {
          const parsedChats = JSON.parse(storedChats);
          const chatsWithDates = parsedChats.map((chat: any) => ({
            ...chat,
            lastUpdated: new Date(chat.lastUpdated),
            messages: chat.messages.map((msg: any) => ({
              ...msg,
              timestamp: new Date(msg.timestamp)
            }))
          }));
          setSavedChats(chatsWithDates);
        } catch (error) {
          console.error('Failed to load saved chats for user', error);
        }
      }
      
      // Load current chat
      const currentChatIdStored = localStorage.getItem(`promptEngineer_currentChatId_${userId}`);
      if (currentChatIdStored) {
        setCurrentChatId(currentChatIdStored);
      }
    }
  }, [userId]);

  // Clear results when switching processing methods
  useEffect(() => {
    setResult(null);
    setWorkflowId(null);
    setIsPolling(false);
    // Don't clear prompt content to preserve user's work
  }, [processingMethod]);

    const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
        
    const inputPrompt = processingMethod === 'memory' ? currentChatPrompt : prompt;
    
    if (!inputPrompt.trim()) {
      toast.error('Please enter a prompt');
      return;
    }

    // Reset previous result for non-memory modes
    if (processingMethod !== 'memory') {
      setResult(null);
    }

    // Prepare the base request
    const request: PromptRequest = {
      prompt: inputPrompt.trim(),
      prompt_type: promptType,
      return_comparison: returnComparison,
      use_langgraph: processingMethod === 'langgraph',
    };

    // Add memory context if using memory mode
    if (processingMethod === 'memory') {
      if (!userId) {
        toast.error('User ID is required when using memory');
        return;
      }
      
      // Build comprehensive chat history for context
      if (chatHistory.length > 0) {
        const contextMessages: Array<{role: 'user' | 'assistant', content: string}> = [];
        
        chatHistory.forEach(chat => {
          // Add user message
          contextMessages.push({
            role: 'user',
            content: chat.prompt
          });
          
          // Add assistant response
          if (chat.response?.output?.optimized_prompt) {
            contextMessages.push({
              role: 'assistant',
              content: chat.response.output.optimized_prompt
            });
          }
        });
        
        request.chat_history = contextMessages;
      }
    }

    try {
      let response: PromptResponse | undefined;
      
      // Check if this is a retry (same prompt that was previously cached or cancelled)
      const cacheKey = getCacheKey(request);
      const existingCache = localStorage.getItem(cacheKey);
      const isRetry = !!existingCache;
      
      // Clear cache if it exists to ensure fresh processing
      if (isRetry) {
        localStorage.removeItem(cacheKey);
      }
      
      if (processingMethod === 'memory') {
        if (!userId.trim()) {
          toast.error('User ID is required when using memory');
          return;
        }
        response = await processPromptWithMemoryMutation.mutateAsync({
          request: { ...request, user_id: userId },
          skipCache: isRetry  // Skip cache check if this is a retry
        });
      } else {
        // For both standard and langgraph methods, use the standard endpoint
        // The use_langgraph flag in the request determines the processing type
        response = await processPromptMutation.mutateAsync({ 
          request,
          skipCache: isRetry  // Skip cache check if this is a retry
        });
      }
      
      if (response) {
        // Start polling for the workflow result
        setWorkflowId(response.workflow_id);
        setIsPolling(true);
        
        // Add start notification
        addNotification({
          type: 'info',
          title: `${processingMethod === 'langgraph' ? 'LangGraph Workflow' : processingMethod === 'memory' ? 'Memory Processing' : 'Standard Processing'} Started`,
          message: `Processing your prompt with workflow ID: ${response.workflow_id.slice(-8)}`
        });
        
        // Remove duplicate toast since we already show notification
      }
    } catch (error: any) {
      console.error('Processing failed:', error);
      // Error is already handled by the mutation's onError
    }
  };

  const isLoading = processPromptMutation.isPending || 
                   processPromptWithMemoryMutation.isPending || 
                   isPolling;
  
  const canCancel = (isPolling && workflowId && workflowStatusQuery.data?.grace_period_active) || 
                    (isPolling && workflowId && gracePeriodCountdown > 0);
  
  // Reset form after successful submission
  React.useEffect(() => {
    if (processPromptMutation.isSuccess || 
        processPromptWithMemoryMutation.isSuccess) {
      // Scroll to results
      window.scrollTo({
        top: document.body.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [processPromptMutation.isSuccess, 
      processPromptWithMemoryMutation.isSuccess]);

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

  // Chat management functions
  const startNewChat = () => {
    setChatHistory([]);
    setCurrentChatId(null);
    setResult(null);
    setCurrentChatPrompt('');
    setShowChatList(false);
    localStorage.removeItem(`promptEngineer_currentChatId_${userId}`);
  };

  const loadChat = (chatId: string) => {
    const chat = savedChats.find(c => c.id === chatId);
    if (chat) {
      setChatHistory(chat.messages);
      setCurrentChatId(chatId);
      setShowChatList(false);
      // Set result to the last response if available
      if (chat.messages.length > 0) {
        setResult(chat.messages[chat.messages.length - 1].response);
      }
    }
  };

    const deleteChatHandler = (chatId: string) => {
    setChatToDelete(chatId);
    setIsDeleteDialogOpen(true);
  };

  const handleTitleChange = (chatId: string, newTitle: string) => {
    setSavedChats(prev =>
      prev.map(chat =>
        chat.id === chatId ? { ...chat, title: newTitle, lastUpdated: new Date() } : chat
      )
    );
    setEditingChatId(null);
    setEditingTitle('');
  };

  const switchToStandardMode = () => {
    setProcessingMethod('standard');
    setShowChatList(false);
    setChatHistory([]);
    setCurrentChatId(null);
    setResult(null);
    setCurrentChatPrompt('');
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
            {/* Processing Method Selection */}
            <div className="mb-6">
              <label className="text-sm font-medium mb-3 block">
                Processing Method
              </label>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Standard Processing */}
                <div className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                  processingMethod === 'standard' 
                    ? 'border-primary bg-primary/5' 
                    : 'border-border hover:border-primary/50'
                }`}
                onClick={() => setProcessingMethod('standard')}
                >
                  <div className="flex items-center space-x-2 mb-2">
                    <div className={`w-4 h-4 rounded-full border-2 ${
                      processingMethod === 'standard' 
                        ? 'border-primary bg-primary' 
                        : 'border-muted-foreground'
                    }`} />
                    <span className="font-medium">Standard Processing</span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Basic prompt optimization with single-agent processing
                  </p>
                </div>

                {/* Memory Enhanced */}
                <div className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                  processingMethod === 'memory' 
                    ? 'border-primary bg-primary/5' 
                    : 'border-border hover:border-primary/50'
                }`}
                onClick={() => setProcessingMethod('memory')}
                >
                  <div className="flex items-center space-x-2 mb-2">
                    <div className={`w-4 h-4 rounded-full border-2 ${
                      processingMethod === 'memory' 
                        ? 'border-primary bg-primary' 
                        : 'border-muted-foreground'
                    }`} />
                    <span className="font-medium">Memory Enhanced</span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Maintain conversation context and chat history
                  </p>
                </div>

                {/* LangGraph Workflow */}
                <div className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                  processingMethod === 'langgraph' 
                    ? 'border-primary bg-primary/5' 
                    : 'border-border hover:border-primary/50'
                }`}
                onClick={() => setProcessingMethod('langgraph')}
                >
                  <div className="flex items-center space-x-2 mb-2">
                    <div className={`w-4 h-4 rounded-full border-2 ${
                      processingMethod === 'langgraph' 
                        ? 'border-primary bg-primary' 
                        : 'border-muted-foreground'
                    }`} />
                    <span className="font-medium">LangGraph Workflow</span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Complex multi-agent workflow processing
                  </p>
                </div>
              </div>
            </div>

            {/* Memory Mode Chat Interface */}
            {processingMethod === 'memory' ? (
              <div className="space-y-4">
                {/* Memory Mode Header with Controls */}
                <div className="flex items-center justify-between p-3 bg-primary/10 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <Brain className="h-5 w-5 text-primary" />
                    <span className="font-medium text-primary">Memory-Enhanced Mode</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setShowChatList(!showChatList)}
                    >
                      <MessageSquare className="h-4 w-4 mr-1" />
                      Chats ({savedChats.length})
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={startNewChat}
                    >
                      <RefreshCw className="h-4 w-4 mr-1" />
                      New Chat
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={switchToStandardMode}
                    >
                      <ArrowLeft className="h-4 w-4 mr-1" />
                      Standard Mode
                    </Button>
                  </div>
                </div>

                {/* Chat List */}
                {showChatList && (
                  <div className="space-y-2 max-h-60 overflow-y-auto border rounded-lg p-3">
                    <div className="text-sm font-medium text-muted-foreground mb-2">
                      Saved Conversations
                    </div>
                    {savedChats.length === 0 ? (
                      <p className="text-sm text-muted-foreground text-center py-4">
                        No saved conversations yet. Start a new chat!
                      </p>
                    ) : (
                      savedChats.map((chat) => (
                        <div
                          key={chat.id}
                          className={`flex items-center justify-between p-2 rounded cursor-pointer hover:bg-muted/50 ${
                            currentChatId === chat.id ? 'bg-primary/10 border border-primary/20' : ''
                          }`}
                          onClick={() => loadChat(chat.id)}
                        >
                                                    <div className="flex-1 min-w-0">
                            {editingChatId === chat.id ? (
                              <input
                                type="text"
                                value={editingTitle}
                                onChange={(e) => setEditingTitle(e.target.value)}
                                onBlur={() => handleTitleChange(chat.id, editingTitle)}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter') {
                                    handleTitleChange(chat.id, editingTitle);
                                  }
                                }}
                                className="w-full p-1 border border-input rounded-md bg-background text-sm"
                                autoFocus
                              />
                            ) : (
                              <>
                                <p className="text-sm font-medium truncate">{chat.title}</p>
                                <p className="text-xs text-muted-foreground">
                                  {chat.messages.length} messages • {chat.lastUpdated.toLocaleDateString()}
                                </p>
                              </>
                            )}
                          </div>
                                                    <div className="flex items-center">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              if (editingChatId === chat.id) {
                                handleTitleChange(chat.id, editingTitle);
                              } else {
                                setEditingChatId(chat.id);
                                setEditingTitle(chat.title);
                              }
                            }}
                          >
                            {editingChatId === chat.id ? <Check className="h-4 w-4" /> : <Edit className="h-4 w-4" />}
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              deleteChatHandler(chat.id);
                            }}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                )}

                {/* Current Chat Display */}
                {currentChatId && (
                  <div className="text-sm text-muted-foreground">
                    Current: {savedChats.find(c => c.id === currentChatId)?.title || 'Untitled Chat'}
                  </div>
                )}

                {/* Chat History */}
                {chatHistory.length > 0 && (
                  <div className="space-y-4 max-h-96 overflow-y-auto border rounded-lg p-4 bg-muted/20">
                    <div className="flex items-center justify-between p-2 bg-card rounded border">
                      <div className="flex items-center text-sm font-medium">
                        <History className="h-4 w-4 mr-2 text-primary" />
                        <span>Conversation History</span>
                      </div>
                      <div className="text-xs text-muted-foreground bg-primary/10 px-2 py-1 rounded">
                        {chatHistory.length} message{chatHistory.length !== 1 ? 's' : ''}
                      </div>
                    </div>
                    {chatHistory.map((chat) => (
                      <div key={chat.id} className="space-y-3 p-4 bg-card border rounded-lg shadow-sm">
                        {/* User Message */}
                        <div className="space-y-2">
                          <div className="flex items-center space-x-2">
                            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                            <span className="font-semibold text-blue-700 dark:text-blue-300 text-sm">You</span>
                          </div>
                          <div className="pl-4 text-sm text-foreground bg-blue-50 dark:bg-blue-950/30 p-3 rounded-lg border-l-4 border-blue-500">
                            {chat.prompt}
                          </div>
                        </div>
                        
                        {/* AI Response */}
                        <div className="space-y-2">
                          <div className="flex items-center space-x-2">
                            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                            <span className="font-semibold text-green-700 dark:text-green-300 text-sm">AI Assistant</span>
                          </div>
                          <div className="pl-4">
                            <div className="bg-green-50 dark:bg-green-950/30 p-4 rounded-lg border-l-4 border-green-500">
                              <SyntaxHighlighter 
                                language="markdown" 
                                style={vscDarkPlus} 
                                customStyle={{ 
                                  background: 'transparent', 
                                  padding: '0',
                                  margin: '0',
                                  color: 'inherit'
                                }} 
                                codeTagProps={{ 
                                  style: { 
                                    fontFamily: 'inherit', 
                                    fontSize: 'inherit',
                                    color: 'hsl(var(--foreground))'
                                  } 
                                }}
                              >
                                {chat.response.output.optimized_prompt}
                              </SyntaxHighlighter>
                            </div>
                          </div>
                        </div>
                        
                        {/* Timestamp */}
                        <div className="flex justify-end">
                          <div className="text-xs text-muted-foreground bg-muted/50 px-2 py-1 rounded">
                            {chat.timestamp.toLocaleString()}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                
                {/* Chat Input */}
                <div className="border-t pt-4">
                  <form onSubmit={handleSubmit} className="space-y-4">
                    <div className="space-y-3">
                      <label className="text-sm font-medium text-foreground">
                        {chatHistory.length === 0 ? '💬 Start New Conversation' : '💭 Continue Conversation'}
                      </label>
                      <div className="flex space-x-3">
                        <div className="flex-1 space-y-2">
                          <Textarea
                            value={currentChatPrompt}
                            onChange={(e) => setCurrentChatPrompt(e.target.value)}
                            placeholder={chatHistory.length === 0 
                              ? "Type your message to start a new conversation with context memory..." 
                              : "Continue the conversation - previous context will be remembered..."
                            }
                            className="min-h-[120px] resize-none border-2 transition-colors focus:border-primary"
                            required
                          />
                          <div className="flex items-center justify-between text-xs text-muted-foreground">
                            <span>Characters: {currentChatPrompt.length}</span>
                            <span className="text-primary">Context preserved across messages</span>
                          </div>
                        </div>
                        
                        <div className="flex flex-col space-y-2 min-w-[120px]">
                          <Button 
                            type="submit" 
                            disabled={isLoading}
                            className="h-full bg-gradient-to-br from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 transition-all duration-200"
                          >
                            {isLoading ? (
                              <>
                                <LoadingSpinner size="sm" className="mr-2" />
                                <span className="text-xs">Sending...</span>
                              </>
                            ) : (
                              <>
                                <MessageSquare className="h-4 w-4 mr-2" />
                                <span className="text-sm">Send</span>
                              </>
                            )}
                          </Button>
                          
                          {canCancel && (
                            <Button
                              type="button"
                              variant="destructive"
                              onClick={handleCancel}
                              className="text-xs"
                            >
                              Cancel ({gracePeriodCountdown}s)
                            </Button>
                          )}
                          
                          {isLoading && !canCancel && (
                            <div className="text-xs text-center text-muted-foreground px-2 py-2 bg-muted/50 rounded border">
                              <div className="animate-pulse">Processing with memory...</div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </form>
                </div>
              </div>
            ) : (
              /* Standard and LangGraph Processing */
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* Method Info Banner */}
                {processingMethod === 'langgraph' && (
                  <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950 rounded-lg border border-blue-200 dark:border-blue-800">
                    <div className="flex items-center space-x-2 mb-2">
                      <Zap className="h-5 w-5 text-blue-600" />
                      <span className="font-medium text-blue-700 dark:text-blue-300">LangGraph Workflow Mode</span>
                    </div>
                    <p className="text-sm text-blue-600 dark:text-blue-400">
                      Advanced multi-agent workflow processing for complex prompts with enhanced analysis and optimization capabilities.
                    </p>
                  </div>
                )}
                
                {processingMethod === 'standard' && (
                  <div className="p-4 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950 dark:to-emerald-950 rounded-lg border border-green-200 dark:border-green-800">
                    <div className="flex items-center space-x-2 mb-2">
                      <Brain className="h-5 w-5 text-green-600" />
                      <span className="font-medium text-green-700 dark:text-green-300">Standard Processing Mode</span>
                    </div>
                    <p className="text-sm text-green-600 dark:text-green-400">
                      Fast and efficient prompt optimization using our core AI agent for quick results.
                    </p>
                  </div>
                )}

                <div className="space-y-3">
                  <label className="text-sm font-medium block">
                    Prompt Content
                  </label>
                  <Textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder={processingMethod === 'langgraph' 
                      ? "Enter your complex prompt here for advanced multi-agent processing...\n\nExamples:\n- Create a comprehensive business strategy with market analysis\n- Design a complex software architecture with multiple components\n- Develop a detailed research methodology with data collection strategies"
                      : "Enter your prompt here for optimization...\n\nExamples:\n- Write a function to sort a list\n- Create a data analysis report\n- Draft a business strategy document"
                    }
                    className="min-h-[200px] resize-none"
                    required
                  />
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>Characters: {prompt.length}</span>
                    {processingMethod === 'langgraph' && prompt.length > 0 && (
                      <span className="text-blue-600 dark:text-blue-400">
                        Complex processing will be applied
                      </span>
                    )}
                  </div>
                </div>

              <div className="space-y-4">
                {/* Prompt Type Selection */}
                <div className="space-y-3">
                  <label className="text-sm font-semibold text-foreground">
                    Prompt Type Configuration
                  </label>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    {/* Auto-detect */}
                    <div 
                      onClick={() => setPromptType('auto')}
                      className={`relative p-4 rounded-lg border-2 cursor-pointer transition-all hover:shadow-md ${
                        promptType === 'auto' 
                          ? 'border-primary bg-primary/5 shadow-md' 
                          : 'border-border hover:border-primary/50 bg-card'
                      }`}
                    >
                      {promptType === 'auto' && (
                        <div className="absolute top-2 right-2">
                          <Check className="h-4 w-4 text-primary" />
                        </div>
                      )}
                      <div className="space-y-2">
                        <div className="flex items-center space-x-2">
                          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center">
                            <Brain className="h-4 w-4 text-white" />
                          </div>
                          <span className="font-medium">Auto-detect</span>
                        </div>
                        <p className="text-xs text-muted-foreground">
                          AI intelligently determines the optimal prompt type
                        </p>
                      </div>
                    </div>
                    
                    {/* Raw Prompt */}
                    <div 
                      onClick={() => setPromptType('raw')}
                      className={`relative p-4 rounded-lg border-2 cursor-pointer transition-all hover:shadow-md ${
                        promptType === 'raw' 
                          ? 'border-primary bg-primary/5 shadow-md' 
                          : 'border-border hover:border-primary/50 bg-card'
                      }`}
                    >
                      {promptType === 'raw' && (
                        <div className="absolute top-2 right-2">
                          <Check className="h-4 w-4 text-primary" />
                        </div>
                      )}
                      <div className="space-y-2">
                        <div className="flex items-center space-x-2">
                          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center">
                            <Edit className="h-4 w-4 text-white" />
                          </div>
                          <span className="font-medium">Raw Prompt</span>
                        </div>
                        <p className="text-xs text-muted-foreground">
                          Process exactly as written without restructuring
                        </p>
                      </div>
                    </div>
                    
                    {/* Structured Prompt */}
                    <div 
                      onClick={() => setPromptType('structured')}
                      className={`relative p-4 rounded-lg border-2 cursor-pointer transition-all hover:shadow-md ${
                        promptType === 'structured' 
                          ? 'border-primary bg-primary/5 shadow-md' 
                          : 'border-border hover:border-primary/50 bg-card'
                      }`}
                    >
                      {promptType === 'structured' && (
                        <div className="absolute top-2 right-2">
                          <Check className="h-4 w-4 text-primary" />
                        </div>
                      )}
                      <div className="space-y-2">
                        <div className="flex items-center space-x-2">
                          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center">
                            <Settings className="h-4 w-4 text-white" />
                          </div>
                          <span className="font-medium">Structured</span>
                        </div>
                        <p className="text-xs text-muted-foreground">
                          Deep analysis with structural optimization
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Basic Options */}
              <div className="space-y-4">
                <div className="p-4 bg-muted/50 rounded-lg border">
                  <h4 className="text-sm font-medium mb-3 flex items-center">
                    ⚙️ Processing Options
                  </h4>
                  <div className="space-y-3">
                    <label className="flex items-center space-x-3 cursor-pointer group">
                      <input
                        type="checkbox"
                        id="returnComparison"
                        checked={returnComparison}
                        onChange={(e) => setReturnComparison(e.target.checked)}
                        className="w-4 h-4 text-primary bg-background border-2 border-muted-foreground rounded focus:ring-primary focus:ring-2 transition-colors"
                      />
                      <div className="flex-1">
                        <span className="text-sm font-medium group-hover:text-primary transition-colors">
                          Show Before/After Comparison
                        </span>
                        <p className="text-xs text-muted-foreground mt-1">
                          Display side-by-side comparison of original and optimized prompts
                        </p>
                      </div>
                    </label>
                  </div>
                </div>
              </div>

              {/* Advanced Options Toggle */}
              <div className="pt-2">
                <button
                  type="button"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="text-sm font-medium text-primary flex items-center hover:underline"
                >
                  {showAdvanced ? (
                    <>
                      <ChevronUp className="h-4 w-4 mr-1" />
                      Hide Advanced Options
                    </>
                  ) : (
                    <>
                      <ChevronDown className="h-4 w-4 mr-1" />
                      Show Advanced Options
                    </>
                  )}
                </button>
              </div>

              {/* Advanced Options */}
              {showAdvanced && (
                <div className="mt-4 space-y-4 p-4 bg-muted/30 rounded-lg border border-dashed">
                  <div className="space-y-3">
                    <div className="flex items-center space-x-2 text-muted-foreground">
                      <Settings className="h-4 w-4" />
                      <span className="text-sm font-medium">Advanced Configuration</span>
                    </div>
                    
                    <div className="text-center py-4">
                      <p className="text-xs text-muted-foreground">
                        No additional advanced options available for {processingMethod} processing method.
                      </p>
                    </div>
                  </div>
                </div>
              )}

                <div className="pt-4 border-t">
                  <div className="flex flex-col space-y-3">
                    <Button 
                      type="submit" 
                      className={`w-full py-3 text-base font-medium transition-all duration-200 ${
                        processingMethod === 'langgraph' 
                          ? 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700' 
                          : processingMethod === 'standard'
                            ? 'bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700'
                            : 'bg-primary hover:bg-primary/90'
                      }`}
                      disabled={isLoading}
                    >
                      {isLoading ? (
                        <>
                          <LoadingSpinner size="sm" className="mr-2" />
                          {processingMethod === 'langgraph' ? 'Running Advanced Workflow...' : 'Processing Prompt...'}
                        </>
                      ) : (
                        <>
                          {processingMethod === 'langgraph' ? (
                            <>
                              <Zap className="mr-2 h-5 w-5" />
                              Start LangGraph Workflow
                            </>
                          ) : (
                            <>
                              <Play className="mr-2 h-5 w-5" />
                              Optimize Prompt
                            </>
                          )}
                        </>
                      )}
                    </Button>
                    
                    {canCancel && (
                      <Button
                        type="button"
                        variant="destructive"
                        onClick={handleCancel}
                        className="w-full"
                      >
                        Cancel Workflow ({gracePeriodCountdown}s)
                      </Button>
                    )}
                    
                    {isLoading && !canCancel && (
                      <div className="flex items-center justify-center p-3 bg-amber-50 dark:bg-amber-950 border border-amber-200 dark:border-amber-800 rounded-lg">
                        <div className="animate-pulse flex items-center space-x-2">
                          <div className="w-2 h-2 bg-amber-500 rounded-full"></div>
                          <div className="w-2 h-2 bg-amber-500 rounded-full"></div>
                          <div className="w-2 h-2 bg-amber-500 rounded-full"></div>
                        </div>
                        <span className="ml-3 text-sm font-medium text-amber-700 dark:text-amber-300">
                          {processingMethod === 'langgraph' ? 'Multi-agent workflow in progress' : 'Processing cannot be cancelled'}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </form>
            )}
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
              <div className="space-y-6">
                {/* Workflow Status Card */}
                {workflowId && (
                  <div className="p-4 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-950 dark:to-purple-950 rounded-lg border border-indigo-200 dark:border-indigo-800">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <div className="animate-spin">
                          <Zap className="h-5 w-5 text-indigo-600" />
                        </div>
                        <span className="font-medium text-indigo-700 dark:text-indigo-300">
                          {processingMethod === 'langgraph' ? 'LangGraph Workflow' : processingMethod === 'memory' ? 'Memory Processing' : 'Standard Processing'} Active
                        </span>
                      </div>
                      {workflowId && (
                        <code className="text-xs bg-white/50 dark:bg-black/20 px-2 py-1 rounded">
                          ID: {workflowId.slice(-8)}
                        </code>
                      )}
                    </div>
                    <div className="text-sm text-indigo-600 dark:text-indigo-400">
                      Status: {workflowStatusQuery.data?.status || 'Initializing...'}
                    </div>
                  </div>
                )}
                
                {/* Processing Animation */}
                <div className="flex flex-col items-center justify-center py-8 space-y-6">
                  <div className="relative">
                    <div className="absolute inset-0 animate-ping">
                      <div className="h-20 w-20 rounded-full bg-primary/20"></div>
                    </div>
                    <LoadingSpinner size="lg" className="relative z-10" />
                  </div>
                  
                  <div className="text-center space-y-2">
                    <h3 className="text-lg font-semibold">
                      {processingMethod === 'langgraph' 
                        ? 'Running Multi-Agent Workflow...'
                        : processingMethod === 'memory' 
                          ? 'Processing with Context Memory...'
                          : 'Optimizing Your Prompt...'
                      }
                    </h3>
                    <p className="text-sm text-muted-foreground max-w-md">
                      {processingMethod === 'langgraph'
                        ? 'Multiple specialized AI agents are collaborating to analyze and enhance your complex prompt.'
                        : processingMethod === 'memory'
                          ? 'Processing your prompt while maintaining conversation context and history.'
                          : 'Our AI agent is analyzing, classifying, and optimizing your prompt for better results.'
                      }
                    </p>
                  </div>
                  
                  {/* Progress Steps */}
                  <div className="flex items-center space-x-3 text-sm">
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-primary rounded-full animate-pulse"></div>
                      <span className="text-muted-foreground">Analyzing</span>
                    </div>
                    <span className="text-muted-foreground">→</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-primary/60 rounded-full animate-pulse"></div>
                      <span className="text-muted-foreground">Optimizing</span>
                    </div>
                    <span className="text-muted-foreground">→</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-primary/30 rounded-full"></div>
                      <span className="text-muted-foreground">Finalizing</span>
                    </div>
                  </div>
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
                {result.comparison && returnComparison ? (
                  <div className="space-y-2">
                    <button
                      onClick={() => setIsComparisonOpen(!isComparisonOpen)}
                      className="w-full text-left flex items-center justify-between p-2 bg-muted/50 rounded-lg"
                    >
                      <h4 className="font-medium">📝 Comparison</h4>
                      {isComparisonOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                    </button>
                    {isComparisonOpen && (
                      <div className="space-y-4 pt-2 pl-4 border-l-2 ml-2">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div>
                            <h5 className="font-medium mb-1 text-sm">Original Prompt</h5>
                            <SyntaxHighlighter language="markdown" style={vscDarkPlus} className="rounded-lg text-sm w-full">
                              {result.comparison.side_by_side.original}
                            </SyntaxHighlighter>
                          </div>
                          <div>
                            <h5 className="font-medium mb-1 text-sm">Optimized Prompt</h5>
                            <SyntaxHighlighter language="markdown" style={vscDarkPlus} className="rounded-lg text-sm w-full">
                              {result.comparison.side_by_side.optimized}
                            </SyntaxHighlighter>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="space-y-4">
                    {/* Optimized Prompt Header */}
                    <div className="flex items-center justify-between p-3 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950 dark:to-emerald-950 rounded-lg border border-green-200 dark:border-green-800">
                      <div className="flex items-center space-x-2">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center">
                          <Check className="h-4 w-4 text-white" />
                        </div>
                        <div>
                          <h4 className="font-semibold text-green-700 dark:text-green-300">Optimized Prompt</h4>
                          <p className="text-xs text-green-600 dark:text-green-400">
                            Enhanced for {result.output.domain || 'general'} domain
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={handleCopyResult}
                          className="text-green-600 hover:text-green-700 dark:text-green-400 dark:hover:text-green-300"
                        >
                          <Copy className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                    
                    {/* Optimized Prompt Content */}
                    <div className="relative group">
                      <div className="absolute inset-0 bg-gradient-to-r from-primary/5 via-transparent to-primary/5 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity"></div>
                      <div className="relative bg-card border-2 border-primary/20 rounded-lg p-6 shadow-lg">
                        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                          <Badge variant="secondary" className="text-xs">
                            Quality: {result.output.quality_score?.toFixed(2) || 'N/A'}
                          </Badge>
                        </div>
                        <SyntaxHighlighter 
                          language="markdown" 
                          style={vscDarkPlus} 
                          className="!bg-transparent rounded-lg text-sm"
                          customStyle={{
                            background: 'transparent',
                            padding: '0',
                            margin: '0',
                            fontSize: '0.95rem',
                            lineHeight: '1.6',
                            fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
                          }}
                        >
                          {result.output.optimized_prompt}
                        </SyntaxHighlighter>
                      </div>
                    </div>
                    
                    {/* Optimization Stats */}
                    <div className="grid grid-cols-3 gap-3">
                      <div className="p-3 bg-muted/50 rounded-lg text-center">
                        <p className="text-2xl font-bold text-primary">
                          {result.output.iterations_used || 1}
                        </p>
                        <p className="text-xs text-muted-foreground">Iterations</p>
                      </div>
                      <div className="p-3 bg-muted/50 rounded-lg text-center">
                        <p className="text-2xl font-bold text-green-600">
                          {((result.output.quality_score || 0) * 10).toFixed(0)}/10
                        </p>
                        <p className="text-xs text-muted-foreground">Quality Score</p>
                      </div>
                      <div className="p-3 bg-muted/50 rounded-lg text-center">
                        <p className="text-2xl font-bold text-blue-600">
                          {result.processing_time_seconds ? `${result.processing_time_seconds.toFixed(1)}s` : 'N/A'}
                        </p>
                        <p className="text-xs text-muted-foreground">Process Time</p>
                      </div>
                    </div>
                  </div>
                )}


                {/* Analysis */}
                {result.analysis && (
                  <div className="space-y-2">
                    <button
                      onClick={() => setIsAnalysisOpen(!isAnalysisOpen)}
                      className="w-full text-left flex items-center justify-between p-2 bg-muted/50 rounded-lg"
                    >
                      <h4 className="font-medium">🔍 Analysis</h4>
                      {isAnalysisOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                    </button>
                    {isAnalysisOpen && (
                      <div className="p-3 bg-muted rounded-lg ml-2 pl-4 border-l-2 space-y-4">
                        <p className="text-sm">
                          <strong>Classification:</strong> {result.analysis.classification.reasoning}
                        </p>
                        <p className="text-sm">
                          <strong>Key Topics:</strong> {result.analysis.classification.key_topics.join(', ')}
                        </p>
                        {result?.comparison?.improvement_ratio !== undefined && (
                          <div className="space-y-2">
                            <p className="text-sm font-medium">Quality Improvement Score</p>
                            <div className="flex items-center space-x-3">
                              <Progress value={result.comparison.improvement_ratio * 100} className="flex-1" />
                              <span className="text-lg font-bold text-green-600">
                                {(result.comparison.improvement_ratio * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
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
      <ConfirmationDialog
        open={isDeleteDialogOpen}
        onOpenChange={setIsDeleteDialogOpen}
        title="Delete Chat"
        description="Are you sure you want to delete this chat? This action cannot be undone."
        onConfirm={() => {
          if (chatToDelete) {
            setSavedChats(prev => prev.filter(chat => chat.id !== chatToDelete));
            if (currentChatId === chatToDelete) {
              startNewChat();
            }
          }
          setIsDeleteDialogOpen(false);
        }}
        confirmText="Delete"
        cancelText="Cancel"
      />
    </div>
  );
}
