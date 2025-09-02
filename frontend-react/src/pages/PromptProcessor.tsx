import React, { useState, useEffect } from 'react';
import { Brain, Zap, Play, Copy, Download, ChevronDown, ChevronUp, Send, History, ArrowLeft, MessageSquare, Trash2, RefreshCw, Edit, Check } from 'lucide-react';
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
  useWorkflowStatus
} from '@/hooks/useApi';
import { formatDuration } from '@/lib/utils';
import { toast } from 'sonner';
import type { PromptRequest, PromptResponse } from '@/types/api';

export function PromptProcessor() {
  const [prompt, setPrompt] = useState('');
  const [promptType, setPromptType] = useState<'auto' | 'raw' | 'structured'>('auto');
  const [returnComparison, setReturnComparison] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [useLangGraph, setUseLangGraph] = useState(false);
  const [useMemory, setUseMemory] = useState(false);
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
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [chatToDelete, setChatToDelete] = useState<string | null>(null);

  const processPromptMutation = useProcessPrompt();
  const processPromptWithMemoryMutation = useProcessPromptWithMemory();
  const cancelWorkflowMutation = useCancelWorkflow();
  
  // Workflow status polling
  const workflowStatusQuery = useWorkflowStatus(workflowId);
  
  // Handle workflow completion
  useEffect(() => {
    if (workflowStatusQuery.data) {
      const status = workflowStatusQuery.data.status;
      
      if (status === 'completed') {
        setResult(workflowStatusQuery.data.result);
        setIsPolling(false);
        setWorkflowId(null);
        
        // Add to chat history if using memory mode
        if (useMemory && workflowStatusQuery.data.result) {
          const newChatEntry = {
            id: workflowId || `entry_${Date.now()}`,
            prompt: useMemory ? currentChatPrompt : prompt,
            response: workflowStatusQuery.data.result,
            timestamp: new Date()
          };
          
          // Create new chat if none exists
          if (!currentChatId) {
            const newChatId = `chat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            const newChat = {
              id: newChatId,
              title: (useMemory ? currentChatPrompt : prompt).trim().substring(0, 50) + ((useMemory ? currentChatPrompt : prompt).trim().length > 50 ? '...' : ''),
              messages: [newChatEntry],
              lastUpdated: new Date()
            };
            
            setSavedChats(prev => [newChat, ...prev]);
            setCurrentChatId(newChatId);
            setChatHistory([newChatEntry]);
          } else {
            // Add to existing chat
            setChatHistory(prev => [...prev, newChatEntry]);
          }
          
          setCurrentChatPrompt('');
        } else {
          setPrompt('');
        }
        
        toast.success('Prompt processed successfully!');
      } else if (status === 'cancelled') {
        setIsPolling(false);
        setWorkflowId(null);
        toast.info('Workflow was cancelled');
      } else if (status === 'failed') {
        setIsPolling(false);
        setWorkflowId(null);
        toast.error(workflowStatusQuery.data.error || 'Workflow failed');
      }
    }
  }, [workflowStatusQuery.data, workflowId, useMemory, currentChatPrompt, prompt, currentChatId]);

  // Handle cancellation
  const handleCancel = () => {
    if (workflowId) {
      cancelWorkflowMutation.mutate(workflowId);
      setIsPolling(false);
      setWorkflowId(null);
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

    const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
        
    const inputPrompt = useMemory ? currentChatPrompt : prompt;
    
    if (!inputPrompt.trim()) {
      toast.error('Please enter a prompt');
      return;
    }

    // Reset previous result for non-memory modes
    if (!useMemory) {
      setResult(null);
    }

    // Prepare the base request
    const request: PromptRequest = {
      prompt: inputPrompt.trim(),
      prompt_type: promptType,
      return_comparison: returnComparison,
      use_langgraph: useLangGraph,
    };

    // Add memory context if using memory mode
    if (useMemory) {
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
      
      if (useMemory) {
        if (!userId.trim()) {
          toast.error('User ID is required when using memory');
          return;
        }
        response = await processPromptWithMemoryMutation.mutateAsync({
          request: { ...request, user_id: userId },
        });
      } else {
        response = await processPromptMutation.mutateAsync({ request });
      }
      
      if (response) {
        // Start polling for the workflow result
        setWorkflowId(response.workflow_id);
        setIsPolling(true);
        toast.info('Processing started. Please wait...');
      }
    } catch (error: any) {
      console.error('Processing failed:', error);
      // Error is already handled by the mutation's onError
    }
  };

  const isLoading = processPromptMutation.isPending || 
                   processPromptWithMemoryMutation.isPending ||
                   isPolling;
  
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
    setUseMemory(false);
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
            {/* Memory Mode Chat Interface */}
            {useMemory ? (
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
                  <div className="space-y-3 overflow-y-auto border rounded-lg p-3">
                    <div className="flex items-center text-sm font-medium text-muted-foreground">
                      <History className="h-4 w-4 mr-2" />
                      Conversation History
                    </div>
                    {chatHistory.map((chat) => (
                      <div key={chat.id} className="space-y-2 p-3 bg-muted/30 rounded-lg">
                        <div className="text-sm">
                          <span className="font-medium text-blue-600">You:</span> {chat.prompt}
                        </div>
                                                <div className="text-sm">
                          <span className="font-medium text-green-600">AI:</span>
                          <SyntaxHighlighter language="markdown" style={vscDarkPlus} customStyle={{ background: 'transparent', padding: '0', marginTop: '0.5rem' }} codeTagProps={{ style: { fontFamily: 'inherit', fontSize: 'inherit' } }}>
                            {chat.response.output.optimized_prompt}
                          </SyntaxHighlighter>
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {chat.timestamp.toLocaleString()}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                
                {/* Chat Input */}
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div className="flex space-x-2">
                    <Textarea
                      value={currentChatPrompt}
                      onChange={(e) => setCurrentChatPrompt(e.target.value)}
                      placeholder={chatHistory.length === 0 ? "Start a new conversation..." : "Continue the conversation..."}
                      className="flex-1 min-h-[100px]"
                      required
                    />
                    <div className="flex flex-col space-y-2">
                      <Button type="submit" disabled={isLoading} className="h-full">
                        {isLoading ? (
                          <LoadingSpinner size="sm" />
                        ) : (
                          <Send className="h-4 w-4" />
                        )}
                      </Button>
                      {isLoading && (
                        <Button
                          type="button"
                          variant="destructive"
                          onClick={handleCancel}
                          className="h-full"
                        >
                          Cancel
                        </Button>
                      )}
                    </div>
                  </div>
                </form>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">
                    Prompt Content
                  </label>
                  <Textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder={
                      "Enter your prompt here...\n\nExamples:\n- Write a function to sort a list\n- Create a data analysis report\n- Draft a business strategy document"
                    }
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

                {useMemory && (
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

              {/* Basic Options */}
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="returnComparison"
                    checked={returnComparison}
                    onChange={(e) => setReturnComparison(e.target.checked)}
                    className="rounded"
                  />
                  <label htmlFor="returnComparison" className="text-sm">
                    Show before/after comparison
                  </label>
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
                <div className="mt-4 space-y-4 pl-4 border-l-2 border-muted">
                  <div className="space-y-3">
                    <div className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id="useLangGraph"
                        checked={useLangGraph}
                        onChange={(e) => setUseLangGraph(e.target.checked)}
                        className="rounded"
                      />
                      <label htmlFor="useLangGraph" className="text-sm">
                        Enable LangGraph Workflow
                        <p className="text-xs text-muted-foreground">
                          For complex prompt processing with multiple agents
                        </p>
                      </label>
                    </div>

                    <div className="flex items-start space-x-2 pt-2">
                      <input
                        type="radio"
                        id="standardMode"
                        name="processingMode"
                        checked={!useMemory}
                        onChange={() => setUseMemory(false)}
                        className="mt-1"
                      />
                      <label htmlFor="standardMode" className="text-sm">
                        <span className="font-medium">Standard Processing</span>
                        <p className="text-xs text-muted-foreground">
                          Basic prompt optimization without additional features
                        </p>
                      </label>
                    </div>

                    <div className="flex items-start space-x-2">
                      <input
                        type="radio"
                        id="useMemory"
                        name="processingMode"
                        checked={useMemory}
                        onChange={() => {
                          setUseMemory(true);
                          setShowChatList(false);
                        }}
                        className="mt-1"
                        disabled={false}
                        title="Use memory to maintain context across requests"
                      />
                      <label htmlFor="useMemory" className="text-sm">
                        <span className="font-medium">Memory-Enhanced Processing</span>
                        <p className="text-xs text-muted-foreground">
                          Maintain conversation context and chat history
                        </p>
                      </label>
                    </div>


                    {userId && (
                      <div className="pl-6 pt-2">
                        <p className="text-xs text-muted-foreground">
                          Session ID: {userId.slice(-12)}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}

                                <div className="flex items-center space-x-2">
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
                  {isLoading && (
                    <Button
                      type="button"
                      variant="destructive"
                      onClick={handleCancel}
                    >
                      Cancel
                    </Button>
                  )}
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
                  <div>
                    <h4 className="font-medium mb-2">✨ Optimized Prompt</h4>
                    <SyntaxHighlighter language="markdown" style={vscDarkPlus} className="rounded-lg text-sm w-full">
                      {result.output.optimized_prompt}
                    </SyntaxHighlighter>
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
