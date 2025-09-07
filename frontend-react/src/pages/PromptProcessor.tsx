import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import { Send, Copy, Download, Trash2, Edit, Sparkles, Bot, User, Plus, Menu, X, Activity, Bell } from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { LoadingSpinner } from '@/components/ui/loading';
import { ConfirmationDialog } from '@/components/ui/confirmation-dialog';
import { 
  useProcessPromptWithMemory,
  useCancelWorkflow,
  useWorkflowStatus
} from '@/hooks/useApi';
import { formatDuration } from '@/utils';
import { toast } from 'sonner';
import { useNotifications } from '@/hooks/useNotifications';
import { NotificationsDropdown } from '@/components/ui/notifications';
import { NavLink } from 'react-router-dom';
import type { PromptRequest, PromptResponse } from '@/types/api';

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

export function PromptProcessor() {
  const location = useLocation();
  const [selectedModel, setSelectedModel] = useState<'standard' | 'langgraph'>('standard');
  const [currentInput, setCurrentInput] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [showSidebar, setShowSidebar] = useState(true);
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');
  const [userId, setUserId] = useState('');
  const [workflowId, setWorkflowId] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [gracePeriodCountdown, setGracePeriodCountdown] = useState(3);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const processPromptWithMemoryMutation = useProcessPromptWithMemory();
  const cancelWorkflowMutation = useCancelWorkflow();
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

  // Workflow status polling
  const workflowStatusQuery = useWorkflowStatus(workflowId);

  // Auto-scroll to bottom when new messages are added
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Initialize user ID and load saved sessions
  useEffect(() => {
    // Get or create user ID
    const getOrCreateUserId = () => {
      let storedUserId = localStorage.getItem('promptEngineer_userId');
      if (!storedUserId) {
        storedUserId = sessionStorage.getItem('promptEngineer_userId');
      }
      if (!storedUserId) {
        const newUserId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        localStorage.setItem('promptEngineer_userId', newUserId);
        sessionStorage.setItem('promptEngineer_userId', newUserId);
        return newUserId;
      }
      return storedUserId;
    };

    const userId = getOrCreateUserId();
    setUserId(userId);

    // Load saved sessions
    const loadSessions = () => {
      const storedSessions = localStorage.getItem(`promptEngineer_sessions_${userId}`);
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
          localStorage.removeItem(`promptEngineer_sessions_${userId}`);
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
      localStorage.setItem(`promptEngineer_sessions_${userId}`, JSON.stringify(sessions));
    }
  }, [sessions, userId]);

  // Handle workflow completion
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
      const { status, result: workflowResult, error } = workflowStatusQuery.data;

      if (status === 'completed' && workflowResult) {
        // Update the loading message with the actual response
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

        // Session will be auto-updated by the useEffect hook

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
        // Remove loading message
        setMessages(prev => prev.filter(msg => msg.id !== `loading_${workflowId}`));
        setIsPolling(false);
        setWorkflowId(null);
        toast.info('Workflow was cancelled.');
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
        toast.error(error || 'Workflow failed');
      }
    }

    return () => {
      if (timer) clearInterval(timer);
    };
  }, [workflowStatusQuery.data, isPolling, workflowId, selectedModel, messages, currentSessionId]);

  // Handle form submission
  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    
    if (!currentInput.trim()) {
      toast.error('Please enter a prompt');
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
      return_comparison: false,
      use_langgraph: selectedModel === 'langgraph',
      chat_history: chatHistory
    };

    try {
      let response: PromptResponse | undefined;
      
      // Always use memory-enhanced processing
      response = await processPromptWithMemoryMutation.mutateAsync({
        request: { ...request, user_id: userId },
        skipCache: false
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
        
        // Add start notification
        addNotification({
          type: 'info',
          title: `${selectedModel === 'langgraph' ? 'LangGraph' : 'Standard'} Model Started`,
          message: `Processing your prompt with workflow ID: ${response.workflow_id.slice(-8)}`
        });
      }
    } catch (error: any) {
      console.error('Processing failed:', error);
      // Remove loading message on error
      setMessages(prev => prev.filter(msg => msg.id !== loadingMessage.id));
      toast.error('Failed to process prompt');
    }
  };

  // Handle cancellation
  const handleCancel = () => {
    if (workflowId) {
      cancelWorkflowMutation.mutate(workflowId, {
        onSuccess: () => {
          toast.info('Workflow cancellation requested.');
        },
        onError: (error) => {
          toast.error(`Failed to cancel workflow: ${error.message}`);
          setIsPolling(false);
          setWorkflowId(null);
        }
      });
    }
  };

  // Create new session
  const createNewSession = () => {
    setCurrentSessionId(null);
    setMessages([]);
  };

  // Load session
  const loadSession = (sessionId: string) => {
    const session = sessions.find(s => s.id === sessionId);
    if (session) {
      setCurrentSessionId(sessionId);
      setMessages(session.messages);
      setSelectedModel(session.model);
    }
  };

  // Delete session
  const deleteSession = (sessionId: string) => {
    setSessions(prev => prev.filter(s => s.id !== sessionId));
    if (currentSessionId === sessionId) {
      setCurrentSessionId(null);
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
          title: firstUserMessage.content.substring(0, 50) + (firstUserMessage.content.length > 50 ? '...' : ''),
          messages: messages,
          lastUpdated: new Date(),
          model: selectedModel
        };
        setSessions(prev => [newSession, ...prev]);
        setCurrentSessionId(newSession.id);
      }
    }
  }, [messages, currentSessionId, selectedModel]);

  // Copy result to clipboard
  const handleCopyMessage = (content: string) => {
    navigator.clipboard.writeText(content);
    toast.success('Copied to clipboard');
  };

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

  const isLoading = processPromptWithMemoryMutation.isPending || isPolling;
  const canCancel = isPolling && workflowId && gracePeriodCountdown > 0;

  return (
    <div className="flex h-screen bg-background">
      {/* Professional Sidebar */}
      <div className={`${showSidebar ? 'w-64' : 'w-0'} transition-all duration-300 border-r border-gray-200 bg-gray-50 overflow-hidden flex flex-col`}>
        {/* Simple Header */}
        <div className="flex items-center px-4 py-4 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-black rounded-sm flex items-center justify-center">
              <Bot className="h-5 w-5 text-white" />
            </div>
            <span className="text-lg font-semibold text-gray-900">Prompt Engineer</span>
          </div>
        </div>

        {/* New Chat Button */}
        <div className="px-3 py-3">
          <Button 
            onClick={createNewSession}
            className="w-full justify-start bg-white hover:bg-gray-100 text-gray-900 border border-gray-300 shadow-none"
            variant="outline"
          >
            <Plus className="mr-2 h-4 w-4" />
            New chat
          </Button>
        </div>

        {/* Chat Sessions */}
        <div className="flex-1 flex flex-col min-h-0">
          <div className="flex-1 overflow-y-auto px-3 space-y-1">
            {sessions.length === 0 ? (
              <p className="text-sm text-gray-500 text-center py-8">
                No conversations yet
              </p>
            ) : (
              sessions.map(session => (
                <div
                  key={session.id}
                  className={`group relative p-2 rounded-md cursor-pointer hover:bg-gray-200 transition-colors ${
                    currentSessionId === session.id ? 'bg-gray-200' : ''
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
                        if (e.key === 'Enter') {
                          updateSessionTitle(session.id, editingTitle);
                        } else if (e.key === 'Escape') {
                          setEditingSessionId(null);
                          setEditingTitle('');
                        }
                      }}
                      className="w-full px-2 py-1 text-sm bg-white border border-gray-300 rounded"
                      autoFocus
                      onClick={(e) => e.stopPropagation()}
                    />
                  ) : (
                    <>
                      <div className="flex items-start justify-between">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm text-gray-900 truncate">{session.title}</p>
                          <p className="text-xs text-gray-500">
                            {session.messages.length} messages
                          </p>
                        </div>
                      </div>
                      <div className="absolute right-1 top-2 opacity-0 group-hover:opacity-100 transition-opacity flex space-x-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 w-6 p-0 hover:bg-gray-300"
                          onClick={(e) => {
                            e.stopPropagation();
                            setEditingSessionId(session.id);
                            setEditingTitle(session.title);
                          }}
                        >
                          <Edit className="h-3 w-3 text-gray-600" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 w-6 p-0 hover:bg-gray-300"
                          onClick={(e) => {
                            e.stopPropagation();
                            setSessionToDelete(session.id);
                            setIsDeleteDialogOpen(true);
                          }}
                        >
                          <Trash2 className="h-3 w-3 text-gray-600" />
                        </Button>
                      </div>
                    </>
                  )}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Simple Footer */}
        <div className="px-3 py-4 border-t border-gray-200">
          <div className="flex items-center justify-between">
            <NavLink
              to="/system-health"
              className="flex items-center space-x-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
            >
              <Activity className="h-4 w-4" />
              <span>System Health</span>
            </NavLink>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="border-b px-6 py-4 flex items-center justify-between bg-background">
          <div className="flex items-center space-x-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowSidebar(!showSidebar)}
            >
              {showSidebar ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </Button>
            
            <div className="flex items-center space-x-2">
              <Bot className="h-5 w-5 text-primary" />
              <h1 className="text-xl font-semibold">Prompt Processor</h1>
            </div>
          </div>

          {/* Model Selector */}
          <div className="flex items-center space-x-2">
            <span className="text-sm text-muted-foreground">Model:</span>
            <div className="flex rounded-lg border bg-muted/30 p-1">
              <Button
                variant={selectedModel === 'standard' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setSelectedModel('standard')}
                className="px-4"
              >
                <Sparkles className="mr-2 h-4 w-4" />
                Standard
              </Button>
              <Button
                variant={selectedModel === 'langgraph' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setSelectedModel('langgraph')}
                className="px-4"
              >
                <Bot className="mr-2 h-4 w-4" />
                LangGraph
              </Button>
            </div>
            
            <div className="flex items-center space-x-2">
              {/* Notification Icon */}
              <div className="relative">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={toggleDropdown}
                  className="relative"
                >
                  <Bell className="h-4 w-4" />
                  {unreadCount > 0 && (
                    <span className="absolute -top-1 -right-1 h-2 w-2 bg-red-500 rounded-full"></span>
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
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleDownloadConversation}
                >
                  <Download className="h-4 w-4" />
                </Button>
              )}
            </div>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center space-y-4">
              <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center">
                <Bot className="h-10 w-10 text-primary" />
              </div>
              <div>
                <h2 className="text-2xl font-semibold mb-2">Welcome to Prompt Processor</h2>
                <p className="text-muted-foreground max-w-md">
                  I'll help you optimize your prompts using our advanced multi-agent system. 
                  Choose between Standard (fast) or LangGraph (advanced) models.
                </p>
              </div>
              <div className="flex flex-col space-y-2 text-sm text-muted-foreground">
                <div className="flex items-center space-x-2">
                  <Badge variant="outline">Standard Model</Badge>
                  <span>Memory-enhanced optimization for quick results</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Badge variant="outline">LangGraph Model</Badge>
                  <span>Complex multi-agent workflow for advanced analysis</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-6 max-w-4xl mx-auto">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`flex space-x-3 ${message.role === 'user' ? 'max-w-[80%] flex-row-reverse space-x-reverse' : 'max-w-[95%]'}`}>
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      message.role === 'user' ? 'bg-primary' : 'bg-muted'
                    }`}>
                      {message.role === 'user' ? (
                        <User className="h-4 w-4 text-primary-foreground" />
                      ) : (
                        <Bot className="h-4 w-4" />
                      )}
                    </div>
                    
                    <div className="flex-1 space-y-2 max-w-full">
                      <div className={`rounded-lg px-4 py-3 break-words overflow-hidden ${
                        message.role === 'user' 
                          ? 'bg-primary text-primary-foreground' 
                          : message.error 
                            ? 'bg-destructive/10 border border-destructive/20'
                            : 'bg-assistant-message'
                      }`}>
                        {message.isLoading ? (
                          <div className="flex items-center space-x-2">
                            <LoadingSpinner size="sm" />
                            <span className="text-sm">
                              Processing with {selectedModel === 'langgraph' ? 'LangGraph' : 'Standard'} model...
                            </span>
                          </div>
                        ) : message.error ? (
                          <div className="text-sm text-destructive">{message.content}</div>
                        ) : message.role === 'assistant' && message.response ? (
                          <div className="space-y-3">
                            <SyntaxHighlighter
                              language="markdown"
                              className="code-block-wrapper"
                              customStyle={{
                                background: 'transparent',
                                padding: '0',
                                margin: '0',
                                fontSize: '0.875rem',
                                maxWidth: '100%',
                                overflowWrap: 'break-word',
                                whiteSpace: 'pre-wrap'
                              }}
                              wrapLines={true}
                            >
                              {message.content}
                            </SyntaxHighlighter>
                            
                            {message.response.output && (
                              <div className="flex items-center space-x-4 pt-2 border-t text-xs text-muted-foreground">
                                <span>Quality: {message.response.output.quality_score?.toFixed(2)}</span>
                                <span>Domain: {message.response.output.domain}</span>
                                <span>Iterations: {message.response.output.iterations_used}</span>
                                {message.response.processing_time_seconds && (
                                  <span>Time: {formatDuration(message.response.processing_time_seconds)}</span>
                                )}
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="text-sm whitespace-pre-wrap">{message.content}</div>
                        )}
                      </div>
                      
                      {!message.isLoading && !message.error && (
                        <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                          <span>{message.timestamp.toLocaleTimeString()}</span>
                          {message.model && (
                            <Badge variant="outline" className="text-xs">
                              {message.model === 'langgraph' ? 'LangGraph' : 'Standard'}
                            </Badge>
                          )}
                          {message.role === 'assistant' && (
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleCopyMessage(message.content)}
                              className="h-6 px-2"
                            >
                              <Copy className="h-3 w-3" />
                            </Button>
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
        <div className="border-t px-6 py-4 bg-background">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
            <div className="flex space-x-4">
              <div className="flex-1 relative">
                <Textarea
                  ref={inputRef}
                  value={currentInput}
                  onChange={(e) => setCurrentInput(e.target.value)}
                  placeholder={`Message Prompt Processor (${selectedModel === 'langgraph' ? 'LangGraph' : 'Standard'} Model)...`}
                  className="resize-none pr-12 min-h-[60px] max-h-[200px]"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSubmit();
                    }
                  }}
                  disabled={isLoading}
                />
                <div className="absolute bottom-2 right-2">
                  {canCancel ? (
                    <Button
                      type="button"
                      size="sm"
                      variant="destructive"
                      onClick={handleCancel}
                    >
                      Cancel ({gracePeriodCountdown}s)
                    </Button>
                  ) : (
                    <Button
                      type="submit"
                      size="sm"
                      disabled={isLoading || !currentInput.trim()}
                    >
                      {isLoading ? (
                        <LoadingSpinner size="sm" />
                      ) : (
                        <Send className="h-4 w-4" />
                      )}
                    </Button>
                  )}
                </div>
              </div>
            </div>
            <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
              <span>Press Enter to send, Shift+Enter for new line</span>
              <span>{currentInput.length} characters</span>
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
    </div>
  );
}
