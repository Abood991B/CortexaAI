import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import { Send, Copy, Download, Trash2, Edit, Sparkles, Bot, User, Plus, Menu, Activity, Bell, ChevronDown, Check, HelpCircle } from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
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
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
  const [isAdvancedMode, setIsAdvancedMode] = useState(false);
  const [isUserGuideOpen, setIsUserGuideOpen] = useState(false);
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
      chat_history: chatHistory,
      advanced_mode: isAdvancedMode
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
      <div className={`${showSidebar ? 'w-64' : 'w-20'} transition-all duration-300 border-r border-gray-200 bg-gray-50 flex flex-col items-center py-4`}>
        <div className="flex items-center justify-center w-full px-4 py-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowSidebar(!showSidebar)}
            className="absolute top-4 left-4"
          >
            <Menu className="h-5 w-5" />
          </Button>
          {showSidebar && (
            <div className="flex items-center space-x-3">
              <img src="/Cortexa Logo.png" alt="Cortexa Logo" className="w-8 h-8" />
              <span className="text-lg font-semibold text-gray-900">Cortexa</span>
            </div>
          )}
        </div>

        <div className="mt-6 w-full flex flex-col items-center space-y-4 px-3">
          <Button 
            onClick={createNewSession}
            className={`w-full ${showSidebar ? 'justify-start' : 'justify-center'} bg-white hover:bg-gray-100 text-gray-900 border border-gray-300 shadow-none`}
            variant="outline"
          >
            <Plus className={`${showSidebar ? 'mr-2' : ''} h-4 w-4`} />
            {showSidebar && 'New chat'}
          </Button>
        </div>

        {showSidebar && (
          <div className="flex-1 flex flex-col min-h-0 w-full mt-4">
            <div className="flex-1 overflow-y-auto px-3 space-y-1">
              {sessions.map(session => (
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
                      <div className="absolute right-1 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity flex space-x-1">
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
              ))}
            </div>
          </div>
        )}

        <div className={`w-full px-3 py-4 ${showSidebar ? 'border-t border-gray-200' : ''}`}>
          <NavLink
            to="/system-health"
            className={`flex items-center space-x-2 text-sm text-gray-600 hover:text-gray-900 transition-colors ${showSidebar ? '' : 'justify-center'}`}
          >
            <Activity className="h-4 w-4" />
            {showSidebar && <span>System Health</span>}
          </NavLink>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="border-b px-6 py-4 flex items-center justify-end bg-background">
          <div className="flex items-center space-x-2">
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
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsUserGuideOpen(true)}
              >
                <HelpCircle className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center max-w-4xl mx-auto p-8 rounded-lg bg-gradient-to-br from-background to-muted/20">
                <div className="flex justify-center">
                  <img src="/Cortexa Logo.png" alt="Cortexa Logo" className="w-48 h-48" />
                </div>
                <div className="text-center md:text-left">
                  <h2 className="text-4xl font-bold mb-4 text-primary">Welcome to Cortexa</h2>
                  <p className="text-lg text-muted-foreground mb-6">
                    Unlock the power of AI with our advanced multi-agent system. Craft, refine, and optimize your prompts for exceptional results.
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div 
                      className="p-4 border rounded-lg shadow-sm hover:shadow-md transition-shadow cursor-pointer bg-background"
                      onClick={() => setSelectedModel('standard')}
                    >
                      <div className="flex items-center mb-2">
                        <Sparkles className="h-6 w-6 mr-3 text-primary" />
                        <h3 className="text-lg font-semibold">Standard Model</h3>
                      </div>
                      <p className="text-sm text-muted-foreground">Memory-enhanced optimization for quick, reliable results.</p>
                    </div>
                    <div 
                      className="p-4 border rounded-lg shadow-sm hover:shadow-md transition-shadow cursor-pointer bg-background"
                      onClick={() => setSelectedModel('langgraph')}
                    >
                      <div className="flex items-center mb-2">
                        <Bot className="h-6 w-6 mr-3 text-primary" />
                        <h3 className="text-lg font-semibold">LangGraph Model</h3>
                      </div>
                      <p className="text-sm text-muted-foreground">Complex multi-agent workflow for in-depth analysis.</p>
                    </div>
                  </div>
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
            <div className="relative rounded-lg border bg-background focus-within:ring-2 focus-within:ring-ring">
              <Textarea
                ref={inputRef}
                value={currentInput}
                onChange={(e) => setCurrentInput(e.target.value)}
                placeholder="Message Cortexa..."
                className="resize-none w-full border-0 bg-transparent pt-3 pb-12 pl-3 pr-12 min-h-[60px] max-h-[400px] focus-visible:ring-0 overflow-y-auto"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit();
                  }
                }}
                disabled={isLoading}
                rows={1}
              />
              <div className="absolute bottom-3 left-3 flex items-center space-x-4">
                <div className="relative">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => setIsModelDropdownOpen(!isModelDropdownOpen)}
                    className="flex items-center space-x-1 text-sm font-semibold"
                  >
                    <span>{selectedModel === 'standard' ? 'Standard' : 'LangGraph'}</span>
                    <ChevronDown className="h-4 w-4" />
                  </Button>
                  {isModelDropdownOpen && (
                    <div className="absolute bottom-full mb-2 w-72 bg-background border rounded-lg shadow-lg z-10">
                      <div
                        className="flex items-center justify-between p-3 hover:bg-muted cursor-pointer"
                        onClick={() => {
                          setSelectedModel('standard');
                          setIsModelDropdownOpen(false);
                        }}
                      >
                        <div>
                          <h4 className="font-semibold">Standard</h4>
                          <p className="text-xs text-muted-foreground">Memory-enhanced optimization for quick results.</p>
                        </div>
                        {selectedModel === 'standard' && <Check className="h-4 w-4 text-primary" />}
                      </div>
                      <div
                        className="flex items-center justify-between p-3 hover:bg-muted cursor-pointer"
                        onClick={() => {
                          setSelectedModel('langgraph');
                          setIsModelDropdownOpen(false);
                        }}
                      >
                        <div>
                          <h4 className="font-semibold">LangGraph</h4>
                          <p className="text-xs text-muted-foreground">Complex multi-agent workflow for in-depth analysis.</p>
                        </div>
                        {selectedModel === 'langgraph' && <Check className="h-4 w-4 text-primary" />}
                      </div>
                    </div>
                  )}
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="advanced-mode"
                    checked={isAdvancedMode}
                    onCheckedChange={setIsAdvancedMode}
                  />
                  <Label htmlFor="advanced-mode" className="text-sm">Advanced</Label>
                </div>
              </div>
              <div className="absolute bottom-3 right-3">
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

      {/* User Guide Dialog */}
      <Dialog open={isUserGuideOpen} onOpenChange={setIsUserGuideOpen}>
        <DialogContent className="sm:max-w-[600px]">
          <DialogHeader>
            <DialogTitle>User Guide</DialogTitle>
            <DialogDescription>
              Welcome to Cortexa! Here's a quick guide to get you started.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4 text-sm">
            <div className="grid grid-cols-1 gap-2">
              <h4 className="font-semibold">What is Cortexa?</h4>
              <p>
                Cortexa is an advanced multi-agent system designed to help you craft, refine, and optimize your prompts for exceptional results from AI models.
              </p>
            </div>
            <div className="grid grid-cols-1 gap-2">
              <h4 className="font-semibold">Choosing a Model</h4>
              <ul className="list-disc list-inside space-y-1">
                <li><b>Standard Model:</b> Use this for quick and reliable prompt optimization. It's enhanced with memory to understand the context of your conversation.</li>
                <li><b>LangGraph Model:</b> This model uses a more complex multi-agent workflow for in-depth analysis and prompt improvement.</li>
              </ul>
            </div>
            <div className="grid grid-cols-1 gap-2">
              <h4 className="font-semibold">Advanced Mode</h4>
              <p>
                The "Advanced" mode enables a conversational prompt engineering session. When you enable this mode, the AI agent will ask you a series of clarifying questions to better understand your needs before generating an improved prompt. This is useful when your initial idea is broad or you're not sure how to best phrase your prompt.
              </p>
            </div>
          </div>
          <DialogFooter>
            <Button onClick={() => setIsUserGuideOpen(false)}>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
