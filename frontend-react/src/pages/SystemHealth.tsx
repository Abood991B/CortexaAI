import { Activity, Server, AlertTriangle, CheckCircle, RefreshCw, Cpu, Bot, Bell, Menu } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { LoadingSpinner } from '@/components/ui/loading';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { useHealth, useStats } from '@/hooks/useApi';
import { useNotifications } from '@/hooks/useNotifications';
import { NotificationsDropdown } from '@/components/ui/notifications';
import { formatDuration } from '@/utils';
import { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';

export function SystemHealth() {
  const location = useLocation();
  const { data: health, isLoading, refetch: refetchHealth, isFetching } = useHealth();
  const { data: stats, refetch: refetchStats } = useStats();
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [autoRefresh, setAutoRefresh] = useState(true);
  const {
    notifications,
    unreadCount,
    isOpen: isDropdownOpen,
    toggle: toggleDropdown,
    markAsRead,
    markAllAsRead,
    removeNotification,
    clearAll,
  } = useNotifications();

  // Auto-refresh every 30 seconds
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      refetchHealth();
      refetchStats();
      setLastUpdated(new Date());
    }, 30000);

    return () => clearInterval(interval);
  }, [autoRefresh, refetchHealth, refetchStats]);

  const handleManualRefresh = () => {
    refetchHealth();
    refetchStats();
    setLastUpdated(new Date());
  };

  const getSuccessRate = () => {
    if (!stats) return 0;
    const total = stats.total_workflows;
    const successful = stats.completed_workflows;
    return total > 0 ? (successful / total) * 100 : 0;
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'available':
      case 'enabled':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'degraded':
      case 'warning':
        return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      default:
        return <AlertTriangle className="h-4 w-4 text-red-600" />;
    }
  };

  const [showSidebar, setShowSidebar] = useState(true);

  if (isLoading) {
    return (
      <div className="flex h-screen bg-background">
        <div className="flex items-center justify-center w-full">
          <LoadingSpinner size="lg" />
        </div>
      </div>
    );
  }

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
          <NavLink
            to="/"
            className={`w-full flex items-center ${showSidebar ? 'justify-start' : 'justify-center'} px-3 py-2 text-sm rounded-md transition-colors ${
              location.pathname === '/' || location.pathname === '/processor'
                ? 'bg-gray-200 text-gray-900'
                : 'text-gray-600 hover:bg-gray-200 hover:text-gray-900'
            }`}
          >
            <Bot className={`${showSidebar ? 'mr-3' : ''} h-4 w-4`} />
            {showSidebar && <span>Prompt Processor</span>}
          </NavLink>
          
          <NavLink
            to="/system-health"
            className={`w-full flex items-center ${showSidebar ? 'justify-start' : 'justify-center'} px-3 py-2 text-sm rounded-md transition-colors ${
              location.pathname === '/system-health'
                ? 'bg-gray-200 text-gray-900'
                : 'text-gray-600 hover:bg-gray-200 hover:text-gray-900'
            }`}
          >
            <Activity className={`${showSidebar ? 'mr-3' : ''} h-4 w-4`} />
            {showSidebar && <span>System Health</span>}
          </NavLink>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="border-b px-6 py-4 bg-background">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Activity className="h-5 w-5 text-primary" />
                <h1 className="text-xl font-semibold">System Health</h1>
              </div>
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
              
              <Button
                variant="outline"
                size="sm"
                onClick={() => setAutoRefresh(!autoRefresh)}
              >
                {autoRefresh ? 'Disable Auto-refresh' : 'Enable Auto-refresh'}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleManualRefresh}
                disabled={isFetching}
              >
                {isFetching ? <LoadingSpinner size="sm" /> : <RefreshCw className="h-4 w-4" />}
                Refresh
              </Button>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          <div className="space-y-6">
            {/* System Overview */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Server className="h-5 w-5" />
                  System Overview
                </CardTitle>
                <CardDescription>
                  Overall system status and key metrics
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <Badge variant={health?.status === 'healthy' ? 'default' : 'destructive'}>
                        {health?.status?.toUpperCase() || 'UNKNOWN'}
                      </Badge>
                      {health?.readiness && health?.liveness && (
                        <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                          OPERATIONAL
                        </Badge>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground">System Status</p>
                  </div>
                  <div className="space-y-2">
                    <p className="text-2xl font-bold">{formatDuration(health?.uptime_seconds || 0)}</p>
                    <p className="text-sm text-muted-foreground">Uptime</p>
                  </div>
                  <div className="space-y-2">
                    <p className="text-2xl font-bold">{stats?.total_workflows || 0}</p>
                    <p className="text-sm text-muted-foreground">Total Prompts</p>
                  </div>
                  <div className="space-y-2">
                    <p className="text-2xl font-bold">{getSuccessRate().toFixed(1)}%</p>
                    <p className="text-sm text-muted-foreground">Success Rate</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Health Checks */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle className="h-5 w-5" />
                  Health Checks
                </CardTitle>
                <CardDescription>
                  System component health status
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-3">
                      {getStatusIcon(health?.readiness ? 'healthy' : 'unhealthy')}
                      <span className="font-medium">Readiness</span>
                    </div>
                    <Badge variant={health?.readiness ? 'default' : 'destructive'}>
                      {health?.readiness ? 'Ready' : 'Not Ready'}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-3">
                      {getStatusIcon(health?.liveness ? 'healthy' : 'unhealthy')}
                      <span className="font-medium">Liveness</span>
                    </div>
                    <Badge variant={health?.liveness ? 'default' : 'destructive'}>
                      {health?.liveness ? 'Alive' : 'Dead'}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Performance Metrics */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Cpu className="h-5 w-5" />
                  Performance Metrics
                </CardTitle>
                <CardDescription>
                  System performance and resource utilization
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Success Rate</span>
                      <span className="text-sm text-muted-foreground">{getSuccessRate().toFixed(1)}%</span>
                    </div>
                    <Progress value={getSuccessRate()} className="h-2" />
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Total Workflows</span>
                      <span className="text-sm text-muted-foreground">{stats?.total_workflows || 0}</span>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Failed Workflows</span>
                      <span className="text-sm text-muted-foreground">{stats?.error_workflows || 0}</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>



            {/* Last Updated */}
            <div className="text-center text-sm text-muted-foreground">
              Last updated: {lastUpdated.toLocaleTimeString()}
              {autoRefresh && " • Auto-refresh enabled"}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
