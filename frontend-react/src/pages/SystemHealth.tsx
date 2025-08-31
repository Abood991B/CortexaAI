import React from 'react';
import { Activity, Server, Database, Wifi, AlertTriangle, CheckCircle } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { LoadingSpinner } from '@/components/ui/loading';
import { useHealth } from '@/hooks/useApi';
import { formatDuration } from '@/lib/utils';

export function SystemHealth() {
  const { data: health, isLoading } = useHealth();

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'available':
      case 'enabled':
        return 'text-green-600';
      case 'unhealthy':
      case 'not_configured':
      case 'disabled':
        return 'text-red-600';
      default:
        return 'text-yellow-600';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'available':
      case 'enabled':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'unhealthy':
      case 'not_configured':
      case 'disabled':
        return <AlertTriangle className="h-4 w-4 text-red-600" />;
      default:
        return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold flex items-center">
          <Activity className="mr-3 h-8 w-8 text-primary" />
          System Health
        </h1>
        <p className="text-muted-foreground">
          Monitor system status and component health
        </p>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-12">
          <LoadingSpinner size="lg" />
        </div>
      ) : health ? (
        <div className="space-y-6">
          {/* Overall Status */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Overall System Status</span>
                <Badge variant={health.status === 'healthy' ? 'default' : 'destructive'}>
                  {health.status}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <p className="text-2xl font-bold">{health.version}</p>
                  <p className="text-sm text-muted-foreground">Version</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold">{formatDuration(health.uptime_seconds)}</p>
                  <p className="text-sm text-muted-foreground">Uptime</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold">{health.readiness ? 'Ready' : 'Not Ready'}</p>
                  <p className="text-sm text-muted-foreground">Readiness</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold">{health.liveness ? 'Live' : 'Down'}</p>
                  <p className="text-sm text-muted-foreground">Liveness</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* LLM Providers */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Server className="mr-2 h-5 w-5" />
                LLM Providers
              </CardTitle>
              <CardDescription>Status of configured language model providers</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {Object.entries(health.components.llm_providers).map(([provider, info]) => (
                  <div key={provider} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      {getStatusIcon(info.status)}
                      <div>
                        <p className="font-medium capitalize">{provider}</p>
                        <p className="text-sm text-muted-foreground">
                          {info.configured ? 'Configured' : 'Not configured'}
                        </p>
                      </div>
                    </div>
                    <Badge variant={info.status === 'available' ? 'default' : 'secondary'}>
                      {info.status}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* System Components */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* LangSmith */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Database className="mr-2 h-5 w-5" />
                  LangSmith
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    {getStatusIcon(health.components.langsmith.status)}
                    <span>{health.components.langsmith.enabled ? 'Enabled' : 'Disabled'}</span>
                  </div>
                  <Badge variant={health.components.langsmith.enabled ? 'default' : 'secondary'}>
                    {health.components.langsmith.status}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            {/* Coordinator */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Wifi className="mr-2 h-5 w-5" />
                  Coordinator
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    {getStatusIcon(health.components.coordinator.status)}
                    <span>
                      {health.components.coordinator.available_domains || 0} domains available
                    </span>
                  </div>
                  <Badge variant={health.components.coordinator.status === 'healthy' ? 'default' : 'destructive'}>
                    {health.components.coordinator.status}
                  </Badge>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* System Metrics */}
          <Card>
            <CardHeader>
              <CardTitle>System Metrics</CardTitle>
              <CardDescription>Current system performance and usage</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-3 border rounded-lg">
                  <p className="text-2xl font-bold">{health.metrics.total_workflows}</p>
                  <p className="text-sm text-muted-foreground">Total Workflows</p>
                </div>
                <div className="text-center p-3 border rounded-lg">
                  <p className="text-2xl font-bold text-green-600">{health.metrics.successful_workflows}</p>
                  <p className="text-sm text-muted-foreground">Successful</p>
                </div>
                <div className="text-center p-3 border rounded-lg">
                  <p className="text-2xl font-bold text-red-600">{health.metrics.failed_workflows}</p>
                  <p className="text-sm text-muted-foreground">Failed</p>
                </div>
                <div className="text-center p-3 border rounded-lg">
                  <p className="text-2xl font-bold">{health.metrics.llm_calls_total}</p>
                  <p className="text-sm text-muted-foreground">LLM Calls</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* System Resources */}
          {health.system && (
            <Card>
              <CardHeader>
                <CardTitle>System Resources</CardTitle>
                <CardDescription>Current resource utilization</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-3 border rounded-lg">
                    <p className="text-2xl font-bold">{health.system.memory_percent?.toFixed(1)}%</p>
                    <p className="text-sm text-muted-foreground">Memory Usage</p>
                  </div>
                  <div className="text-center p-3 border rounded-lg">
                    <p className="text-2xl font-bold">{health.system.cpu_percent?.toFixed(1)}%</p>
                    <p className="text-sm text-muted-foreground">CPU Usage</p>
                  </div>
                  <div className="text-center p-3 border rounded-lg">
                    <p className="text-2xl font-bold">{health.system.active_connections || 0}</p>
                    <p className="text-sm text-muted-foreground">Active Connections</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      ) : (
        <div className="text-center py-12">
          <AlertTriangle className="h-12 w-12 mx-auto mb-4 text-red-500" />
          <h3 className="text-lg font-medium mb-2">Unable to fetch system health</h3>
          <p className="text-muted-foreground">
            The system health endpoint is not responding
          </p>
        </div>
      )}
    </div>
  );
}
