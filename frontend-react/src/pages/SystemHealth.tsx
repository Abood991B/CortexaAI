import {
  Activity, AlertTriangle, CheckCircle, RefreshCw, Cpu, Zap,
  Clock, BarChart3, Wifi, WifiOff,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { LoadingSpinner } from '@/components/ui/loading';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { useHealth, useStats } from '@/hooks/useApi';
import { formatDuration } from '@/utils';
import { useState, useEffect } from 'react';
import { AppLayout, PageHeader } from '@/components/layout';

/* Simple circular gauge */
function CircularGauge({ value, label, color }: { value: number; label: string; color: string }) {
  const clamped = Math.min(100, Math.max(0, value));
  const radius = 36;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (clamped / 100) * circumference;

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative h-24 w-24">
        <svg className="gauge-ring h-full w-full" viewBox="0 0 80 80">
          <circle cx="40" cy="40" r={radius} fill="none" stroke="hsl(var(--border))" strokeWidth="7" />
          <circle
            cx="40" cy="40" r={radius} fill="none"
            stroke={color}
            strokeWidth="7"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className="transition-all duration-700 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-lg font-bold">{clamped.toFixed(0)}%</span>
        </div>
      </div>
      <span className="text-xs text-muted-foreground font-medium">{label}</span>
    </div>
  );
}

export function SystemHealth() {
  const { data: health, isLoading, refetch: refetchHealth, isFetching } = useHealth();
  const { data: stats, refetch: refetchStats } = useStats();
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [autoRefresh, setAutoRefresh] = useState(true);

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
    return stats.total_workflows > 0
      ? (stats.completed_workflows / stats.total_workflows) * 100
      : 0;
  };

  const isPositiveStatus = (status?: string | boolean): boolean => {
    if (typeof status === 'boolean') return status;
    if (!status) return false;
    return ['healthy', 'available', 'enabled', 'ok', 'ready'].includes(status.toLowerCase());
  };

  const getStatusIcon = (status: string) => {
    if (isPositiveStatus(status)) return <CheckCircle className="h-4 w-4 text-green-500" />;
    if (['degraded', 'warning'].includes(status)) return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
    return <AlertTriangle className="h-4 w-4 text-red-500" />;
  };

  if (isLoading) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center flex-1">
          <LoadingSpinner size="lg" />
        </div>
      </AppLayout>
    );
  }

  const cpuPercent = health?.system?.cpu_percent ?? 0;
  const memPercent = health?.system?.memory_percent ?? 0;
  const activeConns = health?.system?.active_connections ?? 0;

  const llmProviders = health?.components?.llm_providers
    ? Object.entries(health.components.llm_providers)
    : [];

  return (
    <AppLayout>
      <PageHeader
        title="System Health"
        icon={<Activity className="h-5 w-5 text-primary" />}
        actions={
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={() => setAutoRefresh(!autoRefresh)}>
              {autoRefresh ? 'Pause' : 'Resume'} Auto-refresh
            </Button>
            <Button variant="outline" size="sm" onClick={handleManualRefresh} disabled={isFetching} className="gap-1.5">
              {isFetching ? <LoadingSpinner size="sm" /> : <RefreshCw className="h-3.5 w-3.5" />}
              Refresh
            </Button>
          </div>
        }
      />

      <div className="flex-1 overflow-auto p-6 space-y-6">
        {/* Top row: Status + Uptime + Success Rate + Connections */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="stat-card-green">
            <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
              <CardDescription className="text-xs font-medium uppercase tracking-wide">Status</CardDescription>
              {health?.status === 'healthy'
                ? <CheckCircle className="h-4 w-4 text-green-500" />
                : <AlertTriangle className="h-4 w-4 text-red-500" />}
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                <Badge variant={health?.status === 'healthy' ? 'success' : 'destructive'} className="text-xs">
                  {health?.status?.toUpperCase() || 'UNKNOWN'}
                </Badge>
                {health?.readiness && health?.liveness && (
                  <Badge variant="outline" className="text-xs bg-green-50 text-green-700 border-green-200">OPERATIONAL</Badge>
                )}
              </div>
            </CardContent>
          </Card>

          <Card className="stat-card-blue">
            <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
              <CardDescription className="text-xs font-medium uppercase tracking-wide">Uptime</CardDescription>
              <Clock className="h-4 w-4 text-blue-500" />
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">{formatDuration(health?.uptime_seconds || 0)}</p>
            </CardContent>
          </Card>

          <Card className="stat-card-purple">
            <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
              <CardDescription className="text-xs font-medium uppercase tracking-wide">Success Rate</CardDescription>
              <BarChart3 className="h-4 w-4 text-purple-500" />
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">{getSuccessRate().toFixed(1)}%</p>
              <p className="text-xs text-muted-foreground">{stats?.completed_workflows ?? 0} / {stats?.total_workflows ?? 0} workflows</p>
            </CardContent>
          </Card>

          <Card className="stat-card-amber">
            <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
              <CardDescription className="text-xs font-medium uppercase tracking-wide">Connections</CardDescription>
              <Wifi className="h-4 w-4 text-amber-500" />
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">{activeConns}</p>
              <p className="text-xs text-muted-foreground">active connections</p>
            </CardContent>
          </Card>
        </div>

        {/* Resource usage gauges */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base"><Cpu className="h-5 w-5 text-primary" /> Resource Utilization</CardTitle>
            <CardDescription>CPU, memory, and system resource metrics</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center gap-12 py-2">
              <CircularGauge value={cpuPercent} label="CPU" color={cpuPercent > 80 ? '#ef4444' : cpuPercent > 50 ? '#f59e0b' : '#22c55e'} />
              <CircularGauge value={memPercent} label="Memory" color={memPercent > 80 ? '#ef4444' : memPercent > 50 ? '#f59e0b' : '#22c55e'} />
            </div>
          </CardContent>
        </Card>

        {/* Health checks */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base"><CheckCircle className="h-5 w-5 text-primary" /> Health Checks</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {[
                { label: 'Readiness', ok: health?.readiness, subtext: health?.readiness ? 'Ready to serve requests' : 'Not ready' },
                { label: 'Liveness', ok: health?.liveness, subtext: health?.liveness ? 'Application is alive' : 'Application is down' },
                { label: 'Coordinator', ok: isPositiveStatus(health?.components?.coordinator?.status), subtext: health?.components?.coordinator?.error || `${health?.components?.coordinator?.available_domains ?? 0} domains` },
                { label: 'LangSmith', ok: isPositiveStatus(health?.components?.langsmith?.status) || !!health?.components?.langsmith?.enabled, subtext: health?.components?.langsmith?.status || 'disabled' },
              ].map(({ label, ok, subtext }) => (
                <div key={label} className="flex items-center justify-between p-3 rounded-lg border">
                  <div className="flex items-center gap-3">
                    {getStatusIcon(ok ? 'healthy' : 'unhealthy')}
                    <div>
                      <span className="font-medium text-sm">{label}</span>
                      <p className="text-xs text-muted-foreground">{subtext}</p>
                    </div>
                  </div>
                  <Badge variant={ok ? 'success' : 'destructive'} className="text-xs">
                    {ok ? 'OK' : 'DOWN'}
                  </Badge>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* LLM Providers */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base"><Zap className="h-5 w-5 text-primary" /> LLM Providers</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {llmProviders.length === 0 ? (
                <p className="text-sm text-muted-foreground">No LLM providers detected.</p>
              ) : (
                  llmProviders.map(([name, info]: [string, any]) => {
                    const ok = isPositiveStatus(info.status);
                    return (
                  <div key={name} className="flex items-center justify-between p-3 rounded-lg border">
                    <div className="flex items-center gap-3">
                      {ok
                        ? <Wifi className="h-4 w-4 text-green-500" />
                        : <WifiOff className="h-4 w-4 text-red-500" />}
                      <div>
                        <span className="font-medium text-sm capitalize">{name}</span>
                        <p className="text-xs text-muted-foreground">{info.configured ? 'Configured' : 'Not configured'}</p>
                      </div>
                    </div>
                    <Badge variant={ok ? 'success' : 'destructive'} className="text-xs capitalize">
                      {info.status || 'unknown'}
                    </Badge>
                  </div>
                    );
                  })
              )}
            </CardContent>
          </Card>
        </div>

        {/* Workflow metrics */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base"><BarChart3 className="h-5 w-5 text-primary" /> Workflow Metrics</CardTitle>
            <CardDescription>Overall processing performance</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
              {[
                { label: 'Total Workflows', value: health?.metrics?.total_workflows ?? stats?.total_workflows ?? 0 },
                { label: 'Successful', value: health?.metrics?.successful_workflows ?? stats?.completed_workflows ?? 0 },
                { label: 'Failed', value: health?.metrics?.failed_workflows ?? stats?.error_workflows ?? 0 },
                { label: 'LLM Calls', value: health?.metrics?.llm_calls_total ?? 0 },
                { label: 'Retries', value: health?.metrics?.retry_attempts ?? 0 },
              ].map(({ label, value }) => (
                <div key={label} className="space-y-1">
                  <p className="text-xs text-muted-foreground uppercase tracking-wide">{label}</p>
                  <p className="text-xl font-bold">{value}</p>
                </div>
              ))}
            </div>
            <div className="mt-6 space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium">Success Rate</span>
                <span className="text-muted-foreground">{getSuccessRate().toFixed(1)}%</span>
              </div>
              <Progress value={getSuccessRate()} className="h-2" />
            </div>
          </CardContent>
        </Card>

        {/* Footer */}
        <p className="text-center text-xs text-muted-foreground pb-2">
          Last updated: {lastUpdated.toLocaleTimeString()}
          {autoRefresh && ' \u00B7 Auto-refresh every 30s'}
        </p>
      </div>
    </AppLayout>
  );
}
