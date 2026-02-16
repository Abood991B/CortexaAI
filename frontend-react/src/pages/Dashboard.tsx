import {
  BarChart3, RefreshCw, TrendingUp, Clock, CheckCircle, XCircle,
  Zap, Activity, Sparkles, AlertTriangle, Cpu, Wifi, WifiOff,
  Database, ShieldCheck, ShieldX, Timer, ArrowRight,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { LoadingSpinner } from '@/components/ui/loading';
import { Progress } from '@/components/ui/progress';
import { useStats, useHealth, useHistory, useCacheStats } from '@/hooks/useApi';
import { formatDuration } from '@/utils';
import { AppLayout, PageHeader } from '@/components/layout';
import { useState, useEffect, useMemo, useCallback } from 'react';
import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip,
  LineChart, Line, YAxis,
} from 'recharts';

// ─── Color Palette ───────────────────────────────────────────
const DOMAIN_COLORS = [
  '#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899',
  '#f43f5e', '#f97316', '#eab308', '#22c55e', '#14b8a6',
];

// ─── Circular Gauge Component ────────────────────────────────
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

// ─── Animated Number Component ───────────────────────────────
function AnimatedStat({ value, suffix = '', decimals = 0 }: { value: number; suffix?: string; decimals?: number }) {
  const [displayed, setDisplayed] = useState(0);

  useEffect(() => {
    const duration = 600;
    const start = displayed;
    const diff = value - start;
    if (diff === 0) return;
    const startTime = performance.now();
    let raf: number;

    const step = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplayed(start + diff * eased);
      if (progress < 1) raf = requestAnimationFrame(step);
    };

    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return <>{decimals > 0 ? displayed.toFixed(decimals) : Math.round(displayed)}{suffix}</>;
}

// ─── Custom Recharts Tooltip ─────────────────────────────────
function CustomPieTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const { name, value, percent } = payload[0].payload;
  return (
    <div className="rounded-lg border bg-popover px-3 py-2 text-sm shadow-md">
      <p className="font-medium capitalize">{name}</p>
      <p className="text-muted-foreground">{value} prompts ({(percent * 100).toFixed(0)}%)</p>
    </div>
  );
}



// ─── Main Dashboard Component ────────────────────────────────
export function Dashboard() {
  const { data: stats, isLoading: statsLoading, refetch: refetchStats } = useStats();
  const { data: health, isLoading: healthLoading, refetch: refetchHealth, isFetching } = useHealth();
  const { data: history, refetch: refetchHistory } = useHistory(10);
  const { data: cacheStats, refetch: refetchCache } = useCacheStats();

  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [autoRefresh, setAutoRefresh] = useState(true);

  // ── Auto-refresh timer ──────────────────────────────────────
  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      refetchStats();
      refetchHealth();
      refetchHistory();
      refetchCache();
      setLastUpdated(new Date());
    }, 30000);
    return () => clearInterval(interval);
  }, [autoRefresh, refetchStats, refetchHealth, refetchHistory, refetchCache]);

  const handleManualRefresh = useCallback(() => {
    refetchStats();
    refetchHealth();
    refetchHistory();
    refetchCache();
    setLastUpdated(new Date());
  }, [refetchStats, refetchHealth, refetchHistory, refetchCache]);

  // ── Derived values ──────────────────────────────────────────
  const successRate = stats && stats.total_workflows > 0
    ? (stats.completed_workflows / stats.total_workflows) * 100
    : 0;

  const recentHistory: any[] = useMemo(
    () => (Array.isArray(history) ? history.slice(0, 8) : []),
    [history],
  );

  const cpuPercent = health?.system?.cpu_percent ?? 0;
  const memPercent = health?.system?.memory_percent ?? 0;

  const llmProviders = useMemo(
    () => (health?.components?.llm_providers ? Object.entries(health.components.llm_providers) : []),
    [health],
  );

  // ── Domain PieChart data ────────────────────────────────────
  const domainChartData = useMemo(() => {
    if (!stats?.domain_distribution) return [];
    const total = stats.total_workflows || 1;
    return Object.entries(stats.domain_distribution)
      .sort(([, a], [, b]) => (b as number) - (a as number))
      .slice(0, 8)
      .map(([domain, count]) => ({
        name: domain,
        value: count as number,
        percent: (count as number) / total,
      }));
  }, [stats]);

  // ── Quality trend sparkline data ────────────────────────────
  const qualityTrend = useMemo(() => {
    if (!recentHistory.length) return [];
    return [...recentHistory]
      .reverse()
      .filter(h => h.quality_score != null)
      .map((h, i) => ({ idx: i + 1, score: +(h.quality_score * 100).toFixed(0) }));
  }, [recentHistory]);

  // ── Helpers ─────────────────────────────────────────────────
  const isPositiveStatus = (status?: string | boolean): boolean => {
    if (typeof status === 'boolean') return status;
    if (!status) return false;
    return ['healthy', 'available', 'enabled', 'ok', 'ready'].includes(status.toLowerCase());
  };

  const getStatusIcon = (ok: boolean) =>
    ok ? <CheckCircle className="h-4 w-4 text-green-500" /> : <AlertTriangle className="h-4 w-4 text-red-500" />;

  // ── Loading state ───────────────────────────────────────────
  if (statsLoading || healthLoading) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center flex-1">
          <LoadingSpinner size="lg" />
        </div>
      </AppLayout>
    );
  }

  // ── Render ──────────────────────────────────────────────────
  return (
    <AppLayout>
      <PageHeader
        title="Dashboard"
        icon={<BarChart3 className="h-5 w-5 text-primary" />}
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

        {/* ═══════════════════════════════════════════════════════
            § 1 — Summary Stat Cards (animated numbers)
            ═══════════════════════════════════════════════════════ */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="stat-card-blue group hover:shadow-lg transition-shadow">
            <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
              <CardDescription className="text-xs font-medium uppercase tracking-wide">Total Prompts</CardDescription>
              <Sparkles className="h-4 w-4 text-blue-500 group-hover:scale-110 transition-transform" />
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold tabular-nums">
                <AnimatedStat value={stats?.total_workflows ?? 0} />
              </p>
              <p className="text-xs text-muted-foreground mt-0.5">prompts processed</p>
            </CardContent>
          </Card>

          <Card className="stat-card-green group hover:shadow-lg transition-shadow">
            <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
              <CardDescription className="text-xs font-medium uppercase tracking-wide">Success Rate</CardDescription>
              <CheckCircle className="h-4 w-4 text-green-500 group-hover:scale-110 transition-transform" />
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold tabular-nums">
                <AnimatedStat value={successRate} suffix="%" decimals={1} />
              </p>
              <p className="text-xs text-muted-foreground mt-0.5">
                {stats?.completed_workflows ?? 0} succeeded
              </p>
            </CardContent>
          </Card>

          <Card className="stat-card-purple group hover:shadow-lg transition-shadow">
            <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
              <CardDescription className="text-xs font-medium uppercase tracking-wide">Avg Quality</CardDescription>
              <TrendingUp className="h-4 w-4 text-purple-500 group-hover:scale-110 transition-transform" />
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold tabular-nums">
                <AnimatedStat value={stats?.average_quality_score ?? 0} decimals={2} />
              </p>
              <p className="text-xs text-muted-foreground mt-0.5">quality score</p>
            </CardContent>
          </Card>

          <Card className="stat-card-amber group hover:shadow-lg transition-shadow">
            <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
              <CardDescription className="text-xs font-medium uppercase tracking-wide">Avg Time</CardDescription>
              <Clock className="h-4 w-4 text-amber-500 group-hover:scale-110 transition-transform" />
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold tabular-nums">{formatDuration(stats?.average_processing_time ?? 0)}</p>
              <p className="text-xs text-muted-foreground mt-0.5">per prompt</p>
            </CardContent>
          </Card>
        </div>

        {/* ═══════════════════════════════════════════════════════
            § 2 — Resource Utilization + System Status
            ═══════════════════════════════════════════════════════ */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Resource Utilization */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Cpu className="h-5 w-5 text-primary" /> Resource Utilization
              </CardTitle>
              <CardDescription>CPU, memory, and system resource metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-center gap-12 py-2">
                <CircularGauge value={cpuPercent} label="CPU" color={cpuPercent > 80 ? '#ef4444' : cpuPercent > 50 ? '#f59e0b' : '#22c55e'} />
                <CircularGauge value={memPercent} label="Memory" color={memPercent > 80 ? '#ef4444' : memPercent > 50 ? '#f59e0b' : '#22c55e'} />
              </div>
            </CardContent>
          </Card>

          {/* System Status */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Activity className="h-5 w-5 text-primary" /> System Status
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {[
                { label: 'System', ok: health?.status === 'healthy', subtext: health?.status === 'healthy' ? `Uptime: ${formatDuration(health?.uptime_seconds || 0)}` : 'System unhealthy' },
                { label: 'Readiness', ok: !!health?.readiness, subtext: health?.readiness ? 'Ready to serve requests' : 'Not ready' },
                { label: 'Coordinator', ok: isPositiveStatus(health?.components?.coordinator?.status), subtext: health?.components?.coordinator?.error || `${health?.components?.coordinator?.available_domains ?? 0} domains` },
                { label: 'LangSmith', ok: isPositiveStatus(health?.components?.langsmith?.status) || !!health?.components?.langsmith?.enabled, subtext: health?.components?.langsmith?.status || 'disabled' },
              ].map(({ label, ok, subtext }) => (
                <div key={label} className="flex items-center justify-between p-3 rounded-lg border hover:bg-accent/20 transition-colors">
                  <div className="flex items-center gap-3">
                    {getStatusIcon(ok)}
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
        </div>

        {/* ═══════════════════════════════════════════════════════
            § 3 — Processing Performance + Domain PieChart
            ═══════════════════════════════════════════════════════ */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Processing Performance */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Zap className="h-5 w-5 text-primary" /> Processing Performance
              </CardTitle>
              <CardDescription>How your prompts are performing</CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium">Success Rate</span>
                  <span className="text-muted-foreground">{successRate.toFixed(1)}%</span>
                </div>
                <Progress value={successRate} className="h-2.5" />
              </div>
              <div className="grid grid-cols-3 gap-4 pt-2">
                <div className="space-y-1 text-center">
                  <div className="flex items-center justify-center gap-1.5">
                    <CheckCircle className="h-3.5 w-3.5 text-green-500" />
                    <p className="text-lg font-bold tabular-nums">
                      <AnimatedStat value={stats?.completed_workflows ?? 0} />
                    </p>
                  </div>
                  <p className="text-xs text-muted-foreground">Succeeded</p>
                </div>
                <div className="space-y-1 text-center">
                  <div className="flex items-center justify-center gap-1.5">
                    <XCircle className="h-3.5 w-3.5 text-red-500" />
                    <p className="text-lg font-bold tabular-nums">
                      <AnimatedStat value={stats?.error_workflows ?? 0} />
                    </p>
                  </div>
                  <p className="text-xs text-muted-foreground">Failed</p>
                </div>
                <div className="space-y-1 text-center">
                  <div className="flex items-center justify-center gap-1.5">
                    <Zap className="h-3.5 w-3.5 text-amber-500" />
                    <p className="text-lg font-bold tabular-nums">
                      <AnimatedStat value={stats?.total_workflows ?? 0} />
                    </p>
                  </div>
                  <p className="text-xs text-muted-foreground">Total</p>
                </div>
              </div>

              {/* Quality trend sparkline */}
              {qualityTrend.length >= 2 && (
                <div className="pt-3 border-t">
                  <p className="text-xs font-medium text-muted-foreground mb-2">Quality Trend (recent)</p>
                  <ResponsiveContainer width="100%" height={60}>
                    <LineChart data={qualityTrend}>
                      <Line
                        type="monotone"
                        dataKey="score"
                        stroke="#8b5cf6"
                        strokeWidth={2}
                        dot={false}
                      />
                      <YAxis domain={[0, 100]} hide />
                      <Tooltip
                        content={({ active, payload }: any) =>
                          active && payload?.length ? (
                            <div className="rounded-lg border bg-popover px-2 py-1 text-xs shadow-md">
                              Quality: {payload[0].value}%
                            </div>
                          ) : null
                        }
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Domain Breakdown — PieChart */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <BarChart3 className="h-5 w-5 text-primary" /> Domain Breakdown
              </CardTitle>
              <CardDescription>Prompt categories you've explored</CardDescription>
            </CardHeader>
            <CardContent>
              {domainChartData.length > 0 ? (
                <div className="flex flex-col items-center gap-4">
                  <ResponsiveContainer width="100%" height={200}>
                    <PieChart>
                      <Pie
                        data={domainChartData}
                        cx="50%"
                        cy="50%"
                        innerRadius={50}
                        outerRadius={80}
                        paddingAngle={3}
                        dataKey="value"
                        nameKey="name"
                        animationDuration={800}
                      >
                        {domainChartData.map((_, index) => (
                          <Cell key={`cell-${index}`} fill={DOMAIN_COLORS[index % DOMAIN_COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip content={<CustomPieTooltip />} />
                    </PieChart>
                  </ResponsiveContainer>

                  {/* Legend */}
                  <div className="flex flex-wrap justify-center gap-x-4 gap-y-1.5">
                    {domainChartData.map((entry, i) => (
                      <div key={entry.name} className="flex items-center gap-1.5 text-xs">
                        <span
                          className="inline-block h-2.5 w-2.5 rounded-full shrink-0"
                          style={{ backgroundColor: DOMAIN_COLORS[i % DOMAIN_COLORS.length] }}
                        />
                        <span className="capitalize text-muted-foreground">{entry.name}</span>
                        <span className="font-medium">{entry.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="py-10 text-center text-muted-foreground">
                  <BarChart3 className="h-12 w-12 mx-auto mb-3 opacity-20" />
                  <p className="text-sm font-medium mb-1">No domain data yet</p>
                  <p className="text-xs mb-4">Process some prompts to see your domain breakdown.</p>
                  <Button variant="outline" size="sm" className="gap-1.5" asChild>
                    <a href="/process">
                      Get Started <ArrowRight className="h-3.5 w-3.5" />
                    </a>
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* ═══════════════════════════════════════════════════════
            § 4 — Cache Performance + LLM Providers
            ═══════════════════════════════════════════════════════ */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Cache Performance */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Database className="h-5 w-5 text-primary" /> Cache Performance
              </CardTitle>
              <CardDescription>
                Backend: <span className="font-medium">{cacheStats?.backend ?? 'N/A'}</span>
              </CardDescription>
            </CardHeader>
            <CardContent>
              {cacheStats ? (
                <div className="space-y-4">
                  {/* Hit rate gauge */}
                  <div className="flex items-center justify-center py-2">
                    <CircularGauge
                      value={(cacheStats.hit_rate ?? 0) * 100}
                      label="Hit Rate"
                      color={(cacheStats.hit_rate ?? 0) >= 0.7 ? '#22c55e' : (cacheStats.hit_rate ?? 0) >= 0.4 ? '#f59e0b' : '#ef4444'}
                    />
                  </div>
                  {/* Key metrics */}
                  <div className="grid grid-cols-3 gap-3 text-center">
                    <div className="rounded-lg border p-3">
                      <p className="text-lg font-bold tabular-nums text-green-600">
                        <AnimatedStat value={cacheStats.hit_count ?? 0} />
                      </p>
                      <p className="text-xs text-muted-foreground">Hits</p>
                    </div>
                    <div className="rounded-lg border p-3">
                      <p className="text-lg font-bold tabular-nums text-red-500">
                        <AnimatedStat value={cacheStats.miss_count ?? 0} />
                      </p>
                      <p className="text-xs text-muted-foreground">Misses</p>
                    </div>
                    <div className="rounded-lg border p-3">
                      <p className="text-lg font-bold tabular-nums">
                        <AnimatedStat value={cacheStats.total_entries ?? 0} />
                      </p>
                      <p className="text-xs text-muted-foreground">Entries</p>
                    </div>
                  </div>
                  <p className="text-xs text-center text-muted-foreground">
                    {cacheStats.total_requests ?? 0} total requests
                  </p>
                </div>
              ) : (
                <div className="py-10 text-center text-muted-foreground">
                  <Database className="h-12 w-12 mx-auto mb-3 opacity-20" />
                  <p className="text-sm font-medium mb-1">Cache stats unavailable</p>
                  <p className="text-xs">The cache statistics endpoint may not be reachable.</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* LLM Providers */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Zap className="h-5 w-5 text-primary" /> LLM Providers
              </CardTitle>
              <CardDescription>Connected language model providers</CardDescription>
            </CardHeader>
            <CardContent>
              {llmProviders.length > 0 ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {llmProviders.map(([name, info]: [string, any]) => {
                    const ok = isPositiveStatus(info.status);
                    const verified = info.verified === true;
                    const latency = info.latency_ms;
                    const model = info.model;
                    const error = info.error;

                    return (
                      <div
                        key={name}
                        className="flex flex-col gap-2 p-3 rounded-lg border hover:bg-accent/20 transition-colors"
                      >
                        {/* Row 1: Icon + name + status badge */}
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2.5">
                            {ok
                              ? <Wifi className="h-4 w-4 text-green-500 shrink-0" />
                              : <WifiOff className="h-4 w-4 text-red-500 shrink-0" />}
                            <span className="font-medium text-sm capitalize truncate">{name}</span>
                          </div>
                          <Badge variant={ok ? 'success' : 'destructive'} className="text-xs capitalize shrink-0">
                            {info.status || 'unknown'}
                          </Badge>
                        </div>

                        {/* Row 2: Metadata chips */}
                        <div className="flex flex-wrap items-center gap-1.5 pl-6">
                          {/* Verification badge */}
                          {verified ? (
                            <Badge variant="outline" className="gap-1 text-[10px] h-5 px-1.5 border-green-300 text-green-700">
                              <ShieldCheck className="h-3 w-3" /> Verified
                            </Badge>
                          ) : info.configured && ok ? (
                            <Badge variant="outline" className="gap-1 text-[10px] h-5 px-1.5 border-amber-300 text-amber-700">
                              <ShieldX className="h-3 w-3" /> Unverified
                            </Badge>
                          ) : null}

                          {/* Model name */}
                          {model && (
                            <Badge variant="outline" className="text-[10px] h-5 px-1.5 font-mono">
                              {model}
                            </Badge>
                          )}

                          {/* Latency */}
                          {latency != null && (
                            <Badge
                              variant="outline"
                              className={`gap-1 text-[10px] h-5 px-1.5 ${
                                latency < 500 ? 'border-green-300 text-green-700' :
                                latency < 2000 ? 'border-amber-300 text-amber-700' :
                                'border-red-300 text-red-700'
                              }`}
                            >
                              <Timer className="h-3 w-3" /> {latency.toFixed(0)}ms
                            </Badge>
                          )}

                          {/* Configured chip */}
                          {!info.configured && (
                            <Badge variant="outline" className="text-[10px] h-5 px-1.5 text-muted-foreground">
                              Not configured
                            </Badge>
                          )}
                        </div>

                        {/* Error display */}
                        {error && (
                          <p className="text-[11px] text-red-500 pl-6 leading-tight truncate" title={error}>
                            <AlertTriangle className="inline h-3 w-3 mr-1 -mt-0.5" />
                            {error}
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="py-10 text-center text-muted-foreground">
                  <Wifi className="h-12 w-12 mx-auto mb-3 opacity-20" />
                  <p className="text-sm font-medium mb-1">No providers detected</p>
                  <p className="text-xs">Configure LLM providers in your environment.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* ═══════════════════════════════════════════════════════
            § 5 — Footer: Last Updated
            ═══════════════════════════════════════════════════════ */}
        <div className="flex items-center justify-center gap-3 pb-4">
          {health && (
            <Badge variant={health.status === 'healthy' ? 'success' : 'destructive'} className="text-xs gap-1">
              <Activity className="h-3 w-3" /> {health.status}
            </Badge>
          )}
          <span className="text-sm text-muted-foreground font-medium">
            Last updated: {lastUpdated.toLocaleTimeString()}
          </span>
          {autoRefresh && (
            <Badge variant="outline" className="text-[10px] h-5 px-1.5">
              <RefreshCw className="h-3 w-3 mr-1 animate-spin-slow" /> Auto-refresh 30s
            </Badge>
          )}
          {health?.version && (
            <span className="text-xs text-muted-foreground">v{health.version}</span>
          )}
        </div>
      </div>
    </AppLayout>
  );
}
