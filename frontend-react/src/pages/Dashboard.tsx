import {
  BarChart3, RefreshCw, TrendingUp, Clock, CheckCircle, XCircle,
  Zap, Activity, Sparkles,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { LoadingSpinner } from '@/components/ui/loading';
import { Progress } from '@/components/ui/progress';
import { useStats, useHealth, useHistory } from '@/hooks/useApi';
import { formatDuration } from '@/utils';
import { AppLayout, PageHeader } from '@/components/layout';

export function Dashboard() {
  const { data: stats, isLoading: statsLoading, refetch: refetchStats } = useStats();
  const { data: health } = useHealth();
  const { data: history } = useHistory(10);

  const successRate = stats && stats.total_workflows > 0
    ? (stats.completed_workflows / stats.total_workflows) * 100
    : 0;

  if (statsLoading) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center flex-1">
          <LoadingSpinner size="lg" />
        </div>
      </AppLayout>
    );
  }

  const recentHistory: any[] = Array.isArray(history) ? history.slice(0, 8) : [];

  return (
    <AppLayout>
      <PageHeader
        title="Overview"
        icon={<BarChart3 className="h-5 w-5 text-primary" />}
        actions={
          <Button variant="outline" size="sm" onClick={() => refetchStats()} className="gap-1.5">
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </Button>
        }
      />

      <div className="flex-1 overflow-auto p-6 space-y-6">
        {/* Summary cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="stat-card-blue">
            <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
              <CardDescription className="text-xs font-medium uppercase tracking-wide">Total Prompts</CardDescription>
              <Sparkles className="h-4 w-4 text-blue-500" />
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">{stats?.total_workflows ?? 0}</p>
              <p className="text-xs text-muted-foreground mt-0.5">prompts processed</p>
            </CardContent>
          </Card>

          <Card className="stat-card-green">
            <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
              <CardDescription className="text-xs font-medium uppercase tracking-wide">Success Rate</CardDescription>
              <CheckCircle className="h-4 w-4 text-green-500" />
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">{successRate.toFixed(1)}%</p>
              <p className="text-xs text-muted-foreground mt-0.5">{stats?.completed_workflows ?? 0} succeeded</p>
            </CardContent>
          </Card>

          <Card className="stat-card-purple">
            <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
              <CardDescription className="text-xs font-medium uppercase tracking-wide">Avg Quality</CardDescription>
              <TrendingUp className="h-4 w-4 text-purple-500" />
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">{(stats?.average_quality_score ?? 0).toFixed(2)}</p>
              <p className="text-xs text-muted-foreground mt-0.5">quality score</p>
            </CardContent>
          </Card>

          <Card className="stat-card-amber">
            <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
              <CardDescription className="text-xs font-medium uppercase tracking-wide">Avg Time</CardDescription>
              <Clock className="h-4 w-4 text-amber-500" />
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">{formatDuration(stats?.average_processing_time ?? 0)}</p>
              <p className="text-xs text-muted-foreground mt-0.5">per prompt</p>
            </CardContent>
          </Card>
        </div>

        {/* Success rate progress + domain breakdown */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Activity className="h-5 w-5 text-primary" /> Processing Performance
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
                    <p className="text-lg font-bold">{stats?.completed_workflows ?? 0}</p>
                  </div>
                  <p className="text-xs text-muted-foreground">Succeeded</p>
                </div>
                <div className="space-y-1 text-center">
                  <div className="flex items-center justify-center gap-1.5">
                    <XCircle className="h-3.5 w-3.5 text-red-500" />
                    <p className="text-lg font-bold">{stats?.error_workflows ?? 0}</p>
                  </div>
                  <p className="text-xs text-muted-foreground">Failed</p>
                </div>
                <div className="space-y-1 text-center">
                  <div className="flex items-center justify-center gap-1.5">
                    <Zap className="h-3.5 w-3.5 text-amber-500" />
                    <p className="text-lg font-bold">{stats?.total_workflows ?? 0}</p>
                  </div>
                  <p className="text-xs text-muted-foreground">Total</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <BarChart3 className="h-5 w-5 text-primary" /> Domain Breakdown
              </CardTitle>
              <CardDescription>Prompt categories you've explored</CardDescription>
            </CardHeader>
            <CardContent>
              {stats?.domain_distribution && Object.keys(stats.domain_distribution).length > 0 ? (
                <div className="space-y-3">
                  {Object.entries(stats.domain_distribution)
                    .sort(([, a], [, b]) => (b as number) - (a as number))
                    .slice(0, 6)
                    .map(([domain, count]) => {
                      const total = stats.total_workflows || 1;
                      const pct = ((count as number) / total) * 100;
                      return (
                        <div key={domain} className="space-y-1.5">
                          <div className="flex items-center justify-between text-sm">
                            <span className="font-medium capitalize">{domain}</span>
                            <span className="text-muted-foreground text-xs">{count as number} ({pct.toFixed(0)}%)</span>
                          </div>
                          <Progress value={pct} className="h-1.5" />
                        </div>
                      );
                    })}
                </div>
              ) : (
                <div className="py-8 text-center text-muted-foreground">
                  <BarChart3 className="h-10 w-10 mx-auto mb-3 opacity-30" />
                  <p className="text-sm">Process some prompts to see your domain breakdown.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Recent activity */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Clock className="h-5 w-5 text-primary" /> Recent Activity
            </CardTitle>
            <CardDescription>Your latest prompt processing results</CardDescription>
          </CardHeader>
          <CardContent>
            {recentHistory.length === 0 ? (
              <div className="py-8 text-center text-muted-foreground">
                <Sparkles className="h-10 w-10 mx-auto mb-3 opacity-30" />
                <p className="text-sm">No activity yet. Start by processing a prompt!</p>
              </div>
            ) : (
              <div className="space-y-2">
                {recentHistory.map((item: any, i: number) => (
                  <div key={item.workflow_id || i} className="flex items-center justify-between p-3 rounded-lg border hover:bg-accent/30 transition-colors">
                    <div className="flex items-center gap-3 min-w-0 flex-1">
                      <div className={`h-8 w-8 rounded-lg flex items-center justify-center shrink-0 ${
                        item.status === 'completed' ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'
                      }`}>
                        {item.status === 'completed' ? <CheckCircle className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
                      </div>
                      <div className="min-w-0">
                        <p className="text-sm font-medium truncate">{item.prompt_preview || 'Prompt'}</p>
                        <p className="text-xs text-muted-foreground">
                          {item.domain && <span className="capitalize">{item.domain}</span>}
                          {item.processing_time != null && <span> &middot; {formatDuration(item.processing_time)}</span>}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      {item.quality_score != null && (
                        <Badge variant={item.quality_score >= 0.8 ? 'success' : item.quality_score >= 0.5 ? 'warning' : 'destructive'} className="text-xs">
                          {item.quality_score.toFixed(2)}
                        </Badge>
                      )}
                      <Badge variant={item.status === 'completed' ? 'success' : 'destructive'} className="text-xs capitalize">
                        {item.status}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* System status footer */}
        {health && (
          <div className="flex items-center justify-center gap-4 text-xs text-muted-foreground pb-2">
            <span className="flex items-center gap-1">
              System:
              <Badge variant={health.status === 'healthy' ? 'success' : 'destructive'} className="text-[10px] h-4 px-1.5">
                {health.status}
              </Badge>
            </span>
            <span>Uptime: {formatDuration(health.uptime_seconds || 0)}</span>
            <span>v{health.version || '1.0.0'}</span>
          </div>
        )}
      </div>
    </AppLayout>
  );
}
