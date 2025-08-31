import { useState } from 'react';
import { 
  BarChart3, 
  TrendingUp, 
  PieChart, 
  Calendar,
  Download
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { LoadingSpinner } from '@/components/ui/loading';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell,
  Pie
} from 'recharts';
import { useAnalytics } from '@/hooks/useApi';

import { formatNumber, formatPercentage } from '@/lib/utils';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

export function Analytics() {
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d'>('30d');
  const { data: analytics, isLoading, error } = useAnalytics(timeRange);

  // Transform data for charts
  const domainData = analytics?.domain_distribution ? 
    Object.entries(analytics.domain_distribution).map(([name, count]) => ({
      name,
      value: Number(count)
    })) : [];

  const qualityTrends = analytics?.daily_stats?.map((day: { date: string; avg_quality_score?: number }) => ({
    date: day.date,
    avg_quality: day.avg_quality_score || 0
  })) || [];

  const exportData = () => {
    if (analytics) {
      const content = JSON.stringify(analytics, null, 2);
      const blob = new Blob([content], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `analytics_${timeRange}_${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <div className="text-destructive mb-4">
          Failed to load analytics data. Please try again later.
        </div>
        <Button 
          variant="outline" 
          onClick={() => window.location.reload()}
        >
          Retry
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold">Analytics Dashboard</h1>
          <p className="text-muted-foreground mt-1">
            Comprehensive insights into your prompt engineering workflows
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <select 
            value={timeRange} 
            onChange={(e) => setTimeRange(e.target.value as '7d' | '30d' | '90d')}
            className="px-3 py-2 border rounded-md bg-background"
          >
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 90 days</option>
          </select>
          <Button variant="outline" onClick={exportData} disabled={!analytics}>
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
        </div>
      </div>

      {analytics ? (
        <div className="space-y-6">
          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Total Prompts</p>
                    <p className="text-2xl font-bold">{formatNumber(analytics.total_prompts)}</p>
                  </div>
                  <TrendingUp className="h-8 w-8 text-blue-600" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Success Rate</p>
                    <p className="text-2xl font-bold text-green-600">
                      {formatPercentage(analytics.success_rate / 100)}
                    </p>
                  </div>
                  <BarChart3 className="h-8 w-8 text-green-600" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Avg Processing Time</p>
                    <p className="text-2xl font-bold text-purple-600">
                      {analytics.avg_processing_time.toFixed(2)}s
                    </p>
                  </div>
                  <PieChart className="h-8 w-8 text-purple-600" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Throughput</p>
                    <p className="text-2xl font-bold text-orange-600">
                      {analytics.performance_metrics?.throughput || 0}/min
                    </p>
                  </div>
                  <Calendar className="h-8 w-8 text-orange-600" />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Domain Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Domain Distribution</CardTitle>
                <CardDescription>Breakdown of prompts by domain</CardDescription>
              </CardHeader>
              <CardContent className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={domainData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    >
                      {domainData.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Quality Trends */}
            <Card>
              <CardHeader>
                <CardTitle>Quality Trends</CardTitle>
                <CardDescription>Average quality score over time</CardDescription>
              </CardHeader>
              <CardContent className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={qualityTrends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis domain={[0, 10]} />
                    <Tooltip />
                    <Line 
                      type="monotone" 
                      dataKey="avg_quality" 
                      name="Average Quality"
                      stroke="#82ca9d" 
                      strokeWidth={2} 
                      dot={false}
                      activeDot={{ r: 6 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Performance Metrics */}
            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics</CardTitle>
                <CardDescription>Key system performance indicators</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <p className="text-sm font-medium text-muted-foreground">Response Time</p>
                    <p className="text-2xl font-semibold">
                      {analytics.performance_metrics?.avg_response_time?.toFixed(2) || 'N/A'}s
                    </p>
                  </div>
                  <div className="space-y-2">
                    <p className="text-sm font-medium text-muted-foreground">Error Rate</p>
                    <p className="text-2xl font-semibold">
                      {analytics.performance_metrics?.error_rate?.toFixed(2) || 'N/A'}%
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Top Performing Domains */}
          <Card>
            <CardHeader>
              <CardTitle>Domain Breakdown</CardTitle>
              <CardDescription>Distribution of prompts across different domains</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(analytics.domain_breakdown || {})
                  .sort(([, a], [, b]) => Number(b) - Number(a))
                  .map(([domain, count]) => (
                    <div key={domain} className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2">
                          <span className="font-medium capitalize">{domain}</span>
                          <Badge variant="outline" className="text-xs">
                            {String(count)} prompts
                          </Badge>
                        </div>
                        <div className="h-2 bg-gray-200 rounded-full mt-1 overflow-hidden">
                          <div 
                            className="h-full bg-blue-600" 
                            style={{ width: `${(Number(count) / analytics.total_prompts) * 100}%` }}
                          />
                        </div>
                      </div>
                      <div className="ml-4 w-16 text-right">
                        <span className="font-semibold">
                          {((Number(count) / analytics.total_prompts) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
              </div>
            </CardContent>
          </Card>
        </div>
      ) : (
        <div className="text-center py-12">
          <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <h3 className="text-lg font-medium mb-2">No analytics data available</h3>
          <p className="text-muted-foreground">
            Process some prompts to generate analytics data
          </p>
        </div>
      )}
    </div>
  );
}
