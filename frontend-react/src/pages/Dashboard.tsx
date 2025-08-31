import React from 'react';
import { 
  Brain, 
  FileText, 
  BarChart3, 
  Clock, 
  TrendingUp, 
  Users,
  Zap,
  CheckCircle
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { LoadingSpinner } from '@/components/ui/loading';
import { useStats, useHistory, useDomains } from '@/hooks/useApi';
import { formatNumber, formatPercentage, formatDuration, formatDate, getDomainColor } from '@/utils';

export function Dashboard() {
  const { data: stats, isLoading: statsLoading } = useStats();
  const { data: history, isLoading: historyLoading } = useHistory(5);
  const { data: domains, isLoading: domainsLoading } = useDomains();

  const quickStats = [
    {
      title: 'Total Workflows',
      value: stats?.total_workflows || 0,
      icon: Zap,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
    },
    {
      title: 'Success Rate',
      value: formatPercentage(stats?.success_rate || 0),
      icon: CheckCircle,
      color: 'text-green-600',
      bgColor: 'bg-green-50',
    },
    {
      title: 'Avg Quality Score',
      value: (stats?.average_quality_score || 0).toFixed(2),
      icon: TrendingUp,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50',
    },
    {
      title: 'Avg Processing Time',
      value: formatDuration(stats?.average_processing_time || 0),
      icon: Clock,
      color: 'text-orange-600',
      bgColor: 'bg-orange-50',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <p className="text-muted-foreground">
          Overview of your multi-agent prompt engineering system
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {quickStats.map((stat) => (
          <Card key={stat.title}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">
                    {stat.title}
                  </p>
                  <p className="text-2xl font-bold">
                    {statsLoading ? <LoadingSpinner size="sm" /> : stat.value}
                  </p>
                </div>
                <div className={`p-3 rounded-full ${stat.bgColor}`}>
                  <stat.icon className={`h-6 w-6 ${stat.color}`} />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Workflows */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Brain className="mr-2 h-5 w-5" />
              Recent Workflows
            </CardTitle>
            <CardDescription>
              Latest prompt processing activities
            </CardDescription>
          </CardHeader>
          <CardContent>
            {historyLoading ? (
              <div className="flex justify-center py-8">
                <LoadingSpinner />
              </div>
            ) : history && history.length > 0 ? (
              <div className="space-y-4">
                {history.map((workflow) => (
                  <div
                    key={workflow.workflow_id}
                    className="flex items-center justify-between p-3 rounded-lg border"
                  >
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <Badge className={getDomainColor(workflow.domain)}>
                          {workflow.domain}
                        </Badge>
                        <Badge 
                          variant={workflow.status === 'completed' ? 'success' : 'destructive'}
                        >
                          {workflow.status}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">
                        {workflow.prompt_preview}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {formatDate(workflow.timestamp)}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium">
                        {workflow.quality_score.toFixed(2)}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {formatDuration(workflow.processing_time)}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-center text-muted-foreground py-8">
                No workflows found
              </p>
            )}
          </CardContent>
        </Card>

        {/* Domain Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Users className="mr-2 h-5 w-5" />
              Domain Distribution
            </CardTitle>
            <CardDescription>
              Available domains and their usage
            </CardDescription>
          </CardHeader>
          <CardContent>
            {domainsLoading ? (
              <div className="flex justify-center py-8">
                <LoadingSpinner />
              </div>
            ) : domains && domains.length > 0 ? (
              <div className="space-y-3">
                {domains.slice(0, 8).map((domain) => (
                  <div
                    key={domain.domain}
                    className="flex items-center justify-between"
                  >
                    <div className="flex items-center space-x-3">
                      <Badge className={getDomainColor(domain.domain)}>
                        {domain.domain}
                      </Badge>
                      <span className="text-sm">
                        {domain.description}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      {domain.has_expert_agent && (
                        <Badge variant="success" className="text-xs">
                          Expert
                        </Badge>
                      )}
                      <span className="text-sm text-muted-foreground">
                        {stats?.domain_distribution?.[domain.domain] || 0}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-center text-muted-foreground py-8">
                No domains found
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* System Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <FileText className="mr-2 h-5 w-5" />
              Prompt Library
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-center">
              <p className="text-3xl font-bold text-blue-600">
                {formatNumber(0)} {/* Will be updated when prompts API is integrated */}
              </p>
              <p className="text-sm text-muted-foreground">
                Stored prompts
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <BarChart3 className="mr-2 h-5 w-5" />
              Templates
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-center">
              <p className="text-3xl font-bold text-green-600">
                {formatNumber(0)} {/* Will be updated when templates API is integrated */}
              </p>
              <p className="text-sm text-muted-foreground">
                Available templates
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Zap className="mr-2 h-5 w-5" />
              Active Experiments
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-center">
              <p className="text-3xl font-bold text-purple-600">
                {formatNumber(0)} {/* Will be updated when experiments API is integrated */}
              </p>
              <p className="text-sm text-muted-foreground">
                Running experiments
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
