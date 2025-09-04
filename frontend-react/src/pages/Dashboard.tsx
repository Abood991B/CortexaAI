import {
  Activity,
  BarChart3,
  Clock,
  CheckCircle,
  RefreshCw,
  Zap,
  Target,
  Brain,
  TrendingUp,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { LoadingSpinner } from '@/components/ui/loading';
import { Button } from '@/components/ui/button';
import { useStats, useHistory } from '@/hooks/useApi';
import { formatNumber, formatPercentage, formatDuration, formatDate } from '@/utils';
import { useNavigate } from 'react-router-dom';

export function Dashboard() {
  const navigate = useNavigate();
  const { data: stats, isLoading: statsLoading, refetch: refetchStats } = useStats();
  const { data: history, isLoading: historyLoading, refetch: refetchHistory } = useHistory(5);

  const handleRefresh = () => {
    refetchStats();
    refetchHistory();
  };


  const quickStats = [
    {
      title: 'Total Workflows',
      value: formatNumber(stats?.total_workflows || 0),
      icon: Zap,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200',
      description: 'All time workflows'
    },
    {
      title: 'Success Rate',
      value: formatPercentage(stats?.success_rate || 0),
      icon: CheckCircle,
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      borderColor: 'border-green-200',
      description: 'Completed successfully'
    },
    {
      title: 'Quality Score',
      value: (stats?.average_quality_score || 0).toFixed(1),
      icon: Target,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50',
      borderColor: 'border-purple-200',
      description: 'Average quality rating'
    },
    {
      title: 'Processing Time',
      value: formatDuration(stats?.average_processing_time || 0),
      icon: Clock,
      color: 'text-orange-600',
      bgColor: 'bg-orange-50',
      borderColor: 'border-orange-200',
      description: 'Average completion time'
    },
  ];

  return (
    <div className="space-y-8 p-6">
      {/* Enhanced Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Dashboard
          </h1>
          <p className="text-muted-foreground mt-2 text-lg">
            Real-time overview of your AI prompt engineering workflows and performance metrics.
          </p>
        </div>
        <Button 
          onClick={handleRefresh} 
          variant="outline" 
          size="sm"
          className="flex items-center gap-2 hover:bg-blue-50"
        >
          <RefreshCw className="h-4 w-4" />
          Refresh
        </Button>
      </div>

      {/* Enhanced Quick Stats */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        {quickStats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <Card key={index} className={`border-2 ${stat.borderColor} hover:shadow-lg transition-all duration-200 hover:scale-105`}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
                <div>
                  <CardTitle className="text-sm font-medium text-gray-600">{stat.title}</CardTitle>
                  <p className="text-xs text-muted-foreground mt-1">{stat.description}</p>
                </div>
                <div className={`p-3 rounded-full ${stat.bgColor} ring-2 ring-white shadow-sm`}>
                  <Icon className={`h-5 w-5 ${stat.color}`} />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-gray-900">
                  {statsLoading ? <LoadingSpinner size="sm" /> : stat.value}
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Content: Recent Workflows (takes 2/3 width) */}
        <div className="lg:col-span-2">
          <Card className="border-2 border-gray-100 hover:shadow-xl transition-all duration-300 h-full">
            <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-t-lg">
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center">
                  <Activity className="mr-2 h-5 w-5 text-blue-600" />
                  <span className="text-lg font-semibold text-gray-800">Recent Workflows</span>
                </div>
                <Badge variant="secondary" className="bg-blue-100 text-blue-700">
                  {history?.length || 0} recent
                </Badge>
              </CardTitle>
              <CardDescription>
                Latest prompt processing activities
              </CardDescription>
            </CardHeader>
            <CardContent className="p-6">
              {historyLoading ? (
                <div className="flex justify-center py-8">
                  <LoadingSpinner />
                </div>
              ) : history && history.length > 0 ? (
                <div className="space-y-4">
                  {history.map((workflow: any, index: number) => (
                    <div key={workflow.workflow_id || index} className="group flex items-center justify-between p-4 border-2 border-gray-100 rounded-xl hover:border-blue-200 hover:bg-blue-50/30 transition-all duration-200">
                      <div className="flex items-center space-x-4">
                        <div className={`w-3 h-3 rounded-full shadow-sm ${
                          workflow.status === 'completed' ? 'bg-green-500 ring-2 ring-green-200' : 
                          workflow.status === 'failed' ? 'bg-red-500 ring-2 ring-red-200' : 
                          workflow.status === 'cancelled' ? 'bg-gray-500 ring-2 ring-gray-200' :
                          'bg-yellow-500 ring-2 ring-yellow-200'
                        }`} />
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <p className="font-semibold text-sm text-gray-900">
                              Workflow {workflow.workflow_id || index + 1}
                            </p>
                            <Badge 
                              variant="outline" 
                              className={`text-xs ${
                                workflow.status === 'completed' ? 'border-green-300 text-green-700 bg-green-50' : 
                                workflow.status === 'failed' ? 'border-red-300 text-red-700 bg-red-50' :
                                workflow.status === 'cancelled' ? 'border-gray-300 text-gray-700 bg-gray-50' :
                                'border-yellow-300 text-yellow-700 bg-yellow-50'
                              }`}
                            >
                              {workflow.status}
                            </Badge>
                          </div>
                          <div className="flex items-center gap-4 mt-1">
                            <p className="text-xs text-gray-500">
                              {formatDate(workflow.timestamp)}
                            </p>
                            {workflow.output?.quality_score && (
                              <div className="flex items-center gap-1">
                                <Target className="h-3 w-3 text-purple-500" />
                                <span className="text-xs text-purple-600 font-medium">
                                  {workflow.output.quality_score.toFixed(1)}
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm text-gray-500 mb-2">
                          Processing Time: {formatDuration(workflow.processing_time_seconds || workflow.processing_time || 0)}
                        </div>
                        {workflow.output?.iterations_used && (
                          <p className="text-xs text-gray-500">
                            {workflow.output.iterations_used} iterations
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <Brain className="mx-auto h-12 w-12 text-gray-300 mb-4" />
                  <p className="text-gray-500 font-medium">No recent workflows found</p>
                  <p className="text-sm text-gray-400 mt-1">Start processing prompts to see workflow history</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Sidebar: Workflow Stats (takes 1/3 width) */}
        <div className="lg:col-span-1 space-y-8">
          <Card className="border-2 border-gray-100 hover:shadow-xl transition-all duration-300">
            <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-t-lg">
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center">
                  <BarChart3 className="mr-2 h-5 w-5 text-purple-600" />
                  <span className="text-lg font-semibold text-gray-800">Workflow Statistics</span>
                </div>
              </CardTitle>
              <CardDescription className="text-gray-600">Performance metrics and insights</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 p-6">
              {statsLoading ? (
                <div className="flex justify-center py-8">
                  <LoadingSpinner />
                </div>
              ) : stats ? (
                <div className="space-y-4">
                  <div className="p-3 rounded-lg bg-gray-50">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-semibold text-gray-800">Total Workflows</span>
                      <span className="text-lg font-bold text-gray-900">{formatNumber(stats.total_workflows || 0)}</span>
                    </div>
                    <p className="text-xs text-gray-500">All processed workflows</p>
                  </div>
                  <div className="p-3 rounded-lg bg-gray-50">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-semibold text-gray-800">Success Rate</span>
                      <span className="text-lg font-bold text-gray-900">{formatPercentage(stats.success_rate || 0)}</span>
                    </div>
                    <p className="text-xs text-gray-500">Completed successfully</p>
                  </div>
                  <div className="p-3 rounded-lg bg-gray-50">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-semibold text-gray-800">Error Rate</span>
                      <div className="flex items-center gap-1">
                        <TrendingUp className="h-4 w-4 text-red-500" />
                        <span className="text-sm font-bold text-red-600">{stats.error_workflows || 0} errors</span>
                      </div>
                    </div>
                    <p className="text-xs text-gray-500">Failed workflows</p>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12">
                  <BarChart3 className="mx-auto h-12 w-12 text-gray-300 mb-4" />
                  <p className="text-gray-500 font-medium">No statistics available</p>
                  <p className="text-sm text-gray-400 mt-1">Process workflows to see statistics</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Action-Oriented Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        <Card className="hover:shadow-lg transition-shadow duration-300 border-2 border-transparent hover:border-blue-300">
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Zap className="h-6 w-6 text-blue-500"/>Prompt Processor</CardTitle>
            <CardDescription>Optimize and test your prompts in real-time.</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => navigate('/processor')} className="w-full bg-blue-600 hover:bg-blue-700">
              Go to Processor
            </Button>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-shadow duration-300 border-2 border-transparent hover:border-yellow-300">
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Activity className="h-6 w-6 text-yellow-500"/>Workflows</CardTitle>
            <CardDescription>Track and analyze prompt processing workflows.</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => navigate('/workflows')} className="w-full bg-yellow-600 hover:bg-yellow-700">
              View Workflows
            </Button>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-shadow duration-300 border-2 border-transparent hover:border-red-300">
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Activity className="h-6 w-6 text-red-500"/>System Health</CardTitle>
            <CardDescription>Monitor system performance and health metrics.</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => navigate('/system-health')} className="w-full bg-red-600 hover:bg-red-700">
              View Health
            </Button>
          </CardContent>
        </Card>
      </div>
      


    </div>
  );
}
