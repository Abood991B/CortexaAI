import { useState } from 'react';
import { 
  Activity, 
  Search, 
  Eye, 
  RefreshCw,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { LoadingSpinner } from '@/components/ui/loading';
import { useWorkflows, useWorkflowDetails } from '@/hooks/useApi';
import { formatDate, formatDuration } from '@/lib/utils';

export function Workflows() {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [domainFilter, setDomainFilter] = useState<string>('all');
  const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const limit = 20;

  const { data: workflows, isLoading, refetch } = useWorkflows({
    status: statusFilter === 'all' ? undefined : statusFilter,
    domain: domainFilter === 'all' ? undefined : domainFilter,
    page,
    limit
  });
  const { data: workflowDetails, isLoading: detailsLoading } = useWorkflowDetails(selectedWorkflow || '');

  // Filter workflows
  const filteredWorkflows = workflows?.data?.filter((workflow: any) => {
    const matchesSearch = workflow.prompt_preview.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         workflow.workflow_id.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesStatus = statusFilter === 'all' || workflow.status === statusFilter;
    const matchesDomain = domainFilter === 'all' || workflow.domain === domainFilter;
    
    return matchesSearch && matchesStatus && matchesDomain;
  }) || [];

  // Get unique domains and statuses
  const domains = ['all', ...new Set(workflows?.data?.map((w: any) => w.domain) || [])];
  const statuses = ['all', 'completed', 'failed', 'running'];

  const getStatusIcon = (status: any) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-600" />;
      case 'running':
        return <Clock className="h-4 w-4 text-blue-600" />;
      default:
        return <AlertCircle className="h-4 w-4 text-yellow-600" />;
    }
  };

  const getStatusVariant = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'failed':
        return 'destructive';
      case 'running':
        return 'info';
      default:
        return 'secondary';
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center">
            <Activity className="mr-3 h-8 w-8 text-primary" />
            Workflow Monitor
          </h1>
          <p className="text-muted-foreground">
            Track and analyze prompt processing workflows
          </p>
        </div>
        <Button onClick={() => refetch()} disabled={isLoading}>
          <RefreshCw className={`mr-2 h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search workflows by ID or prompt content..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>
        
        <div className="flex space-x-2">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-3 py-2 border border-input rounded-md bg-background"
          >
            {statuses.map(status => (
              <option key={status} value={status}>
                {status === 'all' ? 'All Statuses' : status}
              </option>
            ))}
          </select>
          
          <select
            value={domainFilter}
            onChange={(e) => setDomainFilter(e.target.value)}
            className="px-3 py-2 border border-input rounded-md bg-background"
          >
            {domains.map(domain => (
              <option key={String(domain)} value={String(domain)}>
                {domain === 'all' ? 'All Domains' : String(domain)}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Workflows List */}
      {isLoading ? (
        <div className="flex justify-center py-12">
          <LoadingSpinner size="lg" />
        </div>
      ) : filteredWorkflows.length > 0 ? (
        <div className="space-y-4">
          {filteredWorkflows.map((workflow: any) => (
            <Card 
              key={workflow.workflow_id as string}
              className="cursor-pointer hover:bg-gray-50 transition-colors"
              onClick={() => setSelectedWorkflow(workflow.workflow_id)}
            >
              <CardContent className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      {getStatusIcon(workflow.status)}
                      <span className="font-medium">{workflow.workflow_id}</span>
                      <Badge variant="outline">
                        {workflow.domain}
                      </Badge>
                    </div>
                    <p className="text-sm text-gray-600 mb-2">{workflow.prompt_preview}</p>
                    <div className="flex items-center gap-4 text-xs text-gray-500">
                      <span>Created: {formatDate(workflow.created_at)}</span>
                      <span>Duration: {formatDuration(workflow.duration)}</span>
                      <span>Steps: {workflow.total_steps}</span>
                    </div>
                  </div>
                  <Button variant="ghost" size="sm">
                    <Eye className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
          
          {/* Pagination */}
          {workflows && workflows.total > limit && (
            <div className="flex justify-center space-x-2">
              <Button
                variant="outline"
                onClick={() => setPage(Math.max(1, page - 1))}
                disabled={page === 1}
              >
                Previous
              </Button>
              <span className="flex items-center px-4 text-sm text-muted-foreground">
                Page {page} of {Math.ceil(workflows.total / limit)}
              </span>
              <Button
                variant="outline"
                onClick={() => setPage(page + 1)}
                disabled={page >= Math.ceil(workflows.total / limit)}
              >
                Next
              </Button>
            </div>
          )}
        </div>
      ) : (
        <div className="text-center py-12">
          <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <h3 className="text-lg font-medium mb-2">No workflows found</h3>
          <p className="text-muted-foreground">
            {searchTerm || statusFilter !== 'all' || domainFilter !== 'all'
              ? 'Try adjusting your search or filters'
              : 'Process some prompts to see workflows here'
            }
          </p>
        </div>
      )}

      {/* Workflow Details Modal */}
      {selectedWorkflow && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <Card className="w-full max-w-4xl max-h-[90vh] overflow-y-auto">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Workflow Details</CardTitle>
                <Button
                  variant="ghost"
                  onClick={() => setSelectedWorkflow(null)}
                >
                  ×
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {detailsLoading ? (
                <div className="flex justify-center py-8">
                  <LoadingSpinner />
                </div>
              ) : workflowDetails ? (
                <div className="space-y-6">
                  {/* Basic Info */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-medium mb-2">Workflow ID</h4>
                      <p className="font-mono text-sm">{workflowDetails.workflow_id}</p>
                    </div>
                    <div>
                      <h4 className="font-medium mb-2">Status</h4>
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(workflowDetails.status)}
                        <Badge variant={getStatusVariant(workflowDetails.status)}>
                          {workflowDetails.status}
                        </Badge>
                      </div>
                    </div>
                  </div>

                  {/* Original Prompt */}
                  <div>
                    <h4 className="font-medium mb-2">Original Prompt</h4>
                    <div className="p-3 bg-muted rounded-lg font-mono text-sm">
                      {workflowDetails.original_prompt}
                    </div>
                  </div>

                  {/* Optimized Prompt */}
                  {workflowDetails.optimized_prompt && (
                    <div>
                      <h4 className="font-medium mb-2">Optimized Prompt</h4>
                      <div className="p-3 bg-primary/10 rounded-lg font-mono text-sm">
                        {workflowDetails.optimized_prompt}
                      </div>
                    </div>
                  )}

                  {/* Metrics */}
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center">
                      <p className="text-2xl font-bold text-purple-600">
                        {workflowDetails.quality_score.toFixed(2)}
                      </p>
                      <p className="text-sm text-muted-foreground">Quality Score</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-blue-600">
                        {workflowDetails.iterations_used}
                      </p>
                      <p className="text-sm text-muted-foreground">Iterations</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-green-600">
                        {formatDuration(workflowDetails.processing_time)}
                      </p>
                      <p className="text-sm text-muted-foreground">Processing Time</p>
                    </div>
                  </div>

                  {/* Agent Steps */}
                  {workflowDetails.agent_steps && workflowDetails.agent_steps.length > 0 && (
                    <div>
                      <h4 className="font-medium mb-3">Agent Processing Steps</h4>
                      <div className="space-y-3">
                        {workflowDetails.agent_steps.map((step: any, index: number) => (
                          <div key={index} className="border rounded-lg p-3">
                            <div className="flex items-center justify-between mb-2">
                              <Badge variant="outline">{step.agent_type}</Badge>
                              <span className="text-xs text-muted-foreground">
                                {formatDuration(step.processing_time)}
                              </span>
                            </div>
                            <p className="text-sm">{step.output}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-center text-muted-foreground py-8">
                  Failed to load workflow details
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
