import { Beaker, Plus, BarChart3, Pause, Play } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { LoadingSpinner } from '@/components/ui/loading';
import { useExperiments } from '@/hooks/useApi';
import { formatDate } from '@/lib/utils';

export function Experiments() {
  const { data: experiments, isLoading } = useExperiments();

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center">
            <Beaker className="mr-3 h-8 w-8 text-primary" />
            A/B Testing & Experiments
          </h1>
          <p className="text-muted-foreground">
            Test and optimize prompt variations
          </p>
        </div>
        <Button>
          <Plus className="mr-2 h-4 w-4" />
          New Experiment
        </Button>
      </div>

      {/* Experiments List */}
      {isLoading ? (
        <div className="flex justify-center py-12">
          <LoadingSpinner size="lg" />
        </div>
      ) : experiments && experiments.length > 0 ? (
        <div className="space-y-4">
          {experiments.map((experiment) => (
            <Card key={experiment.id} className="hover:shadow-md transition-shadow">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      {experiment.name}
                      <Badge variant={experiment.status === 'completed' ? 'default' : 'secondary'}>
                        {experiment.status}
                      </Badge>
                    </CardTitle>
                    <CardDescription>{experiment.description}</CardDescription>
                  </div>
                  <div className="flex space-x-2">
                    <Button variant="outline" size="sm">
                      <BarChart3 className="h-4 w-4" />
                    </Button>
                    <Button variant="outline" size="sm">
                      {experiment.status === 'running' ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {experiment.variants.map((variant, index) => (
                    <div key={index} className="p-3 border rounded-lg">
                      <h4 className="font-medium">{variant.name}</h4>
                      <p className="text-2xl font-bold text-green-600">
                        {((variant.conversion_rate || 0) * 100).toFixed(1)}%
                      </p>
                      <Beaker className="h-6 w-6 text-blue-500" />
                      <p className="text-sm text-muted-foreground">Conversion Rate</p>
                    </div>
                  ))}
                </div>
                <div className="mt-4 flex items-center justify-between text-sm text-muted-foreground">
                  <span>Created: {formatDate(experiment.created_at)}</span>
                  {experiment.status === 'completed' && experiment.completed_at && (
                    <span>Completed: {formatDate(experiment.completed_at)}</span>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <Beaker className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <h3 className="text-lg font-medium mb-2">No experiments yet</h3>
          <p className="text-muted-foreground mb-4">
            Create A/B tests to optimize your prompts
          </p>
          <Button>
            <Plus className="mr-2 h-4 w-4" />
            Create First Experiment
          </Button>
        </div>
      )}
    </div>
  );
}
