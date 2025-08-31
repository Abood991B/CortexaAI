import React from 'react';
import { Globe, Users, CheckCircle, XCircle } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { LoadingSpinner } from '@/components/ui/loading';
import { useDomains } from '@/hooks/useApi';

export function Domains() {
  const { data: domains, isLoading } = useDomains();

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold flex items-center">
          <Globe className="mr-3 h-8 w-8 text-primary" />
          Domain Management
        </h1>
        <p className="text-muted-foreground">
          Available domains and their expert agents
        </p>
      </div>

      {/* Domains Grid */}
      {isLoading ? (
        <div className="flex justify-center py-12">
          <LoadingSpinner size="lg" />
        </div>
      ) : domains && domains.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {domains.map((domain) => (
            <Card key={domain.domain}>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="capitalize">{domain.domain}</CardTitle>
                    <CardDescription>{domain.description}</CardDescription>
                  </div>
                  <div className="flex items-center space-x-1">
                    {domain.has_expert_agent ? (
                      <CheckCircle className="h-5 w-5 text-green-600" />
                    ) : (
                      <XCircle className="h-5 w-5 text-red-600" />
                    )}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="text-sm font-medium mb-2">Keywords</h4>
                    <div className="flex flex-wrap gap-1">
                      {domain.keywords.map((keyword, index) => (
                        <Badge key={index} variant="outline" className="text-xs">
                          {keyword}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Expert Agent:</span>
                    <Badge variant={domain.has_expert_agent ? 'default' : 'secondary'}>
                      {domain.has_expert_agent ? 'Available' : 'Not Available'}
                    </Badge>
                  </div>
                  
                  {domain.agent_created && (
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Status:</span>
                      <Badge variant="default">Active</Badge>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <Globe className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <h3 className="text-lg font-medium mb-2">No domains available</h3>
          <p className="text-muted-foreground">
            Domains will appear here as they are discovered and configured
          </p>
        </div>
      )}
    </div>
  );
}
