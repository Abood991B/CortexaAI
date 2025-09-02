import React, { useState } from 'react';
import { 
  FileText, 
  Plus, 
  Search, 
  Edit, 
  Filter, 
  Trash2, 
  Copy,
  Calendar,
  Tag
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { LoadingSpinner } from '@/components/ui/loading';
import { formatDate, copyToClipboard } from '@/utils';
import { toast } from 'sonner';
import { useCreatePrompt, useUpdatePrompt } from '@/hooks/useApi';

interface SimplePrompt {
  id: string;
  title: string;
  content: string;
  description: string;
  category: string;
  tags: string[];
  created_at: string;
}

interface CreatePromptData {
  name: string;
  current_version: string;
  status: 'draft' | 'active' | 'archived';
  metadata: {
    domain: string;
    strategy: string;
    author: string;
    tags: string[];
    description: string;
    performance_metrics: Record<string, any>;
    dependencies: string[];
    configuration: Record<string, any>;
  };
  versions: string[];
  created_by: string;
}

export function Prompts() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [editingPrompt, setEditingPrompt] = useState<SimplePrompt | null>(null);
  
  const [newPrompt, setNewPrompt] = useState<CreatePromptData>({
    name: '',
    current_version: '1.0.0',
    status: 'draft',
    metadata: {
      domain: '',
      strategy: 'default',
      author: 'user',
      tags: [],
      description: '',
      performance_metrics: {},
      dependencies: [],
      configuration: {},
    },
    versions: ['1.0.0'],
    created_by: 'user',
  });

  // Mock data for now - will be replaced with real API calls once backend is running
  const prompts: SimplePrompt[] = [];
  const isLoading = false;

  // Filter prompts based on search and category
  const filteredPrompts = prompts.filter((prompt: SimplePrompt) => {
    const matchesSearch = prompt.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         prompt.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         prompt.tags.some((tag: string) => tag.toLowerCase().includes(searchTerm.toLowerCase()));
    
    const matchesCategory = selectedCategory === 'all' || prompt.category === selectedCategory;
    
    return matchesSearch && matchesCategory;
  });

  // Get unique categories
  const categories = ['all', ...new Set(prompts?.map((p: any) => p.metadata?.domain || 'uncategorized') || [])];

  const createPromptMutation = useCreatePrompt();
  const updatePromptMutation = useUpdatePrompt();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newPrompt.name.trim() || !newPrompt.metadata.description.trim()) return;

    try {
      if (editingPrompt) {
        await updatePromptMutation.mutateAsync({
          id: editingPrompt.id,
          prompt: newPrompt
        });
      } else {
        await createPromptMutation.mutateAsync(newPrompt);
      }
      setIsCreateModalOpen(false);
      setNewPrompt({
        name: '',
        current_version: '1.0.0',
        status: 'draft',
        metadata: {
          domain: '',
          strategy: 'default',
          author: 'user',
          tags: [],
          description: '',
          performance_metrics: {},
          dependencies: [],
          configuration: {},
        },
        versions: ['1.0.0'],
        created_by: 'user',
      });
    } catch (error) {
      // Error handling is done in the mutation hooks
    }
  };

  const handleDeletePrompt = async (/* promptId: string */) => {
    if (confirm('Are you sure you want to delete this prompt?')) {
      // TODO: Implement API call when backend is running
      toast.info('Prompt deletion will be available when backend is connected');
    }
  };

  const handleCopyPrompt = (content: string) => {
    copyToClipboard(content);
    toast.success('Prompt copied to clipboard');
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center">
            <FileText className="mr-3 h-8 w-8 text-primary" />
            Prompt Library
          </h1>
          <p className="text-muted-foreground">
            Manage and organize your prompt collection
          </p>
        </div>
        <Button onClick={() => setIsCreateModalOpen(true)}>
          <Plus className="mr-2 h-4 w-4" />
          New Prompt
        </Button>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search prompts by title, content, or tags..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>
        
        <div className="flex items-center space-x-2">
          <Filter className="h-4 w-4 text-muted-foreground" />
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-2 border border-input rounded-md bg-background"
          >
            {categories.map(category => (
              <option key={category} value={category}>
                {category === 'all' ? 'All Categories' : category}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Prompts Grid */}
      {isLoading ? (
        <div className="flex justify-center py-12">
          <LoadingSpinner size="lg" />
        </div>
      ) : filteredPrompts.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredPrompts.map((prompt) => (
            <Card key={prompt.id} className="hover:shadow-md transition-shadow">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <CardTitle className="text-lg">{prompt.title}</CardTitle>
                    <CardDescription className="mt-1">
                      {prompt.description || 'No description'}
                    </CardDescription>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleCopyPrompt(prompt.content)}
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => {
                        setEditingPrompt(prompt);
                        setNewPrompt({
                          name: prompt.title,
                          current_version: '1.0.0',
                          status: 'draft',
                          metadata: {
                            domain: prompt.category,
                            strategy: 'default',
                            author: 'user',
                            tags: prompt.tags,
                            description: prompt.description,
                            performance_metrics: {},
                            dependencies: [],
                            configuration: {},
                          },
                          versions: ['1.0.0'],
                          created_by: 'user',
                        });
                      }}
                    >
                      <Edit className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleDeletePrompt()}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="text-sm font-mono bg-muted p-3 rounded-lg max-h-32 overflow-y-auto">
                    {prompt.content.substring(0, 200)}
                    {prompt.content.length > 200 && '...'}
                  </div>
                  
                  <div className="flex items-center justify-between text-sm text-muted-foreground">
                    <div className="flex items-center space-x-2">
                      <Calendar className="h-3 w-3" />
                      <span>{formatDate(prompt.created_at)}</span>
                    </div>
                    <Badge variant="secondary">{prompt.category}</Badge>
                  </div>
                  
                  {prompt.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {prompt.tags.map((tag, index) => (
                        <Badge key={index} variant="outline" className="text-xs">
                          <Tag className="h-2 w-2 mr-1" />
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <h3 className="text-lg font-medium mb-2">No prompts found</h3>
          <p className="text-muted-foreground mb-4">
            {searchTerm || selectedCategory !== 'all' 
              ? 'Try adjusting your search or filters'
              : 'Get started by creating your first prompt'
            }
          </p>
          <Button onClick={() => setIsCreateModalOpen(true)}>
            <Plus className="mr-2 h-4 w-4" />
            Create First Prompt
          </Button>
        </div>
      )}

      {/* Create/Edit Modal */}
      {(isCreateModalOpen || editingPrompt) && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <Card className="w-full max-w-2xl max-h-[90vh] overflow-y-auto">
            <CardHeader>
              <CardTitle>
                {editingPrompt ? 'Edit Prompt' : 'Create New Prompt'}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Title</label>
                  <Input
                    value={newPrompt.name}
                    onChange={(e) => setNewPrompt({...newPrompt, name: e.target.value})}
                    placeholder="Enter prompt title..."
                    required
                  />
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">Description</label>
                  <Input
                    value={newPrompt.metadata.description}
                    onChange={(e) => setNewPrompt({...newPrompt, metadata: {...newPrompt.metadata, description: e.target.value}})}
                    placeholder="Brief description of the prompt..."
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm font-medium mb-2 block">Category</label>
                    <Input
                      value={newPrompt.metadata.domain}
                      onChange={(e) => setNewPrompt({...newPrompt, metadata: {...newPrompt.metadata, domain: e.target.value}})}
                      placeholder="e.g., coding, writing, analysis"
                    />
                  </div>
                  
                  <div>
                    <label className="text-sm font-medium mb-2 block">Tags (comma-separated)</label>
                    <Input
                      value={newPrompt.metadata.tags.join(', ')}
                      onChange={(e) => {
                        const tags = e.target.value.split(',').map(tag => tag.trim()).filter(Boolean);
                        setNewPrompt({...newPrompt, metadata: {...newPrompt.metadata, tags}});
                      }}
                      placeholder="python, function, algorithm"
                    />
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">Prompt Content</label>
                  <Textarea
                    value={newPrompt.metadata.configuration.content || ''}
                    onChange={(e) => setNewPrompt({...newPrompt, metadata: {...newPrompt.metadata, configuration: {...newPrompt.metadata.configuration, content: e.target.value}}})}
                    placeholder="Enter your prompt content here..."
                    className="min-h-[200px]"
                    required
                  />
                </div>

                <div className="flex justify-end space-x-2">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => {
                      setIsCreateModalOpen(false);
                      setEditingPrompt(null);
                      setNewPrompt({
                        name: '',
                        current_version: '1.0.0',
                        status: 'draft',
                        metadata: {
                          domain: '',
                          strategy: 'default',
                          author: 'user',
                          tags: [],
                          description: '',
                          performance_metrics: {},
                          dependencies: [],
                          configuration: {},
                        },
                        versions: ['1.0.0'],
                        created_by: 'user',
                      });
                    }}
                  >
                    Cancel
                  </Button>
                  <Button 
                    type="submit"
                    disabled={createPromptMutation.isPending || updatePromptMutation.isPending}
                  >
                    {createPromptMutation.isPending || updatePromptMutation.isPending ? (
                      <LoadingSpinner size="sm" className="mr-2" />
                    ) : null}
                    {editingPrompt ? 'Update' : 'Create'} Prompt
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
