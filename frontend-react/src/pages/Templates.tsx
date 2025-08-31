import React, { useState } from 'react';
import { 
  Layout, 
  Plus, 
  Search, 
  Edit, 
  Copy,
  Download,
  Upload
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { LoadingSpinner } from '@/components/ui/loading';
import { useTemplates, useCreateTemplate, useUpdateTemplate } from '@/hooks/useApi';
import { formatDate } from '@/lib/utils';
import { toast } from 'sonner';
import type { Template } from '@/types/api';

export function Templates() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [editingTemplate, setEditingTemplate] = useState<Template | null>(null);

  const handleUpdateTemplate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingTemplate) return;
    
    const { name, description, content, category, variables } = newTemplate;
    
    const templateData: Partial<Template> = {
      name,
      description,
      content,
      category,
      variables,
    };

    try {
      await updateTemplate.mutateAsync({
        id: editingTemplate.id,
        template: templateData
      });
      
      toast.success('Template updated successfully');
      setIsCreateModalOpen(false);
      setEditingTemplate(null);
      setNewTemplate({ name: '', description: '', content: '', category: '', variables: [] });
    } catch (error) {
      toast.error('Failed to update template');
      console.error('Error updating template:', error);
    }
  };
  
  const [newTemplate, setNewTemplate] = useState({
    name: '',
    description: '',
    content: '',
    category: '',
    variables: [] as string[],
  });

  const { data: templates, isLoading } = useTemplates();
  const createTemplate = useCreateTemplate();
  const updateTemplate = useUpdateTemplate();

  // Filter templates
  const filteredTemplates = templates?.filter((template) => {
    const matchesSearch = template.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      template.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = !selectedCategory || selectedCategory === 'all' || template.category === selectedCategory;
    
    return matchesSearch && matchesCategory;
  });

  const categories = [...new Set(templates?.map((t) => t.category) || [])];

  const handleCreateTemplate = async (e: React.FormEvent) => {
    e.preventDefault();
    const { name, description, content, category, variables } = newTemplate;
    
    const templateData: Omit<Template, 'id' | 'created_at'> = {
      name,
      description,
      content,
      variables,
      category
    };

    try {
      await createTemplate.mutateAsync(templateData);
      toast.success('Template created successfully');
      setIsCreateModalOpen(false);
      setNewTemplate({ name: '', description: '', content: '', category: '', variables: [] });
    } catch (error) {
      toast.error('Failed to create template');
      console.error('Error creating template:', error);
    }
  };

  const handleCopyTemplate = (content: string) => {
    navigator.clipboard.writeText(content);
    toast.success('Template copied to clipboard');
  };

  const handleEditTemplate = (id: string) => {
    const template = templates?.find((t) => t.id === id);
    if (template) {
      setEditingTemplate(template);
      setIsCreateModalOpen(true);
      
      // Pre-fill the form with the template data
      setNewTemplate({
        name: template.name,
        description: template.description,
        content: template.content,
        category: template.category,
        variables: template.variables,
      });
    }
  };

  const handleExportTemplate = (template: Template) => {
    const content = JSON.stringify(template, null, 2);
    const blob = new Blob([content], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `template_${template.name.replace(/\s+/g, '_')}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    toast.success('Template exported');
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center">
            <Layout className="mr-3 h-8 w-8 text-primary" />
            Template Library
          </h1>
          <p className="text-muted-foreground">
            Reusable prompt templates with variables
          </p>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline">
            <Upload className="mr-2 h-4 w-4" />
            Import
          </Button>
          <Button onClick={() => setIsCreateModalOpen(true)}>
            <Plus className="mr-2 h-4 w-4" />
            New Template
          </Button>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search templates..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>
        
        <select
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
          className="px-3 py-2 border border-input rounded-md bg-background"
        >
          <option value="">All Categories</option>
          {categories.map(category => (
            <option key={category} value={category}>
              {category}
            </option>
          ))}
        </select>

      </div>

      {/* Templates Grid */}
      {isLoading ? (
        <div className="flex justify-center py-12">
          <LoadingSpinner size="lg" />
        </div>
      ) : filteredTemplates && filteredTemplates.length > 0 ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {filteredTemplates.map((template) => (
            <Card key={template.id} className="hover:shadow-md transition-shadow">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <CardTitle className="text-lg">{template.name}</CardTitle>
                    <CardDescription className="mt-1">
                      {template.description}
                    </CardDescription>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleCopyTemplate(template.content)}
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleEditTemplate(template.id)}
                    >
                      <Edit className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleExportTemplate(template)}
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="text-sm font-mono bg-muted p-3 rounded-lg max-h-40 overflow-y-auto">
                    {template.content}
                  </div>
                  
                  {template.variables.length > 0 && (
                    <div>
                      <h4 className="text-sm font-medium mb-2">Variables:</h4>
                      <div className="flex flex-wrap gap-1">
                        {template.variables.map((variable: string, index: number) => (
                          <Badge key={index} variant="outline" className="text-xs">
                            {variable}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <div className="flex items-center justify-between text-sm text-muted-foreground">
                    <Badge variant="outline">{template.category}</Badge>
                    <span>{formatDate(template.created_at)}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <Layout className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <h3 className="text-lg font-medium mb-2">No templates found</h3>
          <p className="text-muted-foreground mb-4">
            Create reusable prompt templates with variables
          </p>
          <Button onClick={() => setIsCreateModalOpen(true)}>
            <Plus className="mr-2 h-4 w-4" />
            Create First Template
          </Button>
        </div>
      )}

      {/* Create/Edit Modal */}
      {(isCreateModalOpen || editingTemplate) && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <Card className="w-full max-w-3xl max-h-[90vh] overflow-y-auto">
            <CardHeader>
              <CardTitle>
                {editingTemplate ? 'Edit Template' : 'Create New Template'}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={editingTemplate ? handleUpdateTemplate : handleCreateTemplate} className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm font-medium mb-2 block">Name</label>
                    <Input
                      value={editingTemplate ? editingTemplate.name : newTemplate.name}
                      onChange={(e) => editingTemplate 
                        ? setEditingTemplate({...editingTemplate, name: e.target.value})
                        : setNewTemplate({...newTemplate, name: e.target.value})
                      }
                      placeholder="Template name..."
                      required
                    />
                  </div>
                  
                  <div>
                    <label className="text-sm font-medium mb-2 block">Category</label>
                    <Input
                      value={editingTemplate ? editingTemplate.category : newTemplate.category}
                      onChange={(e) => editingTemplate 
                        ? setEditingTemplate({...editingTemplate, category: e.target.value})
                        : setNewTemplate({...newTemplate, category: e.target.value})
                      }
                      placeholder="e.g., business, technology"
                    />
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">Description</label>
                  <Input
                    value={editingTemplate ? editingTemplate.description : newTemplate.description}
                    onChange={(e) => editingTemplate 
                      ? setEditingTemplate({...editingTemplate, description: e.target.value})
                      : setNewTemplate({...newTemplate, description: e.target.value})
                    }
                    placeholder="What does this template do?"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">
                    Variables (comma-separated)
                  </label>
                  <Input
                    value={editingTemplate ? editingTemplate.variables.join(', ') : newTemplate.variables.join(', ')}
                    onChange={(e) => {
                      const variables = e.target.value.split(',').map(v => v.trim()).filter(Boolean);
                      editingTemplate 
                        ? setEditingTemplate({...editingTemplate, variables})
                        : setNewTemplate({...newTemplate, variables});
                    }}
                    placeholder="user_input, context, language"
                  />
                  <p className="text-sm text-muted-foreground mt-1">
                    Use {`{variable_name}`} in your template content
                  </p>
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">Template Content</label>
                  <Textarea
                    value={editingTemplate ? editingTemplate.content : newTemplate.content}
                    onChange={(e) => editingTemplate 
                      ? setEditingTemplate({...editingTemplate, content: e.target.value})
                      : setNewTemplate({...newTemplate, content: e.target.value})
                    }
                    placeholder="Write a {language} function that {user_input}. Consider the following context: {context}"
                    className="min-h-[200px] font-mono"
                    required
                  />
                </div>


                <div className="flex justify-end space-x-2">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => {
                      setIsCreateModalOpen(false);
                      setEditingTemplate(null);
                      setNewTemplate({ name: '', description: '', content: '', category: '', variables: [] });
                    }}
                  >
                    Cancel
                  </Button>
                  <Button 
                    type="submit"
                  >
                    {editingTemplate ? 'Update' : 'Create'} Template
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
