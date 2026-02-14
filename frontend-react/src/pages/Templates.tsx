import { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Layers, RefreshCw, Search, Code, Eye, Sparkles, Plus, Star,
  Filter, Copy, Check, Tag, Grid3X3, List, ChevronDown, X, BookOpen,
  Zap, Briefcase, GraduationCap, Palette, Database, Shield, Globe,
  FileText, ArrowUpDown,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { LoadingSpinner } from '@/components/ui/loading';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import {
  useTemplates,
  useRenderTemplate,
  useCreateTemplate,
} from '@/hooks/useApi';
import { AppLayout, PageHeader } from '@/components/layout';

// ─── Domain Icon Map ──────────────────────────────────────────
const domainIcons: Record<string, React.ReactNode> = {
  software_engineering: <Code className="h-4 w-4" />,
  general: <Zap className="h-4 w-4" />,
  data_science: <Database className="h-4 w-4" />,
  creative_writing: <Palette className="h-4 w-4" />,
  business_strategy: <Briefcase className="h-4 w-4" />,
  marketing: <Globe className="h-4 w-4" />,
  education: <GraduationCap className="h-4 w-4" />,
  report_writing: <FileText className="h-4 w-4" />,
};

// ─── Source Detection ─────────────────────────────────────────
type TemplateSource = 'claude-library' | 'custom';

function getTemplateSource(tpl: any): TemplateSource {
  const tags: string[] = tpl.tags || [];
  if (tags.includes('anthropic-official') || tags.includes('claude-library')) return 'claude-library';
  return 'custom';
}

const sourceConfig: Record<TemplateSource, { label: string; color: string; bgColor: string }> = {
  'claude-library': { label: 'Claude Library', color: 'text-orange-700 dark:text-orange-400', bgColor: 'bg-orange-50 dark:bg-orange-950/40 border-orange-200 dark:border-orange-800' },
  'custom': { label: 'Custom', color: 'text-gray-700 dark:text-gray-400', bgColor: 'bg-gray-50 dark:bg-gray-950/40 border-gray-200 dark:border-gray-800' },
};

// ─── Domain Badge Colors ──────────────────────────────────────
const domainColors: Record<string, string> = {
  coding: 'bg-violet-100 text-violet-800 dark:bg-violet-900/40 dark:text-violet-300',
  productivity: 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300',
  writing: 'bg-pink-100 text-pink-800 dark:bg-pink-900/40 dark:text-pink-300',
  research: 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900/40 dark:text-cyan-300',
  education: 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900/40 dark:text-indigo-300',
  general: 'bg-slate-100 text-slate-800 dark:bg-slate-800/40 dark:text-slate-300',
};

// ─── Sort Options ─────────────────────────────────────────────
type SortOption = 'favorites' | 'name-asc' | 'name-desc' | 'domain';

export function Templates() {
  const navigate = useNavigate();
  const [search, setSearch] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [selectedSource, setSelectedSource] = useState<TemplateSource | ''>('');
  const [selectedTags, setSelectedTags] = useState<Set<string>>(new Set());
  const [viewMode, setViewMode] = useState<'grid' | 'list'>(() => {
    return (localStorage.getItem('cortexa_tpl_view') as 'grid' | 'list') || 'grid';
  });
  const [sortBy, setSortBy] = useState<SortOption>('favorites');
  const [renderInput, setRenderInput] = useState<{ name: string; vars: Record<string, string> } | null>(null);
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [copiedTemplate, setCopiedTemplate] = useState<string | null>(null);
  const [expandedTemplate, setExpandedTemplate] = useState<string | null>(null);
  const [newTemplate, setNewTemplate] = useState({ name: '', domain: '', template_text: '', description: '', variableInput: '' });
  const [favorites, setFavorites] = useState<Set<string>>(() => {
    try {
      const stored = localStorage.getItem('cortexa_favorite_templates');
      return stored ? new Set(JSON.parse(stored)) : new Set();
    } catch { return new Set(); }
  });

  // Persist preferences
  useEffect(() => {
    localStorage.setItem('cortexa_favorite_templates', JSON.stringify([...favorites]));
  }, [favorites]);

  useEffect(() => {
    localStorage.setItem('cortexa_tpl_view', viewMode);
  }, [viewMode]);

  const toggleFavorite = useCallback((name: string) => {
    setFavorites(prev => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }, []);

  const { data: templates, isLoading: tplLoading, refetch: refetchTemplates } = useTemplates();
  const renderMut = useRenderTemplate();
  const createMut = useCreateTemplate();

  const allTemplates: any[] = Array.isArray(templates) ? templates : templates?.templates ?? [];

  // Extract metadata
  const categories = useMemo(() =>
    Array.from(new Set(allTemplates.map((t: any) => t.domain || t.category || 'General').filter(Boolean))).sort(),
    [allTemplates]
  );

  const allTags = useMemo(() => {
    const tagSet = new Set<string>();
    allTemplates.forEach((t: any) => (t.tags || []).forEach((tag: string) => tagSet.add(tag)));
    return Array.from(tagSet).sort();
  }, [allTemplates]);

  // Source counts
  const sourceCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    allTemplates.forEach((t: any) => {
      const src = getTemplateSource(t);
      counts[src] = (counts[src] || 0) + 1;
    });
    return counts;
  }, [allTemplates]);

  // Filtered & sorted templates
  const filteredTemplates = useMemo(() => {
    let result = allTemplates.filter((t: any) => {
      const matchSearch = !search ||
        t.name?.toLowerCase().includes(search.toLowerCase()) ||
        t.description?.toLowerCase().includes(search.toLowerCase()) ||
        (t.tags || []).some((tag: string) => tag.toLowerCase().includes(search.toLowerCase()));
      const cat = t.domain || t.category || 'General';
      const matchCategory = !selectedCategory || cat === selectedCategory;
      const matchSource = !selectedSource || getTemplateSource(t) === selectedSource;
      const matchTags = selectedTags.size === 0 || (t.tags || []).some((tag: string) => selectedTags.has(tag));
      return matchSearch && matchCategory && matchSource && matchTags;
    });

    // Sort
    result.sort((a: any, b: any) => {
      const aFav = favorites.has(a.name) ? 1 : 0;
      const bFav = favorites.has(b.name) ? 1 : 0;
      if (sortBy === 'favorites') return bFav - aFav;
      if (sortBy === 'name-asc') return (a.name || '').localeCompare(b.name || '');
      if (sortBy === 'name-desc') return (b.name || '').localeCompare(a.name || '');
      if (sortBy === 'domain') return (a.domain || '').localeCompare(b.domain || '');
      return bFav - aFav;
    });

    return result;
  }, [allTemplates, search, selectedCategory, selectedSource, selectedTags, sortBy, favorites]);

  const handleUseTemplate = (template: string) => {
    navigate('/', { state: { initialPrompt: template } });
  };

  const handleCopy = useCallback((name: string, template: string) => {
    navigator.clipboard.writeText(template || '');
    setCopiedTemplate(name);
    setTimeout(() => setCopiedTemplate(null), 2000);
  }, []);

  const toggleTag = useCallback((tag: string) => {
    setSelectedTags(prev => {
      const next = new Set(prev);
      if (next.has(tag)) next.delete(tag);
      else next.add(tag);
      return next;
    });
  }, []);

  const clearAllFilters = useCallback(() => {
    setSearch('');
    setSelectedCategory('');
    setSelectedSource('');
    setSelectedTags(new Set());
  }, []);

  const hasActiveFilters = search || selectedCategory || selectedSource || selectedTags.size > 0;

  return (
    <AppLayout>
      <PageHeader
        title="Templates"
        icon={<Layers className="h-5 w-5 text-primary" />}
        actions={
          <div className="flex items-center gap-2">
            <Button variant="default" size="sm" onClick={() => setIsCreateOpen(true)} className="gap-1.5">
              <Plus className="h-3.5 w-3.5" /> Create
            </Button>
            <Button variant="outline" size="sm" onClick={() => refetchTemplates()} className="gap-1.5">
              <RefreshCw className="h-3.5 w-3.5" /> Refresh
            </Button>
          </div>
        }
      />

      <div className="flex-1 overflow-auto p-6 space-y-6">
        {/* ─── Hero Stats ──────────────────────────────────── */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <Card className="border-border/40">
            <CardContent className="p-4 flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <BookOpen className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-2xl font-bold">{allTemplates.length}</p>
                <p className="text-xs text-muted-foreground">Total Templates</p>
              </div>
            </CardContent>
          </Card>
          <Card className="border-border/40">
            <CardContent className="p-4 flex items-center gap-3">
              <div className="p-2 rounded-lg bg-orange-500/10">
                <Shield className="h-5 w-5 text-orange-500" />
              </div>
              <div>
                <p className="text-2xl font-bold">{sourceCounts['claude-library'] || 0}</p>
                <p className="text-xs text-muted-foreground">Claude Library</p>
              </div>
            </CardContent>
          </Card>
          <Card className="border-border/40">
            <CardContent className="p-4 flex items-center gap-3">
              <div className="p-2 rounded-lg bg-yellow-500/10">
                <Star className="h-5 w-5 text-yellow-500" />
              </div>
              <div>
                <p className="text-2xl font-bold">{favorites.size}</p>
                <p className="text-xs text-muted-foreground">Favorites</p>
              </div>
            </CardContent>
          </Card>
          <Card className="border-border/40">
            <CardContent className="p-4 flex items-center gap-3">
              <div className="p-2 rounded-lg bg-violet-500/10">
                <Tag className="h-5 w-5 text-violet-500" />
              </div>
              <div>
                <p className="text-2xl font-bold">{categories.length}</p>
                <p className="text-xs text-muted-foreground">Categories</p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* ─── Source Tabs ─────────────────────────────────── */}
        <Tabs value={selectedSource || 'all'} onValueChange={(v) => setSelectedSource(v === 'all' ? '' : v as TemplateSource)}>
          <TabsList className="w-full justify-start overflow-x-auto flex-wrap h-auto gap-1 p-1">
            <TabsTrigger value="all" className="text-xs gap-1.5">
              <Layers className="h-3 w-3" />
              All <span className="opacity-60">({allTemplates.length})</span>
            </TabsTrigger>
            <TabsTrigger value="claude-library" className="text-xs gap-1.5">
              <Shield className="h-3 w-3" />
              Claude Library <span className="opacity-60">({sourceCounts['claude-library'] || 0})</span>
            </TabsTrigger>
            <TabsTrigger value="custom" className="text-xs gap-1.5">
              Custom <span className="opacity-60">({sourceCounts['custom'] || 0})</span>
            </TabsTrigger>
          </TabsList>
        </Tabs>

        {/* ─── Search, Filter & View Controls ─────────────── */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search templates, tags..."
              className="pl-9 pr-9"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
            {search && (
              <button
                onClick={() => setSearch('')}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>

          {categories.length > 1 && (
            <div className="relative">
              <Filter className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
              <select
                className="pl-9 pr-8 border rounded-lg px-3 py-2 text-sm bg-background focus:ring-2 focus:ring-ring focus:outline-none appearance-none cursor-pointer min-w-[160px]"
                value={selectedCategory}
                onChange={e => setSelectedCategory(e.target.value)}
              >
                <option value="">All Domains</option>
                {categories.map((cat) => (
                  <option key={cat} value={cat}>{cat.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</option>
                ))}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
            </div>
          )}

          {/* Sort */}
          <div className="relative">
            <ArrowUpDown className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
            <select
              className="pl-9 pr-8 border rounded-lg px-3 py-2 text-sm bg-background focus:ring-2 focus:ring-ring focus:outline-none appearance-none cursor-pointer min-w-[140px]"
              value={sortBy}
              onChange={e => setSortBy(e.target.value as SortOption)}
            >
              <option value="favorites">Favorites First</option>
              <option value="name-asc">Name A→Z</option>
              <option value="name-desc">Name Z→A</option>
              <option value="domain">By Domain</option>
            </select>
            <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
          </div>

          {/* View Toggle */}
          <div className="flex items-center border rounded-lg p-0.5 bg-muted/30">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-1.5 rounded-md transition-colors ${viewMode === 'grid' ? 'bg-background shadow-sm' : 'hover:bg-background/50'}`}
              title="Grid view"
            >
              <Grid3X3 className="h-4 w-4" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-1.5 rounded-md transition-colors ${viewMode === 'list' ? 'bg-background shadow-sm' : 'hover:bg-background/50'}`}
              title="List view"
            >
              <List className="h-4 w-4" />
            </button>
          </div>

          {hasActiveFilters && (
            <Button variant="ghost" size="sm" onClick={clearAllFilters} className="text-xs text-muted-foreground hover:text-foreground gap-1">
              <X className="h-3 w-3" /> Clear Filters
            </Button>
          )}
        </div>

        {/* ─── Active Tag Filters ─────────────────────────── */}
        {selectedTags.size > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-xs text-muted-foreground mr-1">Active tags:</span>
            {[...selectedTags].map(tag => (
              <Badge
                key={tag}
                variant="default"
                className="text-xs cursor-pointer gap-1 pr-1"
                onClick={() => toggleTag(tag)}
              >
                {tag} <X className="h-3 w-3" />
              </Badge>
            ))}
          </div>
        )}

        {/* ─── Results Count ──────────────────────────────── */}
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            {tplLoading ? 'Loading...' : `${filteredTemplates.length} template${filteredTemplates.length !== 1 ? 's' : ''}`}
            {hasActiveFilters && ` (filtered from ${allTemplates.length})`}
          </p>
        </div>

        {/* ─── Template Grid / List ───────────────────────── */}
        {tplLoading ? (
          <div className="flex justify-center py-16"><LoadingSpinner size="lg" /></div>
        ) : filteredTemplates.length === 0 ? (
          <Card>
            <CardContent className="py-16 text-center text-muted-foreground">
              <Layers className="h-12 w-12 mx-auto mb-4 opacity-30" />
              <p className="text-lg font-medium">
                {hasActiveFilters ? 'No matching templates' : 'No templates available yet'}
              </p>
              <p className="text-sm mt-1">
                {hasActiveFilters
                  ? 'Try adjusting your search or filters.'
                  : 'Templates will appear here once configured in the backend.'}
              </p>
              {hasActiveFilters && (
                <Button variant="outline" size="sm" onClick={clearAllFilters} className="mt-4 gap-1.5">
                  <X className="h-3.5 w-3.5" /> Clear All Filters
                </Button>
              )}
            </CardContent>
          </Card>
        ) : viewMode === 'grid' ? (
          /* ─── GRID VIEW ─── */
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {filteredTemplates.map((tpl: any, idx: number) => {
              const source = getTemplateSource(tpl);
              const srcCfg = sourceConfig[source];
              const domain = tpl.domain || tpl.category || 'General';
              const domainColor = domainColors[domain] || 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300';

              return (
                <Card
                  key={tpl.id || tpl.name || idx}
                  className={`flex flex-col transition-all duration-200 hover:shadow-lg hover:-translate-y-0.5 group border ${
                    favorites.has(tpl.name) ? 'ring-1 ring-yellow-400/30' : ''
                  }`}
                >
                  <CardHeader className="pb-2 space-y-2">
                    {/* Source & Domain badges */}
                    <div className="flex items-center justify-between gap-2">
                      <div className={`text-[10px] font-medium px-2 py-0.5 rounded-full border ${srcCfg.bgColor} ${srcCfg.color}`}>
                        {srcCfg.label}
                      </div>
                      <div className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${domainColor}`}>
                        <span className="flex items-center gap-1">
                          {domainIcons[domain] || <Layers className="h-3 w-3" />}
                          {domain.replace(/_/g, ' ')}
                        </span>
                      </div>
                    </div>

                    {/* Title + Favorite */}
                    <div className="flex items-start justify-between gap-2">
                      <CardTitle className="text-base leading-snug line-clamp-2">{tpl.name}</CardTitle>
                      <button
                        onClick={(e) => { e.stopPropagation(); toggleFavorite(tpl.name); }}
                        className="shrink-0 p-1 rounded-md hover:bg-accent/60 transition-colors"
                        title={favorites.has(tpl.name) ? 'Remove from favorites' : 'Add to favorites'}
                      >
                        <Star className={`h-4 w-4 transition-colors ${favorites.has(tpl.name) ? 'fill-yellow-400 text-yellow-400' : 'text-muted-foreground/30 group-hover:text-muted-foreground/60'}`} />
                      </button>
                    </div>

                    <CardDescription className="line-clamp-2 text-xs leading-relaxed">
                      {tpl.description || tpl.template?.slice(0, 120)}
                    </CardDescription>
                  </CardHeader>

                  <CardContent className="flex-1 flex flex-col justify-end gap-3 pt-0">
                    {/* Variables */}
                    {tpl.variables && tpl.variables.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {tpl.variables.slice(0, 5).map((v: string) => (
                          <span key={v} className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                            {`{{${v}}}`}
                          </span>
                        ))}
                        {tpl.variables.length > 5 && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                            +{tpl.variables.length - 5} more
                          </span>
                        )}
                      </div>
                    )}

                    {/* Tags */}
                    {tpl.tags && tpl.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {tpl.tags.slice(0, 4).map((tag: string) => (
                          <button
                            key={tag}
                            onClick={() => toggleTag(tag)}
                            className={`text-[10px] px-1.5 py-0.5 rounded-full border transition-colors cursor-pointer ${
                              selectedTags.has(tag)
                                ? 'bg-primary text-primary-foreground border-primary'
                                : 'bg-transparent text-muted-foreground border-border/60 hover:border-primary/40 hover:text-foreground'
                            }`}
                          >
                            {tag}
                          </button>
                        ))}
                        {tpl.tags.length > 4 && (
                          <span className="text-[10px] text-muted-foreground/60 px-1">+{tpl.tags.length - 4}</span>
                        )}
                      </div>
                    )}

                    {/* Actions */}
                    <div className="flex items-center gap-1.5 mt-1 pt-2 border-t border-border/40">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => setRenderInput({ name: tpl.name, vars: {} })}
                        className="gap-1 text-xs h-8 px-2.5"
                      >
                        <Eye className="h-3 w-3" /> Preview
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => handleCopy(tpl.name, tpl.template || '')}
                        className="gap-1 text-xs h-8 px-2.5"
                      >
                        {copiedTemplate === tpl.name ? (
                          <><Check className="h-3 w-3 text-green-500" /> Copied</>
                        ) : (
                          <><Copy className="h-3 w-3" /> Copy</>
                        )}
                      </Button>
                      <Button
                        size="sm"
                        variant="default"
                        onClick={() => handleUseTemplate(tpl.template || '')}
                        className="gap-1 text-xs h-8 px-3 ml-auto"
                      >
                        <Sparkles className="h-3 w-3" /> Use
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        ) : (
          /* ─── LIST VIEW ─── */
          <div className="space-y-2">
            {filteredTemplates.map((tpl: any, idx: number) => {
              const source = getTemplateSource(tpl);
              const srcCfg = sourceConfig[source];
              const domain = tpl.domain || tpl.category || 'General';
              const domainColor = domainColors[domain] || 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300';
              const isExpanded = expandedTemplate === tpl.name;

              return (
                <Card
                  key={tpl.id || tpl.name || idx}
                  className={`transition-all duration-150 hover:shadow-md ${
                    favorites.has(tpl.name) ? 'ring-1 ring-yellow-400/30' : ''
                  }`}
                >
                  <div className="p-4">
                    <div className="flex items-start gap-3">
                      {/* Favorite star */}
                      <button
                        onClick={(e) => { e.stopPropagation(); toggleFavorite(tpl.name); }}
                        className="shrink-0 p-0.5 mt-0.5"
                      >
                        <Star className={`h-4 w-4 transition-colors ${favorites.has(tpl.name) ? 'fill-yellow-400 text-yellow-400' : 'text-muted-foreground/30 hover:text-yellow-400'}`} />
                      </button>

                      {/* Main content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap mb-1">
                          <h3 className="font-semibold text-sm">{tpl.name}</h3>
                          <div className={`text-[10px] font-medium px-2 py-0.5 rounded-full border ${srcCfg.bgColor} ${srcCfg.color}`}>
                            {srcCfg.label}
                          </div>
                          <div className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${domainColor}`}>
                            {domain.replace(/_/g, ' ')}
                          </div>
                        </div>
                        <p className="text-xs text-muted-foreground line-clamp-1 mb-2">
                          {tpl.description || tpl.template?.slice(0, 150)}
                        </p>

                        {/* Expandable content */}
                        {isExpanded && (
                          <div className="mt-3 space-y-3 animate-in fade-in slide-in-from-top-1 duration-200">
                            {tpl.variables && tpl.variables.length > 0 && (
                              <div className="flex flex-wrap gap-1">
                                <span className="text-xs text-muted-foreground mr-1">Variables:</span>
                                {tpl.variables.map((v: string) => (
                                  <span key={v} className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                                    {`{{${v}}}`}
                                  </span>
                                ))}
                              </div>
                            )}
                            {tpl.tags && tpl.tags.length > 0 && (
                              <div className="flex flex-wrap gap-1">
                                <span className="text-xs text-muted-foreground mr-1">Tags:</span>
                                {tpl.tags.map((tag: string) => (
                                  <button
                                    key={tag}
                                    onClick={() => toggleTag(tag)}
                                    className={`text-[10px] px-1.5 py-0.5 rounded-full border cursor-pointer ${
                                      selectedTags.has(tag)
                                        ? 'bg-primary text-primary-foreground border-primary'
                                        : 'bg-transparent text-muted-foreground border-border/60 hover:border-primary/40'
                                    }`}
                                  >
                                    {tag}
                                  </button>
                                ))}
                              </div>
                            )}
                            <pre className="p-3 bg-muted/50 rounded-lg text-xs whitespace-pre-wrap max-h-40 overflow-auto border border-border/40 font-mono">
                              {tpl.template?.slice(0, 500)}{(tpl.template?.length || 0) > 500 ? '...' : ''}
                            </pre>
                          </div>
                        )}
                      </div>

                      {/* Actions */}
                      <div className="flex items-center gap-1 shrink-0">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => setExpandedTemplate(isExpanded ? null : tpl.name)}
                          className="h-8 px-2 text-xs"
                        >
                          {isExpanded ? 'Less' : 'More'}
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => setRenderInput({ name: tpl.name, vars: {} })}
                          className="h-8 px-2"
                        >
                          <Eye className="h-3.5 w-3.5" />
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => handleCopy(tpl.name, tpl.template || '')}
                          className="h-8 px-2"
                        >
                          {copiedTemplate === tpl.name ? <Check className="h-3.5 w-3.5 text-green-500" /> : <Copy className="h-3.5 w-3.5" />}
                        </Button>
                        <Button
                          size="sm"
                          variant="default"
                          onClick={() => handleUseTemplate(tpl.template || '')}
                          className="gap-1 text-xs h-8"
                        >
                          <Sparkles className="h-3 w-3" /> Use
                        </Button>
                      </div>
                    </div>
                  </div>
                </Card>
              );
            })}
          </div>
        )}

        {/* ─── Render / Preview Dialog ────────────────────── */}
        <Dialog open={!!renderInput} onOpenChange={(open) => { if (!open) { setRenderInput(null); renderMut.reset(); } }}>
          <DialogContent className="sm:max-w-lg rounded-2xl">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <Eye className="h-4 w-4 text-primary" />
                Preview: {renderInput?.name}
              </DialogTitle>
              <DialogDescription>
                Fill in the template variables to see the rendered output.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-2">
              {(() => {
                const tpl = allTemplates.find((t: any) => t.name === renderInput?.name);
                const vars = tpl?.variables || [];
                const source = tpl ? getTemplateSource(tpl) : 'custom';
                const srcCfg = sourceConfig[source];
                return (
                  <>
                    {tpl && (
                      <div className={`text-xs font-medium px-2.5 py-1 rounded-md border inline-flex items-center gap-1.5 ${srcCfg.bgColor} ${srcCfg.color}`}>
                        {srcCfg.label}
                      </div>
                    )}
                    {vars.length > 0 && vars.map((v: string) => (
                      <div key={v} className="space-y-1.5">
                        <Label className="text-sm font-medium flex items-center gap-1.5">
                          <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">{`{{${v}}}`}</code>
                        </Label>
                        <Input
                          value={renderInput?.vars[v] || ''}
                          placeholder={`Enter ${v.replace(/_/g, ' ')}...`}
                          onChange={e => renderInput && setRenderInput({ ...renderInput, vars: { ...renderInput.vars, [v]: e.target.value } })}
                        />
                      </div>
                    ))}
                    {vars.length === 0 && (
                      <p className="text-sm text-muted-foreground">This template has no variables — it's ready to use as-is.</p>
                    )}
                    {renderMut.data && (
                      <div className="space-y-1.5">
                        <Label className="text-xs text-muted-foreground">Rendered Output</Label>
                        <pre className="p-3 bg-muted rounded-lg text-sm whitespace-pre-wrap max-h-56 overflow-auto border border-border/40">
                          {typeof renderMut.data === 'string' ? renderMut.data : JSON.stringify(renderMut.data, null, 2)}
                        </pre>
                      </div>
                    )}
                  </>
                );
              })()}
            </div>
            <DialogFooter className="gap-2">
              <Button variant="outline" onClick={() => { setRenderInput(null); renderMut.reset(); }}>Close</Button>
              <Button
                onClick={() => renderInput && renderMut.mutate({ template_id: renderInput.name, variables: renderInput.vars })}
                disabled={renderMut.isPending}
                className="gap-1.5"
              >
                {renderMut.isPending ? <LoadingSpinner size="sm" /> : <><Eye className="h-3.5 w-3.5" /> Render</>}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* ─── Create Template Dialog ─────────────────────── */}
        <Dialog open={isCreateOpen} onOpenChange={(open) => { if (!open) { setIsCreateOpen(false); setNewTemplate({ name: '', domain: '', template_text: '', description: '', variableInput: '' }); createMut.reset(); } }}>
          <DialogContent className="sm:max-w-lg rounded-2xl">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <Plus className="h-4 w-4 text-primary" />
                Create Template
              </DialogTitle>
              <DialogDescription>Save a reusable prompt template. Use {'{{variable}}'} syntax for dynamic parts.</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-2">
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <Label htmlFor="tpl-name">Name *</Label>
                  <Input id="tpl-name" placeholder="e.g. Code Review" value={newTemplate.name} onChange={e => setNewTemplate(p => ({ ...p, name: e.target.value }))} />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="tpl-domain">Domain *</Label>
                  <Input id="tpl-domain" placeholder="e.g. software_engineering" value={newTemplate.domain} onChange={e => setNewTemplate(p => ({ ...p, domain: e.target.value }))} />
                </div>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="tpl-desc">Description</Label>
                <Input id="tpl-desc" placeholder="Brief description of what this template does" value={newTemplate.description} onChange={e => setNewTemplate(p => ({ ...p, description: e.target.value }))} />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="tpl-text">Template Text *</Label>
                <Textarea id="tpl-text" rows={6} placeholder={"Review this {{language}} code for best practices:\n\n{{code}}"} value={newTemplate.template_text} onChange={e => setNewTemplate(p => ({ ...p, template_text: e.target.value }))} className="font-mono text-sm" />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="tpl-vars">Variables <span className="text-muted-foreground font-normal">(comma-separated)</span></Label>
                <Input id="tpl-vars" placeholder="language, code" value={newTemplate.variableInput} onChange={e => setNewTemplate(p => ({ ...p, variableInput: e.target.value }))} />
              </div>
              {newTemplate.template_text && (
                <div className="p-3 bg-muted/50 rounded-lg border border-border/40">
                  <p className="text-xs font-medium text-muted-foreground mb-1.5">Preview</p>
                  <pre className="text-xs whitespace-pre-wrap text-foreground/80 font-mono">{newTemplate.template_text.slice(0, 300)}{newTemplate.template_text.length > 300 ? '...' : ''}</pre>
                </div>
              )}
            </div>
            <DialogFooter className="gap-2">
              <Button variant="outline" onClick={() => setIsCreateOpen(false)}>Cancel</Button>
              <Button
                onClick={() => {
                  if (!newTemplate.name.trim() || !newTemplate.domain.trim() || !newTemplate.template_text.trim()) return;
                  const variables = newTemplate.variableInput.split(',').map(v => v.trim()).filter(Boolean);
                  createMut.mutate({
                    name: newTemplate.name.trim(),
                    domain: newTemplate.domain.trim(),
                    template_text: newTemplate.template_text.trim(),
                    description: newTemplate.description.trim() || undefined,
                    variables: variables.length > 0 ? variables : undefined,
                  }, {
                    onSuccess: () => {
                      setIsCreateOpen(false);
                      setNewTemplate({ name: '', domain: '', template_text: '', description: '', variableInput: '' });
                    }
                  });
                }}
                disabled={createMut.isPending || !newTemplate.name.trim() || !newTemplate.domain.trim() || !newTemplate.template_text.trim()}
                className="gap-1.5"
              >
                {createMut.isPending ? <LoadingSpinner size="sm" /> : <><Plus className="h-3.5 w-3.5" /> Create Template</>}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </AppLayout>
  );
}
