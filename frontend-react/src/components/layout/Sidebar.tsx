import { NavLink, useLocation } from 'react-router-dom';
import { 
  Brain, 
  FileText, 
  BarChart3, 
  Settings, 
  Zap, 
  TestTube, 
  Database,
  Home,
  Users,
  Activity
} from 'lucide-react';
import { cn } from '@/utils';
import { Badge } from '@/components/ui/badge';
import { useHealth } from '@/hooks/useApi';

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: Home },
  { name: 'Prompt Processor', href: '/processor', icon: Brain },
  { name: 'Prompt Library', href: '/prompts', icon: FileText },
  { name: 'Templates', href: '/templates', icon: Database },
  { name: 'Workflows', href: '/workflows', icon: Zap },
  { name: 'Analytics', href: '/analytics', icon: BarChart3 },
  { name: 'Domains', href: '/domains', icon: Users },
  { name: 'Experiments', href: '/experiments', icon: TestTube },
  { name: 'System Health', href: '/system-health', icon: Activity },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export function Sidebar() {
  const location = useLocation();
  const { data: health } = useHealth();

  return (
    <div className="flex flex-col w-64 bg-card border-r border-border">
      {/* Logo */}
      <div className="flex items-center px-6 py-4 border-b border-border">
        <Brain className="h-8 w-8 text-primary" />
        <div className="ml-3">
          <h1 className="text-lg font-semibold">Prompt Engineer</h1>
          <p className="text-xs text-muted-foreground">Multi-Agent System</p>
        </div>
      </div>

      {/* System Status */}
      <div className="px-6 py-3 border-b border-border">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">System Status</span>
          <Badge 
            variant={health?.status === 'healthy' ? 'success' : 'destructive'}
            className="text-xs"
          >
            {health?.status || 'Unknown'}
          </Badge>
        </div>
        {health && (
          <div className="mt-2 text-xs text-muted-foreground">
            <div>Uptime: {Math.floor((health.uptime_seconds || 0) / 3600)}h</div>
            <div>Workflows: {health.metrics.total_workflows}</div>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-4 space-y-1">
        {navigation.map((item) => {
          const isActive = location.pathname === item.href;
          return (
            <NavLink
              key={item.name}
              to={item.href}
              className={cn(
                'flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors',
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:text-foreground hover:bg-accent'
              )}
            >
              <item.icon className="mr-3 h-4 w-4" />
              {item.name}
            </NavLink>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-6 py-4 border-t border-border">
        <div className="text-xs text-muted-foreground">
          <div>Version 1.0.0</div>
          <div>© 2024 Prompt Engineer</div>
        </div>
      </div>
    </div>
  );
}
