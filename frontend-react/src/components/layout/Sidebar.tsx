import { NavLink, useLocation } from 'react-router-dom';
import { Bot, BarChart3, Layers, Activity, Menu, ChevronLeft, Keyboard } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/utils';

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  children?: React.ReactNode;
  bottomContent?: React.ReactNode;
}

const navItems = [
  { to: '/', icon: Bot, label: 'Prompt Processor', matchPaths: ['/', '/processor'], shortcut: 'Alt+1' },
  { to: '/dashboard', icon: BarChart3, label: 'Overview', matchPaths: ['/dashboard'], shortcut: 'Alt+2' },
  { to: '/templates', icon: Layers, label: 'Templates', matchPaths: ['/templates'], shortcut: 'Alt+3' },
  { to: '/system-health', icon: Activity, label: 'System Health', matchPaths: ['/system-health'], shortcut: 'Alt+4' },
];

export function Sidebar({ collapsed, onToggle, children, bottomContent }: SidebarProps) {
  const location = useLocation();

  return (
    <aside
      className={cn(
        'relative flex flex-col h-full border-r border-border/60 bg-sidebar transition-all duration-300 ease-in-out',
        collapsed ? 'w-[68px]' : 'w-64'
      )}
    >
      {/* Header */}
      <div className="flex items-center h-16 px-4 border-b border-border/40">
        {!collapsed && (
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/10">
              <img src="/Cortexa Logo.png" alt="Cortexa" className="w-6 h-6" />
            </div>
            <span className="text-base font-semibold text-foreground tracking-tight truncate">
              Cortexa
            </span>
          </div>
        )}
        {collapsed && (
          <div className="flex items-center justify-center w-full">
            <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/10">
              <img src="/Cortexa Logo.png" alt="Cortexa" className="w-6 h-6" />
            </div>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggle}
          className={cn(
            'h-8 w-8 shrink-0 text-muted-foreground hover:text-foreground hover:bg-accent/60',
            collapsed && 'absolute -right-3 top-5 z-10 rounded-full border bg-background shadow-sm h-6 w-6'
          )}
        >
          {collapsed ? <Menu className="h-3.5 w-3.5" /> : <ChevronLeft className="h-4 w-4" />}
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex-shrink-0 px-3 pt-4 pb-2 space-y-1">
        {navItems.map(({ to, icon: Icon, label, matchPaths, shortcut }) => {
          const isActive = matchPaths.includes(location.pathname);
          return (
            <NavLink
              key={to}
              to={to}
              className={cn(
                'group flex items-center gap-3 px-3 py-2.5 text-sm font-medium rounded-lg transition-all duration-200',
                isActive
                  ? 'bg-primary/10 text-primary shadow-sm'
                  : 'text-muted-foreground hover:bg-accent/60 hover:text-foreground',
                collapsed && 'justify-center px-2'
              )}
              title={collapsed ? `${label} (${shortcut})` : undefined}
            >
              <Icon className={cn('h-[18px] w-[18px] shrink-0', isActive && 'text-primary')} />
              {!collapsed && (
                <div className="flex items-center justify-between flex-1 min-w-0">
                  <span className="truncate">{label}</span>
                  <kbd className="hidden group-hover:inline-block text-[10px] text-muted-foreground/60 bg-muted/80 px-1.5 py-0.5 rounded font-mono">{shortcut}</kbd>
                </div>
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* Keyboard shortcut hint */}
      {!collapsed && !children && (
        <div className="px-3 mt-2">
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted/30 text-[10px] text-muted-foreground/50">
            <Keyboard className="h-3 w-3" />
            <span>Press <kbd className="font-mono bg-muted/80 px-1 rounded">?</kbd> for shortcuts</span>
          </div>
        </div>
      )}

      {/* Optional extra content (e.g., chat sessions) */}
      {children && (
        <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
          {children}
        </div>
      )}

      {/* Bottom content */}
      {bottomContent && !collapsed && (
        <div className="mt-auto px-3 pb-4 pt-2 border-t border-border/40">
          {bottomContent}
        </div>
      )}
    </aside>
  );
}
