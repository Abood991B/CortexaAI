import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Bell, Keyboard } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { NotificationsDropdown } from '@/components/ui/notifications';
import { useNotifications } from '@/hooks/useNotifications';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import { Sidebar } from './Sidebar';

interface PageHeaderProps {
  title: string;
  icon?: React.ReactNode;
  actions?: React.ReactNode;
}

export function PageHeader({ title, icon, actions }: PageHeaderProps) {
  const {
    notifications,
    unreadCount,
    isOpen: isDropdownOpen,
    toggle: toggleDropdown,
    markAsRead,
    markAllAsRead,
    removeNotification,
    clearAll,
  } = useNotifications();

  return (
    <header className="sticky top-0 z-30 flex items-center justify-between h-16 px-6 border-b border-border/60 bg-background/80 backdrop-blur-md">
      <div className="flex items-center gap-2.5">
        {icon}
        <h1 className="text-lg font-semibold tracking-tight">{title}</h1>
      </div>
      <div className="flex items-center gap-2">
        {actions}
        <div className="relative">
          <Button variant="ghost" size="icon" onClick={toggleDropdown} className="relative h-9 w-9">
            <Bell className="h-4 w-4" />
            {unreadCount > 0 && (
              <span className="absolute top-1.5 right-1.5 h-2 w-2 bg-destructive rounded-full ring-2 ring-background" />
            )}
          </Button>
          <NotificationsDropdown
            notifications={notifications}
            isOpen={isDropdownOpen}
            unreadCount={unreadCount}
            onClose={toggleDropdown}
            onMarkAsRead={markAsRead}
            onMarkAllAsRead={markAllAsRead}
            onRemove={removeNotification}
            onClearAll={clearAll}
          />
        </div>
      </div>
    </header>
  );
}

/* ── Global Shortcuts ──────────────────────────────────────────── */
const GLOBAL_SHORTCUTS = [
  { keys: ['Alt', '1'], description: 'Prompt Processor' },
  { keys: ['Alt', '2'], description: 'Overview' },
  { keys: ['Alt', '3'], description: 'Templates' },
  { keys: ['Alt', '4'], description: 'System Health' },
  { keys: ['?'], description: 'Show this dialog' },
];

function GlobalShortcutsDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md rounded-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Keyboard className="h-5 w-5 text-primary" />
            Keyboard Shortcuts
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-2 py-2">
          {GLOBAL_SHORTCUTS.map((s, i) => (
            <div key={i} className="flex items-center justify-between px-2 py-1.5 rounded-lg hover:bg-muted/50">
              <span className="text-sm text-muted-foreground">{s.description}</span>
              <div className="flex items-center gap-1">
                {s.keys.map((k, j) => (
                  <span key={j}>
                    <kbd className="px-2 py-0.5 text-xs font-mono bg-muted border border-border/60 rounded-md shadow-sm">{k}</kbd>
                    {j < s.keys.length - 1 && <span className="text-muted-foreground mx-0.5">+</span>}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
        <DialogFooter>
          <Button onClick={() => onOpenChange(false)} className="rounded-xl">Close</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

interface AppLayoutProps {
  children: React.ReactNode;
  sidebarContent?: React.ReactNode;
  sidebarBottomContent?: React.ReactNode;
}

export function AppLayout({ children, sidebarContent, sidebarBottomContent }: AppLayoutProps) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const navigate = useNavigate();

  // Global keyboard shortcuts — work on every page
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      const inInput = tag === 'INPUT' || tag === 'TEXTAREA';

      // ? key — show shortcuts dialog (only when not typing)
      if (e.key === '?' && !inInput && !e.ctrlKey && !e.metaKey && !e.altKey) {
        e.preventDefault();
        setShortcutsOpen(true);
        return;
      }

      // Alt+1-4 — navigate pages
      if (e.altKey && !e.ctrlKey && !e.metaKey && ['1', '2', '3', '4'].includes(e.key)) {
        e.preventDefault();
        const routes = ['/', '/dashboard', '/templates', '/system-health'];
        navigate(routes[parseInt(e.key) - 1]);
        return;
      }
    };
    window.addEventListener('keydown', handler, true);
    return () => window.removeEventListener('keydown', handler, true);
  }, [navigate]);

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
        bottomContent={sidebarBottomContent}
      >
        {sidebarContent}
      </Sidebar>
      <main className="flex-1 flex flex-col overflow-hidden">
        {children}
      </main>
      <GlobalShortcutsDialog open={shortcutsOpen} onOpenChange={setShortcutsOpen} />
    </div>
  );
}

export { Sidebar };
