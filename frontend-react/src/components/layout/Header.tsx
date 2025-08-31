import { Bell, Search, User, Menu, Sun, Moon, ArrowLeft } from 'lucide-react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { useStats } from '@/hooks/useApi';
import { formatNumber } from '@/utils';

interface HeaderProps {
  onMenuClick?: () => void;
  isDarkMode?: boolean;
  onThemeToggle?: () => void;
}

export function Header({ onMenuClick, isDarkMode, onThemeToggle }: HeaderProps) {
  const { data: stats } = useStats();
  const navigate = useNavigate();
  const location = useLocation();

  const handleBackClick = () => {
    // If we have history and not on the first page, go back
    if (window.history.length > 1) {
      navigate(-1);
    } else {
      // Otherwise go to dashboard as fallback
      navigate('/dashboard');
    }
  };

  return (
    <header className="flex items-center justify-between px-6 py-4 bg-card border-b border-border">
      {/* Left side */}
      <div className="flex items-center space-x-4">
        <Button
          variant="ghost"
          size="icon"
          onClick={onMenuClick}
          className="md:hidden"
        >
          <Menu className="h-5 w-5" />
        </Button>
        
        {location.pathname !== '/dashboard' && (
          <Button
            variant="ghost"
            size="icon"
            onClick={handleBackClick}
            className="hidden md:flex"
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>
        )}
        
        <div className="hidden md:flex items-center space-x-6">
          <div className="flex items-center space-x-2">
            <span className="text-sm text-muted-foreground">Total Workflows:</span>
            <Badge variant="secondary">
              {formatNumber(stats?.total_workflows || 0)}
            </Badge>
          </div>
          
          <div className="flex items-center space-x-2">
            <span className="text-sm text-muted-foreground">Success Rate:</span>
            <Badge variant="success">
              {((stats?.success_rate || 0) * 100).toFixed(1)}%
            </Badge>
          </div>
          
          <div className="flex items-center space-x-2">
            <span className="text-sm text-muted-foreground">Avg Quality:</span>
            <Badge variant="info">
              {(stats?.average_quality_score || 0).toFixed(2)}
            </Badge>
          </div>
        </div>
      </div>

      {/* Center - Search */}
      <div className="flex-1 max-w-md mx-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search prompts, templates, workflows..."
            className="pl-10"
          />
        </div>
      </div>

      {/* Right side */}
      <div className="flex items-center space-x-2">
        <Button
          variant="ghost"
          size="icon"
          onClick={onThemeToggle}
        >
          {isDarkMode ? (
            <Sun className="h-5 w-5" />
          ) : (
            <Moon className="h-5 w-5" />
          )}
        </Button>
        
        <Button variant="ghost" size="icon" className="relative">
          <Bell className="h-5 w-5" />
          <Badge 
            variant="destructive" 
            className="absolute -top-1 -right-1 h-5 w-5 rounded-full p-0 flex items-center justify-center text-xs"
          >
            3
          </Badge>
        </Button>
        
        <Button variant="ghost" size="icon">
          <User className="h-5 w-5" />
        </Button>
      </div>
    </header>
  );
}
