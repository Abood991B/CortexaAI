import { NavLink, useLocation } from 'react-router-dom';
import { 
  Brain, 
  Zap, 
  Home,
  Activity,
  Sparkles,
  Target,
  TrendingUp
} from 'lucide-react';
import { cn } from '@/utils';
import { Badge } from '@/components/ui/badge';
import { useHealth } from '@/hooks/useApi';

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: Home, color: 'text-blue-600', bgColor: 'bg-blue-50' },
  { name: 'Prompt Processor', href: '/processor', icon: Brain, color: 'text-purple-600', bgColor: 'bg-purple-50' },
  { name: 'Workflows', href: '/workflows', icon: Zap, color: 'text-yellow-600', bgColor: 'bg-yellow-50' },
  { name: 'System Health', href: '/system-health', icon: Activity, color: 'text-red-600', bgColor: 'bg-red-50' },
];

export function Sidebar() {
  const location = useLocation();
  const { data: health } = useHealth();

  return (
    <div className="flex flex-col w-72 bg-gradient-to-b from-slate-50 to-white border-r border-gray-200 shadow-lg">
      {/* Enhanced Logo */}
      <div className="flex items-center px-6 py-6 border-b border-gray-200 bg-gradient-to-r from-blue-600 to-purple-600">
        <div className="p-2 bg-white rounded-lg shadow-sm">
          <Brain className="h-6 w-6 text-blue-600" />
        </div>
        <div className="ml-4">
          <h1 className="text-xl font-bold text-white">Prompt Engineer</h1>
          <p className="text-xs text-blue-100 font-medium">AI Multi-Agent System</p>
        </div>
      </div>

      {/* Enhanced System Status */}
      <div className="mx-4 my-4 p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border border-green-200">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm font-semibold text-gray-800">System Status</span>
          <Badge 
            variant={health?.status === 'healthy' ? 'default' : 'destructive'}
            className={`text-xs font-medium ${
              health?.status === 'healthy' 
                ? 'bg-green-100 text-green-700 border-green-300' 
                : 'bg-red-100 text-red-700 border-red-300'
            }`}
          >
            {health?.status === 'healthy' ? '🟢 Healthy' : '🔴 ' + (health?.status || 'Unknown')}
          </Badge>
        </div>
        {health && (
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex items-center gap-1">
              <Activity className="h-3 w-3 text-green-600" />
              <span className="text-gray-600">{Math.floor((health.uptime_seconds || 0) / 3600)}h uptime</span>
            </div>
            <div className="flex items-center gap-1">
              <Zap className="h-3 w-3 text-blue-600" />
              <span className="text-gray-600">{health.metrics?.total_workflows || 0} workflows</span>
            </div>
          </div>
        )}
      </div>

      {/* Enhanced Navigation */}
      <nav className="flex-1 px-4 py-2 space-y-2">
        {navigation.map((item) => {
          const isActive = location.pathname === item.href;
          return (
            <NavLink
              key={item.name}
              to={item.href}
              className={cn(
                'group flex items-center px-4 py-3 text-sm font-medium rounded-xl transition-all duration-200 hover:scale-105',
                isActive
                  ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg transform scale-105'
                  : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900'
              )}
            >
              <div className={cn(
                'p-2 rounded-lg mr-3 transition-colors duration-200',
                isActive 
                  ? 'bg-white/20' 
                  : `${item.bgColor} group-hover:${item.bgColor}`
              )}>
                <item.icon className={cn(
                  'h-4 w-4 transition-colors duration-200',
                  isActive ? 'text-white' : `${item.color} group-hover:${item.color}`
                )} />
              </div>
              <span className="font-medium">{item.name}</span>
              {isActive && (
                <div className="ml-auto">
                  <Sparkles className="h-4 w-4 text-white/80" />
                </div>
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* Enhanced Footer */}
      <div className="mx-4 mb-4 p-4 bg-gradient-to-r from-gray-50 to-slate-50 rounded-xl border border-gray-200">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Target className="h-4 w-4 text-blue-600" />
            <span className="text-sm font-semibold text-gray-800">AI System v1.0</span>
          </div>
          <Badge variant="outline" className="text-xs bg-blue-50 text-blue-700 border-blue-200">
            Active
          </Badge>
        </div>
        <div className="text-xs text-gray-500 space-y-1">
          <div className="flex items-center gap-1">
            <TrendingUp className="h-3 w-3" />
            <span>Multi-Agent Architecture</span>
          </div>
          <div>© 2024 Prompt Engineer Agent</div>
        </div>
      </div>
    </div>
  );
}
