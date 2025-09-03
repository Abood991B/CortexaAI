import { X, CheckCircle, AlertTriangle, Info, AlertCircle, Check, Trash2 } from 'lucide-react';
import { Button } from './button';
import { Badge } from './badge';
import { formatDistanceToNow } from 'date-fns';
import type { Notification } from '@/hooks/useNotifications';
import { cn } from '@/utils';

interface NotificationsDropdownProps {
  notifications: Notification[];
  isOpen: boolean;
  unreadCount: number;
  onClose: () => void;
  onMarkAsRead: (id: string) => void;
  onMarkAllAsRead: () => void;
  onRemove: (id: string) => void;
  onClearAll: () => void;
}

const getNotificationIcon = (type: Notification['type']) => {
  switch (type) {
    case 'success':
      return <CheckCircle className="h-4 w-4 text-green-500" />;
    case 'warning':
      return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
    case 'error':
      return <AlertCircle className="h-4 w-4 text-red-500" />;
    default:
      return <Info className="h-4 w-4 text-blue-500" />;
  }
};

const getNotificationBorderColor = (type: Notification['type']) => {
  switch (type) {
    case 'success':
      return 'border-l-green-500';
    case 'warning':
      return 'border-l-yellow-500';
    case 'error':
      return 'border-l-red-500';
    default:
      return 'border-l-blue-500';
  }
};

export function NotificationsDropdown({
  notifications,
  isOpen,
  unreadCount,
  onClose,
  onMarkAsRead,
  onMarkAllAsRead,
  onRemove,
  onClearAll
}: NotificationsDropdownProps) {
  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm" 
        onClick={onClose}
      />

      {/* Dropdown */}
      <div className="absolute right-0 top-full mt-2 w-96 bg-popover text-popover-foreground shadow-lg border rounded-lg z-50 max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="p-4 border-b bg-muted/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <h3 className="font-semibold text-lg">Notifications</h3>
              {unreadCount > 0 && (
                <Badge variant="destructive" className="text-xs">
                  {unreadCount}
                </Badge>
              )}
            </div>
            <Button variant="ghost" size="sm" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>
          
          {/* Actions */}
          {notifications.length > 0 && (
            <div className="flex items-center space-x-2 mt-3">
              {unreadCount > 0 && (
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={onMarkAllAsRead}
                  className="text-xs"
                >
                  <Check className="h-3 w-3 mr-1" />
                  Mark all read
                </Button>
              )}
              <Button 
                variant="outline" 
                size="sm" 
                onClick={onClearAll}
                className="text-xs text-destructive hover:text-destructive"
              >
                <Trash2 className="h-3 w-3 mr-1" />
                Clear all
              </Button>
            </div>
          )}
        </div>

        {/* Notifications List */}
        <div className="flex-1 overflow-y-auto">
          {notifications.length === 0 ? (
            <div className="p-6 text-center text-muted-foreground">
              <div className="mb-2">
                📭
              </div>
              <p className="text-sm">No notifications yet</p>
              <p className="text-xs text-muted-foreground mt-1">
                You'll see updates about workflows, system status, and more here.
              </p>
            </div>
          ) : (
            <div className="divide-y divide-border">
              {notifications.map((notification) => (
                <NotificationItem
                  key={notification.id}
                  notification={notification}
                  onMarkAsRead={onMarkAsRead}
                  onRemove={onRemove}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}

interface NotificationItemProps {
  notification: Notification;
  onMarkAsRead: (id: string) => void;
  onRemove: (id: string) => void;
}

function NotificationItem({ notification, onMarkAsRead, onRemove }: NotificationItemProps) {
  const handleClick = () => {
    if (!notification.read) {
      onMarkAsRead(notification.id);
    }
    if (notification.action) {
      notification.action.onClick();
    }
  };

  return (
    <div 
      className={cn(
        "p-4 hover:bg-muted/50 transition-colors border-l-4 cursor-pointer group",
        getNotificationBorderColor(notification.type),
        !notification.read && "bg-primary/5"
      )}
      onClick={handleClick}
    >
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0 mt-0.5">
          {getNotificationIcon(notification.type)}
        </div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <p className={cn(
                "text-sm font-medium",
                !notification.read && "font-semibold"
              )}>
                {notification.title}
              </p>
              <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                {notification.message}
              </p>
              
              {notification.action && (
                <Button 
                  variant="link" 
                  size="sm" 
                  className="p-0 h-auto text-xs mt-2 text-primary"
                >
                  {notification.action.label}
                </Button>
              )}
            </div>
            
            <div className="flex items-center space-x-1 ml-2">
              {!notification.read && (
                <div className="w-2 h-2 bg-primary rounded-full" title="Unread" />
              )}
              
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  onRemove(notification.id);
                }}
                className="opacity-0 group-hover:opacity-100 transition-opacity p-1 h-auto"
              >
                <X className="h-3 w-3" />
              </Button>
            </div>
          </div>
          
          <p className="text-xs text-muted-foreground mt-2">
            {formatDistanceToNow(notification.timestamp, { addSuffix: true })}
          </p>
        </div>
      </div>
    </div>
  );
}
