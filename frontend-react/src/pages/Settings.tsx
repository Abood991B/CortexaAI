import React, { useState } from 'react';
import { Settings as SettingsIcon, Save, Key, Database, Bell, Shield } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';

export function Settings() {
  const [apiKeys, setApiKeys] = useState({
    openai: '',
    anthropic: '',
    google: '',
    langsmith: ''
  });

  const [systemConfig, setSystemConfig] = useState({
    maxRetries: 3,
    timeout: 30,
    logLevel: 'info',
    enableAnalytics: true,
    enableNotifications: true
  });

  const handleSaveApiKeys = () => {
    toast.success('API keys saved successfully');
  };

  const handleSaveSystemConfig = () => {
    toast.success('System configuration saved');
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold flex items-center">
          <SettingsIcon className="mr-3 h-8 w-8 text-primary" />
          System Settings
        </h1>
        <p className="text-muted-foreground">
          Configure API keys, system parameters, and preferences
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* API Keys */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Key className="mr-2 h-5 w-5" />
              API Keys
            </CardTitle>
            <CardDescription>
              Configure your LLM provider API keys
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium mb-2 block">OpenAI API Key</label>
              <Input
                type="password"
                value={apiKeys.openai}
                onChange={(e) => setApiKeys({...apiKeys, openai: e.target.value})}
                placeholder="sk-..."
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">Anthropic API Key</label>
              <Input
                type="password"
                value={apiKeys.anthropic}
                onChange={(e) => setApiKeys({...apiKeys, anthropic: e.target.value})}
                placeholder="sk-ant-..."
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">Google API Key</label>
              <Input
                type="password"
                value={apiKeys.google}
                onChange={(e) => setApiKeys({...apiKeys, google: e.target.value})}
                placeholder="AIza..."
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">LangSmith API Key</label>
              <Input
                type="password"
                value={apiKeys.langsmith}
                onChange={(e) => setApiKeys({...apiKeys, langsmith: e.target.value})}
                placeholder="ls__..."
              />
            </div>
            <Button onClick={handleSaveApiKeys} className="w-full">
              <Save className="mr-2 h-4 w-4" />
              Save API Keys
            </Button>
          </CardContent>
        </Card>

        {/* System Configuration */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Database className="mr-2 h-5 w-5" />
              System Configuration
            </CardTitle>
            <CardDescription>
              Adjust system behavior and performance settings
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Max Retries</label>
              <Input
                type="number"
                value={systemConfig.maxRetries}
                onChange={(e) => setSystemConfig({...systemConfig, maxRetries: parseInt(e.target.value)})}
                min="1"
                max="10"
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">Timeout (seconds)</label>
              <Input
                type="number"
                value={systemConfig.timeout}
                onChange={(e) => setSystemConfig({...systemConfig, timeout: parseInt(e.target.value)})}
                min="5"
                max="300"
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">Log Level</label>
              <select
                value={systemConfig.logLevel}
                onChange={(e) => setSystemConfig({...systemConfig, logLevel: e.target.value})}
                className="w-full p-2 border border-input rounded-md bg-background"
              >
                <option value="debug">Debug</option>
                <option value="info">Info</option>
                <option value="warning">Warning</option>
                <option value="error">Error</option>
              </select>
            </div>
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="enableAnalytics"
                  checked={systemConfig.enableAnalytics}
                  onChange={(e) => setSystemConfig({...systemConfig, enableAnalytics: e.target.checked})}
                />
                <label htmlFor="enableAnalytics" className="text-sm">Enable Analytics</label>
              </div>
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="enableNotifications"
                  checked={systemConfig.enableNotifications}
                  onChange={(e) => setSystemConfig({...systemConfig, enableNotifications: e.target.checked})}
                />
                <label htmlFor="enableNotifications" className="text-sm">Enable Notifications</label>
              </div>
            </div>
            <Button onClick={handleSaveSystemConfig} className="w-full">
              <Save className="mr-2 h-4 w-4" />
              Save Configuration
            </Button>
          </CardContent>
        </Card>

        {/* Security Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Shield className="mr-2 h-5 w-5" />
              Security & Privacy
            </CardTitle>
            <CardDescription>
              Manage security settings and data privacy
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <input type="checkbox" id="encryptData" defaultChecked />
                <label htmlFor="encryptData" className="text-sm">Encrypt stored data</label>
              </div>
              <div className="flex items-center space-x-2">
                <input type="checkbox" id="auditLogs" defaultChecked />
                <label htmlFor="auditLogs" className="text-sm">Enable audit logging</label>
              </div>
              <div className="flex items-center space-x-2">
                <input type="checkbox" id="anonymizeData" />
                <label htmlFor="anonymizeData" className="text-sm">Anonymize analytics data</label>
              </div>
            </div>
            <Button variant="outline" className="w-full">
              Export Data
            </Button>
            <Button variant="destructive" className="w-full">
              Clear All Data
            </Button>
          </CardContent>
        </Card>

        {/* Notification Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Bell className="mr-2 h-5 w-5" />
              Notifications
            </CardTitle>
            <CardDescription>
              Configure notification preferences
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <input type="checkbox" id="emailNotifs" defaultChecked />
                <label htmlFor="emailNotifs" className="text-sm">Email notifications</label>
              </div>
              <div className="flex items-center space-x-2">
                <input type="checkbox" id="workflowComplete" defaultChecked />
                <label htmlFor="workflowComplete" className="text-sm">Workflow completion</label>
              </div>
              <div className="flex items-center space-x-2">
                <input type="checkbox" id="errorAlerts" defaultChecked />
                <label htmlFor="errorAlerts" className="text-sm">Error alerts</label>
              </div>
              <div className="flex items-center space-x-2">
                <input type="checkbox" id="weeklyReports" />
                <label htmlFor="weeklyReports" className="text-sm">Weekly reports</label>
              </div>
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">Email Address</label>
              <Input
                type="email"
                placeholder="your@email.com"
              />
            </div>
            <Button className="w-full">
              Save Notification Settings
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
