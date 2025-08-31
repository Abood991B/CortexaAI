import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'sonner';
import { Layout } from '@/components/layout/Layout';
import { Dashboard } from '@/pages/Dashboard';
import { PromptProcessor } from '@/pages/PromptProcessor';
import { Prompts } from '@/pages/Prompts';
import { Templates } from '@/pages/Templates';
import { Workflows } from '@/pages/Workflows';
import { Analytics } from '@/pages/Analytics';
import { Domains } from '@/pages/Domains';
import { Experiments } from '@/pages/Experiments';
import { Settings } from '@/pages/Settings';
import { SystemHealth } from '@/pages/SystemHealth';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-background text-foreground">
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<Navigate to="/dashboard" replace />} />
              <Route path="dashboard" element={<Dashboard />} />
              <Route path="processor" element={<PromptProcessor />} />
              <Route path="prompts" element={<Prompts />} />
              <Route path="templates" element={<Templates />} />
              <Route path="workflows" element={<Workflows />} />
              <Route path="analytics" element={<Analytics />} />
              <Route path="domains" element={<Domains />} />
              <Route path="experiments" element={<Experiments />} />
              <Route path="settings" element={<Settings />} />
              <Route path="system-health" element={<SystemHealth />} />
              {/* 404 Route */}
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Route>
          </Routes>
          <Toaster position="top-right" />
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
