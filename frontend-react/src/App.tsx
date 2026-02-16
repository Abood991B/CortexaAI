import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'sonner';
import { PromptProcessor } from '@/pages/PromptProcessor';
import { Dashboard } from '@/pages/Dashboard';
import { Templates } from '@/pages/Templates';
import { useEffect } from 'react';

// Global error handler for audio play() interruptions
const handleAudioErrors = () => {
  // Suppress common audio autoplay errors
  window.addEventListener('unhandledrejection', (event) => {
    if (event.reason?.name === 'AbortError' && 
        event.reason?.message?.includes('play()')) {
      event.preventDefault();
      console.debug('Audio notification blocked by browser policy');
    }
  });
};

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes (default)
      refetchOnWindowFocus: true, // Refresh when returning to app
      retry: 1,
    },
  },
});

function App() {
  useEffect(() => {
    handleAudioErrors();
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <Router
        future={{
          v7_startTransition: true,
          v7_relativeSplatPath: true,
        }}
      >
        <div className="min-h-screen bg-background text-foreground">
          <Routes>
            <Route index element={<PromptProcessor />} />
            <Route path="processor" element={<Navigate to="/" replace />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="templates" element={<Templates />} />
            <Route path="system-health" element={<Navigate to="/dashboard" replace />} />
            {/* 404 Route */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
          <Toaster
            position="top-right"
            richColors
            closeButton
            duration={3000}
            gap={8}
            visibleToasts={3}
            toastOptions={{
              style: { fontSize: '0.875rem' },
            }}
          />
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
