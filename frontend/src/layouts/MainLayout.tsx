import React from 'react';
import { Outlet } from 'react-router-dom';
import { QueryClientProvider } from '@tanstack/react-query';
import { queryClient } from '../services/query-client';
import { AuthProvider } from '../context/auth-context';
import { ThemeProvider } from '../context/theme-context';
import { ToastProvider } from '../context/toast-context';

export const MainLayout: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <ToastProvider>
          <AuthProvider>
            <div className="min-h-screen bg-background text-foreground transition-colors duration-300">
              <Outlet />
            </div>
          </AuthProvider>
        </ToastProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
};
export default MainLayout;
