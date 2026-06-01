import React from 'react';
import { Outlet } from 'react-router-dom';
import { QueryClientProvider } from '@tanstack/react-query';
import { queryClient } from '../services/query-client';
import { AuthProvider } from '../context/auth-context';
import { ThemeProvider } from '../context/theme-context';

export const MainLayout: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <AuthProvider>
          <div className="min-h-screen bg-background text-foreground transition-colors duration-300">
            <Outlet />
          </div>
        </AuthProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
};
export default MainLayout;
