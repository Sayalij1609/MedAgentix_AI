import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false, // Prevents aggressive clinical re-fetches
      retry: 1,                    // Retry failed connections once before alerting user
      staleTime: 5 * 60 * 1000,     // 5 minutes baseline cache fresh time
    },
  },
});
