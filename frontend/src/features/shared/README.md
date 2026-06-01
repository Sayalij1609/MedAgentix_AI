# Shared Feature Module — Shared (shared)

This module holds components, hooks, services, types, and constants that are shared across two or more specific business features (e.g. unified UI wrappers, network retry rules).

## Directory Responsibility Mapping

- `/components`: Houses shared widgets like `Sidebar.tsx`, `Header.tsx`, and Radix UI primitives.
- `/services`: Global Axios clients and base interceptor configurations (`api-client.ts`).
- `/hooks`: Custom global React hooks (e.g. `useDebounce.ts`, `useLocalStorage.ts`).
- `/types`: Shared models like basic demographic formats or token sessions.
- `/constants`: Global clinical limits, API routes lists, and theme parameters.
