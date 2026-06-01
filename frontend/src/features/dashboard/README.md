# Feature Module — Dashboard (dashboard)

This feature module governs dynamic workspace aggregations, greeting banners, action shortcut hubs, and recent consultation grids.

## Directory Responsibility Mapping

- `/pages`: Renders dashboard workspaces tailored to the logged-in user role (`PatientDashboard.tsx`, `DoctorDashboard.tsx`).
- `/components`: Houses widgets like `GreetingBanner.tsx`, `ActionCards.tsx`, and `VitalsCounters.tsx`.
- `/services`: Pulls base metrics queries and profile details.
- `/hooks`: Declares React Query custom state wrappers (e.g. `useDashboardStats.ts`).
- `/types`: Contains types for metric aggregations and dashboard settings.
- `/constants`: Establishes dashboard card configurations, route permissions, and static widgets definitions.
