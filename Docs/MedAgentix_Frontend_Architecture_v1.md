# MedAgentix AI — Enterprise Frontend Architecture Spec
**Document Version**: 1.0.0  
**Architect**: Frontend Architecture Expert  
**Target Tech Stack**: React 18+ (TypeScript), Vite, Tailwind CSS, Framer Motion, shadcn/ui (Radix-based), TanStack React Query v5, Axios

This document outlines the scalable, decoupled frontend architecture for **MedAgentix AI**. The architecture follows a **Feature-Driven Design Pattern**, ensuring clinical modules remain isolated and easy to scale while maintaining clean integration with the backend APIs.

---

## 1. Global Directory Tree Schema

The frontend code resides in a top-level `/frontend` directory completely isolated from existing backend directories.

```
frontend/
├── .vscode/                     # Shared IDE parameters & extensions
├── public/                      # Static metadata (manifests, favicons)
└── src/
    ├── assets/                  # GLOBAL DIGITAL ASSETS
    │   ├── icons/               # Standard SVG components
    │   └── illustrations/       # Split-screen banners & onboarding visual maps
    │
    ├── components/              # GLOBAL PRESENTATIONAL COMPONENTS
    │   ├── ui/                  # Raw shadcn/ui primitives (Radix wrappers)
    │   │   ├── button.tsx
    │   │   ├── dialog.tsx
    │   │   ├── select.tsx
    │   │   └── toast.tsx
    │   └── common/              # Customized global layout blocks
    │       ├── loading-spinner.tsx
    │       ├── error-boundary.tsx
    │       └── page-transition.tsx
    │
    ├── constants/               # CONFIGURATION CONSTANTS
    │   ├── api-endpoints.ts     # Maps matching API Blueprint routes
    │   ├── clinical-thresholds.ts # High-risk alarm limits (e.g. Oxygen Saturation)
    │   └── symptoms-registry.ts # Lookup dictionaries for symptoms slug mapping
    │
    ├── context/                 # SESSION CONTEXT WRAPPERS
    │   ├── auth-context.tsx     # Session states, roles, and profiles
    │   └── theme-context.tsx    # Handles Dark Mode / Light Mode
    │
    ├── features/                # BUSINESS DOMAIN FEATURES (Isolated Modules)
    │   ├── auth/                # AUTHENTICATION MODULE
    │   │   ├── components/      # LoginForm.tsx, RegistrationForm.tsx
    │   │   ├── hooks/           # useLoginQuery.ts, useRegisterMutation.ts
    │   │   ├── services/        # auth-api-service.ts
    │   │   └── types.ts
    │   │
    │   ├── patient/             # PATIENT MODULE
    │   │   ├── components/      # DemographicForm.tsx, VitalsVibe.tsx
    │   │   ├── hooks/           # usePatientProfile.ts, useVitalsTracker.ts
    │   │   ├── services/        # patient-api-service.ts
    │   │   └── types.ts
    │   │
    │   ├── doctor/              # DOCTOR PORTAL MODULE
    │   │   ├── components/      # TriageQueueGrid.tsx, ClaimButton.tsx
    │   │   ├── hooks/           # useQueueQuery.ts, useAvailabilityState.ts
    │   │   ├── services/        # doctor-api-service.ts
    │   │   └── types.ts
    │   │
    │   ├── reports/             # CLINICAL REPORTS MODULE
    │   │   ├── components/      # PrintableReportView.tsx, ReportsGridTable.tsx
    │   │   ├── hooks/           # usePDFExporter.ts, useHistoryLogs.ts
    │   │   ├── services/        # reports-api-service.ts
    │   │   └── types.ts
    │   │
    │   └── workflow/            # WORKFLOW & DIAGNOSTIC INFERENCE MODULE
    │       ├── components/      # AgentInferenceMonitor.tsx, XAIattributionGraph.tsx
    │       ├── hooks/           # useAgentWorkflow.ts, useSHAPMetrics.ts
    │       ├── services/        # workflow-api-service.ts
    │       └── types.ts
    │
    ├── hooks/                   # GLOBAL CROSS-CUTTING HOOKS
    │   ├── use-debounce.ts      # Optimizes symptom registry searching
    │   ├── use-local-storage.ts # Syncs tokens to local disk
    │   └── use-media-query.ts   # Detects responsive viewport changes
    │
    ├── layouts/                 # STRUCTURAL TEMPLATE VIEWPORT WRAPPERS
    │   ├── auth-layout.tsx      # Dual-screen banner split layout
    │   ├── dashboard-layout.tsx # Collapsible sidebar, profile dropdowns
    │   └── root-layout.tsx      # Provider attachments (React Query, Toast UI)
    │
    ├── pages/                   # SLIM ROUTE COMPONENT WRAPPERS
    │   ├── auth/
    │   │   ├── login-page.tsx
    │   │   └── register-page.tsx
    │   ├── patient/
    │   │   ├── dashboard.tsx
    │   │   ├── intake.tsx
    │   │   └── insights.tsx
    │   ├── doctor/
    │   │   ├── dashboard.tsx
    │   │   ├── queue.tsx
    │   │   └── triage.tsx
    │   └── common/
    │       ├── landing-page.tsx
    │       ├── report-viewer.tsx
    │       └── not-found.tsx
    │
    ├── routes/                  # CLIENT ROUTING ENGINE
    │   ├── app-routes.tsx       # Standard React Router v6 registry map
    │   ├── protected-route.tsx  # RBAC interceptor gate component
    │   └── index.ts
    │
    ├── services/                # CORE API CLIENTS
    │   ├── api-client.ts        # Axios base instances with JWT headers
    │   └── query-client.ts      # TanStack Query instance cache definitions
    │
    ├── types/                   # GLOBAL COMPILATION TYPINGS
    │   ├── clinical.d.ts
    │   └── user.d.ts
    │
    ├── App.tsx                  # Root loader
    ├── main.tsx                 # Entry mount point
    ├── index.css                # Global CSS directives & Tailwind variables
    └── vite.config.ts           # Development proxy & bundler settings
```

---

## 2. Comprehensive Folder Responsibilities

### `/src/assets` (Global Digital Assets)
* **Responsibility**: Holds global static image files, vector icons, and customized fonts.
* **Architecture Guideline**: All SVGs used as active widgets must be defined inside `/icons` as React functional components to enable dynamic styling via Tailwind CSS classes.

### `/src/components` (Global Presentational Components)
* **Responsibility**: Houses clean, stateless, logic-free user interface widgets that are reused across different modules.
* **Architecture Guideline**:
  - `/ui`: Contains standard shadcn/ui components (such as standard buttons, dialogs, selects). These should be styled purely using Tailwind CSS class declarations and rely on state passed down through props.
  - `/common`: Contains structural global helpers (such as loading spinners, error boundaries, page transition wraps) that do not belong to a specific feature module.

### `/src/constants` (Configuration Constants)
* **Responsibility**: Centralizes application configurations, lookup tables, and hardcoded values.
* **Architecture Guideline**: All endpoints, triage limits, and symptom registry definitions must be kept here. Hardcoded strings inside components are strictly prohibited to ensure ease of maintenance.

### `/src/context` (Session Context Providers)
* **Responsibility**: Manages low-frequency global state variables (e.g., active user sessions, login state, dashboard theme settings).
* **Architecture Guideline**: Keep contexts lightweight. Domain-specific clinical data must be handled via React Query caching rather than custom React contexts.

### `/src/features` (Business Domain Features)
* **Responsibility**: The core of the feature-based folder architecture. Isolates the business logic, custom hooks, and presentational elements of individual clinical modules (e.g. `auth`, `patient`, `workflow`).
* **Architecture Guideline**: Feature folders are completely self-contained. A feature is prohibited from importing directly from other features; any shared code must be extracted to global folders (`/components`, `/hooks`, `/services`).

### `/src/hooks` (Global Utility Hooks)
* **Responsibility**: Houses custom React hooks that handle cross-cutting concerns.
* **Architecture Guideline**: Reserved for utility concerns that are not tied to a specific business feature (e.g., debouncers, media queries, local storage sync). Domain-specific hooks must reside inside their respective `/features/**/hooks/` directories.

### `/src/layouts` (Structural Page Templates)
* **Responsibility**: Defines the structural scaffolding (such as sidebars, headers, and navigation bars) that wraps page components.
* **Architecture Guideline**: Isolates the main page layouts, such as split screen formats for login paths and structured dashboards with sidebar navigation for patient/doctor portals.

### `/src/pages` (Slim Route Components)
* **Responsibility**: Serves as the entry-point views for client-side routing.
* **Architecture Guideline**: Keep pages extremely lightweight. Pages must only serve to tie together layouts from `/layouts` and domain feature components from `/features`, without implementing custom business logic.

### `/src/routes` (Client Routing Engine)
* **Responsibility**: Defines the application routing tables and manages role-based access.
* **Architecture Guideline**: Uses React Router v6. Integrates `ProtectedRoute` as a wrapper guard to enforce role-based access control (RBAC) before page components are rendered.

### `/src/services` (Core API Clients)
* **Responsibility**: Manages data fetching configurations, global caches, and API clients.
* **Architecture Guideline**:
  - `api-client.ts`: Configures Axios with automatic JWT header injection and request/response interceptors to handle expired token situations.
  - `query-client.ts`: Sets up TanStack React Query clients to manage global data lifetimes and cache invalidation policies.

### `/src/types` (Global Compilation Typings)
* **Responsibility**: Holds shared, high-level TypeScript interface definitions.
* **Architecture Guideline**: All shared models (e.g. basic demographic shapes or consultation logs) must be registered here. Feature-specific models should remain inside their respective `/features/**/types.ts` folders.

---

## 3. Scale-Up & Performance Guidelines

### A. Performance Optimization with Axios + React Query
* All data operations must go through custom hooks wrapping React Query mutations or queries (e.g., `useQuery`, `useMutation`).
* Automatic stale-time intervals are configured globally in `query-client.ts` to reduce redundant API queries. The Doctor's triage dashboard is configured with a 10-second polling interval to fetch critical alerts dynamically without full page reloads.

### B. Frame Motion & Transition Standards
* Page transitions are managed globally through `page-transition.tsx` wrapping the router viewport.
* Animations should be kept subtle (e.g., standard `0.2s` fade-ins, minor scale expansions for modals) to maintain a highly professional, clinical feel.
