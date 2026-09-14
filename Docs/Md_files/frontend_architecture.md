# MedAgentix AI — Frontend Architecture Blueprint
**Document Version**: 1.0.0  
**Architect**: Frontend Architecture Expert  
**Tech Stack**: React 18+ (TypeScript), Vite, Tailwind CSS, Framer Motion, shadcn/ui, TanStack React Query v5, Axios

This document outlines the scalable, modular frontend directory structure designed specifically to integrate with the **MedAgentix AI** multi-agent backend. The architecture follows a **feature-based folder pattern**, isolating distinct clinical domains (Auth, Patients, Doctors, Reports, Workflow) while maintaining centralized shared utilities.

---

## 1. Modular Directory Tree

The complete frontend application is structured inside a new top-level `frontend/` directory (separate from existing backend code).

```
frontend/
│
├── .vscode/                     # Editor settings & shared extensions
├── public/                      # Static assets (favicons, manifest.json)
└── src/
    ├── assets/                  # Global assets (images, vectors, fonts)
    │   ├── icons/
    │   └── illustrations/
    │
    ├── components/              # Global shared UI components (shadcn & custom)
    │   ├── ui/                  # Raw shadcn/ui primitives (Radix-backed)
    │   │   ├── button.tsx
    │   │   ├── dialog.tsx
    │   │   ├── select.tsx
    │   │   └── toast.tsx
    │   └── common/              # Global customized widgets
    │       ├── loading-spinner.tsx
    │       ├── error-boundary.tsx
    │       └── page-transition.tsx
    │
    ├── constants/               # Global configuration tables & static lists
    │   ├── api-endpoints.ts
    │   ├── clinical-thresholds.ts
    │   └── symptoms-registry.ts
    │
    ├── context/                 # Global state providers
    │   ├── auth-context.tsx
    │   └── theme-context.tsx
    │
    ├── features/                # Clinical business domain modules (Self-contained)
    │   ├── auth/                # AUTHENTICATION MODULE
    │   │   ├── components/      # Login/Register widgets
    │   │   ├── hooks/           # useLogin, useRegister queries
    │   │   ├── services/        # auth-api.ts
    │   │   └── types.ts
    │   │
    │   ├── patient/             # PATIENT MODULE
    │   │   ├── components/      # BioForm, Timeline, AssessmentGrid
    │   │   ├── hooks/           # usePatientProfile, useVitals
    │   │   ├── services/        # patient-api.ts
    │   │   └── types.ts
    │   │
    │   ├── doctor/              # DOCTOR MODULE
    │   │   ├── components/      # QueueTable, AvailabilityToggle
    │   │   ├── hooks/           # useTriageQueue, useAvailability
    │   │   ├── services/        # doctor-api.ts
    │   │   └── types.ts
    │   │
    │   ├── reports/             # REPORTS MODULE
    │   │   ├── components/      # PrintableReport, HistoryFilter
    │   │   ├── hooks/           # useDownloadPDF, useReportHistory
    │   │   ├── services/        # reports-api.ts
    │   │   └── types.ts
    │   │
    │   └── workflow/            # WORKFLOW & DIAGNOSTIC INFERENCE MODULE
    │       ├── components/      # AgentMonitor, XAICanvas, ChatBot
    │       ├── hooks/           # useAgentDiagnostics, useSHAPAttributions
    │       ├── services/        # workflow-api.ts
    │       └── types.ts
    │
    ├── hooks/                   # Global cross-cutting React hooks
    │   ├── use-debounce.ts
    │   ├── use-local-storage.ts
    │   └── use-media-query.ts
    │
    ├── layouts/                 # Structural templates defining viewport wrapping
    │   ├── auth-layout.tsx      # Split screen layout with medical graphic
    │   ├── dashboard-layout.tsx # Collapsible sidebar, header with notifications
    │   └── root-layout.tsx      # Entry provider wrapping, scroll bars
    │
    ├── pages/                   # Route components mapping to views
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
    │   ├── common/
    │   │   ├── landing-page.tsx
    │   │   ├── report-viewer.tsx
    │   │   └── not-found.tsx
    │   └── index.ts
    │
    ├── routes/                  # Client-side router declarations
    │   ├── app-routes.tsx       # Standard route registry
    │   ├── protected-route.tsx  # RBAC gating component
    │   └── index.ts
    │
    ├── services/                # Global API clients and interceptor scripts
    │   ├── api-client.ts        # Axios base instances with JWT headers
    │   └── query-client.ts      # Global React Query cache configuration
    │
    ├── types/                   # Shared TypeScript models
    │   ├── clinical.d.ts
    │   └── user.d.ts
    │
    ├── App.tsx                  # Root application loader
    ├── main.tsx                 # Entry mount point
    ├── index.css                # Global CSS directives & Tailwind variables
    ├── vite-env.d.ts            # Vite TypeScript declarations
    │
    ├── package.json             # Build configurations & module packages
    ├── tailwind.config.js       # Theme definitions, gradients, animations
    ├── tsconfig.json            # Strict compiler configurations
    └── vite.config.ts           # Development proxy & bundler settings
```

---

## 2. Directory Purpose and Architectural Mandate

### A. `/src/assets`
* **Purpose**: Houses static digital resources compiled directly into the application bundle (optimized via Vite's asset compiler).
* **Mandate**: Avoid deep nesting. Subdivide strictly into structural folders like `/icons` for SVG symbols and `/illustrations` for graphic overlays.

### B. `/src/components`
* **Purpose**: Holds presentational UI components that do not contain business logic and can be reused across different modules.
* **Mandate**: Divided into two sections:
  1. `/ui`: Contains raw, highly customizable base primitives (e.g., standard shadcn wrappers for buttons, menus, inputs) that should not hold clinical state.
  2. `/common`: Customized global widgets used throughout the application, such as loading indicators, global error boundaries, and transition panels for animations.

### C. `/src/constants`
* **Purpose**: Serves as a single source of truth for static values, configurations, and lookup maps.
* **Mandate**: All clinical definitions (e.g. system symptom options, temperature scaling, or API endpoints) should be defined here. Do not write raw string constants inside pages or components.

### D. `/src/context`
* **Purpose**: Declares global React Context state wrappers for variables that need to be accessed from any page.
* **Mandate**: Keep contexts minimal. Use exclusively for global parameters like authentication states (`AuthContext`) and theme preferences (`ThemeContext`). Use React Query for clinical data caching instead of custom contexts.

### E. `/src/features`
* **Purpose**: The core of the feature-based folder architecture. Each subdirectory represents a self-contained domain module of the application.
* **Mandate**: Each feature folder (e.g. `patient/`, `workflow/`) encapsulates its own domain-specific components, hooks, API services, and types. Features are prohibited from importing directly from other features; any shared features must be extracted to global components or services.

### F. `/src/hooks`
* **Purpose**: Houses custom React hooks that perform global, cross-cutting tasks.
* **Mandate**: Use exclusively for utility behaviors that do not depend on specific clinical state (e.g., browser resizing hooks or local storage synchronizers). Domain-specific hooks must reside in their respective `/features/**/hooks/` folders.

### G. `/src/layouts`
* **Purpose**: Serves as page templates that define the layout structures (like sidebars and navbars) wrapping page content.
* **Mandate**: Isolates the main page layouts, such as split screen formats for login paths and structured dashboards with sidebar navigation for patient/doctor portals.

### H. `/src/pages`
* **Purpose**: Represents the route components matching the application path mappings.
* **Mandate**: Page components should remain thin wrappers that import logic and layouts from `/features` and `/layouts`. They should not define custom APIs, styles, or complex forms directly.

### I. `/src/routes`
* **Purpose**: Manages routing definitions and page access logic.
* **Mandate**: Houses client-side routes (using React Router v6) and security wrapper guards (like `ProtectedRoute`) to enforce role-based access before page rendering.

### J. `/src/services`
* **Purpose**: Manages global data fetching libraries and cache clients.
* **Mandate**: Configure base clients (e.g. Axios configurations with automatic JWT header attachment) and register global React Query clients to manage cache lifetimes and retry settings.

### K. `/src/types`
* **Purpose**: Holds centralized, reusable TypeScript models.
* **Mandate**: Define structural interfaces shared across multiple features (e.g. baseline demographic models or medical record interfaces).

---

## 3. Scale-Up & Performance Design (Tailwind, Motion, Query)

### I. Tailwind CSS and Theme System
* **Implementation**: Managed in `/src/index.css` and `tailwind.config.js`.
* **Strategy**: Establishes custom theme palettes utilizing Tailwind variables, allowing smooth dynamic transitions between light and dark clinical settings. Color setups avoid harsh primary colors in favor of balanced HSL scales (e.g. clinical blues, calming teals, and alert reds).

### II. Framer Motion and Animation Patterns
* **Implementation**: Controlled inside `/src/components/common/page-transition.tsx` and custom sliders.
* **Strategy**: Implements subtle, professional micro-animations. Screen switches utilize clean fade-ins, modal frames utilize slight scale transitions, and queue components utilize sliding list items to make updates feel smooth and responsive.

### III. React Query and Data Caching
* **Implementation**: Set up in `/src/services/query-client.ts` and consumed in `/features/**/hooks/`.
* **Strategy**: Isolates UI state from server cache state. The hook `useTriageQueue` queries `/api/v1/consultations?status=Pending` using a 10-second automatic polling interval, ensuring that the Doctor's triage dashboard updates dynamically as new emergencies arrive, without requiring page refreshes.
