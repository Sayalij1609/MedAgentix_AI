# Feature Module — Agent Workflow & Diagnostics (workflow)

This feature module orchestrates diagnostic execution runs across specialized AI agents, processing loading loaders, and explainable AI (SHAP/LIME) attribution visualizations.

## Directory Responsibility Mapping

- `/pages`: Renders active monitoring dashboards (`WorkflowMonitor.tsx`).
- `/components`: Houses UI components like `AgentStatusLogger.tsx`, `ProcessingLoader.tsx`, and `XAIattributionChart.tsx`.
- `/services`: Interfaces with backend routes `/api/v1/workflow/diagnose` and `/api/v1/workflow/xai/:id`.
- `/hooks`: Declares react query custom hooks (e.g. `useAgentWorkflow.ts`, `useSHAPMetrics.ts`).
- `/types`: Maps agent process log states, SHAP attributions array structures, and confidence listings.
- `/constants`: Declares polling intervals, process stages, and XAI charting theme profiles.
