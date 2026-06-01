# Feature Module — Symptom Analysis (symptom-analysis)

This feature module is the backbone of the dynamic intake pipeline. It handles searchable symptom checklists, dynamic follow-up chatbot consoles, and vitals vectors uploads.

## Directory Responsibility Mapping

- `/pages`: Renders the intake sandbox and symptom checklists screen (`SymptomAssessment.tsx`).
- `/components`: Houses UI components like `SymptomAutocomplete.tsx`, `ChatIntakeConsole.tsx`, and `VitalsCollector.tsx`.
- `/services`: Interfaces with backend routes `/api/v1/symptoms/assess` and `/api/v1/symptoms/follow-ups`.
- `/hooks`: Declares custom hooks to handle chat state (e.g. `useChatDialog.ts`) and react query wrappers.
- `/types`: Maps symptom list shapes, expected follow-up questions, and vitals payload schemas.
- `/constants`: Declares autocomplete delay timeouts, temperature ranges, and blood pressure scales.
