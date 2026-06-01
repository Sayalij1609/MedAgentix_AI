# Feature Module — Doctor Portal (doctor)

This feature module provides clinical triaging tools, unassigned queue claimant views, and case peer-review panels for authenticated physicians.

## Directory Responsibility Mapping

- `/pages`: Renders queues grids and case review pages (`PatientQueue.tsx`, `PatientReview.tsx`).
- `/components`: Houses UI components like `QueueDataGrid.tsx`, `TreatmentPrescriber.tsx`, and `FeedbackForm.tsx`.
- `/services`: Interfaces with backend routes `/api/v1/feedback`, `/api/v1/consultations/:id/assign`, and `/api/v1/doctors/availability`.
- `/hooks`: Declares react query custom hooks (e.g. `useTriageQueue.ts`, `usePeerReview.ts`).
- `/types`: Maps queue item models, treatment prescription payloads, and doctor credentials.
- `/constants`: Declares triage priority colors, specialized departments, and schedule blocks configurations.
