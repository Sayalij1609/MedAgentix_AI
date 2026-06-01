# Feature Module — Reports (reports)

This feature module compiles clinical reports charts, handles report histories search logs, and manages binary file compilations (PDF exports).

## Directory Responsibility Mapping

- `/pages`: Renders printable medical records layouts (`ClinicalReport.tsx`, `HistoryLogs.tsx`).
- `/components`: Houses UI components like `PrintableChart.tsx`, `HistoryGridTable.tsx`, and `DownloadButton.tsx`.
- `/services`: Interfaces with backend routes `/api/v1/reports/:id/export` and `/api/v1/ocr/upload`.
- `/hooks`: Declares custom export triggers (e.g. `usePDFExporter.ts`) and react query wrappers.
- `/types`: Maps medical chart data shapes, history filters, and download progress logs.
- `/constants`: Declares paper sizing limits, pagination defaults, and document templates parameters.
