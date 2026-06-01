# Feature Module — Authentication (auth)

This feature module manages user login, register, token session invalidation, and roles context mappings (Patient, Doctor, Admin).

## Directory Responsibility Mapping

- `/pages`: Renders client-side routes (e.g., `LoginPage.tsx`, `RegisterPage.tsx`).
- `/components`: Presents isolated auth widgets (e.g., `LoginForm.tsx`, `RegistrationForm.tsx`).
- `/services`: Houses API communication layers (`auth-api-service.ts`) using standard Axios client queries.
- `/hooks`: Declares React Query hooks wrapping API calls (e.g., `useLoginMutation.ts`, `useRegisterMutation.ts`).
- `/types`: Contains TypeScript typings mapping authentication payloads, tokens, and role credentials.
- `/constants`: Declares static variables (e.g., form field rules or authorization schema models).
