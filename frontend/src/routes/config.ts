// Central Client Routing Constants mapping

export const ROUTES = {
  // Public Paths
  LANDING: '/',
  LOGIN: '/login',
  REGISTER: '/register',

  // Patient Sub-routes
  PATIENT_DASHBOARD: '/patient/dashboard',
  PATIENT_INTAKE: '/patient/intake',
  PATIENT_INSIGHTS: '/patient/insights',

  // Doctor Sub-routes
  DOCTOR_DASHBOARD: '/doctor/dashboard',
  DOCTOR_QUEUE: '/doctor/queue',
  DOCTOR_TRIAGE: '/doctor/triage',

  // Common Locked Views
  CONSULTATION_DETAIL: '/consultations/:id',
  CLINICAL_REPORT: '/reports/:id',
} as const;
