import React, { lazy, Suspense } from 'react';
import { Route, Routes } from 'react-router-dom';
import { ProtectedRoute } from './ProtectedRoute';
import { RoleRoute } from './RoleRoute';
import { ROUTES } from './config';
import { useAuth } from '../context/auth-context';

// Layout Wrappers
import { MainLayout } from '../layouts/MainLayout';
import { AuthLayout } from '../layouts/AuthLayout';
import { PatientDashboardLayout } from '../layouts/PatientDashboardLayout';
import { DoctorDashboardLayout } from '../layouts/DoctorDashboardLayout';

// Public pages
const LandingPage = lazy(() => import('../pages/common/landing-page'));
const LoginPage = lazy(() => import('../pages/auth/login-page'));
const RegisterPage = lazy(() => import('../pages/auth/register-page'));

// Patient pages
const PatientDashboard = lazy(() => import('../pages/patient/dashboard'));
const PatientIntake = lazy(() => import('../pages/patient/intake'));
const PatientInsights = lazy(() => import('../pages/patient/insights'));

// Doctor pages
const DoctorDashboard = lazy(() => import('../pages/doctor/dashboard'));
const DoctorQueue = lazy(() => import('../pages/doctor/queue'));
const DoctorTriage = lazy(() => import('../pages/doctor/triage'));
const DoctorAssessment = lazy(() => import('../pages/doctor/assessment'));
const DoctorPatientIntake = lazy(() => import('../pages/doctor/patient-intake'));

// Shared pages (reports, analysis) — accessible to both roles
const ConsultationDetail = lazy(() => import('../pages/common/consultation-detail'));
const ClinicalReport = lazy(() => import('../pages/common/clinical-report'));
const ReportAnalysis = lazy(() => import('../pages/common/report-analysis'));

/**
 * RoleAwareSharedLayout
 * Renders PatientDashboardLayout or DoctorDashboardLayout depending on the
 * logged-in user's role. Used for shared pages (ClinicalReport, ReportAnalysis)
 * so the correct sidebar always appears regardless of which role is viewing.
 */
const RoleAwareSharedLayout: React.FC = () => {
  const { user } = useAuth();
  if (user?.role === 'doctor') {
    return <DoctorDashboardLayout />;
  }
  return <PatientDashboardLayout />;
};

// Loading fallback
const SuspenseLoader = () => (
  <div className="min-h-[50vh] flex items-center justify-center bg-transparent">
    <div className="flex flex-col items-center gap-3">
      <div className="w-8 h-8 border-3 border-primary border-t-transparent rounded-full animate-spin" />
      <p className="text-muted-foreground text-xs animate-pulse">Loading...</p>
    </div>
  </div>
);

export const AppRoutes: React.FC = () => {
  return (
    <Suspense fallback={<SuspenseLoader />}>
      <Routes>
        <Route element={<MainLayout />}>

          {/* ── 1. PUBLIC ── */}
          <Route path={ROUTES.LANDING} element={<LandingPage />} />
          <Route element={<AuthLayout />}>
            <Route path={ROUTES.LOGIN} element={<LoginPage />} />
            <Route path={ROUTES.REGISTER} element={<RegisterPage />} />
          </Route>

          {/* ── 2. PATIENT-ONLY ROUTES (PatientDashboardLayout) ── */}
          <Route element={
            <ProtectedRoute>
              <RoleRoute allowedRoles={['patient']}>
                <PatientDashboardLayout />
              </RoleRoute>
            </ProtectedRoute>
          }>
            <Route path={ROUTES.PATIENT_DASHBOARD} element={<PatientDashboard />} />
            <Route path={ROUTES.PATIENT_INTAKE} element={<PatientIntake />} />
            <Route path={ROUTES.PATIENT_INSIGHTS} element={<PatientInsights />} />
          </Route>

          {/* ── 3. DOCTOR-ONLY ROUTES (DoctorDashboardLayout) ── */}
          <Route element={
            <ProtectedRoute>
              <RoleRoute allowedRoles={['doctor']}>
                <DoctorDashboardLayout />
              </RoleRoute>
            </ProtectedRoute>
          }>
            <Route path={ROUTES.DOCTOR_DASHBOARD} element={<DoctorDashboard />} />
            <Route path={ROUTES.DOCTOR_QUEUE} element={<DoctorQueue />} />
            <Route path={ROUTES.DOCTOR_TRIAGE} element={<DoctorTriage />} />
            <Route path={ROUTES.DOCTOR_ASSESSMENT} element={<DoctorAssessment />} />
            <Route path={ROUTES.DOCTOR_PATIENT_INTAKE} element={<DoctorPatientIntake />} />
          </Route>

          {/* ── 4. SHARED ROUTES — both patient & doctor ──────────────
               RoleAwareSharedLayout automatically picks the right sidebar.
               IMPORTANT: /reports/analyze (static) must be declared BEFORE
               /reports/:id (param) so React Router matches it correctly.
          ─────────────────────────────────────────────────────────────── */}
          <Route element={
            <ProtectedRoute>
              <RoleRoute allowedRoles={['patient', 'doctor']}>
                <RoleAwareSharedLayout />
              </RoleRoute>
            </ProtectedRoute>
          }>
            {/* Static route first — MUST be before the dynamic :id param */}
            <Route path={ROUTES.REPORT_ANALYSIS} element={<ReportAnalysis />} />
            {/* Dynamic param routes */}
            <Route path={ROUTES.CLINICAL_REPORT} element={<ClinicalReport />} />
            <Route path={ROUTES.CONSULTATION_DETAIL} element={<ConsultationDetail />} />
          </Route>

          {/* ── 404 ── */}
          <Route path="*" element={
            <div className="flex flex-col items-center justify-center min-h-[60vh]">
              <h2 className="text-xl font-bold text-slate-800">404 — Page Not Found</h2>
              <a href="/" className="mt-3 text-sm text-primary hover:underline">Return to Home</a>
            </div>
          } />

        </Route>
      </Routes>
    </Suspense>
  );
};
export default AppRoutes;
