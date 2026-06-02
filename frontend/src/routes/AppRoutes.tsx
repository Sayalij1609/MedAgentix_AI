import React, { lazy, Suspense } from 'react';
import { Route, Routes } from 'react-router-dom';
import { ProtectedRoute } from './ProtectedRoute';
import { RoleRoute } from './RoleRoute';
import { ROUTES } from './config';

// Reusable Layout Wrappers
import { MainLayout } from '../layouts/MainLayout';
import { AuthLayout } from '../layouts/AuthLayout';
import { PatientDashboardLayout } from '../layouts/PatientDashboardLayout';
import { DoctorDashboardLayout } from '../layouts/DoctorDashboardLayout';

// Lazy Loaded Page Components
const LandingPage = lazy(() => import('../pages/common/landing-page'));
const LoginPage = lazy(() => import('../pages/auth/login-page'));
const RegisterPage = lazy(() => import('../pages/auth/register-page'));

const PatientDashboard = lazy(() => import('../pages/patient/dashboard'));
const PatientIntake = lazy(() => import('../pages/patient/intake'));
const PatientInsights = lazy(() => import('../pages/patient/insights'));

const DoctorDashboard = lazy(() => import('../pages/doctor/dashboard'));
const DoctorQueue = lazy(() => import('../pages/doctor/queue'));
const DoctorTriage = lazy(() => import('../pages/doctor/triage'));

const ConsultationDetail = lazy(() => import('../pages/common/consultation-detail'));
const ClinicalReport = lazy(() => import('../pages/common/clinical-report'));

// Loading fallbacks
const SuspenseLoader = () => (
  <div className="min-h-[50vh] flex items-center justify-center bg-transparent">
    <div className="flex flex-col items-center gap-3">
      <div className="w-8 h-8 border-3 border-primary border-t-transparent rounded-full animate-spin"></div>
      <p className="text-muted-foreground text-xs animate-pulse">Compiling view components...</p>
    </div>
  </div>
);

export const AppRoutes: React.FC = () => {
  return (
    <Suspense fallback={<SuspenseLoader />}>
      <Routes>
        <Route element={<MainLayout />}>
          {/* ==========================================
              1. PUBLIC CLIENT ROUTES
             ========================================== */}
          <Route path={ROUTES.LANDING} element={<LandingPage />} />
          
          <Route element={<AuthLayout />}>
            <Route path={ROUTES.LOGIN} element={<LoginPage />} />
            <Route path={ROUTES.REGISTER} element={<RegisterPage />} />
          </Route>

          {/* ==========================================
              2. PROTECTED ROLE ROUTES (PATIENT)
             ========================================== */}
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

          {/* ==========================================
              3. PROTECTED ROLE ROUTES (DOCTOR)
             ========================================== */}
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
          </Route>

          {/* ==========================================
              4. SHARED LOCKED DIAGNOSTIC VIEWS (Scoped)
             ========================================== */}
          <Route element={
            <ProtectedRoute>
              <RoleRoute allowedRoles={['patient', 'doctor']}>
                <DoctorDashboardLayout />
              </RoleRoute>
            </ProtectedRoute>
          }>
            <Route path={ROUTES.CONSULTATION_DETAIL} element={<ConsultationDetail />} />
            <Route path={ROUTES.CLINICAL_REPORT} element={<ClinicalReport />} />
          </Route>

          {/* Fallback wildcard router */}
          <Route path="*" element={
            <div className="flex flex-col items-center justify-center min-h-[60vh]">
              <h2 className="text-xl font-bold">404 - Clinical View Not Found</h2>
              <a href="/" className="mt-3 text-sm text-primary hover:underline">Return to Landing Portal</a>
            </div>
          } />
        </Route>
      </Routes>
    </Suspense>
  );
};
export default AppRoutes;
