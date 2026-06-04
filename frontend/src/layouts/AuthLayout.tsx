import React from 'react';
import { Outlet, Navigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth } from '../context/auth-context';
import { ROUTES } from '../routes/config';

export const AuthLayout: React.FC = () => {
  const { isAuthenticated, user } = useAuth();

  if (isAuthenticated && user) {
    const redirectUrl =
      user.role === 'doctor' ? ROUTES.DOCTOR_DASHBOARD : ROUTES.PATIENT_DASHBOARD;
    return <Navigate to={redirectUrl} replace />;
  }

  return (
    <div className="min-h-screen grid grid-cols-1 lg:grid-cols-12 overflow-hidden bg-background">
      {/* Graphic Brand Column (5 cols) */}
      <div className="hidden lg:flex lg:col-span-5 flex-col justify-between p-12 bg-slate-50 border-r border-border/40 relative">
        <div className="absolute inset-0 opacity-10 bg-[linear-gradient(to_right,#808080_1px,transparent_1px),linear-gradient(to_bottom,#808080_1px,transparent_1px)] bg-[size:24px_24px]"></div>

        <div className="relative z-10 flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-teal-50 border border-teal-200 text-teal-600 flex items-center justify-center font-bold text-lg">M</div>
          <span className="font-bold text-xl tracking-wide text-slate-900">
            MedAgentix Clinical Portal
          </span>
        </div>

        <div className="relative z-10 space-y-6">
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-3xl font-extrabold text-slate-900 leading-tight"
          >
            Clinical Decision Support System
          </motion.h2>
          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.8 }}
            className="text-slate-600"
          >
            Providing evidence-based diagnostic suggestions, clinical risk assessments, and real-time case triage logs for professional patient care.
          </motion.p>
        </div>

        <div className="relative z-10 text-slate-500 text-xs">
          &copy; {new Date().getFullYear()} MedAgentix. Secure Clinical Portal Session.
        </div>
      </div>

      {/* Auth Entry Column (7 cols) */}
      <div className="lg:col-span-7 flex flex-col justify-center px-4 sm:px-12 md:px-20 lg:px-24 py-12 relative bg-slate-50">
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="w-full max-w-md mx-auto space-y-8 bg-card p-8 rounded-2xl border border-border shadow-xl"
        >
          <Outlet />
        </motion.div>
      </div>
    </div>
  );
};
export default AuthLayout;
