import React from 'react';
import { Outlet, Navigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth } from '../context/auth-context';
import { ROUTES } from '../routes/config';
import { Shield, Activity, Lock, CheckCircle } from 'lucide-react';

const TRUST = [
  { icon: Shield, label: 'HIPAA Compliant' },
  { icon: Lock, label: 'Encrypted' },
  { icon: Activity, label: 'Real-Time CDSS' },
  { icon: CheckCircle, label: 'ESI Protocol' },
];

export const AuthLayout: React.FC = () => {
  const { isAuthenticated, user } = useAuth();

  if (isAuthenticated && user) {
    const redirectUrl = user.role === 'doctor' ? ROUTES.DOCTOR_DASHBOARD : ROUTES.PATIENT_DASHBOARD;
    return <Navigate to={redirectUrl} replace />;
  }

  return (
    <div className="min-h-screen flex overflow-hidden">

      {/* ═══════════════════════════════════════════════
          LEFT PANEL — Dark, grid, doctor illustration
      ═══════════════════════════════════════════════ */}
      <div className="hidden lg:flex flex-col" style={{ width: '46%', minWidth: '440px' }}>
        <div className="relative flex flex-col h-full bg-gradient-to-br from-slate-900 via-[#0d2e4a] to-[#0a3d62] overflow-hidden">

          {/* Fine grid */}
          <div className="absolute inset-0 pointer-events-none"
            style={{
              backgroundImage: `
                linear-gradient(to right, rgba(56,189,248,0.09) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(56,189,248,0.09) 1px, transparent 1px)
              `,
              backgroundSize: '32px 32px',
            }}
          />

          {/* Glow blobs */}
          <div className="absolute -top-40 -left-40 w-[500px] h-[500px] bg-teal-500/10 rounded-full blur-3xl pointer-events-none" />
          <div className="absolute bottom-0 right-0 w-[400px] h-[350px] bg-sky-500/8 rounded-full blur-3xl pointer-events-none" />
          <motion.div
            className="absolute top-1/3 right-10 w-40 h-40 bg-indigo-500/10 rounded-full blur-2xl pointer-events-none"
            animate={{ scale: [1, 1.2, 1], opacity: [0.4, 0.9, 0.4] }}
            transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
          />

          {/* Content */}
          <div className="relative z-10 flex flex-col h-full px-10 py-10">

            {/* Logo */}
            <motion.div initial={{ opacity: 0, y: -14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.55 }}
              className="flex items-center gap-3">
              <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-teal-500/30 to-sky-500/20 border border-teal-400/40 flex items-center justify-center">
                <span className="text-teal-300 font-black text-xl">M</span>
              </div>
              <div>
                <p className="text-white font-extrabold text-lg leading-none">MedAgentix AI</p>
                <p className="text-teal-400 text-[11px] font-medium mt-0.5 tracking-wide">Clinical Portal · v2.0</p>
              </div>
            </motion.div>

            {/* Headline */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.65 }} className="mt-10 space-y-2">
              <p className="text-teal-400 text-[11px] font-bold uppercase tracking-[0.2em]">Multi-Agent CDSS v2.0</p>
              <h2 className="text-4xl font-extrabold text-white leading-tight">
                AI-Powered<br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-300 via-sky-300 to-indigo-400">
                  Clinical Diagnosis
                </span>
              </h2>
              <p className="text-slate-400 text-sm leading-relaxed max-w-[280px]">
                Evidence-based triage, differential diagnostics, and real-time CDSS reports — all in one secure platform.
              </p>
            </motion.div>

            {/* Doctor illustration — fills remaining vertical space */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3, duration: 0.9, ease: 'easeOut' }}
              className="flex-1 flex items-end justify-center relative mt-4 mb-2"
            >
              {/* Glow beneath illustration */}
              <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-80 h-32 bg-teal-400/12 rounded-full blur-3xl" />

              <img
                src="/doctor-illustration.jpg"
                alt="MedAgentix Doctor AI"
                className="relative z-10 w-full object-contain select-none drop-shadow-2xl"
                style={{ maxHeight: '340px', maxWidth: '320px' }}
                draggable={false}
              />

              {/* Floating: accuracy */}
              <motion.div
                initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.75, duration: 0.5 }}
                className="absolute left-2 top-10 bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-3 text-white shadow-xl"
              >
                <p className="text-[9px] font-bold text-teal-300 uppercase tracking-wide">🧠 Accuracy</p>
                <p className="text-2xl font-black mt-0.5">70%+</p>
                <p className="text-[9px] text-slate-300 mt-0.5">721 diseases</p>
              </motion.div>

              {/* Floating: pipeline */}
              <motion.div
                initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.9, duration: 0.5 }}
                className="absolute right-2 top-8 bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-3 text-white shadow-xl"
              >
                <p className="text-[9px] font-bold text-sky-300 uppercase tracking-wide">⚡ Pipeline</p>
                <p className="text-2xl font-black mt-0.5">10</p>
                <p className="text-[9px] text-slate-300 mt-0.5">AI stages</p>
              </motion.div>
            </motion.div>

            {/* Trust badges */}
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.0, duration: 0.6 }}
              className="flex flex-wrap gap-2 mt-2">
              {TRUST.map((b, i) => (
                <div key={i} className="flex items-center gap-1.5 bg-white/5 border border-white/10 rounded-full px-3 py-1.5">
                  <b.icon className="w-3 h-3 text-teal-400" />
                  <span className="text-[10px] text-slate-300 font-medium">{b.label}</span>
                </div>
              ))}
            </motion.div>

            <p className="mt-4 text-slate-600 text-[10px]">
              © {new Date().getFullYear()} MedAgentix AI — Authorized Clinical Use Only
            </p>
          </div>
        </div>
      </div>

      {/* ═══════════════════════════════════════════════
          RIGHT PANEL — Form
      ═══════════════════════════════════════════════ */}
      <div className="flex-1 flex flex-col justify-center items-center relative bg-slate-50 overflow-y-auto py-10 px-6 sm:px-10">

        {/* Subtle grid */}
        <div className="absolute inset-0 pointer-events-none opacity-[0.025]"
          style={{ backgroundImage: `linear-gradient(to right, #0ea5e9 1px, transparent 1px), linear-gradient(to bottom, #0ea5e9 1px, transparent 1px)`, backgroundSize: '40px 40px' }} />

        {/* Top bar — Home link */}
        <div className="absolute top-0 left-0 right-0 px-8 py-4 flex items-center justify-between">
          {/* Mobile logo */}
          <a href="/" className="lg:hidden flex items-center gap-2">
            <div className="w-8 h-8 rounded-xl bg-teal-50 border border-teal-200 flex items-center justify-center font-black text-base text-teal-600">M</div>
            <span className="font-bold text-base text-foreground">MedAgentix AI</span>
          </a>
          <div className="hidden lg:block" />
          {/* Home link — visible on all sizes */}
          <a
            href="/"
            className="flex items-center gap-1.5 text-xs font-semibold text-slate-500 hover:text-teal-600 transition-colors group"
          >
            <svg className="w-3.5 h-3.5 group-hover:-translate-x-0.5 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back to Home
          </a>
        </div>

        {/* Form card */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
          className="relative z-10 w-full max-w-[480px]">
          <div className="bg-white border border-slate-200 rounded-2xl shadow-xl shadow-slate-200/60 px-10 py-10 space-y-5">
            <Outlet />
          </div>
        </motion.div>

        <p className="mt-5 text-xs text-slate-400 text-center">
          Secure clinical platform — HIPAA-compliant &amp; end-to-end encrypted.
        </p>
      </div>
    </div>
  );
};
export default AuthLayout;
