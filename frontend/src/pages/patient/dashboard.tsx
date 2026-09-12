import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import apiClient from '../../services/api-client';
import { useAuth } from '../../context/auth-context';
import { useToast } from '../../context/toast-context';
import {
  FileText, Activity, Clock, Heart, Wind, RefreshCw,
  Calendar, ChevronRight, Stethoscope, ClipboardList,
  Plus, Sparkles, FileSearch, Upload, TrendingUp, Shield
} from 'lucide-react';

interface CaseSummary {
  id: number;
  status: string;
  created_at: string;
  chief_complaint: string;
  final_diagnosis: string;
  severity: string;
  triage_level?: number;
  vitals?: {
    heart_rate: number; oxygen_level: number;
    bp_reading?: string; temperature: number; cholesterol: number;
  };
}

const calculateAge = (dob: string): number | null => {
  if (!dob) return null;
  const birth = new Date(dob);
  const today = new Date();
  let age = today.getFullYear() - birth.getFullYear();
  if (today.getMonth() < birth.getMonth() ||
    (today.getMonth() === birth.getMonth() && today.getDate() < birth.getDate())) age--;
  return age >= 0 ? age : null;
};

const severityBadge = (s: string) => {
  switch (s?.toLowerCase()) {
    case 'mild': return 'bg-emerald-50 text-emerald-700 border-emerald-200';
    case 'moderate': return 'bg-amber-50 text-amber-700 border-amber-200';
    case 'severe': case 'serious': return 'bg-orange-50 text-orange-700 border-orange-200';
    case 'critical': case 'urgent': return 'bg-red-50 text-red-700 border-red-200';
    default: return 'bg-slate-50 text-slate-600 border-slate-200';
  }
};

const statusBadge = (s: string) => {
  switch (s?.toLowerCase()) {
    case 'completed': return 'bg-emerald-50 text-emerald-700 border-emerald-200';
    case 'reviewed': return 'bg-sky-50 text-sky-700 border-sky-200';
    case 'processing': return 'bg-amber-50 text-amber-700 border-amber-200 animate-pulse';
    default: return 'bg-slate-50 text-slate-600 border-slate-200';
  }
};

// Metric card component
const MetricCard: React.FC<{
  label: string; value: string | number | null; unit?: string;
  icon: React.ComponentType<{ className?: string }>; iconBg: string; iconColor: string; delay?: number;
}> = ({ label, value, unit, icon: Icon, iconBg, iconColor, delay = 0 }) => (
  <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
    transition={{ delay, duration: 0.4 }}
    className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm hover:shadow-md transition-all group cursor-default">
    <div className="flex items-center justify-between mb-3">
      <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">{label}</p>
      <div className={`w-9 h-9 rounded-xl ${iconBg} flex items-center justify-center`}>
        <Icon className={`w-4 h-4 ${iconColor}`} />
      </div>
    </div>
    {value != null ? (
      <div className="flex items-baseline gap-1.5">
        <span className="text-3xl font-black text-slate-900 leading-none">{value}</span>
        {unit && <span className="text-sm font-semibold text-slate-400">{unit}</span>}
      </div>
    ) : (
      <span className="text-sm text-slate-400 italic">No data</span>
    )}
  </motion.div>
);

export default function PatientDashboard() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { toast } = useToast();
  const [loading, setLoading] = useState(true);
  const [totalCases, setTotalCases] = useState(0);
  const [pendingReviews, setPendingReviews] = useState(0);
  const [recentCases, setRecentCases] = useState<CaseSummary[]>([]);
  const [latestAssessment, setLatestAssessment] = useState<CaseSummary | null>(null);
  const [lastUpdated, setLastUpdated] = useState('');

  const displayName = user?.name || user?.email?.split('@')[0] || 'there';
  const userAge = user?.date_of_birth ? calculateAge(user.date_of_birth) : null;

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const response = await apiClient.get('/patient/dashboard');
      if (response.data?.success) {
        setTotalCases(response.data.total_cases);
        setPendingReviews(response.data.pending_reviews);
        setRecentCases(response.data.recent_cases);
        setLatestAssessment(response.data.latest_assessment || null);
        setLastUpdated(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
      }
    } catch {
      toast('Failed to load your dashboard.', 'error');
    } finally { setLoading(false); }
  };

  useEffect(() => { fetchDashboardData(); }, []);

  const hrVal = latestAssessment?.vitals?.heart_rate ?? null;
  const spo2Val = latestAssessment?.vitals?.oxygen_level ?? null;
  const tempVal = latestAssessment?.vitals?.temperature ?? null;
  const bpVal = latestAssessment?.vitals?.bp_reading ?? null;
  const isNewPatient = totalCases === 0;

  if (loading && recentCases.length === 0) {
    return (
      <div className="space-y-6 max-w-6xl mx-auto animate-pulse">
        <div className="h-36 bg-slate-100 rounded-2xl" />
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => <div key={i} className="h-28 bg-slate-100 rounded-2xl" />)}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 h-72 bg-slate-100 rounded-2xl" />
          <div className="h-72 bg-slate-100 rounded-2xl" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-7 max-w-6xl mx-auto">

      {/* ── HERO WELCOME BANNER ── */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
        className="relative overflow-hidden rounded-2xl p-7 text-white"
        style={{ background: 'linear-gradient(135deg, #0f172a 0%, #0d2e4a 50%, #0a3d62 100%)' }}>
        {/* Grid */}
        <div className="absolute inset-0 pointer-events-none opacity-[0.08]"
          style={{ backgroundImage: `linear-gradient(to right, #38bdf8 1px, transparent 1px), linear-gradient(to bottom, #38bdf8 1px, transparent 1px)`, backgroundSize: '28px 28px' }} />
        {/* Glow blobs */}
        <div className="absolute -top-16 -right-16 w-56 h-56 bg-teal-400/15 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute -bottom-8 -left-8 w-40 h-40 bg-sky-500/10 rounded-full blur-2xl pointer-events-none" />

        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-5">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <motion.span className="w-2 h-2 bg-teal-400 rounded-full"
                animate={{ scale: [1, 1.4, 1], opacity: [1, 0.5, 1] }} transition={{ duration: 1.5, repeat: Infinity }} />
              <span className="text-teal-400 text-[10px] font-bold uppercase tracking-widest">Patient Portal</span>
            </div>
            <h1 className="text-2xl md:text-3xl font-extrabold leading-tight text-white">
              {isNewPatient ? `Welcome, ${displayName}! 👋` : `Good to see you, ${displayName}!`}
            </h1>
            <p className="text-slate-400 text-sm max-w-md leading-relaxed">
              {isNewPatient
                ? 'Your AI-powered health assistant is ready. Start your first assessment to get personalized diagnostics.'
                : `You have ${totalCases} assessment${totalCases !== 1 ? 's' : ''} on record.${pendingReviews > 0 ? ` ${pendingReviews} awaiting doctor review.` : ' All up to date!'}`}
            </p>
            {userAge && (
              <div className="flex items-center gap-3 mt-1">
                <span className="text-[10px] text-slate-400 bg-white/10 border border-white/10 rounded-full px-3 py-1">
                  Age: {userAge} yrs
                </span>
                {user?.gender && (
                  <span className="text-[10px] text-slate-400 bg-white/10 border border-white/10 rounded-full px-3 py-1">
                    {user.gender}
                  </span>
                )}
              </div>
            )}
          </div>

          <div className="flex flex-col sm:flex-row gap-3 shrink-0">
            <motion.button whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }}
              onClick={() => navigate('/patient/intake')}
              className="inline-flex items-center gap-2 px-5 py-3 bg-gradient-to-r from-teal-500 to-sky-600 text-white font-bold rounded-xl shadow-lg shadow-teal-500/25 text-sm">
              <Plus className="w-4 h-4" /> New Assessment
            </motion.button>
            <motion.button whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }}
              onClick={() => navigate('/reports/analyze')}
              className="inline-flex items-center gap-2 px-5 py-3 bg-white/10 border border-white/20 hover:bg-white/15 text-white font-semibold rounded-xl text-sm backdrop-blur-sm">
              <FileSearch className="w-4 h-4 text-teal-300" /> Upload Report
            </motion.button>
          </div>
        </div>
      </motion.div>

      {/* ── KPI CARDS ── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="Total Assessments" value={totalCases} icon={FileText}
          iconBg="bg-sky-50" iconColor="text-sky-600" delay={0.05} />
        <MetricCard label="Pending Reviews" value={pendingReviews} icon={Clock}
          iconBg="bg-amber-50" iconColor="text-amber-600" delay={0.1} />
        <MetricCard label="Heart Rate" value={hrVal} unit="bpm" icon={Heart}
          iconBg="bg-rose-50" iconColor="text-rose-500" delay={0.15} />
        <MetricCard label="SpO2" value={spo2Val} unit="%" icon={Wind}
          iconBg="bg-emerald-50" iconColor="text-emerald-600" delay={0.2} />
      </div>

      {/* ── QUICK ACTIONS (new patient) ── */}
      {isNewPatient && (
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            { icon: ClipboardList, bg: 'bg-sky-50', color: 'text-sky-600', title: '1. Describe Symptoms', desc: 'Tell us what you\'re feeling. Select symptoms or type freely in your own words.' },
            { icon: Activity, bg: 'bg-indigo-50', color: 'text-indigo-600', title: '2. AI Analyzes', desc: '10-stage AI pipeline runs differential analysis through specialist agents.' },
            { icon: FileText, bg: 'bg-teal-50', color: 'text-teal-600', title: '3. Get Your Report', desc: 'Receive a detailed clinical report with diagnoses, medications, and next steps.' },
          ].map((s, i) => (
            <div key={i} className="bg-white border border-slate-200 rounded-2xl p-6 space-y-3 shadow-sm">
              <div className={`w-11 h-11 rounded-xl ${s.bg} flex items-center justify-center`}>
                <s.icon className={`w-5 h-5 ${s.color}`} />
              </div>
              <h3 className="font-bold text-slate-800 text-sm">{s.title}</h3>
              <p className="text-xs text-slate-500 leading-relaxed">{s.desc}</p>
            </div>
          ))}
        </motion.div>
      )}

      {/* ── MAIN CONTENT GRID ── */}
      {!isNewPatient && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Left — Assessment History */}
          <div className="lg:col-span-2 space-y-5">

            {/* Latest Assessment */}
            {latestAssessment && (
              <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}
                className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 bg-indigo-50 rounded-xl flex items-center justify-center">
                      <Stethoscope className="w-4 h-4 text-indigo-600" />
                    </div>
                    <h3 className="font-bold text-slate-800 text-sm">Latest Assessment</h3>
                  </div>
                  <span className="text-[10px] text-slate-400 font-semibold">
                    {new Date(latestAssessment.created_at).toLocaleDateString(undefined, { month: 'long', day: 'numeric', year: 'numeric' })}
                  </span>
                </div>

                <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col sm:flex-row justify-between gap-4">
                  <div className="space-y-2">
                    <p className="font-bold text-slate-900">{latestAssessment.final_diagnosis || 'Awaiting Diagnosis'}</p>
                    <p className="text-xs text-slate-500">Chief complaint: {latestAssessment.chief_complaint || 'Not specified'}</p>
                    <div className="flex items-center gap-2">
                      <span className={`px-2.5 py-0.5 text-[9px] rounded-lg border font-bold uppercase ${severityBadge(latestAssessment.severity)}`}>
                        {latestAssessment.severity || 'N/A'}
                      </span>
                      <span className={`px-2.5 py-0.5 text-[9px] rounded-lg border font-bold uppercase ${statusBadge(latestAssessment.status)}`}>
                        {latestAssessment.status}
                      </span>
                    </div>
                  </div>
                  <motion.button whileHover={{ x: 2 }}
                    onClick={() => navigate(`/reports/${latestAssessment.id}`)}
                    className="shrink-0 self-start flex items-center gap-1 text-xs font-bold text-teal-600 hover:text-teal-700">
                    View Report <ChevronRight className="w-3.5 h-3.5" />
                  </motion.button>
                </div>
              </motion.div>
            )}

            {/* Assessment History Table */}
            <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}
              className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
              <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
                <h3 className="font-bold text-slate-800 text-sm">Assessment History</h3>
                <button onClick={() => navigate('/patient/intake')}
                  className="text-xs font-bold text-teal-600 hover:text-teal-700 flex items-center gap-0.5">
                  New <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
              {recentCases.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead>
                      <tr className="bg-slate-50 border-b border-slate-200">
                        {['#', 'Diagnosis', 'Severity', 'Status', 'Date', ''].map(h => (
                          <th key={h} className="px-5 py-3 text-[10px] font-bold text-slate-400 uppercase tracking-wider">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {recentCases.map((c, i) => (
                        <motion.tr key={c.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.35 + i * 0.05 }}
                          className="border-b border-slate-100 hover:bg-slate-50/70 transition-colors">
                          <td className="px-5 py-4 text-xs text-slate-400 font-mono">#{c.id}</td>
                          <td className="px-5 py-4 font-semibold text-slate-800 max-w-[160px] truncate text-xs">{c.final_diagnosis || 'Awaiting'}</td>
                          <td className="px-5 py-4">
                            <span className={`px-2.5 py-0.5 rounded-lg border text-[9px] font-bold uppercase ${severityBadge(c.severity)}`}>{c.severity || 'N/A'}</span>
                          </td>
                          <td className="px-5 py-4">
                            <span className={`px-2.5 py-0.5 rounded-lg border text-[9px] font-bold uppercase ${statusBadge(c.status)}`}>{c.status}</span>
                          </td>
                          <td className="px-5 py-4 text-[10px] text-slate-400">
                            {new Date(c.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                          </td>
                          <td className="px-5 py-4 text-right">
                            <button onClick={() => navigate(`/reports/${c.id}`)}
                              className="text-xs font-bold text-teal-600 hover:text-teal-700 hover:underline">
                              View
                            </button>
                          </td>
                        </motion.tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="py-12 text-center text-sm text-slate-400 italic">No assessments yet.</div>
              )}
            </motion.div>
          </div>

          {/* Right column */}
          <div className="space-y-5">

            {/* Quick Actions */}
            <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.28 }}
              className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-3">
              <h3 className="font-bold text-slate-800 text-sm border-b border-slate-100 pb-3">Quick Actions</h3>
              {[
                { icon: Plus, label: 'New Assessment', sub: 'Run AI diagnostic', color: 'bg-teal-50 text-teal-600', path: '/patient/intake' },
                { icon: FileSearch, label: 'Analyze Medical Report', sub: 'Upload PDF/Image', color: 'bg-indigo-50 text-indigo-600', path: '/reports/analyze' },
                { icon: TrendingUp, label: 'Health Insights', sub: 'View analytics', color: 'bg-sky-50 text-sky-600', path: '/patient/insights' },
              ].map(({ icon: Icon, label, sub, color, path }) => (
                <button key={label} onClick={() => navigate(path)}
                  className="w-full flex items-center gap-3 p-3 rounded-xl border border-slate-100 hover:border-teal-200 hover:bg-slate-50/60 transition-all group text-left">
                  <div className={`w-9 h-9 rounded-xl ${color} flex items-center justify-center shrink-0`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-xs font-bold text-slate-800 group-hover:text-teal-700 transition-colors">{label}</p>
                    <p className="text-[10px] text-slate-400">{sub}</p>
                  </div>
                  <ChevronRight className="w-3.5 h-3.5 text-slate-300 group-hover:text-teal-400 ml-auto shrink-0 transition-colors" />
                </button>
              ))}
            </motion.div>

            {/* Profile Card */}
            <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.33 }}
              className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-3">
              <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
                <div className="w-7 h-7 bg-slate-100 rounded-lg flex items-center justify-center">
                  <Shield className="w-3.5 h-3.5 text-slate-500" />
                </div>
                <h3 className="font-bold text-slate-800 text-sm">Your Profile</h3>
              </div>
              <div className="space-y-2.5">
                {[
                  { label: 'Name', value: displayName },
                  { label: 'Age', value: userAge ? `${userAge} years` : null },
                  { label: 'Gender', value: user?.gender },
                  { label: 'Total Visits', value: `${totalCases} assessments` },
                ].filter(r => r.value).map(r => (
                  <div key={r.label} className="flex justify-between items-center py-1 border-b border-slate-50">
                    <span className="text-xs text-slate-500 font-medium">{r.label}</span>
                    <span className="text-xs font-bold text-slate-800">{r.value}</span>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Emergency alert */}
            {latestAssessment?.triage_level != null && latestAssessment.triage_level <= 2 && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}
                className="bg-red-50 border border-red-200 rounded-2xl p-5 space-y-2">
                <div className="flex items-center gap-2">
                  <motion.span className="w-2.5 h-2.5 bg-red-500 rounded-full"
                    animate={{ scale: [1, 1.5, 1], opacity: [1, 0.4, 1] }} transition={{ duration: 1, repeat: Infinity }} />
                  <p className="text-xs font-extrabold text-red-800 uppercase tracking-wide">⚠ Emergency Alert</p>
                </div>
                <p className="text-[11px] text-red-700 leading-relaxed">
                  Your last assessment indicates urgent findings. Please consult a doctor or visit the nearest emergency room immediately.
                </p>
              </motion.div>
            )}

            {/* Latest vitals */}
            {(hrVal != null || spo2Val != null || bpVal || tempVal != null) && (
              <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.38 }}
                className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-3">
                <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
                  <Heart className="w-4 h-4 text-rose-500" />
                  <h3 className="font-bold text-slate-800 text-sm">Latest Vitals</h3>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  {hrVal != null && (
                    <div className="text-center p-3 bg-rose-50 rounded-xl border border-rose-100">
                      <p className="text-xl font-black text-slate-900">{hrVal}</p>
                      <p className="text-[9px] font-bold text-slate-400 uppercase mt-0.5">bpm</p>
                    </div>
                  )}
                  {spo2Val != null && (
                    <div className="text-center p-3 bg-emerald-50 rounded-xl border border-emerald-100">
                      <p className="text-xl font-black text-slate-900">{spo2Val}%</p>
                      <p className="text-[9px] font-bold text-slate-400 uppercase mt-0.5">SpO2</p>
                    </div>
                  )}
                  {bpVal && (
                    <div className="text-center p-3 bg-sky-50 rounded-xl border border-sky-100">
                      <p className="text-xl font-black text-slate-900">{bpVal}</p>
                      <p className="text-[9px] font-bold text-slate-400 uppercase mt-0.5">BP</p>
                    </div>
                  )}
                  {tempVal != null && (
                    <div className="text-center p-3 bg-amber-50 rounded-xl border border-amber-100">
                      <p className="text-xl font-black text-slate-900">{tempVal}°</p>
                      <p className="text-[9px] font-bold text-slate-400 uppercase mt-0.5">Temp F</p>
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
        className="bg-amber-50 border border-amber-200 rounded-xl px-5 py-3.5 flex items-start gap-3">
        <Sparkles className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
        <p className="text-xs text-amber-800 leading-relaxed">
          <strong>⚕ Medical Disclaimer:</strong> MedAgentix AI provides informational health assessments only —
          it is NOT a substitute for professional medical diagnosis or treatment. Always consult a qualified doctor.
        </p>
      </motion.div>
    </div>
  );
}
