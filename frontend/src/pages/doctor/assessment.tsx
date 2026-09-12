import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import apiClient from '../../services/api-client';
import { motion } from 'framer-motion';
import {
  Users, Activity, Clock, Search, ChevronRight, RefreshCw,
  Stethoscope, ShieldAlert, ShieldCheck, Thermometer, Heart,
  Wind, CheckCircle, AlertTriangle, FileText, Star
} from 'lucide-react';

// ─────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────
interface QueueCase {
  id: number;
  patient_name: string;
  status: string;
  triage_level: number;
  created_at: string;
  chief_complaint: string;
}

interface CaseDetail {
  id: number;
  patient_name: string;
  status: string;
  triage_level: number;
  created_at: string;
  vitals: {
    heart_rate: number;
    oxygen_level: number;
    bp_reading: string;
    temperature: number;
    cholesterol: number;
  };
  symptoms: {
    chief_complaint: string;
    selected_symptoms: { name: string; duration_days: number }[];
  };
  history_and_lifestyle: {
    medical_history: string[];
    lifestyle_factors: string[];
  };
  diagnostic_output: {
    final_diagnosis: string;
    confidence: number;
    severity: string;
    icd_code: string;
    pathophysiology: string;
    patient_age?: number;
    patient_gender?: string;
    pipeline_version: string;
    generated_at: string;
    differential_considerations: { rank: number; condition: string; probability: number }[];
    recommended_drugs: { name: string; dosage: string; purpose: string; route: string; class: string; adr: string }[];
    recommended_tests: { name: string; priority: string; department: string; indication: string }[];
    emergency_status: { is_emergency: boolean; triage_level: number; urgency: string };
    triage?: {
      tier: 'GREEN' | 'YELLOW' | 'RED';
      severity_score: number;
      common_illness_type: string;
      red_flags: string[];
      triage_reason: string;
      display_confidence_label: string;
    };
    lifestyle?: {
      diet: string[];
      when_to_see_doctor: string[];
      precautions: string[];
      workout: string[];
    };
    doctor_note?: string;
    reviewed_by?: string;
    reviewed_at?: string;
  };
}

// ─────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────
const tierConfig = {
  GREEN: { bg: 'bg-emerald-50', border: 'border-emerald-200', text: 'text-emerald-700', icon: '✅', label: 'Likely Mild — No Red Flags' },
  YELLOW: { bg: 'bg-amber-50', border: 'border-amber-200', text: 'text-amber-700', icon: '⚠️', label: 'Clinical Evaluation Recommended' },
  RED: { bg: 'bg-red-50', border: 'border-red-300', text: 'text-red-700', icon: '🚨', label: 'Urgent — Immediate Care Required' },
};

const getESILabel = (lvl: number) => {
  switch (lvl) {
    case 1: return { label: 'ESI 1 — Resuscitation', cls: 'bg-red-100 text-red-800 border-red-300 animate-pulse' };
    case 2: return { label: 'ESI 2 — Emergent', cls: 'bg-orange-100 text-orange-800 border-orange-300' };
    case 3: return { label: 'ESI 3 — Urgent', cls: 'bg-amber-100 text-amber-800 border-amber-300' };
    default: return { label: 'ESI 4 — Routine', cls: 'bg-slate-100 text-slate-700 border-slate-200' };
  }
};

const getSeverityCls = (s: string) => {
  switch (s?.toLowerCase()) {
    case 'mild': return 'bg-green-100 text-green-800 border-green-200';
    case 'moderate': return 'bg-amber-100 text-amber-800 border-amber-200';
    case 'serious': return 'bg-orange-100 text-orange-800 border-orange-200';
    case 'urgent': return 'bg-red-100 text-red-800 border-red-200 animate-pulse';
    default: return 'bg-slate-100 text-slate-700 border-slate-200';
  }
};

// ─────────────────────────────────────────────────────────
// Sub-component: Triage Tier Banner
// ─────────────────────────────────────────────────────────
const TriageBanner: React.FC<{ tier: 'GREEN' | 'YELLOW' | 'RED'; reason: string; illnessType?: string; redFlags?: string[] }> = ({
  tier, reason, illnessType, redFlags
}) => {
  const cfg = tierConfig[tier];
  return (
    <div className={`${cfg.bg} ${cfg.border} border rounded-xl p-4 flex items-start gap-3`}>
      <span className="text-2xl">{cfg.icon}</span>
      <div>
        <p className={`text-xs font-extrabold uppercase tracking-wide ${cfg.text}`}>Triage: {tier} — {cfg.label}</p>
        {illnessType && <p className="text-sm font-semibold text-slate-800 mt-0.5">{illnessType}</p>}
        <p className={`text-xs mt-1 ${cfg.text}`}>{reason}</p>
        {redFlags && redFlags.length > 0 && (
          <p className="text-xs text-red-600 mt-1 font-semibold">⚠ Red flags: {redFlags.slice(0, 3).join(', ')}</p>
        )}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────
// Main Page
// ─────────────────────────────────────────────────────────
export default function DoctorAssessment() {
  const navigate = useNavigate();

  const [cases, setCases] = useState<QueueCase[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [tierFilter, setTierFilter] = useState<'all' | 'GREEN' | 'YELLOW' | 'RED'>('all');
  const [statusFilter, setStatusFilter] = useState('all');

  // Selected case detail panel
  const [selectedCase, setSelectedCase] = useState<CaseDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [isSigningOff, setIsSigningOff] = useState(false);
  const [doctorNote, setDoctorNote] = useState('');
  const [isReviewed, setIsReviewed] = useState(false);
  const [reviewedBy, setReviewedBy] = useState('');

  // ── Fetch case list ──
  const fetchCases = async () => {
    setLoading(true);
    try {
      const res = await apiClient.get('/doctor/cases');
      if (res.data?.success) setCases(res.data.cases);
    } catch { /* silent */ }
    finally { setLoading(false); }
  };

  useEffect(() => { fetchCases(); }, []);

  // ── Fetch single case detail ──
  const openCase = async (id: number) => {
    setDetailLoading(true);
    setSelectedCase(null);
    try {
      const res = await apiClient.get(`/cases/${id}`);
      if (res.data?.success) {
        const c: CaseDetail = res.data.case;
        setSelectedCase(c);
        setIsReviewed(c.status === 'reviewed');
        setDoctorNote(c.diagnostic_output?.doctor_note || '');
        setReviewedBy(c.diagnostic_output?.reviewed_by || '');
      }
    } catch { /* silent */ }
    finally { setDetailLoading(false); }
  };

  // ── Sign off ──
  const handleSignOff = async () => {
    if (!selectedCase) return;
    setIsSigningOff(true);
    try {
      const res = await apiClient.put(`/cases/${selectedCase.id}/review`, { doctor_note: doctorNote });
      if (res.data?.success) {
        setIsReviewed(true);
        setReviewedBy(res.data.reviewed_by || '');
        fetchCases(); // refresh list
      }
    } catch { /* silent */ }
    finally { setIsSigningOff(false); }
  };

  // ── Computed ──
  const filteredCases = cases.filter(c => {
    const matchSearch = c.patient_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      c.chief_complaint.toLowerCase().includes(searchTerm.toLowerCase()) ||
      String(c.id) === searchTerm;
    const matchStatus = statusFilter === 'all' || c.status === statusFilter;
    return matchSearch && matchStatus;
  });

  // Stats
  const total = cases.length;
  const critical = cases.filter(c => c.triage_level <= 2).length;
  const reviewed = cases.filter(c => c.status === 'reviewed').length;
  const pending = cases.filter(c => c.status !== 'reviewed').length;

  const itemVariants = {
    hidden: { opacity: 0, y: 12 },
    show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 120, damping: 18 } }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">

      {/* ── Header ── */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 border-b border-border pb-5">
        <div>
          <h1 className="text-2xl font-extrabold tracking-tight text-foreground flex items-center gap-2">
            <Stethoscope className="w-6 h-6 text-sky-600 shrink-0" />
            Patient Assessments
          </h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            Full triage-aware clinical assessments — tap any case to review the AI diagnostic profile.
          </p>
        </div>
        <button
          onClick={fetchCases}
          className="p-2.5 rounded-xl bg-card border border-border hover:bg-slate-50 text-slate-600 transition flex items-center gap-1.5 text-xs font-semibold"
        >
          <RefreshCw className="w-3.5 h-3.5" /> Refresh
        </button>
      </div>

      {/* ── KPI Row ── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: 'Total Cases', value: total, icon: Users, cls: 'text-sky-700 bg-sky-50' },
          { label: 'Critical Triage', value: critical, icon: ShieldAlert, cls: 'text-red-600 bg-red-50' },
          { label: 'Signed Off', value: reviewed, icon: ShieldCheck, cls: 'text-emerald-600 bg-emerald-50' },
          { label: 'Pending Review', value: pending, icon: Clock, cls: 'text-amber-600 bg-amber-50' },
        ].map(kpi => (
          <div key={kpi.label} className="bg-card border border-border rounded-2xl p-4 shadow-xs flex items-center justify-between">
            <div>
              <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">{kpi.label}</p>
              <p className="text-2xl font-black text-foreground mt-1">{kpi.value}</p>
            </div>
            <div className={`p-2.5 rounded-xl ${kpi.cls}`}>
              <kpi.icon className="w-5 h-5" />
            </div>
          </div>
        ))}
      </div>

      {/* ── Two-panel layout ── */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 items-start">

        {/* LEFT: Case List */}
        <div className="lg:col-span-2 space-y-3">

          {/* Search + Filter */}
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <input
                type="text"
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
                placeholder="Search patient or complaint..."
                className="w-full bg-card border border-border rounded-xl pl-9 pr-3 py-2 text-xs focus:outline-none focus:ring-1 focus:ring-primary placeholder-slate-400"
              />
            </div>
            <select
              value={statusFilter}
              onChange={e => setStatusFilter(e.target.value)}
              className="bg-card border border-border rounded-xl px-3 py-2 text-xs font-semibold focus:outline-none"
            >
              <option value="all">All Status</option>
              <option value="completed">Completed</option>
              <option value="reviewed">Reviewed</option>
              <option value="pending">Pending</option>
            </select>
          </div>

          {/* Case cards */}
          {loading ? (
            <div className="space-y-3">
              {[1, 2, 3].map(i => (
                <div key={i} className="h-20 bg-slate-100 rounded-2xl animate-pulse" />
              ))}
            </div>
          ) : filteredCases.length === 0 ? (
            <div className="bg-card border border-border rounded-2xl p-8 text-center text-muted-foreground text-xs">
              No cases found.
            </div>
          ) : (
            <motion.div
              className="space-y-2"
              initial="hidden"
              animate="show"
              variants={{ show: { transition: { staggerChildren: 0.05 } } }}
            >
              {filteredCases.map(c => {
                const esi = getESILabel(c.triage_level);
                const isSelected = selectedCase?.id === c.id;
                return (
                  <motion.div
                    key={c.id}
                    variants={itemVariants}
                    onClick={() => openCase(c.id)}
                    className={`bg-card border rounded-2xl p-4 cursor-pointer hover:shadow-md transition-all ${
                      isSelected ? 'border-sky-400 ring-1 ring-sky-300 shadow-md' : 'border-border hover:border-slate-300'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <p className="text-sm font-bold text-foreground truncate">{c.patient_name}</p>
                          <span className="text-[9px] text-muted-foreground shrink-0">#{c.id}</span>
                        </div>
                        <p className="text-[11px] text-muted-foreground italic truncate mt-0.5">"{c.chief_complaint}"</p>
                        <div className="flex items-center gap-1.5 mt-2 flex-wrap">
                          <span className={`text-[9px] font-bold px-2 py-0.5 rounded border uppercase ${esi.cls}`}>
                            {esi.label}
                          </span>
                          <span className={`text-[9px] font-bold px-2 py-0.5 rounded uppercase ${
                            c.status === 'reviewed' ? 'bg-emerald-100 text-emerald-700' :
                            c.status === 'completed' ? 'bg-sky-100 text-sky-700' :
                            'bg-slate-100 text-slate-600'
                          }`}>
                            {c.status}
                          </span>
                        </div>
                      </div>
                      <ChevronRight className={`w-4 h-4 shrink-0 mt-1 transition ${isSelected ? 'text-sky-500' : 'text-muted-foreground'}`} />
                    </div>
                    <p className="text-[10px] text-muted-foreground mt-2">
                      {new Date(c.created_at).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' })}
                    </p>
                  </motion.div>
                );
              })}
            </motion.div>
          )}
        </div>

        {/* RIGHT: Assessment Detail Panel */}
        <div className="lg:col-span-3 space-y-5">
          {detailLoading && (
            <div className="bg-card border border-border rounded-2xl p-8 flex flex-col items-center gap-3">
              <div className="w-8 h-8 border-2 border-sky-500 border-t-transparent rounded-full animate-spin" />
              <p className="text-xs text-muted-foreground animate-pulse">Loading assessment...</p>
            </div>
          )}

          {!detailLoading && !selectedCase && (
            <div className="bg-card border border-dashed border-border rounded-2xl p-12 flex flex-col items-center gap-3 text-center">
              <Stethoscope className="w-10 h-10 text-slate-300" />
              <p className="text-sm font-semibold text-muted-foreground">Select a case to view its assessment</p>
              <p className="text-xs text-muted-foreground">Click any patient card on the left</p>
            </div>
          )}

          {!detailLoading && selectedCase && (() => {
            const d = selectedCase.diagnostic_output;
            const triage = d.triage;
            const tier = triage?.tier || 'YELLOW';

            return (
              <motion.div
                key={selectedCase.id}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.25 }}
                className="space-y-5"
              >
                {/* Patient Header */}
                <div className="bg-gradient-to-r from-sky-50 to-blue-50 border border-sky-200 rounded-2xl p-5">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-[10px] uppercase font-bold tracking-widest text-sky-700">Clinical Assessment</p>
                      <h2 className="text-xl font-extrabold text-slate-900 mt-0.5">{selectedCase.patient_name}</h2>
                      <div className="flex flex-wrap gap-3 mt-2 text-xs font-semibold text-slate-600">
                        <span>👤 Age: {d.patient_age || 'N/A'}</span>
                        <span>⚧ {d.patient_gender || 'N/A'}</span>
                        <span>📋 MRN-{selectedCase.id}</span>
                        <span>🗓 {new Date(selectedCase.created_at).toLocaleDateString()}</span>
                      </div>
                    </div>
                    <div className="flex flex-col gap-1.5 items-end shrink-0">
                      <span className={`text-[9px] font-bold px-2.5 py-1 rounded-lg border uppercase ${getSeverityCls(d.severity)}`}>
                        {d.severity}
                      </span>
                      {isReviewed ? (
                        <span className="flex items-center gap-1 bg-emerald-50 border border-emerald-200 text-emerald-700 px-2.5 py-1 rounded-lg text-[9px] font-bold">
                          <CheckCircle className="w-3 h-3" /> Signed Off
                        </span>
                      ) : (
                        <span className="bg-amber-50 border border-amber-200 text-amber-700 px-2.5 py-1 rounded-lg text-[9px] font-bold">
                          Needs Review
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                {/* ─── SECTION: Triage Assessment ─── */}
                <div className="bg-card border border-border rounded-2xl p-5 space-y-4">
                  <h3 className="text-xs font-extrabold text-slate-800 uppercase tracking-wide flex items-center gap-2">
                    <span className="w-1 h-4 bg-violet-500 rounded-full"></span>
                    AI Triage Assessment
                  </h3>

                  {/* Tier Banner */}
                  {triage ? (
                    <TriageBanner
                      tier={tier}
                      reason={triage.triage_reason}
                      illnessType={triage.common_illness_type}
                      redFlags={triage.red_flags}
                    />
                  ) : (
                    <div className="bg-slate-50 border border-border rounded-xl p-3 text-xs text-muted-foreground italic">
                      Triage data unavailable (legacy case — pre-Sprint 8).
                    </div>
                  )}

                  {/* Metrics Grid */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    {[
                      { label: 'Triage Tier', value: tier, color: tier === 'GREEN' ? 'text-emerald-600' : tier === 'RED' ? 'text-red-600' : 'text-amber-600' },
                      { label: 'Severity Score', value: triage ? `${triage.severity_score}/100` : 'N/A', color: 'text-foreground' },
                      { label: 'AI Confidence', value: `${d.confidence}%`, color: 'text-sky-600' },
                      { label: 'ESI Level', value: selectedCase.triage_level || 'N/A', color: 'text-foreground' },
                    ].map(m => (
                      <div key={m.label} className="bg-slate-50 border border-border rounded-xl p-3 text-center">
                        <p className="text-[10px] font-bold text-muted-foreground uppercase mb-1">{m.label}</p>
                        <p className={`text-lg font-black ${m.color}`}>{m.value}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* ─── SECTION: Vitals ─── */}
                <div className="bg-card border border-border rounded-2xl p-5 space-y-3">
                  <h3 className="text-xs font-extrabold text-slate-800 uppercase tracking-wide flex items-center gap-2">
                    <span className="w-1 h-4 bg-rose-500 rounded-full"></span>
                    Vital Signs
                  </h3>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <div className="bg-rose-50 border border-rose-100 rounded-xl p-3">
                      <div className="flex items-center gap-1 mb-1"><Heart className="w-3.5 h-3.5 text-rose-500" /><span className="text-[10px] font-bold text-rose-700 uppercase">HR</span></div>
                      <p className="text-lg font-black text-foreground">{selectedCase.vitals.heart_rate} <span className="text-xs font-normal text-muted-foreground">bpm</span></p>
                      <p className={`text-[9px] font-bold mt-0.5 ${selectedCase.vitals.heart_rate > 100 ? 'text-red-500' : 'text-green-600'}`}>
                        {selectedCase.vitals.heart_rate > 100 ? 'Tachycardia' : selectedCase.vitals.heart_rate < 60 ? 'Bradycardia' : 'Normal'}
                      </p>
                    </div>
                    <div className="bg-blue-50 border border-blue-100 rounded-xl p-3">
                      <div className="flex items-center gap-1 mb-1"><Wind className="w-3.5 h-3.5 text-blue-500" /><span className="text-[10px] font-bold text-blue-700 uppercase">SpO2</span></div>
                      <p className="text-lg font-black text-foreground">{selectedCase.vitals.oxygen_level}<span className="text-xs font-normal text-muted-foreground">%</span></p>
                      <p className={`text-[9px] font-bold mt-0.5 ${selectedCase.vitals.oxygen_level < 95 ? 'text-red-500' : 'text-green-600'}`}>
                        {selectedCase.vitals.oxygen_level < 92 ? 'Severe Hypoxemia' : selectedCase.vitals.oxygen_level < 95 ? 'Mild Hypoxemia' : 'Normal'}
                      </p>
                    </div>
                    <div className="bg-orange-50 border border-orange-100 rounded-xl p-3">
                      <div className="flex items-center gap-1 mb-1"><Thermometer className="w-3.5 h-3.5 text-orange-500" /><span className="text-[10px] font-bold text-orange-700 uppercase">Temp</span></div>
                      <p className="text-lg font-black text-foreground">{selectedCase.vitals.temperature}<span className="text-xs font-normal text-muted-foreground">°F</span></p>
                      <p className={`text-[9px] font-bold mt-0.5 ${selectedCase.vitals.temperature >= 100.4 ? 'text-red-500' : 'text-green-600'}`}>
                        {selectedCase.vitals.temperature >= 100.4 ? 'Pyrexia' : 'Normothermia'}
                      </p>
                    </div>
                    <div className="bg-purple-50 border border-purple-100 rounded-xl p-3">
                      <div className="flex items-center gap-1 mb-1"><Activity className="w-3.5 h-3.5 text-purple-500" /><span className="text-[10px] font-bold text-purple-700 uppercase">BP</span></div>
                      <p className="text-lg font-black text-foreground">{selectedCase.vitals.bp_reading}</p>
                      <p className="text-[9px] font-bold text-green-600 mt-0.5">Normotensive</p>
                    </div>
                  </div>
                </div>

                {/* ─── SECTION: Reported Symptoms ─── */}
                <div className="bg-card border border-border rounded-2xl p-5 space-y-3">
                  <h3 className="text-xs font-extrabold text-slate-800 uppercase tracking-wide flex items-center gap-2">
                    <span className="w-1 h-4 bg-sky-500 rounded-full"></span>
                    Reported Symptoms
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedCase.symptoms.selected_symptoms.map(s => (
                      <span key={s.name} className="bg-sky-50 border border-sky-200 text-sky-800 text-xs font-semibold px-3 py-1 rounded-full">
                        {s.name} <span className="text-sky-500 font-normal">· {s.duration_days === 0 ? 'Today' : `${s.duration_days}d`}</span>
                      </span>
                    ))}
                  </div>
                  {selectedCase.symptoms.chief_complaint && (
                    <div className="bg-slate-50 border border-border rounded-xl p-3">
                      <p className="text-[10px] font-bold text-muted-foreground uppercase mb-1">In Patient's Words</p>
                      <p className="text-xs text-foreground italic">"{selectedCase.symptoms.chief_complaint}"</p>
                    </div>
                  )}
                </div>

                {/* ─── SECTION: Diagnosis ─── */}
                <div className="bg-card border border-border rounded-2xl p-5 space-y-3">
                  <h3 className="text-xs font-extrabold text-slate-800 uppercase tracking-wide flex items-center gap-2">
                    <span className="w-1 h-4 bg-indigo-500 rounded-full"></span>
                    AI Diagnosis
                  </h3>
                  <div className={`rounded-xl p-4 border space-y-2 ${
                    tier === 'GREEN' ? 'bg-emerald-50 border-emerald-200' :
                    tier === 'RED' ? 'bg-red-50 border-red-200' :
                    'bg-amber-50 border-amber-200'
                  }`}>
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-[10px] font-bold text-muted-foreground uppercase">Primary Diagnosis</p>
                        <p className="text-xl font-extrabold text-foreground">{d.final_diagnosis}</p>
                        <p className="text-xs text-muted-foreground font-semibold mt-0.5">ICD-10: {d.icd_code || 'N/A'}</p>
                      </div>
                      <span className={`text-[10px] font-bold px-2.5 py-1 rounded-lg border uppercase shrink-0 ${getSeverityCls(d.severity)}`}>
                        {d.severity}
                      </span>
                    </div>
                    <div className="border-t border-border/50 pt-2">
                      <p className="text-[10px] font-bold text-muted-foreground uppercase mb-1">AI Explanation</p>
                      <p className="text-xs text-foreground/90 leading-relaxed">{d.pathophysiology}</p>
                    </div>
                  </div>

                  {/* Differential Considerations */}
                  {d.differential_considerations.length > 0 && (
                    <div>
                      <p className="text-[10px] font-bold text-muted-foreground uppercase mb-2">Differential Considerations</p>
                      <div className="space-y-1.5">
                        {d.differential_considerations.slice(0, 4).map((diff, i) => (
                          <div key={i} className="flex items-center justify-between bg-slate-50 border border-border rounded-lg px-3 py-2">
                            <div className="flex items-center gap-2">
                              <span className="w-5 h-5 bg-indigo-100 text-indigo-700 rounded-full flex items-center justify-center text-[9px] font-bold">{diff.rank}</span>
                              <span className="text-xs font-semibold text-foreground">{diff.condition}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="w-20 bg-slate-200 rounded-full h-1.5">
                                <div className="bg-indigo-500 h-1.5 rounded-full" style={{ width: `${diff.probability}%` }} />
                              </div>
                              <span className="text-xs font-bold text-indigo-600">{diff.probability}%</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* ─── SECTION: Lifestyle & Groq Recommendations ─── */}
                {d.lifestyle && (
                  <div className="bg-card border border-border rounded-2xl p-5 space-y-4">
                    <h3 className="text-xs font-extrabold text-slate-800 uppercase tracking-wide flex items-center gap-2">
                      <span className="w-1 h-4 bg-emerald-500 rounded-full"></span>
                      AI Lifestyle Recommendations
                      <span className="text-[9px] bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded font-bold">Groq</span>
                    </h3>
                    {(d.lifestyle.diet?.length || 0) > 0 && (
                      <div>
                        <p className="text-[10px] font-bold text-emerald-700 uppercase mb-2">🥗 Diet & Recovery</p>
                        <ul className="space-y-1.5">
                          {d.lifestyle.diet.slice(0, 5).map((tip, i) => (
                            <li key={i} className="flex items-start gap-2 text-xs text-foreground/90">
                              <span className="text-emerald-500 mt-0.5 shrink-0">•</span>{tip}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {(d.lifestyle.when_to_see_doctor?.length || 0) > 0 && (
                      <div>
                        <p className="text-[10px] font-bold text-amber-700 uppercase mb-2">⚠️ Escalation Triggers — Refer If:</p>
                        <ul className="space-y-1.5">
                          {d.lifestyle.when_to_see_doctor.slice(0, 4).map((tip, i) => (
                            <li key={i} className="flex items-start gap-2 text-xs text-foreground/90">
                              <span className="text-amber-500 mt-0.5 shrink-0">•</span>{tip}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {(d.lifestyle.precautions?.length || 0) > 0 && (
                      <div>
                        <p className="text-[10px] font-bold text-slate-600 uppercase mb-2">🛡️ Precautions</p>
                        <ul className="space-y-1.5">
                          {d.lifestyle.precautions.slice(0, 4).map((tip, i) => (
                            <li key={i} className="flex items-start gap-2 text-xs text-foreground/90">
                              <span className="text-slate-400 mt-0.5 shrink-0">•</span>{tip}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}

                {/* ─── SECTION: Doctor Notes + Sign Off ─── */}
                <div className="bg-card border-2 border-violet-200 rounded-2xl p-5 space-y-3">
                  <h3 className="text-xs font-extrabold text-slate-800 uppercase tracking-wide flex items-center gap-2">
                    <span className="w-1 h-4 bg-violet-500 rounded-full"></span>
                    Doctor Notes & Sign-Off
                  </h3>

                  {isReviewed && reviewedBy && (
                    <div className="bg-emerald-50 border border-emerald-200 rounded-xl px-4 py-2 text-xs text-emerald-700 font-semibold flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 shrink-0" />
                      Case signed off by Dr. {reviewedBy}
                      {d.reviewed_at && (
                        <span className="text-[10px] text-emerald-500 font-normal ml-1">
                          · {new Date(d.reviewed_at).toLocaleString()}
                        </span>
                      )}
                    </div>
                  )}

                  <textarea
                    value={doctorNote}
                    onChange={e => setDoctorNote(e.target.value)}
                    disabled={isReviewed}
                    rows={4}
                    placeholder={isReviewed ? 'Notes saved. Case is reviewed.' : 'Add clinical observations, differential notes, or follow-up instructions...'}
                    className="w-full border border-border rounded-xl p-3 text-xs text-foreground bg-slate-50 focus:outline-none focus:ring-2 focus:ring-violet-400 resize-none placeholder-slate-400 disabled:opacity-60 disabled:cursor-not-allowed"
                  />

                  <div className="flex items-center justify-between gap-3">
                    <button
                      onClick={() => navigate(`/reports/${selectedCase.id}`)}
                      className="flex items-center gap-1.5 text-xs text-primary font-semibold hover:underline"
                    >
                      <FileText className="w-4 h-4" /> View Full Clinical Report →
                    </button>
                    {!isReviewed && (
                      <button
                        onClick={handleSignOff}
                        disabled={isSigningOff}
                        className="bg-emerald-600 hover:bg-emerald-700 text-white px-5 py-2.5 rounded-xl text-xs font-bold flex items-center gap-1.5 shadow-sm transition disabled:opacity-60"
                      >
                        <CheckCircle className="w-4 h-4 shrink-0" />
                        {isSigningOff ? 'Signing Off...' : 'Save Notes & Sign Off'}
                      </button>
                    )}
                  </div>
                </div>

              </motion.div>
            );
          })()}
        </div>
      </div>
    </div>
  );
}
