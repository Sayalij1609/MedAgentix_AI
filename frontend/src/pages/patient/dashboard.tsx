import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import apiClient from '../../services/api-client';
import { useAuth } from '../../context/auth-context';
import { useToast } from '../../context/toast-context';
import { 
  FileText, Activity, Clock, 
  Heart, Wind, RefreshCw,
  Calendar, ChevronRight, Stethoscope, ClipboardList, Plus, Sparkles
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
    heart_rate: number;
    oxygen_level: number;
    bp_reading?: string;
    systolic_bp?: number;
    diastolic_bp?: number;
    temperature: number;
    cholesterol: number;
  };
}

export default function PatientDashboard() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { toast } = useToast();
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState('');
  
  const [totalCases, setTotalCases] = useState(0);
  const [pendingReviews, setPendingReviews] = useState(0);
  const [recentCases, setRecentCases] = useState<CaseSummary[]>([]);
  const [latestAssessment, setLatestAssessment] = useState<CaseSummary | null>(null);
  
  const [lastUpdated, setLastUpdated] = useState<string>('');

  const displayName = user?.name || user?.email?.split('@')[0] || 'there';

  // Calculate age from DOB
  const calculateAge = (dob: string): number | null => {
    if (!dob) return null;
    const birthDate = new Date(dob);
    const today = new Date();
    let age = today.getFullYear() - birthDate.getFullYear();
    const monthDiff = today.getMonth() - birthDate.getMonth();
    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
      age--;
    }
    return age >= 0 ? age : null;
  };
  const userAge = user?.date_of_birth ? calculateAge(user.date_of_birth) : null;

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const response = await apiClient.get('/patient/dashboard');
      if (response.data && response.data.success) {
        setTotalCases(response.data.total_cases);
        setPendingReviews(response.data.pending_reviews);
        setRecentCases(response.data.recent_cases);
        setLatestAssessment(response.data.latest_assessment || null);
        
        setLastUpdated(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
      }
    } catch (err: any) {
      console.error(err);
      setErrorMsg('Failed to load dashboard.');
      toast('Failed to load your dashboard.', 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const handleRefresh = () => {
    fetchDashboardData();
    toast('Dashboard refreshed.', 'success');
  };

  // Only extract vitals from actual data — no hardcoded defaults
  const hrVal = latestAssessment?.vitals?.heart_rate ?? null;
  const spo2Val = latestAssessment?.vitals?.oxygen_level ?? null;
  const tempVal = latestAssessment?.vitals?.temperature ?? null;
  const bpVal = latestAssessment?.vitals?.bp_reading ?? null;
  // Only show vitals section if the patient actually provided at least one value
  const hasVitals = hrVal != null || spo2Val != null || tempVal != null || bpVal != null;

  const getSeverityBadgeClass = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'mild':
        return 'bg-green-50 text-green-700 border-green-200';
      case 'moderate':
        return 'bg-amber-50 text-amber-700 border-amber-200';
      case 'severe': case 'serious':
        return 'bg-orange-50 text-orange-700 border-orange-200';
      case 'critical': case 'urgent':
        return 'bg-red-50 text-red-700 border-red-200';
      default:
        return 'bg-slate-50 text-slate-700 border-slate-200';
    }
  };

  const getStatusBadgeClass = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'completed':
        return 'bg-emerald-50 border border-emerald-250 text-emerald-700';
      case 'reviewed':
        return 'bg-sky-50 border border-sky-250 text-sky-700';
      case 'processing':
        return 'bg-amber-50 border border-amber-250 text-amber-700 animate-pulse';
      default:
        return 'bg-slate-50 border border-slate-200 text-slate-655';
    }
  };

  // Skeletons during initial load
  if (loading && recentCases.length === 0) {
    return (
      <div className="space-y-8 max-w-5xl mx-auto animate-pulse">
        <div className="h-32 bg-slate-100 rounded-2xl"></div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          <div className="h-28 bg-slate-100 rounded-2xl"></div>
          <div className="h-28 bg-slate-100 rounded-2xl"></div>
          <div className="h-28 bg-slate-100 rounded-2xl"></div>
        </div>
        <div className="h-64 bg-slate-100 rounded-2xl"></div>
      </div>
    );
  }

  const isNewPatient = totalCases === 0;

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      
      {/* Header bar */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 border-b border-border pb-4 text-left">
        <div>
          <h1 className="text-2xl font-extrabold tracking-tight text-slate-900">My Health Dashboard</h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            View your assessments, track your health, and start new consultations.
          </p>
        </div>
        <div className="flex items-center gap-3 self-end sm:self-auto text-xs text-slate-500 font-semibold">
          {lastUpdated && <span>Updated: {lastUpdated}</span>}
          <button 
            onClick={handleRefresh}
            className="p-2 rounded-xl bg-card border border-border hover:bg-slate-50 text-slate-600 transition flex items-center gap-1.5"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* ============================================================ */}
      {/* NEW PATIENT: Welcome Onboarding */}
      {/* ============================================================ */}
      {isNewPatient ? (
        <div className="space-y-6">
          {/* Welcome Hero */}
          <div className="relative overflow-hidden bg-gradient-to-br from-sky-50 via-indigo-50 to-purple-50 border border-sky-150 p-8 md:p-10 rounded-2xl shadow-xs text-left">
            <div className="absolute top-0 right-0 -translate-y-16 translate-x-16 w-72 h-72 bg-sky-200/20 rounded-full blur-3xl pointer-events-none"></div>
            <div className="absolute bottom-0 left-0 translate-y-8 -translate-x-8 w-48 h-48 bg-indigo-200/20 rounded-full blur-3xl pointer-events-none"></div>
            
            <div className="relative z-10 space-y-4 max-w-xl">
              <div className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-indigo-500" />
                <span className="bg-indigo-100 text-indigo-700 px-2.5 py-0.5 rounded-full text-[10px] font-extrabold uppercase tracking-wider">
                  Welcome
                </span>
              </div>
              <h2 className="text-2xl md:text-3xl font-extrabold tracking-tight text-slate-900">
                Hello, {displayName}! 👋
              </h2>
              <p className="text-slate-600 text-sm md:text-base leading-relaxed">
                Welcome to <strong>MedAgentix AI</strong> — your personal AI health assessment assistant. 
                Describe your symptoms and our AI will analyze them to help you understand what might be going on. 
                This is <strong>not</strong> a substitute for a doctor, but a helpful starting point.
              </p>
            </div>
          </div>

          {/* How it works cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-2xl p-5 text-left space-y-2.5 hover:shadow-sm transition">
              <div className="w-10 h-10 rounded-xl bg-sky-50 flex items-center justify-center">
                <ClipboardList className="w-5 h-5 text-sky-600" />
              </div>
              <h3 className="text-sm font-bold text-slate-800">1. Describe Symptoms</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Tell us what you're feeling in your own words. Select common symptoms or type freely.
              </p>
            </div>

            <div className="bg-card border border-border rounded-2xl p-5 text-left space-y-2.5 hover:shadow-sm transition">
              <div className="w-10 h-10 rounded-xl bg-indigo-50 flex items-center justify-center">
                <Activity className="w-5 h-5 text-indigo-600" />
              </div>
              <h3 className="text-sm font-bold text-slate-800">2. AI Analyzes</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Our 8-stage AI pipeline processes your symptoms through multiple specialist agents.
              </p>
            </div>

            <div className="bg-card border border-border rounded-2xl p-5 text-left space-y-2.5 hover:shadow-sm transition">
              <div className="w-10 h-10 rounded-xl bg-emerald-50 flex items-center justify-center">
                <FileText className="w-5 h-5 text-emerald-600" />
              </div>
              <h3 className="text-sm font-bold text-slate-800">3. Get Your Report</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Receive a detailed health report with possible conditions, medications, and next steps.
              </p>
            </div>
          </div>

          {/* CTA button */}
          <div className="flex justify-center">
            <button
              onClick={() => navigate('/patient/intake')}
              className="bg-primary text-primary-foreground hover:opacity-95 active:scale-[0.98] transition px-8 py-4 rounded-xl font-bold text-sm shadow-lg shadow-primary/20 flex items-center gap-2"
            >
              <Plus className="w-4.5 h-4.5" />
              <span>Start Your First Health Assessment</span>
            </button>
          </div>

          {/* Disclaimer */}
          <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-left">
            <p className="text-xs text-amber-800 leading-relaxed">
              <strong>⚕ Important:</strong> MedAgentix AI provides informational health assessments only. 
              It is NOT a medical diagnosis. Always consult a qualified doctor for proper examination 
              and treatment.
            </p>
          </div>
        </div>
      ) : (
        /* ============================================================ */
        /* RETURNING PATIENT: Full Dashboard */
        /* ============================================================ */
        <>
          {/* Welcome Banner */}
          <div className="relative overflow-hidden flex flex-col md:flex-row md:items-center justify-between gap-6 bg-gradient-to-r from-sky-50 to-indigo-50 border border-sky-150 p-6 md:p-8 rounded-2xl shadow-xs text-left">
            <div className="absolute top-0 right-0 -translate-y-12 translate-x-12 w-64 h-64 bg-sky-200/20 rounded-full blur-3xl pointer-events-none"></div>
            <div className="relative z-10 space-y-1.5 max-w-xl">
              <h2 className="text-xl md:text-2xl font-extrabold tracking-tight text-slate-900">
                Welcome back, {displayName}!
              </h2>
              <p className="text-slate-600 text-xs md:text-sm leading-relaxed">
                You have <strong>{totalCases}</strong> health assessment{totalCases !== 1 ? 's' : ''} on record
                {pendingReviews > 0 && <>, including <strong className="text-amber-600">{pendingReviews} pending</strong> review{pendingReviews !== 1 ? 's' : ''}</>}.
                {pendingReviews === 0 && <>.  Everything looks up to date!</>}
              </p>
            </div>
            <button
              onClick={() => navigate('/patient/intake')}
              className="relative z-10 shrink-0 bg-primary text-primary-foreground hover:opacity-95 active:scale-[0.98] transition px-5 py-3 rounded-xl font-extrabold text-xs shadow-md flex items-center gap-1.5 self-start md:self-auto"
            >
              <Plus className="w-4 h-4" />
              <span>New Assessment</span>
            </button>
          </div>

          {/* Summary Metric Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {/* Total Assessments */}
            <div className="bg-card border border-border rounded-2xl p-4 shadow-xs text-left hover:shadow-sm transition">
              <div className="flex justify-between items-center">
                <span className="text-[10px] font-extrabold text-muted-foreground uppercase tracking-wider">Assessments</span>
                <div className="p-1.5 rounded-lg bg-sky-50 text-sky-600">
                  <FileText className="w-4 h-4" />
                </div>
              </div>
              <div className="mt-2.5">
                <span className="text-2xl font-black text-slate-900 tracking-tight">{totalCases}</span>
                <span className="text-xs text-muted-foreground font-bold ml-1">total</span>
              </div>
            </div>

            {/* Pending Reviews */}
            <div className="bg-card border border-border rounded-2xl p-4 shadow-xs text-left hover:shadow-sm transition">
              <div className="flex justify-between items-center">
                <span className="text-[10px] font-extrabold text-muted-foreground uppercase tracking-wider">Pending</span>
                <div className="p-1.5 rounded-lg bg-amber-50 text-amber-600">
                  <Clock className="w-4 h-4" />
                </div>
              </div>
              <div className="mt-2.5">
                <span className="text-2xl font-black text-slate-900 tracking-tight">{pendingReviews}</span>
                <span className="text-xs text-muted-foreground font-bold ml-1">pending</span>
              </div>
            </div>

            {/* Latest Vitals — Heart Rate */}
            <div className="bg-card border border-border rounded-2xl p-4 shadow-xs text-left hover:shadow-sm transition">
              <div className="flex justify-between items-center">
                <span className="text-[10px] font-extrabold text-muted-foreground uppercase tracking-wider">Heart Rate</span>
                <div className="p-1.5 rounded-lg bg-red-50 text-red-600">
                  <Heart className="w-4 h-4" />
                </div>
              </div>
              <div className="mt-2.5">
                {hrVal != null ? (
                  <>
                    <span className="text-2xl font-black text-slate-900 tracking-tight">{hrVal}</span>
                    <span className="text-xs text-muted-foreground font-bold ml-1">bpm</span>
                  </>
                ) : (
                  <span className="text-sm text-muted-foreground italic">No data yet</span>
                )}
              </div>
            </div>

            {/* Latest Vitals — SpO2 */}
            <div className="bg-card border border-border rounded-2xl p-4 shadow-xs text-left hover:shadow-sm transition">
              <div className="flex justify-between items-center">
                <span className="text-[10px] font-extrabold text-muted-foreground uppercase tracking-wider">Oxygen</span>
                <div className="p-1.5 rounded-lg bg-emerald-50 text-emerald-600">
                  <Wind className="w-4 h-4" />
                </div>
              </div>
              <div className="mt-2.5">
                {spo2Val != null ? (
                  <>
                    <span className="text-2xl font-black text-slate-900 tracking-tight">{spo2Val}</span>
                    <span className="text-xs text-muted-foreground font-bold ml-1">%</span>
                  </>
                ) : (
                  <span className="text-sm text-muted-foreground italic">No data yet</span>
                )}
              </div>
            </div>
          </div>

          {/* Main Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

            {/* Left Column — Assessments List */}
            <div className="lg:col-span-2 space-y-6">

              {/* Latest Assessment Quick Card */}
              {latestAssessment && (
                <div className="bg-card border border-border rounded-2xl p-5 shadow-xs text-left">
                  <div className="flex justify-between items-center mb-3">
                    <div className="flex items-center gap-1.5">
                      <Stethoscope className="w-4 h-4 text-indigo-600" />
                      <h3 className="text-sm font-extrabold text-slate-800 uppercase tracking-wide">Latest Assessment</h3>
                    </div>
                    <span className="text-[10px] text-muted-foreground font-semibold">
                      {new Date(latestAssessment.created_at).toLocaleDateString(undefined, { month: 'long', day: 'numeric', year: 'numeric' })}
                    </span>
                  </div>

                  <div className="flex flex-col sm:flex-row justify-between gap-3 p-4 bg-slate-50 rounded-xl border border-border">
                    <div className="space-y-1">
                      <p className="text-base font-bold text-slate-900">{latestAssessment.final_diagnosis || 'Awaiting Diagnosis'}</p>
                      <p className="text-xs text-muted-foreground">
                        Chief complaint: {latestAssessment.chief_complaint || 'Not specified'}
                      </p>
                      <div className="flex items-center gap-2 mt-1.5">
                        <span className={`px-2 py-0.5 text-[9px] rounded-lg border uppercase font-bold ${getSeverityBadgeClass(latestAssessment.severity)}`}>
                          {latestAssessment.severity || 'N/A'}
                        </span>
                        <span className={`px-2 py-0.5 text-[9px] rounded uppercase font-bold ${getStatusBadgeClass(latestAssessment.status)}`}>
                          {latestAssessment.status}
                        </span>
                      </div>
                    </div>
                    <button
                      onClick={() => navigate(`/reports/${latestAssessment.id}`)}
                      className="shrink-0 self-start text-xs text-primary font-bold hover:underline flex items-center gap-0.5"
                    >
                      View Full Report
                      <ChevronRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              )}

              {/* Assessment History Table */}
              <div className="bg-card border border-border rounded-2xl p-5 shadow-xs text-left space-y-4">
                <div className="flex justify-between items-center">
                  <h3 className="text-sm font-extrabold text-slate-800 uppercase tracking-wide">Assessment History</h3>
                  <button 
                    onClick={() => navigate('/patient/intake')}
                    className="text-xs text-primary font-bold hover:underline flex items-center"
                  >
                    <span>New Assessment</span>
                    <ChevronRight className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div className="border border-border rounded-xl overflow-hidden bg-card">
                  {recentCases.length > 0 ? (
                    <div className="overflow-x-auto">
                      <table className="min-w-full divide-y divide-border text-left">
                        <thead className="bg-slate-50 text-[10px] font-bold text-muted-foreground uppercase">
                          <tr>
                            <th className="px-4 py-2.5">#</th>
                            <th className="px-4 py-2.5">Diagnosis</th>
                            <th className="px-4 py-2.5">Severity</th>
                            <th className="px-4 py-2.5">Status</th>
                            <th className="px-4 py-2.5">Date</th>
                            <th className="px-4 py-2.5 text-right">Action</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border text-xs font-semibold text-foreground">
                          {recentCases.map((c) => (
                            <tr key={c.id} className="hover:bg-slate-50/50 transition">
                              <td className="px-4 py-3 text-muted-foreground">{c.id}</td>
                              <td className="px-4 py-3 font-bold text-slate-900 max-w-[150px] truncate">
                                {c.final_diagnosis || 'Awaiting'}
                              </td>
                              <td className="px-4 py-3">
                                <span className={`px-2 py-0.5 text-[9px] rounded-lg border uppercase ${getSeverityBadgeClass(c.severity)}`}>
                                  {c.severity || 'N/A'}
                                </span>
                              </td>
                              <td className="px-4 py-3">
                                <span className={`px-2 py-0.5 text-[9px] rounded uppercase ${getStatusBadgeClass(c.status)}`}>
                                  {c.status}
                                </span>
                              </td>
                              <td className="px-4 py-3 text-[10px] text-muted-foreground">
                                {new Date(c.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                              </td>
                              <td className="px-4 py-3 text-right">
                                <button
                                  onClick={() => navigate(`/reports/${c.id}`)}
                                  className="text-xs text-primary font-bold hover:underline"
                                >
                                  View Report
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="p-8 text-center text-xs text-muted-foreground italic">
                      No assessments yet. Start your first health assessment above.
                    </div>
                  )}
                </div>
              </div>

            </div>

            {/* Right Column — Quick Info & Triage Alert */}
            <div className="space-y-6 text-left">
              
              {/* Patient Profile Card */}
              <div className="bg-card border border-border rounded-2xl p-5 shadow-xs space-y-3.5">
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground font-bold uppercase tracking-wider">
                  <Activity className="w-4 h-4 text-indigo-600 shrink-0" />
                  <span>Your Profile</span>
                </div>

                <div className="space-y-2.5">
                  <div className="flex justify-between items-center border-b border-border pb-2.5">
                    <span className="text-xs font-semibold text-slate-600">Name</span>
                    <span className="text-xs font-bold text-slate-800">{displayName}</span>
                  </div>
                  {userAge !== null && (
                    <div className="flex justify-between items-center border-b border-border pb-2.5">
                      <span className="text-xs font-semibold text-slate-600">Age</span>
                      <span className="text-xs font-bold text-slate-800">{userAge} years</span>
                    </div>
                  )}
                  {user?.gender && (
                    <div className="flex justify-between items-center border-b border-border pb-2.5">
                      <span className="text-xs font-semibold text-slate-600">Gender</span>
                      <span className="text-xs font-bold text-slate-800">{user.gender}</span>
                    </div>
                  )}
                  <div className="flex justify-between items-center">
                    <span className="text-xs font-semibold text-slate-600">Total Assessments</span>
                    <span className="text-xs font-bold text-slate-800">{totalCases}</span>
                  </div>
                </div>
              </div>

              {/* Active Triage Alert */}
              {latestAssessment && latestAssessment.triage_level != null && latestAssessment.triage_level <= 2 && (
                <div className="bg-red-50 border border-red-200 text-red-800 p-5 rounded-2xl space-y-2.5">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 bg-red-600 rounded-full animate-ping shrink-0"></span>
                    <h4 className="text-xs font-extrabold uppercase tracking-wide">⚠ Emergency Alert</h4>
                  </div>
                  <p className="text-[11px] leading-relaxed font-medium">
                    Your last assessment indicates readings that may need urgent attention. 
                    Please consult a doctor or visit the nearest hospital immediately.
                  </p>
                </div>
              )}

              {/* Latest Vitals Summary */}
              {hasVitals && (
                <div className="bg-card border border-border rounded-2xl p-5 shadow-xs space-y-4">
                  <div className="flex items-center gap-1.5 border-b border-border pb-3">
                    <Heart className="w-4 h-4 text-rose-500 shrink-0" />
                    <h3 className="text-sm font-extrabold text-slate-800 uppercase tracking-wide">Latest Vitals</h3>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    {hrVal != null && (
                      <div className="text-center p-2.5 bg-rose-50 rounded-xl">
                        <p className="text-lg font-black text-slate-800">{hrVal}</p>
                        <p className="text-[9px] font-bold text-muted-foreground uppercase">Heart Rate</p>
                      </div>
                    )}
                    {spo2Val != null && (
                      <div className="text-center p-2.5 bg-emerald-50 rounded-xl">
                        <p className="text-lg font-black text-slate-800">{spo2Val}%</p>
                        <p className="text-[9px] font-bold text-muted-foreground uppercase">Oxygen</p>
                      </div>
                    )}
                    {bpVal && (
                      <div className="text-center p-2.5 bg-sky-50 rounded-xl">
                        <p className="text-lg font-black text-slate-800">{bpVal}</p>
                        <p className="text-[9px] font-bold text-muted-foreground uppercase">Blood Pressure</p>
                      </div>
                    )}
                    {tempVal != null && (
                      <div className="text-center p-2.5 bg-amber-50 rounded-xl">
                        <p className="text-lg font-black text-slate-800">{tempVal}°F</p>
                        <p className="text-[9px] font-bold text-muted-foreground uppercase">Temperature</p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Recent Activity — derived from real cases */}
              {recentCases.length > 0 && (
                <div className="bg-card border border-border rounded-2xl p-5 shadow-xs space-y-4">
                  <div className="flex items-center gap-1.5 border-b border-border pb-3">
                    <Calendar className="w-4 h-4 text-sky-600 shrink-0" />
                    <h3 className="text-sm font-extrabold text-slate-800 uppercase tracking-wide">Recent Activity</h3>
                  </div>

                  <div className="relative border-l border-slate-200 pl-4 ml-2.5 space-y-4 text-xs">
                    {recentCases.slice(0, 3).map((c) => (
                      <div key={c.id} className="relative">
                        <span className={`absolute -left-[21.5px] top-1 w-2.5 h-2.5 rounded-full border-2 border-white ring-4 ${
                          c.status === 'completed' ? 'bg-emerald-600 ring-emerald-50' : 
                          c.status === 'reviewed' ? 'bg-sky-600 ring-sky-50' : 
                          'bg-slate-400 ring-slate-50'
                        }`}></span>
                        <p className="font-bold text-slate-900">{c.final_diagnosis || 'Assessment'}</p>
                        <p className="text-[10px] text-muted-foreground">{c.chief_complaint || 'Health assessment completed'}</p>
                        <span className="text-[9px] text-slate-400 font-semibold block mt-0.5">
                          {new Date(c.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

          </div>
        </>
      )}

    </div>
  );
}
