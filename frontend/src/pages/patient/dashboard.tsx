import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import apiClient from '../../services/api-client';
import { useAuth } from '../../context/auth-context';
import { FileText, Activity, Shield, Clock, ArrowRight, Clipboard } from 'lucide-react';

interface CaseSummary {
  id: number;
  status: string;
  created_at: string;
  chief_complaint: string;
  final_diagnosis: string;
  severity: string;
  triage_level?: number;
}

export default function PatientDashboard() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState('');
  
  const [totalCases, setTotalCases] = useState(0);
  const [pendingReviews, setPendingReviews] = useState(0);
  const [recentCases, setRecentCases] = useState<CaseSummary[]>([]);
  const [latestAssessment, setLatestAssessment] = useState<CaseSummary | null>(null);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const response = await apiClient.get('/patient/dashboard');
      if (response.data && response.data.success) {
        setTotalCases(response.data.total_cases);
        setPendingReviews(response.data.pending_reviews);
        setRecentCases(response.data.recent_cases);
        setLatestAssessment(response.data.latest_assessment);
      }
    } catch (err: any) {
      console.error(err);
      setErrorMsg('Failed to load dashboard metrics.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const getSeverityBadgeClass = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'mild':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'moderate':
        return 'bg-amber-100 text-amber-800 border-amber-200';
      case 'serious':
        return 'bg-orange-100 text-orange-850 border-orange-200';
      case 'urgent':
        return 'bg-red-100 text-red-800 border-red-200 animate-pulse';
      default:
        return 'bg-slate-100 text-slate-800 border-border';
    }
  };

  const getStatusBadgeClass = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return 'bg-emerald-100 text-emerald-800';
      case 'reviewed':
        return 'bg-blue-100 text-blue-800';
      case 'processing':
        return 'bg-yellow-100 text-yellow-800 animate-pulse';
      default:
        return 'bg-slate-100 text-slate-700';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-3 border-primary border-t-transparent rounded-full animate-spin"></div>
          <p className="text-muted-foreground text-xs animate-pulse">Retrieving patient session files...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Welcome Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-sky-50 border border-sky-200 text-sky-950 p-6 md:p-8 rounded-2xl shadow-sm">
        <div className="space-y-1">
          <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight text-sky-950">Welcome, {user?.email.split('@')[0]}!</h1>
          <p className="text-sky-800 text-sm md:text-base">
            Access past assessments, review triage reports, or initialize a new clinical assessment.
          </p>
        </div>
        <button
          onClick={() => navigate('/patient/intake')}
          className="bg-primary text-primary-foreground hover:opacity-90 active:scale-95 transition-all px-6 py-3 rounded-xl font-bold text-sm shadow-md self-start md:self-auto"
        >
          Start New Assessment
        </button>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="bg-card border border-border rounded-2xl shadow-sm hover:shadow-md transition overflow-hidden">
          <div className="p-5 space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Total Assessments</span>
              <div className="p-2 rounded-lg bg-sky-50 text-sky-800">
                <FileText className="w-4 h-4" />
              </div>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-extrabold text-slate-900">{totalCases}</span>
              <span className="text-xs text-muted-foreground font-medium">completed cases</span>
            </div>
          </div>
        </div>

        <div className="bg-card border border-border rounded-2xl shadow-sm hover:shadow-md transition overflow-hidden">
          <div className="p-5 space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Active Follow-ups</span>
              <div className="p-2 rounded-lg bg-sky-50 text-sky-800">
                <Activity className="w-4 h-4" />
              </div>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-extrabold text-slate-900">{pendingReviews}</span>
              <span className="text-xs text-muted-foreground font-medium">awaiting review</span>
            </div>
          </div>
        </div>

        <div className="bg-card border border-border rounded-2xl shadow-sm hover:shadow-md transition overflow-hidden sm:col-span-2 lg:col-span-1">
          <div className="p-5 space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Assigned Clinic</span>
              <div className="p-2 rounded-lg bg-sky-50 text-sky-800">
                <Shield className="w-4 h-4" />
              </div>
            </div>
            <div className="space-y-0.5">
              <span className="text-lg font-bold text-slate-900 block">MedAgentix Main</span>
              <span className="text-xs text-emerald-600 font-semibold flex items-center gap-1">
                <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 animate-ping"></span>
                Secure EMR Link
              </span>
            </div>
          </div>
        </div>
      </div>

      {errorMsg && (
        <div className="bg-red-50 text-red-600 border border-red-200 p-4 rounded-xl text-sm font-medium">
          ⚠️ {errorMsg}
        </div>
      )}

      {/* Content Columns */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Column: Latest Assessment detail card */}
        <div className="lg:col-span-2 space-y-4">
          <h2 className="text-xl font-bold tracking-tight text-foreground">Latest Assessment</h2>
          
          {latestAssessment ? (
            <div className="bg-card border border-border rounded-2xl shadow-sm p-6 space-y-6 hover:shadow-md transition">
              <div className="flex justify-between items-start gap-4">
                <div className="space-y-1.5 text-left">
                  <span className="text-[10px] font-bold text-sky-850 uppercase tracking-widest">Primary Working Diagnosis</span>
                  <h3 className="text-xl font-bold text-slate-900 leading-tight">
                    {latestAssessment.final_diagnosis}
                  </h3>
                  <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                    <Clock className="w-3.5 h-3.5" />
                    <span>
                      Generated on {new Date(latestAssessment.created_at).toLocaleDateString(undefined, { 
                        year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' 
                      })}
                    </span>
                  </div>
                </div>
                <div className="flex flex-col items-end gap-1.5 shrink-0">
                  <span className={`px-2.5 py-1 text-xs font-semibold rounded-lg border ${getSeverityBadgeClass(latestAssessment.severity)}`}>
                    {latestAssessment.severity}
                  </span>
                  <span className={`px-2 py-0.5 text-[10px] font-bold rounded uppercase ${getStatusBadgeClass(latestAssessment.status)}`}>
                    {latestAssessment.status}
                  </span>
                </div>
              </div>

              {/* Chief Complaint block */}
              <div className="bg-slate-50 p-4 rounded-xl border border-border text-left">
                <h4 className="text-xs font-bold text-muted-foreground uppercase tracking-wider mb-1">Presented Chief Complaint</h4>
                <p className="text-sm text-foreground/90 italic leading-relaxed">
                  "{latestAssessment.chief_complaint}"
                </p>
              </div>

              {/* Triage ESI Warning Indicator if serious */}
              {latestAssessment.triage_level && latestAssessment.triage_level <= 2 && (
                <div className="bg-red-50 text-red-800 border border-red-200 p-4 rounded-xl text-xs font-semibold flex items-center gap-2">
                  <span className="inline-block w-2.5 h-2.5 rounded-full bg-red-600 animate-pulse shrink-0"></span>
                  <span>
                    Clinical Alert: Emergency Severity Index (ESI) Level {latestAssessment.triage_level} detected. Direct medical triage recommended.
                  </span>
                </div>
              )}

              {/* View detail button */}
              <button
                onClick={() => navigate(`/reports/${latestAssessment.id}`)}
                className="w-full bg-primary hover:opacity-90 text-primary-foreground py-3 rounded-xl text-sm font-bold shadow-sm transition flex justify-center items-center gap-1.5"
              >
                <span>Access Clinical Workup Report</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <div className="bg-slate-50 border border-dashed border-border rounded-2xl h-64 flex flex-col items-center justify-center p-6 text-center space-y-4">
              <div className="w-12 h-12 rounded-full bg-sky-50 flex items-center justify-center text-sky-700">
                <Clipboard className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <p className="font-bold text-slate-800 text-sm">No Active Medical Files Found</p>
                <p className="text-xs text-muted-foreground max-w-xs leading-relaxed">
                  Submit a diagnostic intake questionnaire to populate your clinical metrics and report cards.
                </p>
              </div>
              <button
                onClick={() => navigate('/patient/intake')}
                className="px-5 py-2.5 bg-primary text-primary-foreground hover:opacity-90 rounded-xl text-xs font-bold transition shadow-sm"
              >
                Initialize Intake Assessment
              </button>
            </div>
          )}
        </div>

        {/* Right Column: Case History Summary List */}
        <div className="space-y-4">
          <h2 className="text-xl font-bold tracking-tight text-foreground">Recent Assessments</h2>
          
          <div className="space-y-3">
            {recentCases.length > 0 ? (
              recentCases.map(c => (
                <div 
                  key={c.id} 
                  onClick={() => navigate(`/reports/${c.id}`)}
                  className="bg-card border border-border p-4 rounded-xl shadow-xs hover:border-primary cursor-pointer hover:shadow-sm transition flex justify-between items-center gap-4"
                >
                  <div className="space-y-1 overflow-hidden">
                    <h4 className="text-sm font-bold text-foreground truncate">{c.final_diagnosis}</h4>
                    <p className="text-[10px] text-muted-foreground truncate">
                      {new Date(c.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} — {c.chief_complaint}
                    </p>
                  </div>
                  <span className={`px-2 py-0.5 text-[9px] font-bold rounded uppercase whitespace-nowrap ${getStatusBadgeClass(c.status)}`}>
                    {c.status}
                  </span>
                </div>
              ))
            ) : (
              <p className="text-xs text-muted-foreground italic text-center py-8">No past cases listed.</p>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
