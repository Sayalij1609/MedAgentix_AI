import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import apiClient from '../../services/api-client';
import { useAuth } from '../../context/auth-context';
import { useToast } from '../../context/toast-context';
import { 
  FileText, Activity, Shield, Clock, ArrowRight, Clipboard, 
  Heart, Thermometer, Wind, CheckCircle, AlertTriangle, RefreshCw,
  TrendingUp, Calendar, User, ChevronRight, Zap
} from 'lucide-react';
import { motion } from 'framer-motion';

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

// Animated Counter Component
const AnimatedCounter: React.FC<{ value: number; duration?: number }> = ({ value, duration = 1 }) => {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let start = 0;
    const end = value;
    if (end === 0) return;
    const totalMiliseconds = duration * 1000;
    const incrementTime = Math.max(10, Math.floor(totalMiliseconds / end));
    
    const timer = setInterval(() => {
      start += 1;
      setCount(start);
      if (start >= end) {
        clearInterval(timer);
        setCount(end);
      }
    }, incrementTime);

    return () => clearInterval(timer);
  }, [value, duration]);

  return <span>{count}</span>;
};

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
  
  const [activeMetricTab, setActiveMetricTab] = useState<'hr' | 'bp' | 'spo2'>('hr');
  const [lastUpdated, setLastUpdated] = useState<string>('');

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const response = await apiClient.get('/patient/dashboard');
      if (response.data && response.data.success) {
        setTotalCases(response.data.total_cases);
        setPendingReviews(response.data.pending_reviews);
        setRecentCases(response.data.recent_cases);
        
        // Fetch full case details for the latest assessment to get its actual vitals
        const latest = response.data.latest_assessment;
        if (latest) {
          try {
            const caseDetailRes = await apiClient.get(`/cases/${latest.id}`);
            if (caseDetailRes.data && caseDetailRes.data.success) {
              setLatestAssessment(caseDetailRes.data.case);
            } else {
              setLatestAssessment(latest);
            }
          } catch (e) {
            setLatestAssessment(latest);
          }
        } else {
          setLatestAssessment(null);
        }
        
        setLastUpdated(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
      }
    } catch (err: any) {
      console.error(err);
      setErrorMsg('Failed to load dashboard metrics.');
      toast('Failed to sync clinical records.', 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const handleRefresh = () => {
    fetchDashboardData();
    toast('Clinical portal synced successfully.', 'success');
  };

  // Safe vitals extraction
  const hrVal = latestAssessment?.vitals?.heart_rate ?? 72;
  const spo2Val = latestAssessment?.vitals?.oxygen_level ?? 98;
  const tempVal = latestAssessment?.vitals?.temperature ?? 98.6;
  const bpVal = latestAssessment?.vitals?.bp_reading ?? '120/80';

  // Calculate Health Score
  const calculateHealthScore = () => {
    let score = 94;
    if (latestAssessment) {
      if (hrVal > 100 || hrVal < 60) score -= 10;
      if (spo2Val < 95) score -= 15;
      if (tempVal > 100.4 || tempVal < 96.5) score -= 8;
      
      const severity = latestAssessment.severity?.toLowerCase();
      if (severity === 'urgent') score -= 25;
      else if (severity === 'serious') score -= 15;
      else if (severity === 'moderate') score -= 8;
    }
    return Math.max(35, score);
  };

  const healthScore = calculateHealthScore();

  const getHealthScoreDetails = (score: number) => {
    if (score >= 90) return { label: 'Optimal Stability', color: 'text-teal-650', bg: 'bg-teal-50', border: 'border-teal-200', text: 'Physiological markers are within recommended clinical baselines.' };
    if (score >= 75) return { label: 'Mild Variation', color: 'text-amber-600', bg: 'bg-amber-50', border: 'border-amber-200', text: 'Minor vitals fluctuations observed. Maintain hydration and log logs.' };
    return { label: 'Clinical Review Needed', color: 'text-red-650', bg: 'bg-red-50', border: 'border-red-200', text: 'Out of range physiological signs recorded. differential screening triggered.' };
  };

  const scoreDetails = getHealthScoreDetails(healthScore);

  const getSeverityBadgeClass = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'mild':
        return 'bg-green-50 text-green-700 border-green-200';
      case 'moderate':
        return 'bg-amber-50 text-amber-700 border-amber-200';
      case 'serious':
        return 'bg-orange-50 text-orange-850 border-orange-200';
      case 'urgent':
        return 'bg-red-50 text-red-700 border-red-200 animate-pulse';
      default:
        return 'bg-slate-50 text-slate-700 border-slate-200';
    }
  };

  const getStatusBadgeClass = (status: string) => {
    switch (status.toLowerCase()) {
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

  // Mock 7-day trend analytics (simulating continuous sensor feeds)
  const getTrendData = () => {
    switch (activeMetricTab) {
      case 'bp':
        return {
          points: [118, 122, 120, 124, 119, 122, hrVal > 110 ? 132 : 120], // systolic
          points2: [78, 82, 79, 81, 78, 80, 78], // diastolic
          labels: ['29 May', '30 May', '31 May', '01 Jun', '02 Jun', '03 Jun', '04 Jun'],
          min: 60,
          max: 150,
          suffix: 'mmHg',
        };
      case 'spo2':
        return {
          points: [99, 98, 99, 98, 99, 98, spo2Val],
          labels: ['29 May', '30 May', '31 May', '01 Jun', '02 Jun', '03 Jun', '04 Jun'],
          min: 90,
          max: 100,
          suffix: '%',
        };
      case 'hr':
      default:
        return {
          points: [68, 72, 70, 75, 71, 74, hrVal],
          labels: ['29 May', '30 May', '31 May', '01 Jun', '02 Jun', '03 Jun', '04 Jun'],
          min: 50,
          max: 110,
          suffix: 'bpm',
        };
    }
  };

  const trend = getTrendData();

  // Create SVG points coordinates helper
  const renderSvgPath = (dataPoints: number[], min: number, max: number) => {
    const width = 500;
    const height = 150;
    const padding = 25;
    const xStep = (width - padding * 2) / (dataPoints.length - 1);
    
    return dataPoints.map((val, idx) => {
      const x = padding + idx * xStep;
      // invert y since SVG y increases downwards
      const normVal = (val - min) / (max - min);
      const y = height - padding - normVal * (height - padding * 2);
      return { x, y, val };
    });
  };

  const svgCoords = renderSvgPath(trend.points, trend.min, trend.max);
  const pathString = svgCoords.map((c, i) => `${i === 0 ? 'M' : 'L'} ${c.x} ${c.y}`).join(' ');
  const areaPathString = svgCoords.length > 0 
    ? `${pathString} L ${svgCoords[svgCoords.length - 1].x} 125 L ${svgCoords[0].x} 125 Z`
    : '';

  // BP secondary line coordinates
  const svgCoords2 = trend.points2 ? renderSvgPath(trend.points2, trend.min, trend.max) : [];
  const pathString2 = svgCoords2.map((c, i) => `${i === 0 ? 'M' : 'L'} ${c.x} ${c.y}`).join(' ');

  // Skeletons during initial load
  if (loading && recentCases.length === 0) {
    return (
      <div className="space-y-8 max-w-5xl mx-auto animate-pulse">
        <div className="h-32 bg-slate-100 rounded-2xl"></div>
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-6">
          <div className="h-28 bg-slate-100 rounded-2xl"></div>
          <div className="h-28 bg-slate-100 rounded-2xl"></div>
          <div className="h-28 bg-slate-100 rounded-2xl"></div>
          <div className="h-28 bg-slate-100 rounded-2xl"></div>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 h-96 bg-slate-100 rounded-2xl"></div>
          <div className="h-96 bg-slate-100 rounded-2xl"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      
      {/* Header bar with controls */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 border-b border-border pb-4 text-left">
        <div>
          <h1 className="text-2xl font-extrabold tracking-tight text-slate-900">Patient Dashboard</h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            Holistic vitals logs, health score, and diagnostic tracking console.
          </p>
        </div>
        <div className="flex items-center gap-3 self-end sm:self-auto text-xs text-slate-500 font-semibold">
          <span>Synced: {lastUpdated || 'Loading...'}</span>
          <button 
            onClick={handleRefresh}
            className="p-2 rounded-xl bg-card border border-border hover:bg-slate-50 text-slate-600 transition flex items-center gap-1.5"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Sync EMR</span>
          </button>
        </div>
      </div>

      {/* Welcome Banner */}
      <div className="relative overflow-hidden flex flex-col md:flex-row md:items-center justify-between gap-6 bg-gradient-to-r from-sky-50 to-indigo-50 border border-sky-150 p-6 md:p-8 rounded-2xl shadow-xs text-left">
        <div className="absolute top-0 right-0 -translate-y-12 translate-x-12 w-64 h-64 bg-sky-200/20 rounded-full blur-3xl pointer-events-none"></div>
        <div className="relative z-10 space-y-1.5 max-w-xl">
          <span className="bg-sky-100 text-sky-850 px-2.5 py-0.5 rounded-full text-[10px] font-extrabold uppercase tracking-wider inline-flex items-center gap-1">
            <Zap className="w-3 h-3 text-sky-700 animate-pulse" />
            Active Clinical Session
          </span>
          <h2 className="text-xl md:text-2xl font-extrabold tracking-tight text-slate-900">
            Welcome back, {user?.email.split('@')[0]}!
          </h2>
          <p className="text-slate-655 text-xs md:text-sm leading-relaxed">
            Your MedAgentix CDSS node is fully synced with the local hospital network. Review your latest diagnostic parameters, view attributions on the Insights tab, or request a diagnostic screening.
          </p>
        </div>
        <button
          onClick={() => navigate('/patient/intake')}
          className="relative z-10 shrink-0 bg-primary text-primary-foreground hover:opacity-95 active:scale-[0.98] transition px-5 py-3 rounded-xl font-extrabold text-xs shadow-md flex items-center gap-1.5 self-start md:self-auto"
        >
          <span>Request Diagnosis</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Vitals & Summary Metrics Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        
        {/* Heart Rate Vitals Card */}
        <div className="bg-card border border-border rounded-2xl p-4 shadow-xs flex flex-col justify-between hover:shadow-sm transition text-left">
          <div className="flex justify-between items-center">
            <span className="text-[10px] font-extrabold text-muted-foreground uppercase tracking-wider">Heart Rate</span>
            <div className="p-1.5 rounded-lg bg-red-50 text-red-650">
              <Heart className="w-4 h-4 animate-[pulse_1.2s_infinite]" />
            </div>
          </div>
          <div className="mt-2.5">
            <span className="text-2xl font-black text-slate-900 tracking-tight">{hrVal}</span>
            <span className="text-xs text-muted-foreground font-bold ml-1">bpm</span>
          </div>
          <span className={`text-[10px] font-bold mt-1 inline-flex items-center gap-1 ${hrVal > 100 || hrVal < 60 ? 'text-amber-600' : 'text-emerald-650'}`}>
            {hrVal > 100 || hrVal < 60 ? 'Fluctuating rhythm' : 'Normal sinus'}
          </span>
        </div>

        {/* Blood Pressure Vitals Card */}
        <div className="bg-card border border-border rounded-2xl p-4 shadow-xs flex flex-col justify-between hover:shadow-sm transition text-left">
          <div className="flex justify-between items-center">
            <span className="text-[10px] font-extrabold text-muted-foreground uppercase tracking-wider">Blood Pressure</span>
            <div className="p-1.5 rounded-lg bg-sky-50 text-sky-750">
              <Activity className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-2.5">
            <span className="text-2xl font-black text-slate-900 tracking-tight">{bpVal}</span>
            <span className="text-xs text-muted-foreground font-bold ml-1">mmHg</span>
          </div>
          <span className="text-[10px] font-bold mt-1 text-emerald-650 inline-flex items-center gap-1">
            Normotensive
          </span>
        </div>

        {/* Body Temperature Vitals Card */}
        <div className="bg-card border border-border rounded-2xl p-4 shadow-xs flex flex-col justify-between hover:shadow-sm transition text-left">
          <div className="flex justify-between items-center">
            <span className="text-[10px] font-extrabold text-muted-foreground uppercase tracking-wider">Temperature</span>
            <div className="p-1.5 rounded-lg bg-amber-50 text-amber-750">
              <Thermometer className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-2.5">
            <span className="text-2xl font-black text-slate-900 tracking-tight">{tempVal}</span>
            <span className="text-xs text-muted-foreground font-bold ml-0.5">°F</span>
          </div>
          <span className={`text-[10px] font-bold mt-1 inline-flex items-center gap-1 ${tempVal >= 100.4 ? 'text-red-600' : 'text-emerald-650'}`}>
            {tempVal >= 100.4 ? 'Fever active' : 'Normal range'}
          </span>
        </div>

        {/* Oxygen level Vitals Card */}
        <div className="bg-card border border-border rounded-2xl p-4 shadow-xs flex flex-col justify-between hover:shadow-sm transition text-left">
          <div className="flex justify-between items-center">
            <span className="text-[10px] font-extrabold text-muted-foreground uppercase tracking-wider">SpO2 Oxygen</span>
            <div className="p-1.5 rounded-lg bg-emerald-50 text-emerald-700">
              <Wind className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-2.5">
            <span className="text-2xl font-black text-slate-900 tracking-tight">{spo2Val}</span>
            <span className="text-xs text-muted-foreground font-bold ml-0.5">%</span>
          </div>
          <span className={`text-[10px] font-bold mt-1 inline-flex items-center gap-1 ${spo2Val < 95 ? 'text-red-600 font-extrabold' : 'text-emerald-650'}`}>
            {spo2Val < 95 ? 'Hypoxemia warning' : 'Optimal oxygen'}
          </span>
        </div>

      </div>

      {/* Main Grid: Health score, 7-day trend, assessments list, activities */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Left Column: Health Score Card & 7-Day Trend Chart */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* Health Score Widget */}
          <div className="bg-card border border-border rounded-2xl p-5 shadow-xs flex flex-col sm:flex-row items-center gap-6 text-left">
            {/* Radial score circle */}
            <div className="relative w-28 h-28 shrink-0 flex items-center justify-center">
              <svg className="w-full h-full transform -rotate-90">
                {/* Background Track */}
                <circle cx="56" cy="56" r="46" fill="transparent" stroke="#f1f5f9" strokeWidth="9" />
                {/* Score Indicator */}
                <circle 
                  cx="56" 
                  cy="56" 
                  r="46" 
                  fill="transparent" 
                  stroke={healthScore >= 90 ? '#0d9488' : healthScore >= 75 ? '#d97706' : '#dc2626'} 
                  strokeWidth="9" 
                  strokeDasharray={`${2 * Math.PI * 46}`}
                  strokeDashoffset={`${2 * Math.PI * 46 * (1 - healthScore / 100)}`}
                  strokeLinecap="round"
                  className="transition-all duration-1000 ease-out"
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-3xl font-black tracking-tight text-slate-800">
                  <AnimatedCounter value={healthScore} />
                </span>
                <span className="text-[9px] font-bold text-muted-foreground uppercase">Stability</span>
              </div>
            </div>

            <div className="space-y-1.5 flex-1">
              <span className={`px-2.5 py-0.5 rounded-full text-[9px] font-extrabold uppercase inline-block border ${scoreDetails.color} ${scoreDetails.bg} ${scoreDetails.border}`}>
                {scoreDetails.label}
              </span>
              <h3 className="text-base font-extrabold text-slate-800">Integrated Health Score Index</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {scoreDetails.text} MedAgentix compiles vitals indices, active diagnoses severity metrics, and chronic risk factors to calculate this index.
              </p>
            </div>
          </div>

          {/* 7-Day Trend Charts Container */}
          <div className="bg-card border border-border rounded-2xl p-5 shadow-xs text-left">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 border-b border-border pb-3">
              <div className="flex items-center gap-1.5">
                <TrendingUp className="w-4 h-4 text-sky-600 shrink-0" />
                <h3 className="text-sm font-extrabold text-slate-800 uppercase tracking-wide">Physiological Trends (7 Days)</h3>
              </div>
              <div className="flex border border-border rounded-xl p-0.5 bg-slate-50 text-[10px] font-bold text-slate-600">
                <button
                  onClick={() => setActiveMetricTab('hr')}
                  className={`px-3 py-1.5 rounded-lg transition ${activeMetricTab === 'hr' ? 'bg-card text-slate-900 border border-border shadow-xs' : 'hover:bg-slate-100'}`}
                >
                  Heart Rate
                </button>
                <button
                  onClick={() => setActiveMetricTab('bp')}
                  className={`px-3 py-1.5 rounded-lg transition ${activeMetricTab === 'bp' ? 'bg-card text-slate-900 border border-border shadow-xs' : 'hover:bg-slate-100'}`}
                >
                  Blood Pressure
                </button>
                <button
                  onClick={() => setActiveMetricTab('spo2')}
                  className={`px-3 py-1.5 rounded-lg transition ${activeMetricTab === 'spo2' ? 'bg-card text-slate-900 border border-border shadow-xs' : 'hover:bg-slate-100'}`}
                >
                  Oxygen (SpO2)
                </button>
              </div>
            </div>

            {/* Custom Responsive SVG Chart */}
            <div className="relative mt-4">
              <svg viewBox="0 0 500 150" className="w-full h-36 overflow-visible">
                <defs>
                  <linearGradient id="gradientArea" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#0d9488" stopOpacity="0.15" />
                    <stop offset="100%" stopColor="#0d9488" stopOpacity="0.00" />
                  </linearGradient>
                </defs>

                {/* Y-axis gridlines */}
                <line x1="25" y1="25" x2="475" y2="25" stroke="#f1f5f9" strokeWidth="1" strokeDasharray="4 4" />
                <line x1="25" y1="75" x2="475" y2="75" stroke="#f1f5f9" strokeWidth="1" strokeDasharray="4 4" />
                <line x1="25" y1="125" x2="475" y2="125" stroke="#e2e8f0" strokeWidth="1" />

                {/* Area under curve (gradient) */}
                {areaPathString && activeMetricTab !== 'bp' && (
                  <path d={areaPathString} fill="url(#gradientArea)" />
                )}

                {/* Main line path */}
                {pathString && (
                  <path 
                    d={pathString} 
                    fill="none" 
                    stroke="#0d9488" 
                    strokeWidth="2.5" 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                  />
                )}

                {/* Secondary BP Line (diastolic) */}
                {activeMetricTab === 'bp' && pathString2 && (
                  <path 
                    d={pathString2} 
                    fill="none" 
                    stroke="#0284c7" 
                    strokeWidth="2" 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                  />
                )}

                {/* Data Points / Intersections */}
                {svgCoords.map((pt, idx) => (
                  <g key={`pt-${idx}`}>
                    <circle 
                      cx={pt.x} 
                      cy={pt.y} 
                      r="4.5" 
                      fill="#ffffff" 
                      stroke="#0d9488" 
                      strokeWidth="2" 
                    />
                    <text 
                      x={pt.x} 
                      y={pt.y - 8} 
                      textAnchor="middle" 
                      className="text-[9px] font-bold fill-slate-700"
                    >
                      {pt.val}
                    </text>
                  </g>
                ))}

                {/* Diastolic Data Points for BP */}
                {activeMetricTab === 'bp' && svgCoords2.map((pt, idx) => (
                  <circle 
                    key={`pt2-${idx}`}
                    cx={pt.x} 
                    cy={pt.y} 
                    r="3.5" 
                    fill="#ffffff" 
                    stroke="#0284c7" 
                    strokeWidth="1.5" 
                  />
                ))}

                {/* X Axis Labels */}
                {trend.labels.map((lbl, idx) => {
                  const x = 25 + idx * ((500 - 50) / (trend.labels.length - 1));
                  return (
                    <text 
                      key={`lbl-${idx}`} 
                      x={x} 
                      y="142" 
                      textAnchor="middle" 
                      className="text-[9px] font-semibold fill-slate-400"
                    >
                      {lbl}
                    </text>
                  );
                })}
              </svg>
            </div>
            
            <div className="flex items-center gap-4 mt-2.5 text-[10px] text-muted-foreground font-semibold">
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full bg-teal-600 block"></span>
                <span>Primary Metric ({trend.suffix})</span>
              </span>
              {activeMetricTab === 'bp' && (
                <span className="flex items-center gap-1">
                  <span className="w-2.5 h-2.5 rounded-full bg-sky-650 block"></span>
                  <span>Diastolic Metric (mmHg)</span>
                </span>
              )}
            </div>
          </div>

          {/* Recent Assessments Table */}
          <div className="bg-card border border-border rounded-2xl p-5 shadow-xs text-left space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-sm font-extrabold text-slate-800 uppercase tracking-wide">Case Assessment Roster</h3>
              <button 
                onClick={() => navigate('/patient/intake')}
                className="text-xs text-primary font-bold hover:underline flex items-center"
              >
                <span>New Intake Case</span>
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>

            <div className="border border-border rounded-xl overflow-hidden bg-card">
              {recentCases.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-border text-left">
                    <thead className="bg-slate-50 text-[10px] font-bold text-muted-foreground uppercase">
                      <tr>
                        <th className="px-4 py-2.5">ID</th>
                        <th className="px-4 py-2.5">Primary Diagnosis</th>
                        <th className="px-4 py-2.5">Urgency</th>
                        <th className="px-4 py-2.5">Status</th>
                        <th className="px-4 py-2.5">Date</th>
                        <th className="px-4 py-2.5 text-right">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border text-xs font-semibold text-foreground">
                      {recentCases.map((c) => (
                        <tr key={c.id} className="hover:bg-slate-50/50 transition">
                          <td className="px-4 py-3 text-muted-foreground">#{c.id}</td>
                          <td className="px-4 py-3 font-bold text-slate-900 max-w-[150px] truncate">
                            {c.final_diagnosis}
                          </td>
                          <td className="px-4 py-3">
                            <span className={`px-2 py-0.5 text-[9px] rounded-lg border uppercase ${getSeverityBadgeClass(c.severity)}`}>
                              {c.severity}
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
                              Open Workup
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="p-8 text-center text-xs text-muted-foreground italic">
                  No registered medical case assessments found in Postgres SQL node.
                </div>
              )}
            </div>
          </div>

        </div>

        {/* Right Column: EMR Info, Triage alerts, & Activity Feed */}
        <div className="space-y-6 text-left">
          
          {/* Clinic Node Connectivity Status */}
          <div className="bg-card border border-border rounded-2xl p-5 shadow-xs space-y-4">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground font-bold uppercase tracking-wider">
              <Shield className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>Security Integration</span>
            </div>
            
            <div className="space-y-3.5">
              <div className="flex justify-between items-center border-b border-border pb-2.5">
                <span className="text-xs font-semibold text-slate-655">Clinician Assignment</span>
                <span className="text-xs font-bold text-slate-800">Assigned Queue</span>
              </div>
              <div className="flex justify-between items-center border-b border-border pb-2.5">
                <span className="text-xs font-semibold text-slate-655">Database Link</span>
                <span className="text-xs font-bold text-emerald-600 flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-ping"></span>
                  Active SQL link
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs font-semibold text-slate-655">HIPAA Compliance</span>
                <span className="text-xs font-bold text-slate-800">Verified</span>
              </div>
            </div>
          </div>

          {/* Active Triage Alert Callout */}
          {latestAssessment && latestAssessment.triage_level && latestAssessment.triage_level <= 2 && (
            <div className="bg-red-50 border border-red-200 text-red-800 p-5 rounded-2xl space-y-2.5">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 bg-red-650 rounded-full animate-ping shrink-0"></span>
                <h4 className="text-xs font-extrabold uppercase tracking-wide">Triage Alert (ESI Level {latestAssessment.triage_level})</h4>
              </div>
              <p className="text-[11px] leading-relaxed font-medium">
                Your last clinical workup indicates serious cardiopulmonary or pulmonary values. EMR has flagged your profile. Please coordinate with doctor1@test.com or visit the nearest ER immediately.
              </p>
            </div>
          )}

          {/* Recent Activity Feed Timeline */}
          <div className="bg-card border border-border rounded-2xl p-5 shadow-xs space-y-4">
            <div className="flex items-center gap-1.5 border-b border-border pb-3">
              <Calendar className="w-4 h-4 text-sky-600 shrink-0" />
              <h3 className="text-sm font-extrabold text-slate-800 uppercase tracking-wide">Recent Portal Activity</h3>
            </div>

            <div className="relative border-l border-slate-200 pl-4 ml-2.5 space-y-4 text-xs">
              <div className="relative">
                <span className="absolute -left-[21.5px] top-1 w-2.5 h-2.5 rounded-full bg-emerald-600 border-2 border-white ring-4 ring-emerald-50"></span>
                <p className="font-bold text-slate-900">Vitals Synchronized</p>
                <p className="text-[10px] text-muted-foreground">Latest heart rate and SpO2 synced with Postgres EMR node.</p>
                <span className="text-[9px] text-slate-400 font-semibold block mt-0.5">Today</span>
              </div>

              <div className="relative">
                <span className="absolute -left-[21.5px] top-1 w-2.5 h-2.5 rounded-full bg-sky-655 border-2 border-white ring-4 ring-sky-50"></span>
                <p className="font-bold text-slate-900">Ensemble ML Pipeline Executed</p>
                <p className="text-[10px] text-muted-foreground">Voting Ensemble ran disease attributions on Case #{latestAssessment?.id || '104'}.</p>
                <span className="text-[9px] text-slate-400 font-semibold block mt-0.5">1 day ago</span>
              </div>

              <div className="relative">
                <span className="absolute -left-[21.5px] top-1 w-2.5 h-2.5 rounded-full bg-slate-350 border-2 border-white"></span>
                <p className="font-bold text-slate-800">Security Session Active</p>
                <p className="text-[10px] text-muted-foreground">JWT Session key validated for Role: patient.</p>
                <span className="text-[9px] text-slate-400 font-semibold block mt-0.5">3 days ago</span>
              </div>
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}
