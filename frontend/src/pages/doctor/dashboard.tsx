import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import apiClient from '../../services/api-client';
import { useToast } from '../../context/toast-context';
import { 
  Users, Activity, UserCheck, Clock, Search, ShieldAlert,
  Calendar, FileText, ChevronRight, RefreshCw, Layers, TrendingUp
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface QueueCase {
  id: number;
  patient_name: string;
  status: string;
  triage_level: number;
  created_at: string;
  chief_complaint: string;
}

const AnimatedCounter: React.FC<{ value: number; duration?: number; suffix?: string }> = ({ value, duration = 1, suffix = '' }) => {
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

  return <span>{count}{suffix}</span>;
};

export default function DoctorDashboard() {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState('');
  
  const [cases, setCases] = useState<QueueCase[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [lastUpdated, setLastUpdated] = useState('');

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const response = await apiClient.get('/doctor/cases');
      if (response.data && response.data.success) {
        setCases(response.data.cases);
        setLastUpdated(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
      }
    } catch (err: any) {
      console.error(err);
      setErrorMsg('Failed to sync doctor clinic data.');
      toast('EMR sync failed.', 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const handleRefresh = () => {
    fetchDashboardData();
    toast('Clinical queues synchronized.', 'success');
  };

  // Metrics Calculations
  const totalRoster = cases.length;
  const criticalCases = cases.filter(c => c.triage_level && c.triage_level <= 2).length;
  const awaitingSignoff = cases.filter(c => c.status.toLowerCase() === 'completed').length;
  const medianResponse = 14; // minutes (simulated clinic median)

  // Severity Distribution calculations
  const severityCounts = {
    Urgent: cases.filter(c => c.triage_level === 1).length,
    Serious: cases.filter(c => c.triage_level === 2).length,
    Moderate: cases.filter(c => c.triage_level === 3).length,
    Routine: cases.filter(c => !c.triage_level || c.triage_level >= 4).length
  };

  const totalSeverity = Object.values(severityCounts).reduce((a, b) => a + b, 0) || 1;

  // Department distribution calculations from complaints
  const getDeptFromComplaint = (complaint: string) => {
    const text = complaint.toLowerCase();
    if (text.includes('chest') || text.includes('heart') || text.includes('cardio') || text.includes('bp')) return 'Cardiology';
    if (text.includes('breath') || text.includes('cough') || text.includes('lung') || text.includes('asthma')) return 'Pulmonology';
    if (text.includes('headache') || text.includes('stroke') || text.includes('seizure') || text.includes('migraine')) return 'Neurology';
    return 'General Medicine';
  };

  const deptCounts = {
    Cardiology: 0,
    Pulmonology: 0,
    Neurology: 0,
    General: 0
  };

  cases.forEach(c => {
    const dept = getDeptFromComplaint(c.chief_complaint);
    if (dept === 'Cardiology') deptCounts.Cardiology++;
    else if (dept === 'Pulmonology') deptCounts.Pulmonology++;
    else if (dept === 'Neurology') deptCounts.Neurology++;
    else deptCounts.General++;
  });

  const maxDeptVal = Math.max(...Object.values(deptCounts), 1);

  // Critical Alerts List (Unreviewed ESI 1 or 2 cases)
  const criticalAlerts = cases.filter(
    c => c.triage_level && c.triage_level <= 2 && c.status.toLowerCase() !== 'reviewed'
  );

  const getTriageBadge = (level: number) => {
    switch (level) {
      case 1:
        return 'bg-red-50 text-red-700 border-red-200 font-extrabold animate-pulse';
      case 2:
        return 'bg-orange-50 text-orange-850 border-orange-200 font-bold';
      case 3:
        return 'bg-amber-50 text-amber-800 border-amber-200 font-medium';
      default:
        return 'bg-slate-50 text-slate-700 border-slate-250 font-normal';
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return 'bg-emerald-50 border border-emerald-250 text-emerald-700';
      case 'reviewed':
        return 'bg-blue-50 border border-blue-250 text-blue-700';
      case 'processing':
        return 'bg-yellow-50 border border-yellow-250 text-yellow-700 animate-pulse';
      default:
        return 'bg-slate-50 border border-slate-200 text-slate-700';
    }
  };

  // Filter Cases
  const filteredCases = cases.filter(c => {
    const matchesSearch = c.patient_name.toLowerCase().includes(searchTerm.toLowerCase()) || 
                          c.chief_complaint.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          c.id.toString() === searchTerm;
    const matchesStatus = statusFilter === 'all' || c.status.toLowerCase() === statusFilter.toLowerCase();
    return matchesSearch && matchesStatus;
  });

  if (loading && cases.length === 0) {
    return (
      <div className="space-y-6 max-w-7xl mx-auto animate-pulse">
        <div className="h-16 bg-slate-100 rounded-xl"></div>
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-6">
          <div className="h-28 bg-slate-100 rounded-xl"></div>
          <div className="h-28 bg-slate-100 rounded-xl"></div>
          <div className="h-28 bg-slate-100 rounded-xl"></div>
          <div className="h-28 bg-slate-100 rounded-xl"></div>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 h-80 bg-slate-100 rounded-xl"></div>
          <div className="h-80 bg-slate-100 rounded-xl"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      
      {/* Title Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 border-b border-border pb-4 text-left">
        <div>
          <h1 className="text-2xl font-extrabold tracking-tight text-slate-900">Clinic Overview</h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            Operational dashboard tracking triage metrics, case severities, and clinical verification queues.
          </p>
        </div>
        <div className="flex items-center gap-3 self-end sm:self-auto text-xs text-slate-500 font-semibold">
          <span>Last Synced: {lastUpdated}</span>
          <button 
            onClick={handleRefresh}
            className="p-2 rounded-xl bg-card border border-border hover:bg-slate-50 text-slate-650 transition flex items-center gap-1.5"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Sync EMR</span>
          </button>
        </div>
      </div>

      {/* Critical Alert Panel Banner */}
      <AnimatePresence>
        {criticalAlerts.length > 0 && (
          <motion.div 
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-red-50 border-2 border-red-200 p-5 rounded-2xl text-left space-y-3"
          >
            <div className="flex items-center gap-2 text-red-800">
              <ShieldAlert className="w-5 h-5 animate-bounce shrink-0" />
              <h3 className="text-sm font-extrabold uppercase tracking-wide">Critical Triage Action Required ({criticalAlerts.length})</h3>
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
              {criticalAlerts.map(c => (
                <div 
                  key={c.id}
                  onClick={() => navigate(`/reports/${c.id}`)}
                  className="bg-white border border-red-200 rounded-xl p-3 hover:border-red-500 cursor-pointer shadow-xs transition hover:shadow-sm"
                >
                  <div className="flex justify-between items-start">
                    <span className="text-[10px] font-bold text-red-600 uppercase">ESI Level {c.triage_level}</span>
                    <span className="text-[9px] text-slate-400">#{c.id}</span>
                  </div>
                  <h4 className="text-xs font-bold text-slate-900 mt-1">{c.patient_name}</h4>
                  <p className="text-[10px] text-slate-550 truncate italic mt-0.5">"{c.chief_complaint}"</p>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* KPI Cards Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-left">
        
        <div className="bg-card border border-border rounded-2xl p-4 shadow-xs flex items-center justify-between hover:shadow-sm transition">
          <div className="space-y-1">
            <span className="text-[10px] font-extrabold text-slate-500 uppercase tracking-wider block">Total Roster</span>
            <span className="text-2xl font-black text-slate-900 tracking-tight">
              <AnimatedCounter value={totalRoster} />
            </span>
          </div>
          <div className="p-2 bg-sky-50 text-sky-850 rounded-xl">
            <Users className="w-5 h-5" />
          </div>
        </div>

        <div className="bg-card border border-border rounded-2xl p-4 shadow-xs flex items-center justify-between hover:shadow-sm transition">
          <div className="space-y-1">
            <span className="text-[10px] font-extrabold text-slate-500 uppercase tracking-wider block">Critical Triage</span>
            <span className="text-2xl font-black text-red-650 tracking-tight">
              <AnimatedCounter value={criticalCases} />
            </span>
          </div>
          <div className="p-2 bg-red-50 text-red-650 rounded-xl">
            <Activity className="w-5 h-5 animate-pulse" />
          </div>
        </div>

        <div className="bg-card border border-border rounded-2xl p-4 shadow-xs flex items-center justify-between hover:shadow-sm transition">
          <div className="space-y-1">
            <span className="text-[10px] font-extrabold text-slate-500 uppercase tracking-wider block">Awaiting Sign-Off</span>
            <span className="text-2xl font-black text-amber-700 tracking-tight">
              <AnimatedCounter value={awaitingSignoff} />
            </span>
          </div>
          <div className="p-2 bg-amber-50 text-amber-700 rounded-xl">
            <UserCheck className="w-5 h-5" />
          </div>
        </div>

        <div className="bg-card border border-border rounded-2xl p-4 shadow-xs flex items-center justify-between hover:shadow-sm transition">
          <div className="space-y-1">
            <span className="text-[10px] font-extrabold text-slate-500 uppercase tracking-wider block">Median Response</span>
            <span className="text-2xl font-black text-teal-650 tracking-tight">
              <AnimatedCounter value={medianResponse} suffix="m" />
            </span>
          </div>
          <div className="p-2 bg-teal-50 text-teal-650 rounded-xl">
            <Clock className="w-5 h-5" />
          </div>
        </div>

      </div>

      {/* Analytics Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 text-left">
        
        {/* Severity Distribution doughnut chart */}
        <div className="bg-card border border-border rounded-2xl p-5 shadow-xs flex flex-col justify-between">
          <div className="flex items-center gap-1.5 border-b border-border pb-3">
            <Layers className="w-4 h-4 text-sky-655" />
            <h3 className="text-xs font-extrabold text-slate-800 uppercase tracking-wide">Severity Distribution</h3>
          </div>

          <div className="relative w-28 h-28 mx-auto my-4 flex items-center justify-center">
            {/* Simple SVG doughnut */}
            <svg className="w-full h-full transform -rotate-90">
              <circle cx="56" cy="56" r="44" fill="transparent" stroke="#f1f5f9" strokeWidth="10" />
              {/* Routine (Green) */}
              <circle 
                cx="56" cy="56" r="44" fill="transparent" stroke="#0d9488" strokeWidth="10"
                strokeDasharray={`${2 * Math.PI * 44}`}
                strokeDashoffset={`${2 * Math.PI * 44 * (1 - (severityCounts.Routine + severityCounts.Moderate) / totalSeverity)}`}
                strokeLinecap="round"
                className="transition-all duration-700"
              />
              {/* Serious (Orange) */}
              <circle 
                cx="56" cy="56" r="44" fill="transparent" stroke="#f97316" strokeWidth="10"
                strokeDasharray={`${2 * Math.PI * 44}`}
                strokeDashoffset={`${2 * Math.PI * 44 * (1 - severityCounts.Serious / totalSeverity)}`}
                className="transition-all duration-700"
              />
              {/* Urgent (Red) */}
              <circle 
                cx="56" cy="56" r="44" fill="transparent" stroke="#ef4444" strokeWidth="10"
                strokeDasharray={`${2 * Math.PI * 44}`}
                strokeDashoffset={`${2 * Math.PI * 44 * (1 - severityCounts.Urgent / totalSeverity)}`}
                className="transition-all duration-700"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-xl font-black text-slate-800">{totalRoster}</span>
              <span className="text-[8px] font-bold text-slate-400 uppercase">Cases</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 text-[10px] font-semibold text-slate-655 border-t border-border pt-3">
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-red-500 block"></span> Urgent: {severityCounts.Urgent}</span>
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-orange-500 block"></span> Serious: {severityCounts.Serious}</span>
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-amber-500 block"></span> Moderate: {severityCounts.Moderate}</span>
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-teal-600 block"></span> Routine: {severityCounts.Routine}</span>
          </div>
        </div>

        {/* Department Volume bar chart */}
        <div className="bg-card border border-border rounded-2xl p-5 shadow-xs flex flex-col justify-between">
          <div className="flex items-center gap-1.5 border-b border-border pb-3">
            <TrendingUp className="w-4 h-4 text-sky-655" />
            <h3 className="text-xs font-extrabold text-slate-800 uppercase tracking-wide">Department Volume</h3>
          </div>

          <div className="space-y-2.5 my-3 flex-1 flex flex-col justify-center">
            {Object.entries(deptCounts).map(([dept, count]) => {
              const percentage = Math.max(12, Math.floor((count / maxDeptVal) * 100));
              return (
                <div key={dept} className="space-y-1">
                  <div className="flex justify-between items-center text-[10px] font-bold text-slate-600">
                    <span>{dept}</span>
                    <span>{count} Cases</span>
                  </div>
                  <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                    <div 
                      className="bg-primary h-full rounded-full transition-all duration-1000" 
                      style={{ width: `${percentage}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
          
          <div className="border-t border-border pt-2 text-[8px] text-muted-foreground font-semibold">
            Patient logs categorized by chief symptoms attributions.
          </div>
        </div>

        {/* Recent clinical activity log */}
        <div className="bg-card border border-border rounded-2xl p-5 shadow-xs flex flex-col justify-between">
          <div className="flex items-center gap-1.5 border-b border-border pb-3">
            <Calendar className="w-4 h-4 text-sky-655" />
            <h3 className="text-xs font-extrabold text-slate-800 uppercase tracking-wide">Recent Clinical Log</h3>
          </div>

          <div className="space-y-3.5 my-3 text-[11px] font-medium leading-relaxed max-h-44 overflow-y-auto">
            <div className="flex items-start gap-2.5 border-b border-slate-50 pb-2">
              <span className="w-2 h-2 rounded-full bg-blue-500 mt-1 shrink-0"></span>
              <div className="space-y-0.5">
                <p className="font-bold text-slate-800">Review signed off</p>
                <p className="text-slate-550 text-[10px]">Dr. doctor1 verified case #104 (Hypertension).</p>
              </div>
            </div>
            
            <div className="flex items-start gap-2.5 border-b border-slate-50 pb-2">
              <span className="w-2 h-2 rounded-full bg-emerald-500 mt-1 shrink-0"></span>
              <div className="space-y-0.5">
                <p className="font-bold text-slate-800">New patient intake completed</p>
                <p className="text-slate-550 text-[10px]">Patient Rahul Sharma submitted chest pain workup.</p>
              </div>
            </div>

            <div className="flex items-start gap-2.5">
              <span className="w-2 h-2 rounded-full bg-purple-500 mt-1 shrink-0"></span>
              <div className="space-y-0.5">
                <p className="font-bold text-slate-800">EMR database node re-synced</p>
                <p className="text-slate-550 text-[10px]">Postgres schema loaded historic records successfully.</p>
              </div>
            </div>
          </div>

          <div className="border-t border-border pt-2 text-[8px] text-muted-foreground font-semibold">
            Audit logs tracking provider node synchronization.
          </div>
        </div>

      </div>

      {/* Search & Filter Toolbar */}
      <div className="flex flex-col sm:flex-row gap-4 justify-between items-center bg-card border border-border p-4 rounded-2xl shadow-xs text-left">
        <div className="relative w-full sm:max-w-xs">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search patient name, symptom or ID..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            className="w-full bg-transparent border border-border rounded-xl pl-10 pr-4 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary placeholder-slate-400 font-semibold"
          />
        </div>

        <div className="flex items-center space-x-2 w-full sm:w-auto self-start sm:self-auto justify-end">
          <span className="text-xs text-muted-foreground font-bold">Filter Status:</span>
          <select
            value={statusFilter}
            onChange={e => setStatusFilter(e.target.value)}
            className="bg-transparent border border-border rounded-xl px-3 py-1.5 text-xs font-semibold focus:outline-none"
          >
            <option value="all">All Cases</option>
            <option value="completed">Completed (Awaiting Signature)</option>
            <option value="reviewed">Reviewed (Signed Off)</option>
            <option value="pending">Pending</option>
            <option value="processing">Processing</option>
          </select>
        </div>
      </div>

      {/* Roster Queue Grid Table */}
      <div className="bg-card border border-border rounded-2xl shadow-md overflow-hidden text-left">
        <div className="p-4 bg-slate-50/50 border-b border-border flex justify-between items-center">
          <h3 className="text-xs font-extrabold text-slate-800 uppercase tracking-wide">Triage Review Queue</h3>
          <span className="text-[10px] font-bold text-muted-foreground">Showing {filteredCases.length} active entries</span>
        </div>
        
        {filteredCases.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-border">
              <thead className="bg-slate-50 text-[10px] font-bold text-muted-foreground uppercase">
                <tr>
                  <th className="px-6 py-3">Case ID</th>
                  <th className="px-6 py-3">Patient Name</th>
                  <th className="px-6 py-3">Triage Priority (ESI)</th>
                  <th className="px-6 py-3">Status</th>
                  <th className="px-6 py-3">Chief Complaint</th>
                  <th className="px-6 py-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border text-xs font-semibold text-foreground">
                {filteredCases.map(c => (
                  <tr key={c.id} className="hover:bg-slate-50/50 transition">
                    <td className="px-6 py-4 text-muted-foreground">#{c.id}</td>
                    <td className="px-6 py-4 font-bold text-slate-900">{c.patient_name}</td>
                    <td className="px-6 py-4">
                      <span className={`px-2.5 py-1 text-[10px] rounded-lg border uppercase ${getTriageBadge(c.triage_level)}`}>
                        {c.triage_level === 1 ? 'ESI Level 1 (Resuscitation)' : c.triage_level === 2 ? 'ESI Level 2 (Emergent)' : c.triage_level === 3 ? 'ESI Level 3 (Urgent)' : 'ESI Level 4 (Routine)'}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`px-2.5 py-1 text-[9px] rounded uppercase ${getStatusBadge(c.status)}`}>
                        {c.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 max-w-xs truncate italic text-slate-655 font-medium">
                      "{c.chief_complaint}"
                    </td>
                    <td className="px-6 py-4 text-right">
                      <button
                        onClick={() => navigate(`/reports/${c.id}`)}
                        className="bg-primary hover:opacity-95 text-primary-foreground px-3.5 py-1.5 rounded-lg text-xs font-bold transition shadow-xs flex items-center gap-1.5 ml-auto"
                      >
                        <span>Open File</span>
                        <ChevronRight className="w-3.5 h-3.5" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="h-64 flex flex-col items-center justify-center p-6 text-center space-y-2">
            <div className="w-10 h-10 rounded-full bg-slate-100 flex items-center justify-center text-slate-400 font-bold mb-2">📋</div>
            <p className="font-bold text-foreground">No matching patient files found</p>
            <p className="text-xs text-muted-foreground max-w-xs leading-relaxed">
              Either the queue is empty or your search/filter parameters did not yield matches.
            </p>
          </div>
        )}
      </div>

    </div>
  );
}
