import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import apiClient from '../../services/api-client';
import { Activity, Clock, UserCheck, Search, Users } from 'lucide-react';

interface QueueCase {
  id: number;
  patient_name: string;
  status: string;
  triage_level: number;
  created_at: string;
  chief_complaint: string;
}

export default function DoctorQueue() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState('');
  const [cases, setCases] = useState<QueueCase[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');

  const totalRoster = cases.length;
  const criticalCases = cases.filter(c => c.triage_level && c.triage_level <= 2).length;
  const awaitingSignoff = cases.filter(c => c.status.toLowerCase() === 'completed').length;

  const fetchQueue = async () => {
    try {
      setLoading(true);
      const response = await apiClient.get('/doctor/cases');
      if (response.data && response.data.success) {
        setCases(response.data.cases);
      }
    } catch (err: any) {
      console.error(err);
      setErrorMsg('Failed to fetch patient triage queue.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchQueue();
  }, []);

  const getTriageBadge = (level: number) => {
    switch (level) {
      case 1:
        return 'bg-red-50 text-red-800 border-red-200 font-extrabold animate-pulse';
      case 2:
        return 'bg-orange-50 text-orange-850 border-orange-200 font-bold';
      case 3:
        return 'bg-amber-50 text-amber-800 border-amber-200 font-medium';
      default:
        return 'bg-slate-50 text-slate-700 border-slate-200 font-normal';
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return 'bg-emerald-100 text-emerald-800';
      case 'reviewed':
        return 'bg-blue-100 text-blue-800';
      case 'processing':
        return 'bg-yellow-100 text-yellow-800 animate-pulse';
      default:
        return 'bg-slate-150 text-slate-700';
    }
  };

  const filteredCases = cases.filter(c => {
    const matchesSearch = c.patient_name.toLowerCase().includes(searchTerm.toLowerCase()) || 
                          c.chief_complaint.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          c.id.toString() === searchTerm;
    const matchesStatus = statusFilter === 'all' || c.status.toLowerCase() === statusFilter.toLowerCase();
    return matchesSearch && matchesStatus;
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-3 border-primary border-t-transparent rounded-full animate-spin"></div>
          <p className="text-muted-foreground text-xs animate-pulse">Loading active triage queues...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      
      {/* Title block */}
      <div className="text-left">
        <h1 className="text-3xl font-extrabold tracking-tight text-foreground">Patient Triage Queue</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Roster grid containing unassigned and completed consultation requests.
        </p>
      </div>

      {/* Clinician Roster Statistics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 text-left">
        <div className="bg-card border border-border rounded-2xl p-5 shadow-sm flex items-center justify-between">
          <div className="space-y-1">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">Total Roster</span>
            <span className="text-3xl font-extrabold text-slate-900">{totalRoster}</span>
          </div>
          <div className="p-3 rounded-xl bg-sky-50 text-sky-800">
            <Users className="w-5 h-5" />
          </div>
        </div>

        <div className="bg-card border border-border rounded-2xl p-5 shadow-sm flex items-center justify-between">
          <div className="space-y-1">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">Critical Triage</span>
            <span className="text-3xl font-extrabold text-red-650">{criticalCases}</span>
          </div>
          <div className="p-3 rounded-xl bg-red-50 text-red-600">
            <Activity className="w-5 h-5" />
          </div>
        </div>

        <div className="bg-card border border-border rounded-2xl p-5 shadow-sm flex items-center justify-between">
          <div className="space-y-1">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">Awaiting Sign-Off</span>
            <span className="text-3xl font-extrabold text-sky-700">{awaitingSignoff}</span>
          </div>
          <div className="p-3 rounded-xl bg-amber-50 text-amber-700">
            <UserCheck className="w-5 h-5" />
          </div>
        </div>
      </div>

      {errorMsg && (
        <div className="bg-red-50 text-red-600 border border-red-200 p-4 rounded-xl text-sm font-medium text-left">
          {errorMsg}
        </div>
      )}

      {/* Filter Toolbar */}
      <div className="flex flex-col sm:flex-row gap-4 justify-between items-center bg-card border border-border p-4 rounded-2xl shadow-xs">
        <div className="relative w-full sm:max-w-xs text-left">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search patient name, symptom or ID..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            className="w-full bg-transparent border border-border rounded-xl pl-10 pr-4 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary placeholder-slate-400"
          />
        </div>

        <div className="flex items-center space-x-2 w-full sm:w-auto self-start sm:self-auto justify-end">
          <span className="text-xs text-muted-foreground font-semibold">Filter status:</span>
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

      {/* Roster Grid Table */}
      <div className="bg-card border border-border rounded-2xl shadow-md overflow-hidden">
        {filteredCases.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-border text-left">
              <thead className="bg-slate-50 text-xs font-bold text-muted-foreground uppercase">
                <tr>
                  <th className="px-6 py-3">Case ID</th>
                  <th className="px-6 py-3">Patient Name</th>
                  <th className="px-6 py-3">Urgency (ESI)</th>
                  <th className="px-6 py-3">Status</th>
                  <th className="px-6 py-3">Chief Complaint</th>
                  <th className="px-6 py-3">Submitted</th>
                  <th className="px-6 py-3">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border text-sm font-medium">
                {filteredCases.map(c => (
                  <tr key={c.id} className="hover:bg-slate-50/50 transition">
                    <td className="px-6 py-4 font-bold text-muted-foreground">#{c.id}</td>
                    <td className="px-6 py-4 font-bold text-foreground">{c.patient_name}</td>
                    <td className="px-6 py-4">
                      <span className={`px-2.5 py-1 text-xs rounded-lg border uppercase ${getTriageBadge(c.triage_level)}`}>
                        {c.triage_level === 1 ? 'ESI Level 1 (Resuscitation)' : c.triage_level === 2 ? 'ESI Level 2 (Emergent)' : c.triage_level === 3 ? 'ESI Level 3 (Urgent)' : 'ESI Level 4 (Routine)'}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`px-2 py-0.5 text-[10px] font-bold rounded uppercase ${getStatusBadge(c.status)}`}>
                        {c.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 max-w-xs truncate italic text-foreground/80">
                      "{c.chief_complaint}"
                    </td>
                    <td className="px-6 py-4 text-xs text-muted-foreground">
                      {new Date(c.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4">
                      <button
                        onClick={() => navigate(`/reports/${c.id}`)}
                        className="bg-primary hover:opacity-90 text-primary-foreground px-3.5 py-1.5 rounded-lg text-xs font-bold transition shadow-xs"
                      >
                        Open File
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
            <p className="text-xs text-muted-foreground max-w-xs">
              Either the queue is empty or your search/filter parameters did not yield matches.
            </p>
          </div>
        )}
      </div>

    </div>
  );
}
