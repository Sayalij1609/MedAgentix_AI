import React from 'react';
import { Shield, Lock, Activity, ArrowRight, UserCheck, FileText } from 'lucide-react';

export default function LandingPage() {
  return (
    <div className="min-h-[80vh] flex flex-col items-center justify-center bg-slate-50/50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl w-full space-y-12 text-center">
        
        {/* Institutional Branding Block */}
        <div className="space-y-4">
          <div className="inline-flex items-center gap-2 bg-sky-50 border border-sky-200 text-sky-800 px-4 py-1.5 rounded-full text-xs font-semibold uppercase tracking-wider">
            <Shield className="w-3.5 h-3.5" />
            Authorized Clinical Access Only
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-slate-900">
            MedAgentix Clinical Portal
          </h1>
          <p className="text-base md:text-lg text-slate-600 max-w-2xl mx-auto leading-relaxed">
            Institutional Clinical Decision Support System (CDSS) and patient triage gateway. Fully integrated with electronic medical records and ESI prioritization protocols.
          </p>
        </div>

        {/* Primary Action Panel */}
        <div className="bg-card border border-border p-8 rounded-2xl shadow-md max-w-lg mx-auto space-y-6">
          <div className="space-y-2">
            <h2 className="text-lg font-bold text-slate-800">Secure Gatekeeper Authentication</h2>
            <p className="text-xs text-muted-foreground">
              Sign in with your clinical credentials or complete patient self-intake registration.
            </p>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a 
              href="/login" 
              className="flex items-center justify-center gap-2 px-6 py-3 bg-primary text-primary-foreground font-semibold rounded-xl hover:opacity-95 transition shadow-sm text-sm"
            >
              <span>Enter Secure Portal</span>
              <ArrowRight className="w-4 h-4" />
            </a>
            <a 
              href="/register" 
              className="flex items-center justify-center gap-2 px-6 py-3 bg-slate-100 hover:bg-slate-200 font-semibold rounded-xl text-slate-800 transition text-sm border border-border"
            >
              <span>Patient Registration</span>
            </a>
          </div>

          <div className="border-t border-border pt-4 flex justify-center items-center gap-6 text-[10px] text-muted-foreground font-medium">
            <span className="flex items-center gap-1">
              <Lock className="w-3.5 h-3.5 text-sky-600" />
              HIPAA Compliant
            </span>
            <span className="flex items-center gap-1">
              <Shield className="w-3.5 h-3.5 text-sky-600" />
              Secure Data Socket
            </span>
            <span className="flex items-center gap-1">
              <Activity className="w-3.5 h-3.5 text-sky-600" />
              Real-time CDSS
            </span>
          </div>
        </div>

        {/* Informational Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-left">
          <div className="bg-card border border-border p-6 rounded-2xl shadow-sm space-y-3">
            <div className="w-10 h-10 rounded-xl bg-sky-50 flex items-center justify-center text-sky-700">
              <Activity className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-sm text-slate-800">Clinical Suggestion Engine</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Synthesizes multi-model diagnostic outputs, physiological analysis, and ICD-10 mapping algorithms.
            </p>
          </div>

          <div className="bg-card border border-border p-6 rounded-2xl shadow-sm space-y-3">
            <div className="w-10 h-10 rounded-xl bg-sky-50 flex items-center justify-center text-sky-700">
              <UserCheck className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-sm text-slate-800">ESI Triage Workup</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Standardized assessment wizard translating patient vitals and chief complaints into triage urgencies.
            </p>
          </div>

          <div className="bg-card border border-border p-6 rounded-2xl shadow-sm space-y-3">
            <div className="w-10 h-10 rounded-xl bg-sky-50 flex items-center justify-center text-sky-700">
              <FileText className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-sm text-slate-800">Electronic Health Records</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Maintains secure case logs, physician reviews, and downloadable clinical summary documents.
            </p>
          </div>
        </div>

      </div>
    </div>
  );
}
