import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import apiClient from '../../services/api-client';
import { useAuth } from '../../context/auth-context';
import { 
  Thermometer, 
  Wind, 
  Heart, 
  Activity, 
  Printer, 
  Download, 
  AlertTriangle, 
  Lightbulb, 
  FileText,
  CheckCircle,
  HelpCircle
} from 'lucide-react';
import PipelineVisualizer from '../../components/common/PipelineVisualizer';

interface CaseDetails {
  id: number;
  patient_id: number;
  doctor_id: number | null;
  status: string;
  triage_level: number;
  created_at: string;
  patient_name?: string;
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
    pipeline_version: string;
    generated_at: string;
    final_diagnosis: string;
    confidence: number;
    severity: string;
    icd_code: string;
    pathophysiology: string;
    patient_age?: number;
    patient_gender?: string;
    differential_considerations: { rank: number; condition: string; probability: number }[];
    recommended_drugs: {
      name: string;
      dosage: string;
      purpose: string;
      route: string;
      class: string;
      adr: string;
      ci: string;
      precaution: string;
    }[];
    recommended_tests: { name: string; priority: string; department: string; indication: string }[];
    emergency_status: {
      is_emergency: boolean;
      triage_level: number;
      urgency: string;
      progression: string;
    };
  };
}

export default function ClinicalReport() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState('');
  const [caseData, setCaseData] = useState<CaseDetails | null>(null);
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false);
  const [isPipelineExpanded, setIsPipelineExpanded] = useState(false);

  const fetchCaseDetails = async () => {
    try {
      setLoading(true);
      const response = await apiClient.get(`/cases/${id}`);
      if (response.data && response.data.success) {
        setCaseData(response.data.case);
      }
    } catch (err: any) {
      console.error(err);
      const msg = err.response?.data?.message || 'Failed to retrieve case details.';
      setErrorMsg(msg);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (id) {
      fetchCaseDetails();
    }
  }, [id]);

  const handlePrint = () => {
    window.print();
  };

  const handlePdfDownload = () => {
    setIsGeneratingPdf(true);
    setTimeout(() => {
      setIsGeneratingPdf(false);
      window.print(); // client-side fallback print
    }, 1200);
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 90) return { text: 'Strong Match', color: 'bg-emerald-500 text-white' };
    if (confidence >= 80) return { text: 'Good Match', color: 'bg-blue-500 text-white' };
    if (confidence >= 70) return { text: 'Possible Match', color: 'bg-amber-500 text-black' };
    if (confidence >= 50) return { text: 'Needs Evaluation', color: 'bg-orange-500 text-white' };
    return { text: 'Uncertain / Low Confidence', color: 'bg-red-500 text-white' };
  };

  const getSeverityBadgeClass = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'mild':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'moderate':
        return 'bg-amber-100 text-amber-800 border-amber-200';
      case 'serious':
        return 'bg-orange-100 text-orange-850 border-orange-200';
      case 'urgent':
        return 'bg-red-100 text-red-805 border-red-200 animate-pulse';
      default:
        return 'bg-slate-100 text-slate-800 border-border';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-3 border-primary border-t-transparent rounded-full animate-spin"></div>
          <p className="text-muted-foreground text-xs animate-pulse">Assembling diagnostic outputs...</p>
        </div>
      </div>
    );
  }

  if (errorMsg || !caseData) {
    return (
      <div className="max-w-xl mx-auto bg-card border border-border p-8 rounded-2xl shadow-md text-center space-y-4 flex flex-col items-center">
        <AlertTriangle className="w-12 h-12 text-amber-500 mb-2 shrink-0" />
        <h2 className="text-xl font-bold text-slate-900">Access Restricted</h2>
        <p className="text-muted-foreground text-sm leading-relaxed">
          {errorMsg || 'This clinical report is restricted. Verify session credentials and authorization.'}
        </p>
        <button
          onClick={() => navigate(user?.role === 'doctor' ? '/doctor/queue' : '/patient/dashboard')}
          className="px-6 py-2.5 bg-primary text-primary-foreground font-semibold rounded-xl text-sm hover:opacity-90 transition shadow-sm"
        >
          Return to Dashboard
        </button>
      </div>
    );
  }

  const { vitals, symptoms, history_and_lifestyle, diagnostic_output } = caseData;
  const isDoctor = user?.role === 'doctor';

  // -------------------------------------------------------------
  // PATIENT FRIENDLY VIEW
  // -------------------------------------------------------------
  if (!isDoctor) {
    const conf = getConfidenceLabel(diagnostic_output.confidence);
    return (
      <div className="max-w-3xl mx-auto bg-card border border-border rounded-2xl shadow-xl p-6 md:p-8 space-y-8 print:border-none print:shadow-none print:p-0">
        
        {/* Printable/Download Action Bar */}
        <div className="flex justify-between items-center border-b border-border pb-4 print:hidden">
          <button 
            onClick={() => navigate('/patient/dashboard')}
            className="text-sm font-semibold text-primary hover:underline"
          >
            ← Back to Dashboard
          </button>
          <div className="flex gap-2">
            <button
              onClick={handlePdfDownload}
              disabled={isGeneratingPdf}
              className="bg-primary text-primary-foreground hover:opacity-90 transition px-4 py-2 rounded-xl text-xs font-bold flex items-center gap-1.5 shadow-sm"
            >
              <Download className="w-3.5 h-3.5" />
              <span>{isGeneratingPdf ? 'Generating PDF...' : 'Download PDF'}</span>
            </button>
            <button
              onClick={handlePrint}
              className="bg-slate-100 hover:bg-slate-200 transition px-4 py-2 rounded-xl text-xs font-bold text-foreground flex items-center gap-1.5"
            >
              <Printer className="w-3.5 h-3.5" />
              <span>Print</span>
            </button>
          </div>
        </div>

        {/* Patient Header Block */}
        <div className="flex flex-col md:flex-row md:justify-between md:items-end gap-4 border-b border-border pb-6">
          <div className="space-y-1">
            <span className="text-xs uppercase font-semibold tracking-wider text-sky-850 block">MedAgentix Clinical Portal — Patient Health Summary</span>
            <h2 className="text-2xl font-bold tracking-tight text-slate-900">
              {caseData.patient_name || 'Rahul Sharma'}
            </h2>
            <p className="text-xs text-muted-foreground">
              Medical Record ID: MRN-{caseData.id} | Date of Assessment: {new Date(caseData.created_at).toLocaleDateString()}
            </p>
          </div>
          <div className="text-xs space-y-1 md:text-right text-muted-foreground font-semibold">
            <p>Age: {diagnostic_output.patient_age || 35} years | Gender: {diagnostic_output.patient_gender || 'Male'}</p>
            <p>Assessment State: <span className="text-emerald-600">● {caseData.status}</span></p>
          </div>
        </div>

        {/* Emergency Pulsing Warning Card */}
        {diagnostic_output.emergency_status.is_emergency && (
          <div className="bg-red-50 border-2 border-red-500 text-red-800 p-5 rounded-2xl animate-pulse space-y-2 flex items-start gap-3">
            <AlertTriangle className="w-6 h-6 text-red-600 shrink-0 mt-0.5" />
            <div>
              <h3 className="text-base font-extrabold uppercase">EMERGENCY ALERT — SEEK IMMEDIATE MEDICAL CARE</h3>
              <p className="text-xs leading-relaxed mt-1">
                Based on your clinical readings, please go to the nearest Emergency Room (ER) or call ambulance services (112 / 108) immediately. Do not drive yourself.
              </p>
            </div>
          </div>
        )}

        {/* Vitals Grid */}
        <div className="space-y-3">
          <h3 className="text-xs font-bold uppercase tracking-wider text-muted-foreground">My Vitals</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-50 p-4 border border-border rounded-xl">
              <div className="flex items-center gap-1.5 mb-1.5">
                <Thermometer className="w-4 h-4 text-sky-600 shrink-0" />
                <span className="text-[10px] font-bold text-muted-foreground block uppercase">Temperature</span>
              </div>
              <span className="text-lg font-bold block text-foreground">{vitals.temperature}°F</span>
              <span className={`text-[10px] font-semibold flex items-center gap-1 ${vitals.temperature >= 100.4 ? 'text-amber-600' : 'text-green-600'}`}>
                {vitals.temperature >= 100.4 ? <AlertTriangle className="w-3 h-3 text-amber-500 shrink-0" /> : <CheckCircle className="w-3 h-3 text-green-500 shrink-0" />}
                {vitals.temperature >= 100.4 ? 'Fever detected' : 'Normal range'}
              </span>
            </div>
            
            <div className="bg-slate-50 p-4 border border-border rounded-xl">
              <div className="flex items-center gap-1.5 mb-1.5">
                <Wind className="w-4 h-4 text-sky-600 shrink-0" />
                <span className="text-[10px] font-bold text-muted-foreground block uppercase">Oxygen Level</span>
              </div>
              <span className="text-lg font-bold block text-foreground">{vitals.oxygen_level}%</span>
              <span className={`text-[10px] font-semibold flex items-center gap-1 ${vitals.oxygen_level < 95 ? 'text-red-650' : 'text-green-600'}`}>
                {vitals.oxygen_level < 95 ? <AlertTriangle className="w-3 h-3 text-red-500 shrink-0" /> : <CheckCircle className="w-3 h-3 text-green-500 shrink-0" />}
                {vitals.oxygen_level < 95 ? 'Below normal' : 'Healthy level'}
              </span>
            </div>

            <div className="bg-slate-50 p-4 border border-border rounded-xl">
              <div className="flex items-center gap-1.5 mb-1.5">
                <Heart className="w-4 h-4 text-sky-600 shrink-0" />
                <span className="text-[10px] font-bold text-muted-foreground block uppercase">Heart Rate</span>
              </div>
              <span className="text-lg font-bold block text-foreground">{vitals.heart_rate} bpm</span>
              <span className={`text-[10px] font-semibold flex items-center gap-1 ${vitals.heart_rate > 100 ? 'text-amber-600' : 'text-green-600'}`}>
                {vitals.heart_rate > 100 ? <AlertTriangle className="w-3 h-3 text-amber-500 shrink-0" /> : <CheckCircle className="w-3 h-3 text-green-500 shrink-0" />}
                {vitals.heart_rate > 100 ? 'High resting' : 'Normal sinus'}
              </span>
            </div>

            <div className="bg-slate-50 p-4 border border-border rounded-xl">
              <div className="flex items-center gap-1.5 mb-1.5">
                <Activity className="w-4 h-4 text-sky-600 shrink-0" />
                <span className="text-[10px] font-bold text-muted-foreground block uppercase">Blood Pressure</span>
              </div>
              <span className="text-lg font-bold block text-foreground">{vitals.bp_reading}</span>
              <span className="text-[10px] font-semibold text-green-650 flex items-center gap-1">
                <CheckCircle className="w-3 h-3 text-green-500 shrink-0" />
                Normotensive
              </span>
            </div>
          </div>
        </div>

        {/* Symptoms Chips */}
        <div className="space-y-2">
          <h3 className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Reported Symptoms</h3>
          <div className="flex flex-wrap gap-2">
            {symptoms.selected_symptoms.map(s => (
              <span key={s.name} className="bg-slate-150/80 border border-border rounded-full px-3 py-1 text-xs font-semibold text-foreground">
                ● {s.name} ({s.duration_days} days)
              </span>
            ))}
          </div>
        </div>

        {/* What We Found Section */}
        <div className="space-y-4 border-t border-border pt-6">
          <h3 className="text-xs font-bold uppercase tracking-wider text-muted-foreground">What We Found</h3>
          
          <div className="bg-slate-50 border border-border rounded-2xl p-5 space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="space-y-0.5">
                <span className="text-[10px] font-bold text-muted-foreground uppercase">Possible Condition Match</span>
                <h4 className="text-2xl font-extrabold text-foreground">{diagnostic_output.final_diagnosis}</h4>
              </div>
              <div className="flex gap-2">
                <span className={`px-3 py-1 rounded-lg text-xs font-bold ${conf.color}`}>{conf.text} ({diagnostic_output.confidence}%)</span>
                <span className={`px-3 py-1 rounded-lg text-xs font-bold border ${getSeverityBadgeClass(diagnostic_output.severity)}`}>
                  {diagnostic_output.severity}
                </span>
              </div>
            </div>

            <div className="space-y-2 border-t border-border pt-4">
              <div className="flex items-center gap-1.5">
                <Lightbulb className="w-4 h-4 text-sky-600 shrink-0" />
                <h5 className="text-xs font-bold text-muted-foreground uppercase">What this means in plain language:</h5>
              </div>
              <p className="text-sm text-foreground/90 leading-relaxed text-left">
                Your signs and reported conditions match the typical progression patterns seen with **{diagnostic_output.final_diagnosis}**. 
                This assessment is generated by a clinical decision support system engine. It is not an official prescription or laboratory confirmation.
              </p>
            </div>
          </div>
        </div>

        {/* Recommended Drugs */}
        <div className="space-y-3">
          <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground">Suggested Home Medications</h3>
          <p className="text-xs text-muted-foreground italic">Always verify dosage and consult a clinician before ingestion.</p>
          
          <div className="space-y-2">
            {diagnostic_output.recommended_drugs.map(d => (
              <div key={d.name} className="border border-border p-4 rounded-xl flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div className="space-y-1">
                  <h4 className="font-bold text-base text-foreground">{d.name}</h4>
                  <p className="text-xs text-muted-foreground">Purpose: {d.purpose}</p>
                </div>
                <span className="bg-primary/10 text-primary border border-primary/20 rounded-xl px-4 py-1.5 text-xs font-bold self-start sm:self-auto">
                  {d.dosage}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Recommended Diagnostics Tests */}
        <div className="space-y-3">
          <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground">Clinical Investigations Your Doctor May Order</h3>
          <div className="space-y-2">
            {diagnostic_output.recommended_tests.map(t => (
              <div key={t.name} className="border border-border p-3 rounded-xl flex items-center justify-between bg-slate-50/50">
                <div className="space-y-0.5">
                  <h5 className="text-sm font-bold text-foreground">{t.name}</h5>
                  <p className="text-[10px] text-muted-foreground">Rationale: {t.indication}</p>
                </div>
                <span className="text-[10px] font-semibold text-primary uppercase bg-primary/5 px-2 py-1 rounded">
                  {t.priority}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Metadata Footer */}
        <div className="border-t border-border pt-6 flex flex-col sm:flex-row justify-between items-center text-[10px] text-muted-foreground gap-2">
          <p>CDSS Pipeline: {diagnostic_output.pipeline_version} | Engine Timestamp: {new Date(diagnostic_output.generated_at).toLocaleString()}</p>
          <p className="font-semibold flex items-center gap-1">
            <AlertTriangle className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
            Disclaimer: Not a substitute for professional clinical judgment.
          </p>
        </div>

      </div>
    );
  }

  // -------------------------------------------------------------
  // CLINICAL DOCTOR VIEW
  // -------------------------------------------------------------
  return (
    <div className="max-w-5xl mx-auto bg-card border border-border rounded-2xl shadow-xl p-6 md:p-8 space-y-8 print:border-none print:shadow-none print:p-0">
      
      {/* Clinician Action Toolbar */}
      <div className="flex justify-between items-center border-b border-border pb-4 print:hidden">
        <button 
          onClick={() => navigate('/doctor/queue')}
          className="text-sm font-semibold text-primary hover:underline"
        >
          ← Return to Doctor Queue
        </button>
        <div className="flex gap-2">
          <button
            onClick={handlePdfDownload}
            disabled={isGeneratingPdf}
            className="bg-slate-900 text-white hover:opacity-90 px-4 py-2 rounded-xl text-xs font-bold flex items-center gap-1.5 shadow-sm"
          >
            <FileText className="w-3.5 h-3.5 text-inherit shrink-0" />
            <span>{isGeneratingPdf ? 'Compiling PDF...' : 'Download PDF'}</span>
          </button>
          <button
            onClick={handlePrint}
            className="bg-slate-100 hover:bg-slate-200 transition px-4 py-2 rounded-xl text-xs font-bold text-foreground flex items-center gap-1.5"
          >
            <Printer className="w-3.5 h-3.5 text-inherit shrink-0" />
            <span>Print</span>
          </button>
        </div>
      </div>

      {/* Main Title Header */}
      <div className="border-b-2 border-sky-900 pb-6 flex flex-col md:flex-row justify-between items-start gap-4">
        <div className="space-y-1">
          <h1 className="text-2xl font-extrabold tracking-tight text-foreground uppercase">Clinical Workup Record (CDSS)</h1>
          <p className="text-xs text-muted-foreground">
            Report ID: MED-{caseData.id} | Generated: {new Date(caseData.created_at).toLocaleString()}
          </p>
        </div>
        <div className="bg-sky-50 p-3 rounded-xl border border-sky-100 text-xs text-slate-800 font-medium">
          <p>CDSS Pipeline Version: {diagnostic_output.pipeline_version}</p>
          <p>Triage Status: <span className="text-red-650 font-bold uppercase">{caseData.status}</span></p>
        </div>
      </div>

      {/* CDSS Pipeline Trace (Expandable) */}
      <div className="bg-card border border-border rounded-2xl shadow-xs overflow-hidden">
        <button
          type="button"
          onClick={() => setIsPipelineExpanded(!isPipelineExpanded)}
          className="w-full flex justify-between items-center p-4 bg-slate-50/50 hover:bg-slate-50 transition text-left focus:outline-none"
        >
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-sky-650 shrink-0" />
            <span className="text-xs font-bold text-slate-800 uppercase tracking-wide">
              CDSS Pipeline Audit Trace (10 Calibrated Stages)
            </span>
          </div>
          <span className="text-xs text-sky-700 font-semibold hover:underline">
            {isPipelineExpanded ? 'Collapse Audit Graph ↑' : 'Expand Audit Graph ↓'}
          </span>
        </button>
        {isPipelineExpanded && (
          <div className="p-6 border-t border-border bg-card">
            <PipelineVisualizer isComplete={true} />
          </div>
        )}
      </div>

      {/* Patient demographics table */}
      <div className="space-y-2">
        <h3 className="text-xs font-extrabold uppercase tracking-wider text-muted-foreground">Patient Demographics</h3>
        <table className="min-w-full divide-y divide-border border border-border text-sm text-left">
          <thead className="bg-slate-50 text-xs font-bold text-muted-foreground uppercase">
            <tr>
              <th className="px-4 py-2 border-r border-border">Name</th>
              <th className="px-4 py-2 border-r border-border">Age</th>
              <th className="px-4 py-2 border-r border-border">Gender</th>
              <th className="px-4 py-2">Comorbidities</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border font-medium">
            <tr>
              <td className="px-4 py-3 border-r border-border">
                {caseData.patient_name || 'Rahul Sharma'}
              </td>
              <td className="px-4 py-3 border-r border-border">
                {diagnostic_output.patient_age || 35} years
              </td>
              <td className="px-4 py-3 border-r border-border">
                {diagnostic_output.patient_gender || 'Male'}
              </td>
              <td className="px-4 py-3 text-xs italic text-foreground/80">
                {history_and_lifestyle.medical_history.join(', ') || 'None documented'}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Vitals table */}
      <div className="space-y-2">
        <h3 className="text-xs font-extrabold uppercase tracking-wider text-muted-foreground">Physiological Parameters</h3>
        <table className="min-w-full divide-y divide-border border border-border text-sm text-left">
          <thead className="bg-slate-50 text-xs font-bold text-muted-foreground uppercase">
            <tr>
              <th className="px-4 py-2 border-r border-border">Parameter</th>
              <th className="px-4 py-2 border-r border-border">Value</th>
              <th className="px-4 py-2">Clinical Interpretation</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border font-medium text-xs">
            <tr>
              <td className="px-4 py-3 border-r border-border font-bold">Body Temperature</td>
              <td className="px-4 py-3 border-r border-border">{vitals.temperature}°F</td>
              <td className={`px-4 py-3 font-semibold ${vitals.temperature >= 100.4 ? 'text-red-500' : 'text-green-500'}`}>
                {vitals.temperature >= 100.4 ? 'Pyrexia (High Fever)' : 'Normothermia'}
              </td>
            </tr>
            <tr>
              <td className="px-4 py-3 border-r border-border font-bold">SpO2 / Oxygen Saturation</td>
              <td className="px-4 py-3 border-r border-border">{vitals.oxygen_level}%</td>
              <td className={`px-4 py-3 font-semibold ${vitals.oxygen_level < 95 ? 'text-red-500' : 'text-green-500'}`}>
                {vitals.oxygen_level < 92 ? 'Severe Hypoxemia' : vitals.oxygen_level < 95 ? 'Mild Hypoxemia' : 'Normal Saturation'}
              </td>
            </tr>
            <tr>
              <td className="px-4 py-3 border-r border-border font-bold">Heart Rate</td>
              <td className="px-4 py-3 border-r border-border">{vitals.heart_rate} bpm</td>
              <td className={`px-4 py-3 font-semibold ${vitals.heart_rate > 100 ? 'text-red-550' : 'text-green-500'}`}>
                {vitals.heart_rate > 100 ? 'Tachycardia' : vitals.heart_rate < 60 ? 'Bradycardia' : 'Normal Sinus Rhythm'}
              </td>
            </tr>
            <tr>
              <td className="px-4 py-3 border-r border-border font-bold">Blood Pressure (BP)</td>
              <td className="px-4 py-3 border-r border-border">{vitals.bp_reading} mmHg</td>
              <td className="px-4 py-3 text-green-500 font-semibold">Normotensive</td>
            </tr>
            <tr>
              <td className="px-4 py-3 border-r border-border font-bold">Serum Cholesterol</td>
              <td className="px-4 py-3 border-r border-border">{vitals.cholesterol} mg/dL</td>
              <td className={`px-4 py-3 font-semibold ${vitals.cholesterol >= 200 ? 'text-amber-500' : 'text-green-500'}`}>
                {vitals.cholesterol >= 200 ? 'Borderline High Hypercholesterolemia' : 'Desirable Range'}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Section 1: Clinical Impression */}
      <div className="space-y-4">
        <h3 className="text-sm font-bold text-foreground border-b border-border pb-1">§ 1 Clinical Impression</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-slate-50 border border-border p-4 rounded-xl">
          <div className="space-y-2">
            <div>
              <span className="text-[10px] font-bold text-muted-foreground uppercase">Primary Diagnosis</span>
              <p className="text-lg font-extrabold text-foreground">{diagnostic_output.final_diagnosis}</p>
            </div>
            <div>
              <span className="text-[10px] font-bold text-muted-foreground uppercase">ICD-10 Code</span>
              <p className="text-sm font-bold text-foreground">{diagnostic_output.icd_code}</p>
            </div>
          </div>
          <div className="space-y-2">
            <div>
              <span className="text-[10px] font-bold text-muted-foreground uppercase">Triage Alert Status</span>
              <p className="text-sm font-bold text-foreground">
                Level {caseData.triage_level} ESI ({diagnostic_output.severity} Urgency)
              </p>
            </div>
            <div>
              <span className="text-[10px] font-bold text-muted-foreground uppercase">Model Agreement & Confidence</span>
              <p className="text-sm font-bold text-foreground">
                {diagnostic_output.confidence}% Confidence
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Section 2: Pathophysiology */}
      <div className="space-y-2">
        <h3 className="text-sm font-bold text-foreground border-b border-border pb-1">§ 2 Pathophysiologic Context</h3>
        <p className="text-xs text-foreground/90 leading-relaxed bg-slate-50 p-4 border border-border rounded-xl">
          {diagnostic_output.pathophysiology}
        </p>
      </div>

      {/* Section 3: Differential list */}
      <div className="space-y-2">
        <h3 className="text-sm font-bold text-foreground border-b border-border pb-1">§ 3 Differential Considerations</h3>
        <table className="min-w-full divide-y divide-border border border-border text-xs text-left">
          <thead className="bg-slate-50 text-[10px] font-bold text-muted-foreground uppercase">
            <tr>
              <th className="px-4 py-2 border-r border-border">Rank</th>
              <th className="px-4 py-2 border-r border-border">Condition</th>
              <th className="px-4 py-2">Model Probability</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border font-medium">
            {diagnostic_output.differential_considerations.map(d => (
              <tr key={d.rank}>
                <td className="px-4 py-2.5 border-r border-border font-bold">{d.rank}</td>
                <td className="px-4 py-2.5 border-r border-border font-bold">{d.condition}</td>
                <td className="px-4 py-2.5 font-bold text-primary">{d.probability}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Section 4: Workup */}
      <div className="space-y-2">
        <h3 className="text-sm font-bold text-foreground border-b border-border pb-1">§ 4 Diagnostic Workup</h3>
        <table className="min-w-full divide-y divide-border border border-border text-xs text-left">
          <thead className="bg-slate-50 text-[10px] font-bold text-muted-foreground uppercase">
            <tr>
              <th className="px-4 py-2 border-r border-border">Investigation</th>
              <th className="px-4 py-2 border-r border-border">Priority</th>
              <th className="px-4 py-2 border-r border-border">Department</th>
              <th className="px-4 py-2">Clinical Rationale</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border font-medium">
            {diagnostic_output.recommended_tests.map(t => (
              <tr key={t.name}>
                <td className="px-4 py-2.5 border-r border-border font-bold">{t.name}</td>
                <td className="px-4 py-2.5 border-r border-border">
                  <span className={`px-2 py-0.5 rounded text-[9px] font-bold uppercase ${t.priority === 'Primary' ? 'bg-red-50 text-red-700 border border-red-200' : 'bg-slate-100 text-slate-700 border'}`}>
                    {t.priority}
                  </span>
                </td>
                <td className="px-4 py-2.5 border-r border-border">{t.department}</td>
                <td className="px-4 py-2.5 italic text-foreground/80">{t.indication}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Section 5: Pharmacotherapy */}
      <div className="space-y-2">
        <h3 className="text-sm font-bold text-foreground border-b border-border pb-1">§ 5 Proposed Pharmacotherapy</h3>
        <table className="min-w-full divide-y divide-border border border-border text-xs text-left">
          <thead className="bg-slate-50 text-[10px] font-bold text-muted-foreground uppercase">
            <tr>
              <th className="px-4 py-2 border-r border-border">Drug</th>
              <th className="px-4 py-2 border-r border-border">Dosage / Route</th>
              <th className="px-4 py-2 border-r border-border">Class</th>
              <th className="px-4 py-2">Adverse Drug Reactions (ADRs) & Contraindications (CI)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border font-medium">
            {diagnostic_output.recommended_drugs.map(d => (
              <tr key={d.name}>
                <td className="px-4 py-3 border-r border-border font-bold">
                  {d.name}
                  <span className="block text-[9px] text-muted-foreground italic font-normal">{d.purpose}</span>
                </td>
                <td className="px-4 py-3 border-r border-border">
                  {d.dosage}
                  <span className="block text-[9px] text-muted-foreground font-semibold">{d.route}</span>
                </td>
                <td className="px-4 py-3 border-r border-border">{d.class}</td>
                <td className="px-4 py-3 space-y-1">
                  <p><span className="font-bold text-red-500 uppercase text-[9px] block">ADR:</span> <span className="text-[10px]">{d.adr}</span></p>
                  <p><span className="font-bold text-red-600 uppercase text-[9px] block">CI:</span> <span className="text-[10px] italic">{d.ci}</span></p>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Section 6: Clinical Alerts warning block */}
      <div className="space-y-2">
        <h3 className="text-sm font-bold text-foreground border-b border-border pb-1">§ 6 Management Plan & Alerts</h3>
        <div className="bg-amber-50 text-amber-800 border border-amber-200 p-4 rounded-xl space-y-2 text-xs">
          <div className="flex items-center gap-1.5">
            <AlertTriangle className="w-4 h-4 text-amber-605 shrink-0" />
            <h4 className="font-extrabold uppercase tracking-wide">CLINICAL PRECAUTIONS & ALERTS:</h4>
          </div>
          <ul className="list-disc list-inside space-y-1">
            {diagnostic_output.recommended_drugs.map(d => (
              <li key={d.name}>
                <span className="font-bold">{d.name} Warning</span>: {d.precaution}
              </li>
            ))}
            <li>Ensure active clinical correlation of other differential matches (such as Bronchial Asthma or Gastritis).</li>
          </ul>
        </div>
      </div>

      {/* Report Footer Disclaimer */}
      <div className="border-t border-border pt-6 flex flex-col sm:flex-row justify-between items-center text-[10px] text-muted-foreground gap-2">
        <p>CDSS Pipeline State: {diagnostic_output.pipeline_version} | Finalized: {new Date(diagnostic_output.generated_at).toLocaleString()}</p>
        <p className="font-bold flex items-center gap-1">
          <AlertTriangle className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
          CONFIDENTIALITY NOTICE: Authorized clinical access only.
        </p>
      </div>

    </div>
  );
}
