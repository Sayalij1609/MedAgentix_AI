import React, { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import {
  Upload, FileText, Image, X, CheckCircle, AlertCircle,
  Loader2, FileSearch, Brain, ChevronRight, Download, Sparkles,
  Copy, Check, Stethoscope, Pill, Activity, HeartPulse,
  ShieldAlert, Clock, ArrowRight, FileCheck, Layers, Eye, User,
  Utensils, Dumbbell, Calendar, Moon, Droplets, AlertTriangle,
  CheckCircle2, ListChecks
} from 'lucide-react';
import apiClient from '../../services/api-client';
import { ROUTES } from '../../routes/config';

// ---------------------------------------------------------------------------
// Type Definitions
// ---------------------------------------------------------------------------
interface LabResultItem {
  test_name?: string;
  original_ocr_name?: string;
  value?: string | number;
  unit?: string;
  reference_range?: string;
  flag?: string;
  interpretation?: string;
}

interface MedicationItem {
  name: string;
  generic_name?: string;
  strength?: string;
  form?: string;
  route?: string;
  frequency?: string;
  timing?: string;
  duration?: string;
  indication?: string;
  instructions?: string;
}

interface DrugInteraction {
  drugs?: string[];
  severity?: 'HIGH' | 'MODERATE' | 'MILD';
  description?: string;
  action_needed?: string;
}

interface PatientGuide {
  matched_conditions?: string[];
  diet_and_nutrition?: {
    foods_to_enjoy?: string[];
    foods_to_limit?: string[];
    hydration_advice?: string;
  };
  physical_activity?: {
    recommended_activities?: string[];
    weekly_target?: string;
    safety_precautions?: string[];
  };
  follow_up_plan?: {
    next_recommended_tests?: string[];
    retest_timeline?: string;
    home_monitoring?: string[];
  };
  lifestyle_and_wellness?: {
    sleep_advice?: string;
    stress_management?: string;
    daily_routines?: string[];
  };
  warning_signs?: string[];
  medication_precautions?: {
    drug?: string;
    dosage?: string;
    precaution?: string;
    side_effects?: string;
  }[];
}

interface ClinicalAnalysis {
  document_type: string;
  patient_information?: {
    name?: string | null;
    age?: string | number | null;
    sex?: string | null;
    date?: string | null;
    mrn?: string | null;
    doctor_name?: string | null;
    facility_name?: string | null;
  };
  executive_summary?: string;
  dual_summary?: {
    doctor_notes?: string;
    patient_explanation?: string;
  };
  patient_guide?: PatientGuide;
  lab_analysis?: {
    test_results?: LabResultItem[];
    abnormal_findings?: { test_name: string; value: string; severity?: string; clinical_note?: string }[];
    organ_system_impact?: Record<string, string>;
    clinical_correlation?: string;
    follow_up_recommendations?: string[];
  };
  prescription_analysis?: {
    medications?: MedicationItem[];
    drug_interactions?: DrugInteraction[];
    precautions_and_warnings?: string[];
    adherence_schedule?: {
      morning?: string[];
      afternoon?: string[];
      evening?: string[];
      bedtime?: string[];
    };
  };
  clinical_notes_analysis?: {
    chief_complaints?: string[];
    vitals?: {
      blood_pressure?: string;
      heart_rate?: string;
      respiratory_rate?: string;
      temperature?: string;
      oxygen_saturation?: string;
    };
    diagnoses?: ({ condition: string; type?: string } | string)[];
    hospital_course?: string;
    red_flag_symptoms?: string[];
    post_discharge_care?: string[];
  };
  radiology_analysis?: {
    modality?: string;
    anatomical_site?: string;
    findings?: string[];
    impression?: string;
    plain_language_summary?: string;
    urgency?: string;
  };
  medications?: any[];
  lab_results?: any[];
  diagnoses?: any[];
  instructions?: string[];
  uncertain_text?: string[];
}

interface ScanResult {
  document_type: string;
  raw_ocr: {
    raw_text: string;
    segments: any[];
    segment_count: number;
    avg_confidence: number;
  };
  analysis: ClinicalAnalysis;
  diagnostic_state_preview: Record<string, any>;
  normalization_method: string;
  normalization_status: string;
  pipeline_time_seconds: number;
}

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'application/pdf'];
const MAX_SIZE_MB = 20;

const formatDocType = (type?: string) => {
  if (!type) return 'Medical Document';
  return type
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
};

const getFlagBadge = (flag?: string) => {
  const f = (flag || '').toUpperCase();
  if (f.includes('CRITICAL')) {
    return 'bg-red-600 text-white font-bold animate-pulse';
  }
  if (f === 'HIGH') {
    return 'bg-amber-100 text-amber-800 border border-amber-300 font-bold';
  }
  if (f === 'LOW') {
    return 'bg-sky-100 text-sky-800 border border-sky-300 font-bold';
  }
  return 'bg-emerald-50 text-emerald-700 border border-emerald-200';
};

export default function ReportAnalysis() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStage, setAnalysisStage] = useState(0);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState<'clinical' | 'patient_guide' | 'raw_text' | 'pipeline'>('clinical');
  const [copied, setCopied] = useState(false);
  const [inputMode, setInputMode] = useState<'file' | 'text'>('file');
  const [pastedText, setPastedText] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const STAGES = [
    'PaddleOCR: Extracting layout & character streams...',
    'ClinicalBERT: Medical NER & abbreviation normalization...',
    'Reference KB: Evaluating biomarker intervals & critical cutoffs...',
    'Groq LLaMA-3.3: Synthesizing multi-schema clinical intelligence...',
    'Building dual-perspective doctor notes and patient guide...',
  ];

  const handleFile = (f: File) => {
    setError('');
    setResult(null);
    if (!ACCEPTED_TYPES.includes(f.type)) {
      setError('Unsupported file type. Please upload a PDF, JPG, PNG, or WEBP file.');
      return;
    }
    if (f.size > MAX_SIZE_MB * 1024 * 1024) {
      setError(`File exceeds maximum size limit of ${MAX_SIZE_MB} MB.`);
      return;
    }
    setFile(f);
    if (f.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = e => setPreview(e.target?.result as string);
      reader.readAsDataURL(f);
    } else {
      setPreview(null);
    }
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) handleFile(dropped);
  }, []);

  const handleAnalyze = async () => {
    if (inputMode === 'file' && !file) return;
    if (inputMode === 'text' && !pastedText.trim()) return;

    setIsAnalyzing(true);
    setError('');
    setResult(null);
    setAnalysisStage(0);

    const stageInterval = setInterval(() => {
      setAnalysisStage(prev => Math.min(prev + 1, STAGES.length - 1));
    }, 1800);

    try {
      let response;
      if (inputMode === 'file' && file) {
        const formData = new FormData();
        formData.append('file', file);
        response = await apiClient.post('/ocr/scan', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 120000,
        });
      } else {
        response = await apiClient.post('/ocr/analyze-text', { text: pastedText }, {
          timeout: 120000,
        });
      }

      clearInterval(stageInterval);

      if (response.data?.success) {
        setResult(response.data);
      } else {
        setError(response.data?.error || 'Document analysis failed. Please try again.');
      }
    } catch (err: any) {
      clearInterval(stageInterval);
      setError(
        err?.response?.data?.error ||
        'Could not connect to the OCRAgent service. Please ensure the backend and PaddleOCR are running.'
      );
    } finally {
      setIsAnalyzing(false);
      setAnalysisStage(0);
    }
  };

  const clearFile = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError('');
    setPastedText('');
  };

  const copyExtractedText = () => {
    if (!result?.raw_ocr?.raw_text) return;
    navigator.clipboard.writeText(result.raw_ocr.raw_text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const sendToDiagnosticPipeline = (autoRun: boolean = true) => {
    if (!result?.diagnostic_state_preview) return;
    // Navigate to patient intake with pre-filled document parameters
    navigate(ROUTES.PATIENT_INTAKE, {
      state: { prefilledFromOCR: result.diagnostic_state_preview, autoRun }
    });
  };

  const analysis = result?.analysis;
  const docType = result?.document_type || analysis?.document_type || 'medical_report';
  const patientInfo = analysis?.patient_information;
  const labTests: LabResultItem[] = analysis?.lab_analysis?.test_results || (analysis as any)?.lab_results || [];
  const medicationsList: MedicationItem[] = analysis?.prescription_analysis?.medications || (analysis as any)?.medications || [];
  const isLab = docType === 'lab_report' || labTests.length > 0;
  const isRx = docType === 'prescription' || medicationsList.length > 0;
  const isNotes = docType === 'discharge_summary' || docType === 'medical_report' || Boolean(analysis?.clinical_notes_analysis);
  const isRadiology = docType === 'radiology_report' || Boolean(analysis?.radiology_analysis);

  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-12">

      {/* Header Glassmorphic Banner */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative overflow-hidden rounded-3xl p-8 text-white shadow-xl"
        style={{ background: 'linear-gradient(135deg, #09131e 0%, #0d283e 50%, #083344 100%)' }}
      >
        <div
          className="absolute inset-0 pointer-events-none opacity-[0.06]"
          style={{
            backgroundImage: `linear-gradient(to right, #38bdf8 1px, transparent 1px), linear-gradient(to bottom, #38bdf8 1px, transparent 1px)`,
            backgroundSize: '28px 28px',
          }}
        />
        <div className="absolute -top-20 -right-20 w-64 h-64 bg-teal-400/15 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute -bottom-20 -left-20 w-64 h-64 bg-sky-400/15 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div className="flex items-center gap-5">
            <div className="w-16 h-16 rounded-2xl bg-teal-500/20 border border-teal-400/30 flex items-center justify-center shrink-0 shadow-inner">
              <FileSearch className="w-8 h-8 text-teal-300" />
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-black leading-tight text-white tracking-tight">
                Clinical Report Analysis
              </h1>
              <p className="text-slate-300 text-sm mt-1 max-w-2xl">
                Upload prescriptions, blood/lab panels, discharge summaries, or written radiology reports for deep entity extraction, biomarker grounding, and dual-perspective clinical insights.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 text-xs bg-slate-900/60 border border-slate-700/60 rounded-xl px-3.5 py-2">
            <ShieldAlert className="w-4 h-4 text-teal-400 shrink-0" />
            <span className="text-slate-300 text-[11px]">
              Textual Reports Scope · HIPAA Compliant Processing
            </span>
          </div>
        </div>
      </motion.div >

      {/* Main Grid: Upload & Controls on Left, Rich Findings on Right */}
      < div className="grid grid-cols-1 lg:grid-cols-12 gap-6" >

        {/* Left Column: Upload / Paste (4 cols) */}
        < div className="lg:col-span-4 space-y-5" >
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white border border-slate-200 rounded-3xl p-6 shadow-sm space-y-5"
          >
            <div className="flex items-center justify-between border-b border-slate-100 pb-4">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-sky-50 rounded-xl flex items-center justify-center">
                  <Upload className="w-4 h-4 text-sky-600" />
                </div>
                <h2 className="font-bold text-slate-800 text-sm">Input Document</h2>
              </div>

              {/* Mode Toggle */}
              <div className="flex items-center bg-slate-100 p-1 rounded-xl text-xs font-semibold">
                <button
                  type="button"
                  onClick={() => setInputMode('file')}
                  className={`px-3 py-1 rounded-lg transition ${inputMode === 'file' ? 'bg-white shadow text-slate-900 font-bold' : 'text-slate-500 hover:text-slate-800'}`}
                >
                  File
                </button>
                <button
                  type="button"
                  onClick={() => setInputMode('text')}
                  className={`px-3 py-1 rounded-lg transition ${inputMode === 'text' ? 'bg-white shadow text-slate-900 font-bold' : 'text-slate-500 hover:text-slate-800'}`}
                >
                  Paste Text
                </button>
              </div>
            </div>

            {inputMode === 'file' ? (
              !file ? (
                <div
                  onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={onDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 ${isDragging
                    ? 'border-teal-400 bg-teal-50/60 scale-[1.01]'
                    : 'border-slate-200 hover:border-teal-400 hover:bg-slate-50/60'
                    }`}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,.jpg,.jpeg,.png,.webp"
                    onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
                    className="hidden"
                  />
                  <div className={`w-14 h-14 mx-auto rounded-2xl flex items-center justify-center mb-3 transition-colors ${isDragging ? 'bg-teal-100' : 'bg-slate-100'}`}>
                    <Upload className={`w-6 h-6 ${isDragging ? 'text-teal-600' : 'text-slate-400'}`} />
                  </div>
                  <p className="font-bold text-slate-700 text-sm mb-1">
                    {isDragging ? 'Release to upload' : 'Drag & drop report here'}
                  </p>
                  <p className="text-xs text-slate-400">or <span className="text-teal-600 font-semibold underline">browse file</span></p>
                  <p className="text-[10px] text-slate-400 mt-2">PDF, JPG, PNG, WEBP (up to {MAX_SIZE_MB} MB)</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative border border-slate-200 rounded-2xl overflow-hidden bg-slate-50">
                    {preview ? (
                      <img src={preview} alt="Report preview" className="w-full max-h-56 object-contain p-2" />
                    ) : (
                      <div className="flex items-center gap-3 p-4">
                        <div className="w-12 h-12 bg-red-50 rounded-xl flex items-center justify-center shrink-0">
                          <FileText className="w-6 h-6 text-red-500" />
                        </div>
                        <div className="min-w-0">
                          <p className="font-semibold text-slate-800 text-sm truncate">{file.name}</p>
                          <p className="text-xs text-slate-400">{(file.size / 1024 / 1024).toFixed(2)} MB · PDF</p>
                        </div>
                      </div>
                    )}
                    <button
                      onClick={clearFile}
                      className="absolute top-2 right-2 w-7 h-7 bg-white border border-slate-200 rounded-full flex items-center justify-center text-slate-500 hover:text-red-500 shadow-sm transition"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>

                  <div className="flex items-center gap-2 bg-emerald-50 border border-emerald-200 rounded-xl px-3.5 py-2">
                    <CheckCircle className="w-4 h-4 text-emerald-600 shrink-0" />
                    <span className="text-xs font-semibold text-emerald-800 truncate">{file.name}</span>
                  </div>
                </div>
              )
            ) : (
              <div className="space-y-2">
                <textarea
                  value={pastedText}
                  onChange={e => setPastedText(e.target.value)}
                  placeholder="Paste medical prescription, blood report values, or clinical notes here directly..."
                  rows={8}
                  className="w-full p-3.5 border border-slate-200 rounded-2xl text-xs font-mono text-slate-800 focus:ring-2 focus:ring-teal-500 focus:outline-none resize-none"
                />
                <div className="flex justify-between text-[10px] text-slate-400">
                  <span>Direct text ingestion</span>
                  <span>{pastedText.length} characters</span>
                </div>
              </div>
            )}

            {error && (
              <div className="flex items-center gap-2 bg-red-50 border border-red-200 text-red-700 rounded-xl px-4 py-3 text-xs font-medium">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {/* Document Types Supported */}
            <div className="grid grid-cols-2 gap-2 text-left">
              {[
                { icon: Activity, title: 'Blood Panels', desc: 'CBC, LFT, KFT, Lipids' },
                { icon: Pill, title: 'Prescriptions', desc: 'Rx, Dosages, DDI checks' },
                { icon: FileText, title: 'Clinical Notes', desc: 'Discharge, Vitals, History' },
                { icon: FileCheck, title: 'Radiology Notes', desc: 'Impressions & Findings' },
              ].map(({ icon: Icon, title, desc }) => (
                <div key={title} className="p-2.5 bg-slate-50 border border-slate-100 rounded-xl">
                  <div className="flex items-center gap-1.5 mb-0.5">
                    <Icon className="w-3.5 h-3.5 text-teal-600" />
                    <p className="text-[11px] font-bold text-slate-700">{title}</p>
                  </div>
                  <p className="text-[9px] text-slate-400">{desc}</p>
                </div>
              ))}
            </div>

            {/* Submit Action */}
            <motion.button
              whileHover={{ scale: (file || pastedText) && !isAnalyzing ? 1.01 : 1 }}
              whileTap={{ scale: (file || pastedText) && !isAnalyzing ? 0.98 : 1 }}
              onClick={handleAnalyze}
              disabled={(!file && !pastedText.trim()) || isAnalyzing}
              className={`w-full py-3.5 rounded-2xl font-bold text-sm transition-all flex items-center justify-center gap-2 ${(file || pastedText.trim()) && !isAnalyzing
                ? 'bg-gradient-to-r from-teal-500 to-sky-600 text-white shadow-lg shadow-teal-500/25 hover:opacity-95'
                : 'bg-slate-100 text-slate-400 cursor-not-allowed'
                }`}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>{STAGES[analysisStage]}</span>
                </>
              ) : (
                <>
                  <Brain className="w-4 h-4" />
                  <span>Analyze with OCRAgent</span>
                </>
              )}
            </motion.button>
          </motion.div>
        </div >

        {/* Right Column: Dynamic Analysis & Intelligence Hub (8 cols) */}
        < div className="lg:col-span-8 space-y-5" >
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white border border-slate-200 rounded-3xl p-6 shadow-sm space-y-5 min-h-[500px]"
          >
            {/* Header & Tabs */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 bg-teal-50 rounded-xl flex items-center justify-center">
                  <Sparkles className="w-4 h-4 text-teal-600" />
                </div>
                <div>
                  <h2 className="font-bold text-slate-800 text-sm">Clinical Intelligence Workspace</h2>
                  <p className="text-[10px] text-slate-400">Structured outputs powered by ClinicalBERT & Groq LLaMA-3.3</p>
                </div>
              </div>

              {result && (
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[10px] font-bold text-teal-800 bg-teal-50 border border-teal-200 rounded-full px-2.5 py-1">
                    {formatDocType(docType)}
                  </span>
                  <span className="text-[10px] font-bold text-slate-700 bg-slate-100 rounded-full px-2.5 py-1">
                    {result.raw_ocr?.avg_confidence ? `${Math.round(result.raw_ocr.avg_confidence * 100)}% Conf` : 'N/A'}
                  </span>
                  <span className="text-[10px] text-slate-500 bg-slate-50 border border-slate-200 rounded-full px-2.5 py-1">
                    {result.pipeline_time_seconds}s
                  </span>
                </div>
              )}
            </div>

            {/* Navigation Tabs if Result is present */}
            {result && (
              <div className="flex items-center gap-2 border-b border-slate-100 pb-2 overflow-x-auto">
                <button
                  type="button"
                  onClick={() => setActiveTab('clinical')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition whitespace-nowrap ${activeTab === 'clinical'
                    ? 'bg-teal-500 text-white shadow-sm shadow-teal-500/30'
                    : 'text-slate-600 hover:bg-slate-100'
                    }`}
                >
                  <Stethoscope className="w-3.5 h-3.5" />
                  <span>Clinical Findings</span>
                </button>

                <button
                  type="button"
                  onClick={() => setActiveTab('patient_guide')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition whitespace-nowrap ${activeTab === 'patient_guide'
                    ? 'bg-teal-500 text-white shadow-sm shadow-teal-500/30'
                    : 'text-slate-600 hover:bg-slate-100'
                    }`}
                >
                  <User className="w-3.5 h-3.5" />
                  <span>Patient Guide</span>
                </button>

                <button
                  type="button"
                  onClick={() => setActiveTab('raw_text')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition whitespace-nowrap ${activeTab === 'raw_text'
                    ? 'bg-teal-500 text-white shadow-sm shadow-teal-500/30'
                    : 'text-slate-600 hover:bg-slate-100'
                    }`}
                >
                  <FileText className="w-3.5 h-3.5" />
                  <span>Raw OCR Text</span>
                </button>

                <button
                  type="button"
                  onClick={() => setActiveTab('pipeline')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition whitespace-nowrap ${activeTab === 'pipeline'
                    ? 'bg-teal-500 text-white shadow-sm shadow-teal-500/30'
                    : 'text-slate-600 hover:bg-slate-100'
                    }`}
                >
                  <Layers className="w-3.5 h-3.5" />
                  <span>Diagnostic Bridge</span>
                </button>
              </div>
            )}

            {/* Loading Animation */}
            {isAnalyzing && (
              <div className="flex flex-col items-center justify-center py-20 space-y-4">
                <div className="w-16 h-16 rounded-3xl bg-teal-50 border border-teal-100 flex items-center justify-center shadow-sm">
                  <Loader2 className="w-8 h-8 text-teal-600 animate-spin" />
                </div>
                <div className="text-center space-y-1">
                  <p className="font-bold text-slate-800 text-sm">{STAGES[analysisStage]}</p>
                  <p className="text-xs text-slate-400">Processing textual medical data with clinical intelligence models...</p>
                </div>
                <div className="w-64 h-2 bg-slate-100 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-teal-500 to-sky-500 rounded-full"
                    animate={{
                      width: [
                        `${(analysisStage / STAGES.length) * 100}%`,
                        `${((analysisStage + 1) / STAGES.length) * 100}%`
                      ]
                    }}
                    transition={{ duration: 1.8 }}
                  />
                </div>
              </div>
            )}

            {/* Empty State */}
            {!isAnalyzing && !result && (
              <div className="flex flex-col items-center justify-center py-24 space-y-4 text-center">
                <div className="w-20 h-20 bg-slate-50 border border-slate-100 rounded-3xl flex items-center justify-center">
                  <FileSearch className="w-9 h-9 text-slate-300" />
                </div>
                <div className="space-y-1">
                  <p className="font-bold text-slate-700 text-base">No Report Analyzed Yet</p>
                  <p className="text-xs text-slate-400 max-w-md">
                    Upload a medical report image, PDF, or paste clinical text on the left.
                    OCRAgent will extract all data and automatically generate dynamic clinical intelligence.
                  </p>
                </div>
              </div>
            )}

            {/* Results Display */}
            {!isAnalyzing && result && (
              <div className="space-y-6">

                {/* Patient Information Bar */}
                {patientInfo && (patientInfo.name || patientInfo.age || patientInfo.doctor_name) && (
                  <div className="bg-slate-50 border border-slate-200/80 rounded-2xl p-4 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                    <div>
                      <span className="text-[10px] text-slate-400 uppercase font-semibold block">Patient Name</span>
                      <span className="font-bold text-slate-800">{patientInfo.name || 'Not specified'}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-400 uppercase font-semibold block">Age / Sex</span>
                      <span className="font-bold text-slate-800">
                        {patientInfo.age ? `${patientInfo.age} yrs` : 'N/A'} {patientInfo.sex ? `· ${patientInfo.sex}` : ''}
                      </span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-400 uppercase font-semibold block">Physician / Facility</span>
                      <span className="font-bold text-slate-800 truncate block">
                        {patientInfo.doctor_name || patientInfo.facility_name || 'Not specified'}
                      </span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-400 uppercase font-semibold block">Report Date</span>
                      <span className="font-bold text-slate-800">{patientInfo.date || 'Undated'}</span>
                    </div>
                  </div>
                )}

                {/* TAB 1: Clinical Findings (Dynamic by Report Type) */}
                {activeTab === 'clinical' && (
                  <div className="space-y-5">
                    {/* Executive Summary */}
                    {analysis?.executive_summary && (
                      <div className="bg-gradient-to-r from-teal-50/70 to-sky-50/70 border border-teal-100 rounded-2xl p-4">
                        <div className="flex items-center gap-1.5 mb-1 text-teal-800 font-bold text-xs">
                          <Brain className="w-3.5 h-3.5 text-teal-600" />
                          <span>Executive Clinical Synthesis</span>
                        </div>
                        <p className="text-xs text-slate-700 leading-relaxed">{analysis.executive_summary}</p>
                      </div>
                    )}

                    {/* SUBVIEW A: Lab Results & Biomarkers */}
                    {isLab && labTests.length > 0 && (
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wide flex items-center gap-1.5">
                            <Activity className="w-4 h-4 text-teal-600" />
                            <span>Biomarker Table ({labTests.length} parameters)</span>
                          </h3>
                          <span className="text-[10px] text-slate-400 font-medium">Grounded with Reference KB</span>
                        </div>

                        <div className="border border-slate-200 rounded-2xl overflow-hidden shadow-sm">
                          <div className="overflow-x-auto">
                            <table className="w-full text-left text-xs">
                              <thead className="bg-slate-50 border-b border-slate-200 text-[10px] text-slate-500 uppercase tracking-wider font-bold">
                                <tr>
                                  <th className="p-3">Biomarker</th>
                                  <th className="p-3">Observed Value</th>
                                  <th className="p-3">Reference Interval</th>
                                  <th className="p-3">Status</th>
                                  <th className="p-3">Interpretation</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-slate-100">
                                {labTests.map((tr, idx) => (
                                  <tr key={idx} className="hover:bg-slate-50/80 transition">
                                    <td className="p-3 font-bold text-slate-800">{tr.test_name || tr.original_ocr_name}</td>
                                    <td className="p-3 font-extrabold text-slate-900">
                                      {tr.value} <span className="text-slate-500 font-normal">{tr.unit || ''}</span>
                                    </td>
                                    <td className="p-3 text-slate-500 font-mono text-[11px]">{tr.reference_range || 'N/A'}</td>
                                    <td className="p-3">
                                      <span className={`inline-block text-[10px] px-2 py-0.5 rounded-full ${getFlagBadge(tr.flag)}`}>
                                        {tr.flag || 'NORMAL'}
                                      </span>
                                    </td>
                                    <td className="p-3 text-slate-600 text-[11px]">{tr.interpretation || 'Within normal range.'}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>

                        {/* Organ System Impacts */}
                        {analysis?.lab_analysis?.organ_system_impact && Object.keys(analysis.lab_analysis.organ_system_impact).length > 0 && (
                          <div className="space-y-2">
                            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wide">Organ System Impact</h4>
                            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                              {Object.entries(analysis.lab_analysis.organ_system_impact).map(([system, impact]) => (
                                <div key={system} className="p-3 bg-slate-50 border border-slate-200 rounded-xl">
                                  <p className="text-[10px] uppercase font-bold text-teal-700 tracking-wider mb-0.5">{system}</p>
                                  <p className="text-xs text-slate-700 font-medium">{String(impact)}</p>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Follow-up Recommendations */}
                        {analysis?.lab_analysis?.follow_up_recommendations && analysis.lab_analysis.follow_up_recommendations.length > 0 && (
                          <div className="p-3.5 bg-amber-50/70 border border-amber-200 rounded-2xl space-y-1.5">
                            <p className="text-xs font-bold text-amber-900 flex items-center gap-1.5">
                              <CheckCircle className="w-3.5 h-3.5 text-amber-700" />
                              <span>Recommended Clinical Follow-Ups</span>
                            </p>
                            <ul className="text-xs text-amber-800 list-disc list-inside space-y-1">
                              {analysis.lab_analysis.follow_up_recommendations.map((rec, i) => (
                                <li key={i}>{rec}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}

                    {/* SUBVIEW B: Prescription & Medications */}
                    {isRx && medicationsList.length > 0 && (
                      <div className="space-y-5">
                        <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wide flex items-center gap-1.5">
                          <Pill className="w-4 h-4 text-teal-600" />
                          <span>Prescribed Medications ({medicationsList.length})</span>
                        </h3>

                        {/* Drug-Drug Interaction Warning */}
                        {analysis?.prescription_analysis?.drug_interactions && analysis.prescription_analysis.drug_interactions.length > 0 && (
                          <div className="p-4 bg-red-50 border border-red-200 rounded-2xl space-y-2">
                            <div className="flex items-center gap-2 text-red-800 font-bold text-xs">
                              <ShieldAlert className="w-4 h-4 text-red-600" />
                              <span>Drug-Drug Interaction Alerts Detected</span>
                            </div>
                            {analysis.prescription_analysis.drug_interactions.map((ddi, i) => (
                              <div key={i} className="text-xs text-red-700 bg-white/70 p-2.5 rounded-xl border border-red-100">
                                <div className="flex items-center justify-between mb-1">
                                  <span className="font-bold">{ddi.drugs?.join(' + ') || 'Medication Combination'}</span>
                                  <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-red-100 text-red-800 uppercase">
                                    {ddi.severity || 'ALERT'}
                                  </span>
                                </div>
                                <p className="text-[11px] mb-1">{ddi.description}</p>
                                {ddi.action_needed && (
                                  <p className="text-[10px] font-semibold text-red-900">Recommended Action: {ddi.action_needed}</p>
                                )}
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Medication Cards */}
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                          {medicationsList.map((med, i) => (
                            <div key={i} className="p-4 bg-slate-50 border border-slate-200 rounded-2xl space-y-2 hover:border-teal-300 transition">
                              <div className="flex items-start justify-between gap-2">
                                <div>
                                  <h4 className="font-extrabold text-slate-900 text-sm">{med.name}</h4>
                                  {med.generic_name && (
                                    <p className="text-[10px] text-slate-500 italic">{med.generic_name}</p>
                                  )}
                                </div>
                                {med.form && (
                                  <span className="text-[10px] font-semibold px-2 py-0.5 bg-teal-50 text-teal-700 border border-teal-200 rounded-full">
                                    {med.form}
                                  </span>
                                )}
                              </div>

                              <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-600">
                                <div>
                                  <span className="text-slate-400 text-[9px] block uppercase">Strength:</span>
                                  <span className="font-bold text-slate-800">{med.strength || 'Standard'}</span>
                                </div>
                                <div>
                                  <span className="text-slate-400 text-[9px] block uppercase">Schedule:</span>
                                  <span className="font-bold text-slate-800">{med.frequency || 'As directed'}</span>
                                </div>
                                <div>
                                  <span className="text-slate-400 text-[9px] block uppercase">Timing:</span>
                                  <span className="font-semibold text-slate-700">{med.timing || 'With meals'}</span>
                                </div>
                                <div>
                                  <span className="text-slate-400 text-[9px] block uppercase">Duration:</span>
                                  <span className="font-semibold text-slate-700">{med.duration || 'Per advice'}</span>
                                </div>
                              </div>

                              {med.indication && (
                                <p className="text-[10px] text-slate-500 bg-white p-2 rounded-xl border border-slate-100">
                                  <strong>Indication:</strong> {med.indication}
                                </p>
                              )}
                            </div>
                          ))}
                        </div>

                        {/* Precautions */}
                        {analysis?.prescription_analysis?.precautions_and_warnings && analysis.prescription_analysis.precautions_and_warnings.length > 0 && (
                          <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-2xl">
                            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wide mb-1.5">Special Precautions & Warnings</h4>
                            <ul className="text-xs text-slate-600 list-disc list-inside space-y-1">
                              {analysis.prescription_analysis.precautions_and_warnings.map((p, idx) => (
                                <li key={idx}>{p}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}

                    {/* SUBVIEW C: Clinical Notes & Discharge Summary */}
                    {isNotes && analysis?.clinical_notes_analysis && (
                      <div className="space-y-4">
                        <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wide flex items-center gap-1.5">
                          <FileText className="w-4 h-4 text-teal-600" />
                          <span>Clinical Trajectory & Vitals</span>
                        </h3>

                        {/* Vitals Grid */}
                        {analysis.clinical_notes_analysis.vitals && (
                          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-center">
                            {Object.entries(analysis.clinical_notes_analysis.vitals).map(([k, v]) => (
                              <div key={k} className="p-3 bg-slate-50 border border-slate-200 rounded-2xl">
                                <span className="text-[10px] uppercase font-bold text-slate-400 block">{k.replace(/_/g, ' ')}</span>
                                <span className="text-sm font-black text-slate-800 mt-1 block">{String(v || 'N/A')}</span>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Diagnoses */}
                        {analysis.clinical_notes_analysis.diagnoses && (
                          <div className="space-y-1.5">
                            <span className="text-xs font-bold text-slate-600 uppercase tracking-wide">Recorded Diagnoses</span>
                            <div className="flex flex-wrap gap-2">
                              {analysis.clinical_notes_analysis.diagnoses.map((d, i) => (
                                <span key={i} className="text-xs font-bold px-3 py-1 rounded-full bg-teal-50 text-teal-800 border border-teal-200">
                                  {typeof d === 'string' ? d : d.condition}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Red Flag Symptoms */}
                        {analysis.clinical_notes_analysis.red_flag_symptoms && analysis.clinical_notes_analysis.red_flag_symptoms.length > 0 && (
                          <div className="p-4 bg-red-50 border border-red-200 rounded-2xl space-y-1.5">
                            <p className="text-xs font-bold text-red-800 flex items-center gap-1.5">
                              <AlertCircle className="w-4 h-4 text-red-600" />
                              <span>Red Flag Symptoms (Seek Emergency Care if Present)</span>
                            </p>
                            <ul className="text-xs text-red-700 list-disc list-inside space-y-1">
                              {analysis.clinical_notes_analysis.red_flag_symptoms.map((rf, idx) => (
                                <li key={idx} className="font-semibold">{rf}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}

                    {/* SUBVIEW D: Radiology Report */}
                    {isRadiology && analysis?.radiology_analysis && (
                      <div className="space-y-4">
                        <div className="flex items-center gap-2">
                          <span className="px-3 py-1 bg-sky-50 text-sky-800 font-bold text-xs rounded-full border border-sky-200">
                            {analysis.radiology_analysis.modality || 'Imaging'}
                          </span>
                          <span className="px-3 py-1 bg-slate-100 text-slate-700 font-semibold text-xs rounded-full">
                            Region: {analysis.radiology_analysis.anatomical_site || 'Unspecified'}
                          </span>
                        </div>

                        <div className="space-y-2">
                          <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wide">Findings</h4>
                          <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-2xl text-xs space-y-1.5">
                            {analysis.radiology_analysis.findings?.map((f, i) => (
                              <p key={i} className="text-slate-700">• {f}</p>
                            ))}
                          </div>
                        </div>

                        <div className="p-4 bg-teal-50 border border-teal-200 rounded-2xl space-y-1">
                          <span className="text-[10px] font-bold text-teal-800 uppercase tracking-wider">Impression / Conclusion</span>
                          <p className="text-sm font-black text-teal-950">{analysis.radiology_analysis.impression}</p>
                        </div>
                      </div>
                    )}

                    {/* Doctor's Technical Clinical Note */}
                    {analysis?.dual_summary?.doctor_notes && (
                      <div className="p-4 bg-slate-900 text-white rounded-2xl space-y-2">
                        <div className="flex items-center gap-2 text-teal-400 font-bold text-xs">
                          <Stethoscope className="w-4 h-4" />
                          <span>Physician Clinical Impression (Healthcare Providers)</span>
                        </div>
                        <p className="text-xs text-slate-300 leading-relaxed whitespace-pre-wrap">
                          {analysis.dual_summary.doctor_notes}
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {/* TAB 2: Patient Guide View */}
                {activeTab === 'patient_guide' && (
                  <div className="space-y-6">
                    {/* Summary Card */}
                    <div className="p-5 bg-gradient-to-br from-emerald-500/10 via-teal-500/5 to-sky-500/10 border border-teal-200/80 rounded-3xl shadow-sm space-y-3">
                      <div className="flex items-center justify-between flex-wrap gap-2">
                        <div className="flex items-center gap-2 text-teal-900 font-extrabold text-sm">
                          <User className="w-4 h-4 text-teal-600" />
                          <span>What This Medical Report Means For You</span>
                        </div>
                        <span className="flex items-center gap-1.5 px-3 py-1 bg-teal-100 text-teal-800 text-[11px] font-bold rounded-full border border-teal-200">
                          <Sparkles className="w-3 h-3 text-teal-600" />
                          <span>Evidence-Grounded Guide</span>
                        </span>
                      </div>

                      <p className="text-sm text-slate-800 leading-relaxed font-normal">
                        {analysis?.dual_summary?.patient_explanation ||
                          analysis?.executive_summary ||
                          'Your document has been analyzed by MedAgentix AI. Below are personalized, evidence-based recommendations to help you understand and manage your health.'}
                      </p>

                      {/* Matched Condition Badges */}
                      {analysis?.patient_guide?.matched_conditions && analysis.patient_guide.matched_conditions.length > 0 && (
                        <div className="flex items-center gap-1.5 flex-wrap pt-1">
                          <span className="text-[10px] uppercase font-bold text-slate-400">Target Clinical Profiles:</span>
                          {analysis.patient_guide.matched_conditions.map((cond, i) => (
                            <span key={i} className="text-[11px] font-bold px-2.5 py-0.5 rounded-full bg-white text-teal-700 border border-teal-200 shadow-2xs">
                              {cond.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Section 1: Dietary & Nutrition Guide */}
                    {analysis?.patient_guide?.diet_and_nutrition && (
                      <div className="bg-white border border-slate-200/80 rounded-3xl p-5 shadow-xs space-y-4">
                        <div className="flex items-center gap-2.5 border-b border-slate-100 pb-3">
                          <div className="w-9 h-9 rounded-2xl bg-emerald-50 text-emerald-600 border border-emerald-200 flex items-center justify-center">
                            <Utensils className="w-5 h-5" />
                          </div>
                          <div>
                            <h4 className="text-sm font-bold text-slate-800">Dietary & Nutrition Guide</h4>
                            <p className="text-xs text-slate-500">Evidence-based foods to balance your biomarkers and promote metabolic health</p>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {/* Foods to Enjoy */}
                          <div className="p-4 bg-emerald-50/50 border border-emerald-100 rounded-2xl space-y-2.5">
                            <div className="flex items-center gap-1.5 text-xs font-bold text-emerald-800 uppercase tracking-wide">
                              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                              <span>Foods to Enjoy & Emphasize</span>
                            </div>
                            <ul className="space-y-2">
                              {(analysis.patient_guide.diet_and_nutrition.foods_to_enjoy || [
                                'Leafy green vegetables', 'Whole grains & fiber-rich legumes', 'Lean protein sources'
                              ]).map((item, idx) => (
                                <li key={idx} className="text-xs text-slate-700 flex items-start gap-2 bg-white/80 p-2 rounded-xl border border-emerald-100/60">
                                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-1.5 shrink-0" />
                                  <span>{item}</span>
                                </li>
                              ))}
                            </ul>
                          </div>

                          {/* Foods to Limit */}
                          <div className="p-4 bg-amber-50/50 border border-amber-100 rounded-2xl space-y-2.5">
                            <div className="flex items-center gap-1.5 text-xs font-bold text-amber-900 uppercase tracking-wide">
                              <AlertTriangle className="w-4 h-4 text-amber-600" />
                              <span>Foods to Limit or Avoid</span>
                            </div>
                            <ul className="space-y-2">
                              {(analysis.patient_guide.diet_and_nutrition.foods_to_limit || [
                                'Refined flour and added sugars', 'Deep-fried foods and trans fats', 'High sodium condiments'
                              ]).map((item, idx) => (
                                <li key={idx} className="text-xs text-slate-700 flex items-start gap-2 bg-white/80 p-2 rounded-xl border border-amber-100/60">
                                  <span className="w-1.5 h-1.5 rounded-full bg-amber-500 mt-1.5 shrink-0" />
                                  <span>{item}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        </div>

                        {/* Hydration advice */}
                        {analysis.patient_guide.diet_and_nutrition.hydration_advice && (
                          <div className="flex items-start gap-2.5 p-3.5 bg-sky-50/80 border border-sky-100 rounded-2xl text-xs text-sky-900">
                            <Droplets className="w-4 h-4 text-sky-600 shrink-0 mt-0.5" />
                            <div>
                              <strong className="font-semibold text-sky-950">Daily Hydration: </strong>
                              <span>{analysis.patient_guide.diet_and_nutrition.hydration_advice}</span>
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Section 2: Physical Activity & Exercise Protocol */}
                    {analysis?.patient_guide?.physical_activity && (
                      <div className="bg-white border border-slate-200/80 rounded-3xl p-5 shadow-xs space-y-4">
                        <div className="flex items-center justify-between flex-wrap gap-2 border-b border-slate-100 pb-3">
                          <div className="flex items-center gap-2.5">
                            <div className="w-9 h-9 rounded-2xl bg-indigo-50 text-indigo-600 border border-indigo-200 flex items-center justify-center">
                              <Dumbbell className="w-5 h-5" />
                            </div>
                            <div>
                              <h4 className="text-sm font-bold text-slate-800">Physical Activity & Safe Movement</h4>
                              <p className="text-xs text-slate-500">Customized movement routine tailored to your physical capacity</p>
                            </div>
                          </div>
                          {analysis.patient_guide.physical_activity.weekly_target && (
                            <span className="text-xs font-bold px-3 py-1 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200">
                              🎯 {analysis.patient_guide.physical_activity.weekly_target}
                            </span>
                          )}
                        </div>

                        <div className="space-y-3">
                          <div className="space-y-2">
                            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wide">Recommended Routines</span>
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                              {(analysis.patient_guide.physical_activity.recommended_activities || [
                                '30 minutes brisk walking daily', 'Gentle mobility and stretching'
                              ]).map((act, i) => (
                                <div key={i} className="flex items-start gap-2 p-3 bg-slate-50 rounded-2xl border border-slate-100 text-xs text-slate-700">
                                  <Activity className="w-4 h-4 text-indigo-600 shrink-0 mt-0.5" />
                                  <span>{act}</span>
                                </div>
                              ))}
                            </div>
                          </div>

                          {analysis.patient_guide.physical_activity.safety_precautions && analysis.patient_guide.physical_activity.safety_precautions.length > 0 && (
                            <div className="p-3.5 bg-slate-50 rounded-2xl border border-slate-200/70 space-y-1.5">
                              <span className="text-[11px] font-bold text-slate-600 uppercase tracking-wide flex items-center gap-1.5">
                                <ShieldAlert className="w-3.5 h-3.5 text-slate-500" />
                                <span>Movement Safety Precautions</span>
                              </span>
                              <ul className="text-xs text-slate-600 space-y-1">
                                {analysis.patient_guide.physical_activity.safety_precautions.map((safe, i) => (
                                  <li key={i} className="flex items-center gap-2">
                                    <span className="w-1.5 h-1.5 rounded-full bg-slate-400 shrink-0" />
                                    <span>{safe}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Section 3: Monitoring & Follow-Up Timetable */}
                    {analysis?.patient_guide?.follow_up_plan && (
                      <div className="bg-white border border-slate-200/80 rounded-3xl p-5 shadow-xs space-y-4">
                        <div className="flex items-center justify-between flex-wrap gap-2 border-b border-slate-100 pb-3">
                          <div className="flex items-center gap-2.5">
                            <div className="w-9 h-9 rounded-2xl bg-amber-50 text-amber-600 border border-amber-200 flex items-center justify-center">
                              <Calendar className="w-5 h-5" />
                            </div>
                            <div>
                              <h4 className="text-sm font-bold text-slate-800">Monitoring & Follow-Up Timetable</h4>
                              <p className="text-xs text-slate-500">Upcoming screenings and recommended home tracking</p>
                            </div>
                          </div>
                          {analysis.patient_guide.follow_up_plan.retest_timeline && (
                            <span className="text-xs font-bold px-3 py-1 rounded-full bg-amber-50 text-amber-800 border border-amber-200 flex items-center gap-1">
                              <Clock className="w-3.5 h-3.5 text-amber-600" />
                              <span>{analysis.patient_guide.follow_up_plan.retest_timeline}</span>
                            </span>
                          )}
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {/* Recommended Next Tests */}
                          <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100 space-y-2">
                            <span className="text-xs font-bold text-slate-700 uppercase tracking-wide flex items-center gap-1.5">
                              <ListChecks className="w-4 h-4 text-teal-600" />
                              <span>Recommended Follow-Up Tests</span>
                            </span>
                            <ul className="space-y-1.5">
                              {(analysis.patient_guide.follow_up_plan.next_recommended_tests || [
                                'Repeat biomarker panel in 90 days', 'Consult attending physician'
                              ]).map((test, idx) => (
                                <li key={idx} className="text-xs text-slate-700 flex items-start gap-2 bg-white p-2 rounded-xl border border-slate-200/60">
                                  <ChevronRight className="w-3.5 h-3.5 text-teal-600 shrink-0 mt-0.5" />
                                  <span>{test}</span>
                                </li>
                              ))}
                            </ul>
                          </div>

                          {/* Home Monitoring */}
                          <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100 space-y-2">
                            <span className="text-xs font-bold text-slate-700 uppercase tracking-wide flex items-center gap-1.5">
                              <Activity className="w-4 h-4 text-amber-600" />
                              <span>Home Self-Monitoring Checklist</span>
                            </span>
                            <ul className="space-y-1.5">
                              {(analysis.patient_guide.follow_up_plan.home_monitoring || [
                                'Keep a daily log of symptoms or vital signs', 'Track morning resting values'
                              ]).map((item, idx) => (
                                <li key={idx} className="text-xs text-slate-700 flex items-start gap-2 bg-white p-2 rounded-xl border border-slate-200/60">
                                  <CheckCircle className="w-3.5 h-3.5 text-amber-500 shrink-0 mt-0.5" />
                                  <span>{item}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Section 4: Daily Habits & Lifestyle Wellness */}
                    {analysis?.patient_guide?.lifestyle_and_wellness && (
                      <div className="bg-white border border-slate-200/80 rounded-3xl p-5 shadow-xs space-y-4">
                        <div className="flex items-center gap-2.5 border-b border-slate-100 pb-3">
                          <div className="w-9 h-9 rounded-2xl bg-teal-50 text-teal-600 border border-teal-200 flex items-center justify-center">
                            <Moon className="w-5 h-5" />
                          </div>
                          <div>
                            <h4 className="text-sm font-bold text-slate-800">Daily Habits & Lifestyle Wellness</h4>
                            <p className="text-xs text-slate-500">Sleep hygiene, stress reduction, and healthy daily rhythms</p>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          {analysis.patient_guide.lifestyle_and_wellness.sleep_advice && (
                            <div className="p-3.5 bg-slate-50 rounded-2xl border border-slate-100 space-y-1">
                              <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wide">Sleep Optimization</span>
                              <p className="text-xs text-slate-700 leading-relaxed">{analysis.patient_guide.lifestyle_and_wellness.sleep_advice}</p>
                            </div>
                          )}

                          {analysis.patient_guide.lifestyle_and_wellness.stress_management && (
                            <div className="p-3.5 bg-slate-50 rounded-2xl border border-slate-100 space-y-1">
                              <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wide">Stress Management</span>
                              <p className="text-xs text-slate-700 leading-relaxed">{analysis.patient_guide.lifestyle_and_wellness.stress_management}</p>
                            </div>
                          )}
                        </div>

                        {analysis.patient_guide.lifestyle_and_wellness.daily_routines && analysis.patient_guide.lifestyle_and_wellness.daily_routines.length > 0 && (
                          <div className="space-y-1.5 pt-1">
                            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wide">Suggested Daily Routines</span>
                            <div className="flex flex-wrap gap-2">
                              {analysis.patient_guide.lifestyle_and_wellness.daily_routines.map((routine, i) => (
                                <span key={i} className="text-xs bg-slate-100 text-slate-700 px-3 py-1.5 rounded-xl border border-slate-200">
                                  ✓ {routine}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Section 5: Warning Signs & Red Flags */}
                    {analysis?.patient_guide?.warning_signs && analysis.patient_guide.warning_signs.length > 0 && (
                      <div className="p-5 bg-rose-50/80 border border-rose-200 rounded-3xl space-y-3 shadow-xs">
                        <div className="flex items-center gap-2 text-rose-900 font-extrabold text-sm">
                          <AlertCircle className="w-5 h-5 text-rose-600 shrink-0" />
                          <span>When to Contact a Doctor / Urgent Red Flags</span>
                        </div>
                        <p className="text-xs text-rose-800">
                          Seek prompt medical evaluation if you experience any of the following symptoms:
                        </p>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
                          {analysis.patient_guide.warning_signs.map((sign, idx) => (
                            <div key={idx} className="flex items-start gap-2 p-2.5 bg-white/90 rounded-2xl border border-rose-200 text-xs text-rose-950 font-medium">
                              <span className="w-1.5 h-1.5 rounded-full bg-rose-600 mt-1.5 shrink-0" />
                              <span>{sign}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* TAB 3: Full Verbatim Raw OCR Text */}
                {activeTab === 'raw_text' && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-slate-600 uppercase tracking-wide">
                        PaddleOCR Extracted Stream ({result.raw_ocr?.segment_count || 0} segments)
                      </span>
                      <button
                        type="button"
                        onClick={copyExtractedText}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-xs font-bold transition"
                      >
                        {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                        <span>{copied ? 'Copied to Clipboard' : 'Copy Full Text'}</span>
                      </button>
                    </div>

                    <div className="relative">
                      <pre className="p-4 bg-slate-900 text-emerald-400 font-mono text-xs rounded-2xl overflow-x-auto whitespace-pre-wrap max-h-96 leading-relaxed shadow-inner">
                        {result.raw_ocr?.raw_text || '(No text extracted)'}
                      </pre>
                    </div>
                  </div>
                )}

                {/* TAB 4: Diagnostic Pipeline Bridge */}
                {activeTab === 'pipeline' && (
                  <div className="space-y-4">
                    <div className="p-4 bg-indigo-50 border border-indigo-200 rounded-2xl space-y-2">
                      <div className="flex items-center gap-2 text-indigo-900 font-extrabold text-xs">
                        <Layers className="w-4 h-4 text-indigo-600" />
                        <span>Parameters Mapped for LangGraph DiagnosticState</span>
                      </div>
                      <p className="text-xs text-indigo-800">
                        These clinical findings were automatically parsed by OCRAgent and can be directly piped into MedAgentix AI's multi-agent diagnostic consultation workflow.
                      </p>
                    </div>

                    <div className="bg-slate-50 border border-slate-200 rounded-2xl p-4">
                      <pre className="text-[11px] font-mono text-slate-700 whitespace-pre-wrap overflow-x-auto max-h-60">
                        {JSON.stringify(result.diagnostic_state_preview, null, 2)}
                      </pre>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                      <motion.button
                        whileHover={{ scale: 1.01 }}
                        whileTap={{ scale: 0.99 }}
                        onClick={() => sendToDiagnosticPipeline(true)}
                        className="py-3.5 px-4 rounded-2xl font-bold text-xs bg-gradient-to-r from-teal-500 to-sky-600 text-white shadow-lg shadow-teal-500/25 flex items-center justify-center gap-2"
                      >
                        <Sparkles className="w-4 h-4" />
                        <span>Run Diagnostic Pipeline Directly</span>
                        <ArrowRight className="w-4 h-4" />
                      </motion.button>

                      <motion.button
                        whileHover={{ scale: 1.01 }}
                        whileTap={{ scale: 0.99 }}
                        onClick={() => sendToDiagnosticPipeline(false)}
                        className="py-3.5 px-4 rounded-2xl font-bold text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 flex items-center justify-center gap-2"
                      >
                        <FileText className="w-4 h-4" />
                        <span>Review & Customize Symptoms First</span>
                      </motion.button>
                    </div>
                  </div>
                )}

                {/* Medical Disclaimer */}
                <div className="bg-slate-50 border border-slate-200/60 rounded-2xl p-3.5 text-[10px] text-slate-500 leading-relaxed">
                  <strong>⚕ Clinical Disclaimer:</strong> MedAgentix OCRAgent provides automated medical document transcription and clinical assistance. All interpretations, reference intervals, and medication schedules should be verified by a licensed healthcare practitioner before clinical action.
                </div>
              </div>
            )}
          </motion.div>
        </div >

      </div >
    </div >
  );
}
