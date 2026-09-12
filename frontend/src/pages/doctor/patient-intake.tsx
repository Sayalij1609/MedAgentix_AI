import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import apiClient from '../../services/api-client';
import { useAuth } from '../../context/auth-context';
import PipelineVisualizer from '../../components/common/PipelineVisualizer';
import {
  User, Calendar, Activity, Stethoscope, ClipboardList,
  ChevronRight, ChevronLeft, AlertCircle, Brain
} from 'lucide-react';

const QUICK_SYMPTOMS = [
  { name: 'Fever', emoji: '🔥' }, { name: 'Cough', emoji: '😷' },
  { name: 'Fatigue', emoji: '😫' }, { name: 'Headache', emoji: '🤕' },
  { name: 'Chest Pain', emoji: '💔' }, { name: 'Breathlessness', emoji: '😤' },
  { name: 'Nausea', emoji: '🤢' }, { name: 'Vomiting', emoji: '🤮' },
  { name: 'Body Pain', emoji: '💪' }, { name: 'Joint Pain', emoji: '🦴' },
  { name: 'Sore Throat', emoji: '🗣️' }, { name: 'Runny Nose', emoji: '🤧' },
  { name: 'Dizziness', emoji: '😵' }, { name: 'Rash', emoji: '🔴' },
  { name: 'Abdominal Pain', emoji: '🫃' }, { name: 'Back Pain', emoji: '🩻' },
];

const MEDICAL_HISTORY_OPTIONS = [
  'Diabetes', 'Hypertension', 'Asthma', 'Heart Disease',
  'Kidney Disease', 'Liver Disease', 'Thyroid Disorder', 'Cancer (any)',
  'Obesity', 'COPD', 'Epilepsy', 'Arthritis',
];

const LIFESTYLE_OPTIONS = [
  'Smoking', 'Alcohol Use', 'Sedentary Lifestyle', 'High-Fat Diet',
  'High-Stress Job', 'Poor Sleep', 'Regular Exercise', 'Vegetarian Diet',
];

const LOADING_STAGES = [
  'Starting clinical assessment...', 'Extracting symptoms with ClinicalBERT...',
  'Running differential diagnosis engine...', 'Evaluating risk factors...',
  'Analyzing symptom timeline...', 'Checking for emergency indicators...',
  'Running AI prediction models...', 'Invoking Meditron-7B clinical reasoning...',
  'Generating differential analysis...', 'Finding recommended treatments...',
  'Cross-referencing drug database...', 'Compiling clinical report...',
  'Running final quality checks...', 'Almost done — preparing results...',
];

const calculateAge = (dob: string): number | null => {
  if (!dob) return null;
  const birth = new Date(dob);
  const today = new Date();
  let age = today.getFullYear() - birth.getFullYear();
  if (today.getMonth() < birth.getMonth() ||
    (today.getMonth() === birth.getMonth() && today.getDate() < birth.getDate())) age--;
  return age >= 0 ? age : null;
};

export default function DoctorPatientIntake() {
  const navigate = useNavigate();
  const { user } = useAuth();

  const [step, setStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [loadingStage, setLoadingStage] = useState(0);
  const [errorMsg, setErrorMsg] = useState('');

  // Step 1 — Patient identity
  const [patientName, setPatientName] = useState('');
  const [dateOfBirth, setDateOfBirth] = useState('');
  const [gender, setGender] = useState('');

  // Step 2 — Chief complaint + symptoms
  const [chiefComplaint, setChiefComplaint] = useState('');
  const [selectedSymptoms, setSelectedSymptoms] = useState<{ name: string; duration_days: number }[]>([]);

  // Step 3 — Vitals
  const [heartRate, setHeartRate] = useState<number | ''>('');
  const [oxygenLevel, setOxygenLevel] = useState<number | ''>('');
  const [systolicBp, setSystolicBp] = useState<number | ''>('');
  const [diastolicBp, setDiastolicBp] = useState<number | ''>('');
  const [temperature, setTemperature] = useState<number | ''>('');
  const [cholesterol, setCholesterol] = useState<number | ''>('');
  const [medicalHistory, setMedicalHistory] = useState<string[]>([]);
  const [lifestyleFactors, setLifestyleFactors] = useState<string[]>([]);

  const toggleSymptom = (name: string) => {
    if (selectedSymptoms.find(s => s.name === name)) {
      setSelectedSymptoms(selectedSymptoms.filter(s => s.name !== name));
    } else {
      setSelectedSymptoms([...selectedSymptoms, { name, duration_days: 1 }]);
    }
  };

  const updateDuration = (name: string, days: number) => {
    setSelectedSymptoms(selectedSymptoms.map(s => s.name === name ? { ...s, duration_days: days } : s));
  };

  const toggleArr = (arr: string[], setArr: React.Dispatch<React.SetStateAction<string[]>>, val: string) => {
    if (arr.includes(val)) setArr(arr.filter(v => v !== val));
    else setArr([...arr, val]);
  };

  const handleNext = () => {
    setErrorMsg('');
    if (step === 1) {
      if (!patientName.trim()) { setErrorMsg('Please enter the patient name.'); return; }
      if (!dateOfBirth) { setErrorMsg('Please enter the patient date of birth.'); return; }
      if (!gender) { setErrorMsg('Please select patient gender.'); return; }
      const age = calculateAge(dateOfBirth);
      if (age === null || age < 0 || age > 120) { setErrorMsg('Please enter a valid date of birth.'); return; }
    } else if (step === 2) {
      if (!chiefComplaint.trim()) { setErrorMsg('Please describe the patient\'s chief complaint.'); return; }
    }
    setStep(s => s + 1);
  };

  const handleSubmit = async () => {
    setErrorMsg('');
    if (heartRate !== '' && (Number(heartRate) < 30 || Number(heartRate) > 220)) {
      setErrorMsg('Heart rate must be 30–220 bpm.'); return;
    }
    if (oxygenLevel !== '' && (Number(oxygenLevel) < 50 || Number(oxygenLevel) > 100)) {
      setErrorMsg('SpO2 must be 50–100%.'); return;
    }
    setIsSubmitting(true);
    setLoadingStage(0);

    const loaderInterval = setInterval(() => {
      setLoadingStage(prev => (prev + 1) % LOADING_STAGES.length);
    }, 2500);

    const vitals: Record<string, number> = {};
    if (heartRate !== '') vitals.heart_rate = Number(heartRate);
    if (oxygenLevel !== '') vitals.oxygen_level = Number(oxygenLevel);
    if (systolicBp !== '') vitals.systolic_bp = Number(systolicBp);
    if (diastolicBp !== '') vitals.diastolic_bp = Number(diastolicBp);
    if (temperature !== '') vitals.temperature = Number(temperature);
    if (cholesterol !== '') vitals.cholesterol = Number(cholesterol);

    const payload = {
      patient_name: patientName.trim(),
      age: calculateAge(dateOfBirth) || 0,
      gender,
      chief_complaint: chiefComplaint,
      selected_symptoms: selectedSymptoms,
      vitals,
      medical_history: medicalHistory,
      lifestyle_factors: lifestyleFactors,
    };

    try {
      const response = await apiClient.post('/doctor/patient-intake', payload, { timeout: 300000 });
      clearInterval(loaderInterval);
      if (response.data?.success) {
        navigate(`/reports/${response.data.case.id}`);
      } else {
        setErrorMsg(response.data?.message || 'Unexpected error.');
        setIsSubmitting(false);
      }
    } catch (err: any) {
      clearInterval(loaderInterval);
      setErrorMsg(err.response?.data?.message || 'Failed to submit. Please try again.');
      setIsSubmitting(false);
    }
  };

  // ── Loading screen ──────────────────────────────────────
  if (isSubmitting) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[70vh] p-4 max-w-2xl mx-auto text-center space-y-8">
        <div className="space-y-2">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-teal-500 to-sky-600 flex items-center justify-center mx-auto shadow-lg">
            <Brain className="w-7 h-7 text-white" />
          </div>
          <h2 className="text-2xl font-bold tracking-tight text-slate-900">
            Analyzing {patientName}'s Health Data
          </h2>
          <p className="text-xs text-muted-foreground max-w-md mx-auto">
            Our AI pipeline is running 10-stage differential analysis on behalf of Dr. {user?.name}.
          </p>
        </div>
        <div className="w-full bg-card border border-border p-6 rounded-2xl shadow-md">
          <PipelineVisualizer activeStage={loadingStage} />
        </div>
        <div className="w-full max-w-md bg-slate-100 h-2.5 rounded-full overflow-hidden border border-border">
          <div className="bg-primary h-full transition-all duration-300"
            style={{ width: `${((loadingStage + 1) / LOADING_STAGES.length) * 100}%` }} />
        </div>
        <p className="text-sky-700 text-sm font-semibold h-8 animate-pulse">{LOADING_STAGES[loadingStage]}</p>
      </div>
    );
  }

  // ── Step labels ──────────────────────────────────────────
  const STEPS = ['Patient Info', 'Symptoms', 'Vitals & History'];

  return (
    <div className="max-w-2xl mx-auto space-y-6">

      {/* Header */}
      <div className="bg-gradient-to-r from-slate-900 via-[#0d2e4a] to-slate-900 rounded-2xl p-6 text-white relative overflow-hidden">
        <div className="absolute inset-0 opacity-[0.07]"
          style={{ backgroundImage: `linear-gradient(to right, #38bdf8 1px, transparent 1px), linear-gradient(to bottom, #38bdf8 1px, transparent 1px)`, backgroundSize: '28px 28px' }} />
        <div className="relative z-10 flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-teal-500/20 border border-teal-400/30 flex items-center justify-center shrink-0">
            <Stethoscope className="w-6 h-6 text-teal-300" />
          </div>
          <div>
            <p className="text-teal-400 text-[10px] font-bold uppercase tracking-widest">Clinical Assessment</p>
            <h1 className="text-2xl font-extrabold leading-tight text-white">New Patient Assessment</h1>
            <p className="text-slate-400 text-sm mt-0.5">
              Enter your patient's details to run the full AI diagnostic pipeline
            </p>
          </div>
        </div>
      </div>

      {/* Form Card */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="bg-card border border-border rounded-2xl shadow-lg p-6 md:p-8 space-y-6"
      >

        {/* Stepper */}
        <div className="flex items-center justify-center gap-2">
          {STEPS.map((label, idx) => {
            const n = idx + 1;
            return (
              <React.Fragment key={n}>
                {idx > 0 && <div className={`h-0.5 w-10 ${step >= n ? 'bg-primary' : 'bg-border'}`} />}
                <div className="flex flex-col items-center gap-1">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-xs border ${step >= n ? 'bg-primary text-white border-primary' : 'bg-background text-muted-foreground border-border'}`}>
                    {n}
                  </div>
                  <span className={`text-[9px] font-semibold ${step >= n ? 'text-primary' : 'text-muted-foreground'}`}>{label}</span>
                </div>
              </React.Fragment>
            );
          })}
        </div>

        {/* Error */}
        {errorMsg && (
          <div className="flex items-center gap-2 bg-red-50 border border-red-200 text-red-700 rounded-xl px-4 py-3 text-sm">
            <AlertCircle className="w-4 h-4 shrink-0" />
            {errorMsg}
          </div>
        )}

        {/* ─── STEP 1: Patient Info ─── */}
        {step === 1 && (
          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="space-y-5">
            <div className="flex items-center gap-2 mb-2">
              <User className="w-4 h-4 text-teal-600" />
              <h2 className="font-bold text-foreground">Patient Information</h2>
            </div>

            {/* Patient Name */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-foreground">Patient Full Name *</label>
              <input
                type="text"
                value={patientName}
                onChange={e => setPatientName(e.target.value)}
                placeholder="e.g., John Doe"
                className="w-full px-4 py-3 border border-input rounded-xl bg-background text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
              />
            </div>

            {/* DOB */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-foreground flex items-center gap-1.5">
                <Calendar className="w-3.5 h-3.5 text-teal-600" /> Date of Birth *
              </label>
              <input
                type="date"
                value={dateOfBirth}
                onChange={e => setDateOfBirth(e.target.value)}
                max={new Date().toISOString().split('T')[0]}
                className="w-full px-4 py-3 border border-input rounded-xl bg-background text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
              />
              {dateOfBirth && (
                <p className="text-xs text-muted-foreground">
                  Age: <span className="font-semibold text-foreground">{calculateAge(dateOfBirth)} years</span>
                </p>
              )}
            </div>

            {/* Gender */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-foreground">Biological Sex *</label>
              <div className="grid grid-cols-3 gap-3">
                {['Male', 'Female', 'Other'].map(g => (
                  <button key={g} onClick={() => setGender(g)}
                    className={`py-3 rounded-xl border text-sm font-semibold transition ${gender === g ? 'bg-primary text-white border-primary' : 'bg-background border-border text-muted-foreground hover:border-primary/50'}`}>
                    {g}
                  </button>
                ))}
              </div>
            </div>
          </motion.div>
        )}

        {/* ─── STEP 2: Symptoms ─── */}
        {step === 2 && (
          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="space-y-5">
            <div className="flex items-center gap-2 mb-2">
              <ClipboardList className="w-4 h-4 text-teal-600" />
              <h2 className="font-bold text-foreground">Chief Complaint & Symptoms</h2>
            </div>

            {/* Patient name reminder */}
            <div className="bg-teal-50 border border-teal-200 rounded-xl px-4 py-2.5 flex items-center gap-2">
              <User className="w-3.5 h-3.5 text-teal-600 shrink-0" />
              <span className="text-xs font-semibold text-teal-700">Patient: {patientName}</span>
            </div>

            {/* Chief complaint */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-foreground">Chief Complaint (in patient's words) *</label>
              <textarea
                value={chiefComplaint}
                onChange={e => setChiefComplaint(e.target.value)}
                rows={3}
                placeholder='e.g., "Patient presents with fever, body aches, and sore throat for 3 days..."'
                className="w-full px-4 py-3 border border-input rounded-xl bg-background text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-primary/40 resize-none"
              />
            </div>

            {/* Quick symptom picker */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-foreground">Select Symptoms</label>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {QUICK_SYMPTOMS.map(({ name, emoji }) => {
                  const selected = !!selectedSymptoms.find(s => s.name === name);
                  return (
                    <button key={name} onClick={() => toggleSymptom(name)}
                      className={`flex items-center gap-1.5 px-3 py-2 rounded-xl border text-xs font-medium transition ${selected ? 'bg-primary/10 border-primary text-primary' : 'bg-background border-border text-muted-foreground hover:border-primary/40'}`}>
                      <span>{emoji}</span>{name}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Duration for selected symptoms */}
            {selectedSymptoms.length > 0 && (
              <div className="space-y-2">
                <label className="text-xs font-semibold text-foreground">Symptom Duration</label>
                <div className="space-y-2">
                  {selectedSymptoms.map(s => (
                    <div key={s.name} className="flex items-center justify-between bg-primary/5 border border-primary/20 rounded-xl px-4 py-2.5">
                      <span className="text-xs font-semibold text-foreground">{s.name}</span>
                      <div className="flex items-center gap-2">
                        <button onClick={() => updateDuration(s.name, Math.max(1, s.duration_days - 1))}
                          className="w-6 h-6 rounded-full bg-background border border-border flex items-center justify-center text-xs font-bold">-</button>
                        <span className="text-xs font-bold text-foreground w-16 text-center">{s.duration_days} day{s.duration_days !== 1 ? 's' : ''}</span>
                        <button onClick={() => updateDuration(s.name, Math.min(365, s.duration_days + 1))}
                          className="w-6 h-6 rounded-full bg-background border border-border flex items-center justify-center text-xs font-bold">+</button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}

        {/* ─── STEP 3: Vitals & History ─── */}
        {step === 3 && (
          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="space-y-5">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-4 h-4 text-teal-600" />
              <h2 className="font-bold text-foreground">Vitals & Medical History</h2>
              <span className="text-[10px] text-muted-foreground ml-1">(all optional)</span>
            </div>

            {/* Patient name reminder */}
            <div className="bg-teal-50 border border-teal-200 rounded-xl px-4 py-2.5 flex items-center gap-2">
              <User className="w-3.5 h-3.5 text-teal-600 shrink-0" />
              <span className="text-xs font-semibold text-teal-700">Patient: {patientName} · Age: {calculateAge(dateOfBirth)} · {gender}</span>
            </div>

            {/* Vitals grid */}
            <div className="grid grid-cols-2 gap-4">
              {[
                { label: 'Heart Rate', unit: 'bpm', val: heartRate, set: setHeartRate, placeholder: '60–100' },
                { label: 'SpO2', unit: '%', val: oxygenLevel, set: setOxygenLevel, placeholder: '95–100' },
                { label: 'Systolic BP', unit: 'mmHg', val: systolicBp, set: setSystolicBp, placeholder: '90–120' },
                { label: 'Diastolic BP', unit: 'mmHg', val: diastolicBp, set: setDiastolicBp, placeholder: '60–80' },
                { label: 'Temperature', unit: '°F', val: temperature, set: setTemperature, placeholder: '98.6' },
                { label: 'Cholesterol', unit: 'mg/dL', val: cholesterol, set: setCholesterol, placeholder: '<200' },
              ].map(({ label, unit, val, set, placeholder }) => (
                <div key={label} className="space-y-1">
                  <label className="text-xs font-semibold text-foreground">{label} <span className="text-muted-foreground font-normal">({unit})</span></label>
                  <input type="number" value={val} onChange={e => set(e.target.value === '' ? '' : Number(e.target.value))}
                    placeholder={placeholder}
                    className="w-full px-3 py-2.5 border border-input rounded-xl bg-background text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-primary/40" />
                </div>
              ))}
            </div>

            {/* Medical history */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-foreground">Medical History</label>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {MEDICAL_HISTORY_OPTIONS.map(item => (
                  <button key={item} onClick={() => toggleArr(medicalHistory, setMedicalHistory, item)}
                    className={`text-left px-3 py-2 rounded-xl border text-xs font-medium transition ${medicalHistory.includes(item) ? 'bg-primary/10 border-primary text-primary' : 'bg-background border-border text-muted-foreground hover:border-primary/40'}`}>
                    {item}
                  </button>
                ))}
              </div>
            </div>

            {/* Lifestyle */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-foreground">Lifestyle Factors</label>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {LIFESTYLE_OPTIONS.map(item => (
                  <button key={item} onClick={() => toggleArr(lifestyleFactors, setLifestyleFactors, item)}
                    className={`text-left px-3 py-2 rounded-xl border text-xs font-medium transition ${lifestyleFactors.includes(item) ? 'bg-primary/10 border-primary text-primary' : 'bg-background border-border text-muted-foreground hover:border-primary/40'}`}>
                    {item}
                  </button>
                ))}
              </div>
            </div>
          </motion.div>
        )}

        {/* Navigation buttons */}
        <div className="flex items-center justify-between pt-2 border-t border-border">
          {step > 1 ? (
            <button onClick={() => { setErrorMsg(''); setStep(s => s - 1); }}
              className="inline-flex items-center gap-1.5 px-5 py-2.5 border border-border rounded-xl text-sm font-semibold text-muted-foreground hover:bg-muted transition">
              <ChevronLeft className="w-4 h-4" /> Back
            </button>
          ) : (
            <button onClick={() => navigate('/doctor/dashboard')}
              className="inline-flex items-center gap-1.5 px-5 py-2.5 border border-border rounded-xl text-sm font-semibold text-muted-foreground hover:bg-muted transition">
              <ChevronLeft className="w-4 h-4" /> Cancel
            </button>
          )}

          {step < 3 ? (
            <button onClick={handleNext}
              className="inline-flex items-center gap-1.5 px-6 py-2.5 bg-primary text-white rounded-xl text-sm font-bold hover:bg-primary/90 transition shadow-md">
              Continue <ChevronRight className="w-4 h-4" />
            </button>
          ) : (
            <button onClick={handleSubmit}
              className="inline-flex items-center gap-2 px-7 py-2.5 bg-gradient-to-r from-teal-500 to-sky-600 text-white rounded-xl text-sm font-bold shadow-lg shadow-teal-500/25 hover:opacity-90 transition">
              <Brain className="w-4 h-4" />
              Run AI Diagnosis
            </button>
          )}
        </div>
      </motion.div>
    </div>
  );
}
