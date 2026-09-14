import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Sparkles, Zap, FileText } from 'lucide-react';
import apiClient from '../../services/api-client';
import { useAuth } from '../../context/auth-context';

import PipelineVisualizer from '../../components/common/PipelineVisualizer';

export default function PatientIntake() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();

  const ocrPrefill = location.state?.prefilledFromOCR;
  const isAutoRun = Boolean(location.state?.autoRun);

  const [currentStep, setCurrentStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [loadingStage, setLoadingStage] = useState(0);
  const [errorMsg, setErrorMsg] = useState('');

  // Helper to calculate age from DOB
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

  // Compute default age/DOB from OCR if needed
  let calculatedDobFromOCR = '';
  if (ocrPrefill?.patient_age) {
    const birthYear = new Date().getFullYear() - Number(ocrPrefill.patient_age);
    calculatedDobFromOCR = `${birthYear}-01-01`;
  }

  // Parse blood pressure if provided as e.g. "120/80"
  let defaultSys: number | '' = '';
  let defaultDia: number | '' = '';
  if (ocrPrefill?.blood_pressure_reading) {
    const parts = String(ocrPrefill.blood_pressure_reading).split('/');
    if (parts.length === 2) {
      defaultSys = parseInt(parts[0].replace(/\D/g, ''), 10) || '';
      defaultDia = parseInt(parts[1].replace(/\D/g, ''), 10) || '';
    }
  }

  // Pre-fill from user profile (from registration) or OCR findings
  const [name, setName] = useState(user?.name || ocrPrefill?.patient_name || '');
  const [dateOfBirth, setDateOfBirth] = useState(user?.date_of_birth || calculatedDobFromOCR);
  const [gender, setGender] = useState(ocrPrefill?.patient_gender || user?.gender || '');
  const [chiefComplaint, setChiefComplaint] = useState(
    ocrPrefill?.patient_text || ocrPrefill?.document_findings || ''
  );
  
  // Quick symptom selector list
  const quickSymptoms = [
    { name: 'Fever', emoji: '🔥' },
    { name: 'Cough', emoji: '😷' },
    { name: 'Fatigue', emoji: '😫' },
    { name: 'Headache', emoji: '🤕' },
    { name: 'Chest Pain', emoji: '💔' },
    { name: 'Breathlessness', emoji: '😤' },
    { name: 'Nausea', emoji: '🤢' },
    { name: 'Vomiting', emoji: '🤮' },
    { name: 'Body Pain', emoji: '💪' },
    { name: 'Joint Pain', emoji: '🦴' },
    { name: 'Sore Throat', emoji: '🗣️' },
    { name: 'Runny Nose', emoji: '🤧' },
    { name: 'Dizziness', emoji: '😵' },
    { name: 'Rash', emoji: '🔴' },
  ];

  const [selectedSymptoms, setSelectedSymptoms] = useState<{name: string, duration_days: number}[]>([]);
  
  // Vitals State — ALL OPTIONAL
  const [heartRate, setHeartRate] = useState<number | ''>(ocrPrefill?.heart_rate || '');
  const [oxygenLevel, setOxygenLevel] = useState<number | ''>(ocrPrefill?.oxygen_level || '');
  const [systolicBp, setSystolicBp] = useState<number | ''>(defaultSys);
  const [diastolicBp, setDiastolicBp] = useState<number | ''>(defaultDia);
  const [temperature, setTemperature] = useState<number | ''>(ocrPrefill?.body_temperature || '');
  const [cholesterol, setCholesterol] = useState<number | ''>(ocrPrefill?.cholesterol || '');
  const [showAdvancedVitals, setShowAdvancedVitals] = useState(Boolean(defaultSys || defaultDia || ocrPrefill?.cholesterol));

  // History & Lifestyle Checkboxes
  const [medicalHistory, setMedicalHistory] = useState<string[]>(
    Array.isArray(ocrPrefill?.medical_history) ? ocrPrefill.medical_history : []
  );
  const [lifestyleFactors, setLifestyleFactors] = useState<string[]>([]);

  const toggleSymptom = (name: string) => {
    const exists = selectedSymptoms.find(s => s.name === name);
    if (exists) {
      setSelectedSymptoms(selectedSymptoms.filter(s => s.name !== name));
    } else {
      setSelectedSymptoms([...selectedSymptoms, { name, duration_days: 1 }]);
    }
  };

  const updateSymptomDuration = (name: string, days: number) => {
    setSelectedSymptoms(selectedSymptoms.map(s => s.name === name ? { ...s, duration_days: days } : s));
  };

  const toggleHistory = (name: string) => {
    if (medicalHistory.includes(name)) {
      setMedicalHistory(medicalHistory.filter(h => h !== name));
    } else {
      setMedicalHistory([...medicalHistory, name]);
    }
  };

  const toggleLifestyle = (name: string) => {
    if (lifestyleFactors.includes(name)) {
      setLifestyleFactors(lifestyleFactors.filter(l => l !== name));
    } else {
      setLifestyleFactors([...lifestyleFactors, name]);
    }
  };

  const handleNext = () => {
    setErrorMsg('');
    if (currentStep === 1) {
      if (!name.trim()) {
        setErrorMsg('Please enter your name.');
        return;
      }
      if (!dateOfBirth) {
        setErrorMsg('Please select your date of birth.');
        return;
      }
      const computedAge = calculateAge(dateOfBirth);
      if (computedAge === null || computedAge < 0 || computedAge > 120) {
        setErrorMsg('Please enter a valid date of birth.');
        return;
      }
      if (!gender) {
        setErrorMsg('Please select your gender.');
        return;
      }
    } else if (currentStep === 2) {
      if (!chiefComplaint.trim()) {
        setErrorMsg('Please describe what you are feeling in your own words.');
        return;
      }
    }
    setCurrentStep(prev => prev + 1);
  };

  const handleBack = () => {
    setErrorMsg('');
    setCurrentStep(prev => prev - 1);
  };

  const loadingStages = [
    'Starting your health assessment...',
    'Extracting symptoms with ClinicalBERT...',
    'Running differential diagnosis engine...',
    'Evaluating risk factors...',
    'Analyzing symptom timeline...',
    'Checking for emergency indicators...',
    'Running AI prediction models...',
    'Invoking Meditron-7B clinical reasoning...',
    'Generating differential analysis...',
    'Finding recommended treatments...',
    'Cross-referencing drug database...',
    'Compiling your health report...',
    'Running final quality checks...',
    'Almost done — preparing your results...',
  ];

  const executeSubmission = async (override?: {
    name?: string;
    dob?: string;
    gender?: string;
    complaint?: string;
    vitals?: Record<string, number>;
  }) => {
    setErrorMsg('');

    const targetName = override?.name ?? name;
    const targetDob = override?.dob ?? dateOfBirth;
    const targetGender = override?.gender ?? gender;
    const targetComplaint = override?.complaint ?? chiefComplaint;

    if (!targetComplaint?.trim()) {
      setErrorMsg('Please describe what you are feeling or the clinical issue.');
      return;
    }

    // Only validate vitals IF they have values (all optional)
    if (heartRate !== '' && (Number(heartRate) < 30 || Number(heartRate) > 220)) {
      setErrorMsg('Heart rate must be between 30 and 220 bpm.');
      return;
    }
    if (oxygenLevel !== '' && (Number(oxygenLevel) < 50 || Number(oxygenLevel) > 100)) {
      setErrorMsg('Oxygen level must be between 50 and 100%.');
      return;
    }
    if (systolicBp !== '' && (Number(systolicBp) < 60 || Number(systolicBp) > 250)) {
      setErrorMsg('Systolic blood pressure must be between 60 and 250 mmHg.');
      return;
    }
    if (diastolicBp !== '' && (Number(diastolicBp) < 30 || Number(diastolicBp) > 150)) {
      setErrorMsg('Diastolic blood pressure must be between 30 and 150 mmHg.');
      return;
    }
    if (temperature !== '' && (Number(temperature) < 90 || Number(temperature) > 110)) {
      setErrorMsg('Temperature must be between 90 and 110 °F.');
      return;
    }

    setIsSubmitting(true);
    setLoadingStage(0);

    // Start loading animation — loops continuously until API responds
    const loaderInterval = setInterval(() => {
      setLoadingStage(prev => (prev + 1) % loadingStages.length);
    }, 2500);
    
    const vitals: Record<string, number> = override?.vitals || {};
    if (!override?.vitals) {
      if (heartRate !== '') vitals.heart_rate = Number(heartRate);
      if (oxygenLevel !== '') vitals.oxygen_level = Number(oxygenLevel);
      if (systolicBp !== '') vitals.systolic_bp = Number(systolicBp);
      if (diastolicBp !== '') vitals.diastolic_bp = Number(diastolicBp);
      if (temperature !== '') vitals.temperature = Number(temperature);
      if (cholesterol !== '') vitals.cholesterol = Number(cholesterol);
    }

    const computedAge = calculateAge(targetDob) || (ocrPrefill?.patient_age ? Number(ocrPrefill.patient_age) : 35);

    const payload = {
      age: computedAge,
      gender: targetGender || 'Other',
      chief_complaint: targetComplaint,
      selected_symptoms: selectedSymptoms,
      vitals: vitals,
      medical_history: medicalHistory,
      lifestyle_factors: lifestyleFactors
    };

    // Fire API call IMMEDIATELY (runs in parallel with animation)
    try {
      const response = await apiClient.post('/patient/intake', payload, {
        timeout: 300000, // 5 minutes — Meditron LLM inference can take time
      });
      clearInterval(loaderInterval);
      if (response.data && response.data.success) {
        const caseId = response.data.case.id;
        navigate(`/reports/${caseId}`);
      } else {
        setErrorMsg(response.data.message || 'An unexpected error occurred.');
        setIsSubmitting(false);
      }
    } catch (err: any) {
      clearInterval(loaderInterval);
      console.error(err);
      const backendError = err.response?.data?.message || 'Failed to submit. Please try again.';
      setErrorMsg(backendError);
      setIsSubmitting(false);
    }
  };

  const handleSubmit = () => executeSubmission();

  useEffect(() => {
    if (isAutoRun && !isSubmitting) {
      const activeName = name || user?.name || ocrPrefill?.patient_name || 'Patient';
      const activeDob = dateOfBirth || calculatedDobFromOCR || '1990-01-01';
      const activeGender = gender || ocrPrefill?.patient_gender || 'Other';
      const activeComplaint = chiefComplaint || ocrPrefill?.patient_text || ocrPrefill?.document_findings || 'Consultation requested from clinical report findings.';

      const autoVitals: Record<string, number> = {};
      if (heartRate !== '') autoVitals.heart_rate = Number(heartRate);
      if (oxygenLevel !== '') autoVitals.oxygen_level = Number(oxygenLevel);
      if (systolicBp !== '') autoVitals.systolic_bp = Number(systolicBp);
      if (diastolicBp !== '') autoVitals.diastolic_bp = Number(diastolicBp);
      if (temperature !== '') autoVitals.temperature = Number(temperature);
      if (cholesterol !== '') autoVitals.cholesterol = Number(cholesterol);

      executeSubmission({
        name: activeName,
        dob: activeDob,
        gender: activeGender,
        complaint: activeComplaint,
        vitals: autoVitals,
      });
    }
  }, []);

  if (isSubmitting) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[70vh] p-4 max-w-2xl mx-auto text-center space-y-8">
        <div className="space-y-2">
          <h2 className="text-2xl font-bold tracking-tight text-slate-900">
            Analyzing Your Health Data
          </h2>
          <p className="text-xs text-muted-foreground max-w-md mx-auto">
            Our AI is processing your symptoms through 8 specialist agents and Meditron-7B clinical reasoning. This may take 1-3 minutes.
          </p>
        </div>

        {/* The 10-stage visualizer */}
        <div className="w-full bg-card border border-border p-6 rounded-2xl shadow-md">
          <PipelineVisualizer activeStage={loadingStage} />
        </div>

        <div className="w-full max-w-md bg-slate-100 h-2.5 rounded-full overflow-hidden shadow-inner border border-border">
          <div 
            className="bg-primary h-full transition-all duration-300 ease-out" 
            style={{ width: `${((loadingStage + 1) / loadingStages.length) * 100}%` }}
          ></div>
        </div>

        <p className="text-sky-700 text-sm font-semibold h-8 animate-pulse">
          {loadingStages[loadingStage]}
        </p>
      </div>
    );
  }

  const stepLabels = ['Your Info', 'Symptoms', 'Vitals & History'];

  return (
    <div className="max-w-2xl mx-auto bg-card border border-border rounded-2xl shadow-xl p-6 md:p-8 space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-extrabold text-foreground tracking-tight">Health Assessment</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Tell us about your symptoms and our AI will help you understand what might be going on.
        </p>
      </div>

      {/* OCR Ingestion Indicator Banner */}
      {ocrPrefill && (
        <div className="bg-gradient-to-r from-teal-50 to-sky-50 border border-teal-200 rounded-2xl p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow-sm">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-teal-500/20 text-teal-700 flex items-center justify-center shrink-0">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <p className="text-xs font-bold text-slate-800">
                Data Ingested from Medical Report (OCRAgent)
              </p>
              <p className="text-[11px] text-slate-500">
                Extracted vitals, diagnoses, and lab findings have pre-filled this clinical assessment.
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => executeSubmission()}
            className="w-full sm:w-auto px-4 py-2 bg-gradient-to-r from-teal-600 to-sky-600 hover:opacity-95 text-white font-bold text-xs rounded-xl shadow-sm transition whitespace-nowrap flex items-center justify-center gap-2"
          >
            <Zap className="w-3.5 h-3.5" />
            <span>Launch Consultation Now</span>
          </button>
        </div>
      )}

      {/* Stepper Header with Labels */}
      <div className="flex items-center justify-center space-x-2">
        {stepLabels.map((label, idx) => {
          const stepNum = idx + 1;
          return (
            <React.Fragment key={stepNum}>
              {idx > 0 && (
                <div className={`h-0.5 w-10 ${currentStep >= stepNum ? 'bg-primary' : 'bg-border'}`}></div>
              )}
              <div className="flex flex-col items-center gap-1">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-xs border ${currentStep >= stepNum ? 'bg-primary text-white border-primary' : 'bg-transparent text-muted-foreground border-border'}`}>
                  {stepNum}
                </div>
                <span className={`text-[9px] font-semibold ${currentStep >= stepNum ? 'text-primary' : 'text-muted-foreground'}`}>{label}</span>
              </div>
            </React.Fragment>
          );
        })}
      </div>

      {/* Error Alert Box */}
      {errorMsg && (
        <div className="bg-red-50 text-red-600 border border-red-200 p-4 rounded-xl text-sm font-medium">
          {errorMsg}
        </div>
      )}

      {/* Step Contents */}
      <div className="space-y-6">
        {currentStep === 1 && (
          <div className="space-y-4">
            <h2 className="text-lg font-bold">About You</h2>
            <p className="text-xs text-muted-foreground -mt-2">Let us know who you are so we can personalize your assessment.</p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="md:col-span-2">
                <label className="block text-sm font-medium mb-1">Full Name <span className="text-red-500">*</span></label>
                <input 
                  type="text" 
                  value={name}
                  onChange={e => setName(e.target.value)}
                  placeholder="Enter your full name"
                  required
                  className="w-full bg-transparent border border-border rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-primary/20 focus:border-primary transition"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Date of Birth <span className="text-red-500">*</span></label>
                <input 
                  type="date" 
                  value={dateOfBirth}
                  onChange={e => setDateOfBirth(e.target.value)}
                  max={new Date().toISOString().split('T')[0]}
                  required
                  className="w-full bg-transparent border border-border rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-primary/20 focus:border-primary transition"
                />
                {dateOfBirth && calculateAge(dateOfBirth) !== null && (
                  <p className="text-xs text-secondary font-semibold mt-1">Age: {calculateAge(dateOfBirth)} years</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Gender <span className="text-red-500">*</span></label>
                <select 
                  value={gender} 
                  onChange={e => setGender(e.target.value)}
                  required
                  className="w-full bg-transparent border border-border rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-primary/20 focus:border-primary transition"
                >
                  <option value="">Select gender</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {currentStep === 2 && (
          <div className="space-y-4">
            <h2 className="text-lg font-bold">What Are You Feeling?</h2>
            <p className="text-xs text-muted-foreground -mt-2">Describe your symptoms in your own words, then select any that apply below.</p>
            
            <div>
              <label className="block text-sm font-medium mb-1.5">Describe in your own words <span className="text-red-500">*</span></label>
              <textarea 
                value={chiefComplaint}
                onChange={e => setChiefComplaint(e.target.value)}
                placeholder="Example: I've been having fever and a bad cough for the last 3 days. I also feel very tired and have body aches..."
                required
                rows={4}
                className="w-full bg-transparent border border-border rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-primary/20 focus:border-primary transition"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Select common symptoms (tap to add)</label>
              <div className="flex flex-wrap gap-2">
                {quickSymptoms.map(s => {
                  const isSelected = !!selectedSymptoms.find(item => item.name === s.name);
                  return (
                    <button
                      type="button"
                      key={s.name}
                      onClick={() => toggleSymptom(s.name)}
                      className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-full text-xs font-semibold border transition ${
                        isSelected 
                          ? 'bg-primary text-white border-primary' 
                          : 'bg-transparent border-border hover:bg-slate-50'
                      }`}
                    >
                      <span>{s.emoji}</span>
                      <span>{s.name}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {selectedSymptoms.length > 0 && (
              <div className="space-y-2 border-t border-border pt-4">
                <label className="block text-sm font-medium mb-2">How long have you had each symptom?</label>
                <div className="space-y-2">
                  {selectedSymptoms.map(s => (
                    <div key={s.name} className="flex items-center justify-between p-2.5 bg-slate-50 border border-border rounded-xl">
                      <span className="text-sm font-medium text-foreground">{s.name}</span>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-muted-foreground">Days:</span>
                        <input
                          type="number"
                          min="1"
                          max="90"
                          value={s.duration_days}
                          onChange={e => updateSymptomDuration(s.name, Number(e.target.value))}
                          className="w-16 bg-transparent border border-border rounded-lg px-2 py-1 text-xs text-center"
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {currentStep === 3 && (
          <div className="space-y-6">
            <div>
              <h2 className="text-lg font-bold">Additional Information</h2>
              <p className="text-xs text-muted-foreground mt-0.5">
                This section is <strong>completely optional</strong>. If you know any of your vital signs, entering them helps our AI give more accurate results. If not, just skip ahead.
              </p>
            </div>

            {/* Common Vitals */}
            <div className="bg-sky-50/50 border border-sky-100 rounded-xl p-4 space-y-3">
              <h3 className="text-xs font-bold text-sky-800 uppercase tracking-wider">Vital Signs (Optional)</h3>
              <p className="text-[10px] text-sky-700">If you have a thermometer, BP monitor, or pulse oximeter at home, enter the readings. Otherwise, skip this.</p>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Heart Rate (bpm)</label>
                  <input 
                    type="number" 
                    value={heartRate}
                    onChange={e => setHeartRate(e.target.value === '' ? '' : Number(e.target.value))}
                    placeholder="e.g. 72"
                    className="w-full bg-white border border-border rounded-xl px-3 py-2 text-sm"
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Oxygen Level (%)</label>
                  <input 
                    type="number" 
                    value={oxygenLevel}
                    onChange={e => setOxygenLevel(e.target.value === '' ? '' : Number(e.target.value))}
                    placeholder="e.g. 98"
                    className="w-full bg-white border border-border rounded-xl px-3 py-2 text-sm"
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Temperature (°F)</label>
                  <input 
                    type="number" 
                    step="0.1"
                    value={temperature}
                    onChange={e => setTemperature(e.target.value === '' ? '' : Number(e.target.value))}
                    placeholder="e.g. 98.6"
                    className="w-full bg-white border border-border rounded-xl px-3 py-2 text-sm"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Systolic BP (mmHg)</label>
                  <input 
                    type="number" 
                    value={systolicBp}
                    onChange={e => setSystolicBp(e.target.value === '' ? '' : Number(e.target.value))}
                    placeholder="e.g. 120"
                    className="w-full bg-white border border-border rounded-xl px-3 py-2 text-sm"
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Diastolic BP (mmHg)</label>
                  <input 
                    type="number" 
                    value={diastolicBp}
                    onChange={e => setDiastolicBp(e.target.value === '' ? '' : Number(e.target.value))}
                    placeholder="e.g. 80"
                    className="w-full bg-white border border-border rounded-xl px-3 py-2 text-sm"
                  />
                </div>
              </div>
            </div>

            {/* Advanced Vitals (Cholesterol) — Collapsible */}
            <div className="border border-border rounded-xl overflow-hidden">
              <button
                type="button"
                onClick={() => setShowAdvancedVitals(!showAdvancedVitals)}
                className="w-full flex items-center justify-between px-4 py-3 bg-slate-50 text-left text-xs font-semibold text-slate-700 hover:bg-slate-100 transition"
              >
                <span>Additional Measurements (if you have lab results)</span>
                <span className="text-muted-foreground">{showAdvancedVitals ? '▲' : '▼'}</span>
              </button>
              {showAdvancedVitals && (
                <div className="px-4 py-3 space-y-3">
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">Cholesterol (mg/dL)</label>
                    <input 
                      type="number" 
                      value={cholesterol}
                      onChange={e => setCholesterol(e.target.value === '' ? '' : Number(e.target.value))}
                      placeholder="e.g. 180"
                      className="w-full bg-transparent border border-border rounded-xl px-3 py-2 text-sm"
                    />
                    <p className="text-[10px] text-muted-foreground mt-1">Usually from a recent blood test report.</p>
                  </div>
                </div>
              )}
            </div>

            {/* Medical History Section */}
            <div className="border-t border-border pt-4">
              <label className="block text-sm font-bold mb-2">Do you have any of these conditions?</label>
              <p className="text-[10px] text-muted-foreground mb-3">Select any conditions you've been diagnosed with before.</p>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {['Diabetes', 'Hypertension', 'Asthma', 'Heart Disease', 'Kidney Disease', 'Thyroid', 'Cancer', 'Liver Disease', 'Arthritis'].map(item => (
                  <label key={item} className="flex items-center space-x-2 text-sm text-foreground/80 cursor-pointer p-1.5 rounded-lg hover:bg-slate-50 transition">
                    <input 
                      type="checkbox" 
                      checked={medicalHistory.includes(item)}
                      onChange={() => toggleHistory(item)}
                      className="rounded border-border text-primary focus:ring-primary w-4 h-4" 
                    />
                    <span>{item}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Lifestyle Factors Section */}
            <div className="border-t border-border pt-4">
              <label className="block text-sm font-bold mb-2">Lifestyle factors</label>
              <p className="text-[10px] text-muted-foreground mb-3">These help us better understand your overall health profile.</p>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {['Smoking', 'Alcohol', 'Obesity', 'Sedentary Lifestyle', 'High Stress', 'Poor Diet'].map(item => (
                  <label key={item} className="flex items-center space-x-2 text-sm text-foreground/80 cursor-pointer p-1.5 rounded-lg hover:bg-slate-50 transition">
                    <input 
                      type="checkbox" 
                      checked={lifestyleFactors.includes(item)}
                      onChange={() => toggleLifestyle(item)}
                      className="rounded border-border text-primary focus:ring-primary w-4 h-4" 
                    />
                    <span>{item}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Footer Navigation Buttons */}
        <div className="flex justify-between border-t border-border pt-6">
          {currentStep > 1 ? (
            <button
              type="button"
              onClick={handleBack}
              className="px-6 py-2.5 border border-border text-foreground hover:bg-slate-50 rounded-xl text-sm font-medium transition"
            >
              ← Back
            </button>
          ) : (
            <div></div>
          )}

          {currentStep < 3 ? (
            <button
              type="button"
              onClick={handleNext}
              className="px-6 py-2.5 bg-primary text-primary-foreground hover:opacity-90 rounded-xl text-sm font-medium transition ml-auto"
            >
              Continue →
            </button>
          ) : (
            <button
              type="button"
              onClick={handleSubmit}
              className="px-6 py-2.5 bg-primary text-primary-foreground hover:opacity-90 rounded-xl text-sm font-semibold shadow-lg shadow-primary/20 transition ml-auto"
            >
              🔍 Analyze My Symptoms
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
