import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import apiClient from '../../services/api-client';
import { useAuth } from '../../context/auth-context';

import PipelineVisualizer from '../../components/common/PipelineVisualizer';

export default function PatientIntake() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [currentStep, setCurrentStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [loadingStage, setLoadingStage] = useState(0);
  const [errorMsg, setErrorMsg] = useState('');

  // Form State
  const [age, setAge] = useState(35);
  const [gender, setGender] = useState('Male');
  const [chiefComplaint, setChiefComplaint] = useState('');
  
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
    { name: 'Joint Pain', emoji: '🦴' }
  ];

  const [selectedSymptoms, setSelectedSymptoms] = useState<{name: string, duration_days: number}[]>([]);
  
  // Vitals State
  const [heartRate, setHeartRate] = useState<number | ''>('');
  const [oxygenLevel, setOxygenLevel] = useState<number | ''>('');
  const [systolicBp, setSystolicBp] = useState<number | ''>('');
  const [diastolicBp, setDiastolicBp] = useState<number | ''>('');
  const [temperature, setTemperature] = useState<number | ''>('');
  const [cholesterol, setCholesterol] = useState<number | ''>('');

  // History & Lifestyle Checkboxes
  const [medicalHistory, setMedicalHistory] = useState<string[]>([]);
  const [lifestyleFactors, setLifestyleFactors] = useState<string[]>([]);

  const toggleSymptom = (name: string) => {
    const exists = selectedSymptoms.find(s => s.name === name);
    if (exists) {
      setSelectedSymptoms(selectedSymptoms.filter(s => s.name !== name));
    } else {
      setSelectedSymptoms([...selectedSymptoms, { name, duration_days: 3 }]);
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
      if (!age || age < 1 || age > 120) {
        setErrorMsg('Please enter a valid age between 1 and 120.');
        return;
      }
    } else if (currentStep === 2) {
      if (!chiefComplaint.trim()) {
        setErrorMsg('Please describe your chief complaint in your own words.');
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
    'Initializing Patient Intake workup...',
    'Running Symptom Analysis & synonym mapping...',
    'Calculating Differential Diagnosis probabilities...',
    'Performing clinical Risk Assessment scoring...',
    'Executing Temporal Analysis chronology checks...',
    'Analyzing Emergency Evaluation ESI urgency levels...',
    'Processing Prediction Engine voting ensemble...',
    'Compiling Recommendation Engine pharmacotherapy classes...',
    'Auditing Supervisor Review consensus parameters...',
    'Generating final secure Clinical Report. Redirecting...'
  ];

  const triggerLoaderCycle = (callback: () => void) => {
    setLoadingStage(0);
    const interval = setInterval(() => {
      setLoadingStage(prev => {
        if (prev >= loadingStages.length - 1) {
          clearInterval(interval);
          callback();
          return prev;
        }
        return prev + 1;
      });
    }, 900);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg('');
    
    // Final Validations
    if (!heartRate || heartRate < 30 || heartRate > 220) {
      setErrorMsg('Heart rate must be between 30 and 220 bpm.');
      return;
    }
    if (!oxygenLevel || oxygenLevel < 50 || oxygenLevel > 100) {
      setErrorMsg('Oxygen level (SpO2) must be between 50 and 100%.');
      return;
    }
    if (!systolicBp || systolicBp < 50 || systolicBp > 250) {
      setErrorMsg('Systolic BP must be between 50 and 250 mmHg.');
      return;
    }
    if (!diastolicBp || diastolicBp < 30 || diastolicBp > 150) {
      setErrorMsg('Diastolic BP must be between 30 and 150 mmHg.');
      return;
    }
    if (!temperature || temperature < 80.0 || temperature > 115.0) {
      setErrorMsg('Body temperature must be between 80.0°F and 115.0°F.');
      return;
    }
    if (!cholesterol || cholesterol < 50 || cholesterol > 600) {
      setErrorMsg('Cholesterol must be between 50 and 600 mg/dL.');
      return;
    }

    setIsSubmitting(true);
    
    const payload = {
      chief_complaint: chiefComplaint,
      selected_symptoms: selectedSymptoms,
      vitals: {
        heart_rate: Number(heartRate),
        oxygen_level: Number(oxygenLevel),
        systolic_bp: Number(systolicBp),
        diastolic_bp: Number(diastolicBp),
        temperature: Number(temperature),
        cholesterol: Number(cholesterol)
      },
      medical_history: medicalHistory,
      lifestyle_factors: lifestyleFactors
    };

    triggerLoaderCycle(async () => {
      try {
        const response = await apiClient.post('/patient/intake', payload);
        if (response.data && response.data.success) {
          const caseId = response.data.case.id;
          navigate(`/reports/${caseId}`);
        } else {
          setErrorMsg(response.data.message || 'An unexpected error occurred.');
          setIsSubmitting(false);
        }
      } catch (err: any) {
        console.error(err);
        const backendError = err.response?.data?.message || 'Failed to submit intake form. Please try again.';
        setErrorMsg(backendError);
        setIsSubmitting(false);
      }
    });
  };

  if (isSubmitting) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[70vh] p-4 max-w-2xl mx-auto text-center space-y-8">
        <div className="space-y-2">
          <h2 className="text-2xl font-bold tracking-tight text-slate-900">
            Executing Clinical Diagnostic Pipeline
          </h2>
          <p className="text-xs text-muted-foreground max-w-md mx-auto">
            Processing patient metrics across all clinical analysis engines. Do not close this browser window.
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

  return (
    <div className="max-w-2xl mx-auto bg-card border border-border rounded-2xl shadow-xl p-6 md:p-8 space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-extrabold text-foreground tracking-tight">Symptom Assessment</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Provide your symptoms and vital signs to request a clinical diagnostic workup suggestion.
        </p>
      </div>

      {/* Stepper Header */}
      <div className="flex items-center justify-center space-x-2">
        <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-xs border ${currentStep >= 1 ? 'bg-primary text-white border-primary' : 'bg-transparent text-muted-foreground border-border'}`}>1</div>
        <div className={`h-0.5 w-12 ${currentStep >= 2 ? 'bg-primary' : 'bg-border'}`}></div>
        <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-xs border ${currentStep >= 2 ? 'bg-primary text-white border-primary' : 'bg-transparent text-muted-foreground border-border'}`}>2</div>
        <div className={`h-0.5 w-12 ${currentStep >= 3 ? 'bg-primary' : 'bg-border'}`}></div>
        <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-xs border ${currentStep >= 3 ? 'bg-primary text-white border-primary' : 'bg-transparent text-muted-foreground border-border'}`}>3</div>
      </div>

      {/* Error Alert Box */}
      {errorMsg && (
        <div className="bg-red-50 text-red-600 border border-red-200 p-4 rounded-xl text-sm font-medium">
          {errorMsg}
        </div>
      )}

      {/* Step Contents */}
      <form onSubmit={handleSubmit} className="space-y-6">
        {currentStep === 1 && (
          <div className="space-y-4">
            <h2 className="text-lg font-bold">Step 1: Patient Information</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Full Name</label>
                <input 
                  type="text" 
                  value={user?.email.split('@')[0] || ''} 
                  disabled 
                  className="w-full bg-slate-100 border border-border rounded-xl px-4 py-2.5 text-sm cursor-not-allowed opacity-75"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Email Address</label>
                <input 
                  type="email" 
                  value={user?.email || ''} 
                  disabled 
                  className="w-full bg-slate-100 border border-border rounded-xl px-4 py-2.5 text-sm cursor-not-allowed opacity-75"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Age (Years)</label>
                <input 
                  type="number" 
                  value={age} 
                  onChange={e => setAge(Number(e.target.value))}
                  min="1" 
                  max="120"
                  required
                  className="w-full bg-transparent border border-border rounded-xl px-4 py-2.5 text-sm"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Gender</label>
                <select 
                  value={gender} 
                  onChange={e => setGender(e.target.value)}
                  className="w-full bg-transparent border border-border rounded-xl px-4 py-2.5 text-sm"
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {currentStep === 2 && (
          <div className="space-y-4">
            <h2 className="text-lg font-bold">Step 2: Describe Your Symptoms</h2>
            
            <div>
              <label className="block text-sm font-medium mb-1.5">What are you feeling? (Free-text description)</label>
              <textarea 
                value={chiefComplaint}
                onChange={e => setChiefComplaint(e.target.value)}
                placeholder="I have had fever, cough, and body aches for the last 5 days..."
                required
                rows={4}
                className="w-full bg-transparent border border-border rounded-xl px-4 py-2.5 text-sm"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Quick Symptoms Selector (Click to add)</label>
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
                <label className="block text-sm font-medium mb-2">Symptom Durations</label>
                <div className="space-y-2">
                  {selectedSymptoms.map(s => (
                    <div key={s.name} className="flex items-center justify-between p-2.5 bg-slate-50 border border-border rounded-xl">
                      <span className="text-sm font-medium text-foreground">{s.name}</span>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-muted-foreground">Duration (days):</span>
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
              <h2 className="text-lg font-bold">Step 3: Vital Signs & History</h2>
              <p className="text-xs text-muted-foreground mt-0.5">Please provide clinical vital measurements.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider mb-1">Heart Rate (bpm)</label>
                <input 
                  type="number" 
                  value={heartRate}
                  onChange={e => setHeartRate(e.target.value === '' ? '' : Number(e.target.value))}
                  placeholder="Normal: 60 - 100"
                  required
                  className="w-full bg-transparent border border-border rounded-xl px-4 py-2.5 text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-bold uppercase tracking-wider mb-1">SpO2 / Oxygen (%)</label>
                <input 
                  type="number" 
                  value={oxygenLevel}
                  onChange={e => setOxygenLevel(e.target.value === '' ? '' : Number(e.target.value))}
                  placeholder="Normal: 95 - 100"
                  required
                  className="w-full bg-transparent border border-border rounded-xl px-4 py-2.5 text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-bold uppercase tracking-wider mb-1">Temperature (°F)</label>
                <input 
                  type="number" 
                  step="0.1"
                  value={temperature}
                  onChange={e => setTemperature(e.target.value === '' ? '' : Number(e.target.value))}
                  placeholder="Normal: 97 - 99"
                  required
                  className="w-full bg-transparent border border-border rounded-xl px-4 py-2.5 text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-bold uppercase tracking-wider mb-1">Systolic BP (mmHg)</label>
                <input 
                  type="number" 
                  value={systolicBp}
                  onChange={e => setSystolicBp(e.target.value === '' ? '' : Number(e.target.value))}
                  placeholder="e.g. 120"
                  required
                  className="w-full bg-transparent border border-border rounded-xl px-4 py-2.5 text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-bold uppercase tracking-wider mb-1">Diastolic BP (mmHg)</label>
                <input 
                  type="number" 
                  value={diastolicBp}
                  onChange={e => setDiastolicBp(e.target.value === '' ? '' : Number(e.target.value))}
                  placeholder="e.g. 80"
                  required
                  className="w-full bg-transparent border border-border rounded-xl px-4 py-2.5 text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-bold uppercase tracking-wider mb-1">Cholesterol (mg/dL)</label>
                <input 
                  type="number" 
                  value={cholesterol}
                  onChange={e => setCholesterol(e.target.value === '' ? '' : Number(e.target.value))}
                  placeholder="Normal: < 200"
                  required
                  className="w-full bg-transparent border border-border rounded-xl px-4 py-2.5 text-sm"
                />
              </div>
            </div>

            {/* Medical History Section */}
            <div className="border-t border-border pt-4">
              <label className="block text-sm font-bold mb-2">Medical History (Check all that apply)</label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {['Diabetes', 'Hypertension', 'Asthma', 'Heart Disease', 'Kidney Disease', 'Thyroid'].map(item => (
                  <label key={item} className="flex items-center space-x-2 text-sm text-foreground/80 cursor-pointer">
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
              <label className="block text-sm font-bold mb-2">Lifestyle Risks (Check all that apply)</label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {['Smoking', 'Alcohol', 'Obesity', 'Sedentary Lifestyle', 'High Stress'].map(item => (
                  <label key={item} className="flex items-center space-x-2 text-sm text-foreground/80 cursor-pointer">
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
              Next Step →
            </button>
          ) : (
            <button
              type="submit"
              className="px-6 py-2.5 bg-primary text-primary-foreground hover:opacity-90 rounded-xl text-sm font-semibold shadow-lg shadow-primary/20 transition ml-auto"
            >
              🔍 Run Diagnostics
            </button>
          )}
        </div>
      </form>
    </div>
  );
}
