import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import apiClient from '../../services/api-client';
import { useToast } from '../../context/toast-context';
import { 
  Users, Search, Filter, User, Calendar, Mail, Phone, 
  ChevronRight, Heart, Thermometer, Wind, Activity, X, FileText
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface CaseRecord {
  id: number;
  patient_name: string;
  status: string;
  triage_level: number;
  created_at: string;
  chief_complaint: string;
}

interface PatientProfile {
  name: string;
  age: number;
  gender: string;
  email: string;
  phone: string;
  bloodType: string;
  comorbidities: string[];
  lifestyleFactors: string[];
  cases: CaseRecord[];
  latestVitals: {
    hr: number;
    bp: string;
    temp: number;
    spo2: number;
    cholesterol: number;
  };
}

export default function DoctorTriage() {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState('');
  
  const [cases, setCases] = useState<CaseRecord[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [genderFilter, setGenderFilter] = useState('all');
  
  const [selectedPatient, setSelectedPatient] = useState<PatientProfile | null>(null);

  const fetchPatientsData = async () => {
    try {
      setLoading(true);
      const response = await apiClient.get('/doctor/cases');
      if (response.data && response.data.success) {
        setCases(response.data.cases);
      }
    } catch (err: any) {
      console.error(err);
      setErrorMsg('Failed to sync patient roster.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPatientsData();
  }, []);

  // Construct patient profiles from case logs
  const getPatientProfiles = (): PatientProfile[] => {
    const patientMap = new Map<string, CaseRecord[]>();
    
    cases.forEach(c => {
      if (!patientMap.has(c.patient_name)) {
        patientMap.set(c.patient_name, []);
      }
      patientMap.get(c.patient_name)!.push(c);
    });

    const profiles: PatientProfile[] = [];

    patientMap.forEach((patientCases, name) => {
      // Simulate static profile data for academic representation
      const isMale = name.toLowerCase().includes('rahul') || name.toLowerCase().includes('doe') || Math.random() > 0.5;
      const calculatedAge = name.toLowerCase().includes('rahul') ? 35 : isMale ? 42 : 29;
      const email = `${name.toLowerCase().replace(/\s+/g, '')}@clinical-emr.org`;
      
      const comorbidities =CalculatedComorbidities(patientCases);
      const lifestyleFactors =CalculatedLifestyle(patientCases);

      profiles.push({
        name,
        age: calculatedAge,
        gender: isMale ? 'Male' : 'Female',
        email,
        phone: '+91 98450 12894',
        bloodType: isMale ? 'O+' : 'A-',
        comorbidities,
        lifestyleFactors,
        cases: patientCases,
        latestVitals: {
          hr: patientCases[0]?.triage_level === 1 ? 112 : 74,
          bp: patientCases[0]?.triage_level === 1 ? '145/95' : '120/80',
          temp: patientCases[0]?.triage_level === 1 ? 101.5 : 98.4,
          spo2: patientCases[0]?.triage_level === 1 ? 92 : 98,
          cholesterol: 185
        }
      });
    });

    // Fallback: Default patient if db is empty
    if (profiles.length === 0) {
      profiles.push({
        name: 'Rahul Sharma',
        age: 35,
        gender: 'Male',
        email: 'rahulsharma@clinical-emr.org',
        phone: '+91 98450 12894',
        bloodType: 'O+',
        comorbidities: ['Hypertension'],
        lifestyleFactors: ['High Stress', 'Sedentary Lifestyle'],
        cases: [],
        latestVitals: {
          hr: 72,
          bp: '120/80',
          temp: 98.6,
          spo2: 99,
          cholesterol: 190
        }
      });
    }

    return profiles;
  };

  const CalculatedComorbidities = (cases: CaseRecord[]) => {
    const list: string[] = [];
    cases.forEach(c => {
      const complaint = c.chief_complaint.toLowerCase();
      if (complaint.includes('chest') || complaint.includes('bp')) {
        if (!list.includes('Hypertension')) list.push('Hypertension');
      }
      if (complaint.includes('sugar') || complaint.includes('diabetes')) {
        if (!list.includes('Diabetes Type II')) list.push('Diabetes Type II');
      }
    });
    if (list.length === 0) list.push('None declared');
    return list;
  };

  const CalculatedLifestyle = (cases: CaseRecord[]) => {
    const list: string[] = [];
    cases.forEach(c => {
      const complaint = c.chief_complaint.toLowerCase();
      if (complaint.includes('chest')) {
        if (!list.includes('High Stress')) list.push('High Stress');
      }
    });
    if (list.length === 0) list.push('Sedentary Lifestyle');
    return list;
  };

  const patientProfiles = getPatientProfiles();

  // Search & Filter Logic
  const filteredPatients = patientProfiles.filter(p => {
    const matchesSearch = p.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
                          p.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          p.comorbidities.some(c => c.toLowerCase().includes(searchTerm.toLowerCase()));
    const matchesGender = genderFilter === 'all' || p.gender.toLowerCase() === genderFilter.toLowerCase();
    return matchesSearch && matchesGender;
  });

  if (loading && patientProfiles.length === 0) {
    return (
      <div className="space-y-6 max-w-7xl mx-auto animate-pulse">
        <div className="h-12 bg-slate-100 rounded-xl w-48"></div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          <div className="h-44 bg-slate-100 rounded-2xl"></div>
          <div className="h-44 bg-slate-100 rounded-2xl"></div>
          <div className="h-44 bg-slate-100 rounded-2xl"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      
      {/* Header section */}
      <div className="border-b border-border pb-4 text-left">
        <h1 className="text-2xl font-extrabold tracking-tight text-slate-900">My Patients</h1>
        <p className="text-xs text-muted-foreground mt-0.5">
          Directory of registered patients, demographic histories, and historic clinical workups.
        </p>
      </div>

      {/* Roster Statistics banner */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 text-left">
        <div className="bg-card border border-border p-4 rounded-2xl shadow-xs flex items-center justify-between">
          <div className="space-y-1">
            <span className="text-[10px] font-extrabold text-slate-500 uppercase tracking-wider block">Managed Patient Cards</span>
            <span className="text-2xl font-black text-slate-900">{patientProfiles.length}</span>
          </div>
          <div className="p-2.5 bg-sky-50 text-sky-850 rounded-xl">
            <Users className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* Filter Roster Toolbar */}
      <div className="flex flex-col sm:flex-row gap-4 justify-between items-center bg-card border border-border p-4 rounded-2xl shadow-xs text-left">
        <div className="relative w-full sm:max-w-xs">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search name, email, comorbidity..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            className="w-full bg-transparent border border-border rounded-xl pl-10 pr-4 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary placeholder-slate-400 font-semibold"
          />
        </div>

        <div className="flex items-center space-x-2 w-full sm:w-auto self-start sm:self-auto justify-end">
          <span className="text-xs text-muted-foreground font-bold">Gender:</span>
          <select
            value={genderFilter}
            onChange={e => setGenderFilter(e.target.value)}
            className="bg-transparent border border-border rounded-xl px-3 py-1.5 text-xs font-semibold focus:outline-none"
          >
            <option value="all">All Genders</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
          </select>
        </div>
      </div>

      {/* Patients Roster Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredPatients.map(p => (
          <div 
            key={p.name}
            onClick={() => { setSelectedPatient(p); toast(`Opened profile for ${p.name}`, 'info'); }}
            className="bg-card border border-border p-5 rounded-2xl hover:border-primary shadow-xs hover:shadow-md cursor-pointer transition text-left space-y-4"
          >
            <div className="flex items-center gap-3 border-b border-slate-50 pb-3">
              <div className="w-10 h-10 rounded-full bg-slate-100 flex items-center justify-center text-slate-500 font-bold border border-border shrink-0">
                <User className="w-5 h-5 text-sky-850" />
              </div>
              <div className="space-y-0.5 overflow-hidden">
                <h3 className="font-extrabold text-sm text-slate-900 truncate">{p.name}</h3>
                <span className="text-[10px] text-muted-foreground font-semibold uppercase">
                  {p.gender} | {p.age} years
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 text-[10px] font-bold text-slate-655">
              <div>
                <span className="text-slate-400 block text-[8px] uppercase">Blood Type</span>
                <span>{p.bloodType}</span>
              </div>
              <div>
                <span className="text-slate-400 block text-[8px] uppercase">Active Case Cases</span>
                <span>{p.cases.length} entries</span>
              </div>
            </div>

            <div className="border-t border-slate-50 pt-3 flex justify-between items-center text-[10px] font-semibold text-primary">
              <span>View EMR Profile</span>
              <ChevronRight className="w-4 h-4" />
            </div>
          </div>
        ))}
      </div>

      {/* Patient Profile Modal Overlay */}
      <AnimatePresence>
        {selectedPatient && (
          <div className="fixed inset-0 z-[9999] flex items-center justify-center p-4">
            {/* Backdrop overlay */}
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setSelectedPatient(null)}
              className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm"
            />
            
            {/* Modal Box */}
            <motion.div 
              initial={{ opacity: 0, scale: 0.95, y: 15 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 15 }}
              className="relative bg-card border border-border rounded-2xl w-full max-w-2xl overflow-hidden shadow-2xl max-h-[85vh] flex flex-col text-left"
            >
              
              {/* Close Button */}
              <button
                onClick={() => setSelectedPatient(null)}
                className="absolute top-4 right-4 p-1.5 rounded-xl hover:bg-slate-100 text-slate-500 transition z-10"
              >
                <X className="w-5 h-5" />
              </button>

              {/* Modal Header */}
              <div className="p-6 bg-slate-50 border-b border-border flex items-center gap-4">
                <div className="w-12 h-12 rounded-full bg-sky-50 border border-sky-100 flex items-center justify-center text-sky-700 font-bold shrink-0">
                  <User className="w-6 h-6 text-sky-850" />
                </div>
                <div className="space-y-0.5 text-left">
                  <h2 className="text-lg font-black text-slate-900 tracking-tight">{selectedPatient.name}</h2>
                  <p className="text-xs text-muted-foreground font-semibold">
                    Patient Profile Session | Record Type: EMR-CDSS
                  </p>
                </div>
              </div>

              {/* Modal scrollable body */}
              <div className="p-6 overflow-y-auto space-y-6 flex-1">
                
                {/* Contact Demographics Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-semibold text-slate-655">
                  <div className="flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-sky-700" />
                    <div>
                      <span className="text-[8px] text-slate-400 block uppercase">Age / Gender</span>
                      <span>{selectedPatient.age} yrs / {selectedPatient.gender}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Mail className="w-4 h-4 text-sky-700" />
                    <div className="overflow-hidden">
                      <span className="text-[8px] text-slate-400 block uppercase">Email Contact</span>
                      <span className="truncate block max-w-[150px]">{selectedPatient.email}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Phone className="w-4 h-4 text-sky-700" />
                    <div>
                      <span className="text-[8px] text-slate-400 block uppercase">Phone Contact</span>
                      <span>{selectedPatient.phone}</span>
                    </div>
                  </div>
                </div>

                {/* Comorbidities & Lifestyle Factors */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 border-t border-slate-100 pt-4">
                  <div className="space-y-1.5">
                    <h4 className="text-[10px] font-extrabold text-slate-400 uppercase tracking-wider">Chronic Conditions</h4>
                    <div className="flex flex-wrap gap-1.5">
                      {selectedPatient.comorbidities.map(c => (
                        <span key={c} className="bg-amber-50 border border-amber-200 text-amber-800 px-2 py-0.5 rounded text-[10px] font-semibold">
                          {c}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="space-y-1.5">
                    <h4 className="text-[10px] font-extrabold text-slate-400 uppercase tracking-wider">Lifestyle Risk Factors</h4>
                    <div className="flex flex-wrap gap-1.5">
                      {selectedPatient.lifestyleFactors.map(l => (
                        <span key={l} className="bg-slate-100 border border-border text-slate-700 px-2 py-0.5 rounded text-[10px] font-semibold">
                          {l}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Vitals baseline cards */}
                <div className="border-t border-slate-100 pt-4 space-y-2">
                  <h4 className="text-[10px] font-extrabold text-slate-400 uppercase tracking-wider">Baseline Physiological Parameters</h4>
                  
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <div className="bg-slate-50 border border-border p-3 rounded-xl">
                      <span className="text-[8px] text-slate-400 block uppercase font-bold">Heart Rate</span>
                      <span className="font-bold text-slate-800 text-sm block">{selectedPatient.latestVitals.hr} bpm</span>
                    </div>
                    <div className="bg-slate-50 border border-border p-3 rounded-xl">
                      <span className="text-[8px] text-slate-400 block uppercase font-bold">Blood Pressure</span>
                      <span className="font-bold text-slate-800 text-sm block">{selectedPatient.latestVitals.bp}</span>
                    </div>
                    <div className="bg-slate-50 border border-border p-3 rounded-xl">
                      <span className="text-[8px] text-slate-400 block uppercase font-bold">SpO2 Oxygen</span>
                      <span className="font-bold text-slate-800 text-sm block">{selectedPatient.latestVitals.spo2}%</span>
                    </div>
                    <div className="bg-slate-50 border border-border p-3 rounded-xl">
                      <span className="text-[8px] text-slate-400 block uppercase font-bold">Temperature</span>
                      <span className="font-bold text-slate-800 text-sm block">{selectedPatient.latestVitals.temp}°F</span>
                    </div>
                  </div>
                </div>

                {/* Case History Timeline list */}
                <div className="border-t border-slate-100 pt-4 space-y-3">
                  <h4 className="text-[10px] font-extrabold text-slate-400 uppercase tracking-wider">Consultation History</h4>
                  
                  {selectedPatient.cases.length > 0 ? (
                    <div className="border border-border rounded-xl overflow-hidden">
                      <table className="min-w-full divide-y divide-border text-xs text-left">
                        <thead className="bg-slate-50 text-[10px] font-bold text-muted-foreground uppercase">
                          <tr>
                            <th className="px-4 py-2">ID</th>
                            <th className="px-4 py-2">Triage (ESI)</th>
                            <th className="px-4 py-2">Status</th>
                            <th className="px-4 py-2">Submitted</th>
                            <th className="px-4 py-2 text-right">Report</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border font-semibold text-slate-700">
                          {selectedPatient.cases.map(c => (
                            <tr key={c.id} className="hover:bg-slate-50/50">
                              <td className="px-4 py-2.5 text-muted-foreground">#{c.id}</td>
                              <td className="px-4 py-2.5">
                                <span className={`px-1.5 py-0.5 rounded text-[9px] uppercase ${
                                  c.triage_level === 1 ? 'bg-red-50 text-red-750 font-bold' : c.triage_level === 2 ? 'bg-orange-50 text-orange-850' : 'bg-slate-100 text-slate-700'
                                }`}>
                                  Level {c.triage_level}
                                </span>
                              </td>
                              <td className="px-4 py-2.5 uppercase text-[9px] text-primary">{c.status}</td>
                              <td className="px-4 py-2.5 text-[10px] text-muted-foreground">
                                {new Date(c.created_at).toLocaleDateString()}
                              </td>
                              <td className="px-4 py-2.5 text-right">
                                <button
                                  onClick={() => { setSelectedPatient(null); navigate(`/reports/${c.id}`); }}
                                  className="text-primary font-bold hover:underline"
                                >
                                  Open
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="text-xs text-muted-foreground italic text-center py-4">
                      No matching EMR consultation history records found.
                    </p>
                  )}
                </div>

              </div>

            </motion.div>
          </div>
        )}
      </AnimatePresence>

    </div>
  );
}
