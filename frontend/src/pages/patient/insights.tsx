import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  LineChart as LucideLineChart, Activity, AlertTriangle, ShieldCheck, Heart, 
  Clock, Thermometer, Zap, Apple, Dribbble, Sparkles, TrendingUp, TrendingDown,
  Calendar, CheckCircle2, AlertCircle, HeartPulse, Stethoscope, ChevronRight,
  ShieldAlert, ActivitySquare, CheckCircle, Info
} from 'lucide-react';
import { 
  ResponsiveContainer, LineChart as RechartsLineChart, Line, XAxis, YAxis, 
  CartesianGrid, Tooltip, Legend 
} from 'recharts';

// Custom Count-Up Animation Component for KPIs
function AnimatedNumber({ value, suffix = '', duration = 1000 }: { value: number; suffix?: string; duration?: number }) {
  const [current, setCurrent] = useState(0);

  useEffect(() => {
    let start = 0;
    const end = value;
    if (start === end) {
      setCurrent(end);
      return;
    }

    const totalMilliseconds = duration;
    const stepTime = Math.max(Math.floor(totalMilliseconds / end), 15);
    
    const timer = setInterval(() => {
      start += 1;
      if (start >= end) {
        setCurrent(end);
        clearInterval(timer);
      } else {
        setCurrent(start);
      }
    }, stepTime);

    return () => clearInterval(timer);
  }, [value, duration]);

  return <span>{current}{suffix}</span>;
}

export default function PatientInsights() {
  const navigate = useNavigate();

  // Demo Vital Trends Data (Mon-Sun)
  const vitalTrendsData = [
    { day: 'Mon', heartRate: 92, spO2: 95, temperature: 100.2 },
    { day: 'Tue', heartRate: 90, spO2: 95, temperature: 99.8 },
    { day: 'Wed', heartRate: 88, spO2: 96, temperature: 99.4 },
    { day: 'Thu', heartRate: 86, spO2: 96, temperature: 99.1 },
    { day: 'Fri', heartRate: 84, spO2: 97, temperature: 98.9 },
    { day: 'Sat', heartRate: 83, spO2: 97, temperature: 98.6 },
    { day: 'Sun', heartRate: 82, spO2: 98, temperature: 98.4 },
  ];

  // Motion variants for staggering child elements
  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.08
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 100, damping: 15 } }
  };

  return (
    <div className="min-h-screen bg-slate-50/30 space-y-8 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 text-left">
      
      {/* Title Header */}
      <motion.div 
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-slate-100 pb-6 gap-4"
      >
        <div className="space-y-1">
          <h1 className="text-3xl font-extrabold tracking-tight text-slate-900 font-sans">Health Insights</h1>
          <p className="text-sm text-slate-500">
            Real-time explainable CDSS analytics, clinical progression, and personalized recovery timeline.
          </p>
        </div>
        <button 
          onClick={() => navigate('/patient/intake')}
          className="w-full sm:w-auto bg-indigo-600 hover:bg-indigo-700 text-white font-bold px-5 py-2.5 rounded-xl text-xs transition duration-200 shadow-sm hover:shadow flex items-center justify-center gap-1.5 self-start sm:self-center"
        >
          <Stethoscope className="w-4 h-4" /> Start New Intake
        </button>
      </motion.div>

      {/* Main Layout: Main Dashboard Area (Col span 2) & Sticky Widget (Col span 1) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
        
        {/* Main Dashboard Columns */}
        <div className="lg:col-span-2 space-y-8">

          {/* ==================================================
              SECTION 1: HEALTH OVERVIEW HERO
              ================================================== */}
          <motion.div 
            variants={containerVariants}
            initial="hidden"
            animate="show"
            className="grid grid-cols-1 sm:grid-cols-2 gap-4"
          >
            {/* Health Score Card */}
            <motion.div 
              variants={itemVariants} 
              whileHover={{ y: -4, boxShadow: '0 10px 15px -3px rgba(148, 163, 184, 0.1)' }}
              className="bg-white border border-slate-100 p-5 rounded-2xl shadow-sm flex flex-col justify-between"
            >
              <div className="flex justify-between items-start">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Health Score</span>
                <span className="bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded-full text-[10px] font-bold">
                  +6% Change
                </span>
              </div>
              <div className="mt-4">
                <h3 className="text-3xl font-extrabold text-slate-900">
                  <AnimatedNumber value={84} suffix=" / 100" />
                </h3>
                <p className="text-xs font-semibold text-slate-500 mt-1.5 flex items-center gap-1">
                  <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
                  Stable Condition
                </p>
                <p className="text-[10px] text-slate-400 mt-1 font-medium">Since previous assessment</p>
              </div>
            </motion.div>

            {/* Clinical Risk Card */}
            <motion.div 
              variants={itemVariants}
              whileHover={{ y: -4, boxShadow: '0 10px 15px -3px rgba(148, 163, 184, 0.1)' }}
              className="bg-white border border-slate-100 p-5 rounded-2xl shadow-sm flex flex-col justify-between"
            >
              <div className="flex justify-between items-start">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Clinical Risk</span>
                <span className="bg-sky-50 text-sky-700 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider">
                  Low
                </span>
              </div>
              <div className="mt-4">
                <h3 className="text-3xl font-extrabold text-slate-900">
                  Low Risk
                </h3>
                <p className="text-xs font-semibold text-slate-500 mt-1.5 flex items-center gap-1">
                  <ShieldCheck className="w-4 h-4 text-sky-655 shrink-0" />
                  Risk Score: <span className="text-slate-800 font-bold">24 / 100</span>
                </p>
                <p className="text-[10px] text-slate-400 mt-1 font-medium">Cardiometabolic & comorbidity parameters</p>
              </div>
            </motion.div>

            {/* Recovery Trend Card */}
            <motion.div 
              variants={itemVariants}
              whileHover={{ y: -4, boxShadow: '0 10px 15px -3px rgba(148, 163, 184, 0.1)' }}
              className="bg-white border border-slate-100 p-5 rounded-2xl shadow-sm flex flex-col justify-between"
            >
              <div className="flex justify-between items-start">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Recovery Trend</span>
                <span className="bg-teal-50 text-teal-700 px-2 py-0.5 rounded-full text-[10px] font-bold flex items-center gap-0.5">
                  <TrendingUp className="w-3 h-3" /> Upward
                </span>
              </div>
              <div className="mt-4">
                <h3 className="text-3xl font-extrabold text-slate-900">
                  Improving
                </h3>
                <p className="text-xs font-semibold text-slate-505 mt-1.5">
                  Previous: <span className="font-bold text-slate-700">72</span> → Current: <span className="font-bold text-slate-700">84</span>
                </p>
                <p className="text-[10px] text-slate-400 mt-1 font-medium">Progressive physiological recovery</p>
              </div>
            </motion.div>

            {/* Assessment Activity Card */}
            <motion.div 
              variants={itemVariants}
              whileHover={{ y: -4, boxShadow: '0 10px 15px -3px rgba(148, 163, 184, 0.1)' }}
              className="bg-white border border-slate-100 p-5 rounded-2xl shadow-sm flex flex-col justify-between"
            >
              <div className="flex justify-between items-start">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Assessment Activity</span>
                <span className="bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded-full text-[10px] font-bold">
                  Active
                </span>
              </div>
              <div className="mt-4">
                <h3 className="text-3xl font-extrabold text-slate-900">
                  <AnimatedNumber value={3} /> Completed
                </h3>
                <p className="text-xs font-semibold text-slate-500 mt-1.5 flex items-center gap-1">
                  <Clock className="w-4 h-4 text-indigo-500 shrink-0" />
                  Last: 2 days ago | Next: 3 days
                </p>
                <p className="text-[10px] text-slate-400 mt-1 font-medium">Intake logs history tracking</p>
              </div>
            </motion.div>
          </motion.div>

          {/* ==================================================
              SECTION 2: HEALTH SCORE GAUGE
              ================================================== */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
            className="bg-white border border-slate-100 p-6 sm:p-8 rounded-3xl shadow-sm flex flex-col sm:flex-row items-center gap-8 text-left"
          >
            {/* SVG Circular Progress Gauge */}
            <div className="relative w-36 h-36 flex items-center justify-center shrink-0">
              <svg className="w-full h-full transform -rotate-90">
                <circle cx="72" cy="72" r="60" fill="transparent" stroke="#f1f5f9" strokeWidth="10" />
                <motion.circle 
                  cx="72" 
                  cy="72" 
                  r="60" 
                  fill="transparent" 
                  stroke="#10b981" // Green color rule (score is 84)
                  strokeWidth="10" 
                  strokeDasharray={`${2 * Math.PI * 60}`}
                  initial={{ strokeDashoffset: 2 * Math.PI * 60 }}
                  animate={{ strokeDashoffset: 2 * Math.PI * 60 * (1 - 0.84) }}
                  transition={{ duration: 1.5, ease: 'easeOut' }}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-3xl font-black text-slate-900">84%</span>
                <span className="text-[10px] font-extrabold text-emerald-600 uppercase tracking-widest mt-0.5">Stable</span>
              </div>
            </div>
            
            {/* Value Explanation & Description */}
            <div className="space-y-3 flex-1">
              <h3 className="text-xl font-bold text-slate-800 font-sans">Overall Health Score</h3>
              <p className="text-xs sm:text-sm text-slate-500 leading-relaxed font-medium">
                Your overall health indicators remain within acceptable ranges. The multi-agent evaluation confirms stable progress with optimal recovery parameters.
              </p>
              
              {/* Color Rules indicators */}
              <div className="flex flex-wrap gap-4 pt-1">
                <span className="flex items-center gap-1.5 text-xs font-semibold text-slate-500">
                  <span className="w-2.5 h-2.5 bg-emerald-500 rounded-full"></span> 80-100 Green (Stable)
                </span>
                <span className="flex items-center gap-1.5 text-xs font-semibold text-slate-500">
                  <span className="w-2.5 h-2.5 bg-amber-500 rounded-full"></span> 60-79 Amber (Observation)
                </span>
                <span className="flex items-center gap-1.5 text-xs font-semibold text-slate-500">
                  <span className="w-2.5 h-2.5 bg-red-500 rounded-full"></span> 0-59 Red (Critical)
                </span>
              </div>
            </div>
          </motion.div>

          {/* ==================================================
              SECTION 3: TODAY'S HEALTH SNAPSHOT
              ================================================== */}
          <div className="space-y-4">
            <h3 className="text-sm font-extrabold text-slate-400 uppercase tracking-wider">Today's Health Snapshot</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              
              {/* Temperature */}
              <div className="bg-white border border-slate-100 p-4 rounded-2xl shadow-sm text-left flex flex-col justify-between">
                <div className="p-2 bg-amber-50 text-amber-600 rounded-xl w-fit">
                  <Thermometer className="w-5 h-5" />
                </div>
                <div className="mt-4">
                  <span className="text-[10px] font-bold text-slate-400 uppercase block">Temperature</span>
                  <span className="text-lg font-extrabold text-slate-800">98.4°F</span>
                  <span className="text-[10px] font-semibold text-emerald-600 block mt-0.5">Normal Range</span>
                </div>
              </div>

              {/* Heart Rate */}
              <div className="bg-white border border-slate-100 p-4 rounded-2xl shadow-sm text-left flex flex-col justify-between">
                <div className="p-2 bg-rose-50 text-rose-600 rounded-xl w-fit">
                  <Heart className="w-5 h-5" />
                </div>
                <div className="mt-4">
                  <span className="text-[10px] font-bold text-slate-400 uppercase block">Heart Rate</span>
                  <span className="text-lg font-extrabold text-slate-800">82 bpm</span>
                  <span className="text-[10px] font-semibold text-emerald-600 block mt-0.5">Normal Resting</span>
                </div>
              </div>

              {/* SpO2 */}
              <div className="bg-white border border-slate-100 p-4 rounded-2xl shadow-sm text-left flex flex-col justify-between">
                <div className="p-2 bg-teal-50 text-teal-600 rounded-xl w-fit">
                  <Activity className="w-5 h-5" />
                </div>
                <div className="mt-4">
                  <span className="text-[10px] font-bold text-slate-400 uppercase block">SpO2</span>
                  <span className="text-lg font-extrabold text-slate-800">98%</span>
                  <span className="text-[10px] font-semibold text-emerald-600 block mt-0.5">Normal Saturation</span>
                </div>
              </div>

              {/* Blood Pressure */}
              <div className="bg-white border border-slate-100 p-4 rounded-2xl shadow-sm text-left flex flex-col justify-between">
                <div className="p-2 bg-sky-50 text-sky-600 rounded-xl w-fit">
                  <ShieldCheck className="w-5 h-5" />
                </div>
                <div className="mt-4">
                  <span className="text-[10px] font-bold text-slate-400 uppercase block">Blood Pressure</span>
                  <span className="text-lg font-extrabold text-slate-800">118 / 76</span>
                  <span className="text-[10px] font-semibold text-emerald-600 block mt-0.5">Normal Range</span>
                </div>
              </div>

            </div>
          </div>

          {/* ==================================================
              SECTION 4: 7 DAY VITAL TRENDS
              ================================================== */}
          <div className="bg-white border border-slate-100 p-5 rounded-3xl shadow-sm space-y-4">
            <div className="flex justify-between items-center pb-3 border-b border-slate-50">
              <div className="space-y-0.5">
                <h4 className="text-sm font-bold text-slate-800">7 Day Vital Trends</h4>
                <p className="text-[10.5px] text-slate-400">Longitudinal monitoring of critical clinical signals.</p>
              </div>
              <span className="bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded-full text-[9px] font-bold flex items-center gap-0.5">
                <TrendingUp className="w-3 h-3" /> Recovery Trend Positive
              </span>
            </div>
            
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsLineChart 
                  data={vitalTrendsData}
                  margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#f8fafc" />
                  <XAxis dataKey="day" stroke="#94a3b8" fontSize={11} tickLine={false} />
                  <YAxis stroke="#94a3b8" fontSize={11} domain={[80, 102]} tickLine={false} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#ffffff', borderRadius: '12px', border: '1px solid #f1f5f9', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.05)' }} 
                    labelStyle={{ fontWeight: 'bold', color: '#1e293b' }}
                  />
                  <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '11px', fontWeight: 'semibold' }} />
                  <Line 
                    type="monotone" 
                    dataKey="heartRate" 
                    name="Heart Rate (bpm)" 
                    stroke="#f43f5e" 
                    strokeWidth={2.5} 
                    dot={{ r: 4 }} 
                    activeDot={{ r: 6 }} 
                  />
                  <Line 
                    type="monotone" 
                    dataKey="spO2" 
                    name="SpO2 (%)" 
                    stroke="#0d9488" 
                    strokeWidth={2.5} 
                    dot={{ r: 4 }} 
                    activeDot={{ r: 6 }} 
                  />
                  <Line 
                    type="monotone" 
                    dataKey="temperature" 
                    name="Temp (°F)" 
                    stroke="#d97706" 
                    strokeWidth={2.5} 
                    dot={{ r: 4 }} 
                    activeDot={{ r: 6 }} 
                  />
                </RechartsLineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* ==================================================
              SECTION 5: HEALTH JOURNEY TIMELINE
              ================================================== */}
          <div className="space-y-4">
            <h3 className="text-sm font-extrabold text-slate-400 uppercase tracking-wider">Health Journey Timeline</h3>
            <div className="bg-white border border-slate-100 p-6 rounded-3xl shadow-sm">
              
              <div className="relative border-l-2 border-slate-100 pl-6 ml-2.5 space-y-8 py-2">
                
                {/* Day 1 */}
                <div className="relative">
                  <span className="absolute -left-[30px] top-1.5 w-3.5 h-3.5 rounded-full border-2 border-white bg-amber-500 ring-4 ring-amber-100"></span>
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-bold text-slate-400 uppercase">Day 1</span>
                      <span className="bg-amber-50 text-amber-700 px-1.5 py-0.5 rounded text-[9px] font-bold">Assessment #1</span>
                    </div>
                    <h4 className="text-xs font-bold text-slate-800">Moderate Risk & High Fever</h4>
                    <p className="text-[11px] text-slate-500 leading-relaxed">
                      Initial CDSS analysis completed. Severe cough with high fever noted.
                    </p>
                  </div>
                </div>

                {/* Day 5 */}
                <div className="relative">
                  <span className="absolute -left-[30px] top-1.5 w-3.5 h-3.5 rounded-full border-2 border-white bg-sky-500 ring-4 ring-sky-100"></span>
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-bold text-slate-400 uppercase">Day 5</span>
                      <span className="bg-sky-50 text-sky-700 px-1.5 py-0.5 rounded text-[9px] font-bold">Assessment #2</span>
                    </div>
                    <h4 className="text-xs font-bold text-slate-800">Symptoms Reduced</h4>
                    <p className="text-[11px] text-slate-500 leading-relaxed">
                      Follow-up check. Fever has broken and cough has diminished.
                    </p>
                  </div>
                </div>

                {/* Day 10 */}
                <div className="relative">
                  <span className="absolute -left-[30px] top-1.5 w-3.5 h-3.5 rounded-full border-2 border-white bg-emerald-500 ring-4 ring-emerald-100"></span>
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-bold text-slate-400 uppercase">Day 10</span>
                      <span className="bg-emerald-50 text-emerald-700 px-1.5 py-0.5 rounded text-[9px] font-bold">Assessment #3</span>
                    </div>
                    <h4 className="text-xs font-bold text-slate-800">Vitals Improved</h4>
                    <p className="text-[11px] text-slate-500 leading-relaxed">
                      Third check. Temperature and oxygen saturation returned to optimal baselines.
                    </p>
                  </div>
                </div>

                {/* Today */}
                <div className="relative">
                  <span className="absolute -left-[30px] top-1.5 w-3.5 h-3.5 rounded-full border-2 border-white bg-teal-500 ring-4 ring-teal-100 animate-pulse"></span>
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-bold text-slate-400 uppercase">Today</span>
                      <span className="bg-teal-50 text-teal-700 px-1.5 py-0.5 rounded text-[9px] font-bold">Current Status</span>
                    </div>
                    <h4 className="text-xs font-bold text-slate-800">Stable Monitoring</h4>
                    <p className="text-[11px] text-slate-500 leading-relaxed">
                      All systems green. Continue basic care plan monitoring.
                    </p>
                  </div>
                </div>

              </div>
              
            </div>
          </div>

          {/* ==================================================
              SECTION 6: AI FINDINGS SUMMARY
              ================================================== */}
          <div className="bg-white border border-slate-100 p-6 rounded-3xl shadow-sm space-y-5">
            <div className="flex items-center gap-2 border-b border-slate-50 pb-3">
              <Sparkles className="w-5 h-5 text-indigo-500 animate-pulse" />
              <h3 className="text-sm font-extrabold text-slate-850 uppercase tracking-wide">AI Findings Summary</h3>
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-1">
                <span className="text-[10px] font-bold text-slate-400 uppercase">Primary Finding</span>
                <p className="text-sm font-extrabold text-slate-800">Upper Respiratory Infection Pattern</p>
              </div>
              <div className="space-y-1">
                <span className="text-[10px] font-bold text-slate-400 uppercase">Current Severity</span>
                <span className="bg-amber-50 text-amber-700 px-2 py-0.5 rounded text-xs font-extrabold inline-block">
                  Moderate
                </span>
              </div>
              <div className="space-y-1">
                <span className="text-[10px] font-bold text-slate-400 uppercase">Clinical Status</span>
                <span className="bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded text-xs font-extrabold inline-block">
                  Improving
                </span>
              </div>
              <div className="space-y-1">
                <span className="text-[10px] font-bold text-slate-400 uppercase">Emergency Status</span>
                <span className="bg-slate-50 text-slate-700 px-2 py-0.5 rounded text-xs font-extrabold inline-block">
                  No Emergency Indicators Detected
                </span>
              </div>
            </div>

            <div className="bg-slate-50 p-4 border border-slate-100 rounded-2xl space-y-1">
              <span className="text-[10px] font-bold text-slate-400 uppercase">Analysis Explanation</span>
              <p className="text-xs text-slate-600 leading-relaxed font-medium">
                Your recent symptoms and vital signs suggest recovery from a respiratory infection. Current indicators show improvement compared to previous assessments.
              </p>
            </div>
          </div>

          {/* ==================================================
              SECTION 7: RISK FACTORS ANALYSIS
              ================================================== */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            
            {/* Risk Factors */}
            <div className="bg-white border border-slate-100 p-5 rounded-3xl shadow-sm space-y-4">
              <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wide flex items-center gap-1.5">
                <AlertCircle className="w-4 h-4 text-amber-500" /> Risk Factors
              </h4>
              <div className="flex flex-col gap-2">
                <span className="bg-amber-50 text-amber-800 px-3 py-2 rounded-xl text-xs font-bold flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-amber-500 rounded-full"></span> Elevated BMI
                </span>
                <span className="bg-amber-50 text-amber-800 px-3 py-2 rounded-xl text-xs font-bold flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-amber-500 rounded-full"></span> Sedentary Lifestyle
                </span>
                <span className="bg-amber-50 text-amber-800 px-3 py-2 rounded-xl text-xs font-bold flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-amber-500 rounded-full"></span> Family History of Diabetes
                </span>
              </div>
            </div>

            {/* Positive Indicators */}
            <div className="bg-white border border-slate-100 p-5 rounded-3xl shadow-sm space-y-4">
              <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wide flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4 text-emerald-500" /> Positive Indicators
              </h4>
              <div className="flex flex-col gap-2">
                <span className="bg-emerald-50 text-emerald-800 px-3 py-2 rounded-xl text-xs font-bold flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></span> Normal Oxygen Levels
                </span>
                <span className="bg-emerald-50 text-emerald-800 px-3 py-2 rounded-xl text-xs font-bold flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></span> Stable Blood Pressure
                </span>
                <span className="bg-emerald-50 text-emerald-800 px-3 py-2 rounded-xl text-xs font-bold flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></span> No Emergency Symptoms
                </span>
                <span className="bg-emerald-50 text-emerald-800 px-3 py-2 rounded-xl text-xs font-bold flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></span> Improving Trend
                </span>
              </div>
            </div>

          </div>

          {/* ==================================================
              SECTION 8: PERSONALIZED CARE PLAN
              ================================================== */}
          <div className="space-y-4">
            <h3 className="text-sm font-extrabold text-slate-400 uppercase tracking-wider">Personalized Care Plan</h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              
              {/* Exercise card */}
              <div className="bg-white border border-slate-100 p-5 rounded-2xl shadow-sm space-y-3">
                <div className="p-2 bg-indigo-50 text-indigo-650 rounded-xl w-fit">
                  <Dribbble className="w-5 h-5" />
                </div>
                <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider">Exercise</h4>
                <ul className="text-xs text-slate-655 space-y-2 list-disc pl-4 font-medium">
                  <li>Walk 30 minutes daily</li>
                  <li>Light stretching</li>
                  <li>Avoid excessive exertion</li>
                </ul>
              </div>

              {/* Nutrition card */}
              <div className="bg-white border border-slate-100 p-5 rounded-2xl shadow-sm space-y-3">
                <div className="p-2 bg-teal-50 text-teal-650 rounded-xl w-fit">
                  <Apple className="w-5 h-5" />
                </div>
                <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider">Nutrition</h4>
                <ul className="text-xs text-slate-655 space-y-2 list-disc pl-4 font-medium">
                  <li>Increase hydration</li>
                  <li>Vitamin C rich foods</li>
                  <li>Balanced protein intake</li>
                </ul>
              </div>

              {/* Monitoring card */}
              <div className="bg-white border border-slate-100 p-5 rounded-2xl shadow-sm space-y-3">
                <div className="p-2 bg-sky-50 text-sky-655 rounded-xl w-fit">
                  <Clock className="w-5 h-5" />
                </div>
                <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider">Monitoring</h4>
                <ul className="text-xs text-slate-655 space-y-2 list-disc pl-4 font-medium">
                  <li>Monitor temperature</li>
                  <li>Observe symptom progression</li>
                  <li>Repeat assessment in 3 days</li>
                </ul>
              </div>

            </div>
          </div>

          {/* ==================================================
              SECTION 9: RECOVERY PROBABILITY
              ================================================== */}
          <div className="bg-white border border-slate-100 p-5 rounded-3xl shadow-sm space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-xs font-bold text-slate-800 uppercase font-sans">Recovery Likelihood</span>
              <span className="text-sm font-extrabold text-emerald-600">88%</span>
            </div>
            
            <div className="w-full bg-slate-150 h-2.5 rounded-full overflow-hidden">
              <motion.div 
                className="bg-emerald-500 h-full rounded-full"
                initial={{ width: 0 }}
                animate={{ width: '88%' }}
                transition={{ duration: 1.2, ease: 'easeOut' }}
              />
            </div>
            
            <p className="text-[11.5px] text-slate-500 font-medium">
              Current trends suggest a high probability of continued improvement.
            </p>
          </div>

          {/* ==================================================
              SECTION 10: EMERGENCY AWARENESS
              ================================================== */}
          <div className="bg-amber-50 border border-amber-250 p-5 rounded-3xl shadow-sm space-y-3 text-left">
            <h4 className="text-xs font-extrabold text-amber-850 uppercase tracking-wider flex items-center gap-1.5">
              <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 animate-pulse" /> Seek immediate medical attention if:
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pl-2 text-xs font-bold text-amber-900">
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-amber-600 rounded-full"></span> Fever exceeds 103°F
              </div>
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-amber-600 rounded-full"></span> Breathing difficulty increases
              </div>
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-amber-600 rounded-full"></span> Oxygen saturation falls below 92%
              </div>
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-amber-600 rounded-full"></span> Severe chest pain develops
              </div>
            </div>
          </div>

        </div>

        {/* ==================================================
            SECTION 11: TODAY'S SNAPSHOT WIDGET (Sticky Sidebar Card)
            ================================================== */}
        <div className="lg:col-span-1 lg:sticky lg:top-8 space-y-6">
          <div className="bg-white border border-slate-100 p-6 rounded-3xl shadow-sm text-left space-y-5">
            <div className="flex items-center justify-between border-b border-slate-50 pb-3.5">
              <h3 className="text-sm font-extrabold text-slate-800 uppercase tracking-wider">Today's Snapshot</h3>
              <span className="bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded-full text-[9px] font-bold">
                Low Risk
              </span>
            </div>

            <div className="space-y-4">
              
              <div className="flex justify-between items-center text-xs font-semibold">
                <span className="text-slate-400">Health Score</span>
                <span className="text-slate-800 font-extrabold">84 / 100</span>
              </div>

              <div className="flex justify-between items-center text-xs font-semibold">
                <span className="text-slate-400">Clinical Risk</span>
                <span className="text-emerald-605 font-extrabold">Low</span>
              </div>

              <div className="flex justify-between items-center text-xs font-semibold">
                <span className="text-slate-400">Trend</span>
                <span className="text-teal-600 font-extrabold flex items-center gap-0.5">
                  <TrendingUp className="w-3.5 h-3.5" /> Improving
                </span>
              </div>

              <div className="flex justify-between items-center text-xs font-semibold">
                <span className="text-slate-400">Oxygen Saturation (SpO2)</span>
                <span className="text-slate-800 font-extrabold">98%</span>
              </div>

              <div className="flex justify-between items-center text-xs font-semibold">
                <span className="text-slate-400">Heart Rate</span>
                <span className="text-slate-800 font-extrabold">82 bpm</span>
              </div>

              <div className="flex justify-between items-center text-xs font-semibold">
                <span className="text-slate-400">Next Assessment</span>
                <span className="text-indigo-650 font-extrabold">In 3 Days</span>
              </div>

            </div>

            <div className="pt-3 border-t border-slate-100">
              <button 
                onClick={() => navigate('/patient/intake')}
                className="w-full bg-indigo-600 text-white font-bold py-3.5 rounded-2xl text-xs hover:bg-indigo-700 transition duration-200 shadow-sm hover:shadow flex items-center justify-center gap-1.5"
              >
                <Stethoscope className="w-4 h-4" /> Start New Intake
              </button>
            </div>
          </div>
        </div>

      </div>

    </div>
  );
}
