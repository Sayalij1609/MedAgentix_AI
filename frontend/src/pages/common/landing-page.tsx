import React, { useEffect, useState, useRef } from 'react';
import { motion, useInView } from 'framer-motion';
import {
  Shield, Lock, Activity, ArrowRight, UserCheck, FileText,
  Brain, Zap, CheckCircle, ChevronRight, Heart, Stethoscope,
  ClipboardList, Cpu, FlaskConical, Pill, Microscope, BookOpen,
  GitMerge, ScanLine, BarChart3, Menu, X, MonitorPlay
} from 'lucide-react';

// ─────────────────────────────────────────────────────────
// Animated Counter (triggers on scroll)
// ─────────────────────────────────────────────────────────
const AnimCounter: React.FC<{ to: number; suffix?: string }> = ({ to, suffix = '' }) => {
  const [count, setCount] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  useEffect(() => {
    if (!inView) return;
    let start = 0;
    const step = Math.ceil(to / 60);
    const timer = setInterval(() => {
      start = Math.min(start + step, to);
      setCount(start);
      if (start >= to) clearInterval(timer);
    }, 18);
    return () => clearInterval(timer);
  }, [inView, to]);
  return <span ref={ref}>{count.toLocaleString()}{suffix}</span>;
};

// ─────────────────────────────────────────────────────────
// ECG Line
// ─────────────────────────────────────────────────────────
const EcgLine: React.FC = () => (
  <svg viewBox="0 0 300 60" className="w-52 h-10" fill="none" stroke="url(#ecg-grad)" strokeWidth="2.5">
    <defs>
      <linearGradient id="ecg-grad" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stopColor="#14b8a6" />
        <stop offset="100%" stopColor="#38bdf8" />
      </linearGradient>
    </defs>
    <motion.polyline
      points="0,30 40,30 55,10 65,50 75,30 130,30 145,5 155,55 165,30 220,30 235,12 245,48 255,30 300,30"
      initial={{ pathLength: 0, opacity: 0 }}
      animate={{ pathLength: 1, opacity: 1 }}
      transition={{ duration: 2, ease: 'easeInOut', repeat: Infinity, repeatDelay: 0.8 }}
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

// ─────────────────────────────────────────────────────────
// Floating Particle
// ─────────────────────────────────────────────────────────
const Particle: React.FC<{ x: number; y: number; size: number; delay: number }> = ({ x, y, size, delay }) => (
  <motion.div
    className="absolute rounded-full bg-teal-400 opacity-20 pointer-events-none"
    style={{ left: `${x}%`, top: `${y}%`, width: size, height: size }}
    animate={{ y: [0, -30, 0], opacity: [0.1, 0.3, 0.1] }}
    transition={{ duration: 4 + delay, repeat: Infinity, delay, ease: 'easeInOut' }}
  />
);

// ─────────────────────────────────────────────────────────
// Architecture Block
// ─────────────────────────────────────────────────────────
interface ArchBlockProps {
  icon?: React.ComponentType<{ className?: string }>;
  title: string;
  bullets?: string[];
  color?: string;
  bg?: string;
  border?: string;
  delay?: number;
  small?: boolean;
  badge?: string;
  badgeColor?: string;
}
const ArchBlock: React.FC<ArchBlockProps> = ({
  icon: Icon, title, bullets = [], color = 'text-teal-300', bg = 'bg-white/5',
  border = 'border-white/15', delay = 0, small = false, badge, badgeColor = 'bg-teal-500/20 text-teal-300'
}) => (
  <motion.div
    initial={{ opacity: 0, y: 16 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ delay, duration: 0.45, ease: 'easeOut' }}
    className={`${bg} border ${border} rounded-xl p-3.5 flex flex-col gap-2 relative overflow-hidden`}
  >
    {badge && (
      <span className={`absolute top-2 right-2 text-[8px] font-bold px-1.5 py-0.5 rounded-full uppercase tracking-wide ${badgeColor}`}>{badge}</span>
    )}
    <div className="flex items-center gap-2">
      {Icon && <Icon className={`${small ? 'w-3.5 h-3.5' : 'w-4 h-4'} ${color} shrink-0`} />}
      <p className={`${small ? 'text-[10px]' : 'text-[11px]'} font-bold text-white leading-tight`}>{title}</p>
    </div>
    {bullets.length > 0 && (
      <ul className="space-y-0.5 pl-0.5">
        {bullets.map((b, i) => (
          <li key={i} className="text-[9px] text-slate-400 flex items-start gap-1">
            <span className={`mt-[3px] w-1 h-1 rounded-full shrink-0 ${color.replace('text-', 'bg-')}`} />
            {b}
          </li>
        ))}
      </ul>
    )}
  </motion.div>
);

// Arrow connector
const Arrow: React.FC<{ label?: string; delay?: number }> = ({ label, delay = 0 }) => (
  <div className="flex flex-col items-center my-1">
    <motion.div
      initial={{ height: 0, opacity: 0 }}
      whileInView={{ height: 24, opacity: 1 }}
      viewport={{ once: true }}
      transition={{ delay, duration: 0.35 }}
      className="w-px bg-gradient-to-b from-teal-400 to-sky-500"
    />
    <motion.div
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      transition={{ delay: delay + 0.2 }}
      className="w-0 h-0 border-l-[5px] border-r-[5px] border-t-[6px] border-l-transparent border-r-transparent border-t-teal-400"
    />
    {label && (
      <span className="text-[9px] text-teal-400 font-bold uppercase tracking-wide mt-1">{label}</span>
    )}
  </div>
);

const HArrow: React.FC<{ label?: string; delay?: number }> = ({ label, delay = 0 }) => (
  <div className="flex items-center gap-0.5 shrink-0">
    <motion.div
      initial={{ width: 0, opacity: 0 }}
      whileInView={{ width: 20, opacity: 1 }}
      viewport={{ once: true }}
      transition={{ delay, duration: 0.3 }}
      className="h-px bg-gradient-to-r from-teal-400 to-sky-500"
    />
    <motion.div
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      transition={{ delay: delay + 0.15 }}
      className="w-0 h-0 border-t-[4px] border-b-[4px] border-l-[5px] border-t-transparent border-b-transparent border-l-teal-400"
    />
    {label && <span className="text-[8px] text-teal-400 font-bold ml-1">{label}</span>}
  </div>
);

// ─────────────────────────────────────────────────────────
// Landing Navbar
// ─────────────────────────────────────────────────────────
const LandingNavbar: React.FC = () => {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);
  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
    setMenuOpen(false);
  };
  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled ? 'bg-white/90 backdrop-blur-xl border-b border-slate-200 shadow-sm' : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Logo */}
        <a href="/" className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-teal-500 to-sky-600 flex items-center justify-center shadow-md">
            <span className="text-white font-black text-sm">M</span>
          </div>
          <div>
            <p className="font-extrabold text-slate-900 text-sm leading-none">MedAgentix AI</p>
            <p className="text-teal-600 text-[9px] font-semibold leading-none mt-0.5">Multi-Agent CDSS</p>
          </div>
        </a>

        {/* Desktop nav */}
        <nav className="hidden md:flex items-center gap-7">
          {[
            { label: 'Features', id: 'features' },
            { label: 'Architecture', id: 'architecture' },
            { label: 'How It Works', id: 'how-it-works' },
          ].map(({ label, id }) => (
            <button
              key={id}
              onClick={() => scrollTo(id)}
              className="text-slate-600 hover:text-teal-600 text-sm font-medium transition-colors"
            >
              {label}
            </button>
          ))}
        </nav>

        {/* CTA */}
        <div className="hidden md:flex items-center gap-3">
          <a href="/login" className="text-sm font-semibold text-slate-700 hover:text-teal-600 transition-colors px-3 py-1.5">
            Sign In
          </a>
          <a
            href="/register"
            className="text-sm font-bold px-5 py-2 bg-gradient-to-r from-teal-500 to-sky-600 text-white rounded-xl shadow-md shadow-teal-500/20 hover:opacity-90 transition"
          >
            Get Started
          </a>
        </div>

        {/* Mobile menu toggle */}
        <button className="md:hidden p-2" onClick={() => setMenuOpen(!menuOpen)}>
          {menuOpen ? <X className="w-5 h-5 text-slate-700" /> : <Menu className="w-5 h-5 text-slate-700" />}
        </button>
      </div>

      {/* Mobile menu */}
      {menuOpen && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="md:hidden bg-white border-b border-slate-200 px-6 py-4 space-y-3"
        >
          {['Features', 'Architecture', 'How It Works'].map(label => (
            <button key={label} onClick={() => scrollTo(label.toLowerCase().replace(/ /g, '-'))}
              className="block w-full text-left text-slate-700 font-medium text-sm py-1">{label}</button>
          ))}
          <div className="flex gap-3 pt-2">
            <a href="/login" className="flex-1 text-center py-2 border border-slate-200 rounded-lg text-sm font-semibold">Sign In</a>
            <a href="/register" className="flex-1 text-center py-2 bg-teal-500 text-white rounded-lg text-sm font-bold">Register</a>
          </div>
        </motion.div>
      )}
    </motion.header>
  );
};

// ─────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────
const PARTICLES = Array.from({ length: 14 }, (_, i) => ({
  x: 5 + (i * 7) % 90, y: 5 + (i * 13) % 85, size: 3 + (i % 4), delay: i * 0.25,
}));

const STATS = [
  { label: 'Diseases Mapped', value: 721, suffix: '+' },
  { label: 'Pipeline Stages', value: 10, suffix: '' },
  { label: 'Drug Profiles', value: 500, suffix: '+' },
  { label: 'ICD-10 Codes', value: 200, suffix: '+' },
];

const FEATURES = [
  { icon: Brain, color: 'text-violet-500', bg: 'bg-violet-50', title: 'Multi-Agent AI Diagnosis', desc: 'Synthesizes outputs from 10 calibrated ML agents to produce differential diagnoses ranked by probability and ICD-10 code.' },
  { icon: Zap, color: 'text-amber-500', bg: 'bg-amber-50', title: 'ESI Triage Prioritization', desc: 'Emergency Severity Index scoring from Level 1 Resuscitation to Level 4 Routine — with real-time red-flag detection.' },
  { icon: Stethoscope, color: 'text-sky-500', bg: 'bg-sky-50', title: 'Clinical Triage Assessment', desc: 'GREEN / YELLOW / RED triage gating backed by Groq LLM explanations, lifestyle guidance, and escalation triggers.' },
  { icon: FileText, color: 'text-emerald-500', bg: 'bg-emerald-50', title: 'Downloadable CDSS Reports', desc: 'Automated prescription PDFs with drug dosages, diagnostic workup, pathophysiology, and physician sign-off records.' },
  { icon: UserCheck, color: 'text-rose-500', bg: 'bg-rose-50', title: 'Doctor Review & Sign-Off', desc: 'Doctors review AI assessments, add clinical notes, and sign off on cases — directly from the triage dashboard.' },
  { icon: ClipboardList, color: 'text-indigo-500', bg: 'bg-indigo-50', title: 'Electronic Health Records', desc: 'Secure case logs, physician annotations, triage history, and full audit traces per patient visit.' },
];

const TRUST = [
  { icon: Shield, label: 'HIPAA Compliant' },
  { icon: Lock, label: 'Encrypted' },
  { icon: Activity, label: 'Real-Time CDSS' },
  { icon: CheckCircle, label: 'ESI Protocol' },
  { icon: Heart, label: 'Patient-First' },
];

const HOW_STEPS = [
  { step: '01', title: 'Patient Self-Intake', desc: 'Patient enters vitals, symptoms, and health history through the guided intake form.' },
  { step: '02', title: 'AI Triage & Diagnosis', desc: '10-stage CDSS pipeline runs differential analysis, triage scoring, and drug recommendations.' },
  { step: '03', title: 'Clinical Report', desc: 'Patient receives a triage-aware report. Doctor reviews, annotates, and signs off on the case.' },
];

const itemVariants = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: 0.55, ease: 'easeOut' } },
};

// ─────────────────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────────────────
export default function LandingPage() {
  return (
    <div className="min-h-screen bg-white text-foreground overflow-x-hidden">
      <LandingNavbar />

      {/* ═══════════════════════════════════════════
          HERO — White, full-split layout
      ═══════════════════════════════════════════ */}
      <section className="relative min-h-[100vh] flex flex-col lg:flex-row items-center overflow-hidden bg-white pt-16">

        {/* Dot grid */}
        <div className="absolute inset-0 opacity-[0.04] pointer-events-none"
          style={{ backgroundImage: 'radial-gradient(circle, #0ea5e9 1px, transparent 1px)', backgroundSize: '32px 32px' }} />

        {/* Particles */}
        {PARTICLES.map((p, i) => <Particle key={i} {...p} />)}

        {/* Glow blobs */}
        <motion.div className="absolute top-0 left-0 w-[600px] h-[600px] bg-teal-300/10 rounded-full blur-3xl pointer-events-none"
          animate={{ scale: [1, 1.08, 1], x: [0, 20, 0] }} transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }} />
        <motion.div className="absolute bottom-0 right-0 w-[500px] h-[500px] bg-sky-300/10 rounded-full blur-3xl pointer-events-none"
          animate={{ scale: [1, 1.1, 1], x: [0, -20, 0] }} transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut', delay: 2 }} />

        {/* LEFT text */}
        <div className="relative z-10 flex-1 flex flex-col justify-center px-8 sm:px-14 lg:px-20 xl:px-24 py-24 lg:py-0">
          <motion.div initial={{ opacity: 0, y: -14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.55 }}
            className="inline-flex items-center gap-2 bg-teal-50 border border-teal-200 text-teal-700 px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider mb-6 self-start">
            <motion.span className="w-2 h-2 bg-teal-500 rounded-full"
              animate={{ scale: [1, 1.4, 1], opacity: [1, 0.5, 1] }} transition={{ duration: 1.5, repeat: Infinity }} />
            Multi-Agent CDSS · Powered by AI
          </motion.div>

          <motion.h1 initial={{ opacity: 0, y: 28 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15, duration: 0.7 }}
            className="text-6xl md:text-7xl xl:text-8xl font-black tracking-tight text-slate-900 leading-[1.05]">
            MedAgentix{' '}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-500 via-sky-400 to-indigo-500">AI</span>
          </motion.h1>

          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.25, duration: 0.65 }}
            className="mt-3 text-xl md:text-2xl font-semibold text-slate-500 tracking-tight">
            Multi-Agent Clinical Decision Support System
          </motion.p>

          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.35, duration: 0.65 }}
            className="mt-4 text-lg text-slate-500 max-w-lg leading-relaxed">
            10 specialized AI agents working in concert — delivering ESI triage, differential diagnostics,
            Groq-powered explanations, and clinical reports in seconds.
          </motion.p>

          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.45, duration: 0.6 }} className="mt-5">
            <EcgLine />
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5, duration: 0.6 }}
            className="mt-6 flex flex-col sm:flex-row gap-4">
            <motion.a href="/login" whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }}
              className="group inline-flex items-center justify-center gap-2.5 px-7 py-3.5 bg-gradient-to-r from-teal-500 to-sky-600 text-white font-bold rounded-xl shadow-lg shadow-teal-500/25 text-sm">
              Enter Clinical Portal
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </motion.a>
            <motion.a href="/register" whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }}
              className="inline-flex items-center justify-center gap-2 px-7 py-3.5 bg-white border border-slate-200 hover:border-teal-300 font-semibold rounded-xl text-slate-700 text-sm shadow-sm">
              <UserCheck className="w-4 h-4 text-teal-600" />
              Patient Registration
            </motion.a>
          </motion.div>

          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.7, duration: 0.6 }} className="mt-8 flex flex-wrap gap-3">
            {TRUST.map((t, i) => (
              <motion.div key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.7 + i * 0.08 }} className="flex items-center gap-1.5 text-slate-500 text-xs font-medium">
                <t.icon className="w-3.5 h-3.5 text-teal-500" />
                <span>{t.label}</span>
              </motion.div>
            ))}
          </motion.div>
        </div>

        {/* RIGHT illustration — badges OUTSIDE image div */}
        <motion.div initial={{ opacity: 0, x: 50 }} animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3, duration: 0.9, ease: 'easeOut' }}
          className="relative z-10 flex-1 flex items-center justify-center px-6 lg:pr-10 py-16 lg:py-8">
          <div className="relative" style={{ width: '100%', maxWidth: '620px', padding: '28px 36px' }}>
            <motion.div className="absolute inset-8 bg-gradient-to-br from-teal-400/20 via-sky-400/10 to-transparent rounded-3xl blur-2xl"
              animate={{ scale: [1, 1.04, 1] }} transition={{ duration: 5, repeat: Infinity, ease: 'easeInOut' }} />
            <img src="/hero-illustration.jpg" alt="MedAgentix AI Clinical Intelligence"
              className="relative z-10 w-full rounded-3xl shadow-2xl shadow-teal-500/15 border border-slate-100 object-cover" draggable={false} />
            {/* Badge: accuracy — bottom-left */}
            <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.9 }}
              whileHover={{ scale: 1.06 }} style={{ position: 'absolute', bottom: 0, left: 0, zIndex: 30 }}
              className="bg-white border border-slate-200 rounded-2xl px-4 py-3 shadow-xl flex items-center gap-3">
              <div className="w-9 h-9 bg-teal-100 rounded-xl flex items-center justify-center shrink-0"><Brain className="w-4 h-4 text-teal-600" /></div>
              <div><p className="text-[10px] font-bold text-slate-500 uppercase">AI Accuracy</p><p className="text-xl font-black text-slate-900">70%+</p></div>
            </motion.div>
            {/* Badge: diseases — top-right */}
            <motion.div initial={{ opacity: 0, y: -16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 1.0 }}
              whileHover={{ scale: 1.06 }} style={{ position: 'absolute', top: 0, right: 0, zIndex: 30 }}
              className="bg-white border border-slate-200 rounded-2xl px-4 py-3 shadow-xl flex items-center gap-3">
              <div className="w-9 h-9 bg-sky-100 rounded-xl flex items-center justify-center shrink-0"><Activity className="w-4 h-4 text-sky-600" /></div>
              <div><p className="text-[10px] font-bold text-slate-500 uppercase">Diseases</p><p className="text-xl font-black text-slate-900">721+</p></div>
            </motion.div>
            {/* Badge: triage — right-center */}
            <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 1.15 }}
              style={{ position: 'absolute', top: '50%', right: '-4px', transform: 'translateY(-50%)', zIndex: 30 }}
              className="bg-white border border-emerald-200 rounded-2xl px-3.5 py-2.5 shadow-xl">
              <div className="flex items-center gap-1.5">
                <motion.div className="w-2 h-2 bg-emerald-500 rounded-full"
                  animate={{ scale: [1, 1.5, 1], opacity: [1, 0.4, 1] }} transition={{ duration: 1.2, repeat: Infinity }} />
                <span className="text-[10px] font-extrabold text-emerald-700 uppercase tracking-wide">GREEN Triage</span>
              </div>
              <p className="text-[10px] text-slate-500 mt-0.5">No red flags detected</p>
            </motion.div>
          </div>
        </motion.div>
      </section>

      {/* ═══════════════════════════════════════════
          STATS — Dark navy
      ═══════════════════════════════════════════ */}
      <section className="relative bg-gradient-to-r from-slate-900 via-[#0d2e4a] to-slate-900 py-14 px-6 overflow-hidden">
        <div className="absolute inset-0 opacity-[0.06] pointer-events-none"
          style={{ backgroundImage: `linear-gradient(to right, #38bdf8 1px, transparent 1px), linear-gradient(to bottom, #38bdf8 1px, transparent 1px)`, backgroundSize: '36px 36px' }} />
        <motion.div initial="hidden" whileInView="show" viewport={{ once: true }}
          variants={{ show: { transition: { staggerChildren: 0.1 } } }}
          className="max-w-5xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8 text-center relative z-10">
          {STATS.map((s, i) => (
            <motion.div key={i} variants={itemVariants} className="space-y-1">
              <p className="text-3xl md:text-4xl font-black text-white"><AnimCounter to={s.value} suffix={s.suffix} /></p>
              <p className="text-sm text-teal-300 font-semibold">{s.label}</p>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* ═══════════════════════════════════════════
          MULTI-AGENT ARCHITECTURE — White
      ═══════════════════════════════════════════ */}
      <section id="architecture" className="py-24 px-6 bg-white overflow-hidden">
        <div className="max-w-7xl mx-auto">

          <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
            className="text-center mb-14 space-y-3">
            <div className="inline-flex items-center gap-2 bg-indigo-50 border border-indigo-200 text-indigo-700 px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider">
              <Cpu className="w-3.5 h-3.5" /> System Architecture
            </div>
            <h2 className="text-4xl font-extrabold text-slate-900 tracking-tight">
              MedAgentix AI —{' '}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-500 to-teal-500">Updated System Workflow</span>
            </h2>
            <p className="text-slate-500 max-w-2xl mx-auto text-sm leading-relaxed">
              With RAG Agent &amp; Agent Orchestrator · 10-stage multi-agent pipeline from patient intake to final clinical output
            </p>
          </motion.div>

          {/* Architecture diagram — dark background */}
          <div className="relative bg-gradient-to-br from-slate-900 via-[#0b1e35] to-slate-950 rounded-3xl border border-slate-700 overflow-hidden p-8">
            {/* Grid overlay */}
            <div className="absolute inset-0 opacity-[0.06] pointer-events-none rounded-3xl"
              style={{ backgroundImage: `linear-gradient(to right, #38bdf8 1px, transparent 1px), linear-gradient(to bottom, #38bdf8 1px, transparent 1px)`, backgroundSize: '24px 24px' }} />
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-40 bg-teal-500/8 blur-3xl pointer-events-none" />

            <div className="relative z-10 space-y-4">

              {/* ── ROW 1: User Access → Input Collection → OCR ── */}
              <div className="flex items-start justify-center gap-3 flex-wrap">
                {/* 1. User Access */}
                <ArchBlock icon={UserCheck} title="1. USER ACCESS" bullets={['Doctor / Patient Login', 'Role-Based Portal']}
                  color="text-teal-300" bg="bg-teal-500/10" border="border-teal-500/30" delay={0.05} badge="Entry" badgeColor="bg-teal-500/20 text-teal-300" />

                <HArrow delay={0.1} />

                {/* 2. Input Collection */}
                <motion.div initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
                  transition={{ delay: 0.15 }}
                  className="bg-sky-500/10 border border-sky-500/30 rounded-xl p-3.5 flex flex-col gap-2" style={{ minWidth: 340 }}>
                  <p className="text-[11px] font-bold text-white text-center border-b border-sky-500/20 pb-2 mb-1">2. INPUT COLLECTION (Multi-Modal)</p>
                  <div className="grid grid-cols-4 gap-2">
                    {[
                      { icon: ScanLine, label: 'Symptoms', sub: 'Text/Voice/Form' },
                      { icon: UserCheck, label: 'Patient Profile', sub: 'Age, Gender, History' },
                      { icon: FileText, label: 'Medical Reports', sub: 'Images / PDFs' },
                      { icon: MonitorPlay, label: 'Chatbot Input', sub: 'Free Text' },
                    ].map((item, i) => (
                      <div key={i} className="flex flex-col items-center gap-1 bg-white/5 rounded-lg p-2 text-center">
                        <item.icon className="w-4 h-4 text-sky-400" />
                        <p className="text-[9px] font-bold text-white leading-tight">{item.label}</p>
                        <p className="text-[8px] text-slate-400">{item.sub}</p>
                      </div>
                    ))}
                  </div>
                </motion.div>

                <HArrow delay={0.2} />

                {/* 3. OCR Processing */}
                <ArchBlock icon={Microscope} title="3. OCR PROCESSING" bullets={['Extract Text & Values', 'Validate Ranges', 'Convert to Structured Data']}
                  color="text-violet-300" bg="bg-violet-500/10" border="border-violet-500/30" delay={0.25} badge="Pre-Process" badgeColor="bg-violet-500/20 text-violet-300" />
              </div>

              <Arrow delay={0.3} />

              {/* ── ROW 2: Agent Orchestrator ── */}
              <motion.div initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
                transition={{ delay: 0.35 }}
                className="bg-gradient-to-r from-indigo-500/20 to-purple-500/20 border-2 border-indigo-400/40 rounded-xl p-4">
                <div className="text-center mb-3">
                  <span className="inline-flex items-center gap-2 text-[12px] font-extrabold text-white">
                    <GitMerge className="w-4 h-4 text-indigo-400" />
                    4. AGENT ORCHESTRATOR
                    <span className="text-[9px] text-slate-400 font-normal">(Coordinates all agents, routes data, resolves dependencies, aggregates results)</span>
                  </span>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
                  {[
                    { icon: ScanLine, n: '4.1', label: 'Symptom Agent', bullets: ['Normalize', 'Extract Features', 'Severity Detection'], color: 'text-sky-400', bg: 'bg-sky-500/10', border: 'border-sky-500/25' },
                    { icon: BarChart3, n: '4.2', label: 'Risk Factor Agent', bullets: ['Analyze History', 'Lifestyle Factors', 'Identify Risks'], color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/25' },
                    { icon: Activity, n: '4.3', label: 'Temporal Agent', bullets: ['Analyze Duration', 'Time Patterns', 'Progression Analysis'], color: 'text-orange-400', bg: 'bg-orange-500/10', border: 'border-orange-500/25' },
                    { icon: GitMerge, n: '4.4', label: 'Differential Agent', bullets: ['Generate Possibilities', 'Compare Diseases', 'Rank Candidates'], color: 'text-indigo-400', bg: 'bg-indigo-500/10', border: 'border-indigo-500/25' },
                    { icon: Zap, n: '4.5', label: 'Emergency Agent', bullets: ['Detect Critical Cond.', 'Risk Level H/M/L', 'Immediate Alerts'], color: 'text-red-400', bg: 'bg-red-500/10', border: 'border-red-500/25' },
                    { icon: FileText, n: '4.6', label: 'OCR/Report Agent', bullets: ['Extract Lab Values', 'Validate Results', 'Abnormality Detection'], color: 'text-cyan-400', bg: 'bg-cyan-500/10', border: 'border-cyan-500/25' },
                    { icon: BookOpen, n: '4.7', label: 'RAG Knowledge Agent', bullets: ['Retrieve Med. Knowledge', 'Guidelines & Literature', 'Support Reasoning'], color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/25' },
                    { icon: Pill, n: '4.8', label: 'Drug Rec. Agent', bullets: ['Suggest Medications', 'Check Contraindications', 'Drug Interaction'], color: 'text-rose-400', bg: 'bg-rose-500/10', border: 'border-rose-500/25' },
                  ].map((a, i) => (
                    <ArchBlock key={i} icon={a.icon} title={`${a.n} ${a.label}`} bullets={a.bullets}
                      color={a.color} bg={a.bg} border={a.border} delay={0.4 + i * 0.05} small />
                  ))}
                </div>
              </motion.div>

              <Arrow delay={0.85} />

              {/* ── ROW 3: Primary Diagnostic Engine ── */}
              <div className="flex items-start justify-center gap-0">
                <ArchBlock icon={Cpu} title="5. PRIMARY DIAGNOSTIC ENGINE"
                  bullets={['ML Models: XGBoost / RF / LightGBM', 'Train on Clinical Datasets', 'Predict & Rank Diseases']}
                  color="text-teal-300" bg="bg-teal-500/10" border="border-teal-500/30" delay={0.9}
                  badge="Core ML" badgeColor="bg-teal-500/20 text-teal-300" />
              </div>

              <Arrow delay={0.95} />

              {/* ── ROW 4: Confidence Evaluation → two branches ── */}
              <div className="flex flex-col items-center gap-2">
                <ArchBlock icon={CheckCircle} title="6. CONFIDENCE EVALUATION"
                  bullets={['Evaluate Prediction Confidence', 'HIGH / MEDIUM / LOW']}
                  color="text-amber-300" bg="bg-amber-500/15" border="border-amber-500/40" delay={1.0}
                  badge="Decision Gate" badgeColor="bg-amber-500/20 text-amber-300" />

                <div className="flex items-start gap-6 mt-1">
                  {/* Branch: High confidence */}
                  <div className="flex flex-col items-center gap-1.5">
                    <div className="flex items-center gap-1.5">
                      <motion.div initial={{ height: 0 }} whileInView={{ height: 32 }} viewport={{ once: true }}
                        transition={{ delay: 1.05 }} className="w-px bg-emerald-500" />
                    </div>
                    <div className="flex items-center gap-1 text-emerald-400 text-[9px] font-bold">
                      <CheckCircle className="w-3 h-3" /> High Confidence ≥ 85%
                    </div>
                    <ArchBlock icon={Brain} title="7A. USE ML PREDICTION"
                      bullets={['Use Primary Model Output', 'Ranked Diagnoses']}
                      color="text-emerald-300" bg="bg-emerald-500/10" border="border-emerald-500/30" delay={1.1}
                      badge="High Conf" badgeColor="bg-emerald-500/20 text-emerald-300" />
                  </div>

                  <div className="w-px h-20 bg-white/10 self-center" />

                  {/* Branch: Low confidence */}
                  <div className="flex flex-col items-center gap-1.5">
                    <div className="flex items-center gap-1 text-rose-400 text-[9px] font-bold">
                      <Zap className="w-3 h-3" /> Low Confidence &lt; 85%
                    </div>
                    <ArchBlock icon={FlaskConical} title="7B. LLM FALLBACK REASONING"
                      bullets={['Meditron 7B / BioGPT / PubMedBERT', 'Advanced Medical Reasoning', 'Handle Ambiguity & Rare Cases', 'Generate Alternative Diagnoses']}
                      color="text-rose-300" bg="bg-rose-500/10" border="border-rose-500/30" delay={1.15}
                      badge="LLM Fallback" badgeColor="bg-rose-500/20 text-rose-300" />
                  </div>
                </div>
              </div>

              <Arrow delay={1.2} />

              {/* ── ROW 5: XAI → Doctor → Final Output ── */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <ArchBlock icon={BarChart3} title="8. EXPLAINABLE AI (XAI)"
                  bullets={['SHAP / LIME Analysis', 'Feature Importance', 'Reasoning & Confidence Scores']}
                  color="text-violet-300" bg="bg-violet-500/10" border="border-violet-500/30" delay={1.25}
                  badge="Interpretability" badgeColor="bg-violet-500/20 text-violet-300" />
                <ArchBlock icon={Stethoscope} title="9. DOCTOR-IN-THE-LOOP"
                  bullets={['Review Results', 'Validate / Correct', 'Add Feedback & Sign-Off']}
                  color="text-sky-300" bg="bg-sky-500/10" border="border-sky-500/30" delay={1.3}
                  badge="Human Review" badgeColor="bg-sky-500/20 text-sky-300" />
                <ArchBlock icon={MonitorPlay} title="10. FINAL OUTPUT TO USER"
                  bullets={['Dashboard / Chatbot Interface', 'Disease Ranking + ICD-10', 'Test Suggestions & Risk Alerts', 'Explanation & References (RAG)']}
                  color="text-teal-300" bg="bg-teal-500/15" border="border-teal-500/40" delay={1.35}
                  badge="Output" badgeColor="bg-teal-500/20 text-teal-300" />
              </div>

              {/* Key Technologies footer */}
              <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}
                transition={{ delay: 1.5 }}
                className="mt-4 border-t border-white/10 pt-4 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
                {[
                  { label: 'NLP', value: 'ClinicalBERT' },
                  { label: 'ML Models', value: 'XGBoost · RF · LightGBM' },
                  { label: 'LLMs', value: 'Meditron · BioGPT' },
                  { label: 'RAG', value: 'PubMedBERT · FAISS' },
                  { label: 'Orchestration', value: 'LangGraph · PostgreSQL' },
                ].map((t, i) => (
                  <div key={i} className="bg-white/5 border border-white/10 rounded-lg px-3 py-2">
                    <p className="text-[9px] font-bold text-teal-400 uppercase tracking-wide">{t.label}</p>
                    <p className="text-[10px] text-slate-300 font-medium mt-0.5">{t.value}</p>
                  </div>
                ))}
              </motion.div>

              {/* Legend */}
              <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}
                transition={{ delay: 1.55 }}
                className="flex flex-wrap justify-center gap-4 pt-2">
                {[
                  { color: 'border-slate-400', dash: '', label: '→  Data Flow' },
                  { color: 'border-slate-500', dash: 'dashed', label: '- - Fallback / Alternate Flow' },
                  { color: 'border-slate-600', dash: 'dotted', label: '··· Feedback Loop' },
                ].map((l, i) => (
                  <span key={i} className="text-[10px] text-slate-500 font-medium">{l.label}</span>
                ))}
              </motion.div>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════
          FEATURES — Slate-50
      ═══════════════════════════════════════════ */}
      <section id="features" className="py-24 px-6 bg-slate-50">
        <motion.div initial="hidden" whileInView="show" viewport={{ once: true }}
          variants={{ show: { transition: { staggerChildren: 0.08 } } }} className="max-w-6xl mx-auto">
          <motion.div variants={itemVariants} className="text-center mb-14 space-y-3">
            <p className="text-teal-600 text-xs font-bold uppercase tracking-widest">What MedAgentix Does</p>
            <h2 className="text-4xl font-extrabold text-slate-900 tracking-tight">
              Clinical-Grade AI,{' '}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-500 to-sky-500">End-to-End</span>
            </h2>
          </motion.div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {FEATURES.map((f, i) => (
              <motion.div key={i} variants={itemVariants} whileHover={{ y: -6 }}
                className="group bg-white border border-slate-200 hover:border-teal-200 rounded-2xl p-6 shadow-sm hover:shadow-md transition-all space-y-4 cursor-default">
                <motion.div className={`w-11 h-11 rounded-xl ${f.bg} flex items-center justify-center`}
                  whileHover={{ rotate: 5, scale: 1.1 }} transition={{ type: 'spring', stiffness: 300 }}>
                  <f.icon className={`w-5 h-5 ${f.color}`} />
                </motion.div>
                <div className="space-y-1.5">
                  <h3 className="font-bold text-slate-800 text-sm group-hover:text-teal-700 transition-colors">{f.title}</h3>
                  <p className="text-xs text-slate-500 leading-relaxed">{f.desc}</p>
                </div>
                <div className="flex items-center gap-1 text-teal-600 text-xs font-semibold opacity-0 group-hover:opacity-100 transition-opacity">
                  Learn more <ChevronRight className="w-3 h-3" />
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      {/* ═══════════════════════════════════════════
          HOW IT WORKS — White
      ═══════════════════════════════════════════ */}
      <section id="how-it-works" className="py-24 px-6 bg-white relative overflow-hidden">
        <div className="absolute inset-0 opacity-[0.03] pointer-events-none"
          style={{ backgroundImage: 'radial-gradient(circle, #0ea5e9 1px, transparent 1px)', backgroundSize: '36px 36px' }} />
        <motion.div initial="hidden" whileInView="show" viewport={{ once: true }}
          variants={{ show: { transition: { staggerChildren: 0.12 } } }} className="max-w-5xl mx-auto relative z-10">
          <motion.div variants={itemVariants} className="text-center mb-14 space-y-3">
            <p className="text-teal-600 text-xs font-bold uppercase tracking-widest">Workflow</p>
            <h2 className="text-4xl font-extrabold text-slate-900 tracking-tight">How It Works</h2>
          </motion.div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
            <div className="hidden md:block absolute top-8 left-[18%] right-[18%] h-px bg-gradient-to-r from-teal-200 via-sky-300 to-teal-200" />
            {HOW_STEPS.map((s, i) => (
              <motion.div key={i} variants={itemVariants} whileHover={{ y: -4 }} className="text-center space-y-4 cursor-default">
                <motion.div className="inline-flex flex-col items-center" whileHover={{ scale: 1.05 }} transition={{ type: 'spring', stiffness: 300 }}>
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-teal-500 to-sky-600 flex items-center justify-center shadow-lg shadow-teal-400/25 relative z-10">
                    <span className="text-white font-black text-lg">{s.step}</span>
                  </div>
                </motion.div>
                <div className="space-y-1">
                  <h3 className="font-bold text-slate-800 text-sm">{s.title}</h3>
                  <p className="text-xs text-slate-500 leading-relaxed max-w-xs mx-auto">{s.desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      {/* ═══════════════════════════════════════════
          CTA — Dark
      ═══════════════════════════════════════════ */}
      <section className="py-20 px-6 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-[#0d2e4a] to-[#0a3d62]" />
        <div className="absolute inset-0 opacity-[0.07] pointer-events-none"
          style={{ backgroundImage: `linear-gradient(to right, #38bdf8 1px, transparent 1px), linear-gradient(to bottom, #38bdf8 1px, transparent 1px)`, backgroundSize: '32px 32px' }} />
        <motion.div className="absolute top-0 left-1/2 -translate-x-1/2 w-[700px] h-64 bg-teal-500/10 rounded-full blur-3xl pointer-events-none"
          animate={{ scale: [1, 1.08, 1] }} transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }} />
        <motion.div initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.7 }}
          className="max-w-3xl mx-auto text-center relative z-10 space-y-6">
          <p className="text-teal-400 text-xs font-bold uppercase tracking-widest">Get Started</p>
          <h2 className="text-4xl md:text-5xl font-extrabold text-white leading-tight">
            Ready to Elevate<br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-300 to-sky-400">Patient Care?</span>
          </h2>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <motion.a href="/login" whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }}
              className="group inline-flex items-center justify-center gap-2.5 px-8 py-4 bg-gradient-to-r from-teal-400 to-sky-500 text-white font-bold rounded-2xl shadow-lg text-sm">
              Enter Clinical Portal <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </motion.a>
            <motion.a href="/register" whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }}
              className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-white/10 hover:bg-white/15 border border-white/20 font-semibold rounded-2xl text-white text-sm backdrop-blur-sm">
              Patient Registration
            </motion.a>
          </div>
        </motion.div>
      </section>

      {/* FOOTER */}
      <footer className="bg-white border-t border-slate-200 py-8 px-6">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-lg bg-teal-50 border border-teal-200 text-teal-600 flex items-center justify-center font-bold text-sm">M</div>
            <span className="font-semibold text-slate-700">MedAgentix AI</span>
            <span>·</span><span>Clinical Decision Support Platform</span>
          </div>
          <div className="flex items-center gap-4">
            {TRUST.slice(0, 3).map((t, i) => (
              <div key={i} className="flex items-center gap-1"><t.icon className="w-3 h-3 text-teal-500" /><span>{t.label}</span></div>
            ))}
          </div>
          <p>© {new Date().getFullYear()} MedAgentix. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
