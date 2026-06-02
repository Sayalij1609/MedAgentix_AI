# MedAgentix AI — Frontend UI Design Guide

> **Version**: 2.0 | **Last Updated**: June 2026
> **Framework**: Flask + Jinja2 Templates | **Styling**: Vanilla CSS + CSS Variables
> **Font Stack**: Inter (Google Fonts) | **Icons**: Lucide Icons (CDN)

---

## Table of Contents

1. [Design System](#1-design-system)
2. [Page Flow & Navigation](#2-page-flow--navigation)
3. [Landing Page](#3-landing-page)
4. [Input Form — Shared](#4-input-form--shared)
5. [Patient Dashboard — Output](#5-patient-dashboard--output)
6. [Doctor Dashboard — Output](#6-doctor-dashboard--output)
7. [Component Library](#7-component-library)
8. [Responsive Breakpoints](#8-responsive-breakpoints)
9. [File Structure](#9-file-structure)
10. [PDF Prescription Report Download](#10-pdf-prescription-report-download)

---

## 1. Design System

### 1.1 Color Palette

```
┌─────────────────────────────────────────────────────────┐
│  PATIENT DASHBOARD (Warm, Calming, Accessible)          │
├─────────────────────────────────────────────────────────┤
│  Primary         #2563EB   (Trustworthy Blue)           │
│  Primary Light   #DBEAFE   (Card backgrounds)           │
│  Accent          #059669   (Success / Good)             │
│  Warning         #F59E0B   (Moderate / Caution)         │
│  Danger          #DC2626   (Emergency / Urgent)         │
│  Background      #F8FAFC   (Light gray-white)           │
│  Card BG         #FFFFFF   (Clean white cards)          │
│  Text Primary    #1E293B   (Dark readable text)         │
│  Text Secondary  #64748B   (Subdued labels)             │
│  Border          #E2E8F0   (Subtle card borders)        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  DOCTOR DASHBOARD (Clinical, Professional, Dense)       │
├─────────────────────────────────────────────────────────┤
│  Primary         #0F172A   (Dark navy header)           │
│  Primary Light   #1E293B   (Sidebar / section headers)  │
│  Accent Clinical #6366F1   (Indigo — section markers)   │
│  Accent Data     #06B6D4   (Cyan — metrics & scores)    │
│  Alert High      #EF4444   (Critical / Emergency)       │
│  Alert Medium    #F97316   (Orange — Warning)           │
│  Alert Low       #22C55E   (Green — Normal)             │
│  Background      #F1F5F9   (Soft clinical gray)         │
│  Card BG         #FFFFFF   (Data cards)                 │
│  Text Primary    #0F172A   (Maximum contrast)           │
│  Text Label      #475569   (Field labels)               │
│  Border          #CBD5E1   (Section dividers)           │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Typography

```css
:root {
  --font-primary: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;

  /* Patient Dashboard — larger, friendlier */
  --fs-hero:    2.25rem;   /* 36px — page title */
  --fs-section: 1.375rem;  /* 22px — section headers */
  --fs-body:    1rem;      /* 16px — main content */
  --fs-label:   0.875rem;  /* 14px — field labels */
  --fs-small:   0.75rem;   /* 12px — captions */

  /* Doctor Dashboard — denser, clinical */
  --fs-doc-title:   1.75rem;
  --fs-doc-section: 1.125rem;
  --fs-doc-body:    0.9375rem;  /* 15px */
  --fs-doc-data:    0.8125rem;  /* 13px — table data */
  --fs-doc-label:   0.75rem;    /* 12px */
}
```

### 1.3 Spacing & Layout

```css
:root {
  --radius-sm: 6px;
  --radius-md: 12px;
  --radius-lg: 16px;
  --radius-xl: 24px;

  --shadow-card: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
  --shadow-hover: 0 10px 25px rgba(0,0,0,0.08);
  --shadow-section: 0 4px 12px rgba(0,0,0,0.05);

  --max-width: 1200px;
  --sidebar-width: 280px;
}
```

---

## 2. Page Flow & Navigation

```
                 ┌──────────────┐
                 │ Landing Page │
                 │  (Choose     │
                 │   Role)      │
                 └──────┬───────┘
                        │
            ┌───────────┴───────────┐
            │                       │
    ┌───────▼────────┐    ┌────────▼────────┐
    │ Patient Portal │    │ Doctor Portal   │
    │ (Symptom Form) │    │ (Clinical Form) │
    └───────┬────────┘    └────────┬────────┘
            │                      │
    ┌───────▼────────┐    ┌────────▼────────┐
    │   Processing   │    │   Processing    │
    │  (Animation)   │    │  (Pipeline Log) │
    └───────┬────────┘    └────────┬────────┘
            │                      │
    ┌───────▼────────┐    ┌────────▼────────┐
    │ Patient Report │    │ Clinical Report │
    │ (Friendly)     │    │ (Detailed)      │
    └────────────────┘    └─────────────────┘
```

### URL Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Landing page with role selection |
| `/patient` | GET | Patient symptom input form |
| `/doctor` | GET | Doctor clinical input form |
| `/api/predict` | POST | Run ML pipeline (JSON) |
| `/patient/result` | POST | Render patient-friendly report |
| `/doctor/result` | POST | Render clinical report |

---

## 3. Landing Page

### Layout

```
┌──────────────────────────────────────────────────────────────┐
│  ╔════════════════════════════════════════════════════════╗   │
│  ║        🏥  MedAgentix AI                              ║   │
│  ║    Intelligent Health Assessment System                ║   │
│  ║    Powered by Multi-Agent AI + Clinical ML            ║   │
│  ╚════════════════════════════════════════════════════════╝   │
│                                                              │
│    ┌─────────────────────┐    ┌─────────────────────┐        │
│    │   👤  I am a        │    │   🩺  I am a        │        │
│    │       PATIENT       │    │       DOCTOR        │        │
│    │                     │    │                     │        │
│    │  "Get a friendly    │    │  "Access detailed   │        │
│    │   health checkup    │    │   clinical analysis │        │
│    │   report"           │    │   with medical      │        │
│    │                     │    │   terminology"      │        │
│    │   [ Start Check ] → │    │   [ Open Portal ] → │        │
│    └─────────────────────┘    └─────────────────────┘        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  How it works:                                       │    │
│  │  ① Enter symptoms  ② AI analyzes  ③ Get report     │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  Footer: "AI-assisted tool. Not a replacement for doctors." │
└──────────────────────────────────────────────────────────────┘
```

### Design Notes
- **Hero gradient**: `linear-gradient(135deg, #1E3A5F 0%, #2563EB 100%)`
- **Role cards**: White with blue border-left (patient) / indigo border-left (doctor)
- **Hover effect**: Card lifts with `transform: translateY(-4px)` + shadow
- **"How it works"**: Horizontal stepper with numbered circles connected by dashed line

---

## 4. Input Form — Shared

Both dashboards use the SAME input form structure, but with different styling:
- **Patient**: Warm, large fields, helper text visible, tooltips on hover
- **Doctor**: Compact, more fields visible, clinical labels

### 4.1 Form Layout (Multi-Step or Single-Page Accordion)

```
┌──────────────────────────────────────────────────────────────┐
│  Step Progress Bar:  ① Patient Info  ② Symptoms  ③ Vitals   │
│                        ●━━━━━━━━━○━━━━━━━━━○                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  STEP 1: PATIENT INFORMATION                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Full Name    │  │ Age          │  │ Gender       │       │
│  │ [__________] │  │ [____] years │  │ [▼ Select  ] │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
│                              [ Next Step → ]                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  STEP 2: DESCRIBE YOUR SYMPTOMS                             │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Tell us what you're feeling (in your own words):     │    │
│  │ ┌──────────────────────────────────────────────────┐ │    │
│  │ │ I have been having fever, cough, and body pain   │ │    │
│  │ │ for the past 5 days. Also experiencing headache  │ │    │
│  │ │ and some difficulty breathing.                    │ │    │
│  │ └──────────────────────────────────────────────────┘ │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  Quick Symptom Selector (click to add):                      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ [🔥 Fever] [😷 Cough] [😫 Fatigue] [🤕 Headache]    │    │
│  │ [💔 Chest Pain] [😤 Breathlessness] [🤢 Nausea]     │    │
│  │ [🤧 Runny Nose] [💪 Body Pain] [🤮 Vomiting]        │    │
│  │ [🩸 Diarrhea] [🦴 Joint Pain] [🔴 Skin Rash]        │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  Selected Symptoms (with duration):                          │
│  ┌────────────────────────────┬──────────────────────┐       │
│  │  🔥 Fever                  │  Duration: [5 days▼] │       │
│  │  😷 Cough                  │  Duration: [5 days▼] │       │
│  │  😫 Fatigue                │  Duration: [3 days▼] │       │
│  │  🤕 Headache               │  Duration: [3 days▼] │       │
│  └────────────────────────────┴──────────────────────┘       │
│                                                              │
│                    [ ← Back ]  [ Next Step → ]               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  STEP 3: VITAL SIGNS & HISTORY                              │
│                                                              │
│  Vital Signs                                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐  │
│  │ Heart Rate   │ │ SpO2 (O2)    │ │ Blood Pressure       │  │
│  │ [___] bpm    │ │ [___] %      │ │ [___]/[___] mmHg     │  │
│  │ Normal:      │ │ Normal:      │ │ Normal:              │  │
│  │ 60-100       │ │ 95-100%      │ │ 120/80               │  │
│  └──────────────┘ └──────────────┘ └──────────────────────┘  │
│                                                              │
│  ┌──────────────┐ ┌──────────────┐                           │
│  │ Temperature  │ │ Cholesterol  │                           │
│  │ [___] °F     │ │ [___] mg/dL  │                           │
│  │ Normal:      │ │ Normal:      │                           │
│  │ 97-99 F      │ │ <200         │                           │
│  └──────────────┘ └──────────────┘                           │
│                                                              │
│  Medical History (check all that apply):                     │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ [ ] Diabetes  [ ] Hypertension  [ ] Cardiac History  │    │
│  │ [ ] Asthma    [ ] Thyroid       [ ] Kidney Disease   │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  Lifestyle Factors:                                          │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ [ ] Smoking  [ ] Alcohol  [ ] Obesity                │    │
│  │ [ ] Sedentary Lifestyle  [ ] Stress                  │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│               [ ← Back ]  [ 🔍 Analyze My Symptoms ]        │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Input Interaction Details

| Input | Type | Patient UI | Doctor UI |
|-------|------|-----------|-----------|
| Patient Name | Text | Large, friendly placeholder | Compact, clinical label |
| Age | Number + slider | Slider with number display | Number-only field |
| Gender | Radio buttons | Illustrated icons (👤/👩) | Standard dropdown |
| Symptom Text | Textarea | "Tell us how you feel..." | "Present complaints (H/O)..." |
| Quick Symptoms | Chip buttons | Emoji + colored chips | Plain text chips |
| Duration | Dropdown | "How long?" (1-2 days, etc.) | "Duration" (precise values) |
| Heart Rate | Number | With normal range hint | With clinical interpretation |
| SpO2 | Number | With color indicator | With hypoxemia threshold |
| BP | Text | Single field "120/80" | Split systolic/diastolic |
| Temperature | Number | F with fever indicator | F/C toggle |
| History | Checkboxes | Friendly names | ICD-coded conditions |
| Lifestyle | Checkboxes | Simple labels | Clinical risk factors |

### 4.3 Submit Button States

```
┌─────────────────────────┐     ┌─────────────────────────┐
│  🔍 Analyze My Symptoms │ ──► │  ⏳ Analyzing...        │
│      (Blue gradient)    │     │   (Pulsing animation)   │
└─────────────────────────┘     └─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Processing Screen:     │
│                         │
│    ┌─────────────────┐  │
│    │ ● Extracting    │  │
│    │   symptoms...   │  │
│    │ ● Analyzing     │  │
│    │   risk factors  │  │
│    │ ○ Running ML    │  │
│    │   prediction    │  │
│    │ ○ Generating    │  │
│    │   report        │  │
│    └─────────────────┘  │
│                         │
│   [Progress bar: ████░] │
└─────────────────────────┘
```

---

## 5. Patient Dashboard — Output

### Design Philosophy
> **Calming, simple, action-oriented.** Use friendly language.
> No medical jargon. Every section has an explanation.
> Large text, plenty of whitespace, soft colors.

### 5.1 Full Page Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Header: "Your Health Assessment Report"                     │
│  Patient: Rahul Sharma | Date: 02 Jun 2026                  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  📋 YOUR INFORMATION                       [Collapse] │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │  Name: Rahul Sharma       Age: 35        Gender: Male │  │
│  │                                                        │  │
│  │  Your Symptoms:                                        │  │
│  │  ● Fever (5 days) ● Cough (5 days) ● Headache (3d)   │  │
│  │                                                        │  │
│  │  Vital Signs:                                          │  │
│  │  ❤️ Heart Rate: 85 bpm ✅                               │  │
│  │  🫁 Oxygen: 96% ✅ (Normal range)                       │  │
│  │  🌡️ Temperature: 101.5°F ⚠️ (You have a fever)         │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  🔍 WHAT WE FOUND                                     │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │                                                        │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │  Possible Condition:  COVID-19                   │  │  │
│  │  │  Confidence:          ████████░░  Possible Match │  │  │
│  │  │  Severity:            ██████████  Serious        │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  │                                                        │  │
│  │  💡 What this means:                                   │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │ Your symptoms match a pattern commonly seen with │  │  │
│  │  │ respiratory viruses like COVID-19. This does NOT  │  │  │
│  │  │ confirm a diagnosis — only a lab test can do that │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  │                                                        │  │
│  │  📖 Understanding your condition:                      │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │ A viral infection that mainly affects your lungs  │  │  │
│  │  │ and airways. Your body is fighting the virus...   │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  │                                                        │  │
│  │  ⚠️ How serious is this?                               │  │
│  │  ┌─────────────── ORANGE CARD ──────────────────────┐  │  │
│  │  │ Your condition may need urgent medical care.      │  │  │
│  │  │ Please visit a doctor or hospital soon.           │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  🚨 EMERGENCY ALERT  (only if is_emergency = true)    │  │
│  ├────────────────── RED PULSING BORDER ──────────────────┤  │
│  │  ⚠️ PLEASE SEEK MEDICAL HELP NOW                      │  │
│  │                                                        │  │
│  │  • Call emergency services (112 / 108)                 │  │
│  │  • Go to the nearest hospital ER                       │  │
│  │  • Do NOT drive yourself                               │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  💊 POSSIBLE MEDICATIONS                               │  │
│  │  "Your doctor will decide what's right for you"        │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │                                                        │  │
│  │  ┌───────────────────────────────────────────┐         │  │
│  │  │ 1. Paracetamol                            │         │  │
│  │  │    What it does: To reduce fever           │         │  │
│  │  │    Usual dosage: 500-1000 mg every 4-6 hrs │         │  │
│  │  │    ⚠️ Watch for: Nausea (rare)             │         │  │
│  │  └───────────────────────────────────────────┘         │  │
│  │                                                        │  │
│  │  ┌───────────────────────────────────────────┐         │  │
│  │  │ 2. Zinc + Vitamin C                       │         │  │
│  │  │    What it does: Immune support            │         │  │
│  │  │    Usual dosage: Zinc 50mg + Vit C daily   │         │  │
│  │  │    ⚠️ Watch for: Nausea at high doses      │         │  │
│  │  └───────────────────────────────────────────┘         │  │
│  │                                                        │  │
│  │  ⚠️ NEVER take any medicine without doctor's approval  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  🧪 TESTS YOUR DOCTOR MAY ASK FOR                     │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │                                                        │  │
│  │  1. RT-PCR Test (Nasal Swab)                           │  │
│  │     Why: To confirm if you have COVID-19               │  │
│  │                                                        │  │
│  │  2. Chest X-Ray                                        │  │
│  │     Why: To check if your lungs are affected           │  │
│  │                                                        │  │
│  │  3. CBC (Blood Test)                                   │  │
│  │     Why: To check your blood cell counts               │  │
│  │                                                        │  │
│  │  💡 These tests help your doctor understand your       │  │
│  │     condition better                                   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  ✅ WHAT TO DO NEXT                                    │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │  1. Visit your doctor and share this report            │  │
│  │  2. Complete any tests your doctor recommends          │  │
│  │  3. Keep track of your symptoms                        │  │
│  │  4. Stay hydrated, rest, and eat healthy meals         │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  🚩 WHEN TO GET HELP RIGHT AWAY                       │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │  ❗ Difficulty breathing or shortness of breath        │  │
│  │  ❗ Severe chest pain or pressure                      │  │
│  │  ❗ High fever (above 103°F) that won't come down     │  │
│  │  ❗ Fainting or confusion                              │  │
│  │  ❗ Any sudden worsening of symptoms                   │  │
│  │                                                        │  │
│  │  → Call 112/108 or go to the nearest ER immediately   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────── GRAY BOX ───────────────────────┐  │
│  │  ℹ️ IMPORTANT NOTICE                                  │  │
│  │  This report was created by an AI health assistant.    │  │
│  │  It is NOT a medical diagnosis. Only a qualified       │  │
│  │  doctor can diagnose after proper examination.         │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  [ 📄 Download Report (PDF) ]  [ 🔄 New Assessment ]        │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 Confidence & Severity Visual Components

#### Confidence Bar (Patient)
```
Strong Match:     ████████████████████░░  90%+   (Green)
Good Match:       ████████████████░░░░░  80-90%  (Blue)
Possible Match:   ██████████████░░░░░░░  70-80%  (Yellow)
Needs Evaluation: ██████████░░░░░░░░░░░  50-70%  (Orange)
Uncertain:        █████░░░░░░░░░░░░░░░░  <50%    (Red)
```

#### Severity Badge (Patient)
```css
/* CSS for severity badges */
.severity-mild     { background: #DCFCE7; color: #166534; border: 1px solid #86EFAC; }
.severity-moderate { background: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }
.severity-serious  { background: #FED7AA; color: #9A3412; border: 1px solid #FB923C; }
.severity-urgent   { background: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5;
                     animation: pulse-border 2s infinite; }
```

---

## 6. Doctor Dashboard — Output

### Design Philosophy
> **Dense, data-rich, clinical.** Show all metrics and medical terms.
> Numbered sections mirror a clinical diagnostic report.
> Tables, badges, and structured data — no fluff.

### 6.1 Full Page Layout

```
┌──────────────────────────────────────────────────────────────┐
│  ┌─────────────┐                                             │
│  │ Sidebar     │  CLINICAL DIAGNOSTIC REPORT                │
│  │             │  Patient: RAHUL SHARMA                      │
│  │ § 1 Impress │  Report ID: MED-202606021644                │
│  │ § 2 Patho   │  Generated: 02-Jun-2026 16:44               │
│  │ § 3 Symptoms│─────────────────────────────────────────────│
│  │ § 4 Risk    │                                             │
│  │ § 5 Triage  │  PATIENT DATA                               │
│  │ § 6 ML      │  ┌──────────┬──────────┬──────────────────┐ │
│  │ § 7 DDx     │  │ Name     │ Age      │ Gender           │ │
│  │ § 8 Workup  │  │ R.Sharma │ 35 years │ Male             │ │
│  │ § 9 Pharma  │  └──────────┴──────────┴──────────────────┘ │
│  │ §10 Plan    │                                             │
│  │             │  VITALS                                     │
│  │ ────────    │  ┌─────────────┬──────────┬────────────────┐│
│  │ Alerts (3)  │  │ Heart Rate  │ 85 bpm   │ Normal sinus  ✅││
│  │ ──────── ─  │  │ SpO2        │ 96%      │ Normal        ✅││
│  └─────────────┘  │ BP          │ 120/80   │ Normotensive  ✅││
│                   │ Temp        │ 101.5 F  │ Pyrexia       🔴││
│                   │ Cholesterol │ 190      │ Borderline    🟡││
│                   └─────────────┴──────────┴────────────────┘│
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  SECTION 1: CLINICAL IMPRESSION                              │
│  ═══════════════════════════════════════════════════════════ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Primary Dx     COVID-19         ICD-10: U07.1        │  │
│  │  Confidence     ████████████░░░  70.0% [Moderate]     │  │
│  │  Severity       ██████████████   Severe               │  │
│  │  Category       Respiratory Viral Infection           │  │
│  │  Source         Meditron-7B LLM Fallback              │  │
│  │  Agreement      0.0 / 1.0                             │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Differential Considerations:                                │
│  ┌──────┬───────────────────────────┬──────────────────┐     │
│  │ Rank │ Condition                 │ Probability      │     │
│  ├──────┼───────────────────────────┼──────────────────┤     │
│  │ 1    │ Bronchial Asthma          │ 22.7%            │     │
│  │ 2    │ Heart attack              │ 6.8%             │     │
│  │ 3    │ Paralysis (brain hem.)    │ 3.8%             │     │
│  └──────┴───────────────────────────┴──────────────────┘     │
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  SECTION 2: PATHOPHYSIOLOGY & CLINICAL CONTEXT               │
│  ═══════════════════════════════════════════════════════════ │
│                                                              │
│  ┌─ Disease Mechanism ─────────────────────────────────────┐ │
│  │ SARS-CoV-2 binds ACE2 receptors on type II pneumocytes, │ │
│  │ triggering inflammatory cascade. Viral replication       │ │
│  │ causes diffuse alveolar damage, cytokine storm in       │ │
│  │ severe cases.                                            │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                              │
│  Key Markers:  CRP ↑ | Lymphopenia | D-dimer ↑             │
│  Prognosis:    Self-limiting in 85%; 10-15% need support    │
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  SECTION 3: CLINICAL SYMPTOM ANALYSIS                        │
│  ═══════════════════════════════════════════════════════════ │
│                                                              │
│  ┌───────────────────┬──────────┬──────────┬────────────┐    │
│  │ Symptom           │ Severity │ Duration │ Match Type │    │
│  ├───────────────────┼──────────┼──────────┼────────────┤    │
│  │ Fever             │ Moderate │ 5 days   │ NLP        │    │
│  │ Cough             │ Moderate │ 5 days   │ NLP        │    │
│  │ Fatigue           │ Moderate │ -        │ NLP        │    │
│  │ Headache          │ Moderate │ 3 days   │ NLP        │    │
│  │ Breathlessness    │ Moderate │ 2 days   │ NLP        │    │
│  └───────────────────┴──────────┴──────────┴────────────┘    │
│  Total: 5 symptoms | Correlation: Partial                    │
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  SECTION 4: RISK STRATIFICATION                              │
│  ═══════════════════════════════════════════════════════════ │
│                                                              │
│  Risk Level: [ LOW ▌ ]                                       │
│  Factors: 0 identified                                       │
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  SECTION 5: EMERGENCY TRIAGE                                 │
│  ═══════════════════════════════════════════════════════════ │
│                                                              │
│  Emergency: No     Triage: Level 3 (ESI)                    │
│  Progression: Stable | Urgency: High | Onset: N/A           │
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  SECTION 6: PREDICTIVE MODEL OUTPUT                          │
│  ═══════════════════════════════════════════════════════════ │
│                                                              │
│  Model: Voting Ensemble (XGBoost + RF + LightGBM)           │
│  ┌──────┬─────────────────────┬────────────┬─────────┐       │
│  │ Rank │ Disease             │ Probability│ Note    │       │
│  ├──────┼─────────────────────┼────────────┼─────────┤       │
│  │ 1    │ Peptic Ulcer        │ 33.95%     │ Primary │       │
│  │ 2    │ Flu                 │ 23.52%     │         │       │
│  │ 3    │ Stroke              │ 12.05%     │         │       │
│  └──────┴─────────────────────┴────────────┴─────────┘       │
│                                                              │
│  Active Features: fever=1, cough=1, fatigue=1, age=35...    │
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  SECTION 7: DIFFERENTIAL DIAGNOSIS & CLINICAL REASONING      │
│  ═══════════════════════════════════════════════════════════ │
│                                                              │
│  LLM Fallback: Yes | Model: Meditron-7B                     │
│                                                              │
│  ┌─ Clinical Reasoning ───────────────────────────────────┐  │
│  │ [Source: Knowledge-Based Clinical Analysis]             │  │
│  │                                                         │  │
│  │ Clinical assessment of a 35-year-old male presenting   │  │
│  │ with Fever, Cough, Fatigue. Pathological basis:        │  │
│  │ SARS-CoV-2 binds ACE2 receptors on type II pneumo-    │  │
│  │ cytes, triggering inflammatory cascade...              │  │
│  │                                                         │  │
│  │ Moderate confidence (70.0%) warrants confirmatory       │  │
│  │ investigations before definitive diagnosis.            │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  SECTION 8: DIAGNOSTIC WORKUP                                │
│  ═══════════════════════════════════════════════════════════ │
│                                                              │
│  ┌────┬────────────────────────────┬──────────┬───────────┐  │
│  │ #  │ Investigation              │ Priority │ Dept      │  │
│  ├────┼────────────────────────────┼──────────┼───────────┤  │
│  │ 1  │ RT-PCR Test                │ Primary  │ Molecular │  │
│  │    │ → Confirm SARS-CoV-2       │          │           │  │
│  │ 2  │ Rapid Antigen Test         │ Primary  │ Immunol.  │  │
│  │    │ → Quick screening          │          │           │  │
│  │ 3  │ Chest X-Ray                │ Primary  │ Radiology │  │
│  │    │ → Lung involvement         │          │           │  │
│  │ 4  │ CBC                        │ Second.  │ Hematol.  │  │
│  │    │ → Lymphopenia check        │          │           │  │
│  └────┴────────────────────────────┴──────────┴───────────┘  │
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  SECTION 9: PHARMACOTHERAPY                                  │
│  ═══════════════════════════════════════════════════════════ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ 1. Paracetamol [Antipyretic/Analgesic]                │  │
│  │    Dosage: 500-1000 mg q4-6h                          │  │
│  │    Route: Oral                                         │  │
│  │    ADR: Nausea (rare)                                  │  │
│  │    CI: Severe liver disease                            │  │
│  │    Note: First-line for fever and body ache            │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │ 2. Dexamethasone [Corticosteroid]                     │  │
│  │    Dosage: 6 mg OD x 10 days                          │  │
│  │    Route: Oral/IV                                      │  │
│  │    ADR: Hyperglycemia, insomnia                        │  │
│  │    CI: Mild cases not requiring O2                     │  │
│  │    Note: Only for severe cases with O2 support        │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  SECTION 10: MANAGEMENT PLAN                                 │
│  ═══════════════════════════════════════════════════════════ │
│                                                              │
│  Treatment:  Hospitalization + medication                    │
│  Urgency:    High                                            │
│  Follow-up:  Continuous monitoring + specialist consult      │
│  Risk:       Low                                             │
│  Emergency:  Standard care pathway                           │
│                                                              │
│  ┌─ CLINICAL ALERTS ──────────────────────────────────────┐  │
│  │ ⚠️ Paracetamol: CI in severe liver disease             │  │
│  │ ⚠️ Dexamethasone: CI in mild cases not requiring O2    │  │
│  │ 🔴 Severe condition — hospitalization may be required  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ FOOTER ───────────────────────────────────────────────┐  │
│  │ AI-Assisted Clinical Report | MedAgentix AI v2.0       │  │
│  │ Pipeline: ClinicalBERT NER + Voting Ensemble           │  │
│  │ Confidence Routing: >85% Direct | 70-85% Validated     │  │
│  │                      <70% Meditron-7B                   │  │
│  │ DISCLAIMER: Verify before prescribing.                  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  [ 📄 Download PDF ]  [ 🖨️ Print ]  [ 🔄 New Patient ]      │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Doctor-Specific Visual Components

#### Confidence Meter (Clinical)
```
High (>90%):       ████████████████████░  96.4%  🟢
Moderate-High:     ████████████████░░░░░  82.0%  🔵
Moderate (70-80%): ██████████████░░░░░░░  70.0%  🟡
Low-Moderate:      ██████████░░░░░░░░░░░  55.0%  🟠
Low (<50%):        █████░░░░░░░░░░░░░░░░  35.0%  🔴
```

#### Triage Level Badge
```css
.triage-1 { background: #DC2626; color: white; } /* Immediate */
.triage-2 { background: #F97316; color: white; } /* Emergent */
.triage-3 { background: #EAB308; color: black; } /* Urgent */
.triage-4 { background: #22C55E; color: white; } /* Less Urgent */
.triage-5 { background: #3B82F6; color: white; } /* Non-Urgent */
```

#### Vital Sign Interpretation Colors
```css
.vital-normal    { color: #16A34A; }  /* ✅ Normal range */
.vital-borderline { color: #CA8A04; } /* 🟡 Borderline */
.vital-abnormal  { color: #DC2626; }  /* 🔴 Abnormal */
.vital-critical  { color: #DC2626; font-weight: 700; animation: blink 1s infinite; }
```

---

## 7. Component Library

### 7.1 Card Component

```css
.card {
  background: var(--card-bg);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-card);
  border: 1px solid var(--border);
  overflow: hidden;
  transition: box-shadow 0.2s ease, transform 0.2s ease;
}

.card:hover {
  box-shadow: var(--shadow-hover);
  transform: translateY(-2px);
}

.card-header {
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 10px;
}

.card-body {
  padding: 20px;
}
```

### 7.2 Section Header (Doctor)

```css
.section-header {
  background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
  color: white;
  padding: 12px 20px;
  font-size: var(--fs-doc-section);
  font-weight: 600;
  letter-spacing: 0.5px;
  border-left: 4px solid #6366F1;
  margin: 24px 0 0;
}
```

### 7.3 Data Row (Key-Value)

```css
.data-row {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px dotted var(--border);
}
.data-row .label {
  color: var(--text-label);
  font-size: var(--fs-doc-label);
  text-transform: uppercase;
  letter-spacing: 0.3px;
}
.data-row .value {
  font-weight: 500;
  font-size: var(--fs-doc-body);
  text-align: right;
}
```

### 7.4 Alert Box

```css
.alert {
  padding: 14px 18px;
  border-radius: var(--radius-sm);
  margin: 8px 0;
  display: flex;
  align-items: flex-start;
  gap: 12px;
}
.alert-emergency {
  background: #FEF2F2;
  border: 2px solid #EF4444;
  color: #991B1B;
  animation: pulse-border 2s infinite;
}
.alert-warning {
  background: #FFFBEB;
  border-left: 4px solid #F59E0B;
  color: #92400E;
}
.alert-info {
  background: #EFF6FF;
  border-left: 4px solid #3B82F6;
  color: #1E40AF;
}
```

### 7.5 Symptom Chip (Input Form)

```css
.symptom-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border-radius: 20px;
  border: 2px solid #E2E8F0;
  background: white;
  cursor: pointer;
  transition: all 0.15s ease;
  font-size: 14px;
}
.symptom-chip:hover {
  border-color: #2563EB;
  background: #EFF6FF;
}
.symptom-chip.selected {
  border-color: #2563EB;
  background: #2563EB;
  color: white;
}
```

### 7.6 Progress Stepper

```css
.stepper {
  display: flex;
  justify-content: center;
  gap: 0;
  margin-bottom: 32px;
}
.step {
  display: flex;
  align-items: center;
  gap: 8px;
}
.step-circle {
  width: 36px; height: 36px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-weight: 700;
  border: 2px solid #CBD5E1;
  color: #94A3B8;
  background: white;
}
.step.active .step-circle {
  border-color: #2563EB;
  background: #2563EB;
  color: white;
}
.step.completed .step-circle {
  border-color: #22C55E;
  background: #22C55E;
  color: white;
}
.step-line {
  width: 80px; height: 2px;
  background: #CBD5E1;
}
.step.completed + .step-line {
  background: #22C55E;
}
```

---

## 8. Responsive Breakpoints

| Breakpoint | Width | Layout |
|-----------|-------|--------|
| Desktop | ≥1024px | Full sidebar + content (doctor) |
| Tablet | 768-1023px | Sidebar collapses to top nav |
| Mobile | <768px | Single column, stacked cards |

### Mobile Considerations
- **Patient**: Full single-column, larger touch targets (48px min)
- **Doctor**: Sidebar becomes horizontal tab bar, tables scroll horizontally
- **Input form**: Steps become full-page (one section per screen)

---

## 9. File Structure

```
MedAgentix_AI/
├── app.py                          # Flask application entry
├── templates/
│   ├── base.html                   # Base template (head, scripts, footer)
│   ├── landing.html                # Role selection page
│   ├── patient/
│   │   ├── input.html              # Patient symptom form
│   │   └── result.html             # Patient-friendly report
│   └── doctor/
│       ├── input.html              # Doctor clinical form
│       └── result.html             # Clinical diagnostic report
├── static/
│   ├── css/
│   │   ├── base.css                # Shared design system
│   │   ├── patient.css             # Patient-specific styles
│   │   └── doctor.css              # Doctor-specific styles
│   ├── js/
│   │   ├── form.js                 # Form validation + symptom chips
│   │   ├── stepper.js              # Multi-step form navigation
│   │   └── report.js               # Report interactions (PDF, print)
│   └── images/
│       └── logo.svg                # MedAgentix logo
├── api/
│   └── prediction_routes.py        # /api/predict endpoint
├── utils/
│   └── pdf_generator.py            # PDF generation (patient + doctor)
└── frontend_guide.md               # This file
```

---

## Key Design Principles

| Principle | Patient Dashboard | Doctor Dashboard |
|-----------|------------------|-----------------|
| **Language** | Simple, no jargon | Full medical terminology |
| **Density** | Spacious, breathable | Dense, data-rich |
| **Colors** | Warm blues, greens | Clinical navy, indigo |
| **Typography** | 16px body, 22px headers | 15px body, 18px headers |
| **Icons** | Emoji + colored | Minimal, monochrome |
| **Tables** | Avoided (use cards) | Prominent (structured data) |
| **Confidence** | Words ("Strong Match") | Numbers (96.4%, High) |
| **Medications** | "What it does" | Dosage, Route, ADR, CI |
| **Tests** | "Why" explanation | Priority, Department, Indication |
| **Emergency** | Pulsing red alert box | ESI Level badge + flags |
| **Sidebar** | None (scrollable page) | Sticky section navigator |
| **Actions** | "Download" + "New Check" | "PDF" + "Print" + "New Patient" |

---

## 10. PDF Prescription Report Download

### 10.1 Overview

Both dashboards include a **Download Report (PDF)** button that generates a professional
prescription-style PDF. The PDF content is role-specific:

| Feature | Patient PDF | Doctor PDF |
|---------|------------|------------|
| **Title** | "Health Assessment Report" | "Clinical Diagnostic Report" |
| **Language** | Simple, patient-friendly | Full medical terminology |
| **Header** | MedAgentix AI logo + patient name | Logo + Report ID + timestamp |
| **Watermark** | "AI-Generated — Not a Prescription" | "AI-Assisted — Verify Before Prescribing" |
| **Branding** | Soft blue header | Dark navy header |
| **Sections** | 7 sections (simplified) | 10 sections (full clinical) |
| **Page size** | A4 Portrait | A4 Portrait |
| **Filename** | `patient_report_<name>_<date>.pdf` | `clinical_report_<name>_<date>.pdf` |

### 10.2 Patient PDF Layout

```
┌──────────────────────────────────────────────────────────────┐
│  ┌────────────────────────────────────────────────────────┐  │
│  │  🏥 MedAgentix AI                                     │  │
│  │  Health Assessment Report          Date: 02-Jun-2026  │  │
│  │  ─────────────────────────────────────────────────────│  │
│  │  Patient: Rahul Sharma | Age: 35 | Gender: Male       │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  YOUR SYMPTOMS                                               │
│  ┌──────────────────────┬──────────────────────────────┐     │
│  │ Symptom              │ Duration                     │     │
│  ├──────────────────────┼──────────────────────────────┤     │
│  │ Fever                │ 5 days                       │     │
│  │ Cough                │ 5 days                       │     │
│  │ Fatigue              │ -                            │     │
│  │ Headache             │ 3 days                       │     │
│  │ Difficulty Breathing │ 2 days                       │     │
│  └──────────────────────┴──────────────────────────────┘     │
│                                                              │
│  VITAL SIGNS                                                 │
│  ┌──────────────────────┬──────────┬───────────────────┐     │
│  │ Measure              │ Value    │ Status            │     │
│  ├──────────────────────┼──────────┼───────────────────┤     │
│  │ Heart Rate           │ 85 bpm   │ ✅ Normal          │     │
│  │ Oxygen Level         │ 96%      │ ✅ Normal          │     │
│  │ Blood Pressure       │ 120/80   │ Normal            │     │
│  │ Temperature          │ 101.5°F  │ ⚠️ Fever           │     │
│  └──────────────────────┴──────────┴───────────────────┘     │
│                                                              │
│  ─────────────────────────────────────────────────────────── │
│  ASSESSMENT RESULT                                           │
│  ─────────────────────────────────────────────────────────── │
│                                                              │
│  Possible Condition:  COVID-19                               │
│  Match Confidence:    Possible Match (70%)                   │
│  Severity Level:      Serious                                │
│                                                              │
│  What this means:                                            │
│  Your symptoms match a pattern commonly seen with            │
│  respiratory viruses like COVID-19. This does NOT confirm     │
│  a diagnosis — only a lab test can do that.                  │
│                                                              │
│  Understanding your condition:                               │
│  A viral infection that mainly affects your lungs and         │
│  airways. Your body is fighting the virus, which is why      │
│  you feel tired and have a fever.                            │
│                                                              │
│  ─────────────────────────────────────────────────────────── │
│  SUGGESTED MEDICATIONS                                       │
│  (Your doctor will decide what's right for you)              │
│  ─────────────────────────────────────────────────────────── │
│                                                              │
│  ┌───────────────────┬───────────────────┬──────────────┐    │
│  │ Medicine           │ Dosage            │ Purpose      │    │
│  ├───────────────────┼───────────────────┼──────────────┤    │
│  │ Paracetamol        │ 500-1000mg q4-6h  │ Reduce fever │    │
│  │ Zinc + Vitamin C   │ Zinc 50mg + C 1g  │ Immunity     │    │
│  │ Dexamethasone      │ 6mg OD x 10 days  │ Inflammation │    │
│  └───────────────────┴───────────────────┴──────────────┘    │
│                                                              │
│  ⚠️ NEVER take any medicine without your doctor's approval   │
│                                                              │
│  ─────────────────────────────────────────────────────────── │
│  RECOMMENDED TESTS                                           │
│  ─────────────────────────────────────────────────────────── │
│                                                              │
│  1. RT-PCR Test — To confirm COVID-19                       │
│  2. Chest X-Ray — To check lung involvement                 │
│  3. CBC — To check blood cell counts                        │
│                                                              │
│  ─────────────────────────────────────────────────────────── │
│  WHAT TO DO NEXT                                             │
│  ─────────────────────────────────────────────────────────── │
│                                                              │
│  1. Visit your doctor and share this report                  │
│  2. Complete any tests your doctor recommends                │
│  3. Keep track of your symptoms                              │
│  4. Stay hydrated, rest, and eat healthy meals               │
│                                                              │
│  ─────────────────────────────────────────────────────────── │
│  ⚠️ WHEN TO GET HELP RIGHT AWAY                              │
│  ─────────────────────────────────────────────────────────── │
│                                                              │
│  ❗ Difficulty breathing             ❗ Severe chest pain    │
│  ❗ High fever (>103°F)              ❗ Fainting/confusion   │
│  → Call 112/108 or go to the nearest ER immediately          │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ ℹ️ DISCLAIMER: This report was created by an AI health │  │
│  │ assistant. It is NOT a medical diagnosis. Only a       │  │
│  │ qualified doctor can diagnose your condition.          │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ─────────────────── Page Footer ──────────────────────────  │
│  MedAgentix AI v2.0 | AI-Generated | Not a Prescription     │
│  Report Date: 02-Jun-2026 | Page 1/1                        │
└──────────────────────────────────────────────────────────────┘
```

### 10.3 Doctor PDF Layout

```
┌──────────────────────────────────────────────────────────────┐
│  ┌────────────────────────────────────────────────────────┐  │
│  │  🏥 MedAgentix AI — CLINICAL DIAGNOSTIC REPORT        │  │
│  │  Report ID: MED-202606021644      Date: 02-Jun-2026   │  │
│  │  ─────────────────────────────────────────────────────│  │
│  │  Patient: RAHUL SHARMA | Age: 35 | Sex: Male          │  │
│  │  PMH: None documented | Social: None documented       │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  PRESENTING COMPLAINTS                                       │
│  ┌──────────────────────┬──────────┬──────────────────────┐  │
│  │ Symptom              │ Duration │ Match Type           │  │
│  ├──────────────────────┼──────────┼──────────────────────┤  │
│  │ Fever                │ 5 days   │ NLP extraction       │  │
│  │ Cough                │ 5 days   │ NLP extraction       │  │
│  │ Fatigue              │ -        │ NLP extraction       │  │
│  │ Headache             │ 3 days   │ NLP extraction       │  │
│  │ Breathlessness       │ 2 days   │ NLP extraction       │  │
│  └──────────────────────┴──────────┴──────────────────────┘  │
│                                                              │
│  VITALS                                                      │
│  ┌────────────────┬──────────┬─────────────────────────┐     │
│  │ Parameter      │ Value    │ Clinical Interpretation │     │
│  ├────────────────┼──────────┼─────────────────────────┤     │
│  │ HR             │ 85 bpm   │ Normal sinus            │     │
│  │ SpO2           │ 96%      │ Normal                  │     │
│  │ BP             │ 120/80   │ Normotensive            │     │
│  │ Temp           │ 101.5°F  │ Pyrexia                 │     │
│  │ Cholesterol    │ 190      │ Borderline              │     │
│  └────────────────┴──────────┴─────────────────────────┘     │
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  § 1  CLINICAL IMPRESSION                                    │
│  ═══════════════════════════════════════════════════════════ │
│  Primary Dx:    COVID-19 (ICD-10: U07.1)                    │
│  Confidence:    70.0% [Moderate (70-80%)]                   │
│  Severity:      Severe                                       │
│  Category:      Respiratory Viral Infection                  │
│  Source:        Meditron-7B LLM Fallback                    │
│  Agreement:     0.0 / 1.0                                    │
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  § 2  PATHOPHYSIOLOGY & CLINICAL CONTEXT                     │
│  ═══════════════════════════════════════════════════════════ │
│  Mechanism:     SARS-CoV-2 binds ACE2 receptors on type II  │
│                 pneumocytes, triggering inflammatory cascade │
│  Key Markers:   CRP ↑, Lymphopenia, D-dimer ↑              │
│  Prognosis:     Self-limiting 85%; 10-15% need support      │
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  § 3  RISK STRATIFICATION                                    │
│  ═══════════════════════════════════════════════════════════ │
│  Risk Level:    Low | Factors: 0 identified                 │
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  § 4  EMERGENCY TRIAGE                                       │
│  ═══════════════════════════════════════════════════════════ │
│  Classification: Non-Emergency | ESI Level: 3               │
│  Progression:    Stable | Urgency: High                     │
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  § 5  ML PREDICTIVE MODEL OUTPUT                             │
│  ═══════════════════════════════════════════════════════════ │
│  ┌──────┬─────────────────────┬────────────┬─────────┐       │
│  │ Rank │ Disease             │ Probability│ Note    │       │
│  ├──────┼─────────────────────┼────────────┼─────────┤       │
│  │ 1    │ Peptic Ulcer        │ 33.95%     │ Primary │       │
│  │ 2    │ Flu                 │ 23.52%     │         │       │
│  │ 3    │ Stroke              │ 12.05%     │         │       │
│  └──────┴─────────────────────┴────────────┴─────────┘       │
│  Model: Voting Ensemble (XGBoost + RF + LightGBM)           │
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  § 6  CLINICAL REASONING                                     │
│  ═══════════════════════════════════════════════════════════ │
│  [Source: Knowledge-Based Clinical Analysis]                 │
│  Clinical assessment of a 35-year-old male presenting with  │
│  Fever, Cough, Fatigue. SARS-CoV-2 binds ACE2 receptors on  │
│  type II pneumocytes, triggering inflammatory cascade...    │
│  Moderate confidence (70.0%) warrants confirmatory           │
│  investigations before definitive diagnosis.                │
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  § 7  DIAGNOSTIC WORKUP                                      │
│  ═══════════════════════════════════════════════════════════ │
│  ┌────┬────────────────────────────┬──────────┬───────────┐  │
│  │ #  │ Investigation              │ Priority │ Dept      │  │
│  ├────┼────────────────────────────┼──────────┼───────────┤  │
│  │ 1  │ RT-PCR Test (Nasal Swab)   │ Primary  │ Molecular │  │
│  │    │ → Confirm SARS-CoV-2       │          │           │  │
│  │ 2  │ Rapid Antigen Test         │ Primary  │ Immunol.  │  │
│  │    │ → Quick screening          │          │           │  │
│  │ 3  │ Chest X-Ray                │ Primary  │ Radiology │  │
│  │    │ → Lung involvement         │          │           │  │
│  │ 4  │ CBC                        │ Second.  │ Hematol.  │  │
│  │    │ → Lymphopenia check        │          │           │  │
│  │ 5  │ CRP                        │ Second.  │ Biochem.  │  │
│  │    │ → Inflammation level       │          │           │  │
│  └────┴────────────────────────────┴──────────┴───────────┘  │
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  § 8  PHARMACOTHERAPY                                        │
│  ═══════════════════════════════════════════════════════════ │
│  ┌──────────────────┬──────────────────┬────────┬─────────┐  │
│  │ Drug             │ Dosage           │ Route  │ Class   │  │
│  ├──────────────────┼──────────────────┼────────┼─────────┤  │
│  │ Paracetamol      │ 500-1000mg q4-6h │ Oral   │ Antipyr │  │
│  │  ADR: Nausea     │ CI: Liver dz     │        │         │  │
│  ├──────────────────┼──────────────────┼────────┼─────────┤  │
│  │ Zinc + Vit C     │ Zn 50mg+C 1g/day│ Oral   │ Suppl.  │  │
│  │  ADR: Nausea     │ CI: Kidney stones│        │         │  │
│  ├──────────────────┼──────────────────┼────────┼─────────┤  │
│  │ Dexamethasone    │ 6mg OD x 10d     │ Oral/IV│ Steroid │  │
│  │  ADR: Hyperglyce │ CI: Mild cases   │        │         │  │
│  └──────────────────┴──────────────────┴────────┴─────────┘  │
│                                                              │
│  ═══════════════════════════════════════════════════════════ │
│  § 9  MANAGEMENT PLAN                                        │
│  ═══════════════════════════════════════════════════════════ │
│  Treatment:  Hospitalization + medication                    │
│  Urgency:    High                                            │
│  Follow-up:  Continuous monitoring + specialist consult      │
│  Risk:       Low | Emergency Protocol: Standard             │
│                                                              │
│  ┌─ CLINICAL ALERTS ──────────────────────────────────────┐  │
│  │ ⚠️ Paracetamol: CI in severe liver disease             │  │
│  │ ⚠️ Dexamethasone: CI in mild cases w/o O2              │  │
│  │ 🔴 Severe — hospitalization may be required            │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ─────────────────── Page Footer ──────────────────────────  │
│  MedAgentix AI v2.0 | ClinicalBERT + Voting Ensemble        │
│  AI-Assisted Clinical Report | Verify Before Prescribing    │
│  Report ID: MED-202606021644 | Page 1/2                     │
└──────────────────────────────────────────────────────────────┘
```

### 10.4 Download Button UI

#### Patient Dashboard — Bottom Action Bar

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   ┌─────────────────────────┐  ┌─────────────────────────┐   │
│   │  📄 Download Report     │  │  🔄 New Health Check    │   │
│   │      (PDF)              │  │                         │   │
│   │ ━━━━━━━━━━━━━━━━━━━━━━━ │  │ ━━━━━━━━━━━━━━━━━━━━━━ │   │
│   │  Blue gradient button   │  │  Outline button         │   │
│   └─────────────────────────┘  └─────────────────────────┘   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

#### Doctor Dashboard — Top-Right Toolbar + Bottom Bar

```
┌──────────────────────────────────────────────────────────────┐
│  Report: RAHUL SHARMA                                        │
│                    [ 📄 PDF ] [ 🖨️ Print ] [ 🔄 New Patient ]│
├──────────────────────────────────────────────────────────────┤
│  ... (report content) ...                                    │
├──────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐             │
│  │ 📄 PDF   │  │ 🖨️ Print │  │ 🔄 New Patient │             │
│  │ Download │  │          │  │                │             │
│  └──────────┘  └──────────┘  └────────────────┘             │
└──────────────────────────────────────────────────────────────┘
```

#### Button States

```css
/* Download button — Patient */
.btn-download-patient {
  background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
  color: white;
  padding: 14px 32px;
  border-radius: 12px;
  font-size: 16px;
  font-weight: 600;
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 10px;
  transition: all 0.2s ease;
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}
.btn-download-patient:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
}
.btn-download-patient:active {
  transform: translateY(0);
}

/* Download button — Doctor */
.btn-download-doctor {
  background: #0F172A;
  color: white;
  padding: 10px 20px;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  border: 1px solid #334155;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: all 0.15s ease;
}
.btn-download-doctor:hover {
  background: #1E293B;
  border-color: #6366F1;
}

/* Loading state during PDF generation */
.btn-download.generating {
  pointer-events: none;
  opacity: 0.7;
}
.btn-download.generating::after {
  content: '';
  width: 16px; height: 16px;
  border: 2px solid white;
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
```

### 10.5 PDF Generation — Technical Approach

#### Option A: Client-Side (jsPDF + html2canvas) — Recommended for MVP

```javascript
// static/js/report.js

async function downloadPDF(mode) {
  const btn = document.getElementById('btn-download');
  btn.classList.add('generating');
  btn.textContent = 'Generating PDF...';

  try {
    // Capture the report section as canvas
    const reportEl = document.getElementById('report-content');
    const canvas = await html2canvas(reportEl, {
      scale: 2,              // High-res
      useCORS: true,
      backgroundColor: '#FFFFFF',
      scrollY: -window.scrollY,
    });

    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF('p', 'mm', 'a4');
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = pdf.internal.pageSize.getHeight();
    const imgWidth = pdfWidth - 20;  // 10mm margin each side
    const imgHeight = (canvas.height * imgWidth) / canvas.width;

    let heightLeft = imgHeight;
    let position = 10;

    // First page
    pdf.addImage(imgData, 'PNG', 10, position, imgWidth, imgHeight);
    heightLeft -= pdfHeight;

    // Additional pages if content is long
    while (heightLeft > 0) {
      position = heightLeft - imgHeight;
      pdf.addPage();
      pdf.addImage(imgData, 'PNG', 10, position, imgWidth, imgHeight);
      heightLeft -= pdfHeight;
    }

    // Add footer watermark on every page
    const totalPages = pdf.getNumberOfPages();
    for (let i = 1; i <= totalPages; i++) {
      pdf.setPage(i);
      pdf.setFontSize(8);
      pdf.setTextColor(150, 150, 150);
      const disclaimer = mode === 'patient'
        ? 'AI-Generated Health Report — Not a Medical Prescription'
        : 'AI-Assisted Clinical Report — Verify Before Prescribing';
      pdf.text(disclaimer, pdfWidth / 2, pdfHeight - 8, { align: 'center' });
      pdf.text(`Page ${i}/${totalPages}`, pdfWidth - 15, pdfHeight - 8);
    }

    // Generate filename
    const name = document.getElementById('patient-name').textContent
                         .replace(/\s+/g, '_');
    const date = new Date().toISOString().slice(0, 10);
    const prefix = mode === 'patient' ? 'patient_report' : 'clinical_report';
    pdf.save(`${prefix}_${name}_${date}.pdf`);

  } catch (err) {
    console.error('PDF generation failed:', err);
    alert('Failed to generate PDF. Please try again.');
  } finally {
    btn.classList.remove('generating');
    btn.textContent = mode === 'patient'
      ? '📄 Download Report (PDF)'
      : '📄 Download PDF';
  }
}
```

#### Option B: Server-Side (Flask + WeasyPrint / ReportLab) — For Production

```python
# utils/pdf_generator.py

from flask import make_response
import datetime


def generate_patient_pdf(report_data):
    """
    Generate a patient-friendly PDF prescription report.

    report_data keys:
      - patient_name, patient_age, patient_gender
      - symptoms_extracted (list of dicts)
      - vitals (dict: heart_rate, oxygen_level, bp, temp)
      - final_diagnosis, final_confidence, severity
      - patient_explanation, what_it_means
      - recommended_drugs (list of dicts)
      - recommended_tests (list of dicts)
      - next_steps (list of strings)
      - emergency_warnings (list of strings)
    """
    # Use WeasyPrint to render HTML template to PDF
    from weasyprint import HTML
    from flask import render_template

    html_content = render_template(
        'pdf/patient_report.html',
        data=report_data,
        generated_at=datetime.datetime.now().strftime('%d-%b-%Y %H:%M'),
        disclaimer=(
            "This report was created by an AI health assistant. "
            "It is NOT a medical diagnosis."
        )
    )

    pdf_bytes = HTML(string=html_content).write_pdf()

    response = make_response(pdf_bytes)
    name = report_data['patient_name'].replace(' ', '_')
    date = datetime.datetime.now().strftime('%Y%m%d')
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = (
        f'attachment; filename=patient_report_{name}_{date}.pdf'
    )
    return response


def generate_doctor_pdf(report_data):
    """
    Generate a clinical diagnostic PDF report.

    report_data keys (in addition to patient PDF keys):
      - icd_code, disease_category, pathophysiology
      - key_markers, prognosis
      - risk_factors (list), risk_level
      - emergency_status (dict), triage_level
      - ml_predictions (list of dicts)
      - clinical_reasoning (str)
      - diagnosis_source (str)
      - agent_agreement (float)
      - drugs with ADR, CI, route, clinical_precaution
      - tests with priority, department, indication
      - management_plan (dict)
      - clinical_alerts (list)
    """
    from weasyprint import HTML
    from flask import render_template

    html_content = render_template(
        'pdf/doctor_report.html',
        data=report_data,
        report_id=f'MED-{datetime.datetime.now().strftime("%Y%m%d%H%M")}',
        generated_at=datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S'),
        disclaimer=(
            "AI-generated diagnostic support tool. All findings "
            "require clinical correlation. Not a substitute for "
            "clinical judgement."
        )
    )

    pdf_bytes = HTML(string=html_content).write_pdf()

    response = make_response(pdf_bytes)
    name = report_data['patient_name'].replace(' ', '_')
    date = datetime.datetime.now().strftime('%Y%m%d')
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = (
        f'attachment; filename=clinical_report_{name}_{date}.pdf'
    )
    return response
```

### 10.6 Flask API Routes for PDF Download

```python
# In app.py or api/prediction_routes.py

from utils.pdf_generator import generate_patient_pdf, generate_doctor_pdf

@app.route('/api/download/patient-pdf', methods=['POST'])
def download_patient_pdf():
    """Generate and return patient prescription PDF."""
    report_data = request.get_json()
    return generate_patient_pdf(report_data)


@app.route('/api/download/doctor-pdf', methods=['POST'])
def download_doctor_pdf():
    """Generate and return clinical diagnostic PDF."""
    report_data = request.get_json()
    return generate_doctor_pdf(report_data)
```

### 10.7 PDF Content Mapping — What Goes Where

#### Patient PDF Sections (7 sections)

| # | Section | Data Source | Notes |
|---|---------|-------------|-------|
| 1 | Patient Info | `patient_name`, `patient_age`, `patient_gender` | Header block |
| 2 | Symptoms | `symptoms_extracted` with durations | Simple table |
| 3 | Vital Signs | `heart_rate`, `oxygen_level`, `bp`, `temp` | With ✅/⚠️ status |
| 4 | Assessment Result | `final_diagnosis`, confidence, severity | With `what_it_means` + `patient_explanation` |
| 5 | Medications | `recommended_drugs` | Name, dosage, purpose only (no ADR/CI) |
| 6 | Tests | `recommended_tests` | Test name + "Why" explanation |
| 7 | Next Steps + Warnings | `next_steps`, `emergency_warnings` | Action list + danger signs |

#### Doctor PDF Sections (9 sections)

| # | Section | Data Source | Notes |
|---|---------|-------------|-------|
| 1 | Clinical Impression | Dx, ICD-10, confidence, severity, source | With differential |
| 2 | Pathophysiology | `pathophysiology`, `key_markers`, `prognosis` | From DISEASE_KB |
| 3 | Risk Stratification | `risk_factors`, `risk_level` | Table with categories |
| 4 | Emergency Triage | ESI, vital flags, progression | Badges |
| 5 | ML Model Output | Top-5 predictions, active features | Full table |
| 6 | Clinical Reasoning | KB-based or LLM reasoning | Full text block |
| 7 | Diagnostic Workup | Tests with priority, department, indication | Structured table |
| 8 | Pharmacotherapy | Drugs with dosage, route, ADR, CI, precaution | Full clinical table |
| 9 | Management Plan | Treatment, urgency, follow-up, alerts | With clinical alerts |

### 10.8 Updated File Structure (with PDF)

```
MedAgentix_AI/
├── app.py
├── templates/
│   ├── base.html
│   ├── landing.html
│   ├── patient/
│   │   ├── input.html
│   │   └── result.html              # Has "Download PDF" button
│   ├── doctor/
│   │   ├── input.html
│   │   └── result.html              # Has "PDF" + "Print" buttons
│   └── pdf/                          # PDF-specific templates
│       ├── patient_report.html       # Patient PDF layout
│       └── doctor_report.html        # Doctor PDF layout
├── static/
│   ├── css/
│   │   ├── base.css
│   │   ├── patient.css
│   │   ├── doctor.css
│   │   └── pdf.css                   # Print/PDF-optimized styles
│   └── js/
│       ├── form.js
│       ├── stepper.js
│       └── report.js                 # PDF download logic
├── utils/
│   └── pdf_generator.py              # Server-side PDF generation
├── api/
│   └── prediction_routes.py
├── reports/                          # Generated text reports
│   ├── patient_report.txt
│   └── doctor_report.txt
└── frontend_guide.md
```
