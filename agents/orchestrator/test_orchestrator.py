# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Dual Dashboard Report Generator (v2.0)
========================================================
Runs the full multi-agent pipeline and generates TWO reports:

  1. PATIENT DASHBOARD  -- Friendly, explained, non-technical
  2. DOCTOR DASHBOARD   -- Detailed clinical report with medical terminology

Usage:
  cd MedAgentix_AI
  python agents/orchestrator/test_orchestrator.py
"""

import os, sys, io, contextlib, re, warnings, datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

W = 78

# ============================================================
# CLINICAL KNOWLEDGE BASE -- Disease-specific information
# ============================================================
DISEASE_KB = {
    "covid-19": {
        "icd_code": "U07.1",
        "category": "Respiratory Viral Infection",
        "pathophysiology": (
            "SARS-CoV-2 binds ACE2 receptors on type II pneumocytes, "
            "triggering inflammatory cascade. Viral replication causes "
            "diffuse alveolar damage, cytokine storm in severe cases."
        ),
        "patient_explanation": (
            "A viral infection that mainly affects your lungs and airways. "
            "Your body is fighting the virus, which is why you feel "
            "tired and have a fever. Most people recover fully with rest."
        ),
        "what_it_means": (
            "Your symptoms match a pattern commonly seen with respiratory "
            "viruses like COVID-19. This does NOT confirm a diagnosis -- "
            "only a lab test can do that."
        ),
        "key_markers": "CRP elevated, Lymphopenia, D-dimer may be elevated",
        "prognosis": "Self-limiting in 85% of cases; 10-15% may require supportive care",
    },
    "flu": {
        "icd_code": "J11.1",
        "category": "Respiratory Viral Infection",
        "pathophysiology": (
            "Influenza virus targets respiratory epithelium. Hemagglutinin "
            "binds sialic acid residues on host cells. Neuraminidase "
            "facilitates viral release and spread."
        ),
        "patient_explanation": (
            "The flu is a very common viral infection. Your immune system "
            "is actively fighting it, which causes fever, body aches, "
            "and tiredness. Most people feel better within 5-7 days."
        ),
        "what_it_means": (
            "Your symptoms suggest you may have the flu. It usually gets "
            "better on its own. Stay hydrated and rest."
        ),
        "key_markers": "WBC normal or low, Rapid Influenza Diagnostic Test (RIDT)",
        "prognosis": "Self-limiting; 5-7 day recovery expected in healthy adults",
    },
    "diabetes": {
        "icd_code": "E11.9",
        "category": "Endocrine / Metabolic",
        "pathophysiology": (
            "Type 2 DM: Progressive insulin resistance with relative "
            "insulin deficiency. Pancreatic beta-cell dysfunction leads "
            "to impaired glucose homeostasis and chronic hyperglycemia."
        ),
        "patient_explanation": (
            "Diabetes means your body has trouble managing sugar in your "
            "blood. This happens when your body doesn't use a hormone "
            "called insulin properly. It can be managed well with the "
            "right diet, exercise, and sometimes medication."
        ),
        "what_it_means": (
            "Your symptoms and risk factors suggest you may have diabetes. "
            "A simple blood test can confirm this. If diagnosed early, "
            "diabetes is very manageable."
        ),
        "key_markers": "Fasting glucose, HbA1c, Postprandial glucose",
        "prognosis": "Chronic but manageable; requires long-term lifestyle modifications",
    },
    "hypertension": {
        "icd_code": "I10",
        "category": "Cardiovascular",
        "pathophysiology": (
            "Sustained elevation of systemic arterial pressure. "
            "Involves RAAS activation, endothelial dysfunction, "
            "increased peripheral vascular resistance, and arterial "
            "stiffness contributing to target organ damage."
        ),
        "patient_explanation": (
            "High blood pressure means your heart is working harder than "
            "normal to push blood through your body. You might not feel "
            "any symptoms, but it's important to manage it to prevent "
            "future heart or kidney problems."
        ),
        "what_it_means": (
            "Your blood pressure readings are higher than normal. "
            "This is very common and treatable. Lifestyle changes and "
            "medication can bring it under control."
        ),
        "key_markers": "BP readings, Renal function (BUN/Cr), ECG, Fundoscopy",
        "prognosis": "Controllable with medication; requires ongoing monitoring",
    },
    "heart failure": {
        "icd_code": "I50.9",
        "category": "Cardiovascular",
        "pathophysiology": (
            "Impaired ventricular filling or ejection fraction. "
            "Neurohormonal activation (SNS, RAAS) leads to "
            "fluid retention, pulmonary congestion, and reduced "
            "cardiac output."
        ),
        "patient_explanation": (
            "Heart failure means your heart isn't pumping blood as "
            "effectively as it should. This can cause tiredness and "
            "shortness of breath. With proper treatment, many people "
            "live active lives."
        ),
        "what_it_means": (
            "Your symptoms suggest your heart may need support. "
            "This is a serious condition that needs medical attention, "
            "but it is treatable."
        ),
        "key_markers": "BNP/NT-proBNP, Echocardiography (EF%), CXR",
        "prognosis": "Requires chronic management; NYHA classification guides therapy",
    },
    "pneumonia": {
        "icd_code": "J18.9",
        "category": "Respiratory Infection",
        "pathophysiology": (
            "Infectious agents cause alveolar inflammation and "
            "consolidation. Exudative infiltration impairs gas "
            "exchange. May be community-acquired (CAP) or "
            "hospital-acquired (HAP)."
        ),
        "patient_explanation": (
            "Pneumonia is an infection in your lungs that causes "
            "inflammation and makes it harder to breathe. It can be "
            "caused by bacteria or viruses. With antibiotics (if "
            "bacterial) and rest, most people recover fully."
        ),
        "what_it_means": (
            "Your symptoms suggest a possible lung infection. A chest "
            "X-ray can confirm this. Treatment usually involves "
            "antibiotics and rest."
        ),
        "key_markers": "CXR consolidation, CBC (WBC elevated), CRP, Procalcitonin",
        "prognosis": "Good with appropriate antimicrobial therapy; 1-3 week recovery",
    },
    "migraine": {
        "icd_code": "G43.9",
        "category": "Neurological",
        "pathophysiology": (
            "Cortical spreading depression triggers trigeminovascular "
            "system activation. Calcitonin gene-related peptide (CGRP) "
            "release causes vasodilation and neurogenic inflammation."
        ),
        "patient_explanation": (
            "A migraine is a type of severe headache that can come with "
            "nausea, light sensitivity, and sometimes visual changes. "
            "It's caused by changes in brain activity and is very common. "
            "There are effective treatments available."
        ),
        "what_it_means": (
            "Your headache pattern suggests a migraine. This is not "
            "dangerous but can be very uncomfortable. Your doctor can "
            "prescribe medication to help prevent and treat them."
        ),
        "key_markers": "Clinical diagnosis; CT/MRI if atypical presentation",
        "prognosis": "Episodic; manageable with prophylactic and abortive therapy",
    },
    "gastroenteritis": {
        "icd_code": "K52.9",
        "category": "Gastrointestinal",
        "pathophysiology": (
            "Infectious or toxic agents damage intestinal mucosa, "
            "disrupting fluid absorption and electrolyte balance. "
            "May involve enterotoxin-mediated secretory diarrhea."
        ),
        "patient_explanation": (
            "A stomach infection that causes nausea, vomiting, and "
            "diarrhea. Your body is fighting off germs in your "
            "digestive system. Most people feel better in 2-3 days."
        ),
        "what_it_means": (
            "Your symptoms point to a stomach bug. The most important "
            "thing is to stay hydrated. It usually clears up on its own."
        ),
        "key_markers": "Stool analysis, Electrolytes, BUN/Cr for dehydration",
        "prognosis": "Self-limiting; 2-5 days with adequate hydration",
    },
    "asthma": {
        "icd_code": "J45.9",
        "category": "Respiratory / Allergic",
        "pathophysiology": (
            "Chronic airway inflammation with bronchial hyperresponsiveness. "
            "Th2-mediated eosinophilic inflammation, mucus hypersecretion, "
            "and smooth muscle contraction cause reversible airflow obstruction."
        ),
        "patient_explanation": (
            "Asthma is a condition where your airways become narrow and "
            "swollen, making it harder to breathe. It can be triggered "
            "by allergies, exercise, or cold air. Inhalers can help "
            "manage it very effectively."
        ),
        "what_it_means": (
            "Your breathing difficulty may be related to asthma. A "
            "breathing test can confirm this. Asthma is very common "
            "and well-controlled with proper medication."
        ),
        "key_markers": "Spirometry (FEV1/FVC), Peak flow, IgE levels, Eosinophil count",
        "prognosis": "Chronic but controllable; stepwise therapy per GINA guidelines",
    },
    "anxiety": {
        "icd_code": "F41.9",
        "category": "Mental Health / Psychiatric",
        "pathophysiology": (
            "Dysregulation of amygdala-prefrontal circuits with "
            "HPA axis hyperactivation. Elevated cortisol, "
            "norepinephrine, and serotonin imbalance contribute "
            "to somatic and cognitive symptoms."
        ),
        "patient_explanation": (
            "Anxiety is when your body's stress response becomes "
            "overactive. It can cause a racing heart, trouble "
            "breathing, and worry. It's very common and treatable "
            "with therapy and sometimes medication."
        ),
        "what_it_means": (
            "Your symptoms may be related to anxiety. This is not "
            "a sign of weakness -- it's a medical condition. "
            "There are many effective treatments available."
        ),
        "key_markers": "Clinical assessment; GAD-7 score; TSH to r/o thyroid",
        "prognosis": "Good with CBT and/or pharmacotherapy",
    },
    "allergy": {
        "icd_code": "T78.4",
        "category": "Immunological",
        "pathophysiology": (
            "IgE-mediated Type I hypersensitivity. Mast cell degranulation "
            "releases histamine, leukotrienes, and prostaglandins causing "
            "vasodilation, edema, and bronchoconstriction."
        ),
        "patient_explanation": (
            "An allergy happens when your immune system overreacts to "
            "something harmless like pollen or food. This can cause "
            "sneezing, itching, and sometimes skin rashes."
        ),
        "what_it_means": (
            "Your symptoms suggest an allergic reaction. Identifying "
            "what triggers it can help you avoid it in the future."
        ),
        "key_markers": "IgE levels, Skin prick test, Eosinophil count",
        "prognosis": "Manageable; avoidance + antihistamines",
    },
    "heart attack": {
        "icd_code": "I21.9",
        "category": "Cardiovascular / Acute Coronary Syndrome",
        "pathophysiology": (
            "Atherosclerotic plaque rupture triggers thrombus formation "
            "in coronary artery, causing myocardial ischemia and necrosis. "
            "ST-elevation indicates transmural infarction (STEMI); "
            "non-ST-elevation (NSTEMI) involves subendocardial injury."
        ),
        "patient_explanation": (
            "A heart attack happens when blood flow to part of your heart "
            "is blocked, usually by a blood clot. The heart muscle doesn't get "
            "enough oxygen and can be damaged. Quick treatment can save your "
            "life and limit heart damage."
        ),
        "what_it_means": (
            "Your symptoms, vital signs, and medical history strongly suggest "
            "a cardiac event. This is a medical emergency. Getting to a hospital "
            "quickly is the most important step right now."
        ),
        "key_markers": "Troponin I/T, ECG (ST-elevation), CK-MB, BNP, Echocardiography",
        "prognosis": "Time-critical; door-to-balloon <90 min for STEMI; variable based on infarct size",
    },
    "rhinitis": {
        "icd_code": "J31.0",
        "category": "ENT / Upper Respiratory",
        "pathophysiology": (
            "Nasal mucosal inflammation driven by allergic (IgE-mediated) or "
            "non-allergic (vasomotor, infectious) mechanisms. Histamine release "
            "causes vasodilation, edema, and hypersecretion of nasal glands."
        ),
        "patient_explanation": (
            "Rhinitis is inflammation inside your nose, causing a runny or "
            "stuffy nose. It's very common and usually caused by allergies, "
            "a cold, or changes in weather. It is not serious and usually "
            "clears up on its own."
        ),
        "what_it_means": (
            "Your symptoms suggest nasal inflammation (rhinitis). This is "
            "one of the most common conditions and is very treatable. "
            "Avoiding triggers and using simple medications can help."
        ),
        "key_markers": "Nasal examination, Allergy skin prick test, Serum IgE",
        "prognosis": "Excellent; self-limiting or controlled with antihistamines/nasal steroids",
    },
    "stroke": {
        "icd_code": "I63.9",
        "category": "Neurological / Cerebrovascular",
        "pathophysiology": (
            "Ischemic: Thrombotic or embolic occlusion of cerebral artery "
            "causing focal neurological deficit. Hemorrhagic: Rupture of "
            "intracerebral vessel leading to parenchymal bleeding."
        ),
        "patient_explanation": (
            "A stroke happens when blood supply to part of your brain is "
            "cut off. This can cause sudden weakness, confusion, or "
            "difficulty speaking. Getting treatment quickly is critical."
        ),
        "what_it_means": (
            "Your symptoms may suggest a stroke-like event. This needs "
            "immediate medical attention. Time is brain -- every minute counts."
        ),
        "key_markers": "CT/MRI Brain, CT Angiography, CBC, Coagulation panel",
        "prognosis": "Time-dependent; tPA window 4.5 hrs; outcomes vary by location/size",
    },
    "bronchial asthma": {
        "icd_code": "J45.9",
        "category": "Respiratory / Allergic",
        "pathophysiology": (
            "Chronic airway inflammation with bronchial hyperresponsiveness. "
            "Th2-driven eosinophilic inflammation, mucus plugging, and "
            "smooth muscle bronchoconstriction cause episodic airflow limitation."
        ),
        "patient_explanation": (
            "Bronchial asthma makes your airways narrow and swell, "
            "causing wheezing, coughing, and shortness of breath. "
            "It can be triggered by dust, cold air, or exercise. "
            "Inhalers provide effective relief."
        ),
        "what_it_means": (
            "Your breathing symptoms may be related to asthma. A breathing "
            "test (spirometry) can confirm this. It is a very manageable condition."
        ),
        "key_markers": "Spirometry (FEV1/FVC <0.7), Peak flow variability, FeNO, IgE",
        "prognosis": "Chronic but well-controlled with inhaled corticosteroids per GINA",
    },
    "depression": {
        "icd_code": "F32.9",
        "category": "Mental Health / Psychiatric",
        "pathophysiology": (
            "Monoamine hypothesis: Deficiency of serotonin (5-HT), "
            "norepinephrine (NE), and/or dopamine in synaptic cleft. "
            "Neuroplasticity and HPA axis dysregulation also implicated."
        ),
        "patient_explanation": (
            "Depression is a medical condition that affects your mood, "
            "energy, and ability to enjoy things. It is NOT a sign of "
            "weakness. It is very common and very treatable with therapy, "
            "medication, or both."
        ),
        "what_it_means": (
            "Your symptoms suggest you may be experiencing depression. "
            "This is a real medical condition with effective treatments. "
            "Please talk to a doctor."
        ),
        "key_markers": "PHQ-9 score, TSH (rule out thyroid), CBC, Vitamin B12/D",
        "prognosis": "Good with SSRIs/SNRIs + CBT; 60-80% respond to first-line treatment",
    },
    "gerd": {
        "icd_code": "K21.0",
        "category": "Gastrointestinal",
        "pathophysiology": (
            "Lower esophageal sphincter (LES) incompetence allows gastric acid "
            "reflux into the esophagus. Chronic exposure causes mucosal erosion, "
            "metaplasia (Barrett's), and esophageal dysmotility."
        ),
        "patient_explanation": (
            "GERD (acid reflux) happens when stomach acid flows back into "
            "your food pipe, causing heartburn and discomfort. It is very "
            "common and can be managed with diet changes and medication."
        ),
        "what_it_means": (
            "Your symptoms suggest acid reflux (GERD). This is uncomfortable "
            "but very treatable. Diet changes and medication usually help."
        ),
        "key_markers": "Upper GI endoscopy, 24-hr pH monitoring, H. pylori test",
        "prognosis": "Good with PPIs and lifestyle modification; monitor for Barrett's",
    },
    "peptic ulcer": {
        "icd_code": "K27.9",
        "category": "Gastrointestinal",
        "pathophysiology": (
            "Mucosal defect in stomach or duodenum caused by imbalance "
            "between aggressive factors (HCl, pepsin, H. pylori, NSAIDs) "
            "and protective factors (mucus, bicarbonate, prostaglandins)."
        ),
        "patient_explanation": (
            "A peptic ulcer is a sore in the lining of your stomach or "
            "upper intestine. It can cause burning pain in your abdomen. "
            "With proper medication, ulcers heal completely in most cases."
        ),
        "what_it_means": (
            "Your symptoms suggest a possible stomach ulcer. A simple test "
            "can check for the bacteria that often causes this. Treatment "
            "is very effective."
        ),
        "key_markers": "Upper GI endoscopy, H. pylori (urea breath test/stool Ag), CBC",
        "prognosis": "Excellent with PPI + H. pylori eradication; >90% healing in 8 weeks",
    },
    "urinary tract infection": {
        "icd_code": "N39.0",
        "category": "Urology / Infectious",
        "pathophysiology": (
            "Ascending bacterial infection (usually E. coli) of the "
            "urinary tract. Lower UTI (cystitis) involves bladder; "
            "upper UTI (pyelonephritis) involves kidneys."
        ),
        "patient_explanation": (
            "A UTI is an infection in your urinary system (bladder or kidneys). "
            "It can cause pain when urinating and frequent urges to go. "
            "Antibiotics clear it up quickly."
        ),
        "what_it_means": (
            "Your symptoms suggest a urinary tract infection. A simple "
            "urine test can confirm this. It is easily treated with antibiotics."
        ),
        "key_markers": "Urinalysis, Urine culture & sensitivity, CBC",
        "prognosis": "Excellent; uncomplicated UTI resolves in 3-5 days with antibiotics",
    },
    "chickenpox": {
        "icd_code": "B01.9",
        "category": "Infectious / Viral",
        "pathophysiology": (
            "Varicella-zoster virus (VZV) causes primary infection with "
            "viremia and characteristic vesicular rash. Virus establishes "
            "latency in dorsal root ganglia; reactivation causes herpes zoster."
        ),
        "patient_explanation": (
            "Chickenpox is a viral infection that causes an itchy rash "
            "with small blisters. It is very common in children but can "
            "affect adults too. Most people recover fully within 1-2 weeks."
        ),
        "what_it_means": (
            "Your symptoms suggest chickenpox. It is usually mild and "
            "clears up on its own. Avoid scratching to prevent scars."
        ),
        "key_markers": "Clinical diagnosis (vesicular rash), VZV IgM/IgG",
        "prognosis": "Self-limiting in immunocompetent; 7-10 day course",
    },
    "appendicitis": {
        "icd_code": "K35.8",
        "category": "Surgical / Gastrointestinal",
        "pathophysiology": (
            "Luminal obstruction of the appendix (fecalith, lymphoid hyperplasia) "
            "leads to increased intraluminal pressure, venous congestion, "
            "bacterial invasion, and eventual perforation if untreated."
        ),
        "patient_explanation": (
            "Appendicitis is inflammation of the appendix, a small tube "
            "attached to your intestine. It causes pain in the lower right "
            "side of your belly. It usually needs surgery to remove the "
            "appendix, which is a very common and safe procedure."
        ),
        "what_it_means": (
            "Your symptoms suggest possible appendicitis. This needs "
            "urgent medical evaluation. Do not eat or drink anything "
            "and go to the hospital."
        ),
        "key_markers": "WBC count, CRP, CT abdomen, Alvarado score",
        "prognosis": "Excellent with early appendectomy; risk of perforation if delayed",
    },
    "common cold": {
        "icd_code": "J00",
        "category": "ENT / Upper Respiratory",
        "pathophysiology": (
            "Rhinovirus (most common), coronavirus, or RSV infection of "
            "upper respiratory epithelium. Self-limiting mucosal inflammation "
            "with serous then mucopurulent nasal discharge."
        ),
        "patient_explanation": (
            "The common cold is a mild viral infection of your nose and "
            "throat. Almost everyone gets it several times a year. It "
            "clears up on its own within 5-7 days."
        ),
        "what_it_means": (
            "Your symptoms are typical of a common cold. No special "
            "treatment is needed -- just rest, fluids, and time."
        ),
        "key_markers": "Clinical diagnosis; no lab tests required",
        "prognosis": "Self-limiting; 5-7 days; symptomatic treatment only",
    },
    "tuberculosis": {
        "icd_code": "A15.0",
        "category": "Infectious / Pulmonary",
        "pathophysiology": (
            "Mycobacterium tuberculosis causes granulomatous inflammation "
            "primarily in lungs. Caseous necrosis with cavitation in "
            "active disease. Cell-mediated immunity (Th1) is the primary "
            "defense mechanism."
        ),
        "patient_explanation": (
            "Tuberculosis (TB) is a bacterial infection that mainly affects "
            "the lungs. It causes persistent cough, weight loss, and night "
            "sweats. TB is curable with a course of antibiotics taken for "
            "several months."
        ),
        "what_it_means": (
            "Your symptoms may suggest tuberculosis. A chest X-ray and "
            "sputum test can confirm this. TB is completely curable "
            "with proper medication."
        ),
        "key_markers": "Sputum AFB smear/culture, Mantoux test, IGRA, CXR",
        "prognosis": "Curable with 6-9 month DOTS regimen; drug-resistant forms require longer treatment",
    },
    "malaria": {
        "icd_code": "B54",
        "category": "Infectious / Parasitic",
        "pathophysiology": (
            "Plasmodium species (P. falciparum most severe) transmitted by "
            "Anopheles mosquito. Erythrocytic cycle causes periodic hemolysis, "
            "cytokine release, and characteristic cyclic fevers."
        ),
        "patient_explanation": (
            "Malaria is an infection spread by mosquito bites. It causes "
            "high fever that comes and goes, chills, and body aches. "
            "It is treatable with medication and most people recover fully."
        ),
        "what_it_means": (
            "Your fever pattern and symptoms may suggest malaria. "
            "A simple blood test can confirm this quickly."
        ),
        "key_markers": "Peripheral blood smear (thick/thin), Rapid Diagnostic Test (RDT), CBC",
        "prognosis": "Excellent with early antimalarial therapy; P. falciparum can be severe",
    },
    "typhoid": {
        "icd_code": "A01.0",
        "category": "Infectious / Enteric",
        "pathophysiology": (
            "Salmonella typhi ingested via contaminated food/water. "
            "Invades intestinal mucosa, multiplies in macrophages, "
            "causes bacteremia with stepladder fever pattern."
        ),
        "patient_explanation": (
            "Typhoid is a bacterial infection spread through contaminated "
            "food or water. It causes prolonged fever, weakness, and stomach "
            "problems. Antibiotics can cure it effectively."
        ),
        "what_it_means": (
            "Your prolonged fever and symptoms may suggest typhoid. "
            "A blood test (Widal/blood culture) can confirm the diagnosis."
        ),
        "key_markers": "Blood culture, Widal test, CBC (leukopenia), Typhidot",
        "prognosis": "Good with appropriate antibiotics; 1-2 week recovery",
    },
}

# Default for diseases not in knowledge base
DEFAULT_KB = {
    "icd_code": "—",
    "category": "General Medicine",
    "pathophysiology": "Clinical assessment required for detailed pathological analysis.",
    "patient_explanation": (
        "Based on your symptoms, our AI system has identified a possible "
        "health condition. This is a starting point for your doctor to "
        "investigate further."
    ),
    "what_it_means": (
        "Your symptoms match a pattern that may need medical attention. "
        "Please see a doctor for proper evaluation and testing."
    ),
    "key_markers": "Comprehensive metabolic panel, CBC with differential",
    "prognosis": "Prognosis depends on confirmed diagnosis and timely intervention",
}

# Patient-friendly drug category explanations
DRUG_CATEGORY_EXPLAIN = {
    "Analgesic": "for pain relief",
    "Antipyretic": "to reduce fever",
    "Antibiotic": "to fight bacterial infection",
    "Antiviral": "to fight viral infection",
    "Anti-inflammatory": "to reduce swelling and inflammation",
    "Antihistamine": "to reduce allergic reactions",
    "Bronchodilator": "to open up airways for easier breathing",
    "Antacid": "to reduce stomach acid",
    "Antihypertensive": "to lower blood pressure",
    "Antidiabetic": "to control blood sugar levels",
    "Anxiolytic": "to help manage anxiety",
    "Mucolytic": "to thin mucus for easier breathing",
    "Corticosteroid": "to reduce inflammation",
    "Decongestant": "to relieve stuffy nose",
}


# ============================================================
# CONFIDENCE LABELS
# ============================================================
def conf_patient(c):
    """Patient-friendly confidence description."""
    if c >= 0.90: return "Strong Match"
    elif c >= 0.80: return "Good Match"
    elif c >= 0.70: return "Possible Match"
    elif c >= 0.50: return "Needs Further Evaluation"
    else: return "Uncertain -- Doctor Visit Recommended"

def conf_doctor(c):
    """Clinical confidence tier."""
    if c >= 0.90: return "High (>90%)"
    elif c >= 0.80: return "Moderate-High (80-90%)"
    elif c >= 0.70: return "Moderate (70-80%)"
    elif c >= 0.50: return "Low-Moderate (50-70%)"
    else: return "Low (<50%)"


# ============================================================
# FORMATTING UTILITIES
# ============================================================
def hr(ch="="):  print(ch * W)
def bl():        print()
def hdr(text):   hr("="); print(f"  {text}"); hr("=")
def sec(title):  print(f"\n  {title}"); print(f"  {'~' * len(title)}")
def row(l, v, ind=6): print(f"{' '*ind}{l:.<42s} {v}")
def bx(lines, ch="!"):
    hr(ch)
    for ln in lines: print(f"  {ln}")
    hr(ch)

def wrap(text, indent=8, width=62):
    pad = " " * indent
    words = text.replace("\n", " ").split()
    cur = pad
    for w in words:
        if len(cur) + len(w) + 1 > width + indent:
            print(cur.rstrip())
            cur = pad
        cur += w + " "
    if cur.strip(): print(cur.rstrip())

def clean_llm(text):
    """Aggressively clean LLM output of template artifacts and junk."""
    if not text:
        return ""
    # Strip common Meditron/BioGPT template markers
    for m in ["Format each diagnosis EXACTLY as:", "### Response:",
              "### Instruction:", "### Explanations:"]:
        i = text.rfind(m)
        if i != -1: text = text[i + len(m):]
    # Remove structured format artifacts
    text = re.sub(r'###\s*(Response|Instruction|Explanations?|Input):?\s*', '', text)
    text = re.sub(r'DIAGNOSIS:.*?\|.*?\n?', '', text)
    text = re.sub(r'CONFIDENCE:.*?\|', '', text)
    text = re.sub(r'<\|im_(start|end)\|>.*?\n?', '', text)
    text = re.sub(r'<[^>]{1,40}>', '', text)  # Remove XML/template tags like <one-line explanation>
    text = re.sub(r'You are a senior physician\..*?REASONING:.*?\n', '', text, flags=re.DOTALL)
    text = re.sub(r'\n{3,}', '\n', text)
    text = re.sub(r'^REASONING:\s*', '', text.strip())
    # Remove leftover ### markers
    text = re.sub(r'###', '', text)
    # Remove repeated hashes, asterisks, etc.
    text = re.sub(r'[#*]{2,}', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_llm_output_useful(text):
    """Check if cleaned LLM output is meaningful clinical text."""
    if not text or len(text) < 30:
        return False
    # Junk patterns that indicate template artifacts
    junk_patterns = [
        r'^\s*$',
        r'one-line explanation',
        r'explanation here',
        r'insert reasoning',
        r'placeholder',
        r'^\s*N/?A\s*$',
    ]
    for pat in junk_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return False
    return True


def generate_clinical_reasoning(disease, kb, inp, final):
    """
    Generate structured clinical reasoning from knowledge base
    when LLM output is unavailable or garbage.
    """
    dl = disease.strip().lower()
    conf = final.get("final_confidence", 0)
    severity = final.get("severity", "N/A")
    symptoms = final.get("symptoms_extracted", [])
    risk_factors = final.get("risk_factors", [])
    emg = final.get("emergency_status", {})

    lines = []

    # Opening statement
    lines.append(
        f"Clinical assessment of a {inp['patient_age']}-year-old "
        f"{inp['patient_gender'].lower()} presenting with "
        f"{', '.join(symptoms[:4]) if symptoms else 'reported symptoms'}."
    )

    # Pathophysiology context
    if kb.get("pathophysiology") and kb["pathophysiology"] != DEFAULT_KB["pathophysiology"]:
        lines.append(f"\nPathological basis: {kb['pathophysiology']}")

    # Symptom correlation
    if symptoms:
        lines.append(
            f"\nThe symptom constellation of {', '.join(symptoms[:3])} "
            f"is consistent with the clinical presentation of {disease}. "
        )
        if conf >= 0.85:
            lines.append(
                f"High diagnostic confidence ({conf:.1%}) supports this as the "
                f"primary working diagnosis."
            )
        elif conf >= 0.70:
            lines.append(
                f"Moderate confidence ({conf:.1%}) warrants confirmatory "
                f"investigations before definitive diagnosis."
            )
        else:
            lines.append(
                f"Low confidence ({conf:.1%}) indicates significant diagnostic "
                f"uncertainty. Broad differential workup is recommended."
            )

    # Risk factor correlation
    if risk_factors:
        rf_names = []
        for rf in risk_factors[:4]:
            rf_names.append(rf.get("factor", str(rf)) if isinstance(rf, dict) else str(rf))
        lines.append(
            f"\nIdentified risk factors ({', '.join(rf_names)}) further "
            f"support the clinical suspicion and may influence disease "
            f"severity and management approach."
        )

    # Vital sign interpretation
    hr_val = inp.get("heart_rate", 0)
    o2_val = inp.get("oxygen_level", 0)
    temp_val = inp.get("body_temperature", 0)
    vitals_notes = []
    if hr_val > 100:
        vitals_notes.append(f"tachycardia ({hr_val} bpm)")
    if hr_val < 60:
        vitals_notes.append(f"bradycardia ({hr_val} bpm)")
    if o2_val < 94:
        vitals_notes.append(f"hypoxemia (SpO2 {o2_val}%)")
    if isinstance(temp_val, (int, float)) and temp_val > 100.4:
        vitals_notes.append(f"pyrexia ({temp_val} F)")
    if vitals_notes:
        lines.append(
            f"\nVital sign abnormalities noted: {', '.join(vitals_notes)}. "
            f"These findings correlate with {severity.lower()} disease severity "
            f"and support the current triage classification."
        )

    # Emergency context
    if emg.get("is_emergency"):
        lines.append(
            f"\nEMERGENCY CLASSIFICATION: Patient meets criteria for "
            f"immediate clinical intervention. Triage Level "
            f"{emg.get('triage_level', 'N/A')}. Initiate emergency "
            f"protocol and stabilize before definitive management."
        )

    # Diagnostic markers
    if kb.get("key_markers") and kb["key_markers"] != DEFAULT_KB["key_markers"]:
        lines.append(
            f"\nRecommended diagnostic markers: {kb['key_markers']}. "
            f"Correlate with clinical findings for diagnostic confirmation."
        )

    # Prognosis
    if kb.get("prognosis") and kb["prognosis"] != DEFAULT_KB["prognosis"]:
        lines.append(f"\nExpected prognosis: {kb['prognosis']}.")

    return " ".join(lines)


# ============================================================
# SEVERITY EXPLANATION (patient-friendly)
# ============================================================
def severity_explain(sev):
    mapping = {
        "Mild": ("Mild", "Your condition appears to be minor. With proper care and rest, you should feel better soon."),
        "Moderate": ("Moderate", "Your condition needs attention. Please see a doctor within the next day or two for proper evaluation."),
        "Severe": ("Serious", "Your condition may need urgent medical care. Please visit a doctor or hospital as soon as possible."),
        "Critical": ("Urgent", "This may require immediate emergency care. Please call emergency services or go to the hospital now."),
    }
    return mapping.get(sev, ("Unknown", "Please consult a doctor for evaluation."))


# ============================================================
# SHARED: Print patient intake section
# ============================================================
def print_intake(inp, mode="patient"):
    sec("PATIENT INFORMATION")
    bl()
    row("Name", inp.get("patient_name", "Patient"))
    row("Age", f'{inp["patient_age"]} years')
    row("Gender", inp["patient_gender"])
    if mode == "doctor":
        row("Report Date", datetime.datetime.now().strftime("%d-%b-%Y  %H:%M"))
        row("Report ID", f'MED-{datetime.datetime.now().strftime("%Y%m%d%H%M")}')

    sec("PRESENTING COMPLAINTS" if mode == "doctor" else "YOUR SYMPTOMS")
    bl()
    kw = {"fever":"Fever","cough":"Cough","fatigue":"Fatigue","tired":"Fatigue",
          "body pain":"Body Pain","headache":"Headache","breathing":"Difficulty Breathing",
          "breathlessness":"Difficulty Breathing","chest pain":"Chest Pain",
          "vomiting":"Vomiting","runny nose":"Runny Nose","nausea":"Nausea",
          "rash":"Skin Rash","diarrhea":"Diarrhea","weight loss":"Weight Loss",
          "joint pain":"Joint Pain","back pain":"Back Pain"}
    text = inp["patient_text"].lower()
    seen = []
    for k, v in kw.items():
        if k in text and v not in seen: seen.append(v)
    for s in seen:
        dur = "-"
        for d in inp.get("symptom_durations", []):
            if d["symptom"].lower() in s.lower() or s.lower() in d["symptom"].lower():
                dur = d["duration"]; break
        if mode == "doctor":
            print(f"      [+] {s:<30s} Duration: {dur}")
        else:
            print(f"      * {s:<30s} ({dur})")
    if not seen: print("      (No specific symptoms identified)")

    sec("VITALS" if mode == "doctor" else "YOUR VITAL SIGNS")
    bl()
    hr_val = inp["heart_rate"]
    o2_val = inp["oxygen_level"]
    bp_val = inp["blood_pressure_reading"]
    temp_val = inp.get("body_temperature", "N/A")

    if mode == "doctor":
        # Doctor: Show raw values with clinical interpretation
        hr_status = "Tachycardia" if hr_val > 100 else "Bradycardia" if hr_val < 60 else "Normal sinus"
        o2_status = "Hypoxemia" if o2_val < 94 else "Normal"
        row("Heart Rate", f'{hr_val} bpm  [{hr_status}]')
        row("SpO2", f'{o2_val}%  [{o2_status}]')
        row("Blood Pressure", f'{bp_val} mmHg')
        row("Temperature", f'{temp_val} F')
        row("Cholesterol", f'{inp.get("cholesterol","N/A")} mg/dL')
    else:
        # Patient: Show with simple explanation
        row("Heart Rate", f'{hr_val} bpm')
        if hr_val > 100:
            wrap("(Your heart rate is a bit fast. This can happen with fever or stress.)")
        elif hr_val < 60:
            wrap("(Your heart rate is slower than usual. This may need checking.)")
        row("Oxygen Level", f'{o2_val}%')
        if o2_val < 94:
            wrap("(Your oxygen is lower than normal. Please monitor this closely.)")
        else:
            wrap("(Your oxygen level is in the normal range. This is good.)")
        row("Blood Pressure", f'{bp_val} mmHg')
        row("Body Temperature", f'{temp_val} F')
        if isinstance(temp_val, (int, float)) and temp_val > 100.4:
            wrap("(You have a fever. This means your body is fighting an infection.)")

    sec("MEDICAL HISTORY" if mode == "doctor" else "YOUR HEALTH BACKGROUND")
    bl()
    h = inp.get("medical_history", [])
    l = inp.get("lifestyle_factors", [])
    if mode == "doctor":
        row("Past Medical History (PMH)", ", ".join(h) if h else "None documented")
        row("Social / Lifestyle Factors", ", ".join(l) if l else "None documented")
    else:
        row("Past Health Conditions", ", ".join(h) if h else "None mentioned")
        row("Lifestyle Factors", ", ".join(l) if l else "None mentioned")


# ################################################################
#  PATIENT DASHBOARD (v2.0 -- Friendly, Explained, No Jargon)
# ################################################################
def patient_dash(name, inp, result):
    final = result.get("final_diagnosis", {})
    if not final or not final.get("final_disease"):
        print("\n  [!] Assessment could not be generated.\n"); return

    disease = final["final_disease"]
    dl = disease.strip().lower()
    conf = final.get("final_confidence", 0)
    severity = final.get("severity", "N/A")
    kb = DISEASE_KB.get(dl, DEFAULT_KB)
    emg = final.get("emergency_status", {})

    sev_label, sev_explain = severity_explain(severity)

    bl()
    hr("=")
    print(f"  YOUR HEALTH ASSESSMENT REPORT")
    print(f"  {inp.get('patient_name', 'Patient').upper()}")
    print(f"  Date: {datetime.datetime.now().strftime('%d %B %Y, %I:%M %p')}")
    hr("=")

    # ── What you told us ──
    bl()
    print_intake(inp, mode="patient")

    # ── ASSESSMENT RESULT ──
    bl()
    hr("-")
    print(f"  WHAT WE FOUND")
    hr("-")
    bl()
    print(f"      Based on the symptoms and information you shared, our")
    print(f"      AI health assistant has prepared this assessment:")
    bl()

    # Show friendly disease name
    friendly_name = kb.get("patient_explanation") and disease.title() or disease.title()
    row("Possible Condition", friendly_name)
    row("Match Confidence", conf_patient(conf))
    row("Severity Level", sev_label)
    bl()

    # What it means
    print(f"      What this means:")
    print(f"      " + "-" * 40)
    wrap(kb.get("what_it_means", DEFAULT_KB["what_it_means"]))
    bl()

    # Detailed explanation
    print(f"      Understanding your condition:")
    print(f"      " + "-" * 40)
    wrap(kb.get("patient_explanation", DEFAULT_KB["patient_explanation"]))
    bl()

    # Severity explanation
    print(f"      How serious is this?")
    print(f"      " + "-" * 40)
    wrap(sev_explain)

    # ── Emergency ──
    if emg.get("is_emergency"):
        bl()
        bx(["!! IMPORTANT: PLEASE SEEK MEDICAL HELP NOW !!",
            "",
            "Your symptoms and vital signs suggest this may need",
            "immediate attention. Please do one of the following:",
            "",
            "  * Call emergency services (112 / 108)",
            "  * Go to the nearest hospital emergency room",
            "  * Ask someone to drive you -- do NOT drive yourself",
            "",
            "It is better to be safe and get checked right away."], "!")
        bl()

    # ── Risk Factors (simplified) ──
    risk_factors = final.get("risk_factors", [])
    if risk_factors:
        bl()
        hr("-")
        print(f"  THINGS THAT MAY INCREASE YOUR RISK")
        hr("-")
        bl()
        print(f"      These factors from your history may affect your health:")
        bl()
        for rf in risk_factors[:5]:
            fname = rf.get("factor", str(rf)) if isinstance(rf, dict) else str(rf)
            print(f"      * {fname}")
        bl()
        wrap("Knowing your risk factors helps your doctor create the best plan for you.")

    # ── Medications (simplified) ──
    meds = final.get("recommended_medications", [])
    bl()
    hr("-")
    print(f"  POSSIBLE MEDICATIONS")
    print(f"  (Your doctor will decide what's right for you)")
    hr("-")
    bl()
    if meds:
        print(f"      These are medications commonly used for this condition.")
        print(f"      NEVER take any medicine without your doctor's approval.")
        bl()
        for i, m in enumerate(meds[:4], 1):
            if isinstance(m, dict):
                drug_name = m.get('drug', '?')
                category = m.get('category', '')
                dosage = m.get('dosage', '')
                side_fx = m.get('side_effects', '')

                # Explain category in simple terms
                cat_explain = DRUG_CATEGORY_EXPLAIN.get(
                    category, f"for treating your condition"
                )

                print(f"      {i}. {drug_name}")
                print(f"         What it does : {cat_explain.capitalize()}")
                if dosage:
                    print(f"         Usual dosage : {dosage}")
                if side_fx:
                    print(f"         Things to watch: {side_fx}")
                bl()
    else:
        print(f"      Your doctor will prescribe the right treatment")
        print(f"      after a proper examination.")

    # ── Tests (simplified) ──
    tests = final.get("recommended_tests", [])
    hr("-")
    print(f"  TESTS YOUR DOCTOR MAY ASK FOR")
    hr("-")
    bl()
    if tests:
        print(f"      These tests can help confirm what's going on:")
        bl()
        for i, t in enumerate(tests[:5], 1):
            if isinstance(t, dict):
                test_name = t.get('test', '?')
                reason = t.get('reason', t.get('why', ''))
                print(f"      {i}. {test_name}")
                if reason:
                    print(f"         Why: {reason}")
        bl()
        wrap("These tests help your doctor understand your condition better "
             "and create the most effective treatment plan.")
    else:
        print(f"      Your doctor will decide which tests are needed")
        print(f"      after examining you in person.")

    # ── What to do next ──
    bl()
    hr("-")
    print(f"  WHAT TO DO NEXT")
    hr("-")
    bl()
    plan = final.get("treatment_plan", {})
    treatment = plan.get("treatment_plan", "")
    follow_up = plan.get("follow_up", "")

    steps = [
        "Visit your doctor and share this report with them.",
        "Complete any tests your doctor recommends.",
    ]
    if treatment:
        steps.append(f"Treatment suggestion: {treatment}")
    if follow_up:
        steps.append(f"Follow-up: {follow_up}")
    steps.extend([
        "Keep track of your symptoms -- note if they get better or worse.",
        "Stay hydrated, rest, and eat healthy meals.",
    ])

    for i, step in enumerate(steps, 1):
        print(f"      {i}. {step}")
    bl()

    # ── When to seek help ──
    hr("-")
    print(f"  WHEN TO GET HELP RIGHT AWAY")
    hr("-")
    bl()
    emergencies = [
        "Difficulty breathing or feeling short of breath",
        "Severe chest pain or pressure",
        "High fever (above 103 F / 39.4 C) that won't come down",
        "Fainting, confusion, or inability to stay awake",
        "Any sudden worsening of your symptoms",
    ]
    for e in emergencies:
        print(f"      [!] {e}")
    bl()
    wrap("If any of the above happen, call 112/108 or go to the nearest "
         "emergency room immediately. Don't wait.")

    # ── Disclaimer ──
    bl()
    hr("_")
    print(f"  IMPORTANT NOTICE")
    print(f"  This report was created by an AI health assistant. It is NOT")
    print(f"  a medical diagnosis. Only a qualified doctor can diagnose your")
    print(f"  condition after proper examination and lab tests. Please use")
    print(f"  this report as a starting point for your doctor visit, not as")
    print(f"  a replacement for professional medical advice.")
    hr("_")
    bl()


# ################################################################
#  DOCTOR DASHBOARD (v2.0 -- Clinical Detail, No Agent Labels)
# ################################################################
def doctor_dash(name, inp, result):
    final = result.get("final_diagnosis", {})
    agents = final.get("agent_outputs", {})
    if not final or not final.get("final_disease"):
        print("\n  [!] Assessment could not be generated.\n"); return

    disease = final["final_disease"]
    dl = disease.strip().lower()
    conf = final.get("final_confidence", 0)
    severity = final.get("severity", "N/A")
    kb = DISEASE_KB.get(dl, DEFAULT_KB)
    emg = final.get("emergency_status", {})

    bl()
    hr("#")
    print(f"  CLINICAL DIAGNOSTIC REPORT")
    print(f"  Patient: {inp.get('patient_name', 'N/A').upper()}")
    print(f"  Report ID: MED-{datetime.datetime.now().strftime('%Y%m%d%H%M')}")
    print(f"  Generated: {datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S')}")
    hr("#")

    # ── Patient Data ──
    bl()
    print_intake(inp, mode="doctor")

    # ================================================================
    # SECTION 1: CLINICAL IMPRESSION
    # ================================================================
    bl()
    hr("=")
    print(f"  SECTION 1: CLINICAL IMPRESSION")
    hr("=")
    bl()
    row("Primary Diagnosis", f'{disease}  (ICD-10: {kb["icd_code"]})')
    row("Diagnostic Confidence", f'{conf:.1%}  [{conf_doctor(conf)}]')
    row("Clinical Severity", severity)
    row("Disease Category", kb["category"])
    # Translate internal source names to professional terminology
    source_raw = final.get("diagnosis_source", "ML Ensemble")
    source_map = {
        "high_confidence_ml": "ML Ensemble (High Confidence Direct Path)",
        "differential_agent": "Differential Diagnosis Engine (Knowledge-Based)",
        "llm_fallback_meditron_7b": "Meditron-7B LLM Fallback (Low-Confidence Override)",
        "llm_fallback_biogpt": "BioGPT LLM Fallback (Low-Confidence Override)",
        "llm_fallback": "LLM Clinical Reasoning Fallback",
        "moderate_validated": "ML Ensemble + Cross-Agent Validation",
        "ensemble_prediction": "ML Voting Ensemble (XGBoost + RF + LightGBM)",
    }
    source_display = source_map.get(source_raw, source_raw.replace("_", " ").title())
    row("Diagnosis Source", source_display)
    row("Agent Agreement Score", f'{final.get("agent_agreement", "N/A")}')

    # Alternatives
    alts = final.get("alternatives", [])
    if alts:
        bl()
        print(f"      Differential Considerations:")
        print(f"      {'Rank':<6s} {'Condition':<34s} {'Probability':<12s}")
        print(f"      {'----':<6s} {'.' * 34} {'.' * 12}")
        for i, a in enumerate(alts[:5], 1):
            if isinstance(a, dict):
                d = a.get('disease', '?')
                c = a.get('confidence', 0)
                flag = "<< Primary" if i == 1 else ""
                print(f"      {i:<6d} {d:<34s} {c:<12.1%} {flag}")

    # ================================================================
    # SECTION 2: PATHOPHYSIOLOGY & CLINICAL CONTEXT
    # ================================================================
    bl()
    hr("=")
    print(f"  SECTION 2: PATHOPHYSIOLOGY & CLINICAL CONTEXT")
    hr("=")
    bl()
    print(f"      Disease Mechanism:")
    wrap(kb["pathophysiology"], indent=8, width=62)
    bl()
    row("Key Diagnostic Markers", kb["key_markers"])
    row("Expected Prognosis", kb["prognosis"])

    # ================================================================
    # SECTION 3: SYMPTOM ANALYSIS
    # ================================================================
    sym = agents.get("symptom", {})
    symptoms = sym.get("extracted_symptoms", [])
    bl()
    hr("=")
    print(f"  SECTION 3: CLINICAL SYMPTOM ANALYSIS")
    hr("=")
    bl()
    if symptoms:
        print(f"      {'Symptom':<28s} {'Severity':<14s} {'Duration':<16s} {'Match Type'}")
        print(f"      {'.' * 28} {'.' * 14} {'.' * 16} {'.' * 16}")
        for s in symptoms:
            sname = s.get("canonical_name", s.get("raw_text", "?"))
            ssev = s.get("severity", "Moderate")
            sdur = "-"
            for d in inp.get("symptom_durations", []):
                if d["symptom"].lower() in sname.lower() or sname.lower() in d["symptom"].lower():
                    sdur = d["duration"]; break
            src = s.get("match_type", "NLP extraction")
            print(f"      {sname:<28s} {ssev:<14s} {sdur:<16s} {src}")
        bl()
        row("Total Symptoms Identified", f'{len(symptoms)}')
        row("Symptom-Disease Correlation", "Consistent" if conf > 0.7 else "Partial")
    else:
        print(f"      No structured symptoms extracted from clinical text.")

    # ================================================================
    # SECTION 4: RISK STRATIFICATION
    # ================================================================
    risk = agents.get("risk", {})
    bl()
    hr("=")
    print(f"  SECTION 4: RISK STRATIFICATION")
    hr("=")
    bl()
    rl = risk.get("overall_risk_level", "Low")
    rf = risk.get("risk_factors_identified", [])
    row("Overall Risk Level", rl)
    row("Identified Risk Factors", f'{len(rf)} factors')
    if rf:
        bl()
        print(f"      {'Risk Factor':<36s} {'Weight':<10s} {'Category'}")
        print(f"      {'.' * 36} {'.' * 10} {'.' * 16}")
        for f in rf:
            fn = f.get("factor", str(f)) if isinstance(f, dict) else str(f)
            fw = f.get("weight", "-") if isinstance(f, dict) else "-"
            fc = f.get("category", "Clinical") if isinstance(f, dict) else "Clinical"
            print(f"      {fn:<36s} {str(fw):<10s} {fc}")

    # ================================================================
    # SECTION 5: EMERGENCY TRIAGE ASSESSMENT
    # ================================================================
    bl()
    hr("=")
    print(f"  SECTION 5: EMERGENCY TRIAGE ASSESSMENT")
    hr("=")
    bl()
    is_emg = emg.get("is_emergency", False)
    triage = emg.get("triage_level", "N/A")
    row("Emergency Classification", "CRITICAL -- IMMEDIATE INTERVENTION REQUIRED" if is_emg else "Non-Emergency")
    row("Triage Level (ESI)", f'Level {triage}')
    flags = emg.get("vital_flags", [])
    if flags:
        bl()
        print(f"      Abnormal Vital Sign Flags:")
        for fl in flags:
            vital = fl.get('vital', '?')
            value = fl.get('value', '?')
            threshold = fl.get('threshold', '?')
            print(f"        [FLAG] {vital}: {value}  (Normal threshold: {threshold})")

    # Temporal analysis
    temporal = agents.get("temporal", {})
    bl()
    row("Symptom Progression", temporal.get("progression_pattern", "Stable"))
    row("Temporal Urgency", temporal.get("overall_urgency", "N/A"))
    row("Onset Category", temporal.get("onset_category", "N/A"))

    # ================================================================
    # SECTION 6: ML PREDICTION ANALYSIS
    # ================================================================
    pred = agents.get("prediction", {})
    bl()
    hr("=")
    print(f"  SECTION 6: PREDICTIVE MODEL OUTPUT")
    hr("=")
    bl()
    ml_disease = pred.get("primary_disease", "N/A")
    ml_conf = pred.get("primary_confidence", 0)
    row("Ensemble Prediction", ml_disease)
    row("Prediction Probability", f'{ml_conf:.4f}  ({ml_conf:.1%})')
    row("Model Architecture", "Voting Ensemble (XGBoost + RF + LightGBM)")

    top = pred.get("top_diseases", [])
    if top:
        bl()
        print(f"      Ranked Differential (ML-based):")
        print(f"      {'Rank':<6s} {'Disease':<34s} {'Probability':<14s} {'Decision'}")
        print(f"      {'----':<6s} {'.' * 34} {'.' * 14} {'.' * 10}")
        for i, tp in enumerate(top[:5], 1):
            d = tp.get("disease", "?")
            p = tp.get("confidence", 0)
            note = "<< Selected" if i == 1 else ""
            print(f"      {i:<6d} {d:<34s} {p:<14.4f} {note}")

    feats = pred.get("features_used", {})
    if feats:
        bl()
        print(f"      Active Feature Signals ({len(feats)} features):")
        feat_items = list(feats.items())[:10]
        for fn, fv in feat_items:
            clean_name = fn.replace('_', ' ').title()
            print(f"        {clean_name:<32s} = {fv}")

    # ================================================================
    # SECTION 7: DIFFERENTIAL DIAGNOSIS & LLM REASONING
    # ================================================================
    diff = agents.get("differential", {})
    diff_list = diff.get("differential_diagnoses", diff.get("differential_list", []))
    llm_source = final.get("llm_source", "none")
    llm_reasoning = final.get("llm_reasoning", "")
    bl()
    hr("=")
    print(f"  SECTION 7: DIFFERENTIAL DIAGNOSIS & CLINICAL REASONING")
    hr("=")
    bl()

    if diff_list:
        print(f"      Differential Diagnosis List:")
        print(f"      {'#':<4s} {'Condition':<36s} {'Match Score':<14s}")
        print(f"      {'--':<4s} {'.' * 36} {'.' * 14}")
        for i, d in enumerate(diff_list[:6], 1):
            if isinstance(d, dict):
                dn = d.get('disease', d.get('condition', '?'))
                dc = d.get('confidence', d.get('match_score', 0))
                print(f"      {i:<4d} {dn:<36s} {dc:<14.1%}")
            elif isinstance(d, str):
                print(f"      {i:<4d} {d:<36s}")

    bl()
    row("LLM Fallback Invoked", "Yes" if llm_source != "none" else "No (confidence sufficient)")
    if llm_source != "none":
        row("LLM Model Used", llm_source)

    # Determine best clinical reasoning to show
    cleaned_llm = clean_llm(llm_reasoning) if llm_reasoning else ""
    has_useful_llm = is_llm_output_useful(cleaned_llm)

    bl()
    print(f"      Clinical Reasoning:")
    print(f"      " + "-" * 55)

    if has_useful_llm:
        # Show cleaned LLM reasoning
        print(f"      [Source: {llm_source} LLM Model]")
        bl()
        wrap(cleaned_llm[:1000], indent=8, width=62)
    else:
        # Generate structured KB-based clinical reasoning
        kb_reasoning = generate_clinical_reasoning(disease, kb, inp, final)
        print(f"      [Source: Knowledge-Based Clinical Analysis]")
        bl()
        wrap(kb_reasoning, indent=8, width=62)

    # ================================================================
    # SECTION 8: DIAGNOSTIC WORKUP
    # ================================================================
    tests = final.get("recommended_tests", [])
    bl()
    hr("=")
    print(f"  SECTION 8: RECOMMENDED DIAGNOSTIC WORKUP")
    hr("=")
    bl()
    if tests:
        print(f"      {'#':<4s} {'Investigation':<36s} {'Priority':<12s} {'Department'}")
        print(f"      {'--':<4s} {'.' * 36} {'.' * 12} {'.' * 16}")
        for i, t in enumerate(tests, 1):
            if isinstance(t, dict):
                tname = t.get('test', '?')
                tpri = t.get('priority', '-')
                tcat = t.get('category', '-')
                print(f"      {i:<4d} {tname:<36s} {tpri:<12s} {tcat}")
                why = t.get("reason", t.get("why", ""))
                if why:
                    print(f"           Clinical Indication: {why}")
    else:
        print(f"      No specific investigations matched in knowledge base.")
        print(f"      Recommend: CMP, CBC w/ diff, disease-specific workup.")

    # ================================================================
    # SECTION 9: PHARMACOTHERAPY
    # ================================================================
    meds = final.get("recommended_medications", [])
    bl()
    hr("=")
    print(f"  SECTION 9: PHARMACOTHERAPY RECOMMENDATIONS")
    hr("=")
    bl()
    if meds:
        for i, m in enumerate(meds, 1):
            if isinstance(m, dict):
                print(f"      {i}. {m.get('drug','?')}  [{m.get('category','')}]")
                print(f"         Dosage              : {m.get('dosage','')}")
                print(f"         Route               : {m.get('route','Oral')}")
                print(f"         Adverse Effects      : {m.get('side_effects','None documented')}")
                print(f"         Contraindications    : {m.get('contraindications','None documented')}")
                note = m.get("precaution", "")
                if note:
                    print(f"         Clinical Precaution  : {note}")
                bl()
    else:
        print(f"      No medications matched in pharmacological knowledge base.")
        print(f"      Prescribe based on clinical judgement and confirmed diagnosis.")

    # ================================================================
    # SECTION 10: MANAGEMENT PLAN
    # ================================================================
    plan = final.get("treatment_plan", {})
    hr("=")
    print(f"  SECTION 10: CLINICAL MANAGEMENT PLAN")
    hr("=")
    bl()
    row("Recommended Treatment", plan.get("treatment_plan", "As per clinical protocol"))
    row("Clinical Urgency", plan.get("urgency", severity))
    row("Follow-up Schedule", plan.get("follow_up", "Per clinical discretion"))
    row("Risk Classification", final.get("risk_level", "N/A"))
    row("Emergency Protocol", "Activated" if emg.get("is_emergency") else "Standard care pathway")

    # ── Clinical Alerts ──
    alerts = final.get("risk_alerts", [])
    if alerts:
        bl()
        hr("-")
        print(f"  CLINICAL ALERTS & CONTRAINDICATION FLAGS")
        hr("-")
        bl()
        for a in alerts:
            ca = a.replace("[!]", "[WARN]").replace("[!!]", "[CRITICAL]").strip()
            wrap(f">> {ca}", indent=6, width=64)
            bl()

    # ── Footer ──
    bl()
    hr("=")
    print(f"  AI-Assisted Clinical Report | MedAgentix AI v2.0")
    print(f"  Pipeline: ClinicalBERT NER + Voting Ensemble (XGB/RF/LGBM)")
    print(f"  Confidence Routing: >85% Direct | 70-85% Validated | <70% Meditron-7B")
    print(f"  Framework: LangGraph Multi-Agent Orchestration")
    hr("-")
    print(f"  DISCLAIMER: AI-generated diagnostic support tool. All findings")
    print(f"  require clinical correlation. Verify diagnosis with laboratory")
    print(f"  investigations and imaging before initiating treatment.")
    print(f"  Not a substitute for clinical judgement.")
    hr("=")
    bl()


# ============================================================
# TEST CASES
# ============================================================
TEST_CASES = [
    {
        "name": "Case 1: Fever with Cough",
        "input": {
            "patient_name": "Rahul Sharma",
            "patient_age": 35,
            "patient_gender": "Male",
            "patient_text": "I have been having fever, cough, fatigue, and body pain for the past 5 days. Also experiencing headache and some difficulty breathing.",
            "symptom_durations": [
                {"symptom": "Fever", "duration": "5 days"},
                {"symptom": "Cough", "duration": "5 days"},
                {"symptom": "Body Pain", "duration": "5 days"},
                {"symptom": "Headache", "duration": "3 days"},
                {"symptom": "Breathing", "duration": "2 days"},
            ],
            "heart_rate": 85, "oxygen_level": 96,
            "blood_pressure": "Normal", "blood_pressure_reading": "120/80",
            "body_temperature": 101.5, "cholesterol": 190,
            "lifestyle_factors": [], "medical_history": [],
        },
    },
    {
        "name": "Case 2: Chest Pain (Elderly)",
        "input": {
            "patient_name": "Ramesh Gupta",
            "patient_age": 72,
            "patient_gender": "Male",
            "patient_text": "Experiencing chest pain, breathlessness, and fatigue. Had some vomiting this morning.",
            "symptom_durations": [
                {"symptom": "Chest Pain", "duration": "since yesterday"},
                {"symptom": "Breathlessness", "duration": "2 days"},
            ],
            "heart_rate": 110, "oxygen_level": 91,
            "blood_pressure": "High", "blood_pressure_reading": "165/100",
            "body_temperature": 99.2, "cholesterol": 260,
            "lifestyle_factors": ["Smoking", "Obesity"],
            "medical_history": ["Cardiac History", "Hypertension"],
        },
    },
    {
        "name": "Case 3: Runny Nose & Headache",
        "input": {
            "patient_name": "Priya Mehta",
            "patient_age": 25,
            "patient_gender": "Female",
            "patient_text": "I have a runny nose and mild headache. Feeling a bit tired.",
            "symptom_durations": [
                {"symptom": "Runny Nose", "duration": "2 days"},
                {"symptom": "Headache", "duration": "today"},
            ],
            "heart_rate": 72, "oxygen_level": 99,
            "blood_pressure": "Normal", "blood_pressure_reading": "110/70",
            "body_temperature": 98.8, "cholesterol": 170,
            "lifestyle_factors": [], "medical_history": [],
        },
    },
]


# ============================================================
# MAIN
# ============================================================
def main():
    warnings.filterwarnings("ignore")

    # Output directory
    report_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'reports')
    os.makedirs(report_dir, exist_ok=True)

    bl()
    hdr("MedAgentix AI -- Intelligent Health Assessment System")
    print(f"  Version 2.0  |  Multi-Agent Diagnostic Pipeline")
    print(f"  Models: ClinicalBERT NER + Voting Ensemble + Meditron 7B LLM")
    hr("-")
    print(f"  Initializing agents...")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        from agents.orchestrator.langgraph_workflow import run_pipeline

    print(f"  All agents loaded. Processing {len(TEST_CASES)} case(s).")
    hr("-")
    print(f"  Reports will be saved to: {os.path.abspath(report_dir)}")
    hr("=")

    patient_buf = io.StringIO()
    doctor_buf = io.StringIO()

    for test in TEST_CASES:
        print(f"\n  >> Processing: {test['name']} ...")

        # Run pipeline silently
        buf = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            result = run_pipeline(test["input"])

        final = result.get("final_diagnosis", {})
        disease = final.get("final_disease", "?")
        conf = final.get("final_confidence", 0)
        severity = final.get("severity", "?")
        print(f"     Diagnosis: {disease} ({conf:.0%}) | Severity: {severity}")

        # Capture patient report
        with contextlib.redirect_stdout(patient_buf):
            patient_dash(test["name"], test["input"], result)

        # Capture doctor report
        with contextlib.redirect_stdout(doctor_buf):
            doctor_dash(test["name"], test["input"], result)

    # Write patient report
    patient_file = os.path.join(report_dir, "patient_report.txt")
    with open(patient_file, "w", encoding="utf-8") as f:
        f.write("=" * W + "\n")
        f.write("  MedAgentix AI -- YOUR HEALTH ASSESSMENT REPORT\n")
        f.write(f"  Generated: {datetime.datetime.now().strftime('%d-%b-%Y %H:%M')}\n")
        f.write("=" * W + "\n")
        f.write(patient_buf.getvalue())
        f.write("\n" + "=" * W + "\n")
        f.write("  End of Patient Report\n")
        f.write("=" * W + "\n")

    # Write doctor report
    doctor_file = os.path.join(report_dir, "doctor_report.txt")
    with open(doctor_file, "w", encoding="utf-8") as f:
        f.write("#" * W + "\n")
        f.write("  MedAgentix AI -- CLINICAL DIAGNOSTIC REPORT\n")
        f.write(f"  Generated: {datetime.datetime.now().strftime('%d-%b-%Y %H:%M')}\n")
        f.write("#" * W + "\n")
        f.write(doctor_buf.getvalue())
        f.write("\n" + "#" * W + "\n")
        f.write("  End of Clinical Report\n")
        f.write("#" * W + "\n")

    # Terminal summary
    hr("=")
    print(f"\n  REPORTS GENERATED SUCCESSFULLY")
    hr("-")
    print(f"  Patient Report : {os.path.abspath(patient_file)}")
    print(f"  Doctor Report  : {os.path.abspath(doctor_file)}")
    hr("-")
    print(f"  Open these files to view the formatted reports.")
    hr("=")
    bl()


if __name__ == "__main__":
    main()
