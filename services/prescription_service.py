# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Prescription Synthesis Service
=================================================
Formats the raw, highly structured Supervisor Agent diagnostic JSON dictionary
into a clean, human-readable, and well-aligned doctor's prescription layout.
"""

from datetime import datetime

class PrescriptionService:
    """
    PrescriptionService — Transforms supervisor diagnostic dictionary 
    into a structured and polished medical prescription text format.
    """

    def __init__(self):
        pass

    def format_prescription(self, final: dict, patient_info: dict = None) -> str:
        """
        Synthesize the supervisor final diagnosis dictionary into a polished textual prescription.
        
        Args:
            final: The final_diagnosis dictionary from the Supervisor Agent.
            patient_info: Optional secondary patient profile dict (vitals, age, gender).
            
        Returns:
            str: Beautifully aligned textual prescription.
        """
        if not final:
            return "  [ERROR] No diagnostic data available to formulate prescription."

        # Extract values safely
        disease = final.get("final_disease", "Unknown")
        confidence = final.get("final_confidence", 0)
        conf_level = final.get("confidence_level", "N/A").upper()
        severity = final.get("severity", "N/A")
        source = final.get("diagnosis_source", "N/A").replace("_", " ")
        agreement = final.get("agent_agreement", 0)
        reasoning = final.get("reasoning", "N/A")

        # Patient profile parsing
        age = final.get("patient_age", patient_info.get("age", "N/A") if patient_info else "N/A")
        gender = final.get("patient_gender", patient_info.get("gender", "N/A") if patient_info else "N/A")
        
        # Vitals extraction
        emerg_status = final.get("emergency_status", {})
        vital_flags = emerg_status.get("vital_flags", [])
        
        hr = patient_info.get("heart_rate", "N/A") if patient_info else "N/A"
        bp = patient_info.get("blood_pressure_reading", "N/A") if patient_info else "N/A"
        spo2 = patient_info.get("oxygen_level", "N/A") if patient_info else "N/A"
        temp = patient_info.get("body_temperature", "N/A") if patient_info else "N/A"

        # Alternative possibilities
        alts = final.get("alternatives", [])
        
        # Recommendations
        tests = final.get("recommended_tests", [])
        meds = final.get("recommended_medications", [])
        alerts = final.get("risk_alerts", [])
        plan = final.get("treatment_plan", {})
        
        # Formulate string builder
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        lines = []
        lines.append("=" * 64)
        lines.append("                    MEDAGENTIX AI CLINICAL PORTAL")
        lines.append("                      AUTOMATED RX PRESCRIPTION")
        lines.append("=" * 64)
        lines.append(f"  Date & Time: {now_str:<32}")
        lines.append(f"  Facility:    MedAgentix Virtual Care Unit")
        lines.append("-" * 64)
        
        # Patient Card
        lines.append("  PATIENT CLINICAL PROFILE:")
        lines.append(f"    Age:    {age:<10} | Gender: {gender}")
        lines.append(f"    Vitals: HR: {hr} bpm | SpO2: {spo2}% | BP: {bp} mmHg | Temp: {temp}°F")
        if vital_flags:
            formatted_flags = []
            for vf in vital_flags:
                if isinstance(vf, dict):
                    name = vf.get("vital", vf.get("name", "Unknown"))
                    val = vf.get("value", "")
                    status = vf.get("status", "Alert")
                    formatted_flags.append(f"{name}={val} ({status})")
                else:
                    formatted_flags.append(str(vf))
            lines.append(f"    Vital Alerts Flags: {', '.join(formatted_flags)}")
        lines.append("-" * 64)
        
        # Diagnostic Assessment
        lines.append("  CLINICAL DIAGNOSIS:")
        lines.append(f"    Primary Suspected: {disease:<30}")
        lines.append(f"    Severity Level:    {severity:<30}")
        lines.append(f"    ML Confidence:     {confidence:.1%} ({conf_level})")
        lines.append(f"    Agent Agreement:   {agreement:.0%} Consensus")
        lines.append(f"    Diagnostic Source: {source.title()}")
        lines.append("-" * 64)
        
        # Reasoning
        lines.append("  CLINICAL REASONING & FINDINGS:")
        # Wrap reasoning text to fit within 60 chars nicely
        wrapped_reasoning = self._wrap_text(reasoning, 58)
        for r_line in wrapped_reasoning:
            lines.append(f"    {r_line}")
            
        # Alternatives if low confidence
        if alts:
            lines.append("")
            lines.append("    Differential Alternatives Investigated:")
            for idx, alt in enumerate(alts[:3], 1):
                name = alt.get("disease", "?")
                score = alt.get("confidence", alt.get("weighted_score", 0))
                lines.append(f"      {idx}. {name:<26} (Score: {score:.4f})")
        lines.append("-" * 64)
        
        # Rx Medications
        lines.append("  Rx - MEDICATIONS RECOMMENDED:")
        if meds:
            for idx, m in enumerate(meds, 1):
                drug = m.get("drug", "Unknown")
                dosage = m.get("dosage", "N/A")
                route = m.get("route", "Oral")
                freq = m.get("frequency", "As directed")
                lines.append(f"    {idx}. {drug:<22} - {dosage} ({route})")
                lines.append(f"       Instructions: {freq}")
        else:
            lines.append("    - No specific medications mapped for this condition.")
            lines.append("      Refer to emergency triage guidelines below.")
        lines.append("-" * 64)
        
        # Recommended Tests
        lines.append("  RECOMMENDED DIAGNOSTIC LAB TESTS:")
        if tests:
            for idx, t in enumerate(tests, 1):
                t_name = t.get("test", t) if isinstance(t, dict) else t
                lines.append(f"    - {t_name}")
        else:
            lines.append("    - No clinical diagnostic lab tests recommended.")
        lines.append("-" * 64)
        
        # Clinical Risk Alerts
        if alerts:
            lines.append("  CLINICAL RISK ALERTS & WARNINGS:")
            for a in alerts:
                lines.append(f"    ! {a}")
            lines.append("-" * 64)
            
        # Treatment & Care plan
        lines.append("  TREATMENT CARE PLAN & FOLLOW-UP:")
        t_plan = plan.get("treatment_plan", "N/A")
        f_up = plan.get("follow_up", "N/A")
        
        wrapped_plan = self._wrap_text(f"Plan: {t_plan}", 58)
        for pl in wrapped_plan:
            lines.append(f"    {pl}")
        
        wrapped_fup = self._wrap_text(f"Follow-up: {f_up}", 58)
        for fu in wrapped_fup:
            lines.append(f"    {fu}")
        lines.append("-" * 64)
        
        # Official Disclaimer
        disclaimer = final.get("disclaimer", "⚕ DISCLAIMER: AI-generated diagnostic assessment.")
        wrapped_disc = self._wrap_text(disclaimer, 58)
        for dl in wrapped_disc:
            lines.append(f"    {dl}")
        lines.append("=" * 64)
        
        return "\n".join(lines)

    def _wrap_text(self, text: str, width: int) -> list:
        """Wrap text into lines of a maximum character width."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            # Check if word fits
            if current_length + len(word) + (1 if current_line else 0) <= width:
                current_line.append(word)
                current_length += len(word) + (1 if len(current_line) > 1 else 0)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
                
        if current_line:
            lines.append(" ".join(current_line))
            
        return lines
