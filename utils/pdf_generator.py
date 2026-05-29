# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Clinical PDF Prescription Generator
======================================================
Compiles raw supervisor diagnostic outcomes and patient profiles
into an elegant, clinical-grade downloadable PDF document using ReportLab.
"""

import os
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

class PDFGenerator:
    """
    PDFGenerator — Renders structured clinical diagnostics and treatment plans
    into an official, polished, printable PDF prescription sheet.
    """

    def __init__(self):
        # Configure clinical color palette
        self.primary_color = colors.HexColor("#0f172a")    # Slate dark for borders
        self.secondary_color = colors.HexColor("#1e3a8a")  # Deep blue for headers
        self.accent_color = colors.HexColor("#dc2626")     # Red for warnings
        self.bg_light = colors.HexColor("#f8fafc")         # Light gray-blue for table headers
        self.text_dark = colors.HexColor("#334155")        # Charcoal for readable text

    def generate_prescription_pdf(self, final: dict, patient_info: dict, output_path: str) -> str:
        """
        Synthesize supervisor output and patient profile into a clinical PDF document.

        Args:
            final (dict): Supervisor final_diagnosis output.
            patient_info (dict): Patient demographics and vital signs.
            output_path (str): File system destination path to write the PDF.

        Returns:
            str: The output path of the generated PDF.
        """
        # Ensure target directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Setup document template
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=35,
            bottomMargin=35
        )

        # Get styles
        styles = getSampleStyleSheet()
        
        # Define custom typography styles
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=22,
            textColor=colors.white,
            alignment=TA_CENTER,
            spaceAfter=5
        )
        
        subtitle_style = ParagraphStyle(
            'DocSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor("#cbd5e1"),
            alignment=TA_CENTER
        )

        h1_style = ParagraphStyle(
            'H1',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=12,
            textColor=self.secondary_color,
            spaceBefore=10,
            spaceAfter=6,
            borderColor=self.secondary_color
        )

        body_style = ParagraphStyle(
            'BodyText',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            textColor=self.text_dark,
            leading=13
        )

        body_bold_style = ParagraphStyle(
            'BodyTextBold',
            parent=body_style,
            fontName='Helvetica-Bold'
        )

        alert_style = ParagraphStyle(
            'AlertText',
            parent=body_style,
            fontName='Helvetica-Bold',
            textColor=self.accent_color
        )

        disclaimer_style = ParagraphStyle(
            'DisclaimerText',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=7,
            textColor=colors.HexColor("#64748b"),
            alignment=TA_CENTER,
            leading=10
        )

        # Story flow list
        story = []

        # ----------------------------------------------------
        # 1. CLINICAL HEADER BANNER (Slate/Blue Banner)
        # ----------------------------------------------------
        banner_data = [
            [Paragraph("MEDAGENTIX AI CLINICAL PORTAL", title_style)],
            [Paragraph("AUTOMATED MEDICAL DIAGNOSTIC PRESCRIPTION SHEET", subtitle_style)]
        ]
        banner_table = Table(banner_data, colWidths=[532])
        banner_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.secondary_color),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 12),
            ('BOX', (0, 0), (-1, -1), 1, self.primary_color),
        ]))
        story.append(banner_table)
        story.append(Spacer(1, 10))

        # ----------------------------------------------------
        # 2. METADATA ROW (Date / Facility)
        # ----------------------------------------------------
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        meta_data = [
            [
                Paragraph(f"<b>Date:</b> {now_str}", body_style),
                Paragraph("<b>Facility:</b> MedAgentix Virtual Care Unit", body_style)
            ]
        ]
        meta_table = Table(meta_data, colWidths=[200, 332])
        meta_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 10))

        # ----------------------------------------------------
        # 3. PATIENT PROFILE CARD TABLE
        # ----------------------------------------------------
        age = final.get("patient_age", patient_info.get("age", "N/A"))
        gender = final.get("patient_gender", patient_info.get("gender", "N/A"))
        hr = patient_info.get("heart_rate", "N/A")
        bp = patient_info.get("blood_pressure_reading", "N/A")
        spo2 = patient_info.get("oxygen_level", "N/A")
        temp = patient_info.get("body_temperature", "N/A")
        vital_flags = final.get("emergency_status", {}).get("vital_flags", [])
        formatted_flags = []
        for vf in vital_flags:
            if isinstance(vf, dict):
                name = vf.get("vital", vf.get("name", "Unknown"))
                val = vf.get("value", "")
                status = vf.get("status", "Alert")
                formatted_flags.append(f"{name}={val} ({status})")
            else:
                formatted_flags.append(str(vf))

        profile_data = [
            [
                Paragraph("<b>PATIENT PROFILE</b>", body_bold_style), "",
                Paragraph("<b>VITALS RECORDED</b>", body_bold_style), ""
            ],
            [
                Paragraph(f"Age:", body_style), Paragraph(f"{age}", body_style),
                Paragraph(f"Heart Rate:", body_style), Paragraph(f"{hr} bpm", body_style)
            ],
            [
                Paragraph(f"Gender:", body_style), Paragraph(f"{gender}", body_style),
                Paragraph(f"Oxygen Sat. (SpO2):", body_style), Paragraph(f"{spo2}%", body_style)
            ],
            [
                Paragraph(f"Vital Alerts:", body_style),
                Paragraph(f"{', '.join(formatted_flags) if formatted_flags else 'Normal / Stable'}", alert_style if formatted_flags else body_style),
                Paragraph(f"Blood Pressure (BP):", body_style), Paragraph(f"{bp} mmHg", body_style)
            ],
            [
                "", "",
                Paragraph(f"Body Temp:", body_style), Paragraph(f"{temp} °F", body_style)
            ]
        ]
        
        profile_table = Table(profile_data, colWidths=[80, 186, 120, 146])
        profile_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), self.bg_light),
            ('BACKGROUND', (2, 0), (3, 0), self.bg_light),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LINEBELOW', (0, 0), (1, 0), 1, self.secondary_color),
            ('LINEBELOW', (2, 0), (3, 0), 1, self.secondary_color),
            ('BOX', (0, 0), (1, 3), 0.5, colors.HexColor("#cbd5e1")),
            ('BOX', (2, 0), (3, 4), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        story.append(profile_table)
        story.append(Spacer(1, 12))

        # ----------------------------------------------------
        # 4. PRIMARY CLINICAL ASSESSMENT BOX
        # ----------------------------------------------------
        disease = final.get("final_disease", "Unknown")
        confidence = final.get("final_confidence", 0)
        conf_level = final.get("confidence_level", "N/A").upper()
        severity = final.get("severity", "N/A")
        agreement = final.get("agent_agreement", 0)
        source = final.get("diagnosis_source", "N/A").replace("_", " ").title()

        assessment_data = [
            [
                Paragraph(f"<b>PRIMARY DIAGNOSIS:</b> {disease}", ParagraphStyle('PDiag', parent=body_bold_style, fontSize=11, textColor=self.secondary_color)),
                Paragraph(f"<b>Severity:</b> {severity}", ParagraphStyle('PSev', parent=body_bold_style, fontSize=10, textColor=self.accent_color if severity in ("Critical", "Severe") else self.text_dark))
            ],
            [
                Paragraph(f"<b>ML Model Confidence:</b> {confidence:.1%} ({conf_level})", body_style),
                Paragraph(f"<b>Consensus Agreement:</b> {agreement:.0%} Agent Voting", body_style)
            ],
            [
                Paragraph(f"<b>Diagnostic Source Path:</b> {source}", body_style), ""
            ]
        ]
        
        assessment_table = Table(assessment_data, colWidths=[266, 266])
        assessment_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#93c5fd")),
        ]))
        story.append(assessment_table)
        story.append(Spacer(1, 10))

        # ----------------------------------------------------
        # 5. CLINICAL FINDINGS & ALTERNATIVES
        # ----------------------------------------------------
        story.append(Paragraph("CLINICAL REASONING & RELEVANT PATHWAYS", h1_style))
        story.append(Paragraph(final.get("reasoning", "N/A"), body_style))
        
        # Alternatives row
        alts = final.get("alternatives", [])
        if alts:
            alt_texts = []
            for alt in alts[:3]:
                a_name = alt.get("disease", "?")
                a_score = alt.get("confidence", alt.get("weighted_score", 0))
                alt_texts.append(f"<b>{a_name}</b> (Score: {a_score:.3f})")
            
            story.append(Spacer(1, 4))
            story.append(Paragraph(f"<b>Differential Candidates Assessed:</b> &nbsp;&nbsp;{', &nbsp;&nbsp;'.join(alt_texts)}", ParagraphStyle('AltText', parent=body_style, fontSize=8)))
        
        story.append(Spacer(1, 10))

        # ----------------------------------------------------
        # 5b. AI DIAGNOSTICS & SHAP EXPLAINABILITY (XAI)
        # ----------------------------------------------------
        shap_explanation = final.get("shap_explanation", {})
        if shap_explanation and isinstance(shap_explanation, dict):
            plot_path = shap_explanation.get("plot_path")
            if plot_path and os.path.exists(plot_path):
                story.append(Paragraph("AI EXPLAINABILITY & BIOMARKER REASONING (SHAP)", h1_style))
                story.append(Paragraph(
                    "Below is the Shapley additive explanations (SHAP) feature importance chart. "
                    "It shows which patient biomarkers and symptoms contributed to the model's primary diagnostic decision. "
                    "<b>Blue bars (positive)</b> indicate features that supported this diagnosis, while "
                    "<b>red bars (negative)</b> indicate factors that pushed against it.", 
                    body_style
                ))
                story.append(Spacer(1, 5))
                # Add the horizontal bar plot centered
                shap_img = Image(plot_path, width=390, height=192)
                shap_img.hAlign = 'CENTER'
                story.append(shap_img)
                story.append(Spacer(1, 10))

        # ----------------------------------------------------
        # 6. Rx - PHARMACEUTICAL DRUG RECOMMENDATIONS
        # ----------------------------------------------------
        story.append(Paragraph("Rx - PRESCRIBED MEDICATIONS", h1_style))
        meds = final.get("recommended_medications", [])
        
        med_headers = [
            Paragraph("<b>Medication Name</b>", body_bold_style),
            Paragraph("<b>Dosage</b>", body_bold_style),
            Paragraph("<b>Route</b>", body_bold_style),
            Paragraph("<b>Timing & Frequency</b>", body_bold_style)
        ]
        
        med_rows = [med_headers]
        if meds:
            for m in meds:
                med_rows.append([
                    Paragraph(f"<b>{m.get('drug', 'Unknown')}</b>", body_style),
                    Paragraph(m.get("dosage", "As directed"), body_style),
                    Paragraph(m.get("route", "Oral"), body_style),
                    Paragraph(m.get("frequency", "Once Daily"), body_style)
                ])
        else:
            med_rows.append([
                Paragraph("<i>No standard medications mapped for this condition. Please consult triage alerts.</i>", body_style),
                "", "", ""
            ])

        med_table = Table(med_rows, colWidths=[180, 100, 80, 172])
        med_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.bg_light),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LINEBELOW', (0, 0), (-1, 0), 1, self.secondary_color),
            ('LINEBELOW', (0, 1), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        story.append(med_table)
        story.append(Spacer(1, 12))

        # ----------------------------------------------------
        # 7. DIAGNOSTIC LAB TESTS & ADVICE
        # ----------------------------------------------------
        story.append(Paragraph("RECOMMENDED CLINICAL DIAGNOSTIC LAB TESTS", h1_style))
        tests = final.get("recommended_tests", [])
        if tests:
            test_items = []
            for t in tests[:5]:
                t_name = t.get("test", t) if isinstance(t, dict) else t
                test_items.append(f"• {t_name}")
            story.append(Paragraph(" &nbsp;&nbsp;&nbsp;&nbsp; ".join(test_items), body_bold_style))
        else:
            story.append(Paragraph("No diagnostic laboratory tests mapped for this condition.", body_style))
            
        story.append(Spacer(1, 10))

        # ----------------------------------------------------
        # 8. CLINICAL WARNINGS & TREATMENT PLAN
        # ----------------------------------------------------
        story.append(Paragraph("CRITICAL CLINICAL ALERTS & TREATMENT PLAN", h1_style))
        
        alerts = final.get("risk_alerts", [])
        if alerts:
            for idx, a in enumerate(alerts, 1):
                story.append(Paragraph(f"🚨 &nbsp;<b>Alert {idx}:</b> {a}", alert_style))
                story.append(Spacer(1, 2))
                
        plan = final.get("treatment_plan", {})
        care_plan = plan.get("treatment_plan", "N/A")
        follow_up = plan.get("follow_up", "N/A")
        
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Care Plan Instruction:</b> {care_plan}", body_style))
        story.append(Paragraph(f"<b>Clinical Follow-Up:</b> {follow_up}", body_style))
        story.append(Spacer(1, 12))

        # ----------------------------------------------------
        # 9. SIGNATURES & OFFICIAL MEDICAL DISCLAIMER
        # ----------------------------------------------------
        story.append(Spacer(1, 10))
        sig_data = [
            [
                Paragraph("<b>Signature:</b> ___________________________<br/>Attending Virtual Physician, MD", body_style),
                Paragraph("<b>Authorized Verification Seal:</b><br/>[ MedAgentix Autonomous System Validated ]", ParagraphStyle('Seal', parent=body_style, textColor=self.secondary_color, alignment=TA_RIGHT))
            ]
        ]
        sig_table = Table(sig_data, colWidths=[266, 266])
        sig_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(sig_table)
        story.append(Spacer(1, 15))

        disclaimer = final.get("disclaimer", "⚕ DISCLAIMER: AI-generated diagnostic assessment.")
        story.append(Paragraph(disclaimer, disclaimer_style))

        # Build document
        doc.build(story)
        return output_path
