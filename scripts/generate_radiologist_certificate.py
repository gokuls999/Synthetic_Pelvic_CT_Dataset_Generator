# -*- coding: utf-8 -*-
"""generate_radiologist_certificate.py
Generates a formal radiologist verification & approval certificate (PDF).
Output: Radiologist_Verification_Certificate.pdf
"""
from fpdf import FPDF
from pathlib import Path

OUT   = Path("Radiologist_Verification_Certificate.pdf")
L, R  = 25, 25
TM    = 28
BODYW = 210 - L - R
LH    = 6.5

THESIS = (
    '"A COMPUTED TOMOGRAPHY EVALUATION OF PELVIC FLOOR THICKNESS AND BONY '
    "DIMENSIONS AMONG WOMEN FROM HILLY AND INLAND REGIONS OF SOUTH INDIA: "
    'A COMPARATIVE CROSS-SECTIONAL STUDY"'
)


class CertPDF(FPDF):
    def header(self):
        self.set_line_width(0.8)
        self.set_draw_color(30, 64, 120)
        self.line(L, 14, 210 - R, 14)

    def footer(self):
        self.set_y(-18)
        self.set_line_width(0.5)
        self.set_draw_color(30, 64, 120)
        self.line(L, self.get_y(), 210 - R, self.get_y())
        self.ln(2)
        self.set_font("Times", "I", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 5,
                  "Confidential - For Academic & Institutional Review Only  |  "
                  "Dept. of Radiodiagnosis & Imaging",
                  align="C")


def hline(pdf, y=None, color=(100, 100, 100), width=0.3):
    if y is None:
        y = pdf.get_y()
    pdf.set_draw_color(*color)
    pdf.set_line_width(width)
    pdf.line(L, y, 210 - R, y)


def para(pdf, text, indent=0, style="", size=11, color=(20, 20, 20)):
    pdf.set_font("Times", style, size)
    pdf.set_text_color(*color)
    pdf.set_x(L + indent)
    pdf.multi_cell(BODYW - indent, LH, text, align="J")
    pdf.ln(2)


def item(pdf, num, heading, body):
    pdf.set_x(L + 4)
    pdf.set_font("Times", "B", 11)
    pdf.set_text_color(15, 40, 90)
    tag = "%s.  %s: " % (num, heading)
    tw  = pdf.get_string_width(tag) + 1
    pdf.cell(tw, LH, tag, ln=0)
    pdf.set_font("Times", "", 11)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(BODYW - 4 - tw, LH, body, align="J")
    pdf.ln(1)


def field(pdf, label, lw=32, blank=90):
    pdf.set_x(L)
    pdf.set_font("Times", "B", 11)
    pdf.cell(lw, LH, label, ln=0)
    pdf.set_font("Times", "", 11)
    x = pdf.get_x()
    y = pdf.get_y() + LH - 1
    pdf.set_draw_color(80, 80, 80)
    pdf.set_line_width(0.3)
    pdf.line(x, y, x + blank, y)
    pdf.ln(LH + 2)


def build():
    pdf = CertPDF(format="A4")
    pdf.set_margins(L, TM, R)
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.add_page()

    # ── Title ──────────────────────────────────────────────────────────────
    pdf.set_y(TM)
    pdf.set_font("Times", "BU", 13)
    pdf.set_text_color(15, 40, 90)
    pdf.multi_cell(BODYW, 8,
                   "CERTIFICATE OF RADIOLOGICAL VERIFICATION\n"
                   "AND DATASET APPROVAL",
                   align="C")
    pdf.ln(4)

    # ── Opening ────────────────────────────────────────────────────────────
    para(pdf,
         "This is to certify that the radiological dataset review, cross-sectional "
         "landmark validation, pelvimetric data verification, and pelvic floor "
         "morphometric assessment for the Ph.D. thesis entitled:")

    # Thesis title indented, bold-italic
    pdf.set_font("Times", "BI", 11)
    pdf.set_text_color(15, 40, 90)
    pdf.set_x(L + 8)
    pdf.multi_cell(BODYW - 16, LH, THESIS, align="C")
    pdf.ln(2)

    para(pdf,
         "submitted by Mr. MUTHUKUMAR C, Research Scholar in Anatomy bearing "
         "USN No. 24PH14ANA01, Sri Siddhartha Academy of Higher Education (SSAHE), "
         "Tumakuru, has been thoroughly reviewed, verified, and counter-checked in "
         "the Department of Radiodiagnosis.")

    # ── Dataset description ────────────────────────────────────────────────
    para(pdf,
         "The candidate has compiled and processed a structured CT pelvimetry dataset "
         "comprising 350 female patient studies - 175 Hilly-region cohort and 175 "
         "Inland/Plain-region cohort - sourced from institutional PACS repositories "
         "across participating medical institutions in Kerala, Karnataka, and Tamil Nadu. "
         "All scans were originally acquired for non-gynaecological clinical indications "
         "and were retrospectively screened to satisfy the strict anatomical and demographic "
         "inclusion criteria of the study. Each dataset entry includes anonymised DICOM "
         "image volumes, mask-segmented bone and soft-tissue structures, and a corresponding "
         "CT Pelvimetry & Pelvic Floor Assessment Report (PDF).")

    # ── Certification items ────────────────────────────────────────────────
    para(pdf, "I hereby confirm and certify that:")

    item(pdf, "1", "Bony Pelvimetry Validation",
         "The following linear and angular morphometric parameters were measured at "
         "the correct predefined cross-sectional anatomical planes and have been "
         "verified for anatomical accuracy: True Conjugate Diameter, Obstetric "
         "Conjugate Diameter, Diagonal Conjugate Diameter, Transverse Diameter of "
         "the Pelvic Inlet, Interspinous Diameter (ISD), Intertuberous Diameter (ITD), "
         "AP Diameter of Mid-pelvis, Subpubic Angle (SPA), Sacral Length, Sacral "
         "Curvature, Pelvic Inclination, and Oblique Diameter. All values fall within "
         "clinically established reference ranges for the South Indian female population.")

    item(pdf, "2", "Pelvic Floor Soft-Tissue Assessment",
         "Thickness measurements and functional parameters for the levator ani muscle "
         "group (puborectalis, pubococcygeus, and iliococcygeus) and the coccygeus have "
         "been reviewed. Bilateral muscle thickness values, levator hiatus dimensions "
         "(AP and lateral), hiatal area, resting thickness, Valsalva thickness, and "
         "contractility indices have been cross-checked for anatomical plausibility and "
         "regional consistency against published normative data.")

    item(pdf, "3", "Cohort Stratification & PFD Grading",
         "Each of the 350 study subjects has been correctly categorised by PFD pattern "
         "(Descent/Plain-type vs. Bulge/Hilly-type) and graded by severity (Grade 1-3) "
         "for the following pelvic floor disorder diagnoses: uterine prolapse, cystocele, "
         "rectocele, vault prolapse, and combined PFD. The cohort distribution and clinical "
         "grading have been verified as appropriate for comparative cross-sectional analysis.")

    item(pdf, "4", "Segmentation & Landmark Accuracy",
         "The automated bone and soft-tissue boundary extractions (hip bones, sacrum, "
         "pelvic floor musculature) were spot-checked against source DICOM slices across "
         "a random sample of cases from both cohorts. Mask bounding-box coordinates, "
         "voxel spacing calibrations, and measurement derivations have been counter-verified "
         "for technical correctness and reproducibility.")

    item(pdf, "5", "Report Completeness & Clinical Consistency",
         "Each patient record is accompanied by a structured two-page CT Pelvimetry & "
         "Pelvic Floor Assessment Report containing: patient demographics, anthropometric "
         "measurements, obstetric history, pelvic floor symptom profile, imaging parameters, "
         "full pelvimetry table with normal references, pelvic floor muscle measurements, "
         "levator hiatus dimensions, and a radiological impression. All 350 reports have "
         "been reviewed and found complete and internally consistent.")

    # ── Closing statement ──────────────────────────────────────────────────
    pdf.ln(1)
    para(pdf,
         "The dataset, pelvimetric measurements, soft-tissue morphometric values, and "
         "associated clinical reports compiled from the Hilly and Inland study cohorts "
         "have been comprehensively audited and are validated as anatomically accurate "
         "and radiologically reliable for advanced statistical comparative analysis in "
         "fulfilment of the above-mentioned Ph.D. thesis.")

    # ── Signature block ────────────────────────────────────────────────────
    pdf.ln(2)
    field(pdf, "Date:",  lw=24, blank=95)
    field(pdf, "Place:", lw=24, blank=95)
    pdf.ln(3)

    pdf.set_font("Times", "B", 11)
    pdf.set_text_color(20, 20, 20)
    pdf.set_x(L)
    pdf.cell(0, LH, "Verified and Approved by:", ln=1)
    pdf.ln(12)

    # Signature underline
    sy = pdf.get_y()
    pdf.set_draw_color(60, 60, 60)
    pdf.set_line_width(0.4)
    pdf.line(L, sy, L + 80, sy)
    pdf.ln(2)
    pdf.set_font("Times", "B", 10)
    pdf.set_x(L)
    pdf.cell(0, LH, "Signature", ln=1)
    pdf.ln(3)

    field(pdf, "Name:",           lw=36, blank=100)
    pdf.set_font("Times", "", 11)
    pdf.set_x(L)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, LH, "MD (Radiodiagnosis)", ln=1)
    pdf.ln(1)
    field(pdf, "Designation:",    lw=42, blank=90)
    field(pdf, "Department:",     lw=42, blank=90)
    field(pdf, "Institution:",    lw=42, blank=90)
    field(pdf, "Reg. No.:",       lw=42, blank=80)

    # Seal box (bottom-right)
    pdf.ln(2)
    seal_x = 210 - R - 54
    seal_y = pdf.get_y()
    pdf.set_draw_color(120, 120, 120)
    pdf.set_line_width(0.4)
    pdf.rect(seal_x, seal_y, 52, 30)
    pdf.set_font("Times", "I", 8.5)
    pdf.set_text_color(140, 140, 140)
    pdf.set_xy(seal_x, seal_y + 12)
    pdf.cell(52, 6, "Official Seal / Stamp", align="C")

    pdf.output(str(OUT))
    kb = OUT.stat().st_size // 1024
    print("Certificate saved: %s  (%d KB)" % (OUT, kb))


build()
