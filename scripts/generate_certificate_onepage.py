# -*- coding: utf-8 -*-
"""generate_certificate_onepage.py
One-page condensed radiologist verification certificate.
Output: Radiologist_Certificate_OnePage.pdf
"""
from fpdf import FPDF
from pathlib import Path

OUT   = Path("Radiologist_Certificate_OnePage.pdf")
L, R  = 20, 20
TM    = 20
BODYW = 210 - L - R
LH    = 5.4

THESIS = (
    '"A COMPUTED TOMOGRAPHY EVALUATION OF PELVIC FLOOR THICKNESS AND BONY DIMENSIONS '
    "AMONG WOMEN FROM HILLY AND INLAND REGIONS OF SOUTH INDIA: "
    'A COMPARATIVE CROSS-SECTIONAL STUDY"'
)


class CertPDF(FPDF):
    def header(self):
        self.set_line_width(0.7)
        self.set_draw_color(30, 64, 120)
        self.line(L, 12, 210 - R, 12)

    def footer(self):
        self.set_y(-14)
        self.set_line_width(0.4)
        self.set_draw_color(30, 64, 120)
        self.line(L, self.get_y(), 210 - R, self.get_y())
        self.ln(1.5)
        self.set_font("Times", "I", 7.5)
        self.set_text_color(120, 120, 120)
        self.cell(0, 4,
                  "Confidential - For Academic & Institutional Review Only  |  "
                  "Dept. of Radiodiagnosis & Imaging",
                  align="C")


def para(pdf, text, indent=0, style="", size=9.5, color=(20, 20, 20), lh=None):
    pdf.set_font("Times", style, size)
    pdf.set_text_color(*color)
    pdf.set_x(L + indent)
    pdf.multi_cell(BODYW - indent, lh or LH, text, align="J")
    pdf.ln(1)


def item(pdf, num, heading, body, size=9.5):
    pdf.set_x(L + 3)
    pdf.set_font("Times", "B", size)
    pdf.set_text_color(15, 40, 90)
    tag = "%s. %s: " % (num, heading)
    tw  = pdf.get_string_width(tag) + 0.5
    pdf.cell(tw, LH, tag, ln=0)
    pdf.set_font("Times", "", size)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(BODYW - 3 - tw, LH, body, align="J")
    pdf.ln(0.5)


def field(pdf, label, lw=30, blank=85):
    pdf.set_x(L)
    pdf.set_font("Times", "B", 9.5)
    pdf.cell(lw, LH, label, ln=0)
    x = pdf.get_x()
    y = pdf.get_y() + LH - 1
    pdf.set_draw_color(80, 80, 80)
    pdf.set_line_width(0.3)
    pdf.line(x, y, x + blank, y)
    pdf.ln(LH + 1)


def build():
    pdf = CertPDF(format="A4")
    pdf.set_margins(L, TM, R)
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    # ── Title ──────────────────────────────────────────────────────────────
    pdf.set_y(TM)
    pdf.set_font("Times", "BU", 12)
    pdf.set_text_color(15, 40, 90)
    pdf.multi_cell(BODYW, 7,
                   "CERTIFICATE OF RADIOLOGICAL VERIFICATION AND DATASET APPROVAL",
                   align="C")
    pdf.ln(2)

    # ── Opening sentence ───────────────────────────────────────────────────
    para(pdf,
         "This is to certify that the radiological dataset review, pelvimetric data "
         "verification, and pelvic floor morphometric assessment for the Ph.D. thesis:")

    pdf.set_font("Times", "BI", 9.5)
    pdf.set_text_color(15, 40, 90)
    pdf.set_x(L + 6)
    pdf.multi_cell(BODYW - 12, LH, THESIS, align="C")
    pdf.ln(1)

    para(pdf,
         "submitted by Mr. MUTHUKUMAR C, Research Scholar in Anatomy, USN No. 24PH14ANA01, "
         "Sri Siddhartha Academy of Higher Education (SSAHE), Tumakuru, has been thoroughly "
         "reviewed and counter-checked in the Department of Radiodiagnosis.")

    # ── Dataset summary ────────────────────────────────────────────────────
    para(pdf,
         "The candidate has compiled a structured CT pelvimetry dataset of 10 female patient "
         "studies (5 Hilly-region + 5 Inland/Plain-region) sourced from PACS repositories "
         "across institutions in Kerala, Karnataka, and Tamil Nadu. Each entry includes "
         "anonymised DICOM volumes, segmented structures, and a CT Pelvimetry & Pelvic Floor "
         "Assessment Report (PDF).")

    # ── Certification items ────────────────────────────────────────────────
    para(pdf, "I hereby confirm and certify that:")

    item(pdf, "1", "Bony Pelvimetry Validation",
         "All linear and angular morphometric parameters (True/Obstetric/Diagonal Conjugate, "
         "Transverse Inlet, ISD, ITD, Subpubic Angle, Sacral Length & Curvature, Pelvic "
         "Inclination, Oblique Diameter, AP Mid-pelvis) were measured at correct anatomical "
         "planes and fall within clinically established reference ranges for the South Indian "
         "female population.")

    item(pdf, "2", "Pelvic Floor Soft-Tissue Assessment",
         "Bilateral thickness values for puborectalis, pubococcygeus, iliococcygeus, and "
         "coccygeus; levator hiatus AP/lateral dimensions; hiatal area; resting, Valsalva, "
         "and contraction thickness indices have been reviewed and verified for plausibility "
         "against published normative data.")

    item(pdf, "3", "Anatomical Edge Validation",
         "The verification of soft-tissue boundary extraction for the pelvic floor "
         "musculature (specifically the puborectalis, pubococcygeus, iliococcygeus, and "
         "coccygeus muscles) was monitored to ensure the clean exclusion of surrounding "
         "adipose tissue components.")

    item(pdf, "4", "Methodological Rigor",
         "The linear and angular morphometric parameters - including the Interspinous "
         "Diameter (ISD), Intertuberous Diameter (ITD), Subpubic Angle (SPA), "
         "Anteroposterior pelvic outlet diameter, and the Area of the Perineum - were "
         "measured at the correct predefined cross-sectional anatomical planes.")

    item(pdf, "5", "Software Verification",
         "The 3D reconstructions, point-capture calibrations, and contour-trace workflows "
         "executed within the MeViSlab medical image-processing platform were "
         "counter-checked for technical accuracy and consistency.")

    # ── Closing ────────────────────────────────────────────────────────────
    pdf.ln(1)
    para(pdf,
         "The dataset, pelvimetric values, and morphometric reports compiled from both cohorts "
         "are validated as anatomically accurate and radiologically reliable for advanced "
         "statistical comparative analysis in fulfilment of the above-mentioned Ph.D. thesis.")

    # ── Signature block ────────────────────────────────────────────────────
    pdf.ln(2)

    # Date / Place on same row to save space
    pdf.set_x(L)
    pdf.set_font("Times", "B", 9.5)
    pdf.cell(20, LH, "Date:", ln=0)
    pdf.set_draw_color(80, 80, 80)
    pdf.set_line_width(0.3)
    dx = pdf.get_x()
    dy = pdf.get_y() + LH - 1
    pdf.line(dx, dy, dx + 55, dy)
    pdf.set_x(dx + 58)
    pdf.cell(22, LH, "Place:", ln=0)
    px = pdf.get_x()
    pdf.line(px, dy, px + 55, dy)
    pdf.ln(LH + 2)

    # Two-column layout: signature fields left, seal box right
    col_w = 110
    seal_w = 52
    seal_x = 210 - R - seal_w
    start_y = pdf.get_y()

    # Left column fields
    field(pdf, "Name:",        lw=28, blank=col_w - 28)
    pdf.set_font("Times", "", 9.5)
    pdf.set_x(L)
    pdf.cell(0, LH, "MD (Radiodiagnosis)", ln=1)
    pdf.ln(1)
    field(pdf, "Designation:", lw=34, blank=col_w - 34)
    field(pdf, "Department:",  lw=34, blank=col_w - 34)
    field(pdf, "Institution:", lw=34, blank=col_w - 34)
    field(pdf, "Reg. No.:",    lw=34, blank=col_w - 34)

    # Signature line
    pdf.ln(2)
    sig_y = pdf.get_y()
    pdf.set_draw_color(60, 60, 60)
    pdf.set_line_width(0.4)
    pdf.line(L, sig_y, L + 65, sig_y)
    pdf.ln(1.5)
    pdf.set_font("Times", "B", 9.5)
    pdf.set_x(L)
    pdf.cell(0, LH, "Signature & Date", ln=1)

    # Seal box (right side, aligned with start_y)
    pdf.set_draw_color(130, 130, 130)
    pdf.set_line_width(0.4)
    seal_h = 36
    pdf.rect(seal_x, start_y, seal_w, seal_h)
    pdf.set_font("Times", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.set_xy(seal_x, start_y + seal_h / 2 - 3)
    pdf.cell(seal_w, 5, "Official Seal / Stamp", align="C")

    pdf.output(str(OUT))
    kb = OUT.stat().st_size // 1024
    print("Certificate saved: %s  (%d KB)" % (OUT, kb))


build()
