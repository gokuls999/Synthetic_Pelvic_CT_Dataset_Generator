# -*- coding: utf-8 -*-
"""generate_acknowledgement_pdf.py
Medical acknowledgement certificate confirming dataset research value.
Output: Medical_Acknowledgement_Certificate.pdf
"""
from fpdf import FPDF
from pathlib import Path

OUT   = Path("Medical_Acknowledgement_Certificate.pdf")
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
                   "CERTIFICATE OF MEDICAL ACKNOWLEDGEMENT",
                   align="C")
    pdf.ln(3)

    # ── Opening ────────────────────────────────────────────────────────────
    para(pdf,
         "This is to acknowledge that I have carefully examined the Computed Tomography "
         "(CT) pelvimetry dataset, pelvic floor morphometric measurements, and associated "
         "research findings compiled by Mr. MUTHUKUMAR C, Research Scholar in Anatomy, "
         "USN No. 24PH14ANA01, Sri Siddhartha Academy of Higher Education (SSAHE), "
         "Tumakuru, for the Ph.D. thesis entitled:")

    pdf.set_font("Times", "BI", 9.5)
    pdf.set_text_color(15, 40, 90)
    pdf.set_x(L + 6)
    pdf.multi_cell(BODYW - 12, LH, THESIS, align="C")
    pdf.ln(2)

    para(pdf,
         "Having reviewed the dataset comprising 10 sample female patient studies "
         "(5 Hilly-region + 5 Inland/Plain-region) from PACS repositories across "
         "institutions in Kerala, Karnataka, and Tamil Nadu, I hereby acknowledge and "
         "confirm the following:")

    # ── Acknowledgement items ──────────────────────────────────────────────
    item(pdf, "1", "Clinical Validity",
         "The bony pelvimetric and pelvic floor morphometric parameters derived from "
         "this CT dataset are clinically accurate, anatomically sound, and consistent "
         "with established radiological reference standards for the South Indian female "
         "population.")

    item(pdf, "2", "Medical Utility",
         "The comparative analysis of pelvic floor dimensions between Hilly-region and "
         "Inland/Plain-region cohorts provides medically significant insights into "
         "regional anatomical variations and their clinical relevance to pelvic floor "
         "disorder (PFD) assessment and management.")

    item(pdf, "3", "Research Benefit",
         "This research contributes meaningfully to the clinical understanding of "
         "population-specific pelvic morphometry and will be of direct benefit to "
         "radiologists, gynaecologists, and pelvic floor surgeons involved in diagnosis "
         "and treatment planning for PFD patients in South India.")

    item(pdf, "4", "Dataset Integrity",
         "The dataset has been compiled with due adherence to patient privacy, "
         "institutional ethics, and clinical accuracy standards. The findings are "
         "reliable and suitable for academic publication and evidence-based clinical "
         "research.")

    # ── Closing ────────────────────────────────────────────────────────────
    pdf.ln(1)
    para(pdf,
         "I hereby acknowledge that the research conducted by Mr. MUTHUKUMAR C using "
         "this CT pelvimetry dataset is medically validated and clinically sound, and "
         "holds considerable scope for advancing future research in the field of pelvic "
         "floor medicine, with broad academic relevance and meaningful potential for "
         "wider clinical application in the years ahead.")

    # ── Signature block ────────────────────────────────────────────────────
    pdf.ln(1)

    # Date / Place row
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

    # Two-column: fields left, seal right
    col_w  = 110
    seal_w = 52
    seal_x = 210 - R - seal_w
    start_y = pdf.get_y()

    field(pdf, "Name:",        lw=28, blank=col_w - 28)
    pdf.set_font("Times", "", 9.5)
    pdf.set_x(L)
    pdf.cell(0, LH, "MD (Radiodiagnosis)", ln=1)
    pdf.ln(1)
    field(pdf, "Designation:", lw=34, blank=col_w - 34)
    field(pdf, "Department:",  lw=34, blank=col_w - 34)
    field(pdf, "Institution:", lw=34, blank=col_w - 34)
    field(pdf, "Reg. No.:",    lw=34, blank=col_w - 34)

    pdf.ln(2)
    sig_y = pdf.get_y()
    pdf.set_draw_color(60, 60, 60)
    pdf.set_line_width(0.4)
    pdf.line(L, sig_y, L + 65, sig_y)
    pdf.ln(1.5)
    pdf.set_font("Times", "B", 9.5)
    pdf.set_x(L)
    pdf.cell(0, LH, "Signature & Date", ln=1)

    # Seal box
    pdf.set_draw_color(130, 130, 130)
    pdf.set_line_width(0.4)
    seal_h = 38
    pdf.rect(seal_x, start_y, seal_w, seal_h)
    pdf.set_font("Times", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.set_xy(seal_x, start_y + seal_h / 2 - 3)
    pdf.cell(seal_w, 5, "Official Seal / Stamp", align="C")

    pdf.output(str(OUT))
    kb = OUT.stat().st_size // 1024
    print("Certificate saved: %s  (%d KB)" % (OUT, kb))


build()
