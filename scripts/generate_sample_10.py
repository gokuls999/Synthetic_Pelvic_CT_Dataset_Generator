# -*- coding: utf-8 -*-
"""generate_sample_10.py
Generates 10 enhanced patient reports (5 Plain + 5 Hilly) into a new
PFD_Sample_10_Patients/ folder. Incorporates all master data collection
sheet fields (Excel). No nurse/paramedic credentials. No institution names.
Only Reporting Radiologist in signature block.
"""
import sys, json, random, math, shutil
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import sys as _sys
_sys.path.insert(0, "scripts")
from _common import add_repo_to_path, load_config
add_repo_to_path()

import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from fpdf import FPDF

cfg        = load_config("configs/default.yaml")
cache_root = Path(cfg["paths"]["cache_dir"])
SRC_ROOT   = Path("synthetic_dataset/PFD_Real_Dataset_350")
OUT_ROOT   = Path("D:/PFD_Sample_10_Patients")

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY   = (14, 42, 71)
BLUE   = (30, 90, 160)
LBLUE  = (210, 230, 250)
WHITE  = (255, 255, 255)
LGRAY  = (245, 245, 245)
DKGRAY = (80, 80, 80)
BLACK  = (0, 0, 0)

# ── Occupation list — no Nurse / paramedic ───────────────────────────────────
OCCUPATIONS = ["Homemaker", "Agricultural Worker", "Tailor", "Teacher",
               "Domestic Worker", "Vendor", "Daily Wage Worker", "Shopkeeper"]
AREAS       = ["Rural", "Semi-Urban", "Urban"]
EDUCATION   = ["Primary School", "Middle School", "Secondary School",
               "Higher Secondary", "Graduate", "Illiterate"]
SOCIO       = ["Lower", "Lower-Middle", "Middle", "Upper-Middle", "Upper"]
CT_MACHINES = ["Philips Brilliance 64", "Siemens SOMATOM Definition",
               "GE LightSpeed VCT", "Toshiba Aquilion 64", "Philips iCT 256"]

PATTERN_LABELS = {
    "combined_pfd":    "Combined PFD",
    "cystocele":       "Cystocele",
    "rectocele":       "Rectocele",
    "uterine_prolapse":"Uterine Prolapse",
}
SHAPE_LABEL = {"plain": "Gynecoid", "hilly": "Android"}


# ── Stats loader ──────────────────────────────────────────────────────────────
def load_stats_and_spacing(source_dataset, source_uid):
    stats_path = cache_root / "masks" / source_dataset / source_uid / "stats.json"
    npz_path   = cache_root / source_dataset / f"{source_uid}.npz"
    if not stats_path.exists() or not npz_path.exists():
        return None, None
    stats = json.loads(stats_path.read_text())
    with np.load(npz_path, allow_pickle=True) as nf:
        spacing = [float(x) for x in nf["spacing"]]
    return stats, spacing


# ── Pelvimetry ────────────────────────────────────────────────────────────────
def compute_pelvimetry(stats, spacing, population, seed):
    rng = random.Random(seed + 7919)
    sz, sy, sx = spacing

    def noise(x, pct=0.02):
        return round(x * (1 + rng.uniform(-pct, pct)), 1)

    sac   = stats.get("sacrum")
    hip_l = stats.get("hip_left")
    hip_r = stats.get("hip_right")
    m     = {}

    if hip_l and hip_r:
        outer_x_cm = (hip_r["bbox_max"][2] - hip_l["bbox_min"][2]) * sx / 10
        if 12.5 <= outer_x_cm <= 17.5:
            t_inlet = outer_x_cm * 0.780
        else:
            t_inlet = rng.uniform(11.5, 13.5) if population == "plain" else rng.uniform(10.5, 12.5)
        t_mid = t_inlet * (0.83 if population == "plain" else 0.82)
        t_out = t_inlet * (1.03 if population == "plain" else 0.90)
        m["transverse_inlet"] = noise(t_inlet)
        m["interspinous"]     = noise(t_mid)
        m["intertuberous"]    = noise(t_out)

        # Inter-pubic ramus (IPR) — from Excel sheet field
        ipr_base = t_inlet * 0.41
        m["ipr_right"] = noise(ipr_base + rng.uniform(-0.3, 0.3))
        m["ipr_left"]  = noise(ipr_base + rng.uniform(-0.3, 0.3))

        # Sacrotuberous ligament (STL) — from Excel sheet field
        stl_base = rng.uniform(5.8, 7.4) if population == "plain" else rng.uniform(5.2, 6.8)
        m["stl_right"] = noise(stl_base + rng.uniform(-0.2, 0.2))
        m["stl_left"]  = noise(stl_base + rng.uniform(-0.2, 0.2))

    if "transverse_inlet" in m:
        T  = m["transverse_inlet"]
        ap = T * (0.88 if population == "plain" else 0.78)
        m["true_conjugate"]      = noise(ap)
        m["obstetric_conjugate"] = noise(ap - rng.uniform(0.3, 0.7))
        m["diagonal_conjugate"]  = noise(ap + rng.uniform(1.3, 1.7))
        m["ap_midpelvis"]        = noise(ap * (0.93 if population == "plain" else 0.89))

        # AP outlet
        m["ap_outlet"] = noise(ap * (0.87 if population == "plain" else 0.82))

    if sac:
        z_ext_mm  = (sac["bbox_max"][0] - sac["bbox_min"][0]) * sz
        sacral_raw = z_ext_mm * 0.67 / 10
        sacral_len = sacral_raw if 9.5 <= sacral_raw <= 13.5 else (
            rng.uniform(10.2, 12.2) if population == "plain" else rng.uniform(9.8, 11.5))
        m["sacral_length"]    = noise(sacral_len)
        m["sacral_curvature"] = noise(sac["extent"][1] * sy / 10 * 0.30)

    m["subpubic_angle"] = round(rng.uniform(90, 112) if population == "plain"
                                else rng.uniform(68, 86))

    if sac and hip_l and hip_r:
        z_range = (max(sac["bbox_max"][0], hip_l["bbox_max"][0], hip_r["bbox_max"][0])
                   - min(sac["bbox_min"][0], hip_l["bbox_min"][0], hip_r["bbox_min"][0]))
        m["pelvic_depth"] = noise(z_range * sz / 10 * 0.50)
        m["pelvic_width"] = m.get("transverse_inlet", 13.0)

    m["pelvic_inclination"] = round(rng.uniform(52, 62))

    if "transverse_inlet" in m and "true_conjugate" in m:
        oblique = math.sqrt(m["transverse_inlet"]**2 + m["true_conjugate"]**2) / math.sqrt(2)
        m["oblique_diameter"] = noise(oblique)

    # Perineal area from Excel sheet field
    if "intertuberous" in m and "ap_outlet" in m:
        m["perineal_area"] = round(m["intertuberous"] * m["ap_outlet"] * 0.5 * math.pi / 4, 1)

    return m


# ── Muscle params ─────────────────────────────────────────────────────────────
def compute_muscle_params(stats, spacing, population, grade, seed):
    rng = random.Random(seed + 3307)

    def noise(x, pct=0.05):
        return round(x * (1 + rng.uniform(-pct, pct)), 1)

    grade_factor = {1: 0.98, 2: 0.91, 3: 0.83}[grade]
    pop_factor   = 1.0 if population == "plain" else 1.05

    la_base = rng.uniform(6.8, 8.5) * pop_factor * grade_factor
    la_r, la_l = noise(la_base + rng.uniform(-0.4, 0.4)), noise(la_base + rng.uniform(-0.4, 0.4))
    la_mean = round((la_r + la_l) / 2, 1)

    pr_base = rng.uniform(5.5, 7.5) * pop_factor * grade_factor
    pr_r, pr_l = noise(pr_base + rng.uniform(-0.3, 0.3)), noise(pr_base + rng.uniform(-0.3, 0.3))
    pr_mean = round((pr_r + pr_l) / 2, 1)

    ic_base = rng.uniform(3.5, 5.5) * grade_factor
    ic_r, ic_l = noise(ic_base + rng.uniform(-0.2, 0.2)), noise(ic_base + rng.uniform(-0.2, 0.2))
    ic_mean = round((ic_r + ic_l) / 2, 1)

    pc_base = rng.uniform(3.2, 4.8) * grade_factor
    pc_r, pc_l = noise(pc_base + rng.uniform(-0.2, 0.2)), noise(pc_base + rng.uniform(-0.2, 0.2))
    pc_mean = round((pc_r + pc_l) / 2, 1)

    hiatus_ap  = noise(3.8 + grade * 0.35 * pop_factor, 0.04)
    hiatus_lat = noise(5.2 + grade * 0.45 * pop_factor, 0.04)
    hiatus_area = round(math.pi * hiatus_ap * hiatus_lat / 4, 1)

    rest_thick     = noise(la_base * 0.9, 0.03)
    contract_thick = noise(la_base * 1.12, 0.03)
    valsalva_thick = noise(la_base * 0.78, 0.03)

    symmetry     = "Symmetric" if abs(la_r - la_l) < 0.6 else "Asymmetric"
    defect       = "Present" if grade == 3 and rng.random() > 0.5 else "Absent"
    contractility = "Reduced" if grade == 3 else ("Mildly Reduced" if grade == 2 else "Normal")

    return {
        "la_r": la_r, "la_l": la_l, "la_mean": la_mean,
        "pr_r": pr_r, "pr_l": pr_l, "pr_mean": pr_mean,
        "ic_r": ic_r, "ic_l": ic_l, "ic_mean": ic_mean,
        "pc_r": pc_r, "pc_l": pc_l, "pc_mean": pc_mean,
        "hiatus_ap": hiatus_ap, "hiatus_lat": hiatus_lat, "hiatus_area": hiatus_area,
        "rest_thick": rest_thick, "contract_thick": contract_thick,
        "valsalva_thick": valsalva_thick,
        "symmetry": symmetry, "defect": defect, "contractility": contractility,
    }


# ── Demographics ──────────────────────────────────────────────────────────────
def _yn(b): return "Yes" if b else "No"

def generate_demographics(seed, population, pattern, grade, patient_num):
    rng = random.Random(seed + 1337)
    age = int(rng.triangular(28, 68, 47))
    bmi = round(rng.uniform(18.5, 30.5), 1)
    height_cm = round(rng.uniform(148, 165), 1)
    weight_kg  = round(bmi * (height_cm / 100) ** 2, 1)
    waist      = round(rng.uniform(72, 98), 1)
    hip        = round(rng.uniform(88, 108), 1)

    gravida = rng.randint(1, 2 + grade)
    para    = gravida
    vag_del = rng.randint(0, para)
    cs      = para - vag_del
    birth_wt = round(rng.uniform(2.5, 4.2), 2)

    menarche   = rng.randint(11, 15)
    cycle_dur  = rng.randint(21, 35)
    men_status = "Regular" if age < 48 else rng.choice(["Regular", "Irregular", "Menopausal"])

    has_ui   = grade >= 1 or pattern in ("cystocele", "combined_pfd")
    has_si   = pattern in ("cystocele", "combined_pfd") and grade >= 1
    has_urge = grade >= 2 and rng.random() > 0.5
    has_fi   = pattern in ("rectocele", "combined_pfd") and grade >= 2
    has_pop  = grade >= 1
    has_pain = grade >= 2 and rng.random() > 0.4
    has_dysp = grade >= 2 and rng.random() > 0.6
    has_cons = pattern in ("rectocele", "combined_pfd")

    date_collected = datetime(2026, 1, 1) + timedelta(days=patient_num * 0.5)

    return {
        "age": age, "sex": "Female", "marital": "Married",
        "occupation": rng.choice(OCCUPATIONS),
        "area": rng.choice(AREAS),
        "education": rng.choice(EDUCATION),
        "socio": rng.choice(SOCIO),
        "height": height_cm, "weight": weight_kg, "bmi": bmi,
        "waist": waist, "hip": hip, "whr": round(waist / hip, 2),
        "menarche": menarche, "men_status": men_status,
        "cycle_dur": cycle_dur, "oc_use": rng.choice(["Yes", "No", "No"]),
        "gravida": gravida, "para": para, "vag_del": vag_del, "cs": cs,
        "prolonged_labour": rng.choice(["Yes", "No", "No"]),
        "birth_wt": birth_wt,
        "episiotomy": rng.choice(["Yes", "No", "No"]),
        "tear": rng.choice(["Yes", "No", "No"]),
        "time_since_del": rng.randint(2, 28),
        "activity": rng.choice(["Sedentary", "Moderate", "Moderate", "Active"]),
        "exercise":      rng.choice(["Yes", "No"]),
        "weight_lift":   rng.choice(["Yes", "No", "No"]),
        "smoking":       "Non-Smoker",
        "alcohol":       rng.choice(["Yes", "No", "No", "No"]),
        "occ_strain":    rng.choice(["Yes", "No"]),
        "ui": _yn(has_ui), "si": _yn(has_si), "urge": _yn(has_urge),
        "fi": _yn(has_fi), "pop": _yn(has_pop), "pain": _yn(has_pain),
        "dysp": _yn(has_dysp), "constip": _yn(has_cons),
        "machine":  rng.choice(CT_MACHINES),
        "kv":       rng.choice([100, 120, 140]),
        "mas":      rng.randint(150, 280),
        "bladder":  rng.choice(["Partially Filled", "Full", "Empty"]),
        "date":     date_collected.strftime("%Y-%m-%d"),
    }


# ── PDF class ─────────────────────────────────────────────────────────────────
class ReportPDF(FPDF):
    def __init__(self, report_no, date_str):
        super().__init__()
        self.report_no = report_no
        self.date_str  = date_str

    def header(self):
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 18, 'F')
        self.set_text_color(*WHITE)
        self.set_font("Arial", "B", 12)
        self.set_xy(8, 4)
        # Generic centre name — no specific institution
        self.cell(120, 6, "CT PELVIMETRY & PELVIC FLOOR RESEARCH UNIT", ln=0)
        self.set_font("Arial", "", 8)
        self.set_xy(140, 3)
        self.cell(60, 5, f"Report No:  {self.report_no}", ln=1, align="R")
        self.set_xy(140, 8)
        self.cell(60, 5, f"Date:  {self.date_str}", ln=1, align="R")
        self.set_fill_color(*BLUE)
        self.rect(0, 18, 210, 7, 'F')
        self.set_text_color(*WHITE)
        self.set_font("Arial", "B", 9)
        self.set_xy(0, 19)
        self.cell(210, 5,
                  "COMPUTED TOMOGRAPHY PELVIMETRY & PELVIC FLOOR ASSESSMENT REPORT",
                  align="C")
        self.set_text_color(*BLACK)
        self.set_xy(0, 27)

    def footer(self):
        self.set_y(-12)
        self.set_fill_color(*NAVY)
        self.rect(0, self.get_y(), 210, 12, 'F')
        self.set_text_color(*WHITE)
        self.set_font("Arial", "I", 7)
        self.set_xy(8, self.get_y() + 3)
        self.cell(100, 5, "CONFIDENTIAL - FOR CLINICAL RESEARCH USE ONLY")
        self.set_xy(140, self.get_y())
        self.cell(60, 5, f"Page {self.page_no()}", align="R")

    def section_header(self, title):
        self.ln(3)
        self.set_fill_color(*BLUE)
        self.set_text_color(*WHITE)
        self.set_font("Arial", "B", 9)
        self.cell(0, 6, f"  {title}", fill=True, ln=1)
        self.set_text_color(*BLACK)
        self.ln(1)

    def two_col_row(self, label, value, label2="", value2="", fill=False):
        bg = LGRAY if fill else WHITE
        self.set_fill_color(*bg)
        self.set_font("Arial", "", 8)
        self.set_text_color(*DKGRAY)
        self.cell(42, 5.5, label, fill=True)
        self.set_font("Arial", "B", 8)
        self.set_text_color(*BLACK)
        self.cell(48, 5.5, str(value), fill=True, border="R")
        self.set_font("Arial", "", 8)
        self.set_text_color(*DKGRAY)
        self.cell(42, 5.5, label2, fill=True)
        self.set_font("Arial", "B", 8)
        self.set_text_color(*BLACK)
        self.cell(0, 5.5, str(value2), fill=True, ln=1)

    def meas_row(self, label, value, normal_range, fill=False):
        bg = LGRAY if fill else WHITE
        self.set_fill_color(*bg)
        self.set_font("Arial", "", 8)
        self.set_text_color(*DKGRAY)
        self.cell(90, 5.5, f"  {label}", fill=True)
        self.set_font("Arial", "B", 8)
        self.set_text_color(*BLACK)
        self.cell(30, 5.5, str(value), fill=True, align="C")
        self.set_font("Arial", "I", 7.5)
        self.set_text_color(*DKGRAY)
        self.cell(0, 5.5, normal_range, fill=True, ln=1)
        self.set_text_color(*BLACK)

    def muscle_row(self, label, right, left, mean, fill=False):
        bg = LGRAY if fill else WHITE
        self.set_fill_color(*bg)
        self.set_font("Arial", "", 8)
        self.set_text_color(*DKGRAY)
        self.cell(70, 5.5, f"  {label}", fill=True)
        self.set_font("Arial", "B", 8)
        self.set_text_color(*BLACK)
        self.cell(30, 5.5, str(right), fill=True, align="C")
        self.cell(30, 5.5, str(left),  fill=True, align="C")
        self.cell(0,  5.5, str(mean),  fill=True, align="C", ln=1)


# ── Report builder ────────────────────────────────────────────────────────────
def build_report(patient_id, report_no, pop, pattern, grade, slices,
                 pelvimetry, muscles, demog):
    pdf = ReportPDF(report_no, demog["date"])
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=14)

    # Patient details
    pdf.section_header("PATIENT DETAILS")
    f = True
    pdf.two_col_row("Patient Reference No.", report_no, "Date of Study", demog["date"], fill=f); f=not f
    pdf.two_col_row("Age / Sex", f"{demog['age']} yrs / {demog['sex']}", "Marital Status", demog["marital"], fill=f); f=not f
    pdf.two_col_row("Occupation", demog["occupation"], "Residential Area", demog["area"], fill=f); f=not f
    pdf.two_col_row("Educational Status", demog["education"], "Socioeconomic Status", demog["socio"], fill=f)

    # Anthropometry
    pdf.section_header("ANTHROPOMETRIC MEASUREMENTS")
    f = True
    pdf.two_col_row("Height (cm)", demog["height"], "Weight (kg)", demog["weight"], fill=f); f=not f
    pdf.two_col_row("BMI (kg/m2)", demog["bmi"], "Waist Circumference (cm)", demog["waist"], fill=f); f=not f
    pdf.two_col_row("Hip Circumference (cm)", demog["hip"], "Waist-Hip Ratio", demog["whr"], fill=f)

    # Lifestyle factors (from Excel: Physical Activity, Exercise, Weight Lifting, Occupational Strain)
    pdf.section_header("LIFESTYLE FACTORS")
    f = True
    pdf.two_col_row("Physical Activity Level", demog["activity"], "Exercise Habit", demog["exercise"], fill=f); f=not f
    pdf.two_col_row("Weight Lifting History", demog["weight_lift"], "Occupational Strain", demog["occ_strain"], fill=f); f=not f
    pdf.two_col_row("Smoking Status", demog["smoking"], "Alcohol Intake", demog["alcohol"], fill=f)

    # Obstetric history
    pdf.section_header("OBSTETRIC HISTORY")
    f = True
    pdf.two_col_row("Gravida", demog["gravida"], "Para", demog["para"], fill=f); f=not f
    pdf.two_col_row("Vaginal Deliveries", demog["vag_del"], "Caesarean Sections", demog["cs"], fill=f); f=not f
    pdf.two_col_row("Prolonged Labour", demog["prolonged_labour"], "Largest Baby (kg)", demog["birth_wt"], fill=f); f=not f
    pdf.two_col_row("Episiotomy History", demog["episiotomy"], "Perineal Tear History", demog["tear"], fill=f); f=not f
    pdf.two_col_row("Time Since Delivery", f"{demog['time_since_del']} yr(s)", "", "", fill=f)

    # Menstrual & hormonal history
    pdf.section_header("MENSTRUAL & HORMONAL HISTORY")
    f = True
    pdf.two_col_row("Age at Menarche", demog["menarche"], "Menstrual Status", demog["men_status"], fill=f); f=not f
    pdf.two_col_row("Menstrual Cycle (days)", demog["cycle_dur"], "Oral Contraceptive Use", demog["oc_use"], fill=f)

    # Clinical symptoms
    pdf.section_header("CLINICAL SYMPTOMS")
    f = True
    pdf.two_col_row("Urinary Incontinence", demog["ui"],    "Stress Incontinence",  demog["si"],      fill=f); f=not f
    pdf.two_col_row("Urge Incontinence",    demog["urge"],  "Faecal Incontinence",  demog["fi"],      fill=f); f=not f
    pdf.two_col_row("Pelvic Organ Prolapse",demog["pop"],   "Chronic Pelvic Pain",  demog["pain"],    fill=f); f=not f
    pdf.two_col_row("Dyspareunia",          demog["dysp"],  "Constipation Symptoms",demog["constip"], fill=f)

    # Imaging details
    pdf.section_header("IMAGING DETAILS")
    f = True
    pdf.two_col_row("Modality", "CT", "CT Machine", demog["machine"], fill=f); f=not f
    pdf.two_col_row("kVp / mAs", f"{demog['kv']} / {demog['mas']}", "Position", "Supine", fill=f); f=not f
    pdf.two_col_row("No. of Slices", slices, "Bladder Status", demog["bladder"], fill=f); f=not f
    pdf.two_col_row("IV Contrast", "Not Administered", "Image Quality", "Adequate", fill=f); f=not f
    pdf.two_col_row("PFD Pattern", PATTERN_LABELS[pattern], "PFD Grade", grade, fill=f)

    # Pelvimetry measurements
    pdf.section_header("PELVIMETRY MEASUREMENTS")
    pdf.set_fill_color(*LBLUE)
    pdf.set_font("Arial", "B", 8)
    pdf.cell(90, 6, "  Parameter", fill=True)
    pdf.cell(30, 6, "Value (cm / deg)", fill=True, align="C")
    pdf.cell(0,  6, "Normal Reference", fill=True, ln=1)

    pel = pelvimetry
    rows_p = [
        ("True Conjugate Diameter (TCD)",        f"{pel.get('true_conjugate','-')} cm",       ">= 10.5 cm"),
        ("Obstetric Conjugate Diameter (OCD)",   f"{pel.get('obstetric_conjugate','-')} cm",  ">= 10.0 cm"),
        ("Diagonal Conjugate (DC)",              f"{pel.get('diagonal_conjugate','-')} cm",   ">= 11.5 cm"),
        ("Transverse Diameter of Inlet (TDI)",   f"{pel.get('transverse_inlet','-')} cm",     "12.5 - 13.5 cm"),
        ("Oblique Diameter (OD)",                f"{pel.get('oblique_diameter','-')} cm",     "12.0 - 13.0 cm"),
        ("Interspinous Diameter (ISD)",          f"{pel.get('interspinous','-')} cm",         ">= 10.0 cm"),
        ("AP Diameter of Midpelvis (APM)",       f"{pel.get('ap_midpelvis','-')} cm",         ">= 11.5 cm"),
        ("Intertuberous Diameter (ITD)",         f"{pel.get('intertuberous','-')} cm",        ">= 10.5 cm"),
        ("AP Diameter of Outlet (APO)",          f"{pel.get('ap_outlet','-')} cm",            ">= 9.5 cm"),
        ("Subpubic Angle (SPA)",                 f"{pel.get('subpubic_angle','-')} deg",      "85-115 deg (Gynecoid)"),
        ("Sacral Curvature (SC)",                f"{pel.get('sacral_curvature','-')} cm",     "3.0 - 5.0 cm"),
        ("Sacral Length (SL)",                   f"{pel.get('sacral_length','-')} cm",        "10.0 - 13.0 cm"),
        ("Pelvic Depth (PD)",                    f"{pel.get('pelvic_depth','-')} cm",         "11.0 - 13.5 cm"),
        ("Pelvic Width (PW)",                    f"{pel.get('pelvic_width','-')} cm",         "22.0 - 26.0 cm"),
        ("Pelvic Inclination (PI)",              f"{pel.get('pelvic_inclination','-')} deg",  "50 - 65 deg"),
        ("Right Inter-Pubic Ramus (Rt. IPR)",    f"{pel.get('ipr_right','-')} cm",            "4.5 - 6.5 cm"),
        ("Left Inter-Pubic Ramus (Lt. IPR)",     f"{pel.get('ipr_left','-')} cm",             "4.5 - 6.5 cm"),
        ("Right Sacrotuberous Ligament (Rt. STL)",f"{pel.get('stl_right','-')} cm",           "5.5 - 7.5 cm"),
        ("Left Sacrotuberous Ligament (Lt. STL)", f"{pel.get('stl_left','-')} cm",            "5.5 - 7.5 cm"),
        ("Perineal Area",                        f"{pel.get('perineal_area','-')} sq cm",     "28.0 - 38.0 sq cm"),
        ("Pelvic Shape Classification",          SHAPE_LABEL.get(pop, "-"),                  "Gynecoid / Android"),
    ]
    for i, (lbl, val, ref) in enumerate(rows_p):
        pdf.meas_row(lbl, val, ref, fill=(i % 2 == 0))

    # Muscle thickness
    pdf.section_header("PELVIC FLOOR MUSCLE MEASUREMENTS (CT Estimation)")
    pdf.set_fill_color(*LBLUE)
    pdf.set_font("Arial", "B", 8)
    pdf.cell(70, 6, "  Muscle", fill=True)
    pdf.cell(30, 6, "Right (mm)", fill=True, align="C")
    pdf.cell(30, 6, "Left (mm)",  fill=True, align="C")
    pdf.cell(0,  6, "Mean (mm)",  fill=True, align="C", ln=1)

    mu = muscles
    rows_m = [
        ("Levator Ani (LA)",   mu["la_r"], mu["la_l"], mu["la_mean"]),
        ("Puborectalis (PR)",  mu["pr_r"], mu["pr_l"], mu["pr_mean"]),
        ("Iliococcygeus (IC)", mu["ic_r"], mu["ic_l"], mu["ic_mean"]),
        ("Pubococcygeus (PC)", mu["pc_r"], mu["pc_l"], mu["pc_mean"]),
    ]
    for i, (lbl, r, l, mn) in enumerate(rows_m):
        pdf.muscle_row(lbl, r, l, mn, fill=(i % 2 == 0))

    pdf.ln(2)
    f = True
    pdf.section_header("MUSCLE FUNCTIONAL PARAMETERS & LEVATOR HIATUS")
    pdf.two_col_row("Muscle Symmetry",              mu["symmetry"],       "Muscle Defect",              mu["defect"],          fill=f); f=not f
    pdf.two_col_row("Hiatal Dimensions AP (cm)",    mu["hiatus_ap"],      "Hiatal Dimensions Lat (cm)", mu["hiatus_lat"],      fill=f); f=not f
    pdf.two_col_row("Levator Hiatus Area (cm2)",    mu["hiatus_area"],    "Muscle Contractility",       mu["contractility"],   fill=f); f=not f
    pdf.two_col_row("Resting Thickness (mm)",       mu["rest_thick"],     "Thickness-Contraction (mm)", mu["contract_thick"],  fill=f); f=not f
    pdf.two_col_row("Thickness-Valsalva (mm)",      mu["valsalva_thick"], "", "",                                               fill=f)

    # Perineal area (from Excel PERINEAL AREA sheet)
    pdf.section_header("PERINEAL AREA MEASUREMENTS")
    f = True
    pdf.two_col_row("Perineal Area (sq cm)", pel.get("perineal_area", "-"),
                    "Pelvic Shape", SHAPE_LABEL.get(pop, "-"), fill=f)

    # Findings & impression
    pdf.section_header("FINDINGS & IMPRESSION")
    pdf.set_font("Arial", "", 8.5)
    grade_desc  = {1: "mild", 2: "moderate", 3: "severe"}[grade]
    pattern_txt = PATTERN_LABELS[pattern].lower()
    shape_txt   = SHAPE_LABEL[pop].lower()
    oc = pel.get("obstetric_conjugate", "-")
    sp = pel.get("interspinous", "-")
    it = pel.get("intertuberous", "-")
    findings = (
        f"CT pelvimetry of the female pelvis was performed in the supine position "
        f"without intravenous contrast. The bony pelvis demonstrates {shape_txt} morphology. "
        f"The obstetric conjugate measures {oc} cm and the interspinous diameter measures "
        f"{sp} cm. The intertuberous diameter at the pelvic outlet measures {it} cm.\n\n"
        f"Pelvic floor assessment demonstrates {grade_desc} {pattern_txt} with "
        f"{'asymmetric' if mu['symmetry'] == 'Asymmetric' else 'symmetric'} levator ani "
        f"complex. The levator hiatus area is {mu['hiatus_area']} cm2 "
        f"({'enlarged' if mu['hiatus_area'] > 25 else 'within expected limits'} for grade {grade} dysfunction). "
        f"Muscle {'defect noted' if mu['defect'] == 'Present' else 'defect not identified'}. "
        f"Contractility is {mu['contractility'].lower()}.\n\n"
        f"IMPRESSION:\n"
        f"  1. {shape_txt.capitalize()} pelvis. Obstetric conjugate "
        f"{'adequate' if float(str(oc).replace('-','0') or 0) >= 10.0 else 'borderline/reduced'}.\n"
        f"  2. Grade {grade} {PATTERN_LABELS[pattern]} - {grade_desc} pelvic floor dysfunction.\n"
        f"  3. Levator ani complex: {mu['symmetry'].lower()}, hiatal area {mu['hiatus_area']} cm2.\n"
        f"  4. {'Muscular defect identified. Clinical correlation advised.' if mu['defect'] == 'Present' else 'No significant muscular defect identified.'}"
    )
    pdf.set_fill_color(*LGRAY)
    pdf.multi_cell(0, 5.5, findings, fill=True)

    # Signature block — Reporting Radiologist only
    pdf.ln(6)
    pdf.set_draw_color(*NAVY)
    x0 = pdf.get_x() + 10
    y0 = pdf.get_y() + 14
    pdf.line(x0, y0, x0 + 65, y0)
    pdf.set_xy(x0, y0 + 1)
    pdf.set_font("Arial", "B", 8)
    pdf.cell(65, 5, "Reporting Radiologist")

    return pdf


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    plain_dirs = sorted((SRC_ROOT / "Plain_175").iterdir())[:5]
    hilly_dirs = sorted((SRC_ROOT / "Hilly_175").iterdir())[:5]
    patients   = [(d, "plain") for d in plain_dirs] + [(d, "hilly") for d in hilly_dirs]

    rng_order = random.Random(42)
    nums = list(range(1, 11))
    rng_order.shuffle(nums)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "Plain").mkdir(exist_ok=True)
    (OUT_ROOT / "Hilly").mkdir(exist_ok=True)

    print(f"Generating 10 patient reports into {OUT_ROOT}/")

    for idx, ((src_dir, pop), num) in enumerate(zip(patients, nums)):
        meta = json.loads((src_dir / "metadata_real.json").read_text())
        pattern = meta["pfd_pattern"]
        grade   = meta["pfd_grade"]
        slices  = meta["slices"]
        pid     = meta["patient_id"]
        report_no = f"CT-2026-{num:04d}"

        subfolder = "Plain" if pop == "plain" else "Hilly"
        out_dir   = OUT_ROOT / subfolder / src_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy all source subfolders and files (DICOM, JPG, PNG, COMPARISON.png, metadata)
        for item in src_dir.iterdir():
            dst_item = out_dir / item.name
            if item.is_dir():
                if not dst_item.exists():
                    shutil.copytree(str(item), str(dst_item))
            else:
                shutil.copy2(str(item), str(dst_item))

        # Load stats
        stats, spacing = load_stats_and_spacing(meta["source_dataset"], meta["source_uid"])
        if stats is None or spacing is None:
            print(f"  WARN: no stats for {pid}, skipping report")
            continue

        seed = num * 13 + idx * 7
        pelv  = compute_pelvimetry(stats, spacing, pop, seed)
        musc  = compute_muscle_params(stats, spacing, pop, grade, seed)
        demog = generate_demographics(seed, pop, pattern, grade, num)

        pdf = build_report(pid, report_no, pop, pattern, grade, slices, pelv, musc, demog)
        out_pdf = out_dir / "patient_report.pdf"
        pdf.output(str(out_pdf))

        kb = out_pdf.stat().st_size // 1024
        print(f"  [{idx+1}/10] {report_no}  {pid}  ({pop} {pattern} G{grade})  {kb} KB")

    total = sum(f.stat().st_size for f in OUT_ROOT.rglob("*") if f.is_file()) // (1024*1024)
    print(f"\nDone. {OUT_ROOT}/  ({total} MB total)")


main()
