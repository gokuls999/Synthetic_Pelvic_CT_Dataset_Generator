"""Synthetic clinical metadata generator + PDF data collection sheet writer.

Generates plausible clinical fields (demographics, obstetric history,
pelvimetry, pelvic floor muscle measurements) for each synthetic patient
and writes both an enriched JSON and a formatted PDF report that mirrors
the Master Data Collection Sheet template.

All values are synthetically generated from seeded RNG using ranges drawn
from South Indian female pelvic clinical literature.  No real patient data
is used or implied.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Clinical plausibility tables
# ---------------------------------------------------------------------------

_OCCUPATIONS = [
    "Homemaker", "Agricultural Worker", "Teacher", "Nurse",
    "Shop Owner", "Tailor", "Daily Wage Labourer", "Office Staff",
    "Vegetable Vendor", "Anganwadi Worker",
]

_EDUCATION = [
    "Illiterate", "Primary School", "Middle School",
    "High School", "Higher Secondary", "Graduate", "Post Graduate",
]

_SES = ["Lower", "Lower Middle", "Middle", "Upper Middle", "Upper"]

_MARITAL = ["Married", "Married", "Married", "Unmarried", "Widowed", "Divorced"]

_ACTIVITY = ["Sedentary", "Moderate", "Heavy"]

_PELVIC_SHAPE = ["Gynecoid", "Android", "Anthropoid", "Platypelloid"]
_PELVIC_SHAPE_WEIGHTS = [0.55, 0.20, 0.18, 0.07]

_MACHINE_MODELS = [
    "Siemens SOMATOM Definition AS+",
    "GE LightSpeed VCT",
    "Philips Brilliance 64",
    "Toshiba Aquilion 64",
]

# PFD pattern → primary clinical symptoms (present = True)
_PFD_SYMPTOM_MAP: dict[str, dict[str, bool]] = {
    "combined_pfd": {
        "urinary_incontinence":        True,
        "stress_incontinence":         True,
        "urge_incontinence":           False,
        "fecal_incontinence":          True,
        "pelvic_organ_prolapse":       True,
        "chronic_pelvic_pain":         False,
        "dyspareunia":                 False,
        "constipation":                True,
    },
    "cystocele": {
        "urinary_incontinence":        True,
        "stress_incontinence":         True,
        "urge_incontinence":           True,
        "fecal_incontinence":          False,
        "pelvic_organ_prolapse":       True,
        "chronic_pelvic_pain":         False,
        "dyspareunia":                 False,
        "constipation":                False,
    },
    "rectocele": {
        "urinary_incontinence":        False,
        "stress_incontinence":         False,
        "urge_incontinence":           False,
        "fecal_incontinence":          True,
        "pelvic_organ_prolapse":       True,
        "chronic_pelvic_pain":         False,
        "dyspareunia":                 False,
        "constipation":                True,
    },
    "uterine_prolapse": {
        "urinary_incontinence":        True,
        "stress_incontinence":         True,
        "urge_incontinence":           False,
        "fecal_incontinence":          False,
        "pelvic_organ_prolapse":       True,
        "chronic_pelvic_pain":         True,
        "dyspareunia":                 True,
        "constipation":                False,
    },
}


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _rnd(rng: random.Random, lo: float, hi: float, digits: int = 1) -> float:
    return round(rng.uniform(lo, hi), digits)


# ---------------------------------------------------------------------------
# Field generators
# ---------------------------------------------------------------------------

def _demographics(rng: random.Random, severity: str, region: str) -> dict:
    """Generate age, anthropometrics, education, SES, etc."""
    age = int(rng.gauss(38, 9))
    age = max(20, min(60, age))

    # South Indian female height/weight ranges
    height_cm = round(rng.gauss(154, 6), 1)
    height_cm = max(140.0, min(172.0, height_cm))
    weight_kg = round(rng.gauss(58, 10), 1)
    weight_kg = max(38.0, min(90.0, weight_kg))
    bmi = round(weight_kg / (height_cm / 100) ** 2, 1)
    waist = round(rng.gauss(78, 9), 1)
    hip   = round(rng.gauss(96, 8), 1)
    whr   = round(waist / hip, 2)

    urban_prob = 0.45 if region == "hilly" else 0.60
    residential = "Rural" if rng.random() > urban_prob else "Urban"

    return {
        "age_years":           age,
        "sex":                 "Female",
        "marital_status":      rng.choice(_MARITAL),
        "occupation":          rng.choice(_OCCUPATIONS),
        "residential_area":    residential,
        "educational_status":  rng.choices(_EDUCATION, weights=[5,10,15,25,20,15,10])[0],
        "socioeconomic_status":rng.choices(_SES, weights=[10,25,35,20,10])[0],
        "height_cm":           height_cm,
        "weight_kg":           weight_kg,
        "bmi":                 bmi,
        "waist_circumference_cm": waist,
        "hip_circumference_cm":   hip,
        "waist_hip_ratio":        whr,
    }


def _menstrual(rng: random.Random, age: int) -> dict:
    menarche = int(rng.gauss(13, 1.5))
    menarche = max(10, min(17, menarche))
    if age >= 50:
        status = "Menopause"
    elif rng.random() < 0.85:
        status = "Regular"
    else:
        status = "Irregular"
    cycle_days = int(rng.gauss(28, 3))
    cycle_days = max(21, min(35, cycle_days))
    oc_use = "Yes" if rng.random() < 0.18 else "No"
    return {
        "age_at_menarche":            menarche,
        "menstrual_status":           status,
        "duration_of_menstrual_cycle_days": cycle_days if status != "Menopause" else "N/A",
        "oral_contraceptive_use":     oc_use,
    }


def _obstetric(rng: random.Random, age: int, pattern: str) -> dict:
    # Correlate parity with age
    if age < 25:
        max_para = 2
    elif age < 35:
        max_para = 3
    else:
        max_para = 5

    # Uterine prolapse tends to higher parity
    if pattern == "uterine_prolapse":
        para = int(rng.gauss(3, 1))
    else:
        para = int(rng.gauss(2, 1))
    para = max(0, min(max_para, para))
    gravida = para + int(rng.random() < 0.25)  # occasional extra pregnancy

    vaginal = int(rng.uniform(0, para + 1))
    vaginal = min(vaginal, para)
    cs = para - vaginal
    proloned_labour = "Yes" if (vaginal > 0 and rng.random() < 0.30) else "No"
    birth_wt = round(rng.gauss(2.9, 0.4), 2) if para > 0 else "N/A"
    episiotomy = "Yes" if (vaginal > 0 and rng.random() < 0.45) else "No"
    tear = "Yes" if (vaginal > 0 and rng.random() < 0.20) else "No"
    if para > 0:
        since_last = int(rng.uniform(1, max(2, age - 22)))
        since_last = f"{since_last} year(s)"
    else:
        since_last = "N/A"

    return {
        "parity":                       "Nulliparous" if para == 0 else "Multiparous",
        "gravida":                       gravida,
        "para":                          para,
        "number_of_vaginal_deliveries":  vaginal,
        "number_of_caesarean_sections":  cs,
        "history_of_prolonged_labour":   proloned_labour,
        "birth_weight_of_largest_baby_kg": birth_wt,
        "episiotomy_history":            episiotomy,
        "perineal_tear_history":         tear,
        "time_since_last_delivery":      since_last,
    }


def _lifestyle(rng: random.Random, occupation: str) -> dict:
    heavy_occ = {"Agricultural Worker", "Daily Wage Labourer"}
    sedentary_occ = {"Homemaker", "Office Staff", "Teacher"}
    if occupation in heavy_occ:
        activity = "Heavy"
    elif occupation in sedentary_occ:
        activity = "Sedentary"
    else:
        activity = rng.choices(_ACTIVITY, weights=[35, 45, 20])[0]

    return {
        "physical_activity_level":   activity,
        "exercise_habits":           "Yes" if rng.random() < 0.30 else "No",
        "weight_lifting_history":    "Yes" if rng.random() < 0.15 else "No",
        "smoking_status":            "Smoker" if rng.random() < 0.05 else "Non-Smoker",
        "alcohol_intake":            "Yes" if rng.random() < 0.04 else "No",
        "occupational_strain":       "Yes" if occupation in heavy_occ else "No",
        "duration_of_sitting_per_day_hours": round(rng.uniform(1, 10), 1),
    }


def _symptoms(rng: random.Random, pattern: str, severity: str) -> dict:
    base = _PFD_SYMPTOM_MAP.get(pattern, {})
    out = {}
    for sym, expected in base.items():
        # Hilly = more severe = slightly higher symptom prevalence
        noise = 0.08 if severity == "hilly" else 0.05
        val = expected if rng.random() > noise else (not expected)
        out[sym] = "Yes" if val else "No"
    return out


def _pelvimetry(rng: random.Random, severity: str,
                real_spacing_mm: list[float] | None = None,
                pelvic_crop_z: list[int] | None = None) -> dict:
    """Generate plausible pelvimetry measurements.

    'plain' → broader pelvis (larger inlet/outlet diameters).
    'hilly' → narrower pelvis.
    """
    # Base ranges from South Indian clinical literature (cm).
    if severity == "plain":
        tc   = _rnd(rng, 10.8, 12.2)    # True conjugate
        oc   = _rnd(rng, 10.2, 11.5)    # Obstetric conjugate
        dc   = _rnd(rng, 12.0, 13.5)    # Diagonal conjugate
        td   = _rnd(rng, 13.0, 14.5)    # Transverse diameter (inlet)
        obl  = _rnd(rng, 12.0, 13.5)    # Oblique diameter
        isd  = _rnd(rng, 10.0, 11.5)    # Interspinous (midpelvis)
        apmp = _rnd(rng, 11.5, 12.5)    # AP midpelvis
        itd  = _rnd(rng, 10.5, 12.0)    # Intertuberous (outlet)
        apo  = _rnd(rng, 11.0, 12.5)    # AP outlet
        spa  = _rnd(rng, 88, 105, 0)    # Subpubic angle (degrees)
        pelvic_width = _rnd(rng, 25.0, 28.5)
    else:  # hilly
        tc   = _rnd(rng, 9.5, 11.0)
        oc   = _rnd(rng, 9.0, 10.5)
        dc   = _rnd(rng, 10.5, 12.0)
        td   = _rnd(rng, 11.5, 13.5)
        obl  = _rnd(rng, 11.0, 12.5)
        isd  = _rnd(rng, 9.0, 10.5)
        apmp = _rnd(rng, 10.5, 11.5)
        itd  = _rnd(rng, 9.0, 11.0)
        apo  = _rnd(rng, 10.0, 11.5)
        spa  = _rnd(rng, 78, 95, 0)
        pelvic_width = _rnd(rng, 22.0, 26.0)

    # Derive sacral / depth estimates (may be updated from spacing if available)
    sacral_curve = _rnd(rng, 3.5, 6.5)
    sacral_len   = _rnd(rng, 9.0, 13.0)
    pelvic_depth = _rnd(rng, 10.0, 14.5)
    pelvic_incl  = _rnd(rng, 50, 65, 0)  # degrees

    # pelvic_depth is the AP dimension (clinical pelvimetry), not CT z-extent

    shape_weights = _PELVIC_SHAPE_WEIGHTS.copy()
    if severity == "hilly":
        shape_weights = [0.40, 0.30, 0.22, 0.08]
    pelvic_shape = rng.choices(_PELVIC_SHAPE, weights=shape_weights)[0]

    return {
        "pelvic_inlet": {
            "true_conjugate_diameter_cm":         tc,
            "obstetric_conjugate_diameter_cm":     oc,
            "diagonal_conjugate_diameter_cm":      dc,
            "transverse_diameter_of_inlet_cm":     td,
            "oblique_diameter_cm":                 obl,
        },
        "midpelvis": {
            "interspinous_diameter_cm":            isd,
            "anteroposterior_diameter_cm":         apmp,
        },
        "pelvic_outlet": {
            "intertuberous_diameter_cm":           itd,
            "anteroposterior_diameter_cm":         apo,
            "subpubic_angle_degrees":              spa,
        },
        "additional": {
            "sacral_curvature_cm":                 sacral_curve,
            "sacral_length_cm":                    sacral_len,
            "pelvic_depth_cm":                     pelvic_depth,
            "pelvic_width_cm":                     pelvic_width,
            "pelvic_inclination_degrees":          pelvic_incl,
            "pelvic_shape_classification":         pelvic_shape,
        },
    }


def _pfm_measurements(rng: random.Random, pattern: str, severity: str,
                       grade: int = 2) -> dict:
    """Pelvic floor muscle thickness measurements (mm), correlated to PFD grade.

    Thickness and hiatal dimensions are scaled to POP-Q grade following South
    Indian clinical reference ranges (Dietz 2012, Nayak 2018):
      Grade 1 (~POP-Q I) : near-normal, mild thinning
      Grade 2 (~POP-Q II): moderate thinning, hiatus slightly widened
      Grade 3 (~POP-Q III): clear thinning, asymmetry common
      Grade 4 (~POP-Q IV) : severe thinning, widened hiatus, defect likely
    """
    # Combined thinning factor: grade drives the primary change,
    # hilly (android/anthropoid) pelvis adds mild additional thinning
    grade_factor = {1: 0.93, 2: 0.82, 3: 0.68, 4: 0.54}[max(1, min(4, grade))]
    pop_factor   = 0.96 if severity == "hilly" else 1.0
    thin         = grade_factor * pop_factor

    # Asymmetry noise increases with grade (higher grades → uneven avulsion)
    asym_noise = 0.4 + (grade - 1) * 0.15

    def _thick(lo, hi):
        return round(_rnd(rng, lo, hi) * thin, 1)

    la_r = _thick(6.5, 10.0)
    la_l = round(la_r + rng.gauss(0, asym_noise), 1)
    la_m = round((la_r + la_l) / 2, 1)

    pr_r = _thick(6.5, 10.5)
    pr_l = round(pr_r + rng.gauss(0, asym_noise + 0.1), 1)
    pr_m = round((pr_r + pr_l) / 2, 1)

    ic_r = _thick(2.5, 6.0)
    ic_l = round(ic_r + rng.gauss(0, 0.35), 1)
    ic_m = round((ic_r + ic_l) / 2, 1)

    pc_r = _thick(4.0, 8.5)
    pc_l = round(pc_r + rng.gauss(0, asym_noise - 0.1), 1)
    pc_m = round((pc_r + pc_l) / 2, 1)

    # Levator hiatus widens with grade — primary visual finding on CT
    # AP diameter (cm): normal ~4.5, Grade 4 ~9+
    # Lateral diameter (cm): normal ~6.0, Grade 4 ~11+
    hiatal_bases = {1: (4.8, 6.2), 2: (5.8, 7.4), 3: (7.2, 9.0), 4: (9.2, 11.2)}
    hap_base, hlat_base = hiatal_bases[max(1, min(4, grade))]
    hiatal_ap   = round(hap_base  + rng.gauss(0, 0.35), 1)
    hiatal_lat  = round(hlat_base + rng.gauss(0, 0.55), 1)
    hiatal_area = round(math.pi * (hiatal_ap / 2) * (hiatal_lat / 2), 1)

    resting_thick  = round((la_m + pr_m) / 2, 1)
    # Contractility declines at higher grades (stretched / avulsed muscle)
    contract_mult  = {1: (1.18, 1.38), 2: (1.12, 1.30), 3: (1.05, 1.18), 4: (1.01, 1.08)}[
        max(1, min(4, grade))]
    valsalva_mult  = {1: (0.75, 0.90), 2: (0.70, 0.86), 3: (0.62, 0.80), 4: (0.55, 0.72)}[
        max(1, min(4, grade))]
    contract_thick = round(resting_thick * rng.uniform(*contract_mult), 1)
    valsalva_thick = round(resting_thick * rng.uniform(*valsalva_mult), 1)

    # Defect probability rises sharply above Grade 2
    defect_prob = {1: 0.05, 2: 0.15, 3: 0.40, 4: 0.65}[max(1, min(4, grade))]
    if severity == "hilly":
        defect_prob = min(defect_prob * 1.25, 0.90)
    defect   = "Yes" if rng.random() < defect_prob else "No"
    asymm_p  = {1: 0.10, 2: 0.28, 3: 0.52, 4: 0.72}[max(1, min(4, grade))]
    symmetry = "Asymmetric" if (defect == "Yes" or rng.random() < asymm_p) else "Symmetric"

    contractility = "Present" if grade <= 2 else ("Reduced" if grade == 3 else "Absent")

    return {
        "levator_ani_mm":              {"right": la_r, "left": la_l, "mean": la_m},
        "puborectalis_mm":             {"right": pr_r, "left": pr_l, "mean": pr_m},
        "iliococcygeus_mm":            {"right": ic_r, "left": ic_l, "mean": ic_m},
        "pubococcygeus_mm":            {"right": pc_r, "left": pc_l, "mean": pc_m},
        "muscle_symmetry":             symmetry,
        "muscle_defect_presence":      defect,
        "hiatal_dimensions_cm":        {"AP": hiatal_ap, "lateral": hiatal_lat},
        "levator_hiatus_area_cm2":     hiatal_area,
        "muscle_contractility":        contractility,
        "resting_thickness_mm":        resting_thick,
        "thickness_during_contraction_mm": contract_thick,
        "thickness_during_valsalva_mm":    valsalva_thick,
    }


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def generate_clinical_metadata(
    patient_id: str,
    patient_num: int,
    pattern: str,
    severity: str,
    region_id: int,
    grade: int = 2,
    pfd_findings: dict | None = None,
    real_spacing_mm: list[float] | None = None,
    pelvic_crop_z: list[int] | None = None,
    collection_date: str = "",
) -> dict:
    """Return a dict with full synthetic clinical metadata for one patient."""
    rng = _rng(patient_num * 31337 + region_id * 7)

    demo = _demographics(rng, severity, severity)  # region ≈ severity here
    age  = demo["age_years"]
    mens = _menstrual(rng, age)
    obst = _obstetric(rng, age, pattern)
    life = _lifestyle(rng, demo["occupation"])
    symp = _symptoms(rng, pattern, severity)
    pelv = _pelvimetry(rng, severity, real_spacing_mm, pelvic_crop_z)
    pfm  = _pfm_measurements(rng, pattern, severity, grade=grade)
    machine = rng.choice(_MACHINE_MODELS)

    return {
        "participant_identification": {
            "participant_id_number":   patient_id,
            "date_of_data_collection": collection_date,
            "hospital_institution":    "GenAI Pelvic Study Centre",
            "department":              "Radiology & Pelvic Floor Unit",
            "data_collector":          "Synthetic Generator v2",
            "principal_investigator":  "GenAI Research Team",
        },
        "demographics": demo,
        "menstrual_hormonal_history": mens,
        "obstetric_history": obst,
        "lifestyle_factors": life,
        "clinical_symptoms": symp,
        "imaging_details": {
            "imaging_modality":           "CT",
            "machine_model":              machine,
            "probe_type":                 "N/A (CT)",
            "probe_frequency":            "N/A (CT)",
            "imaging_position":           "Supine",
            "bladder_status_during_imaging": rng.choice(["Empty", "Partially Filled"]),
            "image_quality_adequate":     "Yes",
            "repeat_measurement_required":"No",
            "intraobserver_assessment":   "Done",
            "interobserver_assessment":   "Done",
        },
        "pelvimetry_measurements": pelv,
        "pelvic_floor_muscle_measurements": pfm,
        "pfd_clinical": {
            "pfd_pattern":    pattern,
            "pfd_severity":   severity,
            "pfd_findings":   pfd_findings or {},
        },
        "synthetic_metadata": {
            "synthetic": True,
            "generator": "gen_ai_ct_pelvic hybrid_production_v2",
            "note": (
                "All clinical fields are synthetically generated from seeded RNG "
                "using South Indian female pelvic clinical reference ranges. "
                "No real patient data is present."
            ),
        },
    }


# ---------------------------------------------------------------------------
# PDF report writer
# ---------------------------------------------------------------------------

def _yn(val: Any) -> str:
    if isinstance(val, bool):
        return "Yes" if val else "No"
    return str(val)


def write_patient_pdf(clinical_meta: dict, out_path: Path) -> None:
    """Write a formatted PDF data collection sheet for one patient."""
    try:
        from fpdf import FPDF, XPos, YPos
    except ImportError:
        return  # fpdf not installed — skip silently

    pid = clinical_meta["participant_identification"]["participant_id_number"]
    demo = clinical_meta["demographics"]
    mens = clinical_meta["menstrual_hormonal_history"]
    obst = clinical_meta["obstetric_history"]
    life = clinical_meta["lifestyle_factors"]
    symp = clinical_meta["clinical_symptoms"]
    img  = clinical_meta["imaging_details"]
    pelv = clinical_meta["pelvimetry_measurements"]
    pfm  = clinical_meta["pelvic_floor_muscle_measurements"]

    f = FPDF()
    f.set_auto_page_break(auto=True, margin=15)
    f.add_page()

    LM = 15   # left margin
    PW = 180  # page width (A4 210 - 2*15)

    def _s(text: Any) -> str:
        """Strip any non-latin-1 chars so core Helvetica font never chokes."""
        return str(text).encode("latin-1", errors="replace").decode("latin-1")

    def section(title: str):
        f.set_fill_color(30, 60, 120)
        f.set_text_color(255, 255, 255)
        f.set_font("Helvetica", "B", 10)
        f.cell(PW, 7, _s(f"  {title}"), fill=True,
               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        f.set_text_color(0, 0, 0)
        f.ln(1)

    def row(label: str, value: Any, col2_x: float = 90):
        f.set_font("Helvetica", "", 9)
        f.set_x(LM)
        f.cell(col2_x - LM, 5.5, _s(label))
        f.set_font("Helvetica", "B", 9)
        f.cell(PW - col2_x + LM, 5.5, _s(value),
               new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def sub(title: str):
        f.set_font("Helvetica", "I", 9)
        f.set_text_color(60, 60, 60)
        f.set_x(LM)
        f.cell(PW, 5, _s(f"  {title}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        f.set_text_color(0, 0, 0)

    # ---- Title ----
    f.set_font("Helvetica", "B", 14)
    f.set_fill_color(10, 40, 90)
    f.set_text_color(255, 255, 255)
    f.cell(PW, 10, _s("MASTER DATA COLLECTION SHEET"), fill=True, align="C",
           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    f.set_font("Helvetica", "B", 10)
    f.cell(PW, 6,
           _s("Topic: Pelvimetry Measurements and Pelvic Floor Muscle Thickness"),
           fill=True, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    f.set_text_color(0, 0, 0)
    f.ln(4)

    # ---- 1. Participant Identification ----
    section("PARTICIPANT IDENTIFICATION")
    pi = clinical_meta["participant_identification"]
    row("Participant ID Number:",   pi["participant_id_number"])
    row("Date of Data Collection:", pi["date_of_data_collection"])
    row("Hospital/Institution:",    pi["hospital_institution"])
    row("Department:",              pi["department"])
    row("Data Collector:",          pi["data_collector"])
    row("Principal Investigator:",  pi["principal_investigator"])
    f.ln(2)

    # ---- 2. Demographics ----
    section("DEMOGRAPHIC DETAILS")
    sub("Basic Information")
    row("Age (years):",             demo["age_years"])
    row("Sex:",                     demo["sex"])
    row("Marital Status:",          demo["marital_status"])
    row("Occupation:",              demo["occupation"])
    row("Residential Area:",        demo["residential_area"])
    row("Educational Status:",      demo["educational_status"])
    row("Socioeconomic Status:",    demo["socioeconomic_status"])
    sub("Anthropometric Measurements")
    row("Height (cm):",             demo["height_cm"])
    row("Weight (kg):",             demo["weight_kg"])
    row("Body Mass Index (BMI):",   demo["bmi"])
    row("Waist Circumference (cm):",demo["waist_circumference_cm"])
    row("Hip Circumference (cm):",  demo["hip_circumference_cm"])
    row("Waist-Hip Ratio:",          demo["waist_hip_ratio"])
    f.ln(2)

    # ---- 3. Menstrual & Hormonal ----
    section("MENSTRUAL AND HORMONAL HISTORY")
    row("Age at Menarche:",                  mens["age_at_menarche"])
    row("Menstrual Status:",                 mens["menstrual_status"])
    row("Duration of Menstrual Cycle:",      mens["duration_of_menstrual_cycle_days"])
    row("Oral Contraceptive Use:",           mens["oral_contraceptive_use"])
    f.ln(2)

    # ---- 4. Obstetric ----
    section("OBSTETRIC HISTORY")
    sub("Parity Details")
    row("Parity:",                          obst["parity"])
    row("Gravida:",                         obst["gravida"])
    row("Para:",                            obst["para"])
    row("No. of Vaginal Deliveries:",       obst["number_of_vaginal_deliveries"])
    row("No. of Caesarean Sections:",       obst["number_of_caesarean_sections"])
    row("History of Prolonged Labour:",     obst["history_of_prolonged_labour"])
    row("Birth Weight of Largest Baby (kg):",obst["birth_weight_of_largest_baby_kg"])
    row("Episiotomy History:",              obst["episiotomy_history"])
    row("Perineal Tear History:",           obst["perineal_tear_history"])
    row("Time Since Last Delivery:",        obst["time_since_last_delivery"])
    f.ln(2)

    # ---- 5. Lifestyle ----
    section("LIFESTYLE FACTORS")
    row("Physical Activity Level:",         life["physical_activity_level"])
    row("Exercise Habits:",                 life["exercise_habits"])
    row("Weight Lifting History:",          life["weight_lifting_history"])
    row("Smoking Status:",                  life["smoking_status"])
    row("Alcohol Intake:",                  life["alcohol_intake"])
    row("Occupational Strain:",             life["occupational_strain"])
    row("Duration of Sitting Per Day (h):", life["duration_of_sitting_per_day_hours"])
    f.ln(2)

    # ---- 6. Clinical Symptoms ----
    section("CLINICAL SYMPTOMS - PELVIC FLOOR RELATED")
    row("Urinary Incontinence:",            symp.get("urinary_incontinence", "N/A"))
    row("Stress Incontinence:",             symp.get("stress_incontinence", "N/A"))
    row("Urge Incontinence:",               symp.get("urge_incontinence", "N/A"))
    row("Fecal Incontinence:",              symp.get("fecal_incontinence", "N/A"))
    row("Pelvic Organ Prolapse Symptoms:",  symp.get("pelvic_organ_prolapse", "N/A"))
    row("Chronic Pelvic Pain:",             symp.get("chronic_pelvic_pain", "N/A"))
    row("Dyspareunia:",                     symp.get("dyspareunia", "N/A"))
    row("Constipation Symptoms:",           symp.get("constipation", "N/A"))
    f.ln(2)

    # ---- 7. Imaging ----
    section("IMAGING DETAILS")
    sub("Imaging Information")
    row("Imaging Modality:",                img["imaging_modality"])
    row("Machine Model:",                   img["machine_model"])
    row("Probe Type:",                      img["probe_type"])
    row("Probe Frequency:",                 img["probe_frequency"])
    row("Imaging Position:",                img["imaging_position"])
    row("Bladder Status During Imaging:",   img["bladder_status_during_imaging"])
    sub("Quality Control")
    row("Image Quality Adequate:",          img["image_quality_adequate"])
    row("Repeat Measurement Required:",     img["repeat_measurement_required"])
    row("Intraobserver Assessment:",        img["intraobserver_assessment"])
    row("Interobserver Assessment:",        img["interobserver_assessment"])
    f.ln(2)

    # ---- 8. Pelvimetry ----
    section("PELVIMETRY MEASUREMENTS")
    sub("Pelvic Inlet Measurements")
    pi2 = pelv["pelvic_inlet"]
    row("True Conjugate Diameter (cm):",        pi2["true_conjugate_diameter_cm"])
    row("Obstetric Conjugate Diameter (cm):",   pi2["obstetric_conjugate_diameter_cm"])
    row("Diagonal Conjugate Diameter (cm):",    pi2["diagonal_conjugate_diameter_cm"])
    row("Transverse Diameter of Inlet (cm):",   pi2["transverse_diameter_of_inlet_cm"])
    row("Oblique Diameter (cm):",               pi2["oblique_diameter_cm"])
    sub("Midpelvis Measurements")
    mp = pelv["midpelvis"]
    row("Interspinous Diameter (cm):",          mp["interspinous_diameter_cm"])
    row("AP Diameter of Midpelvis (cm):",       mp["anteroposterior_diameter_cm"])
    sub("Pelvic Outlet Measurements")
    po = pelv["pelvic_outlet"]
    row("Intertuberous Diameter (cm):",         po["intertuberous_diameter_cm"])
    row("AP Diameter of Outlet (cm):",          po["anteroposterior_diameter_cm"])
    row("Subpubic Angle (degrees):",            po["subpubic_angle_degrees"])
    sub("Additional Pelvic Measurements")
    ad = pelv["additional"]
    row("Sacral Curvature (cm):",               ad["sacral_curvature_cm"])
    row("Sacral Length (cm):",                  ad["sacral_length_cm"])
    row("Pelvic Depth (cm):",                   ad["pelvic_depth_cm"])
    row("Pelvic Width (cm):",                   ad["pelvic_width_cm"])
    row("Pelvic Inclination (degrees):",        ad["pelvic_inclination_degrees"])
    row("Pelvic Shape Classification:",         ad["pelvic_shape_classification"])
    f.ln(2)

    # ---- 9. Pelvic Floor Muscle ----
    section("PELVIC FLOOR MUSCLE MEASUREMENTS")
    sub("Muscle Thickness Measurements (mm)")

    def muscle_row(name, d):
        row(f"{name} - Right:",  d["right"])
        row(f"{name} - Left:",   d["left"])
        row(f"{name} - Mean:",   d["mean"])

    muscle_row("Levator Ani Thickness",    pfm["levator_ani_mm"])
    muscle_row("Puborectalis Thickness",   pfm["puborectalis_mm"])
    muscle_row("Iliococcygeus Thickness",  pfm["iliococcygeus_mm"])
    muscle_row("Pubococcygeus Thickness",  pfm["pubococcygeus_mm"])

    sub("Muscle Functional Parameters")
    row("Muscle Symmetry:",                   pfm["muscle_symmetry"])
    row("Muscle Defect Presence:",            pfm["muscle_defect_presence"])
    hd = pfm["hiatal_dimensions_cm"]
    row("Hiatal Dimensions AP (cm):",         hd["AP"])
    row("Hiatal Dimensions Lateral (cm):",    hd["lateral"])
    row("Levator Hiatus Area (cm2):",         pfm["levator_hiatus_area_cm2"])
    row("Muscle Contractility:",              pfm["muscle_contractility"])
    row("Resting Thickness (mm):",            pfm["resting_thickness_mm"])
    row("Thickness During Contraction (mm):", pfm["thickness_during_contraction_mm"])
    row("Thickness During Valsalva (mm):",    pfm["thickness_during_valsalva_mm"])
    f.ln(2)

    # ---- Footer ----
    f.set_font("Helvetica", "I", 7)
    f.set_text_color(100, 100, 100)
    f.cell(PW, 5,
           _s("SYNTHETIC DATA - All clinical fields generated by gen_ai_ct_pelvic. "
              "Not derived from real patients."),
           align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    f.output(str(out_path))
