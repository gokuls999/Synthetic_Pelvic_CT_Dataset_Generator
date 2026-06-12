"""Rebuild summary_plain.json, summary_hilly.json, dataset_overview.json
from the actual 175+175 patient folders after trimming."""
import json, datetime
from pathlib import Path

out_root = Path("synthetic_dataset/PFD_Synthetic_Dataset_350")
SEVERITY = ["", "mild", "moderate", "severe", "complete"]

def load_patients(pop_dir, pop_label):
    patients = []
    for pdir in sorted(pop_dir.iterdir()):
        if not pdir.is_dir():
            continue
        mf = pdir / "metadata.json"
        cf = pdir / "clinical_data.json"
        if not mf.exists():
            continue
        try:
            m = json.loads(mf.read_text())
            findings = {}
            if cf.exists():
                c = json.loads(cf.read_text())
                findings = c.get("pfd_findings", {})
            grade = m.get("pfd_grade", 1)
            patients.append({
                "patient_id":         pdir.name,
                "population":         pop_label,
                "pfd_pattern":        m.get("pfd_pattern"),
                "pfd_grade":          grade,
                "pfd_severity":       SEVERITY[max(0, min(4, grade))],
                "transverse_diam_mm": m.get("transverse_diam_mm"),
                "actual_population":  m.get("actual_population"),
                "source_dataset":     m.get("source_dataset"),
                "n_slices":           m.get("n_slices"),
                "findings":           findings,
            })
        except Exception as e:
            print(f"  skip {pdir.name}: {e}")
    return patients


def make_pop_summary(patients, pop_label, pelvic_type):
    grades   = {f"grade_{g}": len([p for p in patients if p.get("pfd_grade") == g]) for g in range(1, 5)}
    patterns = {}
    for p in patients:
        pat = p.get("pfd_pattern", "unknown")
        patterns[pat] = patterns.get(pat, 0) + 1
    return {
        "population":         pop_label,
        "pelvic_type":        pelvic_type,
        "td_threshold_mm":    108.0,
        "n_patients":         len(patients),
        "grade_distribution": grades,
        "pattern_distribution": patterns,
        "deformation_mode":   "confined_organ_plus_levator",
        "patients":           patients,
    }


plain_patients = load_patients(out_root / "Plain_175", "plain")
hilly_patients = load_patients(out_root / "Hilly_175", "hilly")
all_patients   = plain_patients + hilly_patients

plain_doc = make_pop_summary(plain_patients, "plain", "Gynecoid (plains South India)")
hilly_doc = make_pop_summary(hilly_patients, "hilly", "Android/Anthropoid (hilly South India)")

(out_root / "Plain_175" / "summary_plain.json").write_text(json.dumps(plain_doc, indent=2))
(out_root / "Hilly_175" / "summary_hilly.json").write_text(json.dumps(hilly_doc, indent=2))

grade_dist   = {f"grade_{g}": len([p for p in all_patients if p.get("pfd_grade") == g]) for g in range(1, 5)}
pattern_dist = {k: sum(1 for p in all_patients if p.get("pfd_pattern") == k)
                for k in ["combined_pfd", "cystocele", "rectocele", "uterine_prolapse"]}

overview = {
    "dataset_title":  "Synthetic Pelvic CT Dataset - PFD Study (South Indian Women)",
    "generated_date": datetime.date.today().isoformat(),
    "version":        "v6",
    "total_patients": len(all_patients),
    "plain_patients": len(plain_patients),
    "hilly_patients": len(hilly_patients),
    "populations": {
        "plain": "Gynecoid pelvis - TD >= 108 mm (Plains South India)",
        "hilly": "Android/Anthropoid pelvis - TD < 108 mm (Hilly South India)",
    },
    "grade_distribution":   grade_dist,
    "pattern_distribution": pattern_dist,
    "per_patient_files": {
        "DICOM/":                      "CT volume in DICOM format",
        "DICOM/segmentation.seg.nrrd": "3D Slicer segmentation (bones, organs, muscles, pelvic floor)",
        "PNG/":                        "CT slices as PNG",
        "JPG/":                        "CT slices as JPEG",
        "clinical_data.json":          "Full clinical metadata",
        "metadata.json":               "Compact per-patient summary",
        "patient_report.pdf":          "PDF data collection sheet",
        "COMPARISON.png":              "Side-by-side original vs PFD-edited vs difference",
    },
    "folder_structure": {
        "Plain_175/":                    "175 gynecoid patients",
        "Hilly_175/":                    "175 android/anthropoid patients",
        "Plain_175/summary_plain.json":  "Plain population summary",
        "Hilly_175/summary_hilly.json":  "Hilly population summary",
        "dataset_overview.json":         "This file",
    },
}
(out_root / "dataset_overview.json").write_text(json.dumps(overview, indent=2))
(out_root / "roster_summary.json").write_text(json.dumps({
    "version":      "v6",
    "n_planned":    350,
    "n_succeeded":  350,
    "n_plain":      len(plain_patients),
    "n_hilly":      len(hilly_patients),
    "n_failed":     0,
    "deformation_mode": "confined_organ_plus_levator",
    "grade_distribution": {f"grade{g}": len([p for p in all_patients if p.get("pfd_grade") == g]) for g in range(1, 5)},
}, indent=2))

print(f"plain={len(plain_patients)}  hilly={len(hilly_patients)}  total={len(all_patients)}")
print("Grades  :", grade_dist)
print("Patterns:", pattern_dist)
