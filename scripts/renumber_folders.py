"""Renumber patient folders sequentially 0001-0175 in each population dir.
Reads metadata.json inside each folder to reconstruct the canonical name."""
import json, os
from pathlib import Path

out_root = Path("synthetic_dataset/PFD_Synthetic_Dataset_350")

PATTERN_TAG = {
    "combined_pfd":     "COMBINEDPFD",
    "cystocele":        "CYSTOCELE",
    "rectocele":        "RECTOCELE",
    "uterine_prolapse": "UTERINEPROLAPSE",
}

for pop_dir, pop_tag in [("Plain_175", "PLAIN"), ("Hilly_175", "HILLY")]:
    full_dir = out_root / pop_dir
    folders  = sorted([f for f in full_dir.iterdir() if f.is_dir()])

    print(f"\n{pop_dir}: {len(folders)} folders")

    # Pass 1: rename all to __TEMPN__ to avoid collisions
    for i, f in enumerate(folders, start=1):
        f.rename(full_dir / f"__TEMP_{i:04d}__")

    # Pass 2: rename to final canonical names using metadata.json
    temps = sorted([f for f in full_dir.iterdir() if f.is_dir()])
    for i, f in enumerate(temps, start=1):
        idx_str = f.name.replace("__TEMP_", "").replace("__", "")
        meta_path = f / "metadata.json"
        try:
            m = json.loads(meta_path.read_text())
            pattern = PATTERN_TAG.get(m.get("pfd_pattern", ""), "UNKNOWN")
            grade   = m.get("pfd_grade", 1)
            new_name = f"PFD_{i:04d}_{pop_tag}_{pattern}_G{grade}"
        except Exception:
            new_name = f"PFD_{i:04d}_{pop_tag}_UNKNOWN"

        f.rename(full_dir / new_name)
        print(f"  {i:3d}. {new_name}")

print("\nDone.")
