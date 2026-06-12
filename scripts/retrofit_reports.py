"""Option A: Retrofit existing 350 patients in D:\\PFD_Dataset_350 with
clinical_data.json + patient_report.pdf.

Reads each patient's existing metadata.json, generates full clinical
metadata and a PDF data collection sheet, and writes both files into the
same patient folder.  Already-retrofitted patients are skipped unless
--force is passed.

Usage:
    python scripts/retrofit_reports.py
    python scripts/retrofit_reports.py --force          # overwrite existing PDFs
    python scripts/retrofit_reports.py --src D:/other   # different source folder
"""

from _common import add_repo_to_path
add_repo_to_path()

import argparse
import datetime
import json
import sys
from pathlib import Path

from src.patient_report import generate_clinical_metadata, write_patient_pdf
from src import web_progress as wp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", default="D:/PFD_Dataset_350",
                    help="Folder containing PFD_NNNN_* patient subdirectories")
    ap.add_argument("--port", type=int, default=8767,
                    help="Dashboard port (default 8767)")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-generate even if patient_report.pdf already exists")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        sys.exit(f"Source folder not found: {src}")

    patients = sorted(p for p in src.iterdir()
                      if p.is_dir() and p.name.startswith("PFD_"))
    if not patients:
        sys.exit(f"No PFD_* patient folders found in {src}")

    today = datetime.date.today().isoformat()

    wp.set_stages([
        ("retrofit", f"Retrofitting {len(patients)} patients with clinical data + PDF"),
    ])
    wp.start_server(port=args.port, open_browser=not args.no_browser,
                    run_label=f"Retrofit Reports  src={src}  n={len(patients)}",
                    expected_total_s=len(patients) * 0.5,
                    cfg=None)
    wp.set_stage("retrofit", total=len(patients), postfix="starting")

    done = skipped = failed = 0

    for idx, pdir in enumerate(patients, start=1):
        meta_path = pdir / "metadata.json"
        pdf_path  = pdir / "patient_report.pdf"
        json_path = pdir / "clinical_data.json"

        if pdf_path.exists() and json_path.exists() and not args.force:
            skipped += 1
            wp.update_stage(current=idx,
                            postfix=f"{idx}/{len(patients)} skipped={skipped} done={done}")
            continue

        wp.update_stage(current=idx - 1,
                        postfix=f"{idx}/{len(patients)} {pdir.name}")

        if not meta_path.exists():
            wp.log_msg(f"  SKIP {pdir.name}: no metadata.json")
            skipped += 1
            wp.update_stage(current=idx,
                            postfix=f"{idx}/{len(patients)} skipped={skipped} done={done}")
            continue

        try:
            meta = json.loads(meta_path.read_text())

            # Parse patient number from folder name PFD_NNNN_...
            parts = pdir.name.split("_")
            patient_num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else idx

            pattern  = meta.get("pfd_pattern", "combined_pfd")
            severity = meta.get("pfd_severity", meta.get("region", "plain"))
            region_id   = int(meta.get("region_id", 0))
            real_spacing = meta.get("real_spacing_mm") or meta.get("pixel_spacing_mm")
            crop_z       = meta.get("pelvic_crop_z")
            pfd_findings = meta.get("pfd_findings")

            clinical = generate_clinical_metadata(
                patient_id=pdir.name,
                patient_num=patient_num,
                pattern=pattern,
                severity=severity,
                region_id=region_id,
                pfd_findings=pfd_findings,
                real_spacing_mm=real_spacing,
                pelvic_crop_z=crop_z,
                collection_date=today,
            )
            json_path.write_text(json.dumps(clinical, indent=2))
            write_patient_pdf(clinical, pdf_path)

            done += 1
            wp.update_stage(current=idx,
                            postfix=f"{idx}/{len(patients)} done={done} skip={skipped} fail={failed}")
            wp.log_msg(f"  OK  {pdir.name}")

        except Exception as e:
            failed += 1
            wp.log_msg(f"  FAIL {pdir.name}: {e}")
            wp.update_stage(current=idx,
                            postfix=f"{idx}/{len(patients)} done={done} skip={skipped} fail={failed}")

    wp.finish_stage(f"done: {done} retrofitted, {skipped} skipped, {failed} failed")

    print()
    print("=" * 60)
    print(f"  RETROFIT COMPLETE")
    print(f"  Retrofitted : {done}")
    print(f"  Skipped     : {skipped}  (already had PDF+JSON)")
    print(f"  Failed      : {failed}")
    print(f"  Source      : {src}")
    print("=" * 60)
    wp.stop_server(grace_s=30.0)


if __name__ == "__main__":
    main()
