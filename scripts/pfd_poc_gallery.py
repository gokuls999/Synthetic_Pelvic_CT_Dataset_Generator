"""Run several PFD POC patterns back-to-back in ONE dashboard session.

Default gallery: 5 patients showing 1 plain combined + 4 severe variants
(combined / cystocele only / rectocele only / uterine prolapse only).

Same dashboard UI as the main training, on port 8766. Each pattern fires its
own 6 stages in sequence so you can watch the gallery build up.

CPU-only -- does NOT touch the GPU training is using.
"""

from _common import add_repo_to_path, load_config
add_repo_to_path()

import argparse
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models import CVAE
from src.preprocessing import unwindow
from src.generate import SyntheticVolume
from src.dicom_builder import write_dicom_series
from src.exports import write_png_jpg_metadata
from src.pfd_deformation import build_pattern, build_displacement_field, apply_deformation
from src import web_progress as wp


# Gallery roster: (pattern_name, severity)
GALLERY = [
    ("combined_pfd",     "plain"),
    ("combined_pfd",     "hilly"),
    ("cystocele",        "hilly"),
    ("rectocele",        "hilly"),
    ("uterine_prolapse", "hilly"),
]


def make_stages(roster):
    out = []
    for idx, (pat, sev) in enumerate(roster, start=1):
        prefix = f"p{idx}"
        label = f"[{idx}/{len(roster)}] {pat} ({sev})"
        out.extend([
            (f"{prefix}_pick",    f"{label} - pick volume"),
            (f"{prefix}_load",    f"{label} - load volume"),
            (f"{prefix}_field",   f"{label} - build deformation"),
            (f"{prefix}_deform",  f"{label} - apply warp"),
            (f"{prefix}_refine",  f"{label} - CVAE refine"),
            (f"{prefix}_write",   f"{label} - write DICOM"),
        ])
    return out


def pick_source_volume(labels_csv: Path, severity: str) -> dict:
    df = pd.read_csv(labels_csv)
    sub = df[df["region"] == severity].copy()
    if len(sub) == 0:
        raise RuntimeError(f"no '{severity}' volumes in {labels_csv}")
    med = sub["pelvic_inlet_mm"].median()
    sub["dist"] = (sub["pelvic_inlet_mm"] - med).abs()
    row = sub.sort_values("dist").iloc[0]
    return {
        "uid": str(row["uid"]), "dataset": str(row["dataset"]),
        "cache_path": str(row["cache_path"]),
        "pelvic_inlet_mm": float(row["pelvic_inlet_mm"]),
        "region": str(row["region"]), "region_id": int(row["region_id"]),
    }


@torch.no_grad()
def refine(cvae, vol_norm, region_id, batch, on_slice):
    Z = vol_norm.shape[0]
    out = np.empty_like(vol_norm)
    region_t = torch.tensor([region_id], dtype=torch.long)
    for start in range(0, Z, batch):
        end = min(start + batch, Z)
        x = torch.from_numpy(vol_norm[start:end]).unsqueeze(1)
        z_indices = np.arange(start, end)
        z_pos = (z_indices / max(Z - 1, 1)) * 2.0 - 1.0
        z_pos_t = torch.from_numpy(z_pos).float()
        rid = region_t.expand(end - start)
        cond_vec = cvae.cond(rid, z_pos_t)
        mu, _ = cvae.encode(x, cond_vec)
        x_rec = cvae.decode(mu, cond_vec)
        out[start:end] = x_rec.squeeze(1).cpu().numpy()
        on_slice(end, Z)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--skip-done", action="store_true",
                    help="Skip patterns whose output dir already has a DICOM/ folder")
    ap.add_argument("--no-refine", action="store_true",
                    help="Skip the CVAE encode-decode pass. Output the deformed REAL "
                         "volume as DICOM directly. Writes to <outputs>/_pfd_raw/ instead "
                         "of <outputs>/_pfd_poc/ so the refined gallery is preserved.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg["paths"]["checkpoints_dir"]) / "cvae_latest.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"CVAE checkpoint not found: {ckpt_path}")

    out_subdir = "_pfd_raw" if args.no_refine else "_pfd_poc"
    out_root = Path(cfg["paths"]["outputs_dir"]) / out_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    wp.set_stages(make_stages(GALLERY))
    refine_tag = "RAW deform (no CVAE)" if args.no_refine else "refined"
    expected_s = len(GALLERY) * (8.0 if args.no_refine else 85.0)
    url = wp.start_server(port=args.port, open_browser=not args.no_browser,
                          run_label=f"PFD Gallery  {refine_tag}  {len(GALLERY)} patients",
                          expected_total_s=expected_s, cfg=None)
    print()
    print("=" * 72)
    print(f"  PFD Gallery dashboard: {url}")
    print(f"  Main training still:   http://127.0.0.1:8765/  (untouched)")
    print(f"  Output root:           {out_root}")
    print("=" * 72)
    print()

    # Build CVAE once and reuse across all patterns -- skipped in --no-refine mode.
    cvae = None
    if not args.no_refine:
        cvae_mcfg = cfg["model"]["cvae"]
        cvae = CVAE(
            in_channels=cvae_mcfg["in_channels"], base_channels=cvae_mcfg["base_channels"],
            latent_channels=cvae_mcfg["latent_channels"], latent_size=cvae_mcfg["latent_size"],
            cond_dim=cvae_mcfg["cond_dim"],
        ).eval()
        state = torch.load(ckpt_path, map_location="cpu")
        cvae.load_state_dict(state["model"])
        wp.log_msg(f"loaded CVAE from {ckpt_path.name}")
    else:
        wp.log_msg("--no-refine: skipping CVAE; output is raw deformed real anatomy")

    hu_min = float(cfg["preprocess"]["hu_min"])
    hu_max = float(cfg["preprocess"]["hu_max"])
    spacing = (float(cfg["output"]["slice_thickness_mm"]),
               float(cfg["output"]["pixel_spacing_mm"][0]),
               float(cfg["output"]["pixel_spacing_mm"][1]))
    labels_csv = Path(cfg["paths"]["labels_csv"])
    summary = []
    t_gallery = time.time()

    for idx, (pattern_name, severity) in enumerate(GALLERY, start=1):
        prefix = f"p{idx}"
        patient_id = f"PFD_POC_{pattern_name.upper()}_{severity.upper()}"
        pdir = out_root / patient_id

        if args.skip_done and (pdir / "DICOM").is_dir():
            wp.log_msg(f"[{idx}/{len(GALLERY)}] skip {patient_id} (already done)")
            # Mark all 6 stages of this patient done so the bar fills in.
            for suffix in ("pick", "load", "field", "deform", "refine", "write"):
                wp.set_stage(f"{prefix}_{suffix}", total=0, postfix="skipped (already on disk)")
                wp.finish_stage("done")
            summary.append({"patient_id": patient_id, "skipped": True})
            continue

        wp.log_msg(f"[{idx}/{len(GALLERY)}] start {patient_id}")
        t_pat = time.time()

        # --- pick ---
        wp.set_stage(f"{prefix}_pick", postfix="reading labels csv")
        info = pick_source_volume(labels_csv, severity)
        wp.update_stage(postfix=f"picked {info['uid']}  inlet={info['pelvic_inlet_mm']:.1f}mm")
        wp.finish_stage("done")

        # --- load ---
        wp.set_stage(f"{prefix}_load", postfix=info["cache_path"])
        with np.load(info["cache_path"]) as npz:
            vol_norm = np.asarray(npz["slices"]).astype(np.float32)
        Z, H, W = vol_norm.shape
        wp.update_stage(postfix=f"Z={Z}  H={H}  W={W}")
        wp.finish_stage("done")

        # --- build field ---
        wp.set_stage(f"{prefix}_field", postfix=f"{pattern_name} ({severity})")
        pattern = build_pattern(pattern_name, (Z, H, W), severity=severity)
        field = build_displacement_field((Z, H, W), pattern.blobs)
        peak = float(np.max(np.linalg.norm(field, axis=0)))
        wp.update_stage(postfix=f"{len(pattern.blobs)} blob(s), peak {peak:.1f} vox")
        wp.log_msg(f"  findings: {json.dumps(pattern.findings)}")
        wp.finish_stage("done")

        # --- deform ---
        wp.set_stage(f"{prefix}_deform", postfix="warping")
        t0 = time.time()
        deformed = apply_deformation(vol_norm, field, order=1, cval=-1.0)
        wp.update_stage(postfix=f"warped in {time.time()-t0:.1f}s")
        wp.finish_stage("done")

        # --- refine ---
        if args.no_refine:
            wp.set_stage(f"{prefix}_refine", total=0,
                         postfix="skipped (--no-refine: outputting raw deformation)")
            refined = deformed                       # pass-through
            wp.finish_stage("done")
        else:
            wp.set_stage(f"{prefix}_refine", total=Z, postfix="encode->decode")
            last = [0.0]
            def on_slice(done, total):
                wp.update_stage(current=done, total=total)
                now = time.time()
                if now - last[0] > 4.0:
                    last[0] = now
                    wp.log_msg(f"  [{idx}/{len(GALLERY)}] refine {done}/{total}")
            t0 = time.time()
            refined = refine(cvae, deformed, info["region_id"], args.batch, on_slice)
            wp.update_stage(current=Z, total=Z,
                            postfix=f"refined in {(time.time()-t0)/60:.1f} min")
            wp.finish_stage("done")

        # --- write ---
        wp.set_stage(f"{prefix}_write", postfix="DICOM + PNG + JPG + comparison")
        vol_hu = np.clip(unwindow(refined, hu_min, hu_max), -1024.0, 3071.0).astype(np.int16)
        vol = SyntheticVolume(
            pixels_hu=vol_hu, spacing=spacing, region=info["region"],
            region_id=info["region_id"], patient_id=patient_id, seed=0,
        )
        dicom_dir = pdir / "DICOM"
        png_dir   = pdir / "PNG"
        jpg_dir   = pdir / "JPG"
        meta_path = pdir / "metadata.json"
        write_dicom_series(vol, dicom_dir, cfg)
        write_png_jpg_metadata(vol, png_dir, jpg_dir, meta_path, cfg)

        # comparison png
        from PIL import Image
        def to_u8(slc):
            return ((np.clip(slc, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
        z_picks = [int(Z * f) for f in np.linspace(0.30, 0.70, 3)]
        rows = []
        for z in z_picks:
            rows.append(np.concatenate(
                [to_u8(vol_norm[z]), to_u8(deformed[z]), to_u8(refined[z])], axis=1))
        Image.fromarray(np.concatenate(rows, axis=0), mode="L").save(pdir / "COMPARISON.png")

        # patch metadata with PFD findings
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        meta.update({
            "synthetic_strategy": (
                "hybrid_deformation_raw" if args.no_refine
                else "hybrid_deformation_poc"
            ),
            "cvae_refinement": (not args.no_refine),
            "source_volume_uid": info["uid"], "source_dataset": info["dataset"],
            "pfd_pattern": pattern.name, "pfd_severity": pattern.severity,
            "pfd_findings": pattern.findings,
            "cvae_checkpoint": (None if args.no_refine else ckpt_path.name),
        })
        meta_path.write_text(json.dumps(meta, indent=2))
        wp.update_stage(postfix=f"wrote {Z} slices -> {pdir.name}")
        wp.finish_stage("done")

        dt = time.time() - t_pat
        wp.log_msg(f"[{idx}/{len(GALLERY)}] done {patient_id}  ({dt/60:.1f} min)")
        summary.append({"patient_id": patient_id, "n_slices": int(Z),
                        "pattern": pattern.name, "severity": pattern.severity,
                        "dir": str(pdir)})

    total_min = (time.time() - t_gallery) / 60.0
    wp.log_msg(f"gallery complete: {len(summary)} patients in {total_min:.1f} min")

    print()
    print("=" * 72)
    print(f"  GALLERY DONE in {total_min:.1f} min")
    for s in summary:
        if s.get("skipped"):
            print(f"    SKIPPED  {s['patient_id']}")
        else:
            print(f"    {s['patient_id']:42s}  {s['n_slices']:3d} slices")
    print(f"  Output root: {out_root}")
    print(f"  Each patient has: DICOM/ PNG/ JPG/ COMPARISON.png  metadata.json")
    print("=" * 72)
    wp.stop_server(grace_s=120.0)


if __name__ == "__main__":
    main()
