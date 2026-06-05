"""Hybrid PFD generation -- Phase 0 Proof of Concept.

End-to-end proof that the hybrid plan works:
    real cached volume -> parametric PFD deformation -> CVAE refine -> DICOM

Runs on CPU only so it does NOT touch the GPU that the main training is using.
Opens its own dashboard (same UI as the main training one) on port 8766 so you
can watch the per-stage progress in your browser.

Stages shown in the dashboard:
    pfd_pick_volume     - choose a real volume from the cache
    pfd_load            - read cached slices and build [Z,H,W] tensor
    pfd_build_field     - build the 3D Gaussian displacement field
    pfd_deform          - apply the displacement to the volume
    pfd_refine          - encode -> decode each slice through current CVAE
    pfd_write_dicom     - serialize the CT series + side-by-side preview PNG

Outputs go to synthetic_dataset/_pfd_poc/<patient_id>/ and the dashboard prints
the URL of the resulting DICOM folder when done. Open it in 3D Slicer / RadiAnt
/ OHIF to inspect.
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
from PIL import Image

from src.models import CVAE
from src.preprocessing import unwindow
from src.generate import SyntheticVolume
from src.dicom_builder import write_dicom_series
from src.exports import write_png_jpg_metadata
from src.pfd_deformation import build_pattern, build_displacement_field, apply_deformation
from src import web_progress as wp


PFD_STAGES = [
    ("pfd_pick_volume",  "Pick real source volume"),
    ("pfd_load",         "Load cached volume"),
    ("pfd_build_field",  "Build PFD displacement field"),
    ("pfd_deform",       "Apply deformation"),
    ("pfd_refine",       "Refine via trained CVAE (encode-decode)"),
    ("pfd_write_dicom",  "Write DICOM + PNG + metadata"),
]


def pick_source_volume(labels_csv: Path, severity: str) -> dict:
    """Pick a representative real volume for the given severity bucket.

    POC mapping: severity='plain' -> region='plain' (broader/healthier pelvis)
                 severity='hilly' -> region='hilly' (narrower/more variable)
    Picks the volume closest to its cluster median pelvic_inlet_mm so the
    output is anatomically representative."""
    df = pd.read_csv(labels_csv)
    sub = df[df["region"] == severity].copy()
    if len(sub) == 0:
        raise RuntimeError(f"No volumes labelled '{severity}' in {labels_csv}")
    med = sub["pelvic_inlet_mm"].median()
    sub["dist"] = (sub["pelvic_inlet_mm"] - med).abs()
    row = sub.sort_values("dist").iloc[0]
    return {
        "uid": str(row["uid"]),
        "dataset": str(row["dataset"]),
        "cache_path": str(row["cache_path"]),
        "pelvic_inlet_mm": float(row["pelvic_inlet_mm"]),
        "region": str(row["region"]),
        "region_id": int(row["region_id"]),
    }


def load_volume(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as npz:
        return np.asarray(npz["slices"]).astype(np.float32)


@torch.no_grad()
def refine_with_cvae(cvae: CVAE, vol_norm: np.ndarray, region_id: int,
                    batch: int = 4, on_slice=None) -> np.ndarray:
    """Encode -> decode each slice through the trained CVAE. Uses mu (no
    sampling) so the refinement is deterministic. on_slice(done, total) is
    called per batch for progress updates."""
    Z, H, W = vol_norm.shape
    out = np.empty_like(vol_norm)
    region_t = torch.tensor([region_id], dtype=torch.long)
    for start in range(0, Z, batch):
        end = min(start + batch, Z)
        chunk = vol_norm[start:end]
        x = torch.from_numpy(chunk).unsqueeze(1)
        z_indices = np.arange(start, end)
        z_pos = (z_indices / max(Z - 1, 1)) * 2.0 - 1.0
        z_pos_t = torch.from_numpy(z_pos).float()
        rid = region_t.expand(end - start)
        cond_vec = cvae.cond(rid, z_pos_t)
        mu, _ = cvae.encode(x, cond_vec)
        x_rec = cvae.decode(mu, cond_vec)
        out[start:end] = x_rec.squeeze(1).cpu().numpy()
        if on_slice:
            on_slice(end, Z)
    return out


def save_comparison_png(orig_norm: np.ndarray, deformed_norm: np.ndarray,
                        refined_norm: np.ndarray, lo: float, hi: float,
                        out_path: Path, n_strips: int = 3) -> None:
    """Stack a small comparison grid: 3 representative axial slices,
    showing ORIGINAL | DEFORMED | REFINED side-by-side."""
    Z = orig_norm.shape[0]
    z_picks = [int(Z * f) for f in np.linspace(0.30, 0.70, n_strips)]

    def to_u8(slc):
        # Slice is in [-1, 1] (windowed); map to 0..255 linearly.
        s = (np.clip(slc, -1.0, 1.0) + 1.0) * 127.5
        return s.astype(np.uint8)

    rows = []
    for z in z_picks:
        row = np.concatenate([to_u8(orig_norm[z]),
                              to_u8(deformed_norm[z]),
                              to_u8(refined_norm[z])], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    img = Image.fromarray(grid, mode="L")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pattern", default="combined_pfd",
                    choices=["cystocele", "rectocele", "uterine_prolapse", "combined_pfd"],
                    help="Which PFD pattern to apply")
    ap.add_argument("--severity", default="hilly", choices=["plain", "hilly"],
                    help="Mild (plain) or severe (hilly) parameter set")
    ap.add_argument("--port", type=int, default=8766,
                    help="Dashboard port (default 8766 to avoid the training one on 8765)")
    ap.add_argument("--no-browser", action="store_true",
                    help="Don't auto-open the dashboard in a browser")
    ap.add_argument("--ckpt", default=None,
                    help="CVAE checkpoint (default: <checkpoints_dir>/cvae_latest.pt)")
    ap.add_argument("--batch", type=int, default=4,
                    help="Slices per CVAE forward batch on CPU (default 4)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["training"]["device"] = "cpu"

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg["paths"]["checkpoints_dir"]) / "cvae_latest.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"CVAE checkpoint not found: {ckpt_path}")

    out_root = Path(cfg["paths"]["outputs_dir"]) / "_pfd_poc"
    out_root.mkdir(parents=True, exist_ok=True)
    patient_id = f"PFD_POC_{args.pattern.upper()}_{args.severity.upper()}"
    pdir = out_root / patient_id

    # ------- Dashboard up FIRST so every subsequent action is visible -------
    wp.set_stages(PFD_STAGES)
    label = (f"PFD POC  pattern={args.pattern}  severity={args.severity}  "
             f"device=cpu  out={pdir.relative_to(Path.cwd()) if pdir.is_absolute() else pdir}")
    url = wp.start_server(port=args.port, open_browser=not args.no_browser,
                          run_label=label, expected_total_s=None, cfg=None)

    print()
    print("=" * 72)
    print(f"  PFD POC dashboard:   {url}")
    print(f"  Main training still: http://127.0.0.1:8765/  (untouched)")
    print(f"  Pattern:             {args.pattern}  ({args.severity})")
    print(f"  Output:              {pdir}")
    print("=" * 72)
    print()

    t_total = time.time()
    try:
        # ============ Stage 1: pick volume ============
        wp.set_stage("pfd_pick_volume", total=0, postfix="reading labels csv")
        labels_csv = Path(cfg["paths"]["labels_csv"])
        # POC: hilly severity picks a hilly real volume so the deformation has
        # a non-trivial baseline. Plain severity picks a plain volume.
        info = pick_source_volume(labels_csv, severity=args.severity)
        wp.update_stage(postfix=f"picked {info['uid']}  inlet={info['pelvic_inlet_mm']:.1f}mm")
        wp.log_msg(f"source: {info['uid']}  region={info['region']}  inlet={info['pelvic_inlet_mm']:.1f}mm")
        wp.finish_stage("done")

        # ============ Stage 2: load cached volume ============
        wp.set_stage("pfd_load", total=0, postfix=f"loading {info['cache_path']}")
        vol_norm = load_volume(Path(info["cache_path"]))
        Z, H, W = vol_norm.shape
        wp.update_stage(postfix=f"loaded  Z={Z}  H={H}  W={W}")
        wp.log_msg(f"loaded volume shape={vol_norm.shape}")
        wp.finish_stage("done")

        # ============ Stage 3: build displacement field ============
        wp.set_stage("pfd_build_field", total=0, postfix=f"pattern={args.pattern}")
        pattern = build_pattern(args.pattern, (Z, H, W), severity=args.severity)
        field = build_displacement_field((Z, H, W), pattern.blobs)
        peak = float(np.max(np.linalg.norm(field, axis=0)))
        wp.update_stage(postfix=f"{len(pattern.blobs)} blob(s), peak displacement {peak:.1f} vox")
        wp.log_msg(f"findings: {json.dumps(pattern.findings)}")
        wp.finish_stage("done")

        # ============ Stage 4: apply deformation ============
        wp.set_stage("pfd_deform", total=0, postfix="warping volume")
        t0 = time.time()
        deformed = apply_deformation(vol_norm, field, order=1, cval=-1.0)
        wp.update_stage(postfix=f"warped in {time.time()-t0:.1f}s")
        wp.finish_stage("done")

        # ============ Stage 5: CVAE refine ============
        cvae_mcfg = cfg["model"]["cvae"]
        wp.set_stage("pfd_refine", total=Z, postfix=f"loading {ckpt_path.name}")
        cvae = CVAE(
            in_channels=cvae_mcfg["in_channels"], base_channels=cvae_mcfg["base_channels"],
            latent_channels=cvae_mcfg["latent_channels"], latent_size=cvae_mcfg["latent_size"],
            cond_dim=cvae_mcfg["cond_dim"],
        ).eval()
        state = torch.load(ckpt_path, map_location="cpu")
        cvae.load_state_dict(state["model"])
        wp.log_msg(f"loaded CVAE from {ckpt_path.name}")

        last = [0.0]
        def on_slice(done, total):
            wp.update_stage(current=done, total=total)
            # rate-limit log messages so we don't spam
            now = time.time()
            if now - last[0] > 3.0:
                last[0] = now
                wp.log_msg(f"refine: {done}/{total} slices")

        t0 = time.time()
        refined = refine_with_cvae(cvae, deformed, info["region_id"],
                                   batch=args.batch, on_slice=on_slice)
        wp.update_stage(current=Z, total=Z,
                        postfix=f"refined {Z} slices in {(time.time()-t0)/60:.1f} min")
        wp.finish_stage("done")

        # ============ Stage 6: write DICOM + comparison PNG ============
        wp.set_stage("pfd_write_dicom", total=0, postfix="serializing CT series")
        hu_min = float(cfg["preprocess"]["hu_min"])
        hu_max = float(cfg["preprocess"]["hu_max"])
        vol_hu = np.clip(unwindow(refined, hu_min, hu_max), -1024.0, 3071.0).astype(np.int16)
        spacing = (float(cfg["output"]["slice_thickness_mm"]),
                   float(cfg["output"]["pixel_spacing_mm"][0]),
                   float(cfg["output"]["pixel_spacing_mm"][1]))
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
        # Side-by-side comparison PNG
        comp_path = pdir / "COMPARISON.png"
        save_comparison_png(vol_norm, deformed, refined,
                            hu_min, hu_max, comp_path)
        # Augment metadata with PFD findings
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        meta.update({
            "synthetic_strategy": "hybrid_deformation_poc",
            "source_volume_uid": info["uid"],
            "source_dataset":    info["dataset"],
            "pfd_pattern":       pattern.name,
            "pfd_severity":      pattern.severity,
            "pfd_findings":      pattern.findings,
            "cvae_checkpoint":   ckpt_path.name,
            "notes": ("POC: deformation field is parametric (no per-volume segmentation). "
                      "Phase-1 will anchor the field to TotalSegmentator masks."),
        })
        meta_path.write_text(json.dumps(meta, indent=2))
        wp.update_stage(postfix=f"wrote {Z} DICOM slices + PNG/JPG/metadata")
        wp.log_msg(f"output: {pdir}")
        wp.finish_stage("done")

        total_min = (time.time() - t_total) / 60.0
        wp.log_msg(f"PFD POC complete in {total_min:.1f} min")
        print()
        print("=" * 72)
        print(f"  DONE in {total_min:.1f} min")
        print(f"  DICOM:      {dicom_dir}")
        print(f"  Comparison: {comp_path}   (original | deformed | refined)")
        print(f"  Metadata:   {meta_path}")
        print("=" * 72)
        wp.stop_server(grace_s=60.0)

    except Exception as e:
        wp.log_msg(f"ERROR: {e}")
        wp.finish_stage("error", error=str(e))
        wp.stop_server(grace_s=30.0)
        raise


if __name__ == "__main__":
    main()
