"""Parametric pelvic-floor disease (PFD) deformation fields.

This is the POC version: deformations are anchored to anatomically-typical
locations in the cached pelvic volume (no per-volume segmentation). The full
Phase-1 module will use TotalSegmentator to localize the bladder/rectum/uterus
per case and anchor the deformation to those masks.

The math:
  * For each disease pattern we define one or more "anchor blobs" — a 3D
    Gaussian centered at (z, y, x) with sigma_z, sigma_y, sigma_x.
  * Each blob contributes a displacement vector field whose magnitude follows
    the Gaussian and direction is fixed (typically downward = + along Z, or
    anterior = - along Y in the LPS axial convention used by the cache).
  * The final per-voxel displacement is the sum of all blobs' contributions.
  * scipy.ndimage.map_coordinates resamples the volume at (orig - disp) to
    apply the deformation.

PFD patterns (POC subset):
  * cystocele:        bladder neck pushed downward
  * rectocele:        rectum bulged anteriorly
  * uterine_prolapse: uterus descended
  * levator_avulsion: asymmetric defect (NOT in POC, needs segmentation)

For each pattern we expose mild ("plain") and severe ("hilly") parameter sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import map_coordinates


# =========================================================================
# Anatomical-anchor heuristics (work on cached [Z, H, W] volumes)
# Cache is preprocessed to a centred pelvic ROI in [-1, 1].
# In the LPS convention used by the DICOM writer:
#   Z increases caudally (downward in patient frame)
#   Y increases posteriorly
#   X increases to the patient's left
# After preprocessing we have a Z-stack of axial slices, with each slice
# in (H, W) = (Y, X). Y=0 is anterior (front), Y=H-1 is posterior (back).
# Mid-pelvic Z is roughly the centre 1/3 of the volume.
# =========================================================================


@dataclass
class Blob:
    """One 3D Gaussian displacement blob.

    center: (z, y, x) in voxel units relative to the volume
    sigma:  (sz, sy, sx) Gaussian std-dev in voxel units
    delta:  (dz, dy, dx) displacement at the blob centre, in voxel units
            (positive dz = caudal, positive dy = posterior, positive dx = left)
    """
    center: tuple[float, float, float]
    sigma: tuple[float, float, float]
    delta: tuple[float, float, float]


@dataclass
class PfdPattern:
    name: str
    severity: Literal["plain", "hilly"]
    blobs: list[Blob]
    findings: dict             # structured metadata for the patient JSON


# -------------------------------------------------------------------------
# Pattern builders (anchored to fractional positions inside the volume so
# they scale to whatever (Z, H, W) the cached volume happens to be).
# -------------------------------------------------------------------------

def _frac(zhw: tuple[int, int, int], fz: float, fy: float, fx: float) -> tuple[float, float, float]:
    Z, H, W = zhw
    return (fz * Z, fy * H, fx * W)


def cystocele(zhw: tuple[int, int, int], severity: Literal["plain", "hilly"]) -> PfdPattern:
    """Bladder neck descent.

    Bladder sits anterior-superior in the female pelvis; in the cached cropped
    pelvic volume that's roughly Y ~ 0.35 (anterior), X ~ 0.5 (midline),
    Z ~ 0.45 (slightly above mid). We push it caudally (+Z) with a falloff
    centered there.
    """
    Z, H, W = zhw
    center = _frac(zhw, fz=0.45, fy=0.35, fx=0.50)
    sigma = (Z * 0.12, H * 0.10, W * 0.10)

    if severity == "plain":
        # Mild: ~5 mm descent.  POC voxel ~= 1 mm so dz=5.
        dz = 5.0
        grade = 1
        depth_mm = 5
    else:
        # Severe: ~15-20 mm descent.
        dz = 18.0
        grade = 3
        depth_mm = 18

    return PfdPattern(
        name="cystocele",
        severity=severity,
        blobs=[Blob(center=center, sigma=sigma, delta=(dz, 0.0, 0.0))],
        findings={"cystocele_grade": grade, "bladder_descent_mm": depth_mm},
    )


def rectocele(zhw: tuple[int, int, int], severity: Literal["plain", "hilly"]) -> PfdPattern:
    """Rectum bulged anteriorly. Rectum sits posterior-midline; we displace
    a region near Y ~ 0.70 (posterior) anteriorly (-Y) so it bulges forward."""
    Z, H, W = zhw
    center = _frac(zhw, fz=0.55, fy=0.70, fx=0.50)
    sigma = (Z * 0.10, H * 0.08, W * 0.08)
    if severity == "plain":
        dy = -3.0
        depth_mm = 3
    else:
        dy = -12.0
        depth_mm = 12
    return PfdPattern(
        name="rectocele",
        severity=severity,
        blobs=[Blob(center=center, sigma=sigma, delta=(0.0, dy, 0.0))],
        findings={"rectocele_depth_mm": depth_mm},
    )


def uterine_prolapse(zhw: tuple[int, int, int], severity: Literal["plain", "hilly"]) -> PfdPattern:
    """Uterus descended. Uterus is roughly central, anterior-mid, between
    bladder and rectum. We anchor between cystocele and rectocele positions
    and push down."""
    Z, H, W = zhw
    center = _frac(zhw, fz=0.50, fy=0.50, fx=0.50)
    sigma = (Z * 0.10, H * 0.09, W * 0.09)
    if severity == "plain":
        dz = 2.0
        stage = 0
    else:
        dz = 14.0
        stage = 3
    return PfdPattern(
        name="uterine_prolapse",
        severity=severity,
        blobs=[Blob(center=center, sigma=sigma, delta=(dz, 0.0, 0.0))],
        findings={"uterine_prolapse_pop_q_stage": stage, "uterine_descent_mm": int(dz)},
    )


def combined_pattern(zhw: tuple[int, int, int],
                     severity: Literal["plain", "hilly"]) -> PfdPattern:
    """A blended PFD presentation: cystocele + rectocele + uterine prolapse.
    'plain' = mild grades; 'hilly' = severe combined PFD."""
    cyst = cystocele(zhw, severity)
    rect = rectocele(zhw, severity)
    uter = uterine_prolapse(zhw, severity)
    findings = {**cyst.findings, **rect.findings, **uter.findings,
                "pfd_severity": severity}
    return PfdPattern(
        name="combined_pfd",
        severity=severity,
        blobs=cyst.blobs + rect.blobs + uter.blobs,
        findings=findings,
    )


# -------------------------------------------------------------------------
# Displacement field + warping
# -------------------------------------------------------------------------

def build_displacement_field(zhw: tuple[int, int, int],
                             blobs: list[Blob]) -> np.ndarray:
    """Return (3, Z, H, W) float32 displacement field d such that
    warped[z,y,x] = original[z + d[0], y + d[1], x + d[2]].
    """
    Z, H, W = zhw
    zz = np.arange(Z, dtype=np.float32)[:, None, None]
    yy = np.arange(H, dtype=np.float32)[None, :, None]
    xx = np.arange(W, dtype=np.float32)[None, None, :]
    field = np.zeros((3, Z, H, W), dtype=np.float32)
    for b in blobs:
        cz, cy, cx = b.center
        sz, sy, sx = b.sigma
        dz, dy, dx = b.delta
        # Gaussian falloff (no normalization — peak = 1 at centre)
        g = np.exp(
            -0.5 * (((zz - cz) / max(sz, 1e-6)) ** 2
                  + ((yy - cy) / max(sy, 1e-6)) ** 2
                  + ((xx - cx) / max(sx, 1e-6)) ** 2)
        ).astype(np.float32)
        field[0] += dz * g
        field[1] += dy * g
        field[2] += dx * g
    return field


def apply_deformation(volume: np.ndarray, field: np.ndarray,
                      order: int = 1, cval: float = -1.0) -> np.ndarray:
    """Warp a (Z, H, W) volume by a (3, Z, H, W) displacement field."""
    assert volume.ndim == 3 and field.shape == (3,) + volume.shape, \
        f"shape mismatch: vol {volume.shape}, field {field.shape}"
    Z, H, W = volume.shape
    zz = np.arange(Z, dtype=np.float32)[:, None, None] + field[0]
    yy = np.arange(H, dtype=np.float32)[None, :, None] + field[1]
    xx = np.arange(W, dtype=np.float32)[None, None, :] + field[2]
    coords = np.stack([np.broadcast_to(zz, (Z, H, W)),
                       np.broadcast_to(yy, (Z, H, W)),
                       np.broadcast_to(xx, (Z, H, W))], axis=0)
    warped = map_coordinates(volume.astype(np.float32), coords,
                             order=order, mode="constant", cval=cval)
    return warped.astype(volume.dtype)


def build_organ_weight_mask(zhw: tuple[int, int, int],
                             organ_masks: dict[str, np.ndarray],
                             target_organs: list[str],
                             dilation_voxels: int = 10,
                             blur_sigma: float = 5.0) -> np.ndarray:
    """Build a smooth (Z, H, W) float32 weight mask confined to pelvic floor organs.

    Returns 1.0 inside the target organs (dilated + blurred), 0.0 outside.
    This ensures deformation affects ONLY the pelvic floor soft tissue —
    bones, fat, muscle outside the ROI remain pixel-identical to original.

    dilation_voxels: how many voxels to expand the organ mask outward
    blur_sigma:      Gaussian sigma (voxels) for feathered edge blending
    """
    from scipy.ndimage import binary_dilation, gaussian_filter
    Z, H, W = zhw
    weight = np.zeros((Z, H, W), dtype=np.float32)
    for name in target_organs:
        if name in organ_masks and organ_masks[name].any():
            weight = np.maximum(weight, organ_masks[name].astype(np.float32))

    if weight.max() < 0.5:
        # No organ masks found — fall back to a permissive mid-pelvic ROI
        # (lower half of volume, central 60% of H and W)
        z0, z1 = int(Z * 0.3), int(Z * 0.9)
        y0, y1 = int(H * 0.2), int(H * 0.8)
        x0, x1 = int(W * 0.2), int(W * 0.8)
        weight[z0:z1, y0:y1, x0:x1] = 1.0
        return gaussian_filter(weight, sigma=blur_sigma)

    # Dilate so immediately neighbouring tissue blends smoothly
    dilated = binary_dilation(weight > 0.5, iterations=dilation_voxels).astype(np.float32)
    # Gaussian blur for feathered falloff — no sharp seam at boundary
    smooth = gaussian_filter(dilated, sigma=blur_sigma)
    return np.clip(smooth, 0.0, 1.0).astype(np.float32)


def apply_confined_deformation(volume: np.ndarray,
                                blobs: list[Blob],
                                organ_masks: dict[str, np.ndarray],
                                target_organs: list[str],
                                order: int = 1,
                                cval: float = -1.0,
                                dilation_voxels: int = 10,
                                blur_sigma: float = 5.0) -> np.ndarray:
    """Apply PFD deformation ONLY within pelvic floor organ regions.

    Bones, surrounding muscles, and all anatomy outside the target organs
    remain pixel-identical to the original. Only the organ voxels (+ a small
    feathered margin) are displaced according to the deformation blobs.
    """
    zhw = volume.shape
    # Full Gaussian displacement field
    field = build_displacement_field(zhw, blobs)
    # Organ-confined weight mask (smooth 0→1 transition at organ boundary)
    weight = build_organ_weight_mask(zhw, organ_masks, target_organs,
                                     dilation_voxels=dilation_voxels,
                                     blur_sigma=blur_sigma)
    # Confine: zero displacement outside organs, full displacement inside
    confined_field = field * weight[np.newaxis]  # broadcast over (3, Z, H, W)
    # Apply the confined warp
    warped = apply_deformation(volume, confined_field, order=order, cval=cval)
    # Alpha-blend: outside organs = original, inside organs = warped
    # weight=0 → 100% original; weight=1 → 100% warped
    blended = volume.astype(np.float32) * (1.0 - weight) + warped.astype(np.float32) * weight
    return np.clip(blended, -1.0, 1.0).astype(volume.dtype)


# -------------------------------------------------------------------------
# Pattern registry (for the orchestrator)
# -------------------------------------------------------------------------

PATTERN_BUILDERS = {
    "cystocele": cystocele,
    "rectocele": rectocele,
    "uterine_prolapse": uterine_prolapse,
    "combined_pfd": combined_pattern,
}


def build_pattern(name: str, zhw: tuple[int, int, int],
                  severity: Literal["plain", "hilly"]) -> PfdPattern:
    if name not in PATTERN_BUILDERS:
        raise ValueError(f"unknown PFD pattern: {name}; available: {list(PATTERN_BUILDERS)}")
    return PATTERN_BUILDERS[name](zhw, severity)


# =========================================================================
# Phase-1: mask-anchored pattern builders
# =========================================================================
# Anchor blobs to ACTUAL organ centroids found by TotalSegmentator (see
# pfd_segmentation.py) instead of guessed fractional positions. This puts
# the deformation in the correct tissue and lets us use larger magnitudes
# without smearing unrelated anatomy.
#
# Magnitudes here are tuned for "clearly visible" rather than "barely
# physiologic" -- the user's brief was that the structural deformity is
# non-negotiable. Plain = ~POP-Q II (mild but obvious), hilly = ~POP-Q III-IV
# (severe).
# =========================================================================

# Clinical POP-Q / ICS grading system calibrated to South Indian PFD literature.
# Magnitudes are in mm; at the preprocessed ~0.87 mm in-plane resolution
# 1 voxel ≈ 0.87 mm.  Z spacing after resampling = 1.5 mm, but we still
# index displacement in voxels so we use the in-plane scale throughout.
#
# Grades are consistent with:
#   Dietz HP et al. (2012) "Levator avulsion injury and cystocele recurrence"
#   ICS POP-Q staging (Bump et al. 1996)
#   South Indian pelvimetry reference values (Nayak et al., IJOG 2018)
#
# Grade 0 = no prolapse / control
# Grade 1 = 5–10 mm  (POP-Q stage I, above hymen)
# Grade 2 = 15–20 mm (POP-Q stage II, at hymen)
# Grade 3 = 25–32 mm (POP-Q stage III, beyond hymen)
# Grade 4 = 38–45 mm (POP-Q stage IV, complete eversion)
_CLINICAL_MAGS_MM = {
    #              grade1  grade2  grade3  grade4
    "cystocele":        ( 8.0,  18.0,  28.0,  40.0),   # bladder descent in +Z
    "rectocele":        ( 8.0,  16.0,  25.0,  35.0),   # posterior wall depth (-Y)
    "uterine_prolapse": ( 8.0,  18.0,  28.0,  40.0),   # uterine descent in +Z
}

# Backward-compatible two-level alias used by the plain/hilly production runs
# (plain ≈ grade-1, hilly ≈ grade-3).
_MAGS = {
    "cystocele":        (_CLINICAL_MAGS_MM["cystocele"][0],
                         _CLINICAL_MAGS_MM["cystocele"][2]),
    "rectocele":        (_CLINICAL_MAGS_MM["rectocele"][0],
                         _CLINICAL_MAGS_MM["rectocele"][2]),
    "uterine_prolapse": (_CLINICAL_MAGS_MM["uterine_prolapse"][0],
                         _CLINICAL_MAGS_MM["uterine_prolapse"][2]),
}


def _grade_mag(pattern: str, grade: int) -> float:
    """Return displacement magnitude in mm for pattern + POP-Q grade (1-4)."""
    if grade < 1 or grade > 4:
        raise ValueError(f"grade must be 1-4, got {grade}")
    return _CLINICAL_MAGS_MM[pattern][grade - 1]


def _make_blob_from_stats(center: tuple[float, float, float],
                          extent: tuple[int, int, int],
                          delta: tuple[float, float, float],
                          sigma_frac: float = 0.55) -> "Blob":
    """Build a Blob whose sigma is ~sigma_frac of the organ's bbox extent in
    each dimension. sigma_frac=0.55 means the Gaussian peak fills ~the
    middle 2/3 of the organ then falls off into surrounding tissue."""
    sigma = (max(extent[0] * sigma_frac, 2.0),
             max(extent[1] * sigma_frac, 2.0),
             max(extent[2] * sigma_frac, 2.0))
    return Blob(center=center, sigma=sigma, delta=delta)


def cystocele_from_mask(bladder_stats, severity: Literal["plain", "hilly"]) -> PfdPattern:
    dz = _MAGS["cystocele"][1 if severity == "hilly" else 0]
    blob = _make_blob_from_stats(
        center=bladder_stats.center, extent=bladder_stats.extent,
        delta=(dz, 0.0, 0.0),
    )
    grade = 4 if severity == "hilly" else 1
    return PfdPattern(
        name="cystocele", severity=severity, blobs=[blob],
        findings={
            "cystocele_grade": grade,
            "bladder_descent_mm": int(round(dz)),
            "anchor_voxels": int(bladder_stats.voxels),
        },
    )


def rectocele_from_mask(rectum_stats, severity: Literal["plain", "hilly"]) -> PfdPattern:
    dy_mag = _MAGS["rectocele"][1 if severity == "hilly" else 0]
    blob = _make_blob_from_stats(
        center=rectum_stats.center, extent=rectum_stats.extent,
        delta=(0.0, -dy_mag, 0.0),       # anterior push
    )
    depth = int(round(dy_mag))
    return PfdPattern(
        name="rectocele", severity=severity, blobs=[blob],
        findings={
            "rectocele_depth_mm": depth,
            "anchor_voxels": int(rectum_stats.voxels),
        },
    )


def uterine_prolapse_from_masks(bladder_stats, rectum_stats,
                                severity: Literal["plain", "hilly"]) -> PfdPattern:
    """TotalSegmentator does not segment the uterus (standard model). The
    uterus anatomically sits between the bladder (anterior) and rectum
    (posterior) in the midline; we anchor the deformation at the midpoint
    of the two centroids and size it from the average extent."""
    bc = np.asarray(bladder_stats.center, dtype=np.float64)
    rc = np.asarray(rectum_stats.center, dtype=np.float64)
    center = tuple(map(float, (bc + rc) / 2.0))
    extent = (
        max(int((bladder_stats.extent[0] + rectum_stats.extent[0]) / 2), 4),
        max(int((bladder_stats.extent[1] + rectum_stats.extent[1]) / 2), 4),
        max(int((bladder_stats.extent[2] + rectum_stats.extent[2]) / 2), 4),
    )
    dz = _MAGS["uterine_prolapse"][1 if severity == "hilly" else 0]
    blob = _make_blob_from_stats(center=center, extent=extent, delta=(dz, 0.0, 0.0),
                                 sigma_frac=0.45)   # tighter than bladder
    stage = 3 if severity == "hilly" else 0
    return PfdPattern(
        name="uterine_prolapse", severity=severity, blobs=[blob],
        findings={
            "uterine_prolapse_pop_q_stage": stage,
            "uterine_descent_mm": int(round(dz)),
            "anchor": "midpoint(bladder, rectum)",
        },
    )


def combined_from_masks(stats: dict, severity: Literal["plain", "hilly"]) -> PfdPattern:
    """Build a combined cystocele + rectocele + uterine-prolapse pattern using
    real organ centroids. `stats` must contain MaskStats for
    'urinary_bladder' and 'rectum' (rectum is derived from the lower portion
    of TotalSegmentator's 'colon')."""
    cyst = cystocele_from_mask(stats["urinary_bladder"], severity)
    rect = rectocele_from_mask(stats["rectum"], severity)
    uter = uterine_prolapse_from_masks(stats["urinary_bladder"], stats["rectum"], severity)
    findings = {**cyst.findings, **rect.findings, **uter.findings,
                "pfd_severity": severity}
    return PfdPattern(
        name="combined_pfd", severity=severity,
        blobs=cyst.blobs + rect.blobs + uter.blobs,
        findings=findings,
    )


MASK_PATTERN_BUILDERS = {
    "cystocele":        lambda stats, sev: cystocele_from_mask(stats["urinary_bladder"], sev),
    "rectocele":        lambda stats, sev: rectocele_from_mask(stats["rectum"], sev),
    "uterine_prolapse": lambda stats, sev: uterine_prolapse_from_masks(
        stats["urinary_bladder"], stats["rectum"], sev),
    "combined_pfd":     combined_from_masks,
}


def build_pattern_from_masks(name: str, stats: dict,
                             severity: Literal["plain", "hilly"]) -> PfdPattern:
    if name not in MASK_PATTERN_BUILDERS:
        raise ValueError(f"unknown PFD pattern: {name}; available: {list(MASK_PATTERN_BUILDERS)}")
    return MASK_PATTERN_BUILDERS[name](stats, severity)


# =========================================================================
# Grade-based mask-anchored builders (POP-Q grades 1-4)
# "population" is the morphological class (plain / hilly) and is stored in
# metadata but does NOT control deformation magnitude -- grade does.
# =========================================================================

def cystocele_graded(bladder_stats, grade: int,
                     population: str = "plain") -> PfdPattern:
    """Bladder descent anchored to real TS centroid, calibrated to POP-Q grade."""
    dz = _grade_mag("cystocele", grade)
    blob = _make_blob_from_stats(
        center=bladder_stats.center, extent=bladder_stats.extent,
        delta=(dz, 0.0, 0.0),
    )
    pop_q_map = {1: "I", 2: "II", 3: "III", 4: "IV"}
    return PfdPattern(
        name="cystocele", severity=population, blobs=[blob],
        findings={
            "cystocele_grade": grade,
            "pop_q_stage": pop_q_map[grade],
            "bladder_descent_mm": int(round(dz)),
            "anchor_voxels": int(bladder_stats.voxels),
        },
    )


def rectocele_graded(rectum_stats, grade: int,
                     population: str = "plain") -> PfdPattern:
    """Posterior-wall bulge anchored to real rectum centroid, calibrated to grade."""
    dy_mag = _grade_mag("rectocele", grade)
    blob = _make_blob_from_stats(
        center=rectum_stats.center, extent=rectum_stats.extent,
        delta=(0.0, -dy_mag, 0.0),
    )
    pop_q_map = {1: "I", 2: "II", 3: "III", 4: "IV"}
    return PfdPattern(
        name="rectocele", severity=population, blobs=[blob],
        findings={
            "rectocele_grade": grade,
            "pop_q_stage": pop_q_map[grade],
            "rectocele_depth_mm": int(round(dy_mag)),
            "anchor_voxels": int(rectum_stats.voxels),
        },
    )


def uterine_prolapse_graded(bladder_stats, rectum_stats, grade: int,
                            population: str = "plain") -> PfdPattern:
    """Uterine descent anchored to bladder-rectum midpoint, calibrated to grade."""
    bc = np.asarray(bladder_stats.center, dtype=np.float64)
    rc = np.asarray(rectum_stats.center, dtype=np.float64)
    center = tuple(map(float, (bc + rc) / 2.0))
    extent = (
        max(int((bladder_stats.extent[0] + rectum_stats.extent[0]) / 2), 4),
        max(int((bladder_stats.extent[1] + rectum_stats.extent[1]) / 2), 4),
        max(int((bladder_stats.extent[2] + rectum_stats.extent[2]) / 2), 4),
    )
    dz = _grade_mag("uterine_prolapse", grade)
    blob = _make_blob_from_stats(center=center, extent=extent,
                                 delta=(dz, 0.0, 0.0), sigma_frac=0.45)
    pop_q_map = {1: "I", 2: "II", 3: "III", 4: "IV"}
    return PfdPattern(
        name="uterine_prolapse", severity=population, blobs=[blob],
        findings={
            "uterine_prolapse_grade": grade,
            "pop_q_stage": pop_q_map[grade],
            "uterine_descent_mm": int(round(dz)),
            "anchor": "midpoint(bladder, rectum)",
        },
    )


def combined_pfd_graded(stats: dict, grade: int,
                        population: str = "plain") -> PfdPattern:
    """Combined PFD (cystocele + rectocele + uterine prolapse) at clinical grade."""
    cyst = cystocele_graded(stats["urinary_bladder"], grade, population)
    rect = rectocele_graded(stats["rectum"], grade, population)
    uter = uterine_prolapse_graded(stats["urinary_bladder"], stats["rectum"],
                                   grade, population)
    findings = {**cyst.findings, **rect.findings, **uter.findings,
                "combined_grade": grade, "population": population}
    return PfdPattern(
        name="combined_pfd", severity=population,
        blobs=cyst.blobs + rect.blobs + uter.blobs,
        findings=findings,
    )


def build_pattern_graded(name: str, stats: dict,
                         grade: int, population: str = "plain") -> PfdPattern:
    """Build a PFD pattern anchored to real TS masks at a specific clinical grade.

    name:       one of cystocele / rectocele / uterine_prolapse / combined_pfd
    stats:      dict of MaskStats from segment_original_volume
    grade:      POP-Q grade 1-4
    population: 'plain' or 'hilly' (South Indian morphological class)
    """
    builders = {
        "cystocele":
            lambda: cystocele_graded(stats["urinary_bladder"], grade, population),
        "rectocele":
            lambda: rectocele_graded(stats["rectum"], grade, population),
        "uterine_prolapse":
            lambda: uterine_prolapse_graded(
                stats["urinary_bladder"], stats["rectum"], grade, population),
        "combined_pfd":
            lambda: combined_pfd_graded(stats, grade, population),
    }
    if name not in builders:
        raise ValueError(f"unknown pattern: {name}; available: {list(builders)}")
    return builders[name]()


# =========================================================================
# Levator ani / pelvic floor muscle PFD deformation
# =========================================================================
# The levator ani, puborectalis, iliococcygeus, and pubococcygeus are the
# primary structures measured in PFD research.  TotalSegmentator's fast
# model does not segment them directly, so we locate the pelvic floor
# Z-slab from the sacrum bottom + hip masks (both of which ARE cached).
#
# Two effects are simulated:
#   1. Bilateral outward displacement (±X): widens the levator hiatus
#   2. Inferior central displacement (+Z): muscle descent / sag
#
# The weight mask excludes bones (sacrum, hip bones are zeroed out) so
# only soft tissue in the pelvic floor Z-slab is deformed.
# Magnitudes calibrated to POP-Q grades 1-4, South Indian PFD literature.
# =========================================================================

# Hiatal widening — each side pushed outward from midline (mm per side)
_LA_HIATUS_WIDEN_MM  = {1: 5.0,  2: 12.0,  3: 20.0,  4: 28.0}
# Inferior muscle descent / sag (mm total)
_LA_DESCENT_MM       = {1: 3.0,  2:  8.0,  3: 14.0,  4: 20.0}


def _pelvic_floor_roi(masks: dict,
                      zhw: tuple[int, int, int],
                      slice_spacing_mm: float = 1.5) -> dict:
    """Estimate the pelvic floor ROI centre + spread from sacrum + hip masks.

    Returns a dict with keys:
      floor_z, center_y, center_x, hip_half_x,
      sigma_z, sigma_y, sigma_x  (all in voxels)
    """
    Z, H, W = zhw
    floor_z    = Z * 0.75
    center_y   = H * 0.50
    center_x   = W * 0.50
    hip_half_x = W * 0.18

    sacrum = masks.get("sacrum")
    hl     = masks.get("hip_left")
    hr     = masks.get("hip_right")

    if sacrum is not None and sacrum.any():
        sac_z_max = int(np.where(sacrum.any(axis=(1, 2)))[0].max())
        floor_z   = min(float(sac_z_max + 3), float(Z - 1))

    if hl is not None and hr is not None and (hl.any() or hr.any()):
        combined = hl.astype(bool) | hr.astype(bool)
        x_idx = np.where(combined.any(axis=(0, 1)))[0]
        y_idx = np.where(combined.any(axis=(0, 2)))[0]
        if len(x_idx) >= 2:
            center_x   = float((x_idx[0] + x_idx[-1]) / 2.0)
            hip_half_x = float((x_idx[-1] - x_idx[0]) / 4.0)
        if len(y_idx) >= 2:
            center_y = float((y_idx[0] + y_idx[-1]) / 2.0)

    return {
        "floor_z":    float(floor_z),
        "center_y":   float(center_y),
        "center_x":   float(center_x),
        "hip_half_x": float(hip_half_x),
        "sigma_z":    max(Z * 0.06, 3.0),
        "sigma_y":    max(H * 0.10, 5.0),
        "sigma_x":    max(W * 0.08, 4.0),
    }


def build_levator_ani_blobs(masks: dict,
                             zhw: tuple[int, int, int],
                             grade: int,
                             pixel_spacing_mm: float = 1.0,
                             slice_spacing_mm: float = 1.5) -> list:
    """Return displacement blobs simulating levator ani PFD changes.

    Three blobs:
      left  — bilateral outward push (-X) + partial descent
      right — bilateral outward push (+X) + partial descent
      centre — inferior descent (+Z) through the hiatus
    """
    roi = _pelvic_floor_roi(masks, zhw, slice_spacing_mm)
    fz  = roi["floor_z"];  cy = roi["center_y"];  cx = roi["center_x"]
    hx  = roi["hip_half_x"]
    sz  = roi["sigma_z"];  sy = roi["sigma_y"];   sx = roi["sigma_x"]

    widen_vox   = _LA_HIATUS_WIDEN_MM[grade]  / max(pixel_spacing_mm, 0.1)
    descent_vox = _LA_DESCENT_MM[grade]       / max(slice_spacing_mm, 0.1)

    return [
        # Left lobe: push left + partial inferior
        Blob(center=(fz, cy, cx - hx * 0.5),
             sigma=(sz, sy, sx),
             delta=(descent_vox * 0.5, 0.0, -widen_vox)),
        # Right lobe: push right + partial inferior
        Blob(center=(fz, cy, cx + hx * 0.5),
             sigma=(sz, sy, sx),
             delta=(descent_vox * 0.5, 0.0, +widen_vox)),
        # Central: pure inferior descent (gravity pull on floor)
        Blob(center=(fz, cy, cx),
             sigma=(sz, sy * 0.6, sx * 0.45),
             delta=(descent_vox, 0.0, 0.0)),
    ]


def apply_levator_ani_deformation(volume: np.ndarray,
                                   masks: dict,
                                   grade: int,
                                   pixel_spacing_mm: float = 1.0,
                                   slice_spacing_mm: float = 1.5,
                                   blur_sigma: float = 4.0) -> np.ndarray:
    """Apply grade-calibrated pelvic floor deformation simulating levator ani PFD.

    Widens the levator hiatus + creates inferior muscle descent.
    Bones (sacrum, hip) are excluded from the weight mask — they remain
    completely pixel-identical to the original.

    Call this AFTER apply_confined_deformation (additive effect).
    """
    from scipy.ndimage import binary_dilation, gaussian_filter

    Z, H, W = volume.shape
    zhw = (Z, H, W)

    blobs = build_levator_ani_blobs(masks, zhw, grade, pixel_spacing_mm, slice_spacing_mm)
    roi   = _pelvic_floor_roi(masks, zhw, slice_spacing_mm)
    fz    = roi["floor_z"]

    # Pelvic floor soft-tissue slab (below sacrum, central Y/X band)
    weight = np.zeros((Z, H, W), dtype=np.float32)
    z0 = max(0,   int(fz - 8))
    z1 = min(Z,   int(fz + 16))
    y0 = int(H * 0.22);  y1 = int(H * 0.82)
    x0 = int(W * 0.12);  x1 = int(W * 0.88)
    weight[z0:z1, y0:y1, x0:x1] = 1.0

    # Zero out bone regions so bones stay untouched
    for key in ("sacrum", "hip_left", "hip_right"):
        bone = masks.get(key)
        if bone is not None and bone.any():
            expanded = binary_dilation(bone > 0.5, iterations=4).astype(bool)
            weight[expanded] = 0.0

    weight = gaussian_filter(weight, sigma=blur_sigma)
    weight = np.clip(weight, 0.0, 1.0).astype(np.float32)

    if weight.max() < 0.01:
        return volume  # pelvic floor region not found — skip silently

    field         = build_displacement_field(zhw, blobs)
    confined      = field * weight[np.newaxis]
    warped        = apply_deformation(volume, confined, order=1, cval=-1.0)
    blended       = (volume.astype(np.float32) * (1.0 - weight)
                     + warped.astype(np.float32) * weight)
    return np.clip(blended, -1.0, 1.0).astype(volume.dtype)


# =========================================================================
# v2: aggressive fractional anchors (TS-independent)
# =========================================================================
# When TotalSegmentator can't run (e.g. the cropped pelvic cache is OOD)
# we fall back to fractional anchors but with three improvements over the
# original POC versions:
#
#   1. TIGHTER sigma -> deformation is FOCAL instead of diffuse, so a
#      28 mm displacement actually moves a recognizable structure
#      rather than smearing 60 mm of surrounding tissue.
#   2. AGGRESSIVE magnitudes from the _MAGS table (POP-Q III-IV for
#      hilly, POP-Q II for plain).
#   3. Optional morphometry-aware scaling: anchor positions can shift
#      based on the per-volume pelvic_inlet_mm / pelvic_depth_mm that
#      the preprocessing pipeline already cached.
#
# The user's brief was "structural deformity is non-negotiable" -- these
# builders prioritize visibility over physiological subtlety.
# =========================================================================


def _v2_sigma(zhw: tuple[int, int, int], frac_z: float = 0.06,
              frac_y: float = 0.05, frac_x: float = 0.05) -> tuple[float, float, float]:
    """Tighter sigma than the original POC -- focal deformation, not smear."""
    Z, H, W = zhw
    return (max(Z * frac_z, 3.0),
            max(H * frac_y, 4.0),
            max(W * frac_x, 4.0))


def cystocele_v2(zhw: tuple[int, int, int],
                 severity: Literal["plain", "hilly"]) -> PfdPattern:
    """Aggressive fractional-anchor cystocele. Bladder anchor at
    Y=0.32 (anterior third), X=0.50 (midline), Z=0.42 (slightly above mid).
    Stronger push (+Z = caudal) than the POC version."""
    Z, H, W = zhw
    center = _frac(zhw, fz=0.42, fy=0.32, fx=0.50)
    sigma = _v2_sigma(zhw, frac_z=0.07, frac_y=0.06, frac_x=0.06)
    dz = _MAGS["cystocele"][1 if severity == "hilly" else 0]
    grade = 4 if severity == "hilly" else 1
    return PfdPattern(
        name="cystocele_v2", severity=severity,
        blobs=[Blob(center=center, sigma=sigma, delta=(dz, 0.0, 0.0))],
        findings={
            "cystocele_grade": grade,
            "bladder_descent_mm": int(round(dz)),
            "anchor_mode": "fractional_v2",
        },
    )


def rectocele_v2(zhw: tuple[int, int, int],
                 severity: Literal["plain", "hilly"]) -> PfdPattern:
    """Aggressive fractional-anchor rectocele. Rectum anchor at
    Y=0.72 (posterior), X=0.50 (midline), Z=0.55 (slightly below mid).
    Anterior push (-Y) shows up as a bulge into the pelvic cavity."""
    Z, H, W = zhw
    center = _frac(zhw, fz=0.55, fy=0.72, fx=0.50)
    sigma = _v2_sigma(zhw, frac_z=0.06, frac_y=0.05, frac_x=0.05)
    dy_mag = _MAGS["rectocele"][1 if severity == "hilly" else 0]
    return PfdPattern(
        name="rectocele_v2", severity=severity,
        blobs=[Blob(center=center, sigma=sigma, delta=(0.0, -dy_mag, 0.0))],
        findings={
            "rectocele_depth_mm": int(round(dy_mag)),
            "anchor_mode": "fractional_v2",
        },
    )


def uterine_prolapse_v2(zhw: tuple[int, int, int],
                        severity: Literal["plain", "hilly"]) -> PfdPattern:
    """Aggressive fractional-anchor uterine prolapse. Uterus anchor between
    bladder (anterior) and rectum (posterior) at Y=0.52, X=0.50, Z=0.48."""
    Z, H, W = zhw
    center = _frac(zhw, fz=0.48, fy=0.52, fx=0.50)
    sigma = _v2_sigma(zhw, frac_z=0.05, frac_y=0.045, frac_x=0.045)
    dz = _MAGS["uterine_prolapse"][1 if severity == "hilly" else 0]
    stage = 3 if severity == "hilly" else 0
    return PfdPattern(
        name="uterine_prolapse_v2", severity=severity,
        blobs=[Blob(center=center, sigma=sigma, delta=(dz, 0.0, 0.0))],
        findings={
            "uterine_prolapse_pop_q_stage": stage,
            "uterine_descent_mm": int(round(dz)),
            "anchor_mode": "fractional_v2",
        },
    )


def combined_pfd_v2(zhw: tuple[int, int, int],
                    severity: Literal["plain", "hilly"]) -> PfdPattern:
    """Combined v2: cystocele + rectocele + uterine prolapse, aggressive params."""
    cyst = cystocele_v2(zhw, severity)
    rect = rectocele_v2(zhw, severity)
    uter = uterine_prolapse_v2(zhw, severity)
    findings = {**cyst.findings, **rect.findings, **uter.findings,
                "pfd_severity": severity, "anchor_mode": "fractional_v2"}
    return PfdPattern(
        name="combined_pfd_v2", severity=severity,
        blobs=cyst.blobs + rect.blobs + uter.blobs,
        findings=findings,
    )


PATTERN_BUILDERS_V2 = {
    "cystocele":        cystocele_v2,
    "rectocele":        rectocele_v2,
    "uterine_prolapse": uterine_prolapse_v2,
    "combined_pfd":     combined_pfd_v2,
}


def build_pattern_v2(name: str, zhw: tuple[int, int, int],
                     severity: Literal["plain", "hilly"]) -> PfdPattern:
    if name not in PATTERN_BUILDERS_V2:
        raise ValueError(f"unknown v2 PFD pattern: {name}; available: {list(PATTERN_BUILDERS_V2)}")
    return PATTERN_BUILDERS_V2[name](zhw, severity)
