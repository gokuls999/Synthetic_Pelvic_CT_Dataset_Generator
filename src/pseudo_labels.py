"""Plain vs Hilly pseudo-labels derived from pelvic morphometry.

The dataset has no real biographical region label. We approximate one by
clustering volumes on morphometric proxies (pelvic_inlet_mm, iliac_flare,
sacral_tilt) computed in `preprocessing.compute_morphometry`.

Robustness against bad morphometry (encountered on the full 211-volume cache
where a previous KMeans put 208/3):
  * Drop volumes with unphysical pelvic_inlet_mm (<100 or >450 mm) -- these are
    scans with no detectable pelvic bone or with extreme partial-coverage.
  * Clip each feature to its 2nd-98th percentile range before scaling, so a
    handful of outliers can't dominate the clustering.
  * Use RobustScaler (median + IQR) instead of StandardScaler (mean + std).
  * Use the volume's pelvic_inlet_mm as a tie-breaker: if KMeans produces a
    badly skewed split (one cluster <10% of data), fall back to a balanced
    median-split on pelvic_inlet_mm.

The label is still a proxy, NOT a real biographical region. See README.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler


FEATURES = ["pelvic_inlet_mm", "iliac_flare", "sacral_tilt"]
MIN_INLET_MM = 100.0       # below this = no real pelvic bone detected
MAX_INLET_MM = 450.0       # above this = morphometry blew up


def _clip_to_iqr(X: np.ndarray, lo_q: float = 2, hi_q: float = 98) -> np.ndarray:
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        lo, hi = np.percentile(col, [lo_q, hi_q])
        out[:, j] = np.clip(col, lo, hi)
    return out


def make_pseudo_labels(cfg: dict) -> dict:
    """Read cache manifest, cluster morphometry, write labels CSV."""
    paths = cfg["paths"]
    cache_dir = Path(paths["cache_dir"])
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run build_cache first; missing {manifest_path}")

    rows = json.loads(manifest_path.read_text())
    rows = [r for r in rows if r.get("status") in ("ok", "cached")]
    if not rows:
        raise RuntimeError("Manifest has no successfully-cached volumes.")

    from .progress import pbar, Spinner

    feat_rows = []
    for r in pbar(rows, desc="Reading morphometry", unit="vol"):
        m = {f: r.get(f"morph_{f}") for f in FEATURES}
        if any(v is None for v in m.values()):
            try:
                npz = np.load(r["cache_path"], allow_pickle=True)
                morph = json.loads(str(npz["morphometry"]))
                m = {f: morph.get(f, 0.0) for f in FEATURES}
            except Exception:
                m = {f: 0.0 for f in FEATURES}
        feat_rows.append({"uid": r["uid"], "dataset": r["dataset"],
                          "cache_path": r["cache_path"], **m})

    df = pd.DataFrame(feat_rows)

    # Drop volumes whose morphometry is obviously broken.
    n_before = len(df)
    df = df[(df["pelvic_inlet_mm"] >= MIN_INLET_MM) &
            (df["pelvic_inlet_mm"] <= MAX_INLET_MM)].reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"[labels] dropped {n_dropped} volumes with unphysical pelvic_inlet_mm "
              f"(kept {len(df)}/{n_before})")
    if len(df) < 4:
        raise RuntimeError(f"Only {len(df)} valid volumes after morphometry filter -- "
                           f"check build_cache output")

    X = df[FEATURES].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = _clip_to_iqr(X)

    with Spinner("Fitting KMeans (plain vs hilly)"):
        Xs = RobustScaler().fit_transform(X)
        km = KMeans(n_clusters=cfg["pseudo_labels"]["n_clusters"],
                    random_state=cfg["pseudo_labels"]["random_state"],
                    n_init=20)
        clusters = km.fit_predict(Xs)

    # Sanity: if KMeans split is extreme (<10% in either cluster), fall back to
    # a balanced median-split on pelvic_inlet_mm.
    counts = np.bincount(clusters, minlength=km.n_clusters)
    min_frac = counts.min() / len(df)
    if min_frac < 0.1:
        print(f"[labels] KMeans split too skewed ({counts.tolist()}); "
              f"falling back to balanced median-split on pelvic_inlet_mm")
        median = df["pelvic_inlet_mm"].median()
        clusters = (df["pelvic_inlet_mm"].to_numpy() < median).astype(int)

    centroids_inlet = []
    for c in range(int(clusters.max()) + 1):
        mask = clusters == c
        centroids_inlet.append(df.loc[mask, "pelvic_inlet_mm"].mean())
    plain_cluster = int(np.argmax(centroids_inlet))
    labels = np.where(clusters == plain_cluster, "plain", "hilly")

    df["cluster"] = clusters
    df["region"] = labels
    df["region_id"] = (df["region"] == "hilly").astype(int)

    out_csv = Path(paths["labels_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    centroids_df = (df.groupby("region")[FEATURES].mean()
                      .round(3).to_dict(orient="index"))

    return {
        "labels_csv": str(out_csv),
        "n_total": len(df),
        "n_plain": int((df["region"] == "plain").sum()),
        "n_hilly": int((df["region"] == "hilly").sum()),
        "n_dropped": int(n_dropped),
        "centroids": centroids_df,
    }
