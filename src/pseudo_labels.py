"""Plain vs Hilly pseudo-labels derived from pelvic morphometry.

The dataset has no real biographical region label. We approximate one by
clustering volumes on morphometric proxies (pelvic_inlet_mm, iliac_flare,
sacral_tilt) computed in `preprocessing.compute_morphometry`. KMeans (k=2)
produces two anatomy groups; we name the broader-pelvis cluster "plain" and
the narrower one "hilly". The label is repeatable for a given seed but is
an anatomical proxy, NOT a real region label -- see README.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


FEATURES = ["pelvic_inlet_mm", "iliac_flare", "sacral_tilt"]


def make_pseudo_labels(cfg: dict) -> dict:
    """Read cache manifest, cluster morphometry, write labels CSV.

    Returns a dict with cluster centroids and the path to the CSV.
    """
    paths = cfg["paths"]
    cache_dir = Path(paths["cache_dir"])
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run build_cache first; missing {manifest_path}")

    rows = json.loads(manifest_path.read_text())
    rows = [r for r in rows if r.get("status") in ("ok", "cached")]
    if not rows:
        raise RuntimeError("Manifest has no successfully-cached volumes.")

    # Some rows (status='cached' from earlier runs) may not have morph_ fields.
    # Read morphometry directly from the .npz for those.
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
    X = df[FEATURES].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    with Spinner("Fitting KMeans (plain vs hilly)"):
        Xs = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=cfg["pseudo_labels"]["n_clusters"],
                    random_state=cfg["pseudo_labels"]["random_state"],
                    n_init=10)
        clusters = km.fit_predict(Xs)

    # Name clusters: the one with the larger mean pelvic_inlet_mm -> "plain".
    centroids_inlet = []
    for c in range(km.n_clusters):
        mask = clusters == c
        centroids_inlet.append(df.loc[mask, "pelvic_inlet_mm"].mean())
    plain_cluster = int(np.argmax(centroids_inlet))
    labels = np.where(clusters == plain_cluster, "plain", "hilly")

    df["cluster"] = clusters
    df["region"] = labels
    df["region_id"] = (df["region"] == "hilly").astype(int)   # plain=0, hilly=1

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
        "centroids": centroids_df,
    }
