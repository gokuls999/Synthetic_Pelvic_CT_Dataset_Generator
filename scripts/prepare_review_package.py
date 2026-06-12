# -*- coding: utf-8 -*-
"""prepare_review_package.py
Clean image-only radiologist review deliverable.
No forms — radiologist annotates by hand.
Output: synthetic_dataset/Radiologist_Review_Package/review.html
"""
import sys, base64, json, io
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from _common import add_repo_to_path
add_repo_to_path()

import numpy as np
import pydicom
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DATASET_ROOT = Path("synthetic_dataset/PFD_Real_Dataset_350/Plain_175")
OUT_DIR      = Path("synthetic_dataset/Radiologist_Review_Package")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED = [
    ("combined_pfd",     "REAL_001_PLAIN_COMBINED_PFD_G2"),
    ("cystocele",        "REAL_003_PLAIN_CYSTOCELE_G2"),
    ("rectocele",        "REAL_005_PLAIN_RECTOCELE_G2"),
    ("uterine_prolapse", "REAL_022_PLAIN_UTERINE_PROLAPSE_G2"),
]

CLINICAL_INFO = {
    "cystocele": {
        "title": "Anterior Compartment Prolapse — Cystocele",
        "desc": "Bladder base descends through the anterior vaginal wall. Grade 2: descent to the level of the vaginal introitus. Assess bladder base position relative to the pubococcygeal line, urethrovesical junction descent, and anterior vaginal wall support.",
    },
    "rectocele": {
        "title": "Posterior Compartment Prolapse — Rectocele",
        "desc": "Anterior herniation of the rectal ampulla through the posterior vaginal wall. Grade 2: moderate anterior bulging. Assess anterior rectal wall contour, rectovaginal septal integrity, and extent of herniation.",
    },
    "uterine_prolapse": {
        "title": "Apical Compartment Prolapse — Uterine Prolapse",
        "desc": "Inferior displacement of the uterine cervix toward or through the vaginal introitus. Grade 2: cervix at the level of the hymen (POP-Q C point). Assess uterine axis, cervical position relative to ischial spines, and associated cul-de-sac changes.",
    },
    "combined_pfd": {
        "title": "Multi-Compartment Pelvic Floor Disorder — Combined PFD",
        "desc": "Simultaneous anterior (bladder), posterior (rectal), and apical (uterine) compartment involvement. Grade 2 across all compartments. The most common symptomatic presentation. Assess coordinated multi-compartment descent and levator ani hiatal changes.",
    },
}

try:
    font_sm = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
except Exception:
    font_sm = ImageFont.load_default()

LO_HU, HI_HU = -200.0, 500.0


def load_volume(dcm_dir):
    files = sorted(dcm_dir.glob("*.dcm"),
                   key=lambda f: int(pydicom.dcmread(str(f), stop_before_pixels=True)
                                     .get("InstanceNumber", 0)))
    return np.stack([pydicom.dcmread(str(f)).pixel_array.astype(np.int32) - 1024
                     for f in files])


def hu_u8(arr, lo=LO_HU, hi=HI_HU):
    return ((np.clip(arr.astype(np.float32), lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def axial_strip(vol, n=6):
    Z = vol.shape[0]
    idxs = np.linspace(int(Z * 0.2), int(Z * 0.8), n, dtype=int)
    panels = []
    for z in idxs:
        g = hu_u8(vol[z])
        img = Image.fromarray(np.stack([g, g, g], -1))
        d = ImageDraw.Draw(img)
        d.text((4, 4), f"z={z}", fill=(255, 220, 0), font=font_sm)
        panels.append(img)
    W, H = panels[0].size
    strip = Image.new("RGB", (W * n, H), (0, 0, 0))
    for i, p in enumerate(panels):
        strip.paste(p, (i * W, 0))
    return strip


def sagittal(vol):
    s = hu_u8(vol[:, :, vol.shape[2] // 2])
    return Image.fromarray(np.stack([s, s, s], -1)).rotate(90, expand=True)


def coronal(vol):
    c = hu_u8(vol[:, vol.shape[1] // 2, :])
    return Image.fromarray(np.stack([c, c, c], -1)).rotate(90, expand=True)


def b64(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


blocks = []
for pattern, pid in SELECTED:
    pdir = DATASET_ROOT / pid
    print(f"Building {pid}")
    vol  = load_volume(pdir / "DICOM")
    meta = json.loads((pdir / "metadata_real.json").read_text())

    comp = Image.open(pdir / "COMPARISON.png")
    if comp.width > 800:
        comp = comp.resize((800, int(comp.height * 800 / comp.width)), Image.LANCZOS)

    ax  = axial_strip(vol)
    sag = sagittal(vol)
    cor = coronal(vol)
    h   = ax.height
    sag = sag.resize((int(sag.width * h / sag.height), h), Image.LANCZOS)
    cor = cor.resize((int(cor.width * h / cor.height), h), Image.LANCZOS)

    info = CLINICAL_INFO[pattern]
    n    = SELECTED.index((pattern, pid)) + 1

    blocks.append(f"""
<div class="case">
  <div class="case-hdr">
    <span class="num">Case {n}</span>
    <span class="title">{info['title']}</span>
    <span class="badge">Grade 2 &mdash; Moderate &nbsp;|&nbsp; {meta['population'].title()} pelvis</span>
  </div>
  <p class="desc">{info['desc']}</p>

  <div class="img-label">Original CT &nbsp;/&nbsp; PFD-Deformed &nbsp;/&nbsp; Difference heatmap &nbsp;(3 highest-deformation slices)</div>
  <img src="data:image/png;base64,{b64(comp)}" class="img-full"/>

  <div class="img-label">Axial survey &mdash; soft-tissue window &nbsp;(inferior to superior)</div>
  <img src="data:image/png;base64,{b64(ax)}" class="img-full"/>

  <div class="mpr">
    <div>
      <div class="img-label">Sagittal MPR</div>
      <img src="data:image/png;base64,{b64(sag)}" class="img-mpr"/>
    </div>
    <div>
      <div class="img-label">Coronal MPR</div>
      <img src="data:image/png;base64,{b64(cor)}" class="img-mpr"/>
    </div>
  </div>
  <div class="patient-id">Patient ID: {pid}</div>
</div>
""")
    print(f"  done")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Synthetic PFD CT Dataset — Radiologist Review</title>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:'Segoe UI',Arial,sans-serif; font-size:13px;
        background:#fff; color:#111; padding:32px 40px; max-width:960px; margin:auto; }}
.cover {{ border-bottom:3px solid #1a3a5c; padding-bottom:24px; margin-bottom:36px; }}
.cover h1 {{ font-size:1.6rem; color:#1a3a5c; margin-bottom:6px; }}
.cover h2 {{ font-size:1rem; font-weight:400; color:#555; margin-bottom:18px; }}
.meta-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:6px 32px;
              font-size:0.83rem; color:#333; }}
.meta-grid b {{ color:#1a3a5c; }}
.cover-note {{ margin-top:18px; padding:12px 16px; background:#f0f4f9;
               border-left:4px solid #1a3a5c; font-size:0.83rem;
               line-height:1.7; color:#333; }}

.case {{ margin-bottom:48px; padding-bottom:36px;
         border-bottom:2px solid #e8ecf0; page-break-after:always; }}
.case-hdr {{ display:flex; align-items:baseline; gap:12px;
             border-left:5px solid #1a3a5c; padding-left:14px;
             margin-bottom:12px; flex-wrap:wrap; }}
.num   {{ font-size:0.72rem; font-weight:700; text-transform:uppercase;
          letter-spacing:.08em; color:#1a3a5c; white-space:nowrap; }}
.title {{ font-size:1.15rem; font-weight:700; color:#1a3a5c; }}
.badge {{ font-size:0.75rem; color:#555; white-space:nowrap; }}
.desc  {{ font-size:0.86rem; line-height:1.75; color:#333; margin-bottom:18px;
          padding:10px 14px; background:#f8f9fc; border-radius:4px; }}

.img-label {{ font-size:0.7rem; font-weight:700; text-transform:uppercase;
              letter-spacing:.07em; color:#1a3a5c; margin:14px 0 5px; }}
.img-full {{ width:100%; display:block; border:1px solid #ddd; border-radius:3px; }}
.mpr {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:4px; }}
.img-mpr {{ width:100%; display:block; border:1px solid #ddd; border-radius:3px;
            background:#000; }}
.patient-id {{ margin-top:10px; font-size:0.72rem; color:#999; }}

@media print {{
  body {{ padding:16px; }}
  .case {{ page-break-after:always; }}
}}
</style>
</head>
<body>

<div class="cover">
  <h1>Synthetic Pelvic Floor Disorder CT Dataset</h1>
  <h2>Radiologist Review &mdash; 4 Representative Cases</h2>
  <div class="meta-grid">
    <div><b>Study:</b> PFD Prevalence Study &mdash; South India Cohort</div>
    <div><b>Reference:</b> Africa PFD Epidemiological Study</div>
    <div><b>Method:</b> Real CT source + computational PFD deformation</div>
    <div><b>Source data:</b> CTPelvic1K (de-identified)</div>
    <div><b>HU range:</b> -1000 to +1500 (full diagnostic range)</div>
    <div><b>Total dataset planned:</b> 350 patients (175 Plain + 175 Hilly)</div>
  </div>
  <div class="cover-note">
    This package shows one representative Grade 2 (moderate) case per PFD pattern:
    combined multi-compartment, cystocele, rectocele, and uterine prolapse.
    Each case shows the original CT, the deformed CT, and a difference heatmap,
    followed by axial survey slices and sagittal/coronal MPR reconstructions.
  </div>
</div>

{"".join(blocks)}

</body>
</html>"""

out = OUT_DIR / "review.html"
out.write_text(html, encoding="utf-8")
print(f"\nSaved: {out}  ({out.stat().st_size // 1024} KB)")
print("Ctrl+P in browser -> Save as PDF to share.")
