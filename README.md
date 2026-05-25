# Synthetic Pelvic CT Generator

End-to-end pipeline to train a Hybrid CVAE + Latent Diffusion model on the
filtered female pelvic CT dataset (see `DATASET_HANDOFF.md`) and emit
300–400 synthetic patient studies as DICOM + PNG/JPG + metadata, validatable
in 3D Slicer. Spec lives in `gen.md`.

## Layout

```
gen_ai_ct_pelvic/
├── configs/default.yaml          # all knobs
├── ct_data_for_generator/        # input (DICOM + NIfTI + masks) — already filtered
├── cache/                        # built by build_cache.py — preprocessed .npz per series
├── checkpoints/                  # CVAE + diffusion .pt files
├── synthetic_dataset/            # final output: Patient_001/{DICOM,PNG,JPG,metadata.json}
├── src/
│   ├── io_loaders.py             # DICOM series + NIfTI loaders, dataset walkers
│   ├── preprocessing.py          # HU window, pelvic crop, axial resample, cache
│   ├── pseudo_labels.py          # Plain/Hilly via pelvic morphometry + KMeans
│   ├── dataset.py                # PyTorch Dataset over the cache
│   ├── models.py                 # CVAE, LatentUNet, DDPM schedule, DDIM sampler
│   ├── train.py                  # stage A (CVAE) + stage B (latent diffusion)
│   ├── generate.py               # sample latents → decode → slice-stacked volume
│   ├── dicom_builder.py          # volume → 3D-Slicer-loadable DICOM series
│   ├── exports.py                # PNG/JPG + metadata.json per patient
│   └── validate.py               # post-hoc DICOM sanity checks
└── scripts/                      # thin CLI wrappers around src/
```

## Pipeline

```
ct_data_for_generator/output/* ─┐
                                ├─► build_cache    → cache/*.npz + manifest.json
ct_data_for_generator/masks/* ──┘                 │
                                                  ├─► make_pseudo_labels → cache/region_labels.csv
                                                  │
                          train (stage A: CVAE)   ◄┘
                                  │
                          train (stage B: diff)
                                  │
                          generate → synthetic_dataset/Patient_NNNN/{DICOM,PNG,JPG,metadata.json}
                                  │
                          validate → synthetic_dataset/validation_report.json
                                  │
                          → open Patient_NNNN/DICOM/ in 3D Slicer
```

## Live progress dashboard

Every pipeline run opens a self-contained HTTP dashboard at
`http://127.0.0.1:8765/` in your default browser. One card per stage
(build_cache, pseudo_labels, train_cvae, train_diffusion, generate, validate)
with status icon, progress bar, percentage, postfix (epoch/loss/region),
elapsed + ETA, and a rolling log at the bottom. Refreshes every 250 ms.

No external deps -- pure Python `http.server` on a background thread.
Pass `--no-browser` to suppress auto-opening, `--port N` to change port.

## Quick start

```powershell
# 1) Install
python -m pip install -r requirements.txt

# 2) Smoke test the full pipeline on 2 volumes (CPU is fine for the smoke run)
python scripts/build_cache.py --smoke
python scripts/make_pseudo_labels.py
python scripts/train.py --smoke
python scripts/generate.py --smoke
python scripts/validate.py
```

## Real run (8–16 GB GPU)

```powershell
python scripts/build_cache.py                       # full 390 GB → ~10–30 GB cache
python scripts/make_pseudo_labels.py
python scripts/train.py --stage cvae                # ~20 epochs, a few hours
python scripts/train.py --stage diff                # ~40 epochs, longer
python scripts/generate.py                          # writes 400 patients
python scripts/validate.py                          # verifies DICOM correctness
```

## Architectural decisions

- **2D slice-based latent diffusion** (not true 3D) — the latent space is
  32×32×4 per slice, which fits comfortably on consumer GPUs. Slice continuity
  is enforced at sampling time via z-position conditioning + EMA on the
  per-slice init noise (see `generate.generate_volume`).
- **CVAE first, then diffusion in latent space.** The CVAE is small (~1 M params
  range depending on `base_channels`) and trains fast; the diffusion UNet then
  only has to learn the latent distribution, which is much cheaper than pixel
  space.
- **Plain/Hilly labels are morphometric pseudo-labels, NOT biographical.**
  We compute pelvic_inlet_mm + iliac_flare + sacral_tilt from the bone mask of
  each cached volume, KMeans-cluster, and name the broader-pelvis cluster
  "plain". This is what the conditioning learns. Real region labels would
  require new data with documented provenance — that's outside this project.

## Output format

```
synthetic_dataset/
└── PF_0001/
    ├── DICOM/
    │   ├── IM-0001.dcm
    │   ├── IM-0002.dcm
    │   └── ...
    ├── PNG/slice_0001.png ...
    ├── JPG/slice_0001.jpg ...
    └── metadata.json
```

Each DICOM series:
* shares one StudyInstanceUID + one SeriesInstanceUID
* `ImagePositionPatient[Z]` increases by `slice_thickness_mm` (default 1.5 mm)
* `Modality=CT`, `BodyPartExamined=PELVIS`, `PatientSex=F`
* `RescaleSlope=1, RescaleIntercept=-1024` so stored pixels are `HU + 1024`
* loads directly in 3D Slicer as a 3D volume.

## Known limits

- Pelvic Z-range detection is bone-area heuristic; series with heavy metal
  artifacts (CTPelvic1K dataset7) may have noisy crops. They're still useful
  for training texture but should ideally be excluded — see
  `configs/default.yaml: preprocess.enable.ctpelvic1k`.
- The CVAE is a deterministic compressor in latent space, not a quantized
  VQ-VAE. Reconstructions are smoother than a VQ-LDM would give, which is fine
  for "research-grade plausibility" (the stated target) but won't fool a
  radiologist.
- True 3D continuity (e.g. vessels tracking across slices) is approximate.
  For better 3D coherence, the next step would be either (a) a small 3D
  conditioning UNet over latents or (b) volumetric finetuning on a bigger GPU.
