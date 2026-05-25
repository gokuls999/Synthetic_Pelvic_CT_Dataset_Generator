# Pelvic CT Training Dataset — Handoff to the Generator Project

This document describes what's been built in `ct_data_filter/` for the
**pelvic-CT generative model** (South Indian hill vs plain populations,
pelvic/foot condition study). Everything below is **female pelvic CT data**
ready (or being readied) for training.

## What you are inheriting

Six datasets, downloaded, filtered, and deduplicated by sex (where needed).
Outputs are under `output/`; supplementary masks under `masks/`.

### Cumulative inventory (as of 2026-05-23)

| Dataset | Volumes | Patients | Size | Format | Location |
|---|---|---|---|---|---|
| RSNA 2023 Abdominal Trauma | **1,116** series | 938 | 96.69 GB | DICOM | `output/rsna-2023-abdominal-trauma-detection/pelvic_cts/` |
| CT Colonography (TCIA) | **1,614** series | 401 (408 studies) | 211.80 GB | DICOM | `output/ct_colonography/female_cts/` |
| CTPelvic1K | **136** NIfTI | 136 | 10.66 GB | NIfTI | `output/ctpelvic1k/female_cts/` |
| TCGA-UCEC (uterine cancer) | **65** patients | 65 | ~35 GB | DICOM | `output/tcga_ucec/female_cts/` |
| TCGA-OV (ovarian cancer) | **143** patients | 143 | ~26 GB | DICOM | `output/tcga_ov/female_cts/` |
| TCGA-CESC (cervical cancer) | **53** patients | 53 | ~9 GB | DICOM | `output/tcga_cesc/female_cts/` |
| **Total** | **~3,127 entries** | **~1,736** | **~390 GB** | mixed | |

All scans cover the pelvic region (verified — see "Validation" below).

**TCGA caveat:** the three TCGA folders are gynecological-cancer cohorts —
all-female by selection. The CTs are diagnostic/staging scans for uterine,
ovarian, and cervical cancers, so they show **pelvic pathology** (tumors,
post-treatment changes). Do not assume normal anatomy — for that, prefer
RSNA / CT Colonography / CTPelvic1K.

### Supplementary masks (`masks/` — pelvic-bone segmentation labels)

These pair with specific CT volumes inside `output/` and are very useful for
anatomy-aware training or conditional generation.

| Folder under `masks/` | Pairs with | Files |
|---|---|---|
| `CTPelvic1K_dataset2_mask_mappingback/` | **CT Colonography CTs** in `output/ct_colonography/female_cts/` | 714 |
| `ipcai2021_dataset6_Anonymized/` | CTPelvic1K dataset6 CTs in `output/ctpelvic1k/female_cts/` | 103 |
| `dataset7_loose/` | CTPelvic1K dataset7 (CLINIC-metal) CTs in `output/ctpelvic1k/female_cts/` | 14 |

Label convention (4-label segmentation per pelvic bone): sacrum, left hip,
right hip, lumbar vertebrae. See CTPelvic1K paper Table 2 for exact IDs.

## Per-dataset notes for the generator

### RSNA 2023 Abdominal Trauma Detection

- **Origin:** Kaggle competition `rsna-2023-abdominal-trauma-detection`.
- **Two-stage filter:**
  1. **Sex** — DICOM tag `(0010,0040)` PatientSex == 'F'. 949 female patients kept.
  2. **Pelvic coverage** — image-content based (`pelvic_filter.py`, Module 4)
     because the Kaggle release stripped anatomy DICOM tags. Heuristic counts
     caudal no-lung slices with a bone object spanning ≥190 mm (iliac wings /
     femoral heads). Validated 8/8 on hand-labelled samples.
- **Output structure:**
  ```
  pelvic_cts/train_images/{patient_id}/{series_id}/*.dcm
  ```
- **Supplementary:** `partial_pelvis/train_images/...` contains 91 series
  with iliac crest visible but not the full pelvis. User's call whether to
  include in training.
- **Per-series log:** `output/rsna-2023-abdominal-trauma-detection/pelvic_results.csv`
  (verdict, n_wide, lung_dist_mm per series).
- **Population:** US trauma data. Note the project's research question is
  about South Indian populations; this is general-pelvic anatomy, not
  population-matched.

### CT Colonography (TCIA)

- **Origin:** TCIA collection "CT COLONOGRAPHY" (NBIA Data Retriever / aria2c).
  Partial download (~454 GB raw) due to disk limits — does not affect filter
  validity; what was downloaded is fully filtered.
- **One-stage filter:** Sex only. No pelvic filter needed because **CT colonography
  is abdomen+pelvis by clinical protocol** (`BodyPartExamined=COLON`). Every series
  in `female_cts/` covers the pelvis by construction.
- **Output structure:**
  ```
  female_cts/{PatientUID}/{StudyUID}/{SeriesUID}/*.dcm
  ```
- **Per-series log:** `output/ct_colonography/filter_results.csv`.
- **Quality:** Contrast usually absent (it's a colonography prep). Patients
  positioned prone/supine — protocol typically captures both. Some series per
  patient (supine + prone + scout). Use SeriesDescription DICOM tag to filter
  if you want only one orientation per patient.

### CTPelvic1K (in progress)

- **Origin:** Zhao et al. dataset (Shanghai 6th People's Hospital). Already
  pelvic-focused (the name says it). Distributed as `.tar.gz` tarballs.
- **Subsets used as CT volumes:**
  - `dataset6_data`: 103 NIfTI (CLINIC, anonymized clinical trauma)
  - `dataset7_data`: 75 NIfTI (CLINIC-metal, with implants — beware of metal
    artifacts in image generation)
- **Subsets used as masks (segmentation labels):** see "Bonus: segmentation
  masks" below — these are very useful for the generator.
- **Sex filter:** TotalSegmentator CNN, prostate + urinary_bladder detection
  (`sex_filter_cnn.py`). Heuristic approach (`sex_filter.py`) was tried and
  abandoned — outer bone bbox doesn't carry strong sex signal. Kept for
  reference but **do not trust its CSV**.
- **Final tally:** 136 female / 22 male / 20 uncertain. Male + uncertain
  were deleted per the female-only convention. The 20 uncertain were all
  "no bladder coverage" — scans that don't extend low enough to detect the
  bladder, mostly dataset7 with metal artifacts.
- **Per-volume log:** `sex_results.csv` at the project root.

## Bonus: paired segmentation masks (high value)

CTPelvic1K shipped **pelvic bone segmentation labels** for several public
datasets. These are gold for pretraining the generator on anatomy-aware
losses or for conditional generation.

| Mask folder | Pairs with | Files |
|---|---|---|
| `datasets/CTPelvic1K/CTPelvic1K_dataset2_mask_mappingback/` | **CT Colonography CTs** (we have these) | 714 |
| `datasets/CTPelvic1K/ipcai2021_dataset6_Anonymized/` | dataset6 CTs (we have these) | 103 |
| Loose `dataset7_*_mask*.nii.gz` at CTPelvic1K root | dataset7 CTs (we have these) | 14 |
| `datasets/CTPelvic1K/CTPelvic1K_dataset3_mask_mappingback/` | MSD-T10 CTs (NOT downloaded) | 155 — useless without CTs |
| `datasets/CTPelvic1K/CTPelvic1K_dataset4_mask_mappingback/` | KITS19 CTs (NOT downloaded) | 44 — useless without CTs |
| Loose `dataset1_*_mask_*.nii.gz` | TCIA Abdomen CTs (NOT downloaded) | ~35 — useless without CTs |
| Loose `dataset5_*_mask_*.nii.gz` | TCIA Cervix CTs (NOT downloaded) | ~41 — useless without CTs |

**Label convention:** 4-label segmentation per pelvic bone — sacrum,
left hip, right hip, lumbar vertebrae (check CTPelvic1K paper Table 2 for
exact label IDs).

## File formats summary

- **DICOM** (RSNA, CT Colonography): use `pydicom` or `pydicom_nifti`. Sort
  slices by ImagePositionPatient[2] or InstanceNumber. Pixel values: apply
  RescaleSlope/Intercept for HU.
- **NIfTI** (CTPelvic1K, masks): use `nibabel`. Already in canonical
  orientations after `nib.as_closest_canonical()`. Pixel values are already HU.
- **Spacings vary** across datasets (0.7–1.0 mm in-plane, 0.5–5 mm slice
  thickness). Resample to a common grid before training.

## Filtering scripts (Modules)

| Module | Script | Use |
|---|---|---|
| 1 | `dicom_tag_filter.py` | Auto sex split on PatientSex DICOM tag |
| 2 | `manual_reviewer.py` | F/M/S/Q keystroke review (kept; not used post-CTPelvic1K decision) |
| 3 | `utils.py` | Shared DICOM/NIfTI loaders + helpers |
| 4 | `pelvic_filter.py` | Image-content pelvic detector (caudal no-lung + bone spread ≥190 mm). Reusable on any new dataset that strips anatomy tags. |
| 5 | `sex_filter_cnn.py` | TotalSegmentator-based sex classifier for no-metadata datasets. Reusable. |
| — | `sex_filter.py` | Failed heuristic approach. Kept for reference. **Do not use.** |

## Validation done

- RSNA pelvic filter: 8/8 hand-labelled samples correct; flagged cases reviewed
  via montage grids before applying.
- CT colonography: gender filter trusted (DICOM tag present and clinically
  reliable in TCIA).
- CTPelvic1K: TotalSegmentator's prostate detection is the validated method
  (it's the same model used for clinical sex inference in research). One spot
  check on CLINIC_0001 confirmed: 0 prostate + 322k bladder voxels → female.

## Known limitations / things the generator should know

- **No demographics beyond sex.** Age, BMI, race are absent. The South Indian
  population question (hill vs plain) is **NOT solved by this dataset** — these
  are US (RSNA), generic-TCIA, and Shanghai (CTPelvic1K) cohorts. Population
  matching is a separate research problem the user owns.
- **CT colonography is colon-prep imaging.** Some series have contrast residue
  in colon, distended colon, prone/supine pairs. Pelvic anatomy is intact but
  bowel state is non-physiologic.
- **RSNA is trauma.** Acute hemorrhage, organ injury, contrast extravasation
  present in many. For a "healthy pelvic anatomy" generator, consider
  filtering RSNA further by injury severity (RSNA labels are available per
  patient).
- **CTPelvic1K dataset7 has metal implants.** Will show as bright streak
  artifacts. Exclude this subset if you want artifact-free training data
  (it's ~75 of ~178 NIfTI volumes — about half of CTPelvic1K).

## How to resume / extend filtering

If new datasets arrive:

1. Drop the raw data under `datasets/<name>/` (NIfTI) or as a folder of DICOM.
2. Has DICOM `PatientSex` tag? → `python dicom_tag_filter.py --input PATH --yes`.
3. No DICOM tags / NIfTI only? → junction into `input/` and
   `python sex_filter_cnn.py` (CPU, ~55 s/CT) then `--apply`.
4. Coverage uncertain? → `python pelvic_filter.py --input PATH` and then `--apply`.

## Key paths

- Project root: `D:\Muthu kumar\ct_data_filter\`
- Final outputs: `D:\Muthu kumar\ct_data_filter\output\`
- Source archives (CTPelvic1K tarballs, NIfTI extracts): `D:\Muthu kumar\ct_data_filter\datasets\`
- Per-dataset filter logs (CSV): `output/<dataset>/filter_results.csv` or
  the per-script CSV (`pelvic_results.csv`, `sex_results.csv`).
