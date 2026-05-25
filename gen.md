# AI-Driven Synthetic Pelvic CT Dataset Generator

## Goal

Generate realistic synthetic female pelvic CT datasets for pelvic floor research using:

* public hospital-grade datasets
* AI/ML generation
* DICOM reconstruction
* metadata synthesis
* pelvic anatomical conditioning

Target output:

* 300–400 synthetic patients
* Plain region + Hilly region distribution

Example:

| Total | Plain | Hilly |
| ----- | ----- | ----- |
| 400   | 200   | 200   |

---

## Dataset Sources

* TCGA-UCEC
* TCGA-CESC
* TCGA-OV
* RSNA Trauma Dataset
* Zenodo Pelvic Datasets

Input formats:

* DICOM CT
* PNG CT slices
* JPG CT slices
* segmentation masks
* metadata files

---

## Core Architecture

```text
Public Datasets
        │
        ▼
DICOM / Image Preprocessing
        │
        ▼
Pelvic Region Extraction
        │
        ▼
AI Training Engine
(Diffusion + CVAE)
        │
        ▼
Synthetic Volume Generation
        │
 ┌──────┴──────┐
 ▼             ▼
DICOM      PNG/JPG
Builder     Export
        │
        ▼
Metadata Generator
        │
        ▼
Validation in 3D Slicer
```

---

## Preprocessing Layer

Responsibilities:

* DICOM reading
* slice ordering
* voxel reconstruction
* image normalization
* pelvic cropping
* metadata extraction

Libraries:

* pydicom
* MONAI
* SimpleITK
* NumPy

---

## Hybrid Dataset Strategy

| Dataset Type       | Role               |
| ------------------ | ------------------ |
| DICOM CT           | volumetric anatomy |
| PNG/JPG CT         | texture realism    |
| Metadata           | scanner realism    |
| Segmentation masks | anatomy guidance   |

DICOM datasets provide:

* voxel continuity
* spacing
* orientation
* HU distributions

PNG/JPG datasets provide:

* CT appearance
* texture patterns
* edge characteristics

---

## AI Model

Primary architecture:

```text
Hybrid Diffusion + CVAE
```

Reason:

| Model     | Purpose                |
| --------- | ---------------------- |
| CVAE      | structure conditioning |
| Diffusion | realistic CT texture   |

Model learns:

* pelvic anatomy
* slice continuity
* density distributions
* region variation
* volumetric consistency

---

## Region Conditioning

Input label:

```text
Plain / Hilly
```

Model learns:

```text
Region → morphology relationship
```

---

## Output Requirements

### 1. Synthetic DICOM Studies

```text
Patient_001/
   ├── IM-0001.dcm
   ├── IM-0002.dcm
   └── ...
```

Requirements:

* load in 3D Slicer
* proper 3D reconstruction
* slice continuity
* believable anatomy

---

### 2. PNG/JPG Exports

Used for:

* research papers
* visualization
* presentation

---

### 3. Synthetic Metadata

Example:

| Field           | Example |
| --------------- | ------- |
| Patient ID      | PF_001  |
| Region          | Hilly   |
| Sex             | Female  |
| Modality        | CT      |
| Slice Thickness | 1 mm    |

---

## Validation Pipeline

```text
Generated Volume
        │
        ▼
DICOM Reconstruction
        │
        ▼
3D Slicer Validation
        │
        ▼
Anatomical Inspection
```

Checks:

* slice continuity
* pelvis geometry
* metadata consistency
* 3D reconstruction

---

## Technology Stack

| Tool      | Purpose          |
| --------- | ---------------- |
| PyTorch   | AI training      |
| MONAI     | medical AI       |
| pydicom   | DICOM handling   |
| SimpleITK | image processing |
| 3D Slicer | validation       |

---

## Final Deliverables

```text
Synthetic_Dataset/
│
├── Patient_001/
│     ├── DICOM/
│     ├── PNG/
│     ├── JPG/
│     └── metadata.json
│
├── Patient_002/
│     └── ...
```

Contains:

* synthetic DICOM studies
* PNG/JPG exports
* synthetic metadata
* region labels
* pelvic anatomical data

---

## Technical Target

Target:

```text
Research-grade radiological plausibility
```

Not:

```text
perfect clinical indistinguishability
```

Focus:

* believable anatomy
* believable 3D reconstruction
* believable CT appearance
* volumetric consistency
* low-cost implementation
