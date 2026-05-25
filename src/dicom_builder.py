"""Write a synthetic HU volume out as a multi-slice DICOM CT series.

Produces files that load cleanly in 3D Slicer:
  * matching StudyInstanceUID across slices
  * matching SeriesInstanceUID across slices
  * incrementing InstanceNumber and ImagePositionPatient[Z] per slice
  * RescaleSlope=1, RescaleIntercept=-1024 (pixels stored as uint16 = HU + 1024)
  * ImageOrientationPatient = standard axial (1,0,0,0,1,0)
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .generate import SyntheticVolume


def _new_uid(prefix: str = "1.2.826.0.1.3680043.10.1424") -> str:
    """Generate a DICOM UID under a private root."""
    from pydicom.uid import generate_uid
    return generate_uid(prefix=prefix + ".")


def _now() -> tuple[str, str]:
    n = datetime.datetime.now()
    return n.strftime("%Y%m%d"), n.strftime("%H%M%S.%f")[:-3]


def write_dicom_series(vol: "SyntheticVolume", out_dir: str | Path, cfg: dict) -> Path:
    """Serialize `vol` as a CT image series under `out_dir`. Returns the dir."""
    from pydicom import Dataset, FileDataset
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import (ExplicitVRLittleEndian, CTImageStorage)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pixels = vol.pixels_hu                       # int16 (Z, H, W) in HU
    sz, sy, sx = vol.spacing
    Z, H, W = pixels.shape

    date, time_str = _now()
    study_uid = _new_uid()
    series_uid = _new_uid()
    frame_of_ref_uid = _new_uid()

    out_cfg = cfg["output"]

    for i in range(Z):
        sop_uid = _new_uid()

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = sop_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = _new_uid()

        ds = FileDataset(str(out_dir / f"IM-{i+1:04d}.dcm"),
                         {}, file_meta=file_meta, preamble=b"\0" * 128)

        # ----- Patient / Study / Series -----
        ds.PatientName = vol.patient_id
        ds.PatientID = vol.patient_id
        ds.PatientSex = "F"
        ds.PatientBirthDate = ""
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = sop_uid
        ds.SOPClassUID = CTImageStorage
        ds.StudyID = "1"
        ds.SeriesNumber = 1
        ds.AccessionNumber = vol.patient_id
        ds.StudyDate = date
        ds.SeriesDate = date
        ds.AcquisitionDate = date
        ds.ContentDate = date
        ds.StudyTime = time_str
        ds.SeriesTime = time_str
        ds.AcquisitionTime = time_str
        ds.ContentTime = time_str
        ds.Manufacturer = out_cfg.get("manufacturer", "SyntheticPelvicCT")
        ds.InstitutionName = out_cfg.get("institution", "GenAI Pelvic Study")
        ds.StudyDescription = f"Synthetic Pelvic CT ({vol.region})"
        ds.SeriesDescription = f"Synthetic axial CT ({vol.region})"
        ds.BodyPartExamined = "PELVIS"
        ds.Modality = "CT"
        ds.FrameOfReferenceUID = frame_of_ref_uid

        # ----- Image geometry -----
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0.0, 0.0, float(i) * sz]
        ds.SliceLocation = float(i) * sz
        ds.SliceThickness = float(sz)
        ds.PixelSpacing = [float(sy), float(sx)]
        ds.InstanceNumber = i + 1
        ds.AcquisitionNumber = 1

        # ----- Pixel data -----
        # Store as uint16; rescale gives back HU.
        stored = (pixels[i].astype(np.int32) + 1024).clip(0, 4095).astype(np.uint16)
        ds.Rows = H
        ds.Columns = W
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.RescaleIntercept = -1024.0
        ds.RescaleSlope = 1.0
        ds.WindowCenter = 40
        ds.WindowWidth = 400
        ds.PixelData = stored.tobytes()

        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(str(out_dir / f"IM-{i+1:04d}.dcm"), write_like_original=False)

    return out_dir
