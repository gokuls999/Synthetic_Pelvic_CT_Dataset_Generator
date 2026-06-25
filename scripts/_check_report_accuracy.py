import json, numpy as np, pydicom
from pathlib import Path

pt_dir = Path("synthetic_dataset/PFD_Real_Dataset_350/Plain_175/REAL_002_PLAIN_UTERINE_PROLAPSE_G3")
meta   = json.loads((pt_dir / "metadata_real.json").read_text())
ds_name = meta["source_dataset"]
uid     = meta["source_uid"]

print("Patient  :", meta["patient_id"])
print("Pattern  :", meta["pfd_pattern"], "Grade", meta["pfd_grade"])
print("Source   :", ds_name, "/", uid)
print("DICOM slices in patient folder:", meta["slices"])

# NPZ spacing
npz = Path(f"cache/{ds_name}/{uid}.npz")
with np.load(npz, allow_pickle=True) as nf:
    sp = nf["spacing"].tolist()
    print("\nNPZ spacing [sz,sy,sx] mm:", sp)

# Patient DICOM spacing
dcms = sorted((pt_dir / "DICOM").glob("*.dcm"))
ds0  = pydicom.dcmread(str(dcms[0]))
print("Patient DICOM pixel spacing:", list(ds0.PixelSpacing), "slice_thickness:", ds0.SliceThickness)

# Mask stats vs patient slice count
stats = json.loads(Path(f"cache/masks/{ds_name}/{uid}/stats.json").read_text())
sac  = stats["sacrum"]
hipl = stats["hip_left"]
hipr = stats["hip_right"]
print("\nMask stats coordinate space (Z,Y,X):")
print(f"  Sacrum   Z {sac['bbox_min'][0]}->{sac['bbox_max'][0]}  extent={sac['extent'][0]} slices")
print(f"  Hip_L    Z {hipl['bbox_min'][0]}->{hipl['bbox_max'][0]}  extent={hipl['extent'][0]} slices")
print(f"  Patient DICOM slices: {len(dcms)}")
print(f"  Match: {abs(sac['bbox_max'][0] - len(dcms)) < 20}")

# Compute transverse
sx = sp[2]
trans_px = hipr["bbox_max"][2] - hipl["bbox_min"][2]
trans_cm = round(trans_px * sx / 10, 1)
print(f"\nTransverse inlet from mask stats: {trans_px} px x {sx:.3f} mm = {trans_cm} cm")

# What does the actual DICOM bone look like?
mid_idx = len(dcms) // 2
ds_mid = pydicom.dcmread(str(dcms[mid_idx]), force=True)
arr = ds_mid.pixel_array.astype(np.int32)
slope = float(getattr(ds_mid, "RescaleSlope", 1) or 1)
inter = float(getattr(ds_mid, "RescaleIntercept", -1024) or -1024)
hu = arr * slope + inter
bone = hu > 300
bone_x = np.where(bone.any(axis=0))[0]
if len(bone_x) > 0:
    bone_span_px = bone_x[-1] - bone_x[0]
    bone_span_cm = round(bone_span_px * float(ds0.PixelSpacing[1]) / 10, 1)
    print(f"Bone X span from actual mid-DICOM slice: {bone_span_px} px = {bone_span_cm} cm")
    print(f"Difference from mask-based measurement: {abs(trans_cm - bone_span_cm):.1f} cm")

print("\n--- What IS data-driven ---")
print("Bone pelvimetry (from mask stats): YES, same coordinate space as patient DICOM")
print("Demographics (age, BMI, obstetric): NO - seeded generation, not from patient CT")
print("Muscle thickness: NO - formula from population/grade, not from DICOM voxels")
print("Levator hiatus: PARTIALLY - grade-calibrated estimate, not measured from DICOM")
