"""Zip PFD_Synthetic_Dataset_350 with progress reporting."""
import zipfile, os, sys
from pathlib import Path

SRC  = Path("synthetic_dataset/PFD_Synthetic_Dataset_350")
DEST = Path("synthetic_dataset/PFD_Synthetic_Dataset_350.zip")

files = [f for f in SRC.rglob("*") if f.is_file()]
total = len(files)
total_bytes = sum(f.stat().st_size for f in files)

print(f"Files  : {total}")
print(f"Size   : {total_bytes/1e9:.2f} GB")
print(f"Output : {DEST}")
print(f"Starting...", flush=True)

done = 0
done_bytes = 0

with zipfile.ZipFile(DEST, "w", compression=zipfile.ZIP_DEFLATED,
                     compresslevel=1, allowZip64=True) as zf:
    for f in files:
        arcname = f.relative_to(SRC.parent)
        zf.write(f, arcname)
        done += 1
        done_bytes += f.stat().st_size
        if done % 500 == 0 or done == total:
            pct = done / total * 100
            gb  = done_bytes / 1e9
            print(f"  {done}/{total} files  {gb:.1f}/{total_bytes/1e9:.1f} GB  {pct:.0f}%", flush=True)

final_mb = DEST.stat().st_size / 1e6
print(f"\nDone.  {DEST.name}  {final_mb:.0f} MB", flush=True)
