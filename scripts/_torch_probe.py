import sys
print("A start", flush=True)
import os
print(f"B CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}", flush=True)
print("C importing torch...", flush=True)
import torch
print(f"D torch {torch.__version__}", flush=True)
print(f"E cuda available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"F device count: {torch.cuda.device_count()}", flush=True)
    print(f"G device 0: {torch.cuda.get_device_name(0)}", flush=True)
print("DONE", flush=True)
